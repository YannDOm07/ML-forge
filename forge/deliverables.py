"""Deliverables required by the competition: submission, reproducible solution, notebook, report.

Everything is generated from the database (the source of truth). Replacing a validated
deliverable needs an explicit approval. The solution package is self-contained: it runs
offline with pandas and scikit-learn, without ML Forge.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forge.approvals import stored_approver
from forge.db import now_iso, session, transaction
from forge.errors import ApprovalRequired, ForgeError
from forge.experiments import get_experiment, get_metrics, list_experiments
from forge.guardrails import Action, ApprovalRequest, Approver, project_decision_recorder, require_approval
from forge.intake import classify_files, competition_card, fact_value
from forge.projects import Project
from forge.report import decisions_by_experiment, write_report
from forge.runner import Protocol, current_best, dataset_hash, latest_profile, latest_protocol, run_experiment, submit_experiment
from forge.submission import SubmissionError, build_submission
from forge.task import METRICS
from forge.workspace import Workspace

DELIVERABLES = "deliverables"
SOLUTION = "solution"
TABULAR_SOURCE = Path(__file__).with_name("tabular.py")


@dataclass
class FinalizeResult:
    files: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    summary: str = ""


def submission_kind(protocol: Protocol) -> str:
    if not protocol.task.is_classification:
        return "value"
    return "proba" if METRICS[protocol.task.metric][2] else "label"


def sample_file(project: Project) -> Path | None:
    with session(project.db_path) as conn:
        paths = [Path(row["path"]) for row in conn.execute("SELECT path FROM data_sources")]
    wanted = fact_value(competition_card(project), "submission_format")
    for path in paths:
        if wanted and path.name == wanted:
            return path
    return classify_files(paths).get("sample")


# --- stability ---------------------------------------------------------------------------

def stability_check(workspace: Workspace, project: Project, protocol: Protocol, best_id: str) -> dict:
    """Re-run the best experiment with other seeds; the spread of its score is measured, not assumed."""
    best = get_experiment(project.db_path, best_id)
    metric = protocol.task.metric
    means = [float(np.mean(get_metrics(project.db_path, best_id)[metric]["fold"]))]
    if best.card.get("role") in {"decision_stage", "baseline"}:
        return {"seeds": 1, "means": means, "std": 0.0, "note": "Non applicable à ce type d'expérience."}
    with session(project.db_path) as conn:
        code = conn.execute("SELECT content FROM experiment_code WHERE code_hash = ?", (best.card["code_hash"],)).fetchone()
    base_seed = int(best.card.get("seed", workspace.config.get("evaluation", "seed")))
    for offset in range(1, workspace.config.get("evaluation", "stability_seeds")):
        card = {key: value for key, value in best.card.items() if key not in {"code_hash", "data_hash", "new_reason"}}
        card |= {"role": "stability", "seed": base_seed + offset, "parent_experiment_id": best_id,
                 "hypothesis": f"Stabilité de {best_id} avec la seed {base_seed + offset}."}
        outcome = run_experiment(workspace, project, submit_experiment(project, card, code["content"]).id)
        if outcome.metric_mean is not None:
            means.append(outcome.metric_mean)
    std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    return {"seeds": len(means), "means": [round(m, 6) for m in means], "std": round(std, 6)}


# --- solution package --------------------------------------------------------------------

MAKE_SUBMISSION = '''"""Reproduit hors ligne la soumission produite par ML Forge.

Usage : python make_submission.py --data DOSSIER_DES_DONNEES [--out submission.csv]
Dépendances : pandas, numpy, scikit-learn.
"""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from forge_tabular import apply_class_weights, class_columns, to_submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="dossier contenant les fichiers de la compétition")
    parser.add_argument("--out", default="submission.csv", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    spec = json.loads((HERE / "submission_spec.json").read_text(encoding="utf-8"))
    context = json.loads((HERE / "context.json").read_text(encoding="utf-8"))
    for key in ("train_path", "test_path"):
        if context["task"].get(key):
            context["task"][key] = str((args.data / context["task"][key]).resolve())
    work = HERE / "run"
    work.mkdir(exist_ok=True)
    (work / "context.json").write_text(json.dumps(context), encoding="utf-8")
    previous = Path.cwd()
    os.chdir(work)
    try:
        runpy.run_path(str(HERE / "script.py"), run_name="__main__")
    finally:
        os.chdir(previous)
    predictions = pd.read_csv(work / "test_pred.csv")
    if spec.get("class_weights"):
        columns = class_columns(predictions)
        predictions[columns] = apply_class_weights(predictions[columns], spec["class_weights"]).to_numpy()
    sample = pd.read_csv(args.data / spec["sample_file"])
    to_submission(sample, predictions, spec["kind"], spec["classes"], spec["positive_label"]).to_csv(out, index=False)
    print(f"Soumission écrite : {out}")


if __name__ == "__main__":
    main()
'''


def _replace_guard(project: Project, target: Path, approver: Approver, source: str) -> None:
    if target.exists():
        request = ApprovalRequest(Action.REPLACE_FINAL_DELIVERABLE, str(target), {"nouvelle source": source})
        require_approval(request, approver, project_decision_recorder(project.db_path, "ml-forge"))
        shutil.rmtree(target) if target.is_dir() else target.unlink()


def build_solution_package(project: Project, protocol: Protocol, best_id: str, sample: Path, approver: Approver) -> Path:
    best = get_experiment(project.db_path, best_id)
    script_owner, weights = best, None
    if best.card.get("role") == "decision_stage":
        info = json.loads((project.path / "experiments" / best_id / "info.json").read_text(encoding="utf-8"))
        weights = info["class_weights"]
        script_owner = get_experiment(project.db_path, best.parent_id)
    target = project.path / DELIVERABLES / SOLUTION
    _replace_guard(project, target, approver, best_id)
    target.mkdir(parents=True)
    with session(project.db_path) as conn:
        code = conn.execute(
            "SELECT content FROM experiment_code WHERE code_hash = ?", (script_owner.card["code_hash"],)
        ).fetchone()["content"]
    (target / "script.py").write_text(code.replace("forge.tabular", "forge_tabular"), encoding="utf-8")
    tabular = TABULAR_SOURCE.read_text(encoding="utf-8").replace(
        "from forge.task import PROBA_PREFIX", 'PROBA_PREFIX = "p:"'
    )
    (target / "forge_tabular.py").write_text(tabular, encoding="utf-8")
    task = protocol.task.to_dict()
    task["train_path"] = Path(task["train_path"]).name
    task["test_path"] = Path(task["test_path"]).name if task["test_path"] else None
    context = {
        "task": task, "folds": protocol.folds, "classes": protocol.classes, "fold_scheme": protocol.scheme,
        "seed": int(script_owner.card.get("seed", 2026)), "parent_dir": None,
        "outputs": {"oof": "oof.csv", "test": "test_pred.csv", "info": "info.json"},
    }
    (target / "context.json").write_text(json.dumps(context), encoding="utf-8")
    spec = {"kind": submission_kind(protocol), "classes": protocol.classes, "positive_label": protocol.task.positive_label,
            "sample_file": sample.name, "class_weights": weights}
    (target / "submission_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "make_submission.py").write_text(MAKE_SUBMISSION, encoding="utf-8")
    files = ", ".join(p for p in (task["train_path"], task["test_path"], sample.name) if p)
    (target / "README.md").write_text(
        f"# Solution reproductible — {project.name}\n\n"
        f"Expérience retenue : {best_id} ({best.method}). Seed : {context['seed']}.\n\n"
        f"```\npython make_submission.py --data <dossier contenant {files}> --out submission.csv\n```\n\n"
        "Dépendances : pandas, numpy, scikit-learn (versions de requirements.txt). Fonctionne hors ligne.\n",
        encoding="utf-8",
    )
    (target / "requirements.txt").write_text(_requirements(), encoding="utf-8")
    return target


def _requirements() -> str:
    from importlib.metadata import PackageNotFoundError, version

    lines = []
    for package in ("numpy", "scipy", "pandas", "scikit-learn"):
        try:
            lines.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            lines.append(package)
    return "\n".join(lines) + "\n"


def build_notebook(project: Project, solution: Path, slug: str | None) -> Path:
    """Thin notebook for code competitions: it runs the solution package (experimental)."""
    data = f"/kaggle/input/{slug}" if slug else "<dossier des données>"
    cells = [
        _cell("markdown", f"# {project.name} — solution ML Forge\n\nAjoutez le dossier `solution` comme jeu de "
                          "données du notebook, puis exécutez. *Expérimental : à vérifier dans l'environnement de "
                          "la plateforme (hors ligne, limite de temps).*"),
        _cell("code", f'import subprocess, sys\nSOLUTION = "solution"\nDATA = "{data}"\n'
                      'subprocess.run([sys.executable, f"{SOLUTION}/make_submission.py", "--data", DATA, '
                      '"--out", "submission.csv"], check=True)'),
    ]
    notebook = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = solution.parent / "solution.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _cell(kind: str, source: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell |= {"execution_count": None, "outputs": []}
    return cell


# --- technical report ---------------------------------------------------------------------

def build_technical_report(project: Project, protocol: Protocol, best_id: str, stability: dict) -> Path:
    decisions = decisions_by_experiment(project)
    experiments = [e for e in list_experiments(project.db_path) if e.protocol_id == protocol.id]
    profile = latest_profile(project, dataset_hash(project, protocol.task)) or {}
    best = get_experiment(project.db_path, best_id)
    best_decision = decisions[best_id]
    accepted = [e for e in experiments if (decisions.get(e.id) or {}).get("status") in {"baseline", "accepted"}]
    rejected = [e for e in experiments if (decisions.get(e.id) or {}).get("status") in {"rejected", "not_significant"}]
    with session(project.db_path) as conn:
        compliance = conn.execute("SELECT item, justification FROM compliance_log").fetchall()
    lines = [
        f"# {project.name} — rapport technique", "",
        "## Résumé",
        f"Meilleur candidat {best_id} ({best.method}) : {protocol.task.metric} = {best_decision['metric_mean']:.5f} "
        f"± {best_decision['metric_std']:.5f} en validation ; écart-type entre seeds : {stability['std']}.", "",
        "## Données et régime",
        f"{profile.get('rows', '?')} lignes, {profile.get('features', '?')} variables. Cible : {profile.get('target')}.",
        f"Plancher d'erreur (jumeaux contradictoires) : {profile.get('twins', {}).get('error_floor')}.", "",
        "## Validation",
        " ".join(protocol.notes) + (" Validation fiable." if protocol.reliability["reliable"] else
                                     " Validation peu fiable : " + " ".join(protocol.reliability["reasons"])), "",
        "## Historique des composants retenus",
        "| Exp. | Méthode | Score | Δ |", "|---|---|---|---|",
    ]
    for experiment in accepted:
        decision = decisions[experiment.id]
        delta = f"{decision['delta']:+.5f}" if decision["delta"] is not None else "—"
        lines.append(f"| {experiment.id} | {experiment.method} | {decision['metric_mean']:.5f} | {delta} |")
    lines += ["", "## Hypothèses rejetées", "| Exp. | Hypothèse | Δ | Décision |", "|---|---|---|---|"]
    for experiment in rejected:
        decision = decisions[experiment.id]
        lines.append(f"| {experiment.id} | {experiment.hypothesis[:90]} | {decision['delta']:+.5f} | {decision['status']} |")
    lines += ["", "## Conformité"]
    lines += [f"- {row['item']} : {row['justification']}" for row in compliance] or ["- Aucune ressource externe utilisée."]
    lines += ["", "## Reproduction", "Voir `solution/README.md` : `python make_submission.py --data <dossier>`."]
    path = project.path / DELIVERABLES / "technical_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- finalisation ---------------------------------------------------------------------------

def finalize(workspace: Workspace, project: Project, approver: Approver | None = None) -> FinalizeResult:
    approver = approver or stored_approver(project)
    protocol = latest_protocol(project)
    best = current_best(project, protocol.id) if protocol else None
    if best is None:
        return FinalizeResult(summary="Aucun candidat : pas de livrable.")
    result = FinalizeResult()
    sample = sample_file(project)
    stability = stability_check(workspace, project, protocol, best)
    if sample is None:
        result.problems.append("Fichier exemple de soumission introuvable : soumission non générée.")
    else:
        _attempt(result, lambda: build_submission(project, best, sample, submission_kind(protocol), approver, "ml-forge"))
        _attempt(result, lambda: build_solution_package(project, protocol, best, sample, approver))
    card = competition_card(project)
    wanted = {fact["value"] for fact in card["fields"]["deliverables"]}
    solution = project.path / DELIVERABLES / SOLUTION
    if solution.is_dir() and ("notebook" in wanted or fact_value(card, "competition_type") == "code_submission"):
        result.files.append(build_notebook(project, solution, card["slug"]))
    if "report" in wanted:
        result.files.append(build_technical_report(project, protocol, best, stability))
    result.files.append(_write_checklist(project, best, wanted, stability, result.problems))
    if workspace.config.get("backup", "after_campaign"):
        from forge.backup import backup_project

        backup_project(workspace, project)
    write_report(project)
    state = "livrables prêts" if not result.problems else f"{len(result.problems)} problème(s)"
    result.summary = f"Meilleur candidat {best} ; {state}."
    return result


def _attempt(result: FinalizeResult, action) -> None:
    try:
        result.files.append(action())
    except (ApprovalRequired, SubmissionError, ForgeError) as exc:
        result.problems.append(str(exc))


def deliverables_checklist(project: Project, wanted: set[str]) -> list[dict]:
    root = project.path / DELIVERABLES
    items = [("submission", root / "submission.csv", "toujours"), ("solution", root / SOLUTION / "make_submission.py", "toujours")]
    extra = {"notebook": root / "solution.ipynb", "report": root / "technical_report.md",
             "inference_script": root / SOLUTION / "make_submission.py", "training_code": root / SOLUTION / "script.py"}
    items += [(name, extra[name], "exigé par la compétition") for name in sorted(wanted) if name in extra]
    return [{"livrable": name, "fichier": str(path), "raison": why, "présent": path.exists()} for name, path, why in items]


def _write_checklist(project: Project, best: str, wanted: set[str], stability: dict, problems: list[str]) -> Path:
    plan = submission_plan(project)
    lines = [f"# Livrables — {project.name}", "", f"Expérience retenue : {best}. Stabilité entre seeds : {stability}.", "",
             "| Livrable | Raison | Présent | Fichier |", "|---|---|---|---|"]
    for item in deliverables_checklist(project, wanted):
        lines.append(f"| {item['livrable']} | {item['raison']} | {'oui' if item['présent'] else 'NON'} | {item['fichier']} |")
    lines += ["", "## Recommandation de soumission", plan["recommandation"]]
    lines += ["", "## Problèmes"] + [f"- {p}" for p in problems] if problems else []
    lines += ["", "La soumission est à déposer par vous : ML Forge ne soumet jamais rien."]
    path = project.path / DELIVERABLES / "README.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- submissions -------------------------------------------------------------------------------

def submissions_today(project: Project) -> list[dict]:
    today = datetime.now(timezone.utc).date().isoformat()
    with session(project.db_path) as conn:
        rows = conn.execute("SELECT * FROM submissions WHERE substr(submitted_at, 1, 10) = ?", (today,)).fetchall()
    return [dict(row) for row in rows]


def submission_plan(project: Project) -> dict:
    """What to submit today within the daily limit, and which final submissions to select."""
    card = competition_card(project)
    limit = fact_value(card, "rules.daily_submission_limit")
    final_count = fact_value(card, "rules.final_selection_count", 2)
    protocol = latest_protocol(project)
    best = current_best(project, protocol.id) if protocol else None
    decisions = decisions_by_experiment(project)
    with session(project.db_path) as conn:
        submitted = {row["experiment_id"] for row in conn.execute("SELECT experiment_id FROM submissions")}
    higher = protocol.task.higher_is_better if protocol else True
    best_mean = decisions[best]["metric_mean"] if best else None

    def better_than_best(decision: dict) -> bool:
        return decision["metric_mean"] > best_mean if higher else decision["metric_mean"] < best_mean

    # Ambitious = a better mean than the robust candidate, but a gain the computed rule did not accept.
    candidates = sorted(
        (d for d in decisions.values() if best and d["status"] == "not_significant" and better_than_best(d)),
        key=lambda d: d["metric_mean"], reverse=higher,
    )
    ambitious = candidates[0]["experiment_id"] if candidates else None
    remaining = None if limit is None else max(limit - len(submissions_today(project)), 0)
    proposals = [e for e in (best, ambitious) if e and e not in submitted]
    if remaining is not None:
        proposals = proposals[:remaining]
    selection = [e for e in (best, ambitious) if e][:final_count]
    text = (f"Soumettre aujourd'hui : {', '.join(proposals) or 'rien'}"
            f"{f' (reste {remaining} soumission(s) aujourd’hui)' if remaining is not None else ''}. "
            f"Sélection finale recommandée ({final_count}) : {', '.join(selection) or '—'} — {best} est le candidat "
            "robuste (gain accepté par la règle calculée) ; le second, s'il existe, est plus ambitieux mais non "
            "significativement meilleur. Le score public sert d'information, jamais de critère de sélection.")
    return {"limite_quotidienne": limit, "restantes_aujourdhui": remaining, "proposées": proposals,
            "sélection_finale": selection, "robuste": best, "ambitieux": ambitious, "recommandation": text}


def declare_submission(project: Project, experiment_id: str, public_score: float | None = None, note: str = "") -> None:
    """Record a submission the user made; refused beyond the daily limit."""
    get_experiment(project.db_path, experiment_id)
    limit = fact_value(competition_card(project), "rules.daily_submission_limit")
    if limit is not None and len(submissions_today(project)) >= limit:
        raise ForgeError(f"Limite quotidienne atteinte ({limit} soumissions) : réessayez demain.")
    path = project.path / DELIVERABLES / "submission.csv"
    digest = None
    if path.is_file():
        from forge.datasources import file_sha256

        digest = file_sha256(path)
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO submissions (experiment_id, file_sha256, submitted_at, public_score, note) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, digest, now_iso(), public_score, note),
        )
