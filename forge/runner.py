"""Experiment orchestration: protocol, profile, prerequisites, sandboxed run, computed decision."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from forge.data_engine import profile
from forge.datasources import check_data_sources
from forge.db import now_iso, session, transaction
from forge.errors import ExperimentError
from forge.evaluation import assess_reliability, decide, design_folds, fold_scores, simulated_delta
from forge.experiments import (
    Experiment,
    finish_experiment,
    get_experiment,
    get_metrics,
    log_event,
    record_metrics,
    register_experiment,
    start_experiment,
)
from forge.projects import Project
from forge.sandbox import SandboxLimits, run_in_sandbox
from forge.task import PROBA_PREFIX, TaskSpec, read_table
from forge.workspace import Workspace

KEPT_DECISIONS = {"baseline", "accepted"}


@dataclass(frozen=True)
class Protocol:
    id: str
    task: TaskSpec
    folds: list[int]
    classes: list[str] | None
    notes: list[str]
    reliability: dict
    scheme: str = "kfold"  # kfold | time


@dataclass(frozen=True)
class RunOutcome:
    experiment_id: str
    status: str  # completed | failed
    decision: str | None
    metric_mean: float | None
    metric_std: float | None
    delta: float | None
    message: str


# --- protocol and profile -----------------------------------------------------

def define_protocol(
    workspace: Workspace, project: Project, task: TaskSpec, paired_scores: list[tuple[float, float]] = ()
) -> Protocol:
    """Fix the validation folds of a task once; the same task and data give the same protocol id."""
    _require_clean_sources(project, task)
    columns = [c for c in (task.target, task.group_column, task.time_column) if c]
    train = read_table(task.train_path, usecols=columns)
    y = train[task.target]
    groups = train[task.group_column] if task.group_column else None
    times = train[task.time_column] if task.time_column else None
    config = workspace.config
    folds, notes = design_folds(task, y, groups, config.get("evaluation", "n_splits"),
                                config.get("evaluation", "seed"), times)
    reliability = assess_reliability(task, y, folds, config.get("evaluation", "min_class_per_fold"), paired_scores)
    classes = sorted(y.astype(str).unique().tolist()) if task.is_classification else None
    scheme = "time" if task.time_column else "kfold"
    spec = {"task": task.to_dict(), "classes": classes, "notes": notes, "scheme": scheme,
            "data_hash": dataset_hash(project, task)}
    folds_json = json.dumps(folds)
    protocol_id = "P" + hashlib.sha256((json.dumps(spec, sort_keys=True) + folds_json).encode()).hexdigest()[:10]
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO protocols (id, spec_json, folds_json, reliability_json, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (protocol_id, json.dumps(spec, ensure_ascii=False), folds_json,
             json.dumps(reliability.to_dict(), ensure_ascii=False), now_iso()),
        )
    return Protocol(protocol_id, task, folds, classes, notes, reliability.to_dict(), scheme)


def get_protocol(project: Project, protocol_id: str) -> Protocol:
    with session(project.db_path) as conn:
        row = conn.execute("SELECT * FROM protocols WHERE id = ?", (protocol_id,)).fetchone()
    if row is None:
        raise ExperimentError(f"Protocole de validation inconnu : {protocol_id}. Définissez-le d'abord.")
    spec = json.loads(row["spec_json"])
    return Protocol(
        row["id"], TaskSpec(**spec["task"]), json.loads(row["folds_json"]),
        spec["classes"], spec["notes"], json.loads(row["reliability_json"]), spec.get("scheme", "kfold"),
    )


def latest_protocol(project: Project) -> Protocol | None:
    with session(project.db_path) as conn:
        row = conn.execute("SELECT id FROM protocols ORDER BY rowid DESC LIMIT 1").fetchone()
    return get_protocol(project, row["id"]) if row else None


def dataset_hash(project: Project, task: TaskSpec) -> str:
    """Fingerprint of the registered files the task reads."""
    wanted = {str(path.resolve()) for path in task.data_paths()}
    media = [str(Path(directory).resolve()) for directory in task.media_dirs]
    with session(project.db_path) as conn:
        rows = conn.execute("SELECT path, sha256 FROM data_sources ORDER BY path").fetchall()
    known = {row["path"]: row["sha256"] for row in rows
             if row["path"] in wanted or any(row["path"].startswith(directory) for directory in media)}
    missing = wanted - set(known)
    if missing:
        raise ExperimentError(f"Fichier(s) non enregistré(s) comme source de données : {', '.join(sorted(missing))}")
    return hashlib.sha256("".join(f"{p}={h};" for p, h in sorted(known.items())).encode()).hexdigest()


def profile_data(workspace: Workspace, project: Project, task: TaskSpec) -> dict:
    _require_clean_sources(project, task)
    max_rows = workspace.config.get("data", "profile_max_rows")
    train = read_table(task.train_path, nrows=max_rows + 1)
    truncated = len(train) > max_rows
    train = train.head(max_rows)
    test = read_table(task.test_path, nrows=max_rows) if task.test_path else None
    report = profile(task, train, test, truncated)
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO data_profiles (data_hash, profile_json, created_at) VALUES (?, ?, ?)",
            (dataset_hash(project, task), json.dumps(report, ensure_ascii=False), now_iso()),
        )
    return report


def latest_profile(project: Project, data_hash: str) -> dict | None:
    with session(project.db_path) as conn:
        row = conn.execute(
            "SELECT profile_json FROM data_profiles WHERE data_hash = ? ORDER BY id DESC LIMIT 1", (data_hash,)
        ).fetchone()
    return json.loads(row["profile_json"]) if row else None


def _require_clean_sources(project: Project, task: TaskSpec) -> None:
    wanted = {path.resolve() for path in task.data_paths()}
    problems = [s for s in check_data_sources(project.db_path) if s.path in wanted and s.state != "ok"]
    if problems:
        details = ", ".join(f"{s.path} ({s.state})" for s in problems)
        raise ExperimentError(f"Données manquantes ou modifiées : {details}. Rien n'est exécuté.")


# --- experiments ---------------------------------------------------------------

def builtin_code(model: str, params: dict | None = None) -> str:
    return (
        "from forge.tabular import run_builtin\n\n"
        f"run_builtin({model!r}, {json.dumps(params or {}, sort_keys=True)})\n"
    )


def submit_experiment(project: Project, card: dict, code: str) -> Experiment:
    """Register an experiment with its code; code and data fingerprints are set by ML Forge."""
    protocol = get_protocol(project, card.get("protocol_id", ""))
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    _refuse_rejected_retest(project, protocol.id, card, code_hash)
    with session(project.db_path) as conn, transaction(conn):
        conn.execute("INSERT OR IGNORE INTO experiment_code (code_hash, content) VALUES (?, ?)", (code_hash, code))
    card = {**card, "code_hash": code_hash, "data_hash": dataset_hash(project, protocol.task)}
    return register_experiment(project.db_path, card)


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def _refuse_rejected_retest(project: Project, protocol_id: str, card: dict, code_hash: str) -> None:
    """An hypothesis already rejected under this protocol is not retested without a new, written reason."""
    if str(card.get("new_reason", "")).strip() or card.get("role") == "stability":
        return
    with session(project.db_path) as conn:
        rows = conn.execute(
            "SELECT e.id, e.hypothesis, e.card_json, d.status, d.delta FROM experiments e"
            " JOIN experiment_decisions d ON d.experiment_id = e.id"
            " WHERE e.protocol_id = ? AND d.status IN ('rejected', 'not_significant')",
            (protocol_id,),
        ).fetchall()
    for row in rows:
        same_code = json.loads(row["card_json"]).get("code_hash") == code_hash
        if same_code or _normalise(row["hypothesis"]) == _normalise(card.get("hypothesis", "")):
            raise ExperimentError(
                f"Hypothèse déjà testée et rejetée ({row['id']}, {row['status']}, Δ = {row['delta']:+.5f}). "
                "Pour la retester, fournissez une raison nouvelle dans le champ new_reason."
            )


def run_experiment(workspace: Workspace, project: Project, experiment_id: str) -> RunOutcome:
    experiment = get_experiment(project.db_path, experiment_id)
    protocol = get_protocol(project, experiment.protocol_id)
    _check_prerequisites(project, experiment, protocol)
    seed = int(experiment.card.get("seed", workspace.config.get("evaluation", "seed")))
    workdir = _prepare_workdir(project, experiment, protocol, seed)
    start_experiment(project.db_path, experiment.id)
    read_paths = (protocol.task.data_paths() + [Path(directory) for directory in protocol.task.media_dirs]
                  + _parent_outputs(project, experiment))
    result = run_in_sandbox(workdir / "script.py", workdir, read_paths, _limits(workspace),
                            workspace.config.get("sandbox", "python"))
    log_event(project.db_path, experiment.id, "resources",
         f"durée {result.duration_s} s, mémoire max {result.peak_memory_mb} Mo, statut {result.status}")
    if not result.ok:
        return _fail(project, experiment.id, workdir, f"Bac à sable : {result.status}. {result.stderr[-1500:]}")
    try:
        scores = _score_outputs(workdir, protocol)
    except ExperimentError as exc:
        return _fail(project, experiment.id, workdir, str(exc))
    record_metrics(project.db_path, experiment.id, protocol.task.metric, "fold", scores)
    mean, std = float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    finish_experiment(project.db_path, experiment.id, "completed", f"{protocol.task.metric} = {mean:.5f} ± {std:.5f}")
    if experiment.card.get("role") == "stability":  # a re-run with another seed: measured, never a candidate
        _retain(workdir, keep=False)
        return RunOutcome(experiment.id, "completed", None, mean, std, None, "Contrôle de stabilité.")
    status, delta, rule = _record_decision(workspace, project, experiment, protocol, scores, workdir)
    _retain(workdir, keep=status in KEPT_DECISIONS)
    from forge.knowledge import learn_from_decision

    learn_from_decision(workspace, project, experiment.id, status, delta)
    from forge.lessons import record_from_decision

    record_from_decision(workspace, project, experiment.id, status, delta)
    _refresh_report(project)
    return RunOutcome(experiment.id, "completed", status, mean, std, delta, rule)


def _parent_outputs(project: Project, experiment: Experiment) -> list[Path]:
    directory = project.path / "experiments" / experiment.parent_id if experiment.parent_id else None
    return [directory] if directory and directory.is_dir() else []


def _check_prerequisites(project: Project, experiment: Experiment, protocol: Protocol) -> None:
    _require_clean_sources(project, protocol.task)
    if experiment.card.get("data_hash") != dataset_hash(project, protocol.task):
        raise ExperimentError("Les données ont changé depuis l'enregistrement de l'expérience : réenregistrez-la.")
    if latest_profile(project, experiment.card["data_hash"]) is None:
        raise ExperimentError("Profil des données absent : lancez le profilage avant toute expérience.")
    if experiment.card.get("role") != "baseline" and current_best(project, protocol.id) is None:
        raise ExperimentError("Aucune baseline terminée pour ce protocole : exécutez d'abord une baseline.")


def _prepare_workdir(project: Project, experiment: Experiment, protocol: Protocol, seed: int) -> Path:
    workdir = project.path / "experiments" / experiment.id
    if workdir.exists():  # leftovers of an interrupted run of this same experiment
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    with session(project.db_path) as conn:
        code = conn.execute(
            "SELECT content FROM experiment_code WHERE code_hash = ?", (experiment.card["code_hash"],)
        ).fetchone()
    if code is None:
        raise ExperimentError(f"Code de l'expérience {experiment.id} introuvable.")
    (workdir / "script.py").write_text(code["content"], encoding="utf-8")
    task = protocol.task.to_dict()
    for key in ("train_path", "test_path"):
        task[key] = str(Path(task[key]).resolve()) if task[key] else None
    if task["prediction_sets"]:
        task["prediction_sets"] = {name: str(Path(path).resolve()) for name, path in task["prediction_sets"].items()}
    task["media_dirs"] = [str(Path(directory).resolve()) for directory in task["media_dirs"]]
    context = {
        "task": task,
        "folds": protocol.folds,
        "classes": protocol.classes,
        "seed": seed,
        "fold_scheme": protocol.scheme,
        "parent_dir": str(project.path / "experiments" / experiment.parent_id) if experiment.parent_id else None,
        "outputs": {"oof": "oof.csv", "test": "test_pred.csv", "info": "info.json",
                    "sets": protocol.task.prediction_files()},
    }
    (workdir / "context.json").write_text(json.dumps(context), encoding="utf-8")
    return workdir


def _limits(workspace: Workspace) -> SandboxLimits:
    config = workspace.config
    return SandboxLimits(
        timeout_s=config.get("sandbox", "run_timeout_minutes") * 60,
        memory_mb=int(config.get("sandbox", "memory_limit_gb") * 1024),
    )


def _score_outputs(workdir: Path, protocol: Protocol) -> list[float]:
    path = workdir / "oof.csv"
    if not path.is_file():
        raise ExperimentError("Le script n'a pas produit oof.csv (prédictions hors pli).")
    oof = pd.read_csv(path)
    task = protocol.task
    expected = [PROBA_PREFIX + c for c in protocol.classes] if protocol.classes else ["prediction"]
    if list(oof.columns) != expected:
        raise ExperimentError(f"Colonnes de oof.csv : {list(oof.columns)} ; attendues : {expected}.")
    if len(oof) != len(protocol.folds):
        raise ExperimentError(f"oof.csv contient {len(oof)} lignes au lieu de {len(protocol.folds)}.")
    if oof.isna().any().any() or not all(pd.api.types.is_numeric_dtype(oof[c]) for c in oof.columns):
        raise ExperimentError("oof.csv contient des valeurs manquantes ou non numériques.")
    y = read_table(task.train_path, usecols=[task.target])[task.target]
    return fold_scores(task, y, oof, protocol.folds)


def _record_decision(
    workspace: Workspace, project: Project, experiment: Experiment, protocol: Protocol, scores: list[float],
    workdir: Path,
) -> tuple[str, float | None, str]:
    mean, std = float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    reference_id = None
    if experiment.card.get("role") == "baseline" and current_best(project, protocol.id) is None:
        status, delta, noise, rule = "baseline", None, None, "Première référence du protocole."
    else:
        reference_id = current_best(project, protocol.id)
        reference = get_metrics(project.db_path, reference_id)[protocol.task.metric]["fold"]
        reliable = protocol.reliability["reliable"]
        second = None if reliable else _second_regime(project, protocol, workdir, reference_id)
        result = decide(
            scores, reference, protocol.task.higher_is_better,
            workspace.config.get("evaluation", "acceptance_std_factor"), reliable, second,
        )
        status, delta, noise, rule = result.status, result.delta, result.noise, f"{result.rule} (réf. {reference_id})"
        if second is not None:
            rule += f" Second régime (simulation des conditions du test) : Δ = {second:+.5f}."
    rule += _suspicion(protocol, scores)
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO experiment_decisions (experiment_id, reference_id, status, metric_mean, metric_std,"
            " delta, noise, rule, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment.id, reference_id, status, mean, std, delta, noise, rule, now_iso()),
        )
    return status, delta, rule


def _second_regime(project: Project, protocol: Protocol, workdir: Path, reference_id: str) -> float | None:
    reference = project.path / "experiments" / reference_id / "oof.csv"
    if not reference.is_file() or not (workdir / "oof.csv").is_file():
        return None
    task = protocol.task
    y = read_table(task.train_path, usecols=[task.target])[task.target]
    profile = latest_profile(project, dataset_hash(project, task)) or {}
    test_size = profile.get("test_rows") or max(len(y) // 5, 2)
    return simulated_delta(task, y, pd.read_csv(workdir / "oof.csv"), pd.read_csv(reference), protocol.folds, test_size)


def _suspicion(protocol: Protocol, scores: list[float]) -> str:
    """A near-perfect score on a bounded metric is flagged: it often reveals a leak."""
    bounded = protocol.task.metric in {"accuracy", "balanced_accuracy", "f1", "f1_macro", "roc_auc", "r2"}
    if bounded and float(np.mean(scores)) >= 0.999:
        return " ATTENTION : score quasi parfait, fuite de la cible possible à vérifier."
    return ""


def current_best(project: Project, protocol_id: str) -> str | None:
    """The latest baseline or accepted experiment of a protocol: each one beat the previous best."""
    with session(project.db_path) as conn:
        row = conn.execute(
            "SELECT d.experiment_id FROM experiment_decisions d JOIN experiments e ON e.id = d.experiment_id"
            " WHERE e.protocol_id = ? AND d.status IN ('baseline', 'accepted') ORDER BY d.rowid DESC LIMIT 1",
            (protocol_id,),
        ).fetchone()
    return row["experiment_id"] if row else None


def compare(project: Project, first_id: str, second_id: str, std_factor: float = 1.0) -> dict:
    """Compare two experiments; refused unless they share the same validation protocol."""
    first, second = get_experiment(project.db_path, first_id), get_experiment(project.db_path, second_id)
    if first.protocol_id != second.protocol_id:
        raise ExperimentError(
            f"{first_id} ({first.protocol_id}) et {second_id} ({second.protocol_id}) n'ont pas le même protocole : "
            "comparaison refusée."
        )
    protocol = get_protocol(project, first.protocol_id)
    metric = protocol.task.metric
    scores = [get_metrics(project.db_path, e.id).get(metric, {}).get("fold") for e in (first, second)]
    if not all(scores):
        raise ExperimentError("Les deux expériences doivent être terminées pour être comparées.")
    result = decide(scores[1], scores[0], protocol.task.higher_is_better, std_factor, protocol.reliability["reliable"])
    return {
        "metric": metric,
        first_id: {"mean": float(np.mean(scores[0])), "folds": scores[0]},
        second_id: {"mean": float(np.mean(scores[1])), "folds": scores[1]},
        "verdict": f"{second_id} vs {first_id} : {result.status}. {result.rule}",
    }


def _fail(project: Project, experiment_id: str, workdir: Path, message: str) -> RunOutcome:
    finish_experiment(project.db_path, experiment_id, "failed", message[:4000])
    _retain(workdir, keep=False)
    return RunOutcome(experiment_id, "failed", None, None, None, None, message)


def _retain(workdir: Path, keep: bool) -> None:
    """Keep outputs of the baseline and accepted experiments only; the code stays in the database."""
    if keep:
        shutil.rmtree(workdir / "tmp", ignore_errors=True)
    else:
        shutil.rmtree(workdir, ignore_errors=True)
        parent = workdir.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()


def _refresh_report(project: Project) -> None:
    from forge.report import write_report  # report reads this module's helpers

    write_report(project)
