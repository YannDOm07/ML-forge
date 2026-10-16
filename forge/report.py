"""Synthetic Markdown report, rewritten in place from the database after each experiment."""

from __future__ import annotations

from forge.db import now_iso, session
from forge.experiments import list_experiments
from forge.projects import Project
from forge.runner import current_best, latest_profile, latest_protocol

REPORT_FILE = "report.md"


def decisions_by_experiment(project: Project) -> dict[str, dict]:
    with session(project.db_path) as conn:
        rows = conn.execute("SELECT * FROM experiment_decisions").fetchall()
    return {row["experiment_id"]: dict(row) for row in rows}


def write_report(project: Project) -> None:
    lines = [f"# {project.name}", "", f"*Mis à jour le {now_iso()} — l'historique complet est dans project.db.*", ""]
    if project.competition_url:
        lines += [f"Compétition : {project.competition_url}", ""]
    protocol = latest_protocol(project)
    if protocol is None:
        lines += ["Aucun protocole de validation défini."]
    else:
        lines += _protocol_section(project, protocol) + _results_section(project, protocol)
    (project.path / REPORT_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _protocol_section(project: Project, protocol) -> list[str]:
    task = protocol.task
    lines = [
        "## Tâche et validation",
        f"- Problème : {task.problem_type}, cible `{task.target}`, métrique **{task.metric}**"
        f" ({'plus haut' if task.higher_is_better else 'plus bas'} = meilleur)",
        f"- Protocole {protocol.id} : " + " ".join(protocol.notes),
        "- Validation " + ("fiable" if protocol.reliability["reliable"] else "**peu fiable**")
        + "".join(f" — {reason}" for reason in protocol.reliability["reasons"]),
    ]
    with session(project.db_path) as conn:
        row = conn.execute("SELECT data_hash FROM data_profiles ORDER BY id DESC LIMIT 1").fetchone()
    profile = latest_profile(project, row["data_hash"]) if row else None
    if profile:
        twins = profile["twins"]
        lines += [
            f"- Données : {profile['rows']} lignes, {profile['features']} variables"
            + (" (profil sur échantillon)" if profile["truncated"] else ""),
            f"- Jumeaux contradictoires : {twins['groups']} groupe(s), plancher d'erreur {twins['error_floor']}",
        ]
        lines += [f"- Fuite possible : {hint}" for hint in profile["leak_hints"]]
    return lines + [""]


def _results_section(project: Project, protocol) -> list[str]:
    decisions = decisions_by_experiment(project)
    best = current_best(project, protocol.id)
    lines = ["## Résultats", ""]
    if best:
        decision = decisions[best]
        lines += [f"**Meilleur candidat : {best}** — {decision['metric_mean']:.5f} ± {decision['metric_std']:.5f}", ""]
    lines += ["| Exp. | Méthode | Statut | Score | Δ | Décision |", "|---|---|---|---|---|---|"]
    for experiment in list_experiments(project.db_path):
        if experiment.protocol_id != protocol.id:
            continue
        decision = decisions.get(experiment.id)
        score = f"{decision['metric_mean']:.5f} ± {decision['metric_std']:.5f}" if decision else "—"
        delta = f"{decision['delta']:+.5f}" if decision and decision["delta"] is not None else "—"
        verdict = decision["status"] if decision else "—"
        lines.append(f"| {experiment.id} | {experiment.method} | {experiment.status} | {score} | {delta} | {verdict} |")
    return lines
