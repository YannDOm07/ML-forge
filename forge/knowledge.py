"""General memory shared by all projects, built only from measured, computed decisions.

What is remembered: for each family of hypotheses (e.g. "modèle:random_forest", "features:dates")
and each data context (problem type, size, imbalance, metric), how often a change was accepted,
the mean measured Δ, and the projects and experiments behind it. Also, the gap between the gain
the agent expected and the gain measured (calibration). Nothing is learned from generated text,
and the language model itself is never retrained.
"""

from __future__ import annotations

import json

from forge.db import now_iso, session, transaction
from forge.experiments import get_experiment
from forge.projects import Project
from forge.workspace import Workspace

MIN_CALIBRATION_SAMPLES = 3


def context_of(problem_type: str, metric: str, rows: int, imbalance: float | None) -> dict:
    size = "<1k" if rows < 1_000 else "1k-10k" if rows < 10_000 else "10k-100k" if rows < 100_000 else ">100k"
    if imbalance is None:
        balance = "n/a"
    else:
        balance = "équilibré" if imbalance < 3 else "déséquilibré" if imbalance < 10 else "très déséquilibré"
    return {"problem_type": problem_type, "metric": metric, "size": size, "balance": balance}


def context_key(context: dict) -> str:
    return "|".join(context[key] for key in ("problem_type", "metric", "size", "balance"))


def family_of(card: dict) -> str:
    return str(card.get("family") or card.get("method", "autre")).strip().lower()


def project_context(project: Project) -> dict | None:
    from forge.runner import dataset_hash, latest_profile, latest_protocol

    protocol = latest_protocol(project)
    if protocol is None:
        return None
    profile = latest_profile(project, dataset_hash(project, protocol.task)) or {}
    imbalance = profile.get("target", {}).get("imbalance_ratio")
    return context_of(protocol.task.problem_type, protocol.task.metric, profile.get("rows", 0), imbalance)


def learn_from_decision(workspace: Workspace, project: Project, experiment_id: str, status: str, delta: float | None) -> None:
    """Record the measured outcome of a compared experiment as evidence and calibration."""
    if status not in {"accepted", "rejected", "not_significant"} or delta is None:
        return
    context = project_context(project)
    if context is None:
        return
    card = get_experiment(project.db_path, experiment_id).card
    family = family_of(card)
    key = context_key(context)
    with session(workspace.global_db) as conn, transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO knowledge_items (family, context_key, context_json, statement, n_supporting,"
            " n_contradicting, n_projects, mean_delta, confidence, updated_at) VALUES (?, ?, ?, '', 0, 0, 0, 0, 0, ?)",
            (family, key, json.dumps(context, ensure_ascii=False), now_iso()),
        )
        item_id = conn.execute(
            "SELECT id FROM knowledge_items WHERE family = ? AND context_key = ?", (family, key)
        ).fetchone()["id"]
        conn.execute(
            "INSERT OR IGNORE INTO knowledge_evidence (knowledge_id, project_id, experiment_id, delta, supports, recorded_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (item_id, project.id, experiment_id, delta, int(status == "accepted"), now_iso()),
        )
        _refresh_item(conn, item_id, family, context)
        expected = card.get("estimate", {}).get("expected_gain")
        if isinstance(expected, (int, float)):
            conn.execute(
                "INSERT OR IGNORE INTO estimate_calibration (project_id, experiment_id, family, expected_gain,"
                " observed_gain, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                (project.id, experiment_id, family, float(expected), delta, now_iso()),
            )


def _refresh_item(conn, item_id: int, family: str, context: dict) -> None:
    stats = conn.execute(
        "SELECT SUM(supports) AS yes, COUNT(*) AS n, AVG(delta) AS mean, COUNT(DISTINCT project_id) AS projects"
        " FROM knowledge_evidence WHERE knowledge_id = ?",
        (item_id,),
    ).fetchone()
    yes, total = stats["yes"] or 0, stats["n"]
    confidence = round((yes + 1) / (total + 2), 3)  # Laplace: one observation is never a certainty
    statement = (
        f"{family} — {context['problem_type']}, {context['size']} lignes, {context['balance']}, {context['metric']} : "
        f"gain accepté {yes}/{total} fois sur {stats['projects']} projet(s), Δ moyen {stats['mean']:+.4f}. "
        "À vérifier sur le nouveau jeu de données."
    )
    conn.execute(
        "UPDATE knowledge_items SET statement = ?, n_supporting = ?, n_contradicting = ?, n_projects = ?,"
        " mean_delta = ?, confidence = ?, updated_at = ? WHERE id = ?",
        (statement, yes, total - yes, stats["projects"], stats["mean"], confidence, now_iso(), item_id),
    )


def memory_search(workspace: Workspace, context: dict, exclude_project: str | None = None, limit: int = 5) -> list[dict]:
    """Lessons for the same context first, then the same problem type; ranked by evidence."""
    with session(workspace.global_db) as conn:
        rows = conn.execute(
            "SELECT k.*, COUNT(e.id) AS evidence FROM knowledge_items k JOIN knowledge_evidence e ON e.knowledge_id = k.id"
            " WHERE e.project_id != ? GROUP BY k.id",
            (exclude_project or "",),
        ).fetchall()
    items = [dict(row) for row in rows if json.loads(row["context_json"])["problem_type"] == context["problem_type"]]
    key = context_key(context)
    items.sort(key=lambda item: (item["context_key"] == key, item["confidence"] * item["evidence"]), reverse=True)
    return [
        {"famille": i["family"], "leçon": i["statement"], "confiance": i["confidence"], "preuves": i["evidence"],
         "même_contexte": i["context_key"] == key}
        for i in items[:limit]
    ]


def calibration(workspace: Workspace) -> dict[str, dict]:
    """Per family: ratio observed/expected gain, once enough samples exist."""
    with session(workspace.global_db) as conn:
        rows = conn.execute(
            "SELECT family, COUNT(*) AS n, SUM(expected_gain) AS expected, SUM(observed_gain) AS observed"
            " FROM estimate_calibration GROUP BY family"
        ).fetchall()
    result = {}
    for row in rows:
        if row["n"] >= MIN_CALIBRATION_SAMPLES and row["expected"] > 0:
            ratio = max(0.0, min(2.0, row["observed"] / row["expected"]))
            result[row["family"]] = {"échantillons": row["n"], "ratio_observé_sur_estimé": round(ratio, 3)}
    return result


def rank_hypotheses(workspace: Workspace, candidates: list[dict]) -> list[dict]:
    """Order candidate hypotheses by expected gain × probability ÷ cost, the gain corrected by calibration."""
    factors = calibration(workspace)
    risk_penalty = {"low": 1.0, "medium": 0.7, "high": 0.4}
    ranked = []
    for candidate in candidates:
        estimate = candidate["estimate"]
        factor = factors.get(family_of(candidate), {}).get("ratio_observé_sur_estimé", 1.0)
        gain = estimate["expected_gain"] * factor
        priority = gain * estimate["success_probability"] / max(estimate["cost_minutes"], 1) * risk_penalty[estimate["risk"]]
        ranked.append({**candidate, "gain_corrigé": round(gain, 6), "priorité": round(priority, 8), "facteur_calibration": factor})
    return sorted(ranked, key=lambda c: c["priorité"], reverse=True)


def memory_digest(workspace: Workspace, project: Project) -> dict:
    context = project_context(project)
    if context is None:
        return {}
    return {
        "leçons_autres_projets": memory_search(workspace, context, exclude_project=project.id),
        "calibration_des_estimations": calibration(workspace),
    }
