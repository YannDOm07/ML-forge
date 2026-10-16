"""Experiment registry: card validation (annex A.2), status lifecycle, metrics."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.db import now_iso, session, transaction
from forge.errors import ExperimentError
from forge.system import process_alive, process_creation_time

REQUIRED_TEXT_FIELDS = ("question", "hypothesis", "method", "protocol_id")
JUSTIFICATION_KINDS = {"observed_fact", "domain_knowledge", "memory", "public_source"}
RISK_LEVELS = {"low", "medium", "high"}
METRIC_SCOPES = {"fold", "seed", "simulation"}
TRANSITIONS = {
    "registered": {"running"},
    "running": {"completed", "failed", "interrupted"},
    "interrupted": {"running"},
    "completed": set(),
    "failed": set(),
}


@dataclass(frozen=True)
class Experiment:
    id: str
    parent_id: str | None
    question: str
    hypothesis: str
    method: str
    protocol_id: str
    card: dict[str, Any]
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None


def validate_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalised copy of the card, or raise listing every problem."""
    problems = [f"{name} : texte obligatoire" for name in REQUIRED_TEXT_FIELDS if not _text(card.get(name))]
    problems += _justification_problems(card.get("justification"))
    problems += _estimate_problems(card.get("estimate"))
    for name in ("parent_experiment_id", "code_hash", "data_hash"):
        if card.get(name) is not None and not _text(card.get(name)):
            problems.append(f"{name} : texte attendu")
    if problems:
        raise ExperimentError("Fiche d'expérience invalide :\n- " + "\n- ".join(problems))
    return json.loads(json.dumps(card, ensure_ascii=False))


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _justification_problems(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["justification : au moins une justification est obligatoire"]
    problems = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or item.get("kind") not in JUSTIFICATION_KINDS:
            problems.append(f"justification[{index}].kind : attendu parmi {sorted(JUSTIFICATION_KINDS)}")
        elif not _text(item.get("text")):
            problems.append(f"justification[{index}].text : texte obligatoire")
    return problems


def _estimate_problems(value: object) -> list[str]:
    if not isinstance(value, Mapping):
        return ["estimate : estimation préalable obligatoire (gain, probabilité, coût, risque)"]
    problems = []
    if not _number(value.get("expected_gain")):
        problems.append("estimate.expected_gain : nombre attendu")
    probability = value.get("success_probability")
    if not _number(probability) or not 0 <= probability <= 1:
        problems.append("estimate.success_probability : nombre entre 0 et 1 attendu")
    cost = value.get("cost_minutes")
    if not _number(cost) or cost < 0:
        problems.append("estimate.cost_minutes : nombre positif attendu")
    if value.get("risk") not in RISK_LEVELS:
        problems.append(f"estimate.risk : attendu parmi {sorted(RISK_LEVELS)}")
    return problems


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def register_experiment(project_db: Path, card: Mapping[str, Any]) -> Experiment:
    card = validate_card(card)
    with session(project_db) as conn, transaction(conn):
        parent_id = card.get("parent_experiment_id")
        if parent_id and not conn.execute("SELECT 1 FROM experiments WHERE id = ?", (parent_id,)).fetchone():
            raise ExperimentError(f"Expérience parente inconnue : {parent_id}")
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        experiment_id = f"E{count + 1:04d}"
        conn.execute(
            "INSERT INTO experiments (id, parent_id, question, hypothesis, method, protocol_id, card_json,"
            " code_hash, data_hash, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'registered', ?)",
            (
                experiment_id,
                parent_id,
                card["question"],
                card["hypothesis"],
                card["method"],
                card["protocol_id"],
                json.dumps(card, ensure_ascii=False),
                card.get("code_hash"),
                card.get("data_hash"),
                now_iso(),
            ),
        )
        _log_event(conn, experiment_id, "registered", "Expérience enregistrée")
    return get_experiment(project_db, experiment_id)


def get_experiment(project_db: Path, experiment_id: str) -> Experiment:
    with session(project_db) as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise ExperimentError(f"Expérience inconnue : {experiment_id}")
    return _to_experiment(row)


def list_experiments(project_db: Path) -> list[Experiment]:
    with session(project_db) as conn:
        rows = conn.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    return [_to_experiment(row) for row in rows]


def _to_experiment(row: Any) -> Experiment:
    return Experiment(
        id=row["id"],
        parent_id=row["parent_id"],
        question=row["question"],
        hypothesis=row["hypothesis"],
        method=row["method"],
        protocol_id=row["protocol_id"],
        card=json.loads(row["card_json"]),
        status=row["status"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def start_experiment(project_db: Path, experiment_id: str, pid: int | None = None) -> None:
    pid = pid if pid is not None else os.getpid()
    with session(project_db) as conn, transaction(conn):
        _transition(conn, experiment_id, "running", "Exécution démarrée")
        conn.execute(
            "UPDATE experiments SET runner_pid = ?, runner_created = ?, started_at = ?, finished_at = NULL"
            " WHERE id = ?",
            (pid, process_creation_time(pid), now_iso(), experiment_id),
        )


def finish_experiment(project_db: Path, experiment_id: str, status: str, message: str) -> None:
    if status not in {"completed", "failed"}:
        raise ExperimentError(f"Statut final invalide : {status}")
    with session(project_db) as conn, transaction(conn):
        _transition(conn, experiment_id, status, message)
        conn.execute("UPDATE experiments SET finished_at = ? WHERE id = ?", (now_iso(), experiment_id))


def record_metrics(project_db: Path, experiment_id: str, name: str, scope: str, values: Sequence[float]) -> None:
    if scope not in METRIC_SCOPES:
        raise ExperimentError(f"Portée de métrique invalide : {scope}")
    with session(project_db) as conn, transaction(conn):
        if _status(conn, experiment_id) != "running":
            raise ExperimentError(f"{experiment_id} n'est pas en cours : métriques refusées.")
        try:
            conn.executemany(
                "INSERT INTO experiment_metrics (experiment_id, name, scope, idx, value) VALUES (?, ?, ?, ?, ?)",
                [(experiment_id, name, scope, index, float(value)) for index, value in enumerate(values)],
            )
        except sqlite3.IntegrityError as exc:
            raise ExperimentError(f"{experiment_id} : métrique « {name} » ({scope}) déjà enregistrée.") from exc


def get_metrics(project_db: Path, experiment_id: str) -> dict[str, dict[str, list[float]]]:
    with session(project_db) as conn:
        rows = conn.execute(
            "SELECT name, scope, value FROM experiment_metrics WHERE experiment_id = ? ORDER BY name, scope, idx",
            (experiment_id,),
        ).fetchall()
    metrics: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        metrics.setdefault(row["name"], {}).setdefault(row["scope"], []).append(row["value"])
    return metrics


def recover_interrupted(project_db: Path) -> list[str]:
    """Mark as interrupted the experiments left running by a process that no longer exists."""
    recovered = []
    with session(project_db) as conn, transaction(conn):
        rows = conn.execute(
            "SELECT id, runner_pid, runner_created FROM experiments WHERE status = 'running'"
        ).fetchall()
        for row in rows:
            if row["runner_pid"] and process_alive(row["runner_pid"], row["runner_created"]):
                continue
            _transition(conn, row["id"], "interrupted", "Arrêt brutal détecté à la réouverture du projet")
            recovered.append(row["id"])
    return recovered


def _status(conn: Any, experiment_id: str) -> str:
    row = conn.execute("SELECT status FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None:
        raise ExperimentError(f"Expérience inconnue : {experiment_id}")
    return row["status"]


def _transition(conn: Any, experiment_id: str, new_status: str, message: str) -> None:
    current = _status(conn, experiment_id)
    if new_status not in TRANSITIONS[current]:
        raise ExperimentError(f"{experiment_id} : passage de « {current} » à « {new_status} » interdit.")
    conn.execute("UPDATE experiments SET status = ? WHERE id = ?", (new_status, experiment_id))
    _log_event(conn, experiment_id, new_status, message)


def log_event(project_db: Path, experiment_id: str | None, kind: str, message: str) -> None:
    with session(project_db) as conn, transaction(conn):
        _log_event(conn, experiment_id, kind, message)


def _log_event(conn: Any, experiment_id: str | None, kind: str, message: str) -> None:
    conn.execute(
        "INSERT INTO execution_events (experiment_id, kind, message, at) VALUES (?, ?, ?, ?)",
        (experiment_id, kind, message, now_iso()),
    )
