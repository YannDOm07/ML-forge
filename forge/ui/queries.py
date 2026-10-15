"""Read-only SQLite queries for ML Forge UI.

This is the ONLY module that knows the database schema.
Opens databases in read-only mode (file:...?mode=ro) via WAL.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from forge.workspace import Workspace


def _connect_ro(path: Path) -> sqlite3.Connection:
    """Open SQLite in read-only mode via URI."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


# ---------- Global DB (forge.db) ----------

def list_projects(global_db: Path) -> list[dict[str, Any]]:
    """All projects from global registry."""
    with _connect_ro(global_db) as conn:
        rows = conn.execute(
            "SELECT id, name, competition_url, status, created_at, updated_at "
            "FROM projects WHERE status IN ('active', 'archived') ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_project(global_db: Path, project_id: str) -> dict[str, Any] | None:
    """Single project from global registry."""
    with _connect_ro(global_db) as conn:
        row = conn.execute(
            "SELECT id, name, competition_url, status, created_at, updated_at "
            "FROM projects WHERE id = ? AND status != 'deleted'", (project_id,)
        ).fetchone()
    return dict(row) if row else None


def get_knowledge_items(global_db: Path) -> list[dict[str, Any]]:
    """All knowledge items for memory screen."""
    with _connect_ro(global_db) as conn:
        rows = conn.execute(
            "SELECT id, family, context_key, context_json, statement, "
            "n_supporting, n_contradicting, n_projects, mean_delta, confidence, updated_at "
            "FROM knowledge_items ORDER BY confidence DESC, updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_knowledge_evidence(global_db: Path, project_id: str) -> list[dict[str, Any]]:
    """Evidence for knowledge items in a specific project."""
    with _connect_ro(global_db) as conn:
        rows = conn.execute(
            "SELECT ke.*, ki.statement, ki.family "
            "FROM knowledge_evidence ke "
            "JOIN knowledge_items ki ON ki.id = ke.knowledge_id "
            "WHERE ke.project_id = ? ORDER BY ke.recorded_at DESC",
            (project_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_estimate_calibration(global_db: Path, project_id: str) -> list[dict[str, Any]]:
    """Calibration history for a project."""
    with _connect_ro(global_db) as conn:
        rows = conn.execute(
            "SELECT project_id, experiment_id, family, expected_gain, observed_gain, recorded_at "
            "FROM estimate_calibration WHERE project_id = ? ORDER BY recorded_at DESC",
            (project_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_configuration_history(global_db: Path) -> list[dict[str, Any]]:
    """Global configuration history."""
    with _connect_ro(global_db) as conn:
        rows = conn.execute(
            "SELECT id, config_hash, content, recorded_at "
            "FROM configuration_history ORDER BY recorded_at DESC LIMIT 50"
        ).fetchall()
    return [dict(row) for row in rows]


# ---------- Project DB (project.db) ----------

def get_data_sources(project_db: Path) -> list[dict[str, Any]]:
    """Data sources registered for the project."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, role, path, sha256, size_bytes, mtime_ns, origin_url, license, registered_at "
            "FROM data_sources ORDER BY registered_at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_experiments(project_db: Path) -> list[dict[str, Any]]:
    """All experiments with their cards."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, parent_id, question, hypothesis, method, protocol_id, "
            "card_json, code_hash, data_hash, status, runner_pid, "
            "created_at, started_at, finished_at, runner_created "
            "FROM experiments ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_experiment(project_db: Path, experiment_id: str) -> dict[str, Any] | None:
    """Single experiment by ID."""
    with _connect_ro(project_db) as conn:
        row = conn.execute(
            "SELECT id, parent_id, question, hypothesis, method, protocol_id, "
            "card_json, code_hash, data_hash, status, runner_pid, "
            "created_at, started_at, finished_at, runner_created "
            "FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
    return dict(row) if row else None


def get_experiment_metrics(project_db: Path, experiment_id: str) -> list[dict[str, Any]]:
    """Metrics for a single experiment."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT name, scope, idx, value "
            "FROM experiment_metrics WHERE experiment_id = ? "
            "ORDER BY scope, idx",
            (experiment_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_experiment_metrics(project_db: Path) -> list[dict[str, Any]]:
    """All metrics for all experiments (for charts)."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT experiment_id, name, scope, idx, value "
            "FROM experiment_metrics ORDER BY experiment_id, scope, idx"
        ).fetchall()
    return [dict(row) for row in rows]


def get_experiment_decisions(project_db: Path) -> list[dict[str, Any]]:
    """Computed decisions for all experiments."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT experiment_id, reference_id, status, metric_mean, metric_std, "
            "delta, noise, rule, computed_at "
            "FROM experiment_decisions ORDER BY computed_at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_experiment_decision(project_db: Path, experiment_id: str) -> dict[str, Any] | None:
    """Computed decision for a single experiment."""
    with _connect_ro(project_db) as conn:
        row = conn.execute(
            "SELECT experiment_id, reference_id, status, metric_mean, metric_std, "
            "delta, noise, rule, computed_at "
            "FROM experiment_decisions WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
    return dict(row) if row else None


def get_latest_protocol(project_db: Path) -> dict[str, Any] | None:
    """Latest protocol for the project."""
    with _connect_ro(project_db) as conn:
        row = conn.execute(
            "SELECT id, spec_json, folds_json, reliability_json, created_at "
            "FROM protocols ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_latest_data_profile(project_db: Path, data_hash: str) -> dict[str, Any] | None:
    """Latest data profile for a given data hash."""
    with _connect_ro(project_db) as conn:
        row = conn.execute(
            "SELECT id, data_hash, profile_json, created_at "
            "FROM data_profiles WHERE data_hash = ? ORDER BY created_at DESC LIMIT 1",
            (data_hash,)
        ).fetchone()
    return dict(row) if row else None


def get_competition_facts(project_db: Path) -> list[dict[str, Any]]:
    """All competition facts."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, field, value_json, status, source_url, quote, recorded_by, recorded_at "
            "FROM competition_facts ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def get_web_pages(project_db: Path) -> list[dict[str, Any]]:
    """All web pages read for this project."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, url, origin, status, title, text, links_json, flags_json, content_hash, fetched_at "
            "FROM web_pages ORDER BY fetched_at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_page_index(project_db: Path) -> list[dict[str, Any]]:
    """Latest version of each page read: metadata and size only (text stays on the server)."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT url, origin, status, title, length(text) AS chars, flags_json, fetched_at, "
            "substr(text, 1, 700) AS head FROM web_pages "
            "WHERE id IN (SELECT MAX(id) FROM web_pages GROUP BY url) ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def get_data_source_rows(project_db: Path) -> list[tuple[str, str, int]]:
    """(role, path, size) of every registered file, aggregated by the caller, never sent whole."""
    with _connect_ro(project_db) as conn:
        return conn.execute("SELECT role, path, size_bytes FROM data_sources").fetchall()


def get_data_source_version(project_db: Path) -> tuple[int, int]:
    """(count, last id) of the registered files: changes whenever a file is registered."""
    with _connect_ro(project_db) as conn:
        row = conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM data_sources").fetchone()
    return int(row[0]), int(row[1])


TABLE_SUFFIX_SQL = " OR ".join(f"lower(path) LIKE '%{suffix}'" for suffix in
                               (".csv", ".tsv", ".json", ".jsonl", ".parquet", ".feather"))


def get_data_sources_shown(project_db: Path, limit: int) -> list[dict[str, Any]]:
    """The files worth listing: tables first, then the largest ones."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, role, path, sha256, size_bytes, license FROM data_sources "
            f"ORDER BY CASE WHEN {TABLE_SUFFIX_SQL} THEN 0 ELSE 1 END, size_bytes DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_lessons(global_db: Path) -> list[sqlite3.Row]:
    """Lessons register (techniques, mistakes, traps); empty before the v4 schema."""
    try:
        with _connect_ro(global_db) as conn:
            return conn.execute("SELECT * FROM lessons").fetchall()
    except sqlite3.OperationalError:
        return []


def get_compliance_log(project_db: Path) -> list[dict[str, Any]]:
    """Compliance log entries."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, item, justification, source, at "
            "FROM compliance_log ORDER BY at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_user_actions(project_db: Path) -> list[dict[str, Any]]:
    """User actions (open/done)."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, kind, message, status, created_at, resolved_at "
            "FROM user_actions ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def get_campaigns(project_db: Path) -> list[dict[str, Any]]:
    """All campaigns for the project."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, mode, driver, status, budget_hours, max_experiments, "
            "started_at, ended_at, reason, runner_pid, runner_created "
            "FROM campaigns ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_active_campaign(project_db: Path) -> dict[str, Any] | None:
    """Currently active campaign (running/paused/waiting_user)."""
    with _connect_ro(project_db) as conn:
        row = conn.execute(
            "SELECT id, mode, driver, status, budget_hours, max_experiments, "
            "started_at, ended_at, reason, runner_pid, runner_created "
            "FROM campaigns "
            "WHERE status IN ('running', 'paused', 'waiting_user') "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_execution_events(project_db: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Execution events for the log screen."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, experiment_id, kind, message, at "
            "FROM execution_events ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_approval_requests(project_db: Path) -> list[dict[str, Any]]:
    """Approval requests."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, action, target, details_json, status, created_at, decided_at "
            "FROM approval_requests ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_submissions(project_db: Path) -> list[dict[str, Any]]:
    """Declared submissions."""
    with _connect_ro(project_db) as conn:
        rows = conn.execute(
            "SELECT id, experiment_id, file_sha256, submitted_at, public_score, note "
            "FROM submissions ORDER BY submitted_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def get_network_log(global_db: Path, project_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Network access log."""
    with _connect_ro(global_db) as conn:
        if project_id:
            rows = conn.execute(
                "SELECT id, project_id, method, url, purpose, status, bytes, error, at "
                "FROM network_log WHERE project_id = ? ORDER BY id DESC LIMIT ?",
                (project_id, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, project_id, method, url, purpose, status, bytes, error, at "
                "FROM network_log ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
    return [dict(row) for row in rows]


# ---------- Helpers ----------

def parse_json_field(value: str | None) -> Any:
    """Safely parse a JSON field from SQLite."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def get_card_field(card_json: str, field: str) -> Any:
    """Extract a field from experiment card_json."""
    card = parse_json_field(card_json)
    if not card:
        return None
    return card.get(field)