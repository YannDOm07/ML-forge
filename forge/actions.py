"""Actions awaited from the user (accept rules, deposit files, paste text), kept until resolved."""

from __future__ import annotations

from forge.db import now_iso, session, transaction
from forge.projects import Project


def open_action(project: Project, kind: str, message: str) -> None:
    """Record what the user must do; an identical open action is not duplicated."""
    with session(project.db_path) as conn, transaction(conn):
        exists = conn.execute(
            "SELECT 1 FROM user_actions WHERE kind = ? AND message = ? AND status = 'open'", (kind, message)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO user_actions (kind, message, status, created_at) VALUES (?, ?, 'open', ?)",
                (kind, message, now_iso()),
            )


def resolve_actions(project: Project, kind: str) -> None:
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "UPDATE user_actions SET status = 'done', resolved_at = ? WHERE kind = ? AND status = 'open'",
            (now_iso(), kind),
        )


def open_actions(project: Project) -> list[dict]:
    with session(project.db_path) as conn:
        rows = conn.execute("SELECT * FROM user_actions WHERE status = 'open' ORDER BY id").fetchall()
    return [dict(row) for row in rows]
