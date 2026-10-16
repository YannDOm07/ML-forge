"""Approvals given out of band by the user (CLI), never by the LLM.

When the agent asks for a guarded action, the request is stored as pending and the agent is
told which command the user must run. The approval is consumed once, by the same action
on the same target.
"""

from __future__ import annotations

import json

from forge.db import now_iso, session, transaction
from forge.errors import ApprovalRequired, ForgeError
from forge.guardrails import ApprovalRequest, Approver
from forge.projects import Project


class ApprovalPending(ApprovalRequired):
    """The user has not answered yet; nothing happens until they do."""


def stored_approver(project: Project) -> Approver:
    def approver(request: ApprovalRequest) -> bool:
        with session(project.db_path) as conn, transaction(conn):
            row = conn.execute(
                "SELECT id, status FROM approval_requests WHERE action = ? AND target = ? AND status != 'used'"
                " ORDER BY id DESC LIMIT 1",
                (str(request.action), request.target),
            ).fetchone()
            if row is None:
                cursor = conn.execute(
                    "INSERT INTO approval_requests (action, target, details_json, status, created_at)"
                    " VALUES (?, ?, ?, 'pending', ?)",
                    (str(request.action), request.target, json.dumps(request.details, ensure_ascii=False), now_iso()),
                )
                request_id = cursor.lastrowid
            elif row["status"] == "pending":
                request_id = row["id"]
            else:
                if row["status"] == "approved":
                    conn.execute("UPDATE approval_requests SET status = 'used' WHERE id = ?", (row["id"],))
                return row["status"] == "approved"
        raise ApprovalPending(
            f"Accord de l'utilisateur requis (demande n° {request_id}) : {request.describe()}\n"
            f"L'utilisateur doit lancer : python -m forge approve {project.id} {request_id} "
            f"(ou refuse). En attendant, poursuivez les pistes qui n'en dépendent pas."
        )

    return approver


def pending_requests(project: Project) -> list[dict]:
    with session(project.db_path) as conn:
        rows = conn.execute("SELECT * FROM approval_requests WHERE status = 'pending' ORDER BY id").fetchall()
    return [dict(row) | {"details": json.loads(row["details_json"])} for row in rows]


def answer_request(project: Project, request_id: int, approved: bool) -> None:
    with session(project.db_path) as conn, transaction(conn):
        row = conn.execute("SELECT status FROM approval_requests WHERE id = ?", (request_id,)).fetchone()
        if row is None or row["status"] != "pending":
            raise ForgeError(f"Aucune demande en attente n° {request_id}.")
        conn.execute(
            "UPDATE approval_requests SET status = ?, decided_at = ? WHERE id = ?",
            ("approved" if approved else "refused", now_iso(), request_id),
        )
