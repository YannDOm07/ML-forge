"""The closed list of actions that require explicit user approval (prompt_v2 §8)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from forge.db import now_iso, session, transaction
from forge.errors import ApprovalRequired

logger = logging.getLogger(__name__)


class Action(StrEnum):
    EXTERNAL_DATA = "external_data"
    PRETRAINED_MODEL = "pretrained_model"
    HEAVY_DEPENDENCY = "heavy_dependency"
    DELETE = "delete"
    REPLACE_FINAL_DELIVERABLE = "replace_final_deliverable"


ACTION_LABELS = {
    Action.EXTERNAL_DATA: "utiliser une donnée externe",
    Action.PRETRAINED_MODEL: "utiliser un modèle préentraîné",
    Action.HEAVY_DEPENDENCY: "installer une dépendance lourde",
    Action.DELETE: "supprimer",
    Action.REPLACE_FINAL_DELIVERABLE: "remplacer un livrable final validé",
}


@dataclass(frozen=True)
class ApprovalRequest:
    action: Action
    target: str
    details: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [f"{ACTION_LABELS[self.action]} : {self.target}"]
        lines += [f"  {key} : {value}" for key, value in self.details.items()]
        return "\n".join(lines)


Approver = Callable[[ApprovalRequest], bool]
DecisionRecorder = Callable[[ApprovalRequest, bool], None]


def refuse_all(request: ApprovalRequest) -> bool:
    """Default approver for unattended runs: nothing guarded happens without a human."""
    return False


def require_approval(
    request: ApprovalRequest, approver: Approver, record: DecisionRecorder | None = None
) -> None:
    """Raise ApprovalRequired unless the approver explicitly returns True."""
    approved = approver(request) is True
    logger.info("Décision %s pour %s : %s", request.action, request.target, approved)
    if record is not None:
        record(request, approved)
    if not approved:
        raise ApprovalRequired(f"Action refusée faute d'accord explicite : {request.describe()}")


def project_decision_recorder(project_db: Path, decided_by: str) -> DecisionRecorder:
    def record(request: ApprovalRequest, approved: bool) -> None:
        with session(project_db) as conn, transaction(conn):
            conn.execute(
                "INSERT INTO decisions (action, target, details_json, outcome, decided_by, at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    str(request.action),
                    request.target,
                    json.dumps(request.details, ensure_ascii=False),
                    "approved" if approved else "refused",
                    decided_by,
                    now_iso(),
                ),
            )

    return record
