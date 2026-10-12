"""Definition of a supervised task: what to predict, from which files, scored how."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from forge.errors import ForgeError

PROBLEM_TYPES = ("binary", "multiclass", "regression")
PROBA_PREFIX = "p:"  # prediction column of a class probability: "p:<class>"
METRICS = {
    # name: (higher_is_better, problem types, needs probabilities)
    "accuracy": (True, ("binary", "multiclass"), False),
    "balanced_accuracy": (True, ("binary", "multiclass"), False),
    "f1": (True, ("binary",), False),
    "f1_macro": (True, ("binary", "multiclass"), False),
    "roc_auc": (True, ("binary", "multiclass"), True),
    "log_loss": (False, ("binary", "multiclass"), True),
    "rmse": (False, ("regression",), False),
    "mae": (False, ("regression",), False),
    "r2": (True, ("regression",), False),
    "rmsle": (False, ("regression",), False),
}


class TaskError(ForgeError):
    """The task definition is inconsistent."""


@dataclass(frozen=True)
class TaskSpec:
    problem_type: str
    target: str
    metric: str
    train_path: str
    test_path: str | None = None
    id_column: str | None = None
    group_column: str | None = None
    positive_label: str | None = None
    time_column: str | None = None

    def __post_init__(self) -> None:
        if self.problem_type not in PROBLEM_TYPES:
            raise TaskError(f"Type de problème inconnu : {self.problem_type} (attendu : {', '.join(PROBLEM_TYPES)})")
        if self.metric not in METRICS:
            raise TaskError(f"Métrique non implémentée : {self.metric} (disponibles : {', '.join(METRICS)})")
        if self.problem_type not in METRICS[self.metric][1]:
            raise TaskError(f"La métrique {self.metric} ne s'applique pas à un problème {self.problem_type}.")
        if not self.target.strip():
            raise TaskError("La colonne cible est vide.")

    @property
    def is_classification(self) -> bool:
        return self.problem_type != "regression"

    @property
    def higher_is_better(self) -> bool:
        return METRICS[self.metric][0]

    def data_paths(self) -> list[Path]:
        return [Path(path) for path in (self.train_path, self.test_path) if path]

    def excluded_columns(self) -> set[str]:
        return {column for column in (self.target, self.id_column, self.group_column) if column}

    def to_dict(self) -> dict:
        return asdict(self)
