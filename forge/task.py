"""Definition of a supervised task: what to predict, from which files, scored how.

Importable inside the sandbox (experiments use `read_table`): standard library and pandas only.
"""

from __future__ import annotations

import json
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
TABLE_FORMATS = (".csv", ".tsv", ".tab", ".json", ".jsonl", ".ndjson", ".parquet", ".feather")
PRIMARY_SET = "test"  # name of the prediction set read from `test_path` when no set is named


class TaskError(ForgeError):
    """The task definition is inconsistent."""


def read_table(path: str | Path, usecols: list[str] | None = None, nrows: int | None = None):
    """Read a data table whatever its format: CSV/TSV, JSON (records or lines), Parquet, Feather."""
    import pandas as pd

    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".tsv", ".tab"} or suffix not in TABLE_FORMATS:
        separator = "\t" if suffix in {".tsv", ".tab"} else ","
        return pd.read_csv(path, sep=separator, usecols=usecols, nrows=nrows, low_memory=False)
    if suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True, nrows=nrows, dtype=False)
    elif suffix == ".json":
        frame = _read_json_records(Path(path))
    elif suffix == ".parquet":
        frame = pd.read_parquet(path, columns=usecols)
    else:
        frame = pd.read_feather(path, columns=usecols)
    if usecols is not None:
        frame = frame[list(usecols)]
    return frame.head(nrows) if nrows is not None else frame


def _read_json_records(path: Path):
    """A JSON table: a list of records, or a dict of records keyed by id."""
    import pandas as pd

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        lists = [value for value in data.values() if isinstance(value, list)]
        if len(lists) == 1 and all(isinstance(item, dict) for item in lists[0]):
            data = lists[0]  # {"data": [...]} envelope
        elif all(isinstance(value, dict) for value in data.values()):
            data = [{"_key": key, **value} for key, value in data.items()]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise TaskError(f"{path.name} n'est pas une table JSON (liste d'enregistrements attendue).")
    return pd.DataFrame.from_records(data)


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
    # Every set to predict, by name ("valid", "eval"…); None means the single `test_path`.
    prediction_sets: dict[str, str] | None = None
    # Directories of files the tables refer to (audio, images…): read-only for experiments.
    media_dirs: tuple[str, ...] | list[str] = ()
    # Columns present in the training table only (labels, annotations): never used as features.
    train_only_columns: tuple[str, ...] | list[str] = ()

    def __post_init__(self) -> None:
        if self.problem_type not in PROBLEM_TYPES:
            raise TaskError(f"Type de problème inconnu : {self.problem_type} (attendu : {', '.join(PROBLEM_TYPES)})")
        if self.metric not in METRICS:
            raise TaskError(f"Métrique non implémentée : {self.metric} (disponibles : {', '.join(METRICS)})")
        if self.problem_type not in METRICS[self.metric][1]:
            raise TaskError(f"La métrique {self.metric} ne s'applique pas à un problème {self.problem_type}.")
        if not self.target.strip():
            raise TaskError("La colonne cible est vide.")
        if self.prediction_sets is not None and self.test_path not in self.prediction_sets.values():
            raise TaskError("test_path doit être l'un des jeux de prédiction (le jeu principal).")

    @property
    def is_classification(self) -> bool:
        return self.problem_type != "regression"

    @property
    def higher_is_better(self) -> bool:
        return METRICS[self.metric][0]

    def sets(self) -> dict[str, str]:
        """Every set to predict, by name; the primary one (test_path) first."""
        if self.prediction_sets:
            primary = next(name for name, path in self.prediction_sets.items() if path == self.test_path)
            return {primary: self.test_path, **{k: v for k, v in self.prediction_sets.items() if k != primary}}
        return {PRIMARY_SET: self.test_path} if self.test_path else {}

    def prediction_files(self) -> dict[str, str]:
        """File an experiment writes for each set: `test_pred.csv` for the primary one."""
        return {name: "test_pred.csv" if index == 0 else f"pred_{name}.csv"
                for index, name in enumerate(self.sets())}

    def data_paths(self) -> list[Path]:
        paths = [self.train_path, *self.sets().values()]
        return [Path(path) for path in dict.fromkeys(paths) if path]

    def excluded_columns(self) -> set[str]:
        named = {column for column in (self.target, self.id_column, self.group_column) if column}
        return named | set(self.train_only_columns)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["media_dirs"] = list(self.media_dirs)
        data["train_only_columns"] = list(self.train_only_columns)
        return data
