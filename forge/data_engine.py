"""Data inspection: profile, contradictory twins and error floor, metric sensitivity, leak hints."""

from __future__ import annotations

import pandas as pd

from forge.tabular import column_kind
from forge.task import TaskSpec

HIGH_CARDINALITY_RATIO = 0.5
LEAK_PURITY = 0.999


def read_table(path: str, max_rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def feature_columns(task: TaskSpec, frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in task.excluded_columns()]


def profile(task: TaskSpec, train: pd.DataFrame, test: pd.DataFrame | None, truncated: bool) -> dict:
    """Facts about the data, all measured; nothing here is a hypothesis."""
    features = feature_columns(task, train)
    report = {
        "rows": len(train),
        "truncated": truncated,
        "test_rows": None if test is None else len(test),
        "features": len(features),
        "columns": [_column_profile(train[column]) for column in features],
        "target": _target_profile(task, train[task.target]),
        "duplicate_rows": int(train.duplicated(subset=features).sum()) if features else 0,
        "twins": contradictory_twins(task, train),
        "leak_hints": leak_hints(task, train),
    }
    if test is not None:
        report["missing_in_test"] = sorted(set(features) - set(test.columns))
    if task.is_classification:
        report["metric_sensitivity"] = metric_sensitivity(train[task.target], test_rows=report["test_rows"])
    return report


def _column_profile(series: pd.Series) -> dict:
    return {
        "name": series.name,
        "kind": column_kind(series),
        "missing_rate": round(float(series.isna().mean()), 4),
        "unique": int(series.nunique(dropna=True)),
    }


def _target_profile(task: TaskSpec, target: pd.Series) -> dict:
    if not task.is_classification:
        return {"missing": int(target.isna().sum()), "mean": float(target.mean()), "std": float(target.std())}
    counts = target.astype(str).value_counts()
    return {
        "missing": int(target.isna().sum()),
        "classes": {label: int(count) for label, count in counts.items()},
        "imbalance_ratio": round(float(counts.max() / counts.min()), 2),
    }


def contradictory_twins(task: TaskSpec, train: pd.DataFrame) -> dict:
    """Rows with identical features but different targets bound the attainable score."""
    features = feature_columns(task, train)
    if not features:
        return {"groups": 0, "rows": 0, "error_floor": None}
    keyed = train[features].astype(str).agg("\x1f".join, axis=1)
    grouped = train.assign(_key=keyed).groupby("_key")[task.target]
    if task.is_classification:
        distinct = grouped.nunique()
        conflicting = distinct[distinct > 1].index
        sizes = grouped.size()[conflicting]
        majority = grouped.agg(lambda s: s.astype(str).value_counts().iloc[0])[conflicting]
        unavoidable = int((sizes - majority).sum())
        return {
            "groups": len(conflicting),
            "rows": int(sizes.sum()),
            "error_floor": round(unavoidable / len(train), 6),
            "note": "Taux d'erreur minimal pour tout modèle n'utilisant que ces colonnes.",
        }
    variance = grouped.var(ddof=0).fillna(0)
    sizes = grouped.size()
    conflicting = variance[variance > 0].index
    floor = float((variance * sizes).sum() / len(train))
    return {
        "groups": len(conflicting),
        "rows": int(sizes[conflicting].sum()),
        "error_floor": round(floor, 6),
        "note": "MSE minimale due aux lignes identiques de cibles différentes.",
    }


def metric_sensitivity(target: pd.Series, test_rows: int | None) -> dict:
    """Approximate macro-metric change caused by one error on one example of each class."""
    counts = target.astype(str).value_counts()
    scale = (test_rows or len(target)) / len(target)
    n_classes = len(counts)
    return {
        label: round(1 / (n_classes * max(count * scale, 1)), 4)
        for label, count in counts.sort_values().items()
    }


def leak_hints(task: TaskSpec, train: pd.DataFrame) -> list[str]:
    """Columns that look like identifiers or that determine the target almost perfectly."""
    hints = []
    target = train[task.target]
    for column in feature_columns(task, train):
        series = train[column]
        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)
        if unique_ratio > 0.99 and ("id" in column.lower() or pd.api.types.is_integer_dtype(series)):
            hints.append(f"{column} : valeurs quasi uniques, ressemble à un identifiant.")
        elif HIGH_CARDINALITY_RATIO > unique_ratio and _determines(series, target, task):
            hints.append(f"{column} : détermine presque parfaitement la cible ; fuite possible à vérifier.")
    return hints


def _determines(series: pd.Series, target: pd.Series, task: TaskSpec) -> bool:
    if series.nunique(dropna=True) < 2:
        return False
    if task.is_classification:
        purity = target.astype(str).groupby(series.astype(str)).agg(lambda s: s.value_counts().iloc[0]).sum()
        baseline = target.astype(str).value_counts().iloc[0]
        return purity / len(target) >= LEAK_PURITY and purity > baseline
    if pd.api.types.is_numeric_dtype(series):
        return abs(series.corr(target)) >= LEAK_PURITY
    return False
