"""Metrics, validation folds, validation reliability and the computed acceptance rule."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from forge.tabular import class_columns, labels_from_probabilities
from forge.task import PROBA_PREFIX, TaskSpec


# --- metrics -----------------------------------------------------------------

def score(task: TaskSpec, y_true: pd.Series, predictions: pd.DataFrame) -> float:
    """Score predictions in the ML Forge format: 'p:<class>' columns, or 'prediction'."""
    if not task.is_classification:
        return _regression_score(task.metric, y_true.to_numpy(dtype=float), predictions["prediction"].to_numpy())
    truth = y_true.astype(str).to_numpy()
    columns = class_columns(predictions)
    classes = [column[len(PROBA_PREFIX):] for column in columns]
    if task.metric == "log_loss":
        return float(skm.log_loss(truth, predictions[columns].to_numpy(), labels=classes))
    if task.metric == "roc_auc":
        return _roc_auc(task, truth, predictions, classes)
    labels = labels_from_probabilities(predictions)
    if task.metric == "accuracy":
        return float(skm.accuracy_score(truth, labels))
    if task.metric == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(truth, labels))
    if task.metric == "f1_macro":
        return float(skm.f1_score(truth, labels, average="macro", labels=classes, zero_division=0))
    return float(skm.f1_score(truth, labels, pos_label=_positive(task, classes), zero_division=0))


def _positive(task: TaskSpec, classes: Sequence[str]) -> str:
    return task.positive_label if task.positive_label is not None else sorted(classes)[-1]


def _roc_auc(task: TaskSpec, truth: np.ndarray, predictions: pd.DataFrame, classes: list[str]) -> float:
    if task.problem_type == "binary":
        positive = _positive(task, classes)
        return float(skm.roc_auc_score(truth == positive, predictions[PROBA_PREFIX + positive]))
    probabilities = predictions[[PROBA_PREFIX + c for c in classes]].to_numpy()
    return float(skm.roc_auc_score(truth, probabilities, multi_class="ovr", labels=classes))


def _regression_score(metric: str, y: np.ndarray, pred: np.ndarray) -> float:
    if metric == "rmse":
        return float(math.sqrt(skm.mean_squared_error(y, pred)))
    if metric == "mae":
        return float(skm.mean_absolute_error(y, pred))
    if metric == "r2":
        return float(skm.r2_score(y, pred))
    return float(math.sqrt(skm.mean_squared_log_error(y, np.clip(pred, 0, None))))


def fold_scores(task: TaskSpec, y_true: pd.Series, oof: pd.DataFrame, folds: Sequence[int]) -> list[float]:
    folds = np.asarray(folds)
    return [
        score(task, y_true[folds == fold].reset_index(drop=True), oof[folds == fold].reset_index(drop=True))
        for fold in sorted(set(folds.tolist()))
        if fold >= 0
    ]


# --- folds -------------------------------------------------------------------

def design_folds(
    task: TaskSpec, y: pd.Series, groups: pd.Series | None, n_splits: int, seed: int,
    times: pd.Series | None = None,
) -> tuple[list[int], list[str]]:
    """Return a fold index per row (-1: never validated) and the notes explaining the choice."""
    if times is not None:
        return _time_folds(times, n_splits)
    notes = []
    if task.is_classification:
        smallest = int(y.astype(str).value_counts().min())
        if smallest < n_splits:
            notes.append(f"Plus petite classe : {smallest} exemple(s) ; plis réduits de {n_splits} à {max(smallest, 2)}.")
            n_splits = max(smallest, 2)
    if groups is not None:
        n_splits = min(n_splits, groups.nunique())
        splitter = (
            StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
            if task.is_classification
            else GroupKFold(n_splits)
        )
        notes.append(f"Plis par groupe ({task.group_column}) : un groupe n'est jamais à la fois en entraînement et en validation.")
        split = splitter.split(np.zeros(len(y)), y.astype(str) if task.is_classification else y, groups)
    elif task.is_classification:
        split = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y.astype(str))
        notes.append(f"Validation croisée stratifiée, {n_splits} plis, seed {seed}.")
    else:
        split = KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)))
        notes.append(f"Validation croisée, {n_splits} plis, seed {seed}.")
    folds = np.full(len(y), -1)
    for fold, (_, valid) in enumerate(split):
        folds[valid] = fold
    return folds.tolist(), notes


def _time_folds(times: pd.Series, n_splits: int) -> tuple[list[int], list[str]]:
    """Forward chaining: rows sorted by time, cut into n_splits + 1 blocks; block k + 1 is
    predicted by a model trained on blocks 0..k only, the first block is never validated."""
    order = np.argsort(pd.to_datetime(times, errors="coerce", format="mixed").to_numpy(), kind="stable")
    folds = np.full(len(times), -1)
    for fold, block in enumerate(np.array_split(order, n_splits + 1)[1:]):
        folds[block] = fold
    note = (f"Validation chronologique : {n_splits} plis ; chaque pli n'est prédit qu'à partir des lignes "
            "antérieures, le premier bloc sert seulement à l'entraînement.")
    return folds.tolist(), [note]


# --- reliability -------------------------------------------------------------

@dataclass(frozen=True)
class Reliability:
    reliable: bool
    reasons: list[str] = field(default_factory=list)
    smallest_class_per_fold: int | None = None
    rank_agreement: float | None = None

    def to_dict(self) -> dict:
        return {
            "reliable": self.reliable,
            "reasons": self.reasons,
            "smallest_class_per_fold": self.smallest_class_per_fold,
            "rank_agreement": self.rank_agreement,
        }


def assess_reliability(
    task: TaskSpec,
    y: pd.Series,
    folds: Sequence[int],
    min_class_per_fold: int,
    paired_scores: Sequence[tuple[float, float]] = (),
) -> Reliability:
    """Flag a validation as unreliable when folds are too thin for rare classes, or when
    it ranks configurations differently from a second regime (official split, simulator)."""
    reasons = []
    smallest = None
    if task.is_classification:
        validated = np.asarray(folds) >= 0
        table = pd.crosstab(np.asarray(folds)[validated], y.astype(str).to_numpy()[validated])
        smallest = int(table.to_numpy().min())
        if smallest < min_class_per_fold:
            reasons.append(
                f"Une classe n'a que {smallest} exemple(s) dans un pli (< {min_class_per_fold}) : "
                "un seul exemple déplace fortement le score."
            )
    agreement = rank_agreement(paired_scores) if len(paired_scores) >= 3 else None
    if agreement is not None and agreement < 0.5:
        reasons.append(
            f"La validation classe les configurations autrement que le second régime (Spearman ρ = {agreement:.2f})."
        )
    return Reliability(not reasons, reasons, smallest, agreement)


def rank_agreement(pairs: Sequence[tuple[float, float]]) -> float:
    """Spearman correlation between two score lists."""
    first = pd.Series([a for a, _ in pairs]).rank()
    second = pd.Series([b for _, b in pairs]).rank()
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def simulated_delta(
    task: TaskSpec, y: pd.Series, candidate: pd.DataFrame, reference: pd.DataFrame, folds: Sequence[int],
    test_size: int, draws: int = 60, seed: int = 0,
) -> float:
    """Second regime: mean paired gain over random draws shaped like the test set (prompt_v2 §11.2)."""
    rows = np.flatnonzero(np.asarray(folds) >= 0)
    size = max(2, min(test_size, len(rows)))
    rng = np.random.default_rng(seed)
    sign = 1.0 if task.higher_is_better else -1.0
    gains = []
    for _ in range(draws):
        sample = rng.choice(rows, size=size, replace=False)
        truth = y.iloc[sample].reset_index(drop=True)
        if task.is_classification and truth.astype(str).nunique() < 2:
            continue
        try:
            gains.append(sign * (score(task, truth, candidate.iloc[sample].reset_index(drop=True))
                                 - score(task, truth, reference.iloc[sample].reset_index(drop=True))))
        except ValueError:
            continue
    return float(np.mean(gains)) if gains else 0.0


# --- acceptance --------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    status: str  # accepted | rejected | not_significant
    delta: float
    noise: float
    rule: str


def decide(
    candidate: Sequence[float],
    reference: Sequence[float],
    higher_is_better: bool,
    std_factor: float,
    reliable: bool,
    second_regime_delta: float | None = None,
) -> Decision:
    """Computed acceptance rule (prompt_v2 §11.3). Same inputs always give the same decision.

    delta = mean of paired fold differences, oriented so that positive means better;
    noise = standard deviation of those differences. A change is accepted only if
    delta > std_factor × noise; if the validation is unreliable, the second regime
    must improve too, and without a second regime the change cannot be accepted."""
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("Les deux expériences doivent avoir les mêmes plis.")
    sign = 1.0 if higher_is_better else -1.0
    differences = sign * (np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float))
    delta = float(differences.mean())
    noise = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    threshold = std_factor * noise
    if delta <= 0:
        return Decision("rejected", delta, noise, f"Δ = {delta:+.5f} ≤ 0 : pas d'amélioration.")
    if delta <= threshold:
        return Decision("not_significant", delta, noise, f"Δ = {delta:+.5f} ≤ {std_factor} × bruit ({noise:.5f}).")
    if not reliable:
        if second_regime_delta is None:
            return Decision(
                "not_significant", delta, noise,
                f"Δ = {delta:+.5f} > seuil, mais validation peu fiable : double régime requis et absent.",
            )
        if second_regime_delta <= 0:
            return Decision(
                "rejected", delta, noise,
                f"Δ = {delta:+.5f} en validation, mais {second_regime_delta:+.5f} dans le second régime.",
            )
    return Decision("accepted", delta, noise, f"Δ = {delta:+.5f} > {std_factor} × bruit ({noise:.5f}).")
