"""Tests 15 (deterministic acceptance) and 18 (unreliable validation detected), plus metrics and folds."""

import numpy as np
import pandas as pd
import pytest

from forge.evaluation import assess_reliability, decide, design_folds, fold_scores, rank_agreement, score
from forge.task import TaskError, TaskSpec

BINARY = TaskSpec("binary", "y", "roc_auc", "train.csv")
MULTI = TaskSpec("multiclass", "y", "f1_macro", "train.csv")
REGRESSION = TaskSpec("regression", "y", "rmse", "train.csv")


def test_task_spec_rejects_inconsistent_definitions():
    with pytest.raises(TaskError, match="ne s'applique pas"):
        TaskSpec("regression", "y", "roc_auc", "train.csv")
    with pytest.raises(TaskError, match="non implémentée"):
        TaskSpec("binary", "y", "map@5", "train.csv")


def test_scores_use_ml_forge_prediction_format():
    y = pd.Series([0, 1, 1, 0])
    predictions = pd.DataFrame({"p:0": [0.9, 0.2, 0.3, 0.6], "p:1": [0.1, 0.8, 0.7, 0.4]})
    assert score(BINARY, y, predictions) == 1.0
    assert score(MULTI, y, predictions) == 1.0
    regression = pd.DataFrame({"prediction": [1.0, 2.0, 3.0]})
    assert score(REGRESSION, pd.Series([1.0, 2.0, 5.0]), regression) == pytest.approx((4 / 3) ** 0.5)


def test_stratified_folds_keep_every_class_in_every_fold():
    y = pd.Series([0] * 40 + [1] * 10)
    folds, notes = design_folds(BINARY, y, None, n_splits=5, seed=1)
    table = pd.crosstab(np.array(folds), y)
    assert (table.to_numpy() == [[8, 2]] * 5).all()
    assert "stratifiée" in notes[0]


def test_folds_shrink_when_a_class_is_tiny():
    y = pd.Series([0] * 30 + [1] * 3)
    folds, notes = design_folds(BINARY, y, None, n_splits=5, seed=1)
    assert len(set(folds)) == 3
    assert "réduits" in notes[0]


def test_group_folds_never_split_a_group():
    task = TaskSpec("binary", "y", "roc_auc", "train.csv", group_column="g")
    y = pd.Series([0, 1] * 30)
    groups = pd.Series(np.repeat(np.arange(12), 5))
    folds, _ = design_folds(task, y, groups, n_splits=4, seed=1)
    assert pd.DataFrame({"g": groups, "f": folds}).groupby("g")["f"].nunique().max() == 1


def test_fold_scores_are_computed_per_fold():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    oof = pd.DataFrame({"prediction": [1.0, 2.0, 3.0, 6.0]})
    assert fold_scores(REGRESSION, y, oof, [0, 0, 1, 1]) == [0.0, pytest.approx(2 ** 0.5)]


def test_acceptance_is_deterministic_and_oriented():
    reference = [0.70, 0.72, 0.71, 0.69, 0.70]
    better = [0.75, 0.76, 0.74, 0.75, 0.76]
    first = decide(better, reference, True, 1.0, True)
    assert first == decide(better, reference, True, 1.0, True)
    assert first.status == "accepted"
    assert decide(reference, better, True, 1.0, True).status == "rejected"
    # Lower is better: smaller errors are an improvement.
    assert decide([0.30, 0.31, 0.29], [0.40, 0.41, 0.42], False, 1.0, True).status == "accepted"


def test_gain_below_noise_is_not_significant():
    reference = [0.70, 0.72, 0.71, 0.69, 0.70]
    noisy = [0.74, 0.69, 0.75, 0.67, 0.72]
    assert decide(noisy, reference, True, 1.0, True).status == "not_significant"


def test_unreliable_validation_requires_the_second_regime():
    reference, better = [0.70, 0.71, 0.70], [0.80, 0.81, 0.80]
    assert decide(better, reference, True, 1.0, reliable=False).status == "not_significant"
    assert decide(better, reference, True, 1.0, reliable=False, second_regime_delta=-0.01).status == "rejected"
    assert decide(better, reference, True, 1.0, reliable=False, second_regime_delta=0.02).status == "accepted"


def test_inverted_ranking_marks_validation_unreliable():
    """Built like the HeriTech case: local validation ranks configurations against the test split."""
    pairs = [(0.80, 0.84), (0.82, 0.82), (0.84, 0.80), (0.86, 0.78)]
    assert rank_agreement(pairs) == pytest.approx(-1.0)
    y = pd.Series([0, 1] * 50)
    folds, _ = design_folds(BINARY, y, None, 5, 1)
    reliability = assess_reliability(BINARY, y, folds, min_class_per_fold=5, paired_scores=pairs)
    assert not reliability.reliable
    assert "Spearman" in reliability.reasons[0]


def test_thin_rare_class_marks_validation_unreliable():
    y = pd.Series([0] * 95 + [1] * 5)
    folds, _ = design_folds(BINARY, y, None, 5, 1)
    reliability = assess_reliability(BINARY, y, folds, min_class_per_fold=5)
    assert not reliability.reliable
    assert reliability.smallest_class_per_fold == 1


def test_consistent_validation_is_reliable():
    y = pd.Series([0, 1] * 100)
    folds, _ = design_folds(BINARY, y, None, 5, 1)
    pairs = [(0.70, 0.71), (0.75, 0.74), (0.80, 0.81)]
    assert assess_reliability(BINARY, y, folds, 5, pairs).reliable
