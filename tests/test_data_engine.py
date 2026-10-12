"""Test 17: contradictory twins and error floor; profile, sensitivity and leak hints."""

import pandas as pd
import pytest

from forge.data_engine import contradictory_twins, leak_hints, metric_sensitivity, profile
from forge.task import TaskSpec

CLASSIFICATION = TaskSpec("multiclass", "app", "f1_macro", "train.csv", id_column="flow_id")
REGRESSION = TaskSpec("regression", "y", "rmse", "train.csv")


def test_twins_give_the_classification_error_floor():
    # Same packet counters, different applications (as in the ITU packet-flow case).
    train = pd.DataFrame({
        "flow_id": range(8),
        "bytes": [794, 794, 794, 794, 2762, 2762, 100, 200],
        "pkts": [3, 3, 3, 3, 2, 2, 1, 1],
        "app": ["X", "X", "Mozilla", "YouTube", "YouTube", "Google", "X", "Mozilla"],
    })
    twins = contradictory_twins(CLASSIFICATION, train)
    assert twins["groups"] == 2
    assert twins["rows"] == 6
    # Group 1: 4 rows, majority 2 → 2 unavoidable errors; group 2: 2 rows → 1 error.
    assert twins["error_floor"] == pytest.approx(3 / 8)


def test_twins_give_the_regression_error_floor():
    train = pd.DataFrame({"x": [1, 1, 2, 3], "y": [0.0, 2.0, 5.0, 7.0]})
    twins = contradictory_twins(REGRESSION, train)
    assert twins["groups"] == 1
    assert twins["error_floor"] == pytest.approx(2 * 1.0 / 4)


def test_no_twins_means_no_floor():
    train = pd.DataFrame({"flow_id": [1, 2], "bytes": [1, 2], "app": ["X", "Y"]})
    assert contradictory_twins(CLASSIFICATION, train)["error_floor"] == 0.0


def test_metric_sensitivity_is_largest_for_the_rarest_class():
    target = pd.Series(["Google"] * 33 + ["Facebook"] * 2 + ["Instagram"] * 4)
    sensitivity = metric_sensitivity(target, test_rows=39)
    assert list(sensitivity)[0] == "Facebook"
    assert sensitivity["Facebook"] == pytest.approx(1 / (3 * 2), abs=1e-4)


def test_leak_hints_flag_identifiers_and_target_proxies():
    train = pd.DataFrame({
        "flow_id": range(200),
        "row_number": range(200),
        "label_copy": ["a", "b"] * 100,
        "noise": [1, 1, 2, 2] * 50,
        "app": ["A", "B"] * 100,
    })
    hints = " ".join(leak_hints(CLASSIFICATION, train))
    assert "row_number" in hints and "identifiant" in hints
    assert "label_copy" in hints and "fuite" in hints
    assert "noise" not in hints


def test_profile_reports_measured_facts():
    train = pd.DataFrame({
        "flow_id": range(6),
        "size": [1.0, None, 3.0, 4.0, 5.0, 6.0],
        "proto": ["tcp", "udp", "tcp", "udp", "tcp", "udp"],
        "when": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"],
        "app": ["X", "X", "X", "Y", "Y", "Z"],
    })
    report = profile(CLASSIFICATION, train, train.drop(columns=["app", "size"]), truncated=False)
    kinds = {column["name"]: column["kind"] for column in report["columns"]}
    assert kinds == {"size": "numeric", "proto": "categorical", "when": "datetime"}
    assert report["target"]["classes"] == {"X": 3, "Y": 2, "Z": 1}
    assert report["target"]["imbalance_ratio"] == 3.0
    assert report["missing_in_test"] == ["size"]
    assert next(c for c in report["columns"] if c["name"] == "size")["missing_rate"] == pytest.approx(1 / 6, abs=1e-4)
