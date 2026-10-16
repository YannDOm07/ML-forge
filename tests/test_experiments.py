"""Tests 4 (record and retrieve an experiment) and 5 (recovery after a crash)."""

import sqlite3
import subprocess
import sys

import pytest

from conftest import make_card
from forge.db import session
from forge.errors import ExperimentError
from forge.experiments import (
    finish_experiment,
    get_experiment,
    get_metrics,
    record_metrics,
    register_experiment,
    start_experiment,
)
from forge.projects import create_project, open_project


@pytest.fixture
def project(workspace):
    return create_project(workspace, "Expériences")


def test_experiment_is_recorded_and_retrieved(project):
    card = make_card(code_hash="abc123", data_hash="def456")
    experiment = register_experiment(project.db_path, card)
    start_experiment(project.db_path, experiment.id)
    record_metrics(project.db_path, experiment.id, "macro_f1", "fold", [0.71, 0.74, 0.69])
    record_metrics(project.db_path, experiment.id, "macro_f1", "seed", [0.72, 0.73])
    finish_experiment(project.db_path, experiment.id, "completed", "5 plis terminés")

    stored = get_experiment(project.db_path, experiment.id)
    assert stored.status == "completed"
    assert stored.card == card
    assert stored.started_at and stored.finished_at
    assert get_metrics(project.db_path, experiment.id) == {
        "macro_f1": {"fold": [0.71, 0.74, 0.69], "seed": [0.72, 0.73]}
    }


def test_experiment_ids_are_sequential(project):
    ids = [register_experiment(project.db_path, make_card()).id for _ in range(3)]
    assert ids == ["E0001", "E0002", "E0003"]


def test_correction_creates_a_child_experiment(project):
    parent = register_experiment(project.db_path, make_card())
    child = register_experiment(project.db_path, make_card(parent_experiment_id=parent.id, method="RF 500"))
    assert child.parent_id == parent.id
    assert get_experiment(project.db_path, parent.id).method == "RandomForest 300 arbres"


def test_unknown_parent_is_refused(project):
    with pytest.raises(ExperimentError, match="parente inconnue"):
        register_experiment(project.db_path, make_card(parent_experiment_id="E9999"))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"hypothesis": " "}, "hypothesis"),
        ({"justification": []}, "justification"),
        ({"justification": [{"kind": "intuition", "text": "x"}]}, "kind"),
        ({"estimate": None}, "estimation préalable"),
        ({"estimate": {"expected_gain": 0.1, "success_probability": 1.5, "cost_minutes": 1, "risk": "low"}},
         "success_probability"),
        ({"estimate": {"expected_gain": True, "success_probability": 0.5, "cost_minutes": 1, "risk": "low"}},
         "expected_gain"),
        ({"estimate": {"expected_gain": 0.1, "success_probability": 0.5, "cost_minutes": -1, "risk": "low"}},
         "cost_minutes"),
    ],
)
def test_invalid_card_is_refused(project, overrides, expected):
    with pytest.raises(ExperimentError, match=expected):
        register_experiment(project.db_path, make_card(**overrides))


def test_card_is_immutable_and_experiments_never_deleted(project):
    experiment = register_experiment(project.db_path, make_card())
    with session(project.db_path) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("UPDATE experiments SET hypothesis = 'réécrite' WHERE id = ?", (experiment.id,))
        with pytest.raises(sqlite3.DatabaseError, match="never deleted"):
            conn.execute("DELETE FROM experiments WHERE id = ?", (experiment.id,))


def test_metrics_are_append_only(project):
    experiment = register_experiment(project.db_path, make_card())
    start_experiment(project.db_path, experiment.id)
    record_metrics(project.db_path, experiment.id, "rmse", "fold", [1.0])
    with pytest.raises(ExperimentError, match="déjà enregistrée"):
        record_metrics(project.db_path, experiment.id, "rmse", "fold", [0.5])
    with session(project.db_path) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE experiment_metrics SET value = 0.1")


def test_invalid_transitions_are_refused(project):
    experiment = register_experiment(project.db_path, make_card())
    with pytest.raises(ExperimentError, match="interdit"):
        finish_experiment(project.db_path, experiment.id, "completed", "trop tôt")
    with pytest.raises(ExperimentError, match="pas en cours"):
        record_metrics(project.db_path, experiment.id, "rmse", "fold", [1.0])


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_crash_is_recovered_on_reopen(workspace, project):
    crashed = register_experiment(project.db_path, make_card())
    alive = register_experiment(project.db_path, make_card())
    start_experiment(project.db_path, crashed.id, pid=_dead_pid())
    start_experiment(project.db_path, alive.id)  # this test process is alive

    open_project(workspace, project.id)

    assert get_experiment(project.db_path, crashed.id).status == "interrupted"
    assert get_experiment(project.db_path, alive.id).status == "running"
    start_experiment(project.db_path, crashed.id)  # an interrupted experiment can be resumed
    assert get_experiment(project.db_path, crashed.id).status == "running"
    with session(project.db_path) as conn:
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM execution_events WHERE experiment_id = ? ORDER BY id", (crashed.id,)
        )]
    assert kinds == ["registered", "running", "interrupted", "running"]
