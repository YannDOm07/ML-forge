"""Tests for the ML Forge MCP tools used by OpenCode."""

import pandas as pd
import pytest

from ml_forge.core.config import config
from ml_forge.mcp_server import (
    create_project,
    get_project,
    inspect_dataset,
    list_experiments,
    list_projects,
    run_baseline_experiment,
)


@pytest.fixture
def isolated_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "projects_dir", tmp_path / "projects")


def test_mcp_project_lifecycle(isolated_projects):
    created = create_project("OpenCode test", "Projet via outils MCP")
    project_id = created["project_id"]

    assert get_project(project_id)["name"] == "OpenCode test"
    assert any(project["id"] == project_id for project in list_projects())
    assert list_experiments(project_id) == []


def test_mcp_dataset_tools_inspect_and_run_experiment(isolated_projects, tmp_path):
    project_id = create_project("ML test")["project_id"]
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "feature": list(range(1, 21)),
            "category": ["a", "b"] * 10,
            "target": [0] * 10 + [1] * 10,
        }
    ).to_csv(dataset_path, index=False)

    summary = inspect_dataset(project_id, str(dataset_path), "target")
    assert summary["task_type"] == "classification"

    result = run_baseline_experiment(project_id, str(dataset_path), "target")
    assert result["experiment_id"]
    assert len(list_experiments(project_id)) == 1


def test_mcp_tools_reject_unknown_project():
    with pytest.raises(ValueError, match="introuvable"):
        get_project("missing")
