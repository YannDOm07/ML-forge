"""Tests de validation pour la Phase 3 : moteur ML de base."""

import pandas as pd

from ml_forge.core.config import config
from ml_forge.core.project_manager import ProjectManager
from ml_forge.engines import DataEngine, ExperimentEngine


def test_data_engine_inspection_classification(tmp_path):
    """Vérifie que le DataEngine détecte correctement une classification."""
    dataset = pd.DataFrame(
        {
            "age": [18, 22, 25, 29, 31, 35, 40, 44, 47, 52],
            "income": [1500, 2000, 1800, 2100, 2600, 3000, 2800, 3300, 3100, 3500],
            "segment": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    dataset_path = tmp_path / "phase3_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    summary = DataEngine().inspect_dataset(dataset, target_column="target")

    assert summary["task_type"] == "classification"
    assert summary["rows"] == 10
    assert summary["class_distribution"]
    assert len(summary["suggested_hypotheses"]) > 0


def test_experiment_engine_runs_baseline_on_classification(tmp_path):
    """Vérifie qu'une baseline est entraînée et retourne des métriques."""
    dataset = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "feature_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )

    result = ExperimentEngine().run_experiment(
        dataset[["feature_a", "feature_b"]],
        dataset["target"],
        task_type="classification",
    )

    assert result["task_type"] == "classification"
    assert result["metrics"]["accuracy"] >= 0.0
    assert result["metrics"]["f1"] >= 0.0
    assert result["n_samples"] == 10


def test_project_run_baseline_experiment_persists_results(tmp_path):
    """Vérifie qu'un projet peut exécuter une expérience ML et la persister."""
    original_projects_dir = config.projects_dir
    config.projects_dir = tmp_path / "projects"

    try:
        pm = ProjectManager()
        project_id = pm.create_project(name="Phase 3 Project")
        project = pm.open_project(project_id)

        dataset = pd.DataFrame(
            {
                "feature_x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature_y": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            }
        )
        dataset_path = tmp_path / "project_dataset.csv"
        dataset.to_csv(dataset_path, index=False)

        result = project.run_baseline_experiment(
            str(dataset_path),
            target_column="target",
            hypothesis="Le modèle de référence doit séparer clairement les classes.",
            method="Baseline logistic regression",
        )

        assert result["experiment_id"] is not None
        assert result["metrics"]["accuracy"] >= 0.0

        experiments = project.get_experiments()
        assert len(experiments) == 1
        assert experiments[0]["status"] == "completed"
    finally:
        config.projects_dir = original_projects_dir
