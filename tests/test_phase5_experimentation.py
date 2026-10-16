"""Tests de validation pour la Phase 5 : expérimentation avancée."""

import pandas as pd

from ml_forge.core.config import config
from ml_forge.core.project_manager import ProjectManager
from ml_forge.engines import FeatureEngineer


def test_feature_engineering_and_global_memory(tmp_path):
    """Vérifie la génération de features et la mémoire inter-projets."""
    original_projects_dir = config.projects_dir
    config.projects_dir = tmp_path / "projects"

    try:
        pm = ProjectManager()
        project_id = pm.create_project(name="Phase 5 Project")
        project = pm.open_project(project_id)

        dataset = pd.DataFrame(
            {
                "age": [20, 22, 25, 30, 35, 40, 45, 50],
                "income": [1500, 1800, 1900, 2000, 2500, 2600, 3000, 3300],
                "target": [0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        dataset_path = tmp_path / "phase5_dataset.csv"
        dataset.to_csv(dataset_path, index=False)

        engineered = FeatureEngineer().generate_features(dataset.drop(columns=["target"]))
        assert "age_squared" in engineered.columns
        assert "income_per_age" in engineered.columns

        result = project.run_advanced_experiment(
            str(dataset_path),
            target_column="target",
            feature_strategy="ratio",
            optimize_hyperparameters=False,
            hypothesis="Les ratios de revenus amplifient le signal de décision.",
            method="LogisticRegression avec features élaborées",
        )

        assert result["task_type"] == "classification"
        assert result["metrics"]["accuracy"] >= 0.0
        assert result["experiment_id"] is not None

        project.add_knowledge(
            category="feature_insight",
            content="Les features de ratio de revenu sont utiles pour la cible binaire.",
            confidence_score=0.85,
            ref_sources="phase5_dataset",
        )

        relevant = pm.search_global_knowledge(task_type="classification", limit=5)
        assert len(relevant) >= 1

        comparison = project.compare_experiments(
            result["experiment_id"],
            result["experiment_id"],
        )
        assert "metric_name" in comparison
    finally:
        config.projects_dir = original_projects_dir
