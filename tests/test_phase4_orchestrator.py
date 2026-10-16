"""Tests pour la Phase 4 : orchestration locale et mémoire de projet."""

import json

from ml_forge.agents import LocalExperimentOrchestrator
from ml_forge.core.config import config
from ml_forge.core.project_manager import ProjectManager


def test_orchestrator_builds_safe_fallback_plan():
    """Vérifie que le plan d'expérience est construit sans LLM."""
    orchestrator = LocalExperimentOrchestrator()
    summary = {
        "task_type": "classification",
        "missing_values": {"age": 2},
        "duplicate_rows": 1,
        "class_distribution": {"0": 0.6, "1": 0.4},
        "suggested_hypotheses": ["Tester les variables numériques avec une baseline simple."],
    }

    plan = orchestrator.build_plan(summary, project_name="Churn")
    assert plan["task_type"] == "classification"
    assert plan["objective"]
    assert len(plan["steps"]) >= 4
    assert "accuracy" in plan["recommended_metrics"]
    assert plan["llm_generated"] is False


def test_project_memory_persists_knowledge(tmp_path):
    """Vérifie la persistance des connaissances dans la base du projet."""
    original = config.projects_dir
    config.projects_dir = tmp_path / "projects"
    try:
        pm = ProjectManager()
        project_id = pm.create_project(name="Memory Test")
        project = pm.open_project(project_id)

        inserted_id = project.add_knowledge(
            category="dataset_insight",
            content="Les valeurs manquantes sont faibles et localisées.",
            confidence_score=0.9,
            ref_sources="dataset_summary",
        )

        assert inserted_id is not None
        items = project.list_knowledge(category="dataset_insight")
        assert len(items) == 1
        assert "faibles" in items[0]["content"]
    finally:
        config.projects_dir = original


def test_project_build_experiment_plan_persists_it(tmp_path):
    """Vérifie qu'un plan d'expérience peut être construit puis mémorisé."""
    original = config.projects_dir
    config.projects_dir = tmp_path / "projects"
    try:
        pm = ProjectManager()
        project_id = pm.create_project(name="Planning Test")
        project = pm.open_project(project_id)

        dataset_path = tmp_path / "demo.csv"
        dataset_path.write_text(
            "age,income,target\n"
            "22,1500,0\n"
            "25,2000,0\n"
            "31,2600,1\n"
            "40,3200,1\n",
            encoding="utf-8",
        )

        plan = project.build_experiment_plan(str(dataset_path), target_column="target")
        assert plan["task_type"] == "classification"
        assert plan["steps"]
        knowledge = project.list_knowledge(category="experiment_plan")
        assert len(knowledge) == 1
        payload = json.loads(knowledge[0]["content"])
        assert payload["task_type"] == "classification"
    finally:
        config.projects_dir = original
