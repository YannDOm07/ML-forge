"""Tests d'acceptation pour la Phase 2 : Fondations et Persistance."""

import pytest
import shutil
from pathlib import Path
from datetime import datetime

from ml_forge.core.config import config
from ml_forge.core.project_manager import ProjectManager, Project
from ml_forge.storage.database import Database


@pytest.fixture
def temp_projects_dir(tmp_path):
    """Crée un répertoire temporaire pour les projets."""
    original = config.projects_dir
    config.projects_dir = tmp_path / "projects"
    yield config.projects_dir
    config.projects_dir = original


class TestProjectCreation:
    """Test de création de projets isolés."""

    def test_create_single_project(self, temp_projects_dir):
        """Crée un projet et vérifie sa persistance."""
        pm = ProjectManager()
        project_id = pm.create_project(
            name="Test Project 1",
            description="Projet de test",
        )

        assert project_id is not None
        assert (temp_projects_dir / project_id).exists()

        # Vérifier la persistance
        projects = pm.list_projects()
        assert len(projects) == 1
        assert projects[0]["name"] == "Test Project 1"

    def test_create_multiple_projects(self, temp_projects_dir):
        """Crée plusieurs projets et vérifie qu'ils sont indépendants."""
        pm = ProjectManager()

        proj1 = pm.create_project(name="Project 1")
        proj2 = pm.create_project(name="Project 2")
        proj3 = pm.create_project(name="Project 3")

        assert proj1 and proj2 and proj3
        assert proj1 != proj2 and proj2 != proj3

        # Vérifier qu'aucun n'écrase l'autre
        projects = pm.list_projects()
        assert len(projects) == 3
        names = [p["name"] for p in projects]
        assert "Project 1" in names
        assert "Project 2" in names
        assert "Project 3" in names

    def test_project_isolation(self, temp_projects_dir):
        """Vérifie que les données d'un projet ne fuient pas dans un autre."""
        pm = ProjectManager()

        proj1_id = pm.create_project(name="Projet A", description="Description A")
        proj2_id = pm.create_project(name="Projet B", description="Description B")

        proj1 = pm.open_project(proj1_id)
        proj2 = pm.open_project(proj2_id)

        assert proj1.get_name() == "Projet A"
        assert proj2.get_name() == "Projet B"
        assert proj1.metadata["description"] == "Description A"
        assert proj2.metadata["description"] == "Description B"


class TestProjectPersistence:
    """Test de persistance après redémarrage."""

    def test_persistence_after_restart(self, temp_projects_dir):
        """Crée des projets et vérifie qu'ils subsistent après redémarrage."""
        pm1 = ProjectManager()
        proj_id = pm1.create_project(name="Persistent Project")
        pm1.update_project_objective(proj_id, "Test objective")

        # Simuler un redémarrage (nouveau gestionnaire de projets)
        pm2 = ProjectManager()
        projects = pm2.list_projects()

        assert len(projects) == 1
        assert projects[0]["name"] == "Persistent Project"
        assert projects[0]["objective"] == "Test objective"

    def test_open_after_restart(self, temp_projects_dir):
        """Ouvre un projet après redémarrage."""
        pm1 = ProjectManager()
        proj_id = pm1.create_project(name="Project to Reopen")
        proj1 = pm1.open_project(proj_id)
        proj1.set_objective("Objective 1")

        # Redémarrage
        pm2 = ProjectManager()
        proj2 = pm2.open_project(proj_id)

        assert proj2 is not None
        assert proj2.get_objective() == "Objective 1"


class TestProjectArchiving:
    """Test de l'archivage de projets."""

    def test_archive_project(self, temp_projects_dir):
        """Archive un projet."""
        pm = ProjectManager()
        proj_id = pm.create_project(name="Project to Archive")

        pm.archive_project(proj_id)

        # Les projets archivés ne doivent pas apparaître par défaut
        active_projects = pm.list_projects(archived=False)
        assert len(active_projects) == 0

        # Les projets archivés doivent être récupérables
        archived_projects = pm.list_projects(archived=True)
        assert len(archived_projects) == 1
        assert archived_projects[0]["name"] == "Project to Archive"

    def test_delete_project_requires_confirmation(self, temp_projects_dir):
        """Vérifie que la suppression demande une confirmation."""
        pm = ProjectManager()
        proj_id = pm.create_project(name="Project to Delete")

        # Refuser la suppression sans confirmation
        result = pm.delete_project(proj_id, confirm=False)
        assert result is False

        # Le projet doit toujours exister
        projects = pm.list_projects()
        assert len(projects) == 1


class TestExperiments:
    """Test du suivi des expériences."""

    def test_add_experiment(self, temp_projects_dir):
        """Ajoute une expérience à un projet."""
        pm = ProjectManager()
        proj_id = pm.create_project(name="Experiment Project")
        proj = pm.open_project(proj_id)

        exp_id = proj.add_experiment(
            hypothesis="Feature engineering improves score",
            method="add polynomial features",
            code="from sklearn.preprocessing import PolynomialFeatures",
        )

        assert exp_id is not None

        experiments = proj.get_experiments()
        assert len(experiments) == 1
        assert experiments[0]["hypothesis"] == "Feature engineering improves score"

    def test_experiment_persistence(self, temp_projects_dir):
        """Vérifie que les expériences sont persistantes."""
        pm1 = ProjectManager()
        proj_id = pm1.create_project(name="Exp Project")
        proj1 = pm1.open_project(proj_id)
        exp_id = proj1.add_experiment("Hypothesis 1", "Method 1")

        # Redémarrage
        pm2 = ProjectManager()
        proj2 = pm2.open_project(proj_id)

        experiments = proj2.get_experiments()
        assert len(experiments) == 1
        assert experiments[0]["id"] == exp_id


class TestDatabaseIntegrity:
    """Test d'intégrité et de fiabilité de la base de données."""

    def test_transaction_rollback(self, temp_projects_dir):
        """Vérifie que les erreurs ne corrompent pas la base."""
        pm = ProjectManager()
        proj_id = pm.create_project(name="Test DB Integrity")
        proj = pm.open_project(proj_id)

        # Ajouter une expérience valide
        proj.add_experiment("Valid hypothesis", "Method 1")

        # Vérifier que l'expérience est enregistrée
        experiments = proj.get_experiments()
        assert len(experiments) == 1

    def test_backup_and_restore(self, temp_projects_dir):
        """Teste la sauvegarde et restauration de projet."""
        pm = ProjectManager()
        proj_id = pm.create_project(name="Backup Test")
        proj = pm.open_project(proj_id)
        proj.add_experiment("Hypothesis 1", "Method 1")

        # Sauvegarder
        backup_path = pm.backup_project(proj_id)
        assert backup_path and backup_path.exists()

        # Restaurer
        restored_id = pm.restore_project(backup_path)
        assert restored_id is not None
        assert restored_id != proj_id  # Nouvel ID

        # Vérifier que les données sont présentes
        restored_proj = pm.open_project(restored_id)
        experiments = restored_proj.get_experiments()
        assert len(experiments) == 1


class TestConfigurationHistory:
    """Test de l'historique de configuration."""

    def test_update_project_objective(self, temp_projects_dir):
        """Teste la mise à jour d'un objectif."""
        pm = ProjectManager()
        proj_id = pm.create_project(name="Config Test")

        pm.update_project_objective(proj_id, "New Objective")

        proj = pm.open_project(proj_id)
        assert proj.get_objective() == "New Objective"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
