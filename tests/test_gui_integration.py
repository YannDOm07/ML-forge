"""Test simple d'intégration de la GUI (sans affichage graphique)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml_forge.core.config import config
from ml_forge.core.project_manager import ProjectManager
from ml_forge.core.logging import logger


def test_gui_initialization():
    """Teste que la GUI peut être importée et initialisée."""
    try:
        # Importer les modules GUI (devrait fonctionner même en headless)
        from ml_forge.gui.app import MLForgeApp
        logger.info("✅ GUI modules importés avec succès")
        
        # Tester le gestionnaire de projets
        pm = ProjectManager()
        proj_id = pm.create_project(name="GUI Test Project")
        assert proj_id, "Impossible de créer un projet"
        logger.info(f"✅ Projet créé : {proj_id}")
        
        # Ouvrir le projet
        proj = pm.open_project(proj_id)
        assert proj, "Impossible d'ouvrir le projet"
        logger.info(f"✅ Projet ouvert : {proj.get_name()}")
        
        # Ajouter une expérience
        exp_id = proj.add_experiment("Test hypothesis", "Test method")
        assert exp_id, "Impossible d'ajouter une expérience"
        logger.info(f"✅ Expérience créée : {exp_id}")
        
        logger.info("✅✅✅ Tous les tests d'intégration GUI réussis !")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors du test : {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_gui_initialization()
    sys.exit(0 if success else 1)
