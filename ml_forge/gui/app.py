"""Application de bureau principal ML Forge.

Interface graphique de gestion des projets et expériences ML.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTabWidget,
)
from PySide6.QtCore import Qt

from ml_forge.core.config import config
from ml_forge.core.logging import logger
from ml_forge.core.project_manager import ProjectManager
from ml_forge.gui.widgets.dashboard import DashboardWidget
from ml_forge.gui.widgets.project_manager import ProjectManagerWidget


class MLForgeApp(QMainWindow):
    """Fenêtre principale de ML Forge."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Forge - Machine Learning Laboratory")
        self.setGeometry(100, 100, 1200, 800)

        self.project_manager = ProjectManager()
        self.current_project = None

        # Widget central avec onglets
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Barre d'informations système
        info_layout = QHBoxLayout()
        sys_info = config.get_system_info()
        info_text = (
            f"CPU: {sys_info['cpu_cores']} cores | "
            f"Memory: {sys_info['memory_gb']}GB | "
            f"GPU: {'Yes' if sys_info['has_gpu'] else 'No'}"
        )
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        info_layout.addStretch()
        info_layout.addWidget(info_label)
        layout.addLayout(info_layout)

        # Onglets
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Ajouter les onglets
        self.dashboard = DashboardWidget(self.project_manager)
        self.project_manager_widget = ProjectManagerWidget(
            self.project_manager, self.on_project_selected
        )

        self.tabs.addTab(self.dashboard, "Dashboard")
        self.tabs.addTab(self.project_manager_widget, "Projects")

        logger.info("ML Forge démarré")

    def on_project_selected(self, project_id: str):
        """Appelé quand un projet est sélectionné."""
        self.current_project = self.project_manager.open_project(project_id)
        self.dashboard.refresh(self.current_project)
        self.tabs.setCurrentIndex(0)


def main():
    """Point d'entrée de l'application."""
    app = QApplication(sys.argv)
    window = MLForgeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
