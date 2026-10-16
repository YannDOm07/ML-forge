"""Widgets GUI pour le tableau de bord."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
)
from PySide6.QtCore import Qt

from ml_forge.core.logging import logger


class DashboardWidget(QWidget):
    """Tableau de bord principal."""

    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.current_project = None
        self.init_ui()

    def init_ui(self):
        """Initialise l'interface."""
        layout = QVBoxLayout()

        title = QLabel("ML Forge Dashboard")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.content_label = QLabel("No project selected. Create or open a project to begin.")
        layout.addWidget(self.content_label)

        layout.addStretch()
        self.setLayout(layout)

    def refresh(self, project):
        """Rafraîchit le tableau de bord pour un projet."""
        self.current_project = project
        if project:
            self.content_label.setText(f"Project: {project.get_name()}\nObjective: {project.get_objective()}")
        else:
            self.content_label.setText("No project selected.")
