"""Widgets GUI pour la gestion des projets."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QFormLayout,
    QTextEdit,
    QFileDialog,
)
from PySide6.QtCore import Qt

from ml_forge.core.logging import logger


class CreateProjectDialog(QDialog):
    """Dialogue de création de projet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setGeometry(200, 200, 500, 400)

        layout = QFormLayout()

        self.name_input = QLineEdit()
        self.description_input = QTextEdit()
        self.dataset_input = QLineEdit()
        self.target_input = QLineEdit()

        layout.addRow("Project Name:", self.name_input)
        layout.addRow("Description:", self.description_input)
        layout.addRow("Dataset Path:", self.dataset_input)
        layout.addRow("Target Column:", self.target_input)

        button_layout = QHBoxLayout()
        create_btn = QPushButton("Create")
        cancel_btn = QPushButton("Cancel")

        create_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(create_btn)
        button_layout.addWidget(cancel_btn)

        layout.addRow(button_layout)
        self.setLayout(layout)


class ProjectManagerWidget(QWidget):
    """Gestion des projets."""

    def __init__(self, project_manager, on_project_selected):
        super().__init__()
        self.project_manager = project_manager
        self.on_project_selected = on_project_selected
        self.init_ui()

    def init_ui(self):
        """Initialise l'interface."""
        layout = QVBoxLayout()

        title = QLabel("Projects")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Boutons
        button_layout = QHBoxLayout()
        create_btn = QPushButton("Create New Project")
        refresh_btn = QPushButton("Refresh")

        create_btn.clicked.connect(self.create_project)
        refresh_btn.clicked.connect(self.refresh_projects)

        button_layout.addWidget(create_btn)
        button_layout.addWidget(refresh_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Liste des projets
        self.project_list = QListWidget()
        self.project_list.itemClicked.connect(self.on_project_clicked)
        layout.addWidget(self.project_list)

        self.setLayout(layout)
        self.refresh_projects()

    def create_project(self):
        """Crée un nouveau projet."""
        dialog = CreateProjectDialog(self)
        if dialog.exec() == QDialog.Accepted:
            name = dialog.name_input.text()
            description = dialog.description_input.toPlainText()
            dataset_path = dialog.dataset_input.text() or None
            target_column = dialog.target_input.text() or None

            if name.strip():
                project_id = self.project_manager.create_project(
                    name=name,
                    description=description,
                    dataset_path=dataset_path,
                    target_column=target_column,
                )
                if project_id:
                    logger.info(f"Projet créé : {name}")
                    self.refresh_projects()
                else:
                    logger.error("Impossible de créer le projet")

    def refresh_projects(self):
        """Rafraîchit la liste des projets."""
        self.project_list.clear()
        projects = self.project_manager.list_projects(archived=False)
        for project in projects:
            item = QListWidgetItem(f"{project['name']} ({project['id']})")
            item.setData(Qt.UserRole, project['id'])
            self.project_list.addItem(item)

    def on_project_clicked(self, item):
        """Appelé quand un projet est cliqué."""
        project_id = item.data(Qt.UserRole)
        self.on_project_selected(project_id)
