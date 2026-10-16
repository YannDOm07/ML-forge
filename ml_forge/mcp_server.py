"""MCP tools exposing ML Forge to OpenCode."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from ml_forge.core.project_manager import ProjectManager

mcp = FastMCP("ML Forge")


def _project_or_raise(project_id: str):
    project = ProjectManager().open_project(project_id)
    if project is None:
        raise ValueError(f"Projet ML Forge introuvable : {project_id}")
    return project


@mcp.tool()
def list_projects(include_archived: bool = False) -> list[Dict[str, Any]]:
    """Liste les projets ML Forge disponibles."""
    return ProjectManager().list_projects(archived=include_archived)


@mcp.tool()
def create_project(name: str, description: str = "") -> Dict[str, str]:
    """Crée un projet ML Forge isolé et retourne son identifiant."""
    project_id = ProjectManager().create_project(name=name, description=description)
    if project_id is None:
        raise RuntimeError(f"Impossible de créer le projet « {name} ».")
    return {"project_id": project_id, "name": name}


@mcp.tool()
def get_project(project_id: str) -> Dict[str, Any]:
    """Retourne les informations principales d'un projet."""
    project = _project_or_raise(project_id)
    return {
        "id": project.id,
        "name": project.get_name(),
        "description": project.metadata.get("description", ""),
        "objective": project.get_objective(),
        "dataset_path": project.get_dataset_path(),
        "target_column": project.get_target_column(),
    }


@mcp.tool()
def inspect_dataset(
    project_id: str,
    dataset_path: str,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspecte un dataset local et retourne dimensions, colonnes, manquants et risques."""
    return _project_or_raise(project_id).inspect_dataset(dataset_path, target_column)


@mcp.tool()
def run_baseline_experiment(
    project_id: str,
    dataset_path: str,
    target_column: str,
    hypothesis: Optional[str] = None,
) -> Dict[str, Any]:
    """Exécute une baseline scikit-learn et enregistre expérience et métriques au projet."""
    return _project_or_raise(project_id).run_baseline_experiment(
        dataset_path=dataset_path,
        target_column=target_column,
        hypothesis=hypothesis,
    )


@mcp.tool()
def list_experiments(project_id: str) -> list[Dict[str, Any]]:
    """Retourne les expériences déjà enregistrées dans un projet."""
    return _project_or_raise(project_id).get_experiments()


def main() -> None:
    """Lance le transport stdio MCP utilisé par OpenCode."""
    ml_forge_logger = logging.getLogger("ml_forge")
    for handler in ml_forge_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.stream = sys.stderr
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
