"""Projects: each one has a unique id, its own directory, manifest and database."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path

from forge.db import migrate_database, now_iso, session, transaction
from forge.errors import ProjectError
from forge.experiments import recover_interrupted
from forge.guardrails import Action, ApprovalRequest, Approver, project_decision_recorder, require_approval
from forge.schema import PROJECT_MIGRATIONS
from forge.workspace import Workspace

MANIFEST = "manifest.json"
PROJECT_DB = "project.db"
PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9-]{1,40}-[0-9a-f]{6}$")


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    path: Path
    competition_url: str | None
    status: str
    created_at: str

    @property
    def db_path(self) -> Path:
        return self.path / PROJECT_DB


def new_project_id(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:32].strip("-") or "projet"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def create_project(workspace: Workspace, name: str, competition_url: str | None = None) -> Project:
    name = name.strip()
    if not name:
        raise ProjectError("Le nom du projet est vide.")
    project_id = new_project_id(name)
    created_at = now_iso()
    path = workspace.projects_dir / project_id
    with session(workspace.global_db) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO projects (id, name, competition_url, status, created_at, updated_at)"
            " VALUES (?, ?, ?, 'active', ?, ?)",
            (project_id, name, competition_url, created_at, created_at),
        )
        _create_project_files(workspace, path, project_id, name, competition_url, created_at)
    return Project(project_id, name, path, competition_url, "active", created_at)


def _create_project_files(
    workspace: Workspace, path: Path, project_id: str, name: str, url: str | None, created_at: str
) -> None:
    path.mkdir(parents=False, exist_ok=False)
    try:
        manifest = {"id": project_id, "name": name, "competition_url": url, "created_at": created_at, "format": 1}
        (path / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        migrate_database(path / PROJECT_DB, PROJECT_MIGRATIONS, workspace.migration_backup_dir())
    except BaseException:
        # Only the directory created a moment ago by this call is removed.
        shutil.rmtree(path, ignore_errors=True)
        raise


def list_projects(workspace: Workspace, include_archived: bool = False) -> list[Project]:
    statuses = ("active", "archived") if include_archived else ("active",)
    placeholders = ", ".join("?" for _ in statuses)
    with session(workspace.global_db) as conn:
        rows = conn.execute(
            f"SELECT * FROM projects WHERE status IN ({placeholders}) ORDER BY created_at, rowid", statuses
        ).fetchall()
    return [_to_project(workspace, row) for row in rows]


def open_project(workspace: Workspace, project_id: str) -> Project:
    """Check the project's files, migrate its database and recover interrupted experiments."""
    project = _get_project(workspace, project_id)
    if not (project.path / MANIFEST).is_file() or not project.db_path.is_file():
        raise ProjectError(
            f"Les fichiers du projet {project_id} sont introuvables dans {project.path}. "
            "Restaurez une sauvegarde avec : python -m forge restore <fichier.zip>"
        )
    manifest = json.loads((project.path / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("id") != project_id:
        raise ProjectError(f"Le manifeste de {project.path} ne correspond pas au projet {project_id}.")
    migrate_database(project.db_path, PROJECT_MIGRATIONS, workspace.migration_backup_dir())
    recover_interrupted(project.db_path)
    return project


def archive_project(workspace: Workspace, project_id: str) -> None:
    """Hide a project from the default listing. Nothing is deleted."""
    _get_project(workspace, project_id)
    _set_status(workspace, project_id, "archived")


def delete_project(workspace: Workspace, project_id: str, approver: Approver, decided_by: str) -> Path:
    """Delete a project's files after explicit approval, keeping a backup. Returns the backup path."""
    from forge.backup import backup_project  # backup depends on this module

    project = open_project(workspace, project_id)
    request = ApprovalRequest(
        Action.DELETE,
        f"projet {project.id} ({project.name})",
        {"dossier": str(project.path), "note": "une sauvegarde zip est faite avant suppression"},
    )
    require_approval(request, approver, project_decision_recorder(project.db_path, decided_by))
    backup = backup_project(workspace, project)
    shutil.rmtree(project.path)
    _set_status(workspace, project_id, "deleted")
    return backup


def _get_project(workspace: Workspace, project_id: str) -> Project:
    with session(workspace.global_db) as conn:
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None or row["status"] == "deleted":
        raise ProjectError(f"Projet inconnu : {project_id}")
    return _to_project(workspace, row)


def _set_status(workspace: Workspace, project_id: str, status: str) -> None:
    with session(workspace.global_db) as conn, transaction(conn):
        conn.execute(
            "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?", (status, now_iso(), project_id)
        )


def _to_project(workspace: Workspace, row) -> Project:
    return Project(
        id=row["id"],
        name=row["name"],
        path=workspace.projects_dir / row["id"],
        competition_url=row["competition_url"],
        status=row["status"],
        created_at=row["created_at"],
    )
