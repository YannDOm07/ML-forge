"""Backups (zip per project, snapshot of the global database) and restoration."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from forge.db import now_iso, session, snapshot_database, timestamp, transaction
from forge.errors import ProjectError
from forge.projects import MANIFEST, PROJECT_DB, PROJECT_ID_PATTERN, Project, open_project
from forge.workspace import Workspace

_DB_FILES = {PROJECT_DB, f"{PROJECT_DB}-wal", f"{PROJECT_DB}-shm"}


def backup_project(workspace: Workspace, project: Project) -> Path:
    """Zip the project directory, with a consistent snapshot of its database."""
    dest = workspace.backups_dir / f"{project.id}-{timestamp()}.zip"
    workspace.backups_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / PROJECT_DB
        with session(project.db_path) as conn:
            snapshot_database(conn, snapshot)
        with zipfile.ZipFile(dest, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(project.path.rglob("*")):
                relative = file.relative_to(project.path).as_posix()
                if file.is_file() and relative not in _DB_FILES:
                    archive.write(file, relative)
            archive.write(snapshot, PROJECT_DB)
    return dest


def backup_global(workspace: Workspace) -> Path:
    with session(workspace.global_db) as conn:
        return snapshot_database(conn, workspace.backups_dir / f"forge-{timestamp()}.db")


def restore_project(workspace: Workspace, archive_path: Path) -> Project:
    """Restore a project from a zip. Never overwrites an existing project."""
    with zipfile.ZipFile(archive_path) as archive:
        _check_members(archive)
        manifest = json.loads(archive.read(MANIFEST))
        project_id = manifest.get("id", "")
        if not PROJECT_ID_PATTERN.match(project_id):
            raise ProjectError(f"{archive_path} : identifiant de projet invalide dans le manifeste.")
        _ensure_restorable(workspace, project_id)
        target = workspace.projects_dir / project_id
        target.mkdir(parents=True, exist_ok=False)
        archive.extractall(target)
    _register_restored(workspace, manifest)
    return open_project(workspace, project_id)


def _check_members(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    if MANIFEST not in names or PROJECT_DB not in names:
        raise ProjectError("Archive invalide : manifeste ou base de projet absent.")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
            raise ProjectError(f"Archive refusée : chemin dangereux « {name} ».")


def _ensure_restorable(workspace: Workspace, project_id: str) -> None:
    with session(workspace.global_db) as conn:
        row = conn.execute("SELECT status FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is not None and row["status"] != "deleted":
        raise ProjectError(f"Le projet {project_id} existe déjà : restauration refusée pour ne pas l'écraser.")
    if (workspace.projects_dir / project_id).exists():
        raise ProjectError(f"Le dossier du projet {project_id} existe déjà : restauration refusée.")


def _register_restored(workspace: Workspace, manifest: dict) -> None:
    now = now_iso()
    with session(workspace.global_db) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO projects (id, name, competition_url, status, created_at, updated_at)"
            " VALUES (?, ?, ?, 'active', ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET status = 'active', updated_at = excluded.updated_at",
            (manifest["id"], manifest.get("name", manifest["id"]), manifest.get("competition_url"),
             manifest.get("created_at", now), now),
        )
