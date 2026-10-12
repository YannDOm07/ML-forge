"""The data root: directory layout, global database, configuration history."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from forge.config import Config
from forge.db import migrate_database, now_iso, session, transaction
from forge.schema import GLOBAL_MIGRATIONS


@dataclass(frozen=True)
class Workspace:
    config: Config

    @property
    def root(self) -> Path:
        return self.config.root

    @property
    def projects_dir(self) -> Path:
        return self.root / "projects"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def backups_dir(self) -> Path:
        return self.root / "backups"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def global_db(self) -> Path:
        return self.root / "forge.db"

    def migration_backup_dir(self) -> Path | None:
        return self.backups_dir if self.config.get("backup", "before_migration") else None


def open_workspace(config: Config) -> Workspace:
    workspace = Workspace(config)
    for directory in (workspace.projects_dir, workspace.data_dir, workspace.backups_dir):
        directory.mkdir(parents=True, exist_ok=True)
    migrate_database(workspace.global_db, GLOBAL_MIGRATIONS, workspace.migration_backup_dir())
    _record_configuration(workspace)
    return workspace


def _record_configuration(workspace: Workspace) -> None:
    config_hash = hashlib.sha256(workspace.config.text.encode("utf-8")).hexdigest()
    with session(workspace.global_db) as conn, transaction(conn):
        last = conn.execute(
            "SELECT config_hash FROM configuration_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last is None or last["config_hash"] != config_hash:
            conn.execute(
                "INSERT INTO configuration_history (config_hash, content, recorded_at) VALUES (?, ?, ?)",
                (config_hash, workspace.config.text, now_iso()),
            )


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def disk_usage(workspace: Workspace) -> dict[str, int]:
    """Bytes used per area of the data root, projects listed individually."""
    usage = {"data": directory_size(workspace.data_dir), "backups": directory_size(workspace.backups_dir)}
    usage["logs"] = directory_size(workspace.logs_dir)
    usage["forge.db"] = workspace.global_db.stat().st_size if workspace.global_db.exists() else 0
    if workspace.projects_dir.exists():
        for project_dir in sorted(workspace.projects_dir.iterdir()):
            usage[f"projects/{project_dir.name}"] = directory_size(project_dir)
    return usage
