"""Data sources stay where they are; ML Forge records their fingerprint and detects changes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from forge.db import now_iso, session, transaction
from forge.errors import DataSourceError

ROLES = {"official", "external", "inbox"}
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class SourceStatus:
    path: Path
    role: str
    state: str  # "ok", "missing" or "modified"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def register_data_source(
    project_db: Path, path: Path, role: str, origin_url: str | None = None, license: str | None = None
) -> None:
    """Record a file once. Re-registering a changed file is refused, never silently replaced."""
    if role not in ROLES:
        raise DataSourceError(f"Rôle de donnée invalide : {role}")
    path = path.resolve()
    if not path.is_file():
        raise DataSourceError(f"Fichier introuvable : {path}")
    stat = path.stat()
    sha256 = file_sha256(path)
    with session(project_db) as conn, transaction(conn):
        known = conn.execute("SELECT sha256 FROM data_sources WHERE path = ?", (str(path),)).fetchone()
        if known is not None:
            if known["sha256"] != sha256:
                raise DataSourceError(f"{path} a changé depuis son enregistrement : nouvelle version à valider.")
            return
        conn.execute(
            "INSERT INTO data_sources (role, path, sha256, size_bytes, mtime_ns, origin_url, license, registered_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (role, str(path), sha256, stat.st_size, stat.st_mtime_ns, origin_url, license, now_iso()),
        )


def check_data_sources(project_db: Path, full: bool = False) -> list[SourceStatus]:
    """Detect missing or modified files. Unchanged size and mtime skip hashing unless `full`."""
    with session(project_db) as conn:
        rows = conn.execute("SELECT * FROM data_sources ORDER BY id").fetchall()
    return [SourceStatus(Path(row["path"]), row["role"], _state(row, full)) for row in rows]


def _state(row, full: bool) -> str:
    path = Path(row["path"])
    if not path.is_file():
        return "missing"
    stat = path.stat()
    unchanged_stat = stat.st_size == row["size_bytes"] and stat.st_mtime_ns == row["mtime_ns"]
    if unchanged_stat and not full:
        return "ok"
    return "ok" if file_sha256(path) == row["sha256"] else "modified"
