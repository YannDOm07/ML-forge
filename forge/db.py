"""SQLite helpers: connections, transactions, versioned migrations, snapshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from forge.errors import StorageError


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp() -> str:
    """Sortable, filename-safe local timestamp."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path) -> Iterator[sqlite3.Connection]:
    """Short-lived connection, always closed (no lingering -wal/-shm files)."""
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate_database(path: Path, migrations: Sequence[str], backup_dir: Path | None) -> int:
    """Bring the database at `path` to the latest version; return that version."""
    existed = path.exists()
    target = len(migrations)
    with session(path) as conn:
        current = schema_version(conn)
        if current > target:
            raise StorageError(
                f"{path} utilise le schéma v{current}, plus récent que cette version de ML Forge (v{target})."
            )
        if current == target:
            return current
        if existed and current > 0 and backup_dir is not None:
            snapshot_database(conn, backup_dir / f"{path.parent.name}-{path.stem}-v{current}-{timestamp()}.db")
        for version in range(current + 1, target + 1):
            _apply_migration(conn, version, migrations[version - 1])
    return target


def _apply_migration(conn: sqlite3.Connection, version: int, sql: str) -> None:
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{sql}\nPRAGMA user_version = {version};\nCOMMIT;")
    except sqlite3.Error as exc:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise StorageError(f"La migration v{version} a échoué : {exc}") from exc


def snapshot_database(conn: sqlite3.Connection, dest: Path) -> Path:
    """Consistent copy of a live database, never overwriting an existing file."""
    if dest.exists():
        raise StorageError(f"{dest} existe déjà : instantané non écrasé.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(dest)) as copy:
        conn.backup(copy)
    return dest
