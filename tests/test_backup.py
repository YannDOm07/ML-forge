"""Backup and restore, never overwriting; migration backups."""

import zipfile

import pytest

from conftest import make_card
from forge.backup import backup_global, backup_project, restore_project
from forge.db import migrate_database, session
from forge.errors import ProjectError
from forge.experiments import list_experiments, register_experiment
from forge.projects import create_project, delete_project


def test_backup_then_restore_after_deletion(workspace):
    project = create_project(workspace, "Sauvé")
    register_experiment(project.db_path, make_card())
    archive = backup_project(workspace, project)
    delete_project(workspace, project.id, lambda request: True, decided_by="test")

    restored = restore_project(workspace, archive)

    assert restored.id == project.id
    assert [e.id for e in list_experiments(restored.db_path)] == ["E0001"]


def test_restore_never_overwrites_an_existing_project(workspace):
    project = create_project(workspace, "Unique")
    archive = backup_project(workspace, project)
    with pytest.raises(ProjectError, match="existe déjà"):
        restore_project(workspace, archive)


def test_restore_rejects_path_traversal(workspace, tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("manifest.json", '{"id": "evil-abcdef"}')
        archive.writestr("project.db", b"")
        archive.writestr("../../outside.txt", "x")
    with pytest.raises(ProjectError, match="chemin dangereux"):
        restore_project(workspace, evil)
    assert not (workspace.root.parent / "outside.txt").exists()


def test_global_backup_is_a_readable_database(workspace):
    create_project(workspace, "Global")
    snapshot = backup_global(workspace)
    with session(snapshot) as conn:
        assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1


def test_migration_backs_up_existing_database_first(workspace, tmp_path):
    db = tmp_path / "x.db"
    migrate_database(db, ["CREATE TABLE a (id INTEGER);"], workspace.backups_dir)
    assert list(workspace.backups_dir.iterdir()) == []  # new database: nothing to save

    migrate_database(db, ["CREATE TABLE a (id INTEGER);", "CREATE TABLE b (id INTEGER);"], workspace.backups_dir)

    backups = list(workspace.backups_dir.iterdir())
    assert len(backups) == 1 and "-v1-" in backups[0].name


def test_failed_migration_is_rolled_back(tmp_path):
    db = tmp_path / "y.db"
    migrate_database(db, ["CREATE TABLE a (id INTEGER);"], None)
    with pytest.raises(Exception, match="v2"):
        migrate_database(db, ["CREATE TABLE a (id INTEGER);", "CREATE TABLE b (id); SYNTAX ERROR;"], None)
    with session(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == {"a"}
