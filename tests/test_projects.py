"""Tests 1 (no loss), 2 (persistence), 3 (isolation), 9 (destructive ops), 10 (file sobriety)."""

import pytest

from conftest import make_card
from forge.config import load_config
from forge.datasources import register_data_source
from forge.db import session
from forge.errors import ApprovalRequired, ProjectError, StorageError
from forge.experiments import list_experiments, register_experiment
from forge.guardrails import refuse_all
from forge.projects import (
    archive_project,
    create_project,
    delete_project,
    list_projects,
    new_project_id,
    open_project,
)
from forge.workspace import open_workspace


def test_creating_projects_never_alters_previous_ones(workspace):
    first = create_project(workspace, "Titanic")
    register_experiment(first.db_path, make_card())
    manifest_before = (first.path / "manifest.json").read_bytes()
    db_before = first.db_path.read_bytes()

    others = [create_project(workspace, "Titanic") for _ in range(3)]

    assert len({p.id for p in [first, *others]}) == 4
    assert [p.id for p in list_projects(workspace)] == [first.id] + [p.id for p in others]
    assert (first.path / "manifest.json").read_bytes() == manifest_before
    assert first.db_path.read_bytes() == db_before


def test_projects_persist_after_restart(config_path):
    workspace = open_workspace(load_config(config_path))
    project = create_project(workspace, "Playground S5", "https://www.kaggle.com/competitions/playground")
    register_experiment(project.db_path, make_card())

    restarted = open_workspace(load_config(config_path))
    reopened = open_project(restarted, project.id)

    assert reopened.competition_url == "https://www.kaggle.com/competitions/playground"
    assert [e.id for e in list_experiments(reopened.db_path)] == ["E0001"]


def test_projects_are_isolated(workspace, tmp_path):
    a = create_project(workspace, "A")
    b = create_project(workspace, "B")
    data = tmp_path / "train.csv"
    data.write_text("x,y\n1,0\n", encoding="utf-8")
    register_experiment(a.db_path, make_card())
    register_data_source(a.db_path, data, "official")

    assert a.db_path != b.db_path
    assert list_experiments(b.db_path) == []
    with session(b.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM data_sources").fetchone()[0] == 0


def test_project_id_is_slugged_and_unique():
    first, second = new_project_id("Défi ITU — Packet Flows!"), new_project_id("Défi ITU — Packet Flows!")
    assert first.startswith("defi-itu-packet-flows-")
    assert first != second


def test_empty_name_is_refused(workspace):
    with pytest.raises(ProjectError):
        create_project(workspace, "   ")


def test_archive_hides_without_deleting(workspace):
    project = create_project(workspace, "Ancien")
    archive_project(workspace, project.id)
    assert list_projects(workspace) == []
    assert [p.id for p in list_projects(workspace, include_archived=True)] == [project.id]
    assert project.db_path.is_file()
    assert open_project(workspace, project.id).status == "archived"


def test_delete_without_approval_is_blocked_and_recorded(workspace):
    project = create_project(workspace, "Garder")
    with pytest.raises(ApprovalRequired):
        delete_project(workspace, project.id, refuse_all, decided_by="test")
    assert project.db_path.is_file()
    with session(project.db_path) as conn:
        decision = conn.execute("SELECT action, outcome FROM decisions").fetchone()
    assert (decision["action"], decision["outcome"]) == ("delete", "refused")


def test_truthy_but_not_true_answer_is_not_an_approval(workspace):
    project = create_project(workspace, "Garder")
    with pytest.raises(ApprovalRequired):
        delete_project(workspace, project.id, lambda request: "oui", decided_by="test")
    assert project.path.is_dir()


def test_approved_delete_keeps_a_backup(workspace):
    project = create_project(workspace, "Jetable")
    backup = delete_project(workspace, project.id, lambda request: True, decided_by="test")
    assert backup.is_file()
    assert not project.path.exists()
    assert list_projects(workspace, include_archived=True) == []
    with pytest.raises(ProjectError, match="inconnu"):
        open_project(workspace, project.id)


def test_missing_project_files_are_reported(workspace):
    project = create_project(workspace, "Fragile")
    (project.path / "manifest.json").rename(project.path / "manifest.bak")
    with pytest.raises(ProjectError, match="introuvables"):
        open_project(workspace, project.id)


def test_newer_schema_is_refused(workspace):
    project = create_project(workspace, "Futur")
    with session(project.db_path) as conn:
        conn.execute("PRAGMA user_version = 99")
    with pytest.raises(StorageError, match="plus récent"):
        open_project(workspace, project.id)


def test_no_excessive_file_creation(workspace, tmp_path):
    project = create_project(workspace, "Sobre")
    data = tmp_path / "train.csv"
    data.write_text("x,y\n1,0\n", encoding="utf-8")
    for _ in range(5):
        register_experiment(project.db_path, make_card())
    register_data_source(project.db_path, data, "official")
    open_project(workspace, project.id)

    assert sorted(p.name for p in project.path.iterdir()) == ["manifest.json", "project.db"]
    assert sorted(p.name for p in workspace.root.iterdir()) == ["backups", "data", "forge.db", "projects"]
    assert list(workspace.backups_dir.iterdir()) == []
    assert list(workspace.data_dir.iterdir()) == []
