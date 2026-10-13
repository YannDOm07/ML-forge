import sqlite3

import pytest

from forge.db import now_iso, session, transaction
from forge.projects import create_project
from forge.ui import queries
from forge.ui.viewmodel import _column_profile, _protocol_view, build_viewmodel


def test_empty_workspace_has_all_nine_panels(workspace):
    state = build_viewmodel(workspace)

    assert state["projects"] == []
    assert state["focus"]["screen"] == "overview"
    assert {
        "overview", "campaign", "experiments", "hypotheses", "data",
        "validation", "deliverables", "memory", "journal",
    } <= state.keys()


def test_project_viewmodel_is_display_only_and_has_no_data_rows(workspace):
    project = create_project(workspace, "Titanic", "https://www.kaggle.com/competitions/titanic")

    state = build_viewmodel(workspace, project.id)

    assert state["selected_project"] == project.id
    assert state["overview"]["platform"] == "kaggle"
    assert state["overview"]["project"]["name"] == "Titanic"
    assert state["data"]["profile"]["available"] is False
    assert state["focus"]["phase"] == "acquire"
    assert state["experiments"] == []
    assert queries.list_projects(workspace.global_db)[0]["id"] == project.id
    with queries._connect_ro(project.db_path) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO user_actions (kind, message, status, created_at) VALUES ('x', 'x', 'open', 'x')")


def test_unknown_project_is_rejected(workspace):
    with pytest.raises(ValueError, match="Projet inconnu"):
        build_viewmodel(workspace, "inconnu-000000")


def test_facts_from_local_data_are_redacted(workspace):
    project = create_project(workspace, "Privé")
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO competition_facts "
            "(field, value_json, status, source_url, quote, recorded_by, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("target", '"Alice"', "inferred", "file:///private/train.csv#target",
             "name,target\\nAlice,secret", "test", now_iso()),
        )

    state = build_viewmodel(workspace, project.id)

    target = next(fact for fact in state["data"]["facts"] if fact["field"] == "target")
    assert target["value"] == "donnée locale masquée"
    assert target["source"] is None
    assert "Alice" not in str(state)
    assert "secret" not in str(state)


def test_protocol_and_profile_omit_paths_and_target_values():
    protocol = {
        "id": "P123",
        "spec_json": (
            '{"task":{"problem_type":"binary","target":"label","metric":"accuracy",'
            '"train_path":"C:/private/train.csv","test_path":"C:/private/test.csv"},'
            '"classes":["private-label"],"notes":[],"scheme":"kfold"}'
        ),
        "reliability_json": "{}",
        "folds_json": "[]",
    }
    view = _protocol_view(protocol, [])
    profile = _column_profile(
        {"rows": 4, "features": 1, "target": {"missing": 0, "classes": {"private-label": 4}},
         "columns": [], "twins": {}},
        view["task"],
    )

    assert "train_path" not in view["task"]
    assert "private-label" not in str(view)
    assert "private-label" not in str(profile)
    assert profile["target"] == "label"
    assert profile["target_stats"]["class_count"] == 1
