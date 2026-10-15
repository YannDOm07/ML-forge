"""v3 screens: every project at a glance, the competition, data at scale, files, project journal
and the lessons of the project's context — and the same state served by both front ends."""

import pytest

from forge.datasources import register_data_sources
from forge.intake import store_page
from forge.lessons import record
from forge.organize import create_from_link, journal
from forge.projects import create_project
from forge.ui.viewmodel import SOURCES_SHOWN, build_viewmodel


@pytest.fixture
def projects(workspace, tmp_path):
    first = create_project(workspace, "Lyrics Cup", "https://eval.ai/web/challenges/challenge-page/2703/overview")
    second = create_project(workspace, "House Prices", "https://www.kaggle.com/competitions/house-prices")
    journal(first, "Acquérir", "Archive d'entraînement en attente de l'utilisateur.", problem=True)
    journal(first, "Comprendre", "27 pages lues, 9 dépôts décrits.")
    store_page(first, "https://zenodo.org/api/records/222", "api",
               "Enregistrement Zenodo (jeu de données) : 222\nAdresse : https://zenodo.org/records/222\n"
               '"access_right": "open"\nFichiers : 3, taille totale 16.67 Go\n- train.tar.gz (13.79 Go)',
               "Enregistrement Zenodo 222")
    clips = []
    for index in range(SOURCES_SHOWN + 20):
        clip = tmp_path / "data" / "train" / "signals" / f"clip{index:04d}.flac"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"x" * (index + 1))
        clips.append(clip)
    table = tmp_path / "data" / "train_metadata.json"
    table.write_text("[]", encoding="utf-8")
    register_data_sources(first.db_path, [table, *clips], "official")
    return first, second


def test_portfolio_shows_every_project(workspace, projects):
    first, second = projects
    state = build_viewmodel(workspace, first.id)
    names = {item["name"]: item for item in state["portfolio"]}
    assert set(names) == {"Lyrics Cup", "House Prices"}
    assert names["Lyrics Cup"]["platform"] == "evalai" and names["House Prices"]["platform"] == "kaggle"
    assert names["Lyrics Cup"]["last_step"]["step"] == "Comprendre" and names["Lyrics Cup"]["problems"] == 1


def test_competition_repositories_journal_and_files(workspace, projects):
    first, _ = projects
    state = build_viewmodel(workspace, first.id)
    repository = state["competition"]["repositories"][0]
    assert repository["address"] == "https://zenodo.org/records/222"
    assert repository["files"] == 3 and repository["size"] == "16.67 Go" and repository["access"] == "open"
    entries = state["project_journal"]
    assert [entry["step"] for entry in entries[:2]] == ["Comprendre", "Acquérir"]  # newest first
    assert entries[1]["problem"] and not entries[0]["problem"]
    assert "JOURNAL.md" in state["files"]["documents"] or state["files"]["documents"] == []


def test_thousands_of_files_are_summarised_never_sent_whole(workspace, projects):
    first, _ = projects
    data = build_viewmodel(workspace, first.id)["data"]
    assert data["sources_total"] == SOURCES_SHOWN + 21 and len(data["sources"]) == SOURCES_SHOWN
    assert data["sources"][0]["name"] == "train_metadata.json"  # tables first
    summary = data["summary"]
    assert summary["files"] == SOURCES_SHOWN + 21
    assert {"type": ".flac", "files": SOURCES_SHOWN + 20} in summary["types"]
    assert summary["folders"][0]["files"] == SOURCES_SHOWN + 20


def test_lessons_follow_the_project_context(workspace, projects):
    first, second = projects
    record(workspace, "comprendre", "piege", "EvalAI : l'API est sur un autre domaine",
           "La configuration du site déclare l'adresse de l'API.", {"platform": "evalai"})
    lessons = build_viewmodel(workspace, first.id)["lessons"]
    titles = [item["titre"] for item in lessons["by_step"].get("comprendre", [])]
    assert "EvalAI : l'API est sur un autre domaine" in titles
    assert lessons["context"]["platform"] == "evalai" and lessons["register"]["total"] >= 1
    assert build_viewmodel(workspace, second.id)["lessons"]["context"]["platform"] == "kaggle"


def test_project_named_from_link_appears_in_the_portfolio(workspace):
    from fake_web import FakeTransport, json_response, raw

    site, api = "https://challenges.example.org", "https://challenges.example.org/api/"
    url = f"{site}/web/challenges/challenge-page/5/overview"
    transport = FakeTransport({
        f"{site}/dist/js/config.min.js": raw("{}", content_type="application/javascript"),
        f"{api}challenges/challenge/5/": json_response('{"id": 5, "title": "Radio Cup"}'),
        f"{api}challenges/challenge/5/challenge_phase": json_response('{"results": []}'),
    })
    project = create_from_link(workspace, url, transport=transport)
    assert [item["name"] for item in build_viewmodel(workspace, project.id)["portfolio"]] == ["Radio Cup"]


def test_fastapi_serves_the_same_detailed_dashboard(workspace, projects, monkeypatch, config_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import forge.web_server as web

    first, _ = projects
    monkeypatch.setenv("MLFORGE_CONFIG", str(config_path))
    with TestClient(web.app) as client:
        state = client.get("/api/state", params={"project_id": first.id})
        assert state.status_code == 200
        assert state.json()["selected_project"] == first.id and len(state.json()["portfolio"]) == 2
        assert client.get("/ui/").status_code == 200 and "ML Forge" in client.get("/ui/").text
        assert client.get("/ui/static/ui.js").status_code == 200
        assert client.get("/api/state", params={"project_id": "inconnu-000000"}).status_code == 404
