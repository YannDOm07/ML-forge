"""Tests 20 (no API: inbox), 21 (403 then resume), 23 (refused external data: nothing downloaded),
24 (forbidden resource never proposed), and the link-to-task chain."""

import json
from collections import namedtuple

import pytest

from fake_web import FakeTransport, html, json_response, raw, zip_of
from forge.acquisition import ExternalResource, ForbiddenByRules, acquire_official, ingest_inbox, request_external
from forge.actions import open_actions
from forge.datasources import check_data_sources
from forge.db import session
from forge.demo import make_synthetic_competition
from forge.errors import ApprovalRequired, ForgeError
from forge.intake import competition_card, read_competition, task_from_card
from forge.kaggle import download_url, files_url, metadata_url
from forge.projects import create_project
from forge.runner import define_protocol
from forge.web import NetworkError

KAGGLE_URL = "https://www.kaggle.com/competitions/flows-cup"
SLUG = "flows-cup"


@pytest.fixture
def competition(tmp_path):
    return make_synthetic_competition(tmp_path / "source", n_train=120, n_test=30)


@pytest.fixture
def kaggle_token(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    monkeypatch.setenv("KAGGLE_KEY", "secret-key-123")


def kaggle_routes(competition, download):
    files = [{"name": p.name, "totalBytes": p.stat().st_size} for p in competition.values()]
    metadata = [{"ref": KAGGLE_URL, "title": "Flows Cup", "evaluationMetric": "Area Under ROC Curve"}]
    return {
        KAGGLE_URL: html("<div id='app'></div>"),
        metadata_url(SLUG): json_response(json.dumps(metadata)),
        files_url(SLUG): json_response(json.dumps(files)),
        download_url(SLUG): download,
    }


def sources(project):
    with session(project.db_path) as conn:
        return [dict(row) for row in conn.execute("SELECT role, path FROM data_sources")]


def test_missing_token_gives_precise_instructions(workspace, competition):
    project = create_project(workspace, "Kaggle", KAGGLE_URL)
    transport = FakeTransport({})
    result = acquire_official(workspace, project, transport)
    assert result.status == "needs_user"
    assert "kaggle.json" in result.message and "Ne collez jamais" in result.message
    assert transport.calls == []
    assert open_actions(project)[0]["kind"] == "credentials"


def test_rules_not_accepted_then_resume(workspace, competition, kaggle_token):
    project = create_project(workspace, "Kaggle", KAGGLE_URL)
    routes = kaggle_routes(competition, json_response('{"message": "You must accept the rules"}', 403))
    transport = FakeTransport(routes)

    blocked = acquire_official(workspace, project, transport)
    assert blocked.status == "needs_user"
    assert f"https://www.kaggle.com/competitions/{SLUG}/rules" in blocked.message
    assert [a["kind"] for a in open_actions(project)] == ["accept_rules"]
    assert sources(project) == []

    routes[download_url(SLUG)] = raw(zip_of({p.name: p for p in competition.values()}), content_type="application/zip")
    done = acquire_official(workspace, project, transport)
    assert done.status == "done"
    assert sorted(p.name for p in done.files) == ["sample_submission.csv", "test.csv", "train.csv"]
    assert open_actions(project) == []
    assert {s["role"] for s in sources(project)} == {"official"}
    assert not (workspace.data_dir / "kaggle" / f"{SLUG}.zip").exists()  # temporary archive removed


def test_link_to_task_chain(workspace, competition, kaggle_token):
    project = create_project(workspace, "Kaggle", KAGGLE_URL)
    routes = kaggle_routes(competition, raw(zip_of({p.name: p for p in competition.values()}), content_type="application/zip"))
    transport = FakeTransport(routes)
    read_competition(workspace, project, transport=transport)
    acquire_official(workspace, project, transport)

    card = competition_card(project)
    assert card["critical_unknowns"] == []
    assert card["fields"]["target"]["value"] == "target"
    assert card["fields"]["problem_type"]["value"] == "binary"
    task = task_from_card(project, [s.path for s in check_data_sources(project.db_path)])
    assert (task.metric, task.id_column) == ("roc_auc", "id")
    assert define_protocol(workspace, project, task).id.startswith("P")


def test_shared_cache_is_reused_without_download(workspace, competition, kaggle_token):
    routes = kaggle_routes(competition, raw(zip_of({p.name: p for p in competition.values()}), content_type="application/zip"))
    first = create_project(workspace, "Un", KAGGLE_URL)
    acquire_official(workspace, first, FakeTransport(routes))
    second = create_project(workspace, "Deux", KAGGLE_URL)
    transport = FakeTransport(routes)
    result = acquire_official(workspace, second, transport)
    assert result.status == "done" and "réutilisées" in result.message
    assert transport.calls == []
    assert sorted(s["path"] for s in sources(first)) == sorted(s["path"] for s in sources(second))


def test_oversized_competition_is_not_downloaded(workspace, competition, kaggle_token):
    project = create_project(workspace, "Gros", KAGGLE_URL)
    routes = kaggle_routes(competition, raw(b"zip"))
    routes[files_url(SLUG)] = json_response(json.dumps([{"name": "huge.csv", "totalBytes": 50 * 1024**3}]))
    transport = FakeTransport(routes)
    result = acquire_official(workspace, project, transport)
    assert result.status == "refused" and "max_download_gb" in result.message
    assert all(method != "DOWNLOAD" for method, _, _ in transport.calls)


def test_disk_reserve_is_enforced(workspace, competition, kaggle_token, monkeypatch):
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("forge.web.shutil.disk_usage", lambda path: Usage(100, 90, 10 * 1024**3))
    project = create_project(workspace, "Disque", KAGGLE_URL)
    transport = FakeTransport(kaggle_routes(competition, raw(b"zip")))
    result = acquire_official(workspace, project, transport)
    assert result.status == "refused" and "Espace disque insuffisant" in result.message
    assert all(method != "DOWNLOAD" for method, _, _ in transport.calls)


def test_platform_without_api_falls_back_to_inbox(workspace, competition):
    url = "https://zindi.africa/competitions/flows"
    project = create_project(workspace, "Zindi", url)
    transport = FakeTransport({
        url: html("<p>Data</p><a href='/competitions/flows/data/Train.csv'>Train.csv</a>"),
        "https://zindi.africa/competitions/flows/data/Train.csv": html("<form>Please log in</form>"),
    })
    read_competition(workspace, project, transport=transport)
    result = acquire_official(workspace, project, transport)
    inbox = project.path / "inbox"
    assert result.status == "needs_user"
    assert str(inbox) in result.message and "Train.csv" in result.message
    assert any(a["kind"] == "inbox" for a in open_actions(project))

    for path in competition.values():
        (inbox / path.name).write_bytes(path.read_bytes())
    ingested = ingest_inbox(workspace, project)
    assert ingested.status == "done" and len(ingested.files) == 3
    assert {s["role"] for s in sources(project)} == {"inbox"}
    assert all(a["kind"] != "inbox" for a in open_actions(project))
    assert competition_card(project)["fields"]["target"]["value"] == "target"


def test_public_data_links_are_downloaded(workspace, competition):
    url = "https://example.org/challenge"
    project = create_project(workspace, "Public", url)
    transport = FakeTransport({
        url: html("<a href='/files/train.csv'>train</a><a href='/files/test.csv'>test</a>"),
        "https://example.org/files/train.csv": raw(competition["train"].read_bytes()),
        "https://example.org/files/test.csv": raw(competition["test"].read_bytes()),
    })
    read_competition(workspace, project, transport=transport)
    result = acquire_official(workspace, project, transport)
    assert result.status == "done"
    assert sorted(p.name for p in result.files) == ["test.csv", "train.csv"]
    assert all(str(workspace.data_dir) in s["path"] for s in sources(project))


def test_archive_with_path_traversal_is_refused(workspace, kaggle_token, competition):
    project = create_project(workspace, "Piège", KAGGLE_URL)
    evil = raw(zip_of({"../../evil.csv": "x"}), content_type="application/zip")
    with pytest.raises(ForgeError, match="chemin dangereux"):
        acquire_official(workspace, project, FakeTransport(kaggle_routes(competition, evil)))
    assert not (workspace.data_dir / "evil.csv").exists()


RESOURCE = ExternalResource("data", "https://open.example/geo.csv", "Open Geo", "CC-BY-4.0", 1, "features géographiques",
                            "décalage de distribution possible")


def rules_page(workspace, project, sentence):
    url = project.competition_url
    read_competition(workspace, project, transport=FakeTransport({url: html(f"<p>{sentence}</p>")}))


def test_refused_external_data_is_never_downloaded(workspace):
    project = create_project(workspace, "Externe", "https://example.org/c")
    rules_page(workspace, project, "Good luck to everyone.")
    transport = FakeTransport({RESOURCE.url: raw("a,b\n1,2\n")})
    with pytest.raises(ApprovalRequired, match="incertain"):
        request_external(workspace, project, RESOURCE, lambda request: False, "test", transport)
    assert transport.calls == []
    assert all(s["role"] != "external" for s in sources(project))
    with session(project.db_path) as conn:
        assert conn.execute("SELECT outcome FROM decisions").fetchone()["outcome"] == "refused"


def test_forbidden_external_data_is_never_proposed(workspace):
    project = create_project(workspace, "Interdit", "https://example.org/c")
    rules_page(workspace, project, "External data is not allowed in this competition.")

    def approver(request):
        raise AssertionError("une ressource interdite ne doit jamais être proposée")

    transport = FakeTransport({RESOURCE.url: raw("a,b\n1,2\n")})
    with pytest.raises(ForbiddenByRules, match="External data is not allowed"):
        request_external(workspace, project, RESOURCE, approver, "test", transport)
    assert transport.calls == []


def test_accepted_external_data_is_traced(workspace):
    project = create_project(workspace, "Accepté", "https://example.org/c")
    rules_page(workspace, project, "External data is allowed if it is publicly available.")
    shown = []
    transport = FakeTransport({RESOURCE.url: raw("a,b\n1,2\n")})
    result = request_external(workspace, project, RESOURCE, lambda r: shown.append(r) or True, "test", transport)
    assert result.status == "done"
    assert "incertain" in shown[0].details["conformité"]  # an inferred permission is not a green light
    assert [s["role"] for s in sources(project)] == ["external"]
    with session(project.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM compliance_log").fetchone()[0] == 1


def test_non_http_urls_are_refused(workspace):
    from forge.web import Web

    with pytest.raises(NetworkError, match="http ou https"):
        Web(workspace, FakeTransport({})).read("file:///C:/Users/secret.txt", "test")
