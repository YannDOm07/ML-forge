"""Competitions hosted on EvalAI and the repositories they cite (Hugging Face, GitHub),
read from the link alone; access secrets printed on pages are never kept."""

import json

import pytest

from fake_web import FakeTransport, html, json_response, raw
from forge.db import session
from forge.intake import competition_card, identify_platform, read_competition, read_url, stored_pages
from forge.projects import create_project
from forge.repositories import parse_repository
from forge.web import SECRET_MASK, redact_secrets

SITE = "https://challenges.example.org"
URL = f"{SITE}/web/challenges/challenge-page/77/overview"
API = "https://api.example.net/api/"
DATASET = "https://huggingface.co/datasets/lab/cup-data"
CHALLENGE = {
    "id": 77, "title": "Radio Cup", "end_date": "2026-12-15T23:59:59Z", "is_docker_based": False,
    "is_registration_open": True, "github_repository": "lab/cup-baseline",
    "description": "<p>Predict the channel.</p>",
    "evaluation_details": "<p>Submissions are evaluated using the mean absolute error.</p>",
    "submission_guidelines": f"<p>Data: <a href='{DATASET}'>{DATASET}</a></p>"
                             "<p>Dataset Access Password: <code>HUNTER2026</code></p>",
}
PHASES = {"results": [
    {"name": "Dev", "is_public": True, "is_active": False, "end_date": "2026-09-01T00:00:00Z",
     "max_submissions_per_day": 10},
    {"name": "Test", "is_public": True, "is_active": True, "end_date": "2026-12-01T12:00:00Z",
     "max_submissions_per_day": 5, "description": "<p>Submit a JSON file.</p>"},
]}


def routes():
    return {
        URL: html("<html ng-app='evalai'><div ui-view></div></html>"),
        f"{SITE}/dist/js/config.min.js": raw(f'angular.constant("EnvironmentConfig", {{"API": "{API}"}});',
                                             content_type="application/javascript"),
        f"{API}challenges/challenge/77/": json_response(json.dumps(CHALLENGE)),
        f"{API}challenges/challenge/77/challenge_phase": json_response(json.dumps(PHASES)),
        "https://huggingface.co/api/datasets/lab/cup-data": json_response(json.dumps(
            {"id": "lab/cup-data", "gated": False, "cardData": {"license": "mit"}})),
        "https://huggingface.co/api/datasets/lab/cup-data/tree/main?recursive=true": json_response(json.dumps([
            {"type": "directory", "path": "Task1"},
            {"type": "file", "path": "Task1/H_train.mat", "size": 2_000_000},
            {"type": "file", "path": "demo_submission.json", "size": 30_000},
        ])),
        "https://api.github.com/repos/lab/cup-baseline": json_response(json.dumps(
            {"default_branch": "main", "license": {"spdx_id": "Apache-2.0"}})),
        "https://api.github.com/repos/lab/cup-baseline/git/trees/main?recursive=1": json_response(json.dumps(
            {"tree": [{"type": "blob", "path": "README.md", "size": 100}, {"type": "blob", "path": "main.py", "size": 50}]})),
        "https://raw.githubusercontent.com/lab/cup-baseline/main/README.md": raw(
            "# Baseline\nWeights: https://huggingface.co/lab/cup-model\n", content_type="text/plain"),
        "https://huggingface.co/api/models/lab/cup-model": json_response(json.dumps(
            {"id": "lab/cup-model", "gated": "manual"})),
        "https://huggingface.co/api/models/lab/cup-model/tree/main?recursive=true": json_response(json.dumps(
            [{"type": "file", "path": "model.pth", "size": 90_000_000}])),
    }


def test_evalai_and_repositories_are_read_from_the_link_alone(workspace):
    project = create_project(workspace, "Radio Cup", URL)
    assert identify_platform(URL).name == "evalai"
    report = read_competition(workspace, project, transport=FakeTransport(routes()))
    fields = competition_card(project)["fields"]
    assert fields["title"]["value"] == "Radio Cup"
    assert fields["competition_type"]["value"] == "file_submission"
    # The deadline and the daily limit come from the phase that accepts submissions now.
    assert fields["deadline"]["value"] == "2026-12-01T12:00:00Z"
    assert fields["rules.daily_submission_limit"]["value"] == 5
    assert fields["metric"]["value"] == "mae"
    assert fields["submission_format"]["value"] == "demo_submission.json"
    assert fields["submission_format"]["status"] == "found"
    # Dataset cited by the guidelines, baseline from the metadata, model cited by the baseline README.
    described = " ".join(report.repositories)
    assert DATASET in described and "github.com/lab/cup-baseline" in described
    assert "huggingface.co/lab/cup-model" in described and "accès sur demande (manual)" in described
    assert not any("presque vide" in note for note in report.notes)


def test_access_secrets_are_masked_and_never_stored(workspace):
    project = create_project(workspace, "Radio Cup", URL)
    report = read_competition(workspace, project, transport=FakeTransport(routes()))
    assert report.secrets == 1 and any("masqué" in note for note in report.notes)
    texts = " ".join(page["text"] for page in stored_pages(project))
    assert "HUNTER2026" not in texts and SECRET_MASK in texts
    with session(project.db_path) as conn:
        dump = " ".join(str(tuple(row)) for table in ("web_pages", "competition_facts", "execution_events")
                        for row in conn.execute(f"SELECT * FROM {table}"))
    assert "HUNTER2026" not in dump


def test_redaction_keeps_ordinary_mentions():
    assert redact_secrets("open the Submit tab to obtain the Dataset Access Password.")[1] == 0
    masked, count = redact_secrets("API key = sk-12345678 and token: abcd1234")
    assert count == 2 and "sk-12345678" not in masked and "abcd1234" not in masked


def test_repository_addresses_are_recognised():
    assert parse_repository("https://huggingface.co/datasets/PPASS/som2026/tree/main").id == "PPASS/som2026"
    assert parse_repository("https://huggingface.co/pku-pcni-lab/WiFo-2").kind == "models"
    assert parse_repository("https://github.com/JiashuoTang/SoM2026-baseline.git").id == "JiashuoTang/SoM2026-baseline"
    assert parse_repository("https://huggingface.co/pku-pcni-lab") is None
    assert parse_repository("https://github.com/orgs/lab") is None


def test_agent_reads_a_page_that_becomes_citable(workspace):
    project = create_project(workspace, "Radio Cup", URL)
    page = "https://forum.example.org/faq"
    report = read_url(workspace, project, page, FakeTransport({page: html("<p>Pretrained models are allowed.</p>")}))
    assert report.pages == [page]
    repository = read_url(workspace, project, DATASET, FakeTransport(routes()))
    assert repository.pages == ["https://huggingface.co/api/datasets/lab/cup-data"]


# --- an organiser site and Zenodo records: acquisition from the link alone ---------------------

DOCS = "https://lyrics.example.org/docs/cup2"
CURRENT, OLD = "https://zenodo.org/api/records/222", "https://zenodo.org/api/records/111"


def tar_of(files: dict[str, str]) -> bytes:
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def zenodo_routes():
    challenge = dict(CHALLENGE, title="CUP2 - The 2nd Lyrics Cup", github_repository="", submission_guidelines="",
                     description=f"<p>See the <a href='{DOCS}/intro'>website</a>, the <a href='{DOCS}/rules'>rules</a>"
                                 f" and the <a href='https://lyrics.example.org/docs/cup1/intro'>first edition</a>.</p>")
    archive = tar_of({"train/metadata.csv": "id,score\na,0.5\n", "train/audio/a.flac": "flac"})

    def record(number, title, files):
        return json_response(json.dumps({"metadata": {"title": title, "access_right": "open", "license": {"id": "cc-by-4.0"}},
                                         "files": [{"key": key, "size": size} for key, size in files]}))

    return {
        URL: html("<div ui-view></div>"),
        f"{SITE}/dist/js/config.min.js": raw(f'{{"API": "{API}"}}', content_type="application/javascript"),
        f"{API}challenges/challenge/77/": json_response(json.dumps(challenge)),
        f"{API}challenges/challenge/77/challenge_phase": json_response(json.dumps(PHASES)),
        f"{DOCS}/intro": html(f"<p>Data: <a href='https://zenodo.org/records/222'>CUP2 data</a>, "
                              f"<a href='{DOCS}/data'>data</a></p>"),
        f"{DOCS}/rules": html("<p>Systems are evaluated using the root mean squared error.</p>"),
        f"{DOCS}/data": html("<p>Older data: <a href='https://zenodo.org/records/111'>CUP1</a></p>"),
        "https://lyrics.example.org/docs/cup1/intro": html("<p>First edition.</p>"),
        CURRENT: record(222, "Second Lyrics Cup (CUP2) Dataset",
                        [("cup2.train.tar.gz", len(archive)), ("cup2.huge.tar.gz", 50 * 1024**3)]),
        OLD: record(111, "Lyrics Cup (CUP1) Dataset", [("cup1.tar.gz", 10)]),
        f"{CURRENT}/files/cup2.train.tar.gz/content": raw(archive, content_type="application/octet-stream"),
    }


def test_data_found_through_the_organiser_site_are_downloaded_and_extracted(workspace):
    from forge.acquisition import acquire_official
    from forge.actions import open_actions

    project = create_project(workspace, "Lyrics", URL)
    transport = FakeTransport(zenodo_routes())
    report = read_competition(workspace, project, transport=transport)
    # This edition's section is read first; the rules cited there give the metric.
    assert report.pages.index(f"{DOCS}/intro") < report.pages.index("https://lyrics.example.org/docs/cup1/intro")
    assert competition_card(project)["fields"]["metric"]["value"] == "rmse"
    result = acquire_official(workspace, project, transport)
    # The record matching the competition title is chosen over the first edition's.
    assert "zenodo.org/records/222" in result.message.split(";")[0]
    names = sorted(path.name for path in result.files)
    assert names == ["a.flac", "metadata.csv"]
    assert not any(path.name.endswith(".tar.gz") for path in result.files[0].parents[2].rglob("*"))
    # The file above the size limit is listed for the user, never silently dropped.
    assert result.status == "needs_user" and "cup2.huge.tar.gz" in result.message
    assert any(action["kind"] == "large_download" for action in open_actions(project))
    from forge.campaign import next_step

    # Files are registered, but the set is incomplete: the campaign never formalises it.
    assert next_step(project)[0] == "acquire"
    downloads = [call for call in transport.calls if call[0] == "DOWNLOAD"]
    acquire_official(workspace, project, transport)
    assert [call for call in transport.calls if call[0] == "DOWNLOAD"] == downloads  # extracted once, kept


def test_explicit_source_and_unsafe_archives(workspace, tmp_path):
    import io
    import tarfile

    from forge.acquisition import acquire_official
    from forge.errors import ForgeError

    project = create_project(workspace, "Lyrics", URL)
    routes = zenodo_routes()
    read_competition(workspace, project, transport=FakeTransport(routes))
    assert acquire_official(workspace, project, FakeTransport(routes), source="https://zenodo.org/records/999").status == "refused"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("../../evil.txt")
        info.size = 4
        bundle.addfile(info, io.BytesIO(b"evil"))
    routes[f"{OLD}/files/cup1.tar.gz/content"] = raw(buffer.getvalue(), content_type="application/octet-stream")
    with pytest.raises(ForgeError, match="Archive refusée"):
        acquire_official(workspace, project, FakeTransport(routes), source="https://zenodo.org/records/111")
    assert not any(path.name == "evil.txt" for path in workspace.data_dir.parent.rglob("evil.txt"))


def test_interrupted_download_is_resumed(workspace):
    """A real HTTP exchange on the loopback interface: the second run only fetches the missing bytes."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from forge.web import Web

    body = bytes(range(256)) * 400
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-"))
            ranges.append(start)
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body) - start))
            self.end_headers()
            self.wfile.write(body[start:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        dest = workspace.data_dir / "resume" / "file.bin"
        dest.parent.mkdir(parents=True)
        dest.with_name("file.bin.part").write_bytes(body[:30_000])  # left by an interrupted run
        result = Web(workspace).download(f"http://127.0.0.1:{server.server_port}/file.bin", dest, "test", 10**6)
    finally:
        server.shutdown()
    assert ranges == [30_000]
    assert result.status == 206 and dest.read_bytes() == body
