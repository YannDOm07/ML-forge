"""Competitions hosted on EvalAI and the repositories they cite (Hugging Face, GitHub),
read from the link alone; access secrets printed on pages are never kept."""

import json

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
