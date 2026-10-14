"""Phase 5: link to deliverables, test 27 (resume from campaign_next alone), test 28 (LLM pause and resume)."""

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fake_web import FakeTransport, kaggle_routes
from forge.campaign import (
    BuiltinDriver,
    OpencodeDriver,
    active_campaign,
    campaign_summary,
    prepare,
    render_summary,
    request_stop,
    run_campaign,
    start_campaign,
    stop_reason,
)
from forge.config import load_config
from forge.demo import make_synthetic_competition
from forge.experiments import list_experiments
from forge.projects import create_project
from forge.submission import verify_submission
from forge.workspace import open_workspace

URL = "https://www.kaggle.com/competitions/flows-cup"
SLUG = "flows-cup"


@pytest.fixture
def kaggle(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "alice")
    monkeypatch.setenv("KAGGLE_KEY", "secret-key-123")
    competition = make_synthetic_competition(tmp_path / "source", n_train=200, n_test=50)
    return competition, FakeTransport(kaggle_routes(competition, URL, SLUG, "Area Under ROC Curve"))


def no_sleep(seconds):
    return None


def test_from_link_to_verified_submission_without_llm(workspace, kaggle):
    """Version 1 milestone, with the reference plan: a link in, verified deliverables out."""
    competition, transport = kaggle
    project = create_project(workspace, "Flows", URL)
    campaign = run_campaign(workspace, project, BuiltinDriver(), transport=transport, sleep=no_sleep)

    assert campaign["status"] == "finished", campaign["reason"]
    assert "Plan de référence épuisé" in campaign["reason"]
    deliverables = project.path / "deliverables"
    assert verify_submission(deliverables / "submission.csv", competition["sample_submission"]) == []
    assert (deliverables / "solution" / "make_submission.py").is_file()
    assert "submission" in (deliverables / "README.md").read_text(encoding="utf-8")
    roles = [e.card.get("role") for e in list_experiments(project.db_path)]
    assert roles.count("baseline") == 1 and roles.count("stability") == 2
    assert len(list(workspace.backups_dir.glob(f"{project.id}-*.zip"))) == 1


def test_cycle_resumes_from_the_summary_alone(workspace, kaggle, config_path):
    _, transport = kaggle
    project = create_project(workspace, "Flows", URL)
    assert prepare(workspace, project, transport).step == "experiment"
    BuiltinDriver().run_cycle(workspace, project)

    restarted = open_workspace(load_config(config_path))  # a new process, no conversation
    summary = campaign_summary(restarted, project)
    assert summary["étape"] == "experiment"
    assert summary["meilleur"]["id"] in {"E0001", "E0002"}
    assert [line["id"] for line in summary["récentes"]] == ["E0001", "E0002"]
    assert summary["fiche"]["metric"] == "roc_auc [found]"
    text = render_summary(summary, 3000 * 4)
    assert len(text) <= 3000 * 4
    assert len(render_summary(summary, 500)) <= 500


def test_missing_token_waits_for_the_user(workspace, tmp_path):
    project = create_project(workspace, "Sans jeton", URL)
    campaign = run_campaign(workspace, project, BuiltinDriver(), transport=FakeTransport({}), sleep=no_sleep)
    assert campaign["status"] == "waiting_user"
    assert "kaggle.json" in campaign["reason"]


def fake_llm(tmp_path, body: str):
    script = tmp_path / "fake_opencode.py"
    script.write_text(body, encoding="utf-8")
    return OpencodeDriver(command=[sys.executable, str(script)], timeout_s=60)


def test_opencode_cycle_is_noninteractive(workspace, monkeypatch):
    project = create_project(workspace, "Non interactif", URL)
    captured = {}

    def fake_run(arguments, **kwargs):
        captured["arguments"] = arguments
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("forge.campaign.subprocess.run", fake_run)

    result = OpencodeDriver().run_cycle(workspace, project)

    assert result.kind == "ok"
    assert captured["arguments"][1:3] == ["run", "--auto"]


def test_rate_limited_llm_pauses_then_resumes(workspace, kaggle, tmp_path):
    _, transport = kaggle
    project = create_project(workspace, "Limité", URL)
    sleeps = []
    driver = fake_llm(tmp_path, "import sys\nprint('Error: 429 Too Many Requests (rate limit)')\nsys.exit(1)\n")
    campaign = run_campaign(workspace, project, driver, transport=transport, sleep=sleeps.append)
    assert campaign["status"] == "paused"
    assert "rate_limited" in campaign["reason"] and "campaign resume" in campaign["reason"]
    assert len(sleeps) == 2  # two waits, then pause after the third failure
    assert len(list_experiments(project.db_path)) == 1  # only the baseline

    resumed = run_campaign(workspace, project, BuiltinDriver(), transport=transport, sleep=no_sleep)
    assert resumed["id"] == campaign["id"] and resumed["status"] == "finished"


def test_unavailable_llm_pauses_immediately(workspace, kaggle):
    _, transport = kaggle
    project = create_project(workspace, "Absent", URL)
    driver = OpencodeDriver(command=["opencode-introuvable-xyz"])
    campaign = run_campaign(workspace, project, driver, transport=transport, sleep=no_sleep)
    assert campaign["status"] == "paused" and "unavailable" in campaign["reason"]


def test_llm_cycles_without_experiments_end_the_campaign(workspace, kaggle, tmp_path):
    _, transport = kaggle
    project = create_project(workspace, "Inactif", URL)
    driver = fake_llm(tmp_path, "print('{\"type\": \"text\", \"text\": \"rien\"}')\n")
    campaign = run_campaign(workspace, project, driver, transport=transport, sleep=no_sleep)
    assert campaign["status"] == "finished" and "sans nouvelle expérience" in campaign["reason"]


def test_stop_conditions(workspace, kaggle):
    _, transport = kaggle
    project = create_project(workspace, "Arrêts", URL)
    prepare(workspace, project, transport)
    campaign = start_campaign(workspace, project, "interactive", "builtin")
    assert stop_reason(workspace, project, campaign) is None
    later = datetime.now(timezone.utc) + timedelta(hours=3)
    assert "Budget de temps" in stop_reason(workspace, project, campaign, later)
    BuiltinDriver().run_cycle(workspace, project)
    assert "Nombre maximal" in stop_reason(workspace, project, {**campaign, "max_experiments": 1})
    request_stop(project)
    assert "Arrêt demandé" in stop_reason(workspace, project, campaign)
    assert active_campaign(project) is None


def test_reference_plan_is_reproducible_from_the_solution_package(workspace, kaggle, tmp_path):
    from forge.sandbox import SandboxLimits, run_in_sandbox

    competition, transport = kaggle
    project = create_project(workspace, "Reproduction", URL)
    run_campaign(workspace, project, BuiltinDriver(), transport=transport, sleep=no_sleep)
    work = tmp_path / "replay"
    import shutil

    shutil.copytree(project.path / "deliverables" / "solution", work)
    data = competition["train"].parent
    (work / "replay.py").write_text(
        "import runpy, sys\n"
        f"sys.argv = ['make_submission.py', '--data', r'{data}', '--out', 'submission.csv']\n"
        "runpy.run_path('make_submission.py', run_name='__main__')\n",
        encoding="utf-8",
    )
    result = run_in_sandbox(work / "replay.py", work, [data], SandboxLimits(300, 2048))
    assert result.ok, result.stderr
    replayed = pd.read_csv(work / "submission.csv")
    original = pd.read_csv(project.path / "deliverables" / "submission.csv")
    pd.testing.assert_frame_equal(replayed, original)
