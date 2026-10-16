"""Tests 13 (reproducibility), 14 (comparison needs the same protocol), 16 (prerequisites),
19 (verified submission), and the full phase-3 chain."""

import pandas as pd
import pytest

from forge.datasources import register_data_source
from forge.demo import builtin_card, make_synthetic_competition, run_demo
from forge.errors import ApprovalRequired, ExperimentError
from forge.experiments import get_experiment, get_metrics
from forge.guardrails import refuse_all
from forge.projects import create_project
from forge.runner import (
    builtin_code,
    compare,
    current_best,
    define_protocol,
    profile_data,
    run_experiment,
    submit_experiment,
)
from forge.submission import SubmissionError, build_submission, verify_submission
from forge.task import TaskSpec


@pytest.fixture
def setup(workspace, tmp_path):
    paths = make_synthetic_competition(tmp_path / "comp", n_train=200, n_test=50)
    project = create_project(workspace, "Runner")
    for name in ("train", "test"):
        register_data_source(project.db_path, paths[name], "official")
    task = TaskSpec("binary", "target", "roc_auc", str(paths["train"]), str(paths["test"]), id_column="id")
    return project, task, paths


def run_builtin(workspace, project, protocol, model, role=None, params=None):
    card = builtin_card(protocol.id, model, f"Hypothèse {model}", 0.05, role)
    experiment = submit_experiment(project, card, builtin_code(model, params))
    return run_experiment(workspace, project, experiment.id)


def test_full_chain_on_synthetic_competition(workspace, tmp_path):
    project, outcomes, submission = run_demo(workspace, tmp_path / "demo")
    assert [o.decision for o in outcomes] == ["baseline", "accepted", "accepted"]
    assert outcomes[0].metric_mean == pytest.approx(0.5)
    assert outcomes[2].metric_mean > 0.9
    assert verify_submission(submission, tmp_path / "demo" / "sample_submission.csv") == []
    report = (project.path / "report.md").read_text(encoding="utf-8")
    assert "Meilleur candidat : E0003" in report


def test_run_requires_profile_then_baseline(workspace, setup):
    project, task, _ = setup
    protocol = define_protocol(workspace, project, task)
    baseline = submit_experiment(project, builtin_card(protocol.id, "baseline", "h", 0.0, "baseline"), builtin_code("baseline"))
    with pytest.raises(ExperimentError, match="Profil des données absent"):
        run_experiment(workspace, project, baseline.id)

    profile_data(workspace, project, task)
    linear = submit_experiment(project, builtin_card(protocol.id, "linear", "h", 0.1), builtin_code("linear"))
    with pytest.raises(ExperimentError, match="baseline"):
        run_experiment(workspace, project, linear.id)
    assert get_experiment(project.db_path, linear.id).status == "registered"


def test_unknown_protocol_is_refused(setup):
    project, _, _ = setup
    with pytest.raises(ExperimentError, match="Protocole de validation inconnu"):
        submit_experiment(project, builtin_card("P000", "linear", "h", 0.1), builtin_code("linear"))


def test_modified_data_blocks_the_run(workspace, setup):
    project, task, paths = setup
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    experiment = submit_experiment(project, builtin_card(protocol.id, "baseline", "h", 0.0, "baseline"), builtin_code("baseline"))
    with paths["train"].open("a", encoding="utf-8") as handle:
        handle.write("999999,0,0,0,0,0,0,0,0,milieu,1\n")
    with pytest.raises(ExperimentError, match="modifiées"):
        run_experiment(workspace, project, experiment.id)


def test_same_code_and_seed_reproduce_the_same_scores(workspace, setup):
    project, task, _ = setup
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    run_builtin(workspace, project, protocol, "baseline", "baseline")
    first = run_builtin(workspace, project, protocol, "random_forest", params={"n_estimators": 40})
    second = run_builtin(workspace, project, protocol, "random_forest", params={"n_estimators": 40})
    scores = [get_metrics(project.db_path, o.experiment_id)["roc_auc"]["fold"] for o in (first, second)]
    assert scores[0] == scores[1]
    assert first.decision == "accepted"
    assert second.decision == "rejected"  # identical scores: Δ = 0 is not an improvement
    assert not (project.path / "experiments" / second.experiment_id).exists()
    assert (project.path / "experiments" / first.experiment_id / "test_pred.csv").is_file()


def test_invalid_outputs_fail_cleanly(workspace, setup):
    project, task, _ = setup
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    run_builtin(workspace, project, protocol, "baseline", "baseline")
    cheat = "import pandas as pd\npd.DataFrame({'score': [0.99]}).to_csv('oof.csv', index=False)\n"
    experiment = submit_experiment(project, builtin_card(protocol.id, "custom", "Rapporter son propre score", 0.5), cheat)
    outcome = run_experiment(workspace, project, experiment.id)
    assert outcome.status == "failed"
    assert "Colonnes" in outcome.message
    assert get_metrics(project.db_path, experiment.id) == {}
    assert not (project.path / "experiments" / experiment.id).exists()


def test_generated_script_following_the_contract_is_scored(workspace, setup):
    project, task, _ = setup
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    run_builtin(workspace, project, protocol, "baseline", "baseline")
    code = """
import pandas as pd
from forge.tabular import load_context, load_data, prediction_columns, save_predictions
context = load_context()
train, _ = load_data(context)
score = 1 / (1 + (-train["f1"]).rpow(2.718281828))
save_predictions(context, pd.DataFrame({"p:0": 1 - score, "p:1": score})[prediction_columns(context)])
"""
    experiment = submit_experiment(project, builtin_card(protocol.id, "custom", "f1 seule sépare les classes", 0.1), code)
    outcome = run_experiment(workspace, project, experiment.id)
    assert outcome.status == "completed"
    assert outcome.decision in {"accepted", "rejected", "not_significant"}
    assert len(get_metrics(project.db_path, experiment.id)["roc_auc"]["fold"]) == 5


def test_comparison_requires_the_same_protocol(workspace, setup):
    project, task, _ = setup
    first_protocol = define_protocol(workspace, project, task)
    other_task = TaskSpec("binary", "target", "accuracy", task.train_path, task.test_path, id_column="id")
    second_protocol = define_protocol(workspace, project, other_task)
    profile_data(workspace, project, task)
    a = run_builtin(workspace, project, first_protocol, "baseline", "baseline")
    b = run_builtin(workspace, project, second_protocol, "baseline", "baseline")
    with pytest.raises(ExperimentError, match="pas le même protocole"):
        compare(project, a.experiment_id, b.experiment_id)
    c = run_builtin(workspace, project, first_protocol, "linear")
    assert "accepted" in compare(project, a.experiment_id, c.experiment_id)["verdict"]


def test_submission_is_verified_and_replacement_needs_approval(workspace, setup):
    project, task, paths = setup
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    run_builtin(workspace, project, protocol, "baseline", "baseline")
    run_builtin(workspace, project, protocol, "linear")
    best = current_best(project, protocol.id)

    submission = build_submission(project, best, paths["sample_submission"], "proba", refuse_all, "test")
    assert verify_submission(submission, paths["sample_submission"]) == []
    before = submission.read_bytes()
    with pytest.raises(ApprovalRequired):
        build_submission(project, best, paths["sample_submission"], "label", refuse_all, "test")
    assert submission.read_bytes() == before
    build_submission(project, best, paths["sample_submission"], "label", lambda request: True, "test")
    assert set(pd.read_csv(submission)["target"]) <= {0, 1}


def test_submission_needs_kept_predictions(workspace, setup):
    project, task, paths = setup
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    run_builtin(workspace, project, protocol, "baseline", "baseline")
    rejected = run_builtin(workspace, project, protocol, "baseline")
    with pytest.raises(SubmissionError, match="Pas de prédictions"):
        build_submission(project, rejected.experiment_id, paths["sample_submission"], "proba", refuse_all, "test")


def test_malformed_submissions_are_rejected(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"id": [1, 2, 3], "target": [0.5, 0.5, 0.5]}).to_csv(sample, index=False)
    cases = {
        "colonnes": pd.DataFrame({"id": [1, 2, 3], "pred": [0, 1, 0]}),
        "lignes": pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]}),
        "double": pd.DataFrame({"id": [1, 1, 3], "target": [0.1, 0.2, 0.3]}),
        "manquante": pd.DataFrame({"id": [1, 2, 3], "target": [0.1, None, 0.3]}),
        "numériques": pd.DataFrame({"id": [1, 2, 3], "target": ["a", "b", "c"]}),
    }
    for expected, frame in cases.items():
        path = tmp_path / f"{expected}.csv"
        frame.to_csv(path, index=False)
        problems = " ".join(verify_submission(path, sample)).lower()
        assert expected in problems, (expected, problems)
