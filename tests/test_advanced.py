"""Phases 6 and 7: rejected-hypothesis registry (29), calibration (30), submission plan (31),
deliverables (32, 33), decision stage, test simulator, memory across projects, time series,
dates and text, leak suspicion."""

import json

import numpy as np
import pandas as pd
import pytest

from forge.campaign import run_decision_stage
from forge.datasources import register_data_source
from forge.db import now_iso, session, transaction
from forge.deliverables import declare_submission, finalize, submission_plan
from forge.demo import builtin_card
from forge.errors import ExperimentError, ForgeError
from forge.evaluation import design_folds
from forge.experiments import get_metrics, list_experiments
from forge.guardrails import refuse_all
from forge.intake import paste_text, record_fact
from forge.knowledge import calibration, memory_search, project_context, rank_hypotheses
from forge.projects import create_project
from forge.report import decisions_by_experiment
from forge.runner import builtin_code, current_best, define_protocol, profile_data, run_experiment, submit_experiment
from forge.tabular import prepare_frames, training_mask
from forge.task import TaskSpec


def setup_project(workspace, tmp_path, frame: pd.DataFrame, task_kwargs: dict, name="Projet", test=None):
    directory = tmp_path / name
    directory.mkdir()
    train_path = directory / "train.csv"
    frame.to_csv(train_path, index=False)
    project = create_project(workspace, name, "https://example.org/c")
    register_data_source(project.db_path, train_path, "official")
    test_path = None
    if test is not None:
        test_path = directory / "test.csv"
        test.to_csv(test_path, index=False)
        register_data_source(project.db_path, test_path, "official")
        sample = directory / "sample_submission.csv"
        id_column = task_kwargs["id_column"]
        placeholder = frame[task_kwargs["target"]].iloc[0]
        pd.DataFrame({id_column: test[id_column], task_kwargs["target"]: placeholder}).to_csv(sample, index=False)
        register_data_source(project.db_path, sample, "official")
    task = TaskSpec(train_path=str(train_path), test_path=str(test_path) if test_path else None, **task_kwargs)
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    return project, protocol


def run(workspace, project, protocol, model, role=None, family=None, hypothesis=None, **card):
    payload = builtin_card(protocol.id, model, hypothesis or f"Hypothèse {model}", 0.05, role)
    payload.update(card)
    if family:
        payload["family"] = family
    return run_experiment(workspace, project, submit_experiment(project, payload, builtin_code(model)).id)


def binary_frame(n=240, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"id": range(n), **{f"x{i}": x[:, i] for i in range(4)}, "y": y})


def test_rejected_hypothesis_needs_a_new_reason(workspace, tmp_path):
    project, protocol = setup_project(workspace, tmp_path, binary_frame(), {"problem_type": "binary", "target": "y",
                                                                             "metric": "roc_auc", "id_column": "id"})
    run(workspace, project, protocol, "baseline", role="baseline")
    rejected = run(workspace, project, protocol, "baseline", hypothesis="Une seconde constante suffit.")
    assert rejected.decision == "rejected"
    with pytest.raises(ExperimentError, match="déjà testée et rejetée"):
        submit_experiment(project, builtin_card(protocol.id, "baseline", "Autre formulation", 0.0), builtin_code("baseline"))
    with pytest.raises(ExperimentError, match="new_reason"):
        submit_experiment(project, builtin_card(protocol.id, "linear", "Une seconde constante suffit.", 0.0),
                          builtin_code("linear"))
    card = builtin_card(protocol.id, "baseline", "Autre formulation", 0.0) | {"new_reason": "Données corrigées depuis."}
    assert submit_experiment(project, card, builtin_code("baseline")).id == "E0003"


def test_calibration_corrects_the_ranking(workspace):
    with session(workspace.global_db) as conn, transaction(conn):
        for index in range(3):
            conn.execute(
                "INSERT INTO estimate_calibration (project_id, experiment_id, family, expected_gain, observed_gain,"
                " recorded_at) VALUES ('p', ?, 'features:optimiste', 0.10, 0.01, ?)", (f"E{index}", now_iso()))
    assert calibration(workspace)["features:optimiste"]["ratio_observé_sur_estimé"] == 0.1
    estimate = {"success_probability": 0.5, "cost_minutes": 10, "risk": "low"}
    candidates = [
        {"family": "features:optimiste", "estimate": {**estimate, "expected_gain": 0.10}},
        {"family": "modèle:inconnu", "estimate": {**estimate, "expected_gain": 0.03}},
    ]
    ranked = rank_hypotheses(workspace, candidates)
    assert ranked[0]["family"] == "modèle:inconnu"
    assert ranked[1]["gain_corrigé"] == pytest.approx(0.01)


def test_memory_is_shared_across_projects_with_evidence(workspace, tmp_path):
    kwargs = {"problem_type": "binary", "target": "y", "metric": "roc_auc", "id_column": "id"}
    first, protocol = setup_project(workspace, tmp_path, binary_frame(seed=1), kwargs, "Premier")
    run(workspace, first, protocol, "baseline", role="baseline")
    run(workspace, first, protocol, "linear", family="modèle:linear")
    second, _ = setup_project(workspace, tmp_path, binary_frame(seed=2), kwargs, "Second")
    lessons = memory_search(workspace, project_context(second), exclude_project=second.id)
    assert lessons[0]["famille"] == "modèle:linear" and lessons[0]["même_contexte"]
    assert "1/1" in lessons[0]["leçon"] and "À vérifier" in lessons[0]["leçon"]
    assert memory_search(workspace, project_context(first), exclude_project=first.id) == []


def test_submission_plan_respects_the_daily_limit(workspace, tmp_path):
    kwargs = {"problem_type": "binary", "target": "y", "metric": "roc_auc", "id_column": "id"}
    project, protocol = setup_project(workspace, tmp_path, binary_frame(), kwargs)
    run(workspace, project, protocol, "baseline", role="baseline")
    run(workspace, project, protocol, "linear")
    source = paste_text(project, "You may make 2 submissions per day. You can select 2 final submissions.")
    record_fact(project, "rules.daily_submission_limit", 2, source, "You may make 2 submissions per day.", "user")
    plan = submission_plan(project)
    assert plan["robuste"] == current_best(project, protocol.id) and plan["restantes_aujourdhui"] == 2
    assert plan["ambitieux"] is None  # the superseded baseline is worse: never proposed as "ambitious"
    assert plan["proposées"] == [plan["robuste"]]
    declare_submission(project, "E0002", public_score=0.81)
    declare_submission(project, "E0001")
    with pytest.raises(ForgeError, match="Limite quotidienne"):
        declare_submission(project, "E0002")
    assert submission_plan(project)["restantes_aujourdhui"] == 0
    assert submission_plan(project)["proposées"] == []
    assert "jamais de critère" in plan["recommandation"]


def multiclass_frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.choice(["a", "b", "c"], size=n, p=[0.7, 0.2, 0.1])
    shift = {"a": 0.0, "b": 1.0, "c": 2.0}
    x = np.array([shift[label] for label in y]) + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"id": range(n), "x": x, "z": rng.normal(size=n), "label": y})


def test_decision_stage_and_required_deliverables(workspace, tmp_path):
    frame = multiclass_frame()
    test = frame.drop(columns="label").iloc[:40].assign(id=lambda d: d["id"] + 10_000)
    kwargs = {"problem_type": "multiclass", "target": "label", "metric": "f1_macro", "id_column": "id"}
    project, protocol = setup_project(workspace, tmp_path, frame, kwargs, test=test)
    run(workspace, project, protocol, "baseline", role="baseline")
    run(workspace, project, protocol, "linear")
    stage = run_decision_stage(workspace, project)
    assert stage.status == "completed" and stage.decision in {"accepted", "rejected", "not_significant"}
    assert len(get_metrics(project.db_path, stage.experiment_id)["f1_macro"]["fold"]) == 5

    source = paste_text(project, "Participants must submit a technical report and a notebook with their solution.")
    for name in ("report", "notebook"):
        record_fact(project, "deliverables", name, source,
                    "Participants must submit a technical report and a notebook with their solution.", "user")
    result = finalize(workspace, project, refuse_all)
    assert result.problems == [], result.problems
    deliverables = project.path / "deliverables"
    for name in ("submission.csv", "technical_report.md", "solution.ipynb", "README.md", "solution/make_submission.py"):
        assert (deliverables / name).is_file(), name
    assert "NON" not in (deliverables / "README.md").read_text(encoding="utf-8")
    notebook = json.loads((deliverables / "solution.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and len(notebook["cells"]) == 2
    report = (deliverables / "technical_report.md").read_text(encoding="utf-8")
    assert "Hypothèses rejetées" in report and "Reproduction" in report

    again = finalize(workspace, project, refuse_all)  # replacing validated deliverables needs approval
    assert any("refusée" in p for p in again.problems)


def test_unreliable_validation_uses_the_test_simulator(workspace, tmp_path):
    rng = np.random.default_rng(5)
    n = 200
    y = np.array([1] * 12 + [0] * (n - 12))
    x = y * 2.5 + rng.normal(size=n)
    frame = pd.DataFrame({"id": range(n), "x": x, "noise": rng.normal(size=n), "y": y})
    kwargs = {"problem_type": "binary", "target": "y", "metric": "roc_auc", "id_column": "id"}
    project, protocol = setup_project(workspace, tmp_path, frame, kwargs)
    assert not protocol.reliability["reliable"]
    run(workspace, project, protocol, "baseline", role="baseline")
    outcome = run(workspace, project, protocol, "linear")
    assert "Second régime" in outcome.message
    assert outcome.decision == "accepted"


def test_time_folds_never_train_on_the_future():
    times = pd.Series(pd.date_range("2026-01-01", periods=60, freq="D")).sample(frac=1, random_state=0).astype(str)
    task = TaskSpec("regression", "y", "rmse", "train.csv", time_column="date")
    folds, notes = design_folds(task, pd.Series(range(60)), None, 4, 0, times.reset_index(drop=True))
    folds = np.asarray(folds)
    moments = pd.to_datetime(times.reset_index(drop=True))
    assert "chronologique" in notes[0] and (folds == -1).sum() == 12
    for fold in range(4):
        train = training_mask({"fold_scheme": "time"}, folds, fold)
        assert moments[train].max() < moments[folds == fold].min()


def test_time_series_protocol_runs_end_to_end(workspace, tmp_path):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=200, freq="D")
    y = np.sin(np.arange(200) / 10) * 5 + np.arange(200) * 0.05 + rng.normal(scale=0.3, size=200)
    frame = pd.DataFrame({"id": range(200), "date": dates.astype(str), "lag": np.r_[0, y[:-1]], "y": y})
    kwargs = {"problem_type": "regression", "target": "y", "metric": "rmse", "id_column": "id", "time_column": "date"}
    project, protocol = setup_project(workspace, tmp_path, frame, kwargs)
    assert protocol.scheme == "time"
    run(workspace, project, protocol, "baseline", role="baseline")
    outcome = run(workspace, project, protocol, "linear")
    assert outcome.decision == "accepted"


def test_dates_and_text_become_features():
    frame = pd.DataFrame({
        "when": ["2026-03-01 10:00", "2026-07-15 18:30"] * 10,
        "review": ["the product is great and works well", "terrible quality, it broke after one day"] * 10,
        "n": range(20),
    })
    train, _, features, text = prepare_frames(frame, None, ["when", "review", "n"])
    assert {"when__month", "when__hour", "review__words"} <= set(features) and "when" not in features
    assert text == ["review"] and train["when__month"].tolist()[:2] == [3, 7]


def test_text_model_beats_the_baseline(workspace, tmp_path):
    rng = np.random.default_rng(11)
    good = ["great product works perfectly", "excellent quality very happy", "love it great value"]
    bad = ["terrible broke quickly", "awful quality very disappointed", "waste of money broken"]
    labels = rng.integers(0, 2, size=200)
    reviews = [f"{rng.choice(good if label else bad)} item {i}" for i, label in enumerate(labels)]
    frame = pd.DataFrame({"id": range(200), "review": reviews, "y": labels})
    kwargs = {"problem_type": "binary", "target": "y", "metric": "roc_auc", "id_column": "id"}
    project, protocol = setup_project(workspace, tmp_path, frame, kwargs)
    run(workspace, project, protocol, "baseline", role="baseline")
    outcome = run(workspace, project, protocol, "text_linear")
    assert outcome.decision == "accepted" and outcome.metric_mean > 0.95


def test_near_perfect_score_is_flagged_as_possible_leak(workspace, tmp_path):
    frame = binary_frame()
    frame["leak"] = frame["y"] * 10.0
    kwargs = {"problem_type": "binary", "target": "y", "metric": "roc_auc", "id_column": "id"}
    project, protocol = setup_project(workspace, tmp_path, frame, kwargs)
    run(workspace, project, protocol, "baseline", role="baseline")
    outcome = run(workspace, project, protocol, "random_forest", n_estimators=None)
    assert "fuite" in outcome.message
    decision = decisions_by_experiment(project)[outcome.experiment_id]
    assert "fuite" in decision["rule"]
    assert any(e.card.get("role") == "baseline" for e in list_experiments(project.db_path))
