"""Reproducible demonstration on a synthetic competition: the full phase-3 chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from forge.datasources import register_data_source
from forge.guardrails import refuse_all
from forge.projects import Project, create_project
from forge.runner import (
    RunOutcome,
    builtin_code,
    current_best,
    define_protocol,
    profile_data,
    run_experiment,
    submit_experiment,
)
from forge.submission import build_submission
from forge.task import TaskSpec
from forge.workspace import Workspace


def make_synthetic_competition(directory: Path, seed: int = 2026, n_train: int = 600, n_test: int = 200) -> dict:
    """Binary task with an imbalanced target, a categorical column and missing values."""
    X, y = make_classification(
        n_samples=n_train + n_test, n_features=8, n_informative=5, n_redundant=1,
        weights=[0.7, 0.3], class_sep=1.2, random_state=seed,
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    frame.insert(0, "id", np.arange(100_000, 100_000 + len(frame)))
    frame["segment"] = pd.cut(frame["f0"], bins=[-np.inf, -0.5, 0.5, np.inf], labels=["bas", "milieu", "haut"])
    frame.loc[np.random.default_rng(seed).random(len(frame)) < 0.05, "f3"] = np.nan
    frame["target"] = y
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.csv" for name in ("train", "test", "sample_submission")}
    frame.iloc[:n_train].to_csv(paths["train"], index=False)
    frame.iloc[n_train:].drop(columns="target").to_csv(paths["test"], index=False)
    pd.DataFrame({"id": frame["id"].iloc[n_train:], "target": 0.5}).to_csv(paths["sample_submission"], index=False)
    return paths


def builtin_card(protocol_id: str, model: str, hypothesis: str, gain: float, role: str | None = None) -> dict:
    card = {
        "question": f"Le modèle « {model} » améliore-t-il la référence ?",
        "hypothesis": hypothesis,
        "method": f"Modèle intégré {model}",
        "protocol_id": protocol_id,
        "justification": [{"kind": "domain_knowledge", "text": "Démonstration sur données synthétiques."}],
        "estimate": {"expected_gain": gain, "success_probability": 0.5, "cost_minutes": 1, "risk": "low"},
    }
    if role:
        card["role"] = role
    return card


def run_demo(workspace: Workspace, directory: Path) -> tuple[Project, list[RunOutcome], Path]:
    paths = make_synthetic_competition(directory)
    project = create_project(workspace, "Démo synthétique")
    for name in ("train", "test"):
        register_data_source(project.db_path, paths[name], "official")
    task = TaskSpec("binary", "target", "roc_auc", str(paths["train"]), str(paths["test"]), id_column="id")
    protocol = define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    plan = [
        ("baseline", "Une prédiction constante donne le score plancher.", 0.0, "baseline"),
        ("linear", "Une frontière linéaire capte l'essentiel du signal.", 0.3, None),
        ("random_forest", "Des arbres captent des interactions non linéaires.", 0.05, None),
    ]
    outcomes = []
    for model, hypothesis, gain, role in plan:
        experiment = submit_experiment(project, builtin_card(protocol.id, model, hypothesis, gain, role), builtin_code(model))
        outcomes.append(run_experiment(workspace, project, experiment.id))
    best = current_best(project, protocol.id)
    submission = build_submission(project, best, paths["sample_submission"], "proba", refuse_all, decided_by="demo")
    return project, outcomes, submission
