"""Build the submission file from a retained experiment and verify it against the expected format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forge.errors import ForgeError
from forge.experiments import get_experiment
from forge.guardrails import Action, ApprovalRequest, Approver, project_decision_recorder, require_approval
from forge.projects import Project
from forge.runner import get_protocol
from forge.tabular import to_submission

DELIVERABLES_DIR = "deliverables"


class SubmissionError(ForgeError):
    """The submission cannot be built or does not match the expected format."""


def verify_submission(submission_path: Path, sample_path: Path) -> list[str]:
    """Return every format problem; an empty list means the file matches the sample."""
    try:
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return [f"Lecture impossible : {exc}"]
    if list(submission.columns) != list(sample.columns):
        return [f"Colonnes {list(submission.columns)} ; attendues {list(sample.columns)} (même ordre)."]
    problems = []
    id_column = sample.columns[0]
    if len(submission) != len(sample):
        problems.append(f"{len(submission)} lignes ; attendues {len(sample)}.")
    duplicated = int(submission[id_column].duplicated().sum())
    if duplicated:
        problems.append(f"{duplicated} identifiant(s) en double dans {id_column}.")
    expected_ids = set(sample[id_column].astype(str))
    given_ids = set(submission[id_column].astype(str))
    if expected_ids != given_ids:
        problems.append(
            f"Identifiants : {len(expected_ids - given_ids)} manquant(s), {len(given_ids - expected_ids)} inattendu(s)."
        )
    missing = int(submission.isna().sum().sum())
    if missing:
        problems.append(f"{missing} valeur(s) manquante(s).")
    for column in sample.columns[1:]:
        if pd.api.types.is_numeric_dtype(sample[column]) and not pd.api.types.is_numeric_dtype(submission[column]):
            problems.append(f"{column} : valeurs numériques attendues.")
    return problems


def build_submission(
    project: Project, experiment_id: str, sample_path: Path, kind: str, approver: Approver, decided_by: str
) -> Path:
    """Write deliverables/submission.csv from an experiment's test predictions, then verify it.

    kind: 'label' (predicted class), 'proba' (probabilities) or 'value' (regression)."""
    experiment = get_experiment(project.db_path, experiment_id)
    protocol = get_protocol(project, experiment.protocol_id)
    predictions_path = project.path / "experiments" / experiment_id / "test_pred.csv"
    if not predictions_path.is_file():
        raise SubmissionError(
            f"Pas de prédictions de test conservées pour {experiment_id} (seules la baseline et les expériences "
            "acceptées les conservent)."
        )
    sample = pd.read_csv(sample_path)
    predictions = pd.read_csv(predictions_path)
    try:
        submission = to_submission(sample, predictions, kind, protocol.classes, protocol.task.positive_label)
    except ValueError as exc:
        raise SubmissionError(str(exc)) from exc
    target = project.path / DELIVERABLES_DIR / "submission.csv"
    if target.exists():
        request = ApprovalRequest(Action.REPLACE_FINAL_DELIVERABLE, str(target), {"nouvelle source": experiment_id})
        require_approval(request, approver, project_decision_recorder(project.db_path, decided_by))
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(".tmp")
    submission.to_csv(temporary, index=False)
    problems = verify_submission(temporary, sample_path)
    if problems:
        temporary.unlink()
        raise SubmissionError("Soumission non conforme :\n- " + "\n- ".join(problems))
    temporary.replace(target)
    return target

