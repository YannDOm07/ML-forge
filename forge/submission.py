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



# --- submissions described by the rules (no sample file) ---------------------------------------

SPEC_KEYS = {"files", "columns", "scale", "clip", "kind"}


def parse_spec(text: str) -> dict:
    """Validate a submission specification recorded on the card (field `submission_spec`).

    {"files": {"<set>": "<file name>"}, "columns": {"id": "<id header>", "prediction": "<header>"},
     "scale": 100, "clip": [0, 100], "kind": "value" | "label" | "proba"} — scale, clip, kind optional.
    `{TEAM_ID}`-like placeholders in file names are kept for the user to fill in."""
    import json

    try:
        spec = json.loads(text) if isinstance(text, str) else dict(text)
    except ValueError as exc:
        raise SubmissionError(f"Spécification de soumission illisible (JSON attendu) : {exc}") from exc
    if not isinstance(spec, dict) or not {"files", "columns"} <= set(spec) or set(spec) - SPEC_KEYS:
        raise SubmissionError(f"Spécification de soumission : clés attendues {sorted(SPEC_KEYS)} (files et columns requis).")
    if not isinstance(spec["files"], dict) or not spec["files"]:
        raise SubmissionError("Spécification de soumission : « files » doit associer un jeu à un nom de fichier.")
    if not {"id", "prediction"} <= set(spec["columns"]):
        raise SubmissionError("Spécification de soumission : « columns » doit nommer « id » et « prediction ».")
    if "clip" in spec and (len(spec["clip"]) != 2 or spec["clip"][0] >= spec["clip"][1]):
        raise SubmissionError("Spécification de soumission : « clip » doit être [minimum, maximum].")
    return spec


def build_from_spec(project: Project, experiment_id: str, spec: dict, approver: Approver, decided_by: str) -> list[Path]:
    """One verified file per set named by the specification, from the experiment's predictions."""
    import re

    experiment = get_experiment(project.db_path, experiment_id)
    protocol = get_protocol(project, experiment.protocol_id)
    task = protocol.task
    files = task.prediction_files()
    folder = project.path / "experiments" / experiment_id
    unknown = set(spec["files"]) - set(files)
    if unknown:
        raise SubmissionError(f"Jeux inconnus de la tâche : {sorted(unknown)} (jeux de la tâche : {sorted(files)}).")
    built = []
    for set_name, file_name in spec["files"].items():
        source = folder / files[set_name]
        if not source.is_file():
            raise SubmissionError(f"Prédictions absentes pour le jeu {set_name} ({source.name}) de {experiment_id}.")
        predictions = pd.read_csv(source)
        submission = _from_predictions(predictions, task, protocol.classes, spec)
        expected = set(_set_ids(task, set_name))
        problems = _check(submission, spec, expected)
        if problems:
            raise SubmissionError(f"Soumission {set_name} non conforme :\n- " + "\n- ".join(problems))
        safe = re.sub(r'[<>:"/\|?*]', "_", file_name.replace("{", "").replace("}", ""))
        target = project.path / DELIVERABLES_DIR / "submission" / safe
        if target.exists():
            request = ApprovalRequest(Action.REPLACE_FINAL_DELIVERABLE, str(target), {"nouvelle source": experiment_id})
            require_approval(request, approver, project_decision_recorder(project.db_path, decided_by))
        target.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(target, index=False)
        built.append(target)
    return built


def _from_predictions(predictions: pd.DataFrame, task, classes, spec: dict) -> pd.DataFrame:
    from forge.tabular import labels_from_probabilities

    id_column = task.id_column
    if id_column not in predictions.columns:
        raise SubmissionError(f"Les prédictions n'ont pas la colonne d'identifiant {id_column}.")
    kind = spec.get("kind") or ("value" if not task.is_classification else "label")
    if kind == "value":
        values = predictions["prediction"].astype(float) * float(spec.get("scale", 1))
        if "clip" in spec:
            values = values.clip(*spec["clip"])
    elif kind == "label":
        values = pd.Series(labels_from_probabilities(predictions))
    else:
        positive = task.positive_label or sorted(classes)[-1]
        values = predictions["p:" + str(positive)] * float(spec.get("scale", 1))
    return pd.DataFrame({spec["columns"]["id"]: predictions[id_column].to_numpy(),
                         spec["columns"]["prediction"]: values.to_numpy()})


def _set_ids(task, set_name: str) -> list[str]:
    from forge.task import read_table

    return read_table(task.sets()[set_name], usecols=[task.id_column])[task.id_column].astype(str).tolist()


def _check(submission: pd.DataFrame, spec: dict, expected: set[str]) -> list[str]:
    id_name, value_name = spec["columns"]["id"], spec["columns"]["prediction"]
    ids = submission[id_name].astype(str)
    problems = []
    if int(ids.duplicated().sum()):
        problems.append(f"{int(ids.duplicated().sum())} identifiant(s) en double.")
    if set(ids) != expected:
        problems.append(f"Identifiants : {len(expected - set(ids))} manquant(s), {len(set(ids) - expected)} inattendu(s).")
    if int(submission[value_name].isna().sum()):
        problems.append(f"{int(submission[value_name].isna().sum())} valeur(s) manquante(s).")
    if "clip" in spec and pd.api.types.is_numeric_dtype(submission[value_name]):
        low, high = spec["clip"]
        outside = int(((submission[value_name] < low) | (submission[value_name] > high)).sum())
        if outside:
            problems.append(f"{outside} valeur(s) hors de [{low}, {high}].")
    return problems
