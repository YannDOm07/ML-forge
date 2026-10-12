"""Experiment-side helpers, executed INSIDE the sandbox (and copied into exported solutions).

Contract for every experiment script (built-in or generated):
- read `context.json` from the working directory (task, folds, fold scheme, seed, classes, paths);
- write `oof.csv`: one row per training row, in order, with columns 'p:<class>' (classification)
  or 'prediction' (regression), predicted by a model that never trained on that row
  (use `training_mask`: with a time scheme, only the past is used);
- optionally write `test_pred.csv`: the id column then the same prediction columns;
- optionally write `info.json` (model, parameters, features).
ML Forge computes the metrics itself from `oof.csv`; scripts never report their own score.
This module must stay importable on its own: it only imports `forge` inside lab-only functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from forge.task import PROBA_PREFIX

CONTEXT_FILE = "context.json"
DATE_PARTS = ("year", "month", "day", "dayofweek", "hour")
WEIGHT_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


# --- contract helpers ----------------------------------------------------------------

def load_context() -> dict:
    return json.loads(Path(CONTEXT_FILE).read_text(encoding="utf-8"))


def load_data(context: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    task = context["task"]
    train = pd.read_csv(task["train_path"], low_memory=False)
    test = pd.read_csv(task["test_path"], low_memory=False) if task.get("test_path") else None
    return train, test


def feature_names(context: dict, train: pd.DataFrame) -> list[str]:
    task = context["task"]
    excluded = {task["target"], task.get("id_column"), task.get("group_column")}
    return [column for column in train.columns if column not in excluded]


def prediction_columns(context: dict) -> list[str]:
    classes = context.get("classes")
    return [PROBA_PREFIX + c for c in classes] if classes else ["prediction"]


def validation_folds(folds: np.ndarray) -> list[int]:
    return sorted(set(folds.tolist()) - {-1})


def training_mask(context: dict, folds: np.ndarray, fold: int) -> np.ndarray:
    """Rows a model may train on when predicting `fold`: never that fold, never the future."""
    if context.get("fold_scheme") == "time":
        return folds < fold
    return folds != fold


def save_predictions(context: dict, oof: pd.DataFrame, test: pd.DataFrame | None = None, info: dict | None = None) -> None:
    oof.to_csv(context["outputs"]["oof"], index=False)
    if test is not None:
        test.to_csv(context["outputs"]["test"], index=False)
    if info is not None:
        Path(context["outputs"]["info"]).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def class_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith(PROBA_PREFIX)]


def labels_from_probabilities(frame: pd.DataFrame) -> np.ndarray:
    columns = class_columns(frame)
    classes = np.array([column[len(PROBA_PREFIX):] for column in columns])
    return classes[frame[columns].to_numpy().argmax(axis=1)]


# --- column kinds and derived features ------------------------------------------------------

def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    sample = series.dropna().astype(str).head(200)
    if len(sample) and pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.9:
        return "datetime"
    if len(sample) and sample.str.len().mean() > 30 and sample.str.contains(" ").mean() > 0.5:
        return "text"
    return "categorical"


def prepare_frames(train: pd.DataFrame, test: pd.DataFrame | None, features: list[str]):
    """Dates become numeric parts; text gets length features and is kept for TF-IDF models."""
    kinds = {column: column_kind(train[column]) for column in features}
    frames = [train.copy()] + ([test.copy()] if test is not None else [])
    derived = []
    for column, kind in kinds.items():
        if kind == "datetime":
            for frame in frames:
                values = pd.to_datetime(frame[column], errors="coerce", format="mixed")
                for part in DATE_PARTS:
                    frame[f"{column}__{part}"] = getattr(values.dt, part)
                frame.drop(columns=column, inplace=True)
            derived += [f"{column}__{part}" for part in DATE_PARTS]
        elif kind == "text":
            for frame in frames:
                frame[column] = frame[column].fillna("").astype(str)
                frame[f"{column}__chars"] = frame[column].str.len()
                frame[f"{column}__words"] = frame[column].str.split().str.len()
            derived += [f"{column}__chars", f"{column}__words"]
    kept = [c for c in features if kinds[c] != "datetime"] + derived
    text = [c for c in features if kinds[c] == "text"]
    return frames[0], (frames[1] if test is not None else None), kept, text


# --- built-in models ----------------------------------------------------------------------

def _tree_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), numeric), ("cat", encoder, categorical)],
        remainder="drop",
    )


def _linear_preprocessor(numeric: list[str], categorical: list[str], text: list[str]) -> ColumnTransformer:
    parts = [
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
        ), categorical),
    ]
    parts += [(f"txt_{column}", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2), column)
              for column in text]
    return ColumnTransformer([part for part in parts if part[2]], remainder="drop")


def _booster(name: str, classification: bool, params: dict, seed: int):
    if name == "lightgbm":
        import lightgbm

        model = lightgbm.LGBMClassifier if classification else lightgbm.LGBMRegressor
        return model(random_state=seed, verbose=-1, **params)
    if name == "xgboost":
        import xgboost

        model = xgboost.XGBClassifier if classification else xgboost.XGBRegressor
        return model(random_state=seed, **params)
    import catboost

    model = catboost.CatBoostClassifier if classification else catboost.CatBoostRegressor
    return model(random_seed=seed, verbose=0, allow_writing_files=False, **params)


def _estimator(name: str, classification: bool, params: dict, seed: int):
    forest = {"n_estimators": 300, **params}
    factories = {
        "baseline": (lambda: DummyClassifier(strategy="prior"), lambda: DummyRegressor(strategy="mean")),
        "linear": (lambda: LogisticRegression(max_iter=2000, random_state=seed, **params),
                   lambda: Ridge(random_state=seed, **params)),
        "random_forest": (lambda: RandomForestClassifier(random_state=seed, n_jobs=-1, **forest),
                          lambda: RandomForestRegressor(random_state=seed, n_jobs=-1, **forest)),
        "extra_trees": (lambda: ExtraTreesClassifier(random_state=seed, n_jobs=-1, **forest),
                        lambda: ExtraTreesRegressor(random_state=seed, n_jobs=-1, **forest)),
        "hist_gb": (lambda: HistGradientBoostingClassifier(random_state=seed, **params),
                    lambda: HistGradientBoostingRegressor(random_state=seed, **params)),
    }
    factories["text_linear"] = factories["linear"]
    if name in {"lightgbm", "xgboost", "catboost"}:
        return _booster(name, classification, params, seed)
    if name not in factories:
        raise ValueError(f"Modèle intégré inconnu : {name} (disponibles : {', '.join(BUILTIN_MODELS)})")
    return factories[name][0 if classification else 1]()


BUILTIN_MODELS = ("baseline", "linear", "text_linear", "random_forest", "extra_trees", "hist_gb",
                  "lightgbm", "xgboost", "catboost")


def build_pipeline(name: str, context: dict, train: pd.DataFrame, features: list[str], text: list[str],
                   params: dict) -> Pipeline:
    plain = [c for c in features if c not in text]
    numeric = [c for c in plain if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in plain if c not in numeric]
    if name in {"linear", "text_linear"}:
        preprocessor = _linear_preprocessor(numeric, categorical, text if name == "text_linear" else [])
    else:
        preprocessor = _tree_preprocessor(numeric, categorical)
    classification = bool(context.get("classes"))
    return Pipeline([("prep", preprocessor), ("model", _estimator(name, classification, params, context["seed"]))])


def _predict(pipeline: Pipeline, frame: pd.DataFrame, classes: list[str] | None) -> np.ndarray:
    if not classes:
        return np.asarray(pipeline.predict(frame)).reshape(-1, 1)
    probabilities = pipeline.predict_proba(frame)
    aligned = np.zeros((len(frame), len(classes)))
    for index, label in enumerate(str(c) for c in pipeline.classes_):
        aligned[:, classes.index(label)] = probabilities[:, index]
    return aligned


def run_builtin(name: str, params: dict | None = None) -> None:
    """Train a built-in model fold by fold; test predictions average the fold models."""
    params = params or {}
    context = load_context()
    train, test = load_data(context)
    train, test, features, text = prepare_frames(train, test, feature_names(context, train))
    classes = context.get("classes")
    target = train[context["task"]["target"]]
    target = target.astype(str) if classes else target
    folds = np.asarray(context["folds"])
    columns = prediction_columns(context)
    oof = np.zeros((len(train), len(columns)))
    test_sum = None if test is None else np.zeros((len(test), len(columns)))
    for fold in validation_folds(folds):
        fit_rows, valid_rows = training_mask(context, folds, fold), folds == fold
        pipeline = build_pipeline(name, context, train, features, text, params)
        pipeline.fit(train.loc[fit_rows, features], target[fit_rows])
        oof[valid_rows] = _predict(pipeline, train.loc[valid_rows, features], classes)
        if test is not None:
            test_sum += _predict(pipeline, test[features], classes)
    save_predictions(
        context,
        pd.DataFrame(oof, columns=columns),
        _test_frame(context, test, test_sum / len(validation_folds(folds)) if test is not None else None, columns),
        {"model": name, "params": params, "features": features},
    )


def _test_frame(context: dict, test: pd.DataFrame | None, values, columns: list[str]) -> pd.DataFrame | None:
    if test is None:
        return None
    frame = pd.DataFrame(values, columns=columns)
    id_column = context["task"].get("id_column")
    if id_column:
        frame.insert(0, id_column, test[id_column].to_numpy())
    return frame


# --- decision stage ---------------------------------------------------------------------------

def apply_class_weights(probabilities: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    weighted = probabilities.copy()
    for column in class_columns(weighted):
        weighted[column] = weighted[column] * weights[column[len(PROBA_PREFIX):]]
    total = weighted[class_columns(weighted)].sum(axis=1).replace(0, 1)
    for column in class_columns(weighted):
        weighted[column] = weighted[column] / total
    return weighted


def fit_class_weights(probabilities: pd.DataFrame, labels: pd.Series, classes: list[str], scorer) -> dict[str, float]:
    """Coordinate search of one multiplicative weight per class, maximising `scorer`."""
    weights = {label: 1.0 for label in classes}
    best = scorer(labels, apply_class_weights(probabilities, weights))
    for _ in range(2):
        for label in classes:
            for value in WEIGHT_GRID:
                trial = {**weights, label: value}
                result = scorer(labels, apply_class_weights(probabilities, trial))
                if result > best + 1e-12:
                    best, weights = result, trial
    return weights


def run_decision_stage(parent_dir: str) -> None:
    """Re-weight the parent's class probabilities for a label metric (accuracy, F1…).

    For each fold, weights are fitted on the other folds' out-of-fold predictions only, so the
    reported score never uses the labels it is evaluated on. Lab-only: needs the forge package."""
    from forge.evaluation import score
    from forge.task import TaskSpec

    context = load_context()
    task = TaskSpec(**context["task"])
    sign = 1.0 if task.higher_is_better else -1.0

    def scorer(labels, frame):
        return sign * score(task, labels.reset_index(drop=True), frame.reset_index(drop=True))

    parent = Path(parent_dir)
    oof = pd.read_csv(parent / "oof.csv")
    train, _ = load_data(context)
    labels = train[task.target].astype(str)
    folds = np.asarray(context["folds"])
    classes = context["classes"]
    result = oof.copy()
    for fold in validation_folds(folds):
        fit_rows = (folds != fold) & (folds >= 0)
        weights = fit_class_weights(oof[fit_rows], labels[fit_rows], classes, scorer)
        result.loc[folds == fold] = apply_class_weights(oof[folds == fold], weights).to_numpy()
    final = fit_class_weights(oof[folds >= 0], labels[folds >= 0], classes, scorer)
    test = None
    if (parent / "test_pred.csv").is_file():
        parent_test = pd.read_csv(parent / "test_pred.csv")
        test = parent_test.copy()
        test[class_columns(test)] = apply_class_weights(parent_test[class_columns(parent_test)], final).to_numpy()
    save_predictions(context, result, test, {"model": "decision_stage", "class_weights": final, "parent": parent.name})


# --- submission format ---------------------------------------------------------------------------

def to_submission(sample: pd.DataFrame, predictions: pd.DataFrame, kind: str, classes: list[str] | None,
                  positive_label: str | None) -> pd.DataFrame:
    """Map ML Forge test predictions onto the columns and row order of the sample submission.

    kind: 'label' (predicted class), 'proba' (probabilities) or 'value' (regression)."""
    id_column = sample.columns[0]
    if id_column not in predictions.columns:
        raise ValueError(f"Les prédictions de test n'ont pas la colonne d'identifiant {id_column}.")
    targets = list(sample.columns[1:])
    values = pd.DataFrame({id_column: predictions[id_column]})
    if kind == "value":
        values[targets[0]] = predictions["prediction"]
    elif kind == "label":
        labels = pd.Series(labels_from_probabilities(predictions))
        if pd.api.types.is_numeric_dtype(sample[targets[0]]):
            numeric = pd.to_numeric(labels, errors="coerce")
            labels = numeric if numeric.notna().all() else labels  # mismatch left to the format check
        values[targets[0]] = labels
    elif kind == "proba":
        if len(targets) == 1:
            positive = positive_label if positive_label is not None else sorted(classes)[-1]
            values[targets[0]] = predictions[PROBA_PREFIX + positive]
        else:
            missing = [t for t in targets if PROBA_PREFIX + str(t) not in predictions.columns]
            if missing:
                raise ValueError(f"Classes attendues par l'échantillon sans prédiction : {missing}.")
            for target in targets:
                values[target] = predictions[PROBA_PREFIX + str(target)]
    else:
        raise ValueError(f"Type de soumission inconnu : {kind} (label, proba ou value).")
    merged = sample[[id_column]].merge(values, on=id_column, how="left")
    return merged[list(sample.columns)]
