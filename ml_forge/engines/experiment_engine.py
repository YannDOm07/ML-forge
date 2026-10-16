"""Baseline experimentation pipeline for ML Forge."""

from __future__ import annotations

from time import time
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_forge.engines.data_engine import DataEngine
from ml_forge.engines.feature_engineering import FeatureEngineer


class ExperimentEngine:
    """Exécute une baseline ML simple et robuste."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def infer_task_type(self, y: pd.Series) -> str:
        return DataEngine.infer_task_type(y)

    @staticmethod
    def _make_model(task_type: str, **kwargs):
        model_kwargs = dict(kwargs)
        random_state = model_kwargs.pop("random_state", 42)

        if task_type == "classification":
            model_kwargs.setdefault("random_state", random_state)
            model_kwargs.setdefault("max_iter", 2000)
            model_kwargs.setdefault("class_weight", "balanced")
            return LogisticRegression(**model_kwargs)

        return LinearRegression(**model_kwargs)

    def build_pipeline(self, X: pd.DataFrame, task_type: str, model: Optional[Any] = None):
        """Crée un pipeline de base adapté au type de tâche."""
        if model is None:
            model = self._make_model(task_type, random_state=self.random_state)

        numeric_features = [
            col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])
        ]
        categorical_features = [
            col for col in X.columns if col not in numeric_features
        ]

        transformers: List[Any] = []
        if numeric_features:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                )
            )
        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                )
            )

        if not transformers:
            raise ValueError("Aucune colonne exploitable pour construire le pipeline.")

        return Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
                ("model", model),
            ]
        )

    def generate_feature_matrix(
        self,
        X: pd.DataFrame,
        feature_strategy: str = "ratio",
    ) -> pd.DataFrame:
        """Applique une génération de features légère et reproductible."""
        return FeatureEngineer(strategy=feature_strategy).generate_features(X, strategy=feature_strategy)

    def run_experiment(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Optional[str] = None,
        cv_folds: int = 5,
        feature_strategy: str = "ratio",
        optimize_hyperparameters: bool = False,
        feature_generator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Entraîne une baseline et retourne les métriques principales."""
        if X is None or y is None:
            raise ValueError("Les données X et y doivent être fournies.")

        if task_type is None:
            task_type = self.infer_task_type(y)

        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Type de tâche non supporté : {task_type}")

        X = X.copy()
        y = y.copy()

        if feature_generator is not None:
            X = feature_generator(X)
        elif feature_strategy and feature_strategy.lower() != "none":
            X = self.generate_feature_matrix(X, feature_strategy=feature_strategy)

        for column in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[column]):
                X[column] = X[column].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y if task_type == "classification" else None,
        )

        if optimize_hyperparameters:
            optimized = self.optimize_hyperparameters(X_train, X_test, y_train, y_test, task_type, max_trials=5)
            return {
                "task_type": task_type,
                "n_samples": len(y),
                "n_features": X.shape[1],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "duration_seconds": round(optimized["duration_seconds"], 4),
                "metrics": optimized["metrics"],
                "model": optimized["model"],
                "best_params": optimized.get("best_params", {}),
            }

        pipeline = self.build_pipeline(X, task_type)
        started_at = time()
        pipeline.fit(X_train, y_train)
        elapsed = time() - started_at

        if task_type == "classification":
            predictions = pipeline.predict(X_test)
            metrics = self._evaluate_classification(y_test, predictions)
            metrics["cross_val_accuracy"] = self._cross_validation_score(
                X,
                y,
                task_type,
                cv_folds,
                scoring="accuracy",
            )
            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            metrics["classification_report"] = report
        else:
            predictions = pipeline.predict(X_test)
            metrics = self._evaluate_regression(y_test, predictions)
            metrics["cross_val_r2"] = self._cross_validation_score(
                X,
                y,
                task_type,
                cv_folds,
                scoring="r2",
            )

        return {
            "task_type": task_type,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "duration_seconds": round(elapsed, 4),
            "metrics": metrics,
            "model": pipeline.named_steps["model"].__class__.__name__,
        }

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        task_type: str,
        max_trials: int = 5,
    ) -> Dict[str, Any]:
        """Choisit un réglage simple de C pour la classification ou de positive/negative pour la régression."""
        if task_type == "classification":
            candidate_values = [0.01, 0.1, 1.0, 5.0, 10.0]
            best_score = -1.0
            best_model = None
            best_params = {"C": 1.0}
            for candidate in candidate_values[:max_trials]:
                model = LogisticRegression(
                    C=candidate,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=self.random_state,
                )
                pipeline = self.build_pipeline(X_train, task_type, model=model)
                started_at = time()
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                metrics = self._evaluate_classification(y_test, predictions)
                score = metrics.get("accuracy", 0.0)
                if score > best_score:
                    best_score = score
                    best_model = pipeline
                    best_params = {"C": candidate}
                duration = time() - started_at
            if best_model is None:
                best_model = self.build_pipeline(X_train, task_type)
                best_model.fit(X_train, y_train)
                duration = 0.0
            return {
                "model": best_model.named_steps["model"].__class__.__name__,
                "best_params": best_params,
                "duration_seconds": duration,
                "metrics": self._evaluate_classification(y_test, best_model.predict(X_test)),
            }

        model = LinearRegression()
        pipeline = self.build_pipeline(X_train, task_type, model=model)
        started_at = time()
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        duration = time() - started_at
        return {
            "model": pipeline.named_steps["model"].__class__.__name__,
            "best_params": {},
            "duration_seconds": duration,
            "metrics": self._evaluate_regression(y_test, predictions),
        }

    def _cross_validation_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv_folds: int,
        scoring: str,
    ) -> float:
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=min(cv_folds, len(y.unique())), shuffle=True, random_state=self.random_state)
        else:
            cv = None

        base_estimator = self.build_pipeline(X, task_type)
        if task_type == "classification":
            scores = cross_val_score(base_estimator, X, y, cv=cv, scoring=scoring)
        else:
            cv = None if len(X) < 3 else min(cv_folds, 3)
            scores = cross_val_score(base_estimator, X, y, cv=cv, scoring=scoring)
        return float(scores.mean()) if len(scores) else 0.0

    @staticmethod
    def _evaluate_classification(y_true: pd.Series, y_pred: List[Any]) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        return metrics

    @staticmethod
    def _evaluate_regression(y_true: pd.Series, y_pred: List[Any]) -> Dict[str, float]:
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        }
