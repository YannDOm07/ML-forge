"""Data inspection utilities for ML Forge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class DataEngine:
    """Analyse les jeux de données avant l'entraînement."""

    def __init__(self, sample_limit: int = 5000):
        self.sample_limit = sample_limit

    def load_dataset(self, dataset_path: str | Path, **kwargs) -> pd.DataFrame:
        """Charge un dataset CSV/Excel/Parquet."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {path}")

        suffix = path.suffix.lower()

        if suffix in {".csv", ".txt"}:
            sep = kwargs.pop("sep", ",")
            if ".tsv" in path.name.lower() or sep == "\t":
                sep = "\t"
            return pd.read_csv(path, sep=sep, **kwargs)
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path, **kwargs)
        if suffix == ".parquet":
            return pd.read_parquet(path, **kwargs)

        return pd.read_csv(path, **kwargs)

    def inspect_dataset(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retourne une synthèse exploitable du dataset."""
        if df is None or df.empty:
            return {
                "rows": 0,
                "columns": 0,
                "numeric_columns": [],
                "categorical_columns": [],
                "datetime_columns": [],
                "text_columns": [],
                "missing_values": {},
                "duplicate_rows": 0,
                "target_column": target_column,
                "task_type": "unknown",
                "class_distribution": {},
                "suggested_hypotheses": ["Le dataset est vide ou inactif."],
            }

        rows = len(df)
        cols = len(df.columns)
        missing_values = {
            str(column): int(value)
            for column, value in df.isna().sum().items()
            if value > 0
        }

        numeric_columns = list(df.select_dtypes(include=["number"]).columns)
        categorical_columns = list(df.select_dtypes(exclude=["number"]).columns)
        datetime_columns = list(df.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns)
        text_columns = [
            column
            for column in categorical_columns
            if column not in datetime_columns and df[column].astype(str).str.len().mean() > 15
        ]
        categorical_columns = [
            column for column in categorical_columns if column not in datetime_columns and column not in text_columns
        ]

        summary: Dict[str, Any] = {
            "rows": rows,
            "columns": cols,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_columns": datetime_columns,
            "text_columns": text_columns,
            "missing_values": missing_values,
            "duplicate_rows": int(df.duplicated().sum()),
            "target_column": target_column,
            "task_type": "unknown",
            "class_distribution": {},
        }

        if target_column and target_column in df.columns:
            target_series = df[target_column]
            summary["task_type"] = self.infer_task_type(target_series)
            summary["class_distribution"] = self._class_distribution(target_series)

        summary["suggested_hypotheses"] = self.suggest_hypotheses(summary)
        summary["notes"] = self._build_notes(summary)
        return summary

    @staticmethod
    def infer_task_type(series: pd.Series) -> str:
        """Déduit le type d'objectif : classification ou régression."""
        if series.empty:
            return "unknown"

        if pd.api.types.is_numeric_dtype(series):
            unique_count = series.nunique(dropna=True)
            if unique_count <= 20:
                return "classification"
            return "regression"

        return "classification"

    @staticmethod
    def _class_distribution(series: pd.Series) -> Dict[str, float]:
        """Calcule la répartition des classes."""
        if series.empty:
            return {}

        counts = series.dropna().value_counts(normalize=True)
        return {str(key): float(value) for key, value in counts.items()}

    def suggest_hypotheses(self, summary: Dict[str, Any]) -> List[str]:
        """Propose des hypothèses de travail à partir des observations."""
        hypotheses: List[str] = []

        if summary.get("duplicate_rows", 0) > 0:
            hypotheses.append("Supprimer les doublons avant d'entraîner le modèle pour éviter la fuite de données.")

        missing_count = sum(summary.get("missing_values", {}).values())
        if missing_count > 0:
            hypotheses.append("Imputer les valeurs manquantes ou utiliser un pipeline de prétraitement adapté.")

        if summary.get("task_type") == "classification":
            distribution = summary.get("class_distribution", {})
            if distribution:
                imbalance = max(distribution.values()) - min(distribution.values())
                if imbalance > 0.25:
                    hypotheses.append("Le jeu de données est déséquilibré ; tester une pondération des classes ou une métrique robuste.")
        else:
            hypotheses.append("Tester une régression linéaire ou un modèle robuste sur les variables numériques.")

        if summary.get("categorical_columns"):
            hypotheses.append("Convertir les variables catégorielles en features utilisables et vérifier leur cardinalité.")

        if not hypotheses:
            hypotheses.append("Le dataset semble exploitable ; lancer une baseline simple pour établir un point de référence.")

        return hypotheses

    def _build_notes(self, summary: Dict[str, Any]) -> List[str]:
        """Construit des notes lisibles."""
        notes: List[str] = []
        if summary.get("rows", 0) > 10000:
            notes.append("Le dataset est volumineux ; une lecture par bloc ou un échantillonnage peut être préférable.")
        if summary.get("missing_values"):
            notes.append(f"{sum(summary['missing_values'].values())} valeurs manquantes détectées.")
        if summary.get("duplicate_rows", 0):
            notes.append(f"{summary['duplicate_rows']} doublons détectés.")
        if summary.get("categorical_columns"):
            notes.append(f"{len(summary['categorical_columns'])} colonnes catégorielles détectées.")
        return notes

    def generate_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Construit un rapport d'analyse lisible."""
        return {
            "rows": summary.get("rows", 0),
            "columns": summary.get("columns", 0),
            "task_type": summary.get("task_type", "unknown"),
            "numeric_columns": summary.get("numeric_columns", []),
            "categorical_columns": summary.get("categorical_columns", []),
            "datetime_columns": summary.get("datetime_columns", []),
            "text_columns": summary.get("text_columns", []),
            "missing_values": summary.get("missing_values", {}),
            "duplicate_rows": summary.get("duplicate_rows", 0),
            "suggested_hypotheses": summary.get("suggested_hypotheses", []),
            "notes": summary.get("notes", []),
        }
