"""Utilities for lightweight feature generation used in advanced experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Ajoute des features heuristiques simples et robustes."""

    def __init__(self, strategy: str = "ratio"):
        self.strategy = strategy

    def generate_features(self, X: pd.DataFrame, strategy: Optional[str] = None) -> pd.DataFrame:
        """Génère des features dérivées en gardant les données lisibles."""
        engineered = X.copy()
        strategy = (strategy or self.strategy or "none").lower()
        if strategy == "none":
            return engineered

        numeric_columns = [
            column for column in engineered.columns if pd.api.types.is_numeric_dtype(engineered[column])
        ]

        if not numeric_columns:
            return engineered

        additions: dict[str, pd.Series] = {}
        for column in numeric_columns:
            additions[f"{column}_squared"] = engineered[column].pow(2)
            if engineered[column].std(ddof=0) not in (0, np.nan):
                additions[f"{column}_zscore"] = (
                    engineered[column] - engineered[column].mean()
                ) / engineered[column].std(ddof=0)

        if strategy in {"ratio", "all"} and len(numeric_columns) > 1:
            for index, left in enumerate(numeric_columns):
                for right in numeric_columns[index + 1 :]:
                    left_div_right = engineered[left] / engineered[right].replace(0, np.nan)
                    right_div_left = engineered[right] / engineered[left].replace(0, np.nan)
                    additions[f"{left}_per_{right}"] = left_div_right
                    additions[f"{right}_per_{left}"] = right_div_left
                    additions[f"{left}_minus_{right}"] = engineered[left] - engineered[right]
                    additions[f"{right}_minus_{left}"] = engineered[right] - engineered[left]

        if strategy == "all":
            for column in numeric_columns:
                additions[f"{column}_log"] = np.log1p(engineered[column].clip(lower=0))

        if additions:
            additions = {name: value for name, value in additions.items() if name not in engineered.columns}
            if additions:
                engineered = pd.concat([engineered, pd.DataFrame(additions)], axis=1)

        return engineered


FeatureGenerationEngine = FeatureEngineer
