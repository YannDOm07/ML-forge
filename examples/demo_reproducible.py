"""Démonstration reproductible de ML Forge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_forge.core.project_manager import ProjectManager


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "examples" / "demo_dataset.csv"


def build_demo_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22, 25, 28, 31, 35, 39, 44, 52, 60, 65],
            "income": [1500, 1700, 1800, 2000, 2300, 2600, 2900, 3200, 3600, 4000],
            "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_demo_dataset().to_csv(DATA_PATH, index=False)

    pm = ProjectManager()
    project_id = pm.create_project(
        name="Demo Reproducible",
        description="Démonstration locale de reproductibilité ML Forge",
    )
    project = pm.open_project(project_id)
    if project is None:
        raise RuntimeError("Projet de démonstration non créé.")

    result = project.run_baseline_experiment(
        str(DATA_PATH),
        target_column="target",
        hypothesis="Une baseline linéaire simple doit fournir un point de comparaison utile.",
        method="Baseline logistic regression",
    )

    print(json.dumps({
        "project_id": project_id,
        "task_type": result["task_type"],
        "metrics": result["metrics"],
        "experiment_id": result["experiment_id"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
