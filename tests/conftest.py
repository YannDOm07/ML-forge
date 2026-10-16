from pathlib import Path

import pytest

from forge.config import load_config, write_default_config
from forge.workspace import Workspace, open_workspace


def make_card(**overrides):
    card = {
        "question": "La baseline est-elle battue par un modèle d'arbres ?",
        "hypothesis": "Un RandomForest capte des interactions ignorées par la baseline.",
        "method": "RandomForest 300 arbres",
        "protocol_id": "cv5-seed2026",
        "justification": [{"kind": "observed_fact", "text": "Interactions visibles entre x1 et x2"}],
        "estimate": {"expected_gain": 0.02, "success_probability": 0.5, "cost_minutes": 3, "risk": "low"},
    }
    card.update(overrides)
    return card


@pytest.fixture(autouse=True)
def no_real_kaggle_token(tmp_path: Path, monkeypatch):
    """Tests never read the user's real kaggle.json nor environment token."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "kaggle-config"))


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "mlforge.toml"
    write_default_config(path, tmp_path / "root")
    return path


@pytest.fixture
def workspace(config_path: Path) -> Workspace:
    return open_workspace(load_config(config_path))
