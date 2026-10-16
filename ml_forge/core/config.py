"""Configuration centrale de ML Forge.

Gère les paramètres globaux, la détection des ressources,
et les chemins de base du système.
"""

import os
import json
import psutil
from pathlib import Path
from typing import Optional, Dict, Any


class Config:
    """Configuration globale et persistante."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.projects_dir = self.base_dir / "projects"
        self.backups_dir = self.base_dir / "backups"
        self.config_dir = self.base_dir / "config"
        self.config_file = self.config_dir / "ml_forge_config.json"

        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._load_or_create_config()

    def _load_or_create_config(self):
        """Charge ou crée la configuration par défaut."""
        defaults = {
            "llm_model": "opencode",
            "max_experiments": 50,
            "hyperparameter_budget": 20,
            "memory_limit_gb": self._get_available_memory() * 0.7,
            "timeout_seconds": 3600,
            "auto_download_models": False,
            "enable_logging": True,
            "log_level": "INFO",
            "validation_strategy": "cross_validation",
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
            "artifact_retention": "all",  # all, best, none
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                # Merge with defaults
                for key, value in defaults.items():
                    if key not in self.settings:
                        self.settings[key] = value
            except Exception:
                self.settings = defaults
        else:
            self.settings = defaults
            self._save_config()

    def _save_config(self):
        """Persiste la configuration."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2)

    def set(self, key: str, value: Any):
        """Définit une clé de configuration et la persiste."""
        self.settings[key] = value
        self._save_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une clé de configuration."""
        return self.settings.get(key, default)

    @staticmethod
    def _get_available_memory() -> float:
        """Détecte la mémoire disponible en GB."""
        try:
            mem = psutil.virtual_memory()
            return mem.available / (1024 ** 3)
        except Exception:
            return 4.0  # Défaut conservateur

    def get_system_info(self) -> Dict[str, Any]:
        """Retourne les informations système pertinentes."""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            has_gpu = self._detect_gpu()
            return {
                "cpu_cores": cpu_count,
                "memory_gb": round(memory_gb, 2),
                "has_gpu": has_gpu,
                "platform": os.name,
            }
        except Exception:
            return {
                "cpu_cores": 4,
                "memory_gb": 8,
                "has_gpu": False,
                "platform": os.name,
            }

    @staticmethod
    def _detect_gpu() -> bool:
        """Détecte la présence d'un GPU (CUDA ou similaire)."""
        try:
            import torch
            return torch.cuda.is_available()
        except (ImportError, Exception):
            return False


# Instance globale
config = Config()
