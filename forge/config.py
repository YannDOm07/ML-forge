"""Loading, validation and generation of the ML Forge configuration (TOML)."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from forge.errors import ConfigError

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = APP_DIR / "config" / "mlforge.toml"
CONFIG_ENV_VAR = "MLFORGE_CONFIG"
DEFAULT_ROOT = Path("C:/MLForge") if os.name == "nt" else Path.home() / "MLForge"

SYNC_ENV_VARS = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
SYNC_DIR_PREFIXES = ("onedrive", "dropbox", "google drive", "icloud drive", "iclouddrive")

Value = bool | int | float | str


@dataclass(frozen=True)
class Setting:
    default: Value
    help: str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


SETTINGS: dict[str, dict[str, Setting]] = {
    "campaign": {
        "interactive_hours": Setting(2.0, "Durée maximale d'une campagne interactive (heures)", 0.1, 72),
        "unattended_hours": Setting(8.0, "Durée maximale d'une campagne sans surveillance (heures)", 0.1, 72),
        "max_experiments": Setting(40, "Nombre maximal d'expériences par campagne", 1, 10_000),
        "patience": Setting(5, "Expériences consécutives sans gain accepté avant arrêt", 1, 1_000),
        "finalize_hours_before_deadline": Setting(
            24.0, "Passage en mode finalisation avant l'échéance (heures)", 0, 720
        ),
        "keep_awake": Setting(True, "Empêcher la mise en veille du PC pendant une campagne"),
    },
    "sandbox": {
        "python": Setting("", "Interpréteur du bac à sable (vide : celui qui exécute ML Forge)"),
        "run_timeout_minutes": Setting(30, "Temps maximal d'une exécution (minutes)", 1, 1_440),
        "memory_limit_gb": Setting(10.0, "Mémoire maximale du bac à sable (Go)", 0.5, 1_024),
    },
    "data": {
        "profile_max_rows": Setting(200_000, "Lignes lues au maximum pour profiler un fichier", 1_000, 100_000_000),
    },
    "search": {
        "optuna_trials": Setting(50, "Essais Optuna par étude", 1, 100_000),
    },
    "evaluation": {
        "n_splits": Setting(5, "Nombre de plis de la validation croisée", 2, 20),
        "seed": Setting(2026, "Seed du découpage et des modèles", 0, 2**31 - 1),
        "min_class_per_fold": Setting(
            5, "En dessous de ce nombre d'exemples d'une classe par pli, la validation est jugée fragile", 1, 1_000
        ),
        "stability_seeds": Setting(3, "Seeds utilisées pour vérifier la stabilité du candidat final", 1, 100),
        "acceptance_std_factor": Setting(
            1.0, "Un gain est accepté s'il dépasse ce multiple de l'écart-type mesuré", 0, 10
        ),
    },
    "acquisition": {
        "max_download_gb": Setting(5.0, "Taille maximale d'un téléchargement automatique (Go)", 0.01, 10_000),
        "min_free_disk_gb": Setting(20.0, "Espace disque minimal à préserver (Go)", 0, 100_000),
        "link_depth": Setting(2, "Profondeur de suivi des liens depuis la page de compétition", 0, 5),
    },
    "llm": {
        "max_raw_rows": Setting(
            20, "Lignes brutes maximales envoyées au LLM (0 si le partage est interdit ou incertain)", 0, 1_000
        ),
        "cycle_summary_tokens": Setting(3_000, "Taille maximale du résumé de cycle (tokens)", 500, 200_000),
        "assumed_context_tokens": Setting(
            32_000, "Fenêtre de contexte supposée du LLM, remplacée par la valeur mesurée", 2_000, 10_000_000
        ),
    },
    "output": {
        "verbosity": Setting("synthetic", "Niveau de détail des comptes rendus", choices=("synthetic", "detailed")),
        "language": Setting("fr", "Langue de l'interface et des rapports", choices=("fr",)),
    },
    "backup": {
        "before_migration": Setting(True, "Sauvegarder une base avant toute migration de schéma"),
        "after_campaign": Setting(True, "Sauvegarder le projet à la fin de chaque campagne"),
    },
}


@dataclass(frozen=True)
class Config:
    root: Path
    values: dict[str, dict[str, Value]]
    source: Path
    text: str

    def get(self, section: str, key: str) -> Value:
        return self.values[section][key]


def resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    from_env = os.environ.get(CONFIG_ENV_VAR)
    return Path(from_env) if from_env else DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None) -> Config:
    source = resolve_config_path(path)
    if not source.is_file():
        raise ConfigError(
            f"Aucune configuration trouvée à {source}. Lancez d'abord : python -m forge init"
        )
    return parse_config(source.read_text(encoding="utf-8"), source)


def parse_config(text: str, source: Path) -> Config:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source} : syntaxe TOML invalide ({exc}).") from exc
    errors: list[str] = []
    root = _validate_root(raw.get("root"), errors)
    values = _validate_sections(raw, errors)
    if errors:
        raise ConfigError(f"Configuration invalide ({source}) :\n- " + "\n- ".join(errors))
    return Config(root=root, values=values, source=source, text=text)


def _validate_root(value: object, errors: list[str]) -> Path:
    if not isinstance(value, str) or not value.strip():
        errors.append("root : chemin de la racine des données manquant")
        return Path()
    root = Path(value)
    if not root.is_absolute():
        errors.append(f"root : « {value} » doit être un chemin absolu")
    synced = synced_folder(root)
    if synced:
        errors.append(
            f"root : « {value} » est dans un dossier synchronisé ({synced}). "
            "SQLite et les datasets doivent être hors OneDrive/Dropbox, par exemple C:/MLForge"
        )
    return root


def _validate_sections(raw: dict, errors: list[str]) -> dict[str, dict[str, Value]]:
    for section in raw:
        if section != "root" and section not in SETTINGS:
            errors.append(f"section inconnue [{section}]")
    values: dict[str, dict[str, Value]] = {}
    for section, settings in SETTINGS.items():
        given = raw.get(section, {})
        if not isinstance(given, dict):
            errors.append(f"[{section}] doit être une section")
            given = {}
        for key in given:
            if key not in settings:
                errors.append(f"clé inconnue {section}.{key}")
        values[section] = {
            key: _validate_value(f"{section}.{key}", given.get(key, setting.default), setting, errors)
            for key, setting in settings.items()
        }
    return values


def _validate_value(name: str, value: object, setting: Setting, errors: list[str]) -> Value:
    expected = type(setting.default)
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if type(value) is not expected:
        errors.append(f"{name} : type {type(value).__name__} reçu, {expected.__name__} attendu")
        return setting.default
    if setting.minimum is not None and value < setting.minimum:
        errors.append(f"{name} : {value} est inférieur au minimum {setting.minimum}")
    if setting.maximum is not None and value > setting.maximum:
        errors.append(f"{name} : {value} est supérieur au maximum {setting.maximum}")
    if setting.choices is not None and value not in setting.choices:
        errors.append(f"{name} : « {value} » n'est pas parmi {', '.join(setting.choices)}")
    return value


def synced_folder(path: Path) -> str | None:
    """Return the synchronised folder containing `path`, if any."""
    resolved = Path(os.path.abspath(path))
    for var in SYNC_ENV_VARS:
        base = os.environ.get(var)
        if base and resolved.is_relative_to(Path(os.path.abspath(base))):
            return base
    for part in resolved.parts:
        if part.lower().startswith(SYNC_DIR_PREFIXES):
            return part
    return None


def render_default_config(root: Path, overrides: dict[str, dict[str, Value]] | None = None) -> str:
    lines = [
        "# Configuration ML Forge. Toutes les valeurs sont modifiables ;",
        "# une clé inconnue ou une valeur hors limites est refusée au démarrage.",
        "",
        "# Racine des données et des bases, hors de tout dossier synchronisé.",
        f"root = {json.dumps(root.as_posix())}",
    ]
    for section, settings in SETTINGS.items():
        lines += ["", f"[{section}]"]
        for key, setting in settings.items():
            value = (overrides or {}).get(section, {}).get(key, setting.default)
            lines += [f"# {setting.help}", f"{key} = {_toml_value(value)}"]
    return "\n".join(lines) + "\n"


def _toml_value(value: Value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def write_default_config(path: Path, root: Path, overrides: dict[str, dict[str, Value]] | None = None) -> Config:
    """Create a configuration file; never overwrite an existing one."""
    text = render_default_config(root, overrides)
    config = parse_config(text, path)
    if path.exists():
        raise ConfigError(f"{path} existe déjà : il n'est pas écrasé.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return config
