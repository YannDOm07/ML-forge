"""Tests 7 (invalid configuration) and 8 (synchronised root refused)."""

from pathlib import Path

import pytest

from forge.config import load_config, parse_config, render_default_config, write_default_config
from forge.db import session
from forge.errors import ConfigError
from forge.workspace import open_workspace


def test_default_config_is_valid(tmp_path):
    config = parse_config(render_default_config(tmp_path / "root"), tmp_path / "c.toml")
    assert config.get("campaign", "unattended_hours") == 8.0
    assert config.get("llm", "max_raw_rows") == 20
    assert config.get("acquisition", "max_download_gb") == 5.0


@pytest.mark.parametrize(
    "replace, expected",
    [
        (("max_experiments = 40", "max_experiments = 0"), "inférieur au minimum"),
        (("max_experiments = 40", 'max_experiments = "40"'), "type str"),
        (("keep_awake = true", "keep_awake = 1"), "type int"),
        (('verbosity = "synthetic"', 'verbosity = "bavard"'), "n'est pas parmi"),
        (("[search]", "[search]\nunknown_key = 3"), "clé inconnue search.unknown_key"),
        (("[search]", "[mystery]\n[search]"), "section inconnue [mystery]"),
        (("optuna_trials = 50", "optuna_trials = "), "syntaxe TOML invalide"),
    ],
)
def test_invalid_config_is_refused(tmp_path, replace, expected):
    text = render_default_config(tmp_path / "root").replace(*replace)
    with pytest.raises(ConfigError, match=expected.replace("[", r"\[").replace("]", r"\]")):
        parse_config(text, tmp_path / "c.toml")


def test_relative_root_is_refused(tmp_path):
    text = render_default_config(tmp_path / "root")
    text = text.replace(f'root = "{(tmp_path / "root").as_posix()}"', 'root = "relatif/racine"')
    with pytest.raises(ConfigError, match="chemin absolu"):
        parse_config(text, tmp_path / "c.toml")


def test_float_accepts_integer_value(tmp_path):
    text = render_default_config(tmp_path / "root").replace("unattended_hours = 8.0", "unattended_hours = 6")
    assert parse_config(text, tmp_path / "c.toml").get("campaign", "unattended_hours") == 6.0


def test_root_inside_onedrive_env_is_refused(tmp_path, monkeypatch):
    fake_onedrive = tmp_path / "SyncedHome"
    monkeypatch.setenv("OneDrive", str(fake_onedrive))
    with pytest.raises(ConfigError, match="dossier synchronisé"):
        write_default_config(tmp_path / "c.toml", fake_onedrive / "MLForge")
    assert not (tmp_path / "c.toml").exists()


def test_root_with_dropbox_component_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="dossier synchronisé"):
        parse_config(render_default_config(tmp_path / "Dropbox" / "MLForge"), tmp_path / "c.toml")


def test_missing_config_explains_init(tmp_path):
    with pytest.raises(ConfigError, match="forge init"):
        load_config(tmp_path / "absent.toml")


def test_existing_config_is_never_overwritten(config_path):
    before = config_path.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match="existe déjà"):
        write_default_config(config_path, Path(config_path.parent / "other"))
    assert config_path.read_text(encoding="utf-8") == before


def test_configuration_history_records_changes_once(config_path):
    workspace = open_workspace(load_config(config_path))
    open_workspace(load_config(config_path))
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("optuna_trials = 50", "optuna_trials = 80"),
        encoding="utf-8",
    )
    open_workspace(load_config(config_path))
    with session(workspace.global_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM configuration_history").fetchone()[0]
    assert count == 2


def test_init_adapts_the_sandbox_memory_to_the_machine(tmp_path):
    from forge.system import Hardware, prudent_settings

    small = prudent_settings(Hardware(cpu_threads=4, ram_gb=7.4, free_disk_gb=100))
    assert small == {"sandbox": {"memory_limit_gb": 4.4}}
    assert prudent_settings(Hardware(cpu_threads=16, ram_gb=64, free_disk_gb=100))["sandbox"]["memory_limit_gb"] == 10.0
    config = write_default_config(tmp_path / "c.toml", tmp_path / "root", small)
    assert config.get("sandbox", "memory_limit_gb") == 4.4
