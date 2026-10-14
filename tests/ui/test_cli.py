from pathlib import Path

from forge.cli import main
from forge.config import load_config
from forge.projects import create_project
from forge.workspace import open_workspace


def test_ui_command_passes_project_and_port(tmp_path, monkeypatch):
    config = str(tmp_path / "mlforge.toml")
    root = tmp_path / "root"
    assert main(["--config", config, "init", "--root", str(root)]) == 0
    project = create_project(open_workspace(load_config(Path(config))), "Interface")
    called = []

    monkeypatch.setattr("forge.ui.server.serve", lambda workspace, selected, port: called.append((selected, port)))

    assert main(["--config", config, "ui", "--project", project.id, "--port", "9012"]) == 0
    assert called == [(project.id, 9012)]


def test_ui_port_must_be_valid(tmp_path):
    config = str(tmp_path / "mlforge.toml")
    main(["--config", config, "init", "--root", str(tmp_path / "root")])

    try:
        main(["--config", config, "ui", "--port", "70000"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Un port hors plage doit être refusé.")
