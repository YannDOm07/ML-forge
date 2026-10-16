"""End-to-end CLI smoke tests (French output, exit codes)."""

from forge.cli import main


def test_cli_project_lifecycle(tmp_path, capsys):
    config = str(tmp_path / "mlforge.toml")
    root = tmp_path / "root"

    assert main(["--config", config, "init", "--root", str(root)]) == 0
    assert main(["--config", config, "create", "Titanic", "--url", "https://www.kaggle.com/c/titanic"]) == 0
    project_id = capsys.readouterr().out.split("Projet créé : ")[1].split()[0]

    assert main(["--config", config, "list"]) == 0
    assert project_id in capsys.readouterr().out
    assert main(["--config", config, "status", project_id]) == 0
    assert "Expériences : aucune" in capsys.readouterr().out
    assert main(["--config", config, "backup", project_id]) == 0
    assert main(["--config", config, "disk"]) == 0
    assert "total" in capsys.readouterr().out


def test_cli_delete_refused_when_not_interactive(tmp_path, capsys):
    config = str(tmp_path / "mlforge.toml")
    main(["--config", config, "init", "--root", str(tmp_path / "root")])
    main(["--config", config, "create", "Garder"])
    project_id = capsys.readouterr().out.split("Projet créé : ")[1].split()[0]
    # Under pytest, stdin is not a terminal: the approver must refuse.

    assert main(["--config", config, "delete", project_id]) == 2

    assert "Action refusée" in capsys.readouterr().err
    assert (tmp_path / "root" / "projects" / project_id / "project.db").is_file()


def test_cli_reports_invalid_config(tmp_path, capsys):
    config = tmp_path / "mlforge.toml"
    main(["--config", str(config), "init", "--root", str(tmp_path / "root")])
    config.write_text(config.read_text(encoding="utf-8") + "\n[oops]\n", encoding="utf-8")

    assert main(["--config", str(config), "list"]) == 2
    assert "section inconnue [oops]" in capsys.readouterr().err
