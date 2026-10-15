"""Step 3 (a project named after the competition, laid out the same way) and step 9 (journal,
progress shared by every project, notebooks regenerated from the database)."""

import json

from fake_web import FakeTransport, html, json_response, raw
from forge.organize import (
    DIRECTORIES,
    competition_name,
    create_from_link,
    journal,
    progress_path,
    read_journal,
    write_notebooks,
)
from forge.projects import create_project

SITE = "https://challenges.example.org"
URL = f"{SITE}/web/challenges/challenge-page/12/overview"
API = f"{SITE}/api/"


def evalai_routes():
    return {
        f"{SITE}/dist/js/config.min.js": raw("{}", content_type="application/javascript"),
        f"{API}challenges/challenge/12/": json_response(json.dumps({"id": 12, "title": "Lyrics Cup 2026"})),
        f"{API}challenges/challenge/12/challenge_phase": json_response(json.dumps({"results": []})),
    }


def test_project_is_named_after_the_competition_and_laid_out(workspace):
    project = create_from_link(workspace, URL, transport=FakeTransport(evalai_routes()))
    assert project.name == "Lyrics Cup 2026" and project.id.startswith("lyrics-cup-2026-")
    assert all((project.path / name).is_dir() for name in DIRECTORIES)
    assert "Projet créé" in (project.path / "JOURNAL.md").read_text(encoding="utf-8")


def test_names_from_other_links(workspace):
    assert competition_name(workspace, "https://www.kaggle.com/competitions/house-prices", FakeTransport({})) == "House Prices"
    page = "https://cup.example.org/2026"
    assert competition_name(workspace, page, FakeTransport({page: html("<title>Radio Cup</title><p>x</p>")})) == "Radio Cup"


def test_journal_and_progress_keep_problems_for_the_next_projects(workspace):
    first = create_project(workspace, "Première")
    second = create_project(workspace, "Seconde")
    journal(first, "Acquérir", "Les étiquettes ne sont que dans l'archive d'entraînement.", problem=True)
    journal(second, "Comprendre", "12 pages lues.")
    assert progress_path(first).parent.name == "progress"
    progress = progress_path(first).read_text(encoding="utf-8")
    assert "| Première | Acquérir | ⚠ Les étiquettes" in progress and "| Seconde | Comprendre |" in progress
    from forge.lessons import record

    record(workspace, "acquerir", "erreur", "Formaliser sans l'archive qui contient les étiquettes",
           "Validation et évaluation n'ont pas de cible : rien ne peut être appris.", {})
    text = read_journal(workspace, second)
    # The project's own journal, then only the mistakes and traps of a close context (never everything).
    assert "12 pages lues." in text and "Erreurs et pièges déjà rencontrés" in text
    assert "Formaliser sans l'archive" in text.split("Erreurs et pièges")[1]


def test_notebooks_are_valid_and_describe_the_project(workspace):
    project = create_project(workspace, "Notebooks", URL)
    paths = write_notebooks(workspace, project)
    assert [p.name for p in paths] == ["01_competition.ipynb", "02_donnees.ipynb", "03_experiences.ipynb"]
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["cells"]
        assert all("id" in cell and "source" in cell for cell in notebook["cells"])
    assert "Fiche de la compétition" in "".join(json.loads(paths[0].read_text(encoding="utf-8"))["cells"][1]["source"])
    assert (project.path / "README.md").is_file()
