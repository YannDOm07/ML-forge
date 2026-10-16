"""The project as a working directory: named after the competition, laid out the same way every
time, with a progress journal and notebooks that ML Forge regenerates at each step.

Layout of a project directory:

    README.md        what the competition asks, where things are (regenerated)
    JOURNAL.md       progress and problems met, step by step (append-only)
    notebooks/       01_competition, 02_data, 03_experiments (regenerated, ML Forge's view)
    notebooks/perso/ the user's own notebooks (never touched)
    src/             code of the experiments
    submissions/     files ready to be submitted by the user
    reports/         reports and deliverables
    inbox/           files the user drops when a download needs them
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from forge.db import now_iso, session
from forge.experiments import log_event
from forge.projects import Project
from forge.workspace import Workspace

DIRECTORIES = ("notebooks", "notebooks/perso", "src", "submissions", "reports", "inbox")
JOURNAL = "JOURNAL.md"
GENERATED = "Généré par ML Forge à chaque étape : ne pas modifier (vos notebooks vont dans notebooks/perso/)."
TABLE_SUFFIXES = (".csv", ".tsv", ".json", ".parquet", ".feather")


# --- name from the link ---------------------------------------------------------------------

def competition_name(workspace: Workspace, url: str, transport=None) -> str:
    """The competition's title read from its platform, else a readable name taken from the link."""
    from forge.evalai import challenge_id, read_challenge
    from forge.intake import identify_platform
    from forge.web import NetworkError, Web

    web = Web(workspace, transport)
    platform = identify_platform(url)
    try:
        if platform.name == "evalai" and challenge_id(url):
            pages = read_challenge(web, url, [])
            if pages and pages[0].title:
                return pages[0].title
        elif platform.name not in {"kaggle", "zindi"}:
            page = web.read(url, "Nom de la compétition")
            if page.ok and page.title and len(page.title) > 3:
                return page.title
    except NetworkError:
        pass
    if platform.slug and not platform.slug.isdigit():
        return platform.slug.replace("-", " ").replace("_", " ").strip().title()
    parsed = urlparse(url)
    words = [part for part in parsed.path.split("/") if part and not part.isdigit()]
    return " ".join([parsed.netloc.removeprefix("www.")] + words[-1:]) or "Compétition"


def create_from_link(workspace: Workspace, url: str, name: str | None = None, transport=None) -> Project:
    """A project from the competition link alone: named after the competition, laid out, journaled."""
    from forge.projects import create_project

    project = create_project(workspace, name or competition_name(workspace, url, transport), url)
    ensure_layout(project)
    journal(project, "Création", f"Projet créé depuis {url} : dossier {project.path}.")
    return project


# --- layout and journal ----------------------------------------------------------------------

def ensure_layout(project: Project) -> None:
    for name in DIRECTORIES:
        (project.path / name).mkdir(parents=True, exist_ok=True)
    journal = project.path / JOURNAL
    if not journal.exists():
        journal.write_text(
            f"# Journal — {project.name}\n\n"
            f"Compétition : {project.competition_url or '—'}\n\n"
            "Chaque étape y laisse son résultat, ses problèmes et ce qui a été décidé.\n",
            encoding="utf-8",
        )


PROGRESS = "progress/chronologie.md"  # under the workspace root: one line per step, every project


def progress_path(project: Project) -> Path:
    path = project.path.parent.parent / PROGRESS
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def journal(project: Project, step: str, message: str, problem: bool = False) -> None:
    """Append one entry to the project journal, to its event log and to the workspace progress."""
    ensure_layout(project)
    marker = "⚠ " if problem else ""
    stamp = now_iso()[:16].replace("T", " ")
    with (project.path / JOURNAL).open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {stamp} — {step}\n\n{marker}{message.strip()}\n")
    log_event(project.db_path, None, "journal", f"{step} : {message.strip()[:500]}")
    progress = progress_path(project)
    if not progress.exists():
        progress.write_text("# Progression de tous les projets ML Forge\n\n"
                            "Une ligne par étape ; ⚠ marque un problème à ne pas reproduire.\n\n"
                            "| Date | Projet | Étape | Résultat |\n|---|---|---|---|\n", encoding="utf-8")
    summary = " ".join(message.strip().split())[:220].replace("|", "/")
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(f"| {stamp} | {project.name} | {step} | {marker}{summary} |\n")


def read_journal(workspace: Workspace, project: Project, max_chars: int = 6000) -> str:
    """The end of the project journal, then the mistakes and traps recorded for a similar context
    (never the whole register: only the closest lessons)."""
    from forge.lessons import brief, project_context, search

    ensure_layout(project)
    own = (project.path / JOURNAL).read_text(encoding="utf-8")[-max_chars:]
    context = project_context(project)
    warnings = search(workspace, context, kind="erreur", limit=5) + search(workspace, context, kind="piege", limit=5)
    lessons = "\n\n# Erreurs et pièges déjà rencontrés dans un contexte proche\n" + brief(warnings) if warnings else ""
    return own + lessons


# --- notebooks -------------------------------------------------------------------------------

def _cell(kind: str, text: str, index: int) -> dict:
    lines = text.strip("\n").splitlines(keepends=True)
    cell = {"cell_type": kind, "id": f"c{index:03d}", "metadata": {}, "source": lines}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def write_notebook(path: Path, cells: list[tuple[str, str]]) -> Path:
    notebook = {
        "cells": [_cell(kind, text, index) for index, (kind, text) in enumerate(cells)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def write_notebooks(workspace: Workspace, project: Project) -> list[Path]:
    """(Re)generate the three notebooks describing the project's current state."""
    ensure_layout(project)
    folder = project.path / "notebooks"
    paths = [
        write_notebook(folder / "01_competition.ipynb", _competition_cells(project)),
        write_notebook(folder / "02_donnees.ipynb", _data_cells(project)),
        write_notebook(folder / "03_experiences.ipynb", _experiment_cells(project)),
    ]
    write_readme(project)
    return paths


def _setup_cell(project: Project) -> str:
    return (
        "from pathlib import Path\nimport json\nimport pandas as pd\n\n"
        f"PROJET = Path(r\"{project.path}\")"
    )


def _competition_cells(project: Project) -> list[tuple[str, str]]:
    from forge.actions import open_actions
    from forge.intake import FIELDS, competition_card, stored_pages

    card = competition_card(project)
    rows = []
    for name, spec in FIELDS.items():
        value = card["fields"][name]
        facts = value if isinstance(value, list) else [value] if value else []
        shown = "; ".join(f"{fact['value']} [{fact['status']}]" for fact in facts) or "—"
        rows.append(f"| {spec.label} | {shown} |")
    sources = [page for page in stored_pages(project) if page["origin"] in {"web", "api"}]
    cells = [
        ("markdown", f"# {project.name}\n\n{GENERATED}\n\nLien : {project.competition_url}\n\n"
                     f"Plateforme : **{card['platform']}** — prise en charge : **{card['support_level']}**"),
        ("markdown", "## Fiche de la compétition\n\nChaque fait est cité mot pour mot depuis sa source "
                     "(`found`), déduit (`inferred`) ou fourni par l'utilisateur (`user`).\n\n"
                     "| Champ | Valeur |\n|---|---|\n" + "\n".join(rows)),
        ("markdown", "## Sources lues\n\n" + "\n".join(f"- [{page['title'] or page['url']}]({page['url']})"
                                                     for page in sources[:60])),
    ]
    actions = open_actions(project)
    if actions:
        cells.append(("markdown", "## Ce qui attend l'utilisateur\n\n" + "\n".join(
            f"- **{action['kind']}** : {action['message']}" for action in actions)))
    cells.append(("code", _setup_cell(project) + "\n\n# Citations complètes de la fiche\n"
                  "import sqlite3\nconn = sqlite3.connect(PROJET / 'project.db')\n"
                  "pd.read_sql('SELECT field, value_json, status, source_url, quote FROM competition_facts', conn)"))
    return cells


def _data_files(project: Project) -> list[dict]:
    with session(project.db_path) as conn:
        return [dict(row) for row in conn.execute("SELECT role, path, size_bytes FROM data_sources ORDER BY path")]


def _data_cells(project: Project) -> list[tuple[str, str]]:
    from forge.repositories import size_label

    files = _data_files(project)
    if not files:
        return [("markdown", f"# Données\n\n{GENERATED}\n\nAucune donnée enregistrée pour l'instant.")]
    kinds = Counter(Path(item["path"]).suffix.lower() or "(sans extension)" for item in files)
    total = sum(item["size_bytes"] for item in files)
    tables = [item["path"] for item in files if item["path"].lower().endswith(TABLE_SUFFIXES)]
    root = Path(_common_root([item["path"] for item in files]))
    lines = "\n".join(f"| `{suffix}` | {count} |" for suffix, count in kinds.most_common())
    cells = [
        ("markdown", f"# Données\n\n{GENERATED}\n\n{len(files)} fichiers, {size_label(total)}, sous `{root}`.\n\n"
                     f"| Type | Fichiers |\n|---|---|\n{lines}"),
        ("code", _setup_cell(project) + f"\nDONNEES = Path(r\"{root}\")\n\n"
                 "inventaire = pd.Series([p.suffix.lower() for p in DONNEES.rglob('*') if p.is_file()]).value_counts()\n"
                 "inventaire"),
    ]
    for path in tables[:8]:
        reader = _reader(path)
        cells.append(("markdown", f"## `{Path(path).name}`"))
        cells.append(("code", f"table = {reader}\nprint(table.shape)\ntable.head()"))
        cells.append(("code", "table.describe(include='all').T"))
    return cells


def _reader(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return f"pd.read_json(r\"{path}\")"
    if suffix == ".parquet":
        return f"pd.read_parquet(r\"{path}\")"
    if suffix == ".feather":
        return f"pd.read_feather(r\"{path}\")"
    separator = "\\t" if suffix == ".tsv" else ","
    return f"pd.read_csv(r\"{path}\", sep=\"{separator}\")"


def _common_root(paths: list[str]) -> str:
    import os

    try:
        return os.path.commonpath(paths) if len(paths) > 1 else str(Path(paths[0]).parent)
    except ValueError:
        return str(Path(paths[0]).parent)


def _experiment_cells(project: Project) -> list[tuple[str, str]]:
    from forge.experiments import list_experiments
    from forge.report import decisions_by_experiment

    experiments = list_experiments(project.db_path)
    decisions = decisions_by_experiment(project)
    cells = [("markdown", f"# Expériences\n\n{GENERATED}\n\nScores calculés par ML Forge ; chaque décision compare "
                          "l'expérience à la meilleure référence du même protocole.")]
    if not experiments:
        return cells + [("markdown", "Aucune expérience pour l'instant.")]
    rows = []
    for experiment in experiments:
        decision = decisions.get(experiment.id) or {}
        score = "" if decision.get("metric_mean") is None else f"{decision['metric_mean']:.5f}"
        rows.append(f"| `{experiment.id}` | {experiment.method[:40]} | {experiment.hypothesis[:80]} | "
                    f"{experiment.status} | {score} | {decision.get('status') or ''} |")
    cells.append(("markdown", "| Expérience | Méthode | Hypothèse | État | Score | Décision |\n"
                              "|---|---|---|---|---|---|\n" + "\n".join(rows)))
    for experiment in experiments:
        script = project.path / "experiments" / experiment.id / "script.py"
        if script.is_file():
            cells.append(("markdown", f"## `{experiment.id}` — {experiment.method}\n\n{experiment.hypothesis}"))
            cells.append(("code", script.read_text(encoding="utf-8")))
    return cells


def write_readme(project: Project) -> None:
    from forge.actions import open_actions
    from forge.intake import competition_card, fact_value

    card = competition_card(project)
    actions = open_actions(project)
    lines = [
        f"# {project.name}", "", f"Compétition : {project.competition_url}", "",
        f"- Métrique : {fact_value(card, 'metric', '—')}",
        f"- Type de problème : {fact_value(card, 'problem_type', '—')}",
        f"- Échéance : {fact_value(card, 'deadline', '—')}",
        f"- Soumissions par jour : {fact_value(card, 'rules.daily_submission_limit', '—')}",
        "", "## Dossiers", "",
        "- `notebooks/` : compétition, données, expériences (régénérés) ; `notebooks/perso/` : les vôtres",
        "- `src/` : code ; `submissions/` : fichiers à soumettre ; `reports/` : rapports",
        "- `JOURNAL.md` : progression et problèmes rencontrés",
    ]
    if actions:
        lines += ["", "## À faire par l'utilisateur", ""] + [f"- {a['kind']} : {a['message'].splitlines()[0]}" for a in actions]
    (project.path / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
