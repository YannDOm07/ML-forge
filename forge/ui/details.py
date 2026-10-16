"""Details of the v3 screens: every project at a glance, the competition, data at scale, files,
the project journal and the lessons of its context.

Display only: databases are read through `queries` (read-only), files are listed, never opened
for writing. Large collections are summarised here — a project can hold tens of thousands of
media files, and the page is refreshed every few seconds.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from forge.ui import queries

JOURNAL_ENTRY = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) — (.+)$", re.MULTILINE)
REPOSITORY_TITLE = re.compile(r"^(Dépôt|Enregistrement) ")
TOP_FOLDERS = 15
FILE_LIMIT = 200


def _json(value: str | None) -> Any:
    try:
        return json.loads(value) if value else None
    except (TypeError, ValueError):
        return None


def _platform(url: str | None) -> str:
    from forge.intake import identify_platform

    return identify_platform(url).name if url else "—"


# --- every project -----------------------------------------------------------------------------

def portfolio(workspace, projects: list[dict]) -> list[dict]:
    """One line per project: where it stands, its best score, what waits for the user."""
    items = []
    for project in projects:
        folder = workspace.projects_dir / project["id"]
        database = folder / "project.db"
        if not database.is_file():
            items.append({**project, "platform": _platform(project.get("competition_url")), "missing": True})
            continue
        decisions = queries.get_experiment_decisions(database)
        kept = [d for d in decisions if d["status"] in ("baseline", "accepted")]
        protocol = queries.get_latest_protocol(database)
        task = (_json(protocol["spec_json"]) or {}).get("task", {}) if protocol else {}
        campaigns = queries.get_campaigns(database)
        actions = [a for a in queries.get_user_actions(database) if a["status"] == "open"]
        entries = project_journal(folder, limit=1)
        items.append({
            "id": project["id"], "name": project["name"], "url": project.get("competition_url"),
            "status": project["status"], "created_at": project["created_at"],
            "platform": _platform(project.get("competition_url")),
            "problem_type": task.get("problem_type"), "metric": task.get("metric"),
            "experiments": len(queries.get_experiments(database)),
            "best": ({"experiment_id": kept[-1]["experiment_id"], "score": kept[-1]["metric_mean"],
                      "status": kept[-1]["status"]} if kept else None),
            "campaign": campaigns[0]["status"] if campaigns else None,
            "open_actions": len(actions),
            "first_action": actions[0]["message"].splitlines()[0][:200] if actions else None,
            "last_step": entries[0] if entries else None,
            "problems": sum(1 for entry in project_journal(folder, limit=500) if entry["problem"]),
            "deliverables": (folder / "deliverables").is_dir() and any((folder / "deliverables").iterdir()),
        })
    return items


# --- the competition -----------------------------------------------------------------------------

def competition(project_db: Path) -> dict:
    """Cited facts, pages read and repositories described during the intake."""
    facts = []
    for fact in queries.get_competition_facts(project_db):
        local = str(fact.get("source_url", "")).startswith("file:")
        facts.append({
            # A fact measured on the data files may reveal raw values: never shown.
            "field": fact["field"], "value": "donnée locale masquée" if local else _json(fact["value_json"]),
            "status": fact["status"],
            "source": None if local else fact.get("source_url"),
            "quote": None if local else (fact.get("quote") or "")[:300],
            "by": fact.get("recorded_by"),
        })
    pages, repositories = [], []
    for page in queries.get_page_index(project_db):
        if page["origin"] == "data":
            continue
        flags = _json(page["flags_json"]) or []
        entry = {"url": page["url"], "title": page["title"] or page["url"], "origin": page["origin"],
                 "status": page["status"], "chars": page["chars"], "flags": len(flags), "at": page["fetched_at"]}
        if page["origin"] == "api" and REPOSITORY_TITLE.match(page["title"] or ""):
            head = page["head"] or ""
            files = re.search(r"Fichiers : (\d+), taille totale ([^\n]+)", head)
            gated = re.search(r'"gated": "?(\w+)"?|"access_right": "(\w+)"', head)
            address = re.search(r"Adresse : (\S+)", head)
            repositories.append({**entry, "address": address.group(1) if address else page["url"],
                                 "files": int(files.group(1)) if files else None,
                                 "size": files.group(2).strip() if files else None,
                                 "access": next((g for g in gated.groups() if g), None) if gated else None})
        else:
            pages.append(entry)
    return {"facts": facts, "pages": pages, "repositories": repositories}


# --- data at scale -----------------------------------------------------------------------------------

_SUMMARY_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}
_MODALITY_CACHE: dict[tuple, str] = {}


def data_summary(project_db: Path, task: dict) -> dict:
    """Totals by role, type and folder; the task's tables, sets and media folders."""
    sets = task.get("prediction_sets") or ({"test": task["test_path"]} if task.get("test_path") else {})
    return {**_file_totals(project_db), "task": {
        "train": Path(task["train_path"]).name if task.get("train_path") else None,
        "sets": {name: Path(path).name for name, path in sets.items()},
        "id_column": task.get("id_column"), "group_column": task.get("group_column"),
        "target": task.get("target"), "train_only_columns": task.get("train_only_columns") or [],
        "media_dirs": [str(Path(d).name) for d in task.get("media_dirs") or []],
    }}


def _file_totals(project_db: Path) -> dict:
    """Aggregates of the registered files, recomputed only when a file was registered (cache)."""
    version = queries.get_data_source_version(project_db)
    cached = _SUMMARY_CACHE.get(str(project_db))
    if cached and cached[0] == version:
        return cached[1]
    by_role: Counter = Counter()
    sizes: Counter = Counter()
    by_type: Counter = Counter()
    folders: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for role, path, size in queries.get_data_source_rows(project_db):
        cut = max(path.rfind("\\"), path.rfind("/"))  # string operations: 50 000 paths stay instant
        folder, name = path[:cut], path[cut + 1:]
        dot = name.rfind(".")
        by_role[role] += 1
        sizes[role] += size or 0
        by_type[name[dot:].lower() if dot > 0 else "(sans extension)"] += 1
        folders[folder][0] += 1
        folders[folder][1] += size or 0
    top = sorted(folders.items(), key=lambda item: -item[1][0])[:TOP_FOLDERS]
    totals = {
        "files": sum(by_role.values()), "bytes": sum(sizes.values()),
        "roles": [{"role": role, "files": count, "bytes": sizes[role]} for role, count in by_role.most_common()],
        "types": [{"type": kind, "files": count} for kind, count in by_type.most_common(12)],
        "folders": [{"folder": name, "files": count, "bytes": size} for name, (count, size) in top],
        "more_folders": max(0, len(folders) - TOP_FOLDERS),
    }
    _SUMMARY_CACHE[str(project_db)] = (version, totals)
    return totals


# --- files of the project ----------------------------------------------------------------------------

def project_files(folder: Path) -> dict:
    """Deliverables (recursively), notebooks, derived tables and the project's documents."""
    def listing(root: Path, pattern: str = "*") -> list[dict]:
        if not root.is_dir():
            return []
        found = sorted(p for p in root.rglob(pattern) if p.is_file() and not p.name.startswith(".")
                       and "__pycache__" not in p.parts)[:FILE_LIMIT]
        return [{"path": str(p.relative_to(root)).replace("\\", "/"), "bytes": p.stat().st_size,
                 "modified": p.stat().st_mtime} for p in found]

    return {
        "folder": str(folder),
        "deliverables": listing(folder / "deliverables"),
        "notebooks": listing(folder / "notebooks", "*.ipynb"),
        "derived": listing(folder / "data" / "derived"),
        "documents": [name for name in ("README.md", "JOURNAL.md", "report.md") if (folder / name).is_file()],
    }


# --- the project journal ------------------------------------------------------------------------------

def project_journal(folder: Path, limit: int = 80) -> list[dict]:
    """Entries of JOURNAL.md, newest first: date, step, text, problem or not."""
    path = folder / "JOURNAL.md"
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    marks = list(JOURNAL_ENTRY.finditer(text))
    entries = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        body = text[mark.end():end].strip()
        entries.append({"at": mark.group(1), "step": mark.group(2).strip(), "text": body[:1500],
                        "problem": "⚠" in body})
    return entries[::-1][:limit]


# --- lessons of the project's context -------------------------------------------------------------------

def lessons(workspace, task: dict, facts: dict, url: str | None, rows: int | None) -> dict:
    """The lessons closest to the project (never the whole register), and the register's size."""
    from forge.lessons import CONTEXT_FIELDS, STEPS, modality_of, progress_dir, rank, size_bucket

    media = tuple(task.get("media_dirs") or ()) if task else None
    if media is not None and media not in _MODALITY_CACHE:
        _MODALITY_CACHE[media] = modality_of(media)  # sampling the media folders once per task
    context = {
        "modality": _MODALITY_CACHE[media] if media is not None else "*",
        "problem_type": task.get("problem_type") or facts.get("problem_type") or "*",
        "metric": task.get("metric") or facts.get("metric") or "*",
        "platform": _platform(url) if url else "*", "size": size_bucket(rows),
    }
    register = queries.get_lessons(workspace.global_db)
    by_step = {step: rank(register, context, step=step, limit=6) for step in STEPS}
    counts = Counter(row["kind"] for row in register)
    chronology = progress_dir(workspace) / "chronologie.md"
    recent = []
    if chronology.is_file():
        recent = [line for line in chronology.read_text(encoding="utf-8", errors="replace").splitlines()
                  if line.startswith("| 20")][-25:][::-1]
    return {
        "context": {field: context[field] for field in CONTEXT_FIELDS},
        "by_step": {step: items for step, items in by_step.items() if items},
        "register": {"total": len(register), **{kind: counts.get(kind, 0) for kind in
                                                ("technique", "erreur", "reussite", "piege")}},
        "progress_dir": str(progress_dir(workspace)),
        "chronology": recent,
    }
