"""Lessons register, shared by every project: what was tried, in which context, did it work, and why.

The register grows with every project, so nobody reads it whole. Each lesson carries:
- the step of the plan it belongs to (docs/PLAN.md: comprendre … capitaliser);
- its kind: technique, erreur (a mistake made), reussite (something that worked well), piege (a trap);
- its outcome for a technique: marche, echoue, neutre;
- why (mandatory): the explanation that lets another agent judge whether it transfers;
- its context: modality, problem type, metric, platform, size ("*" = any);
- its evidence (projects, experiments, measured Δ) and how many times it was seen.

ML Forge records the measured outcome of every compared experiment by itself; the agent records
everything else (lesson_record). Retrieval (lessons_search) ranks lessons by closeness of context
and only returns a few. The `progress/` files are a rendering of this register for humans and for
agents without the MCP tools: a short index, then one file per modality, split by step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from forge.db import now_iso, session, transaction
from forge.errors import ForgeError
from forge.workspace import Workspace

STEPS = ("comprendre", "acquerir", "organiser", "formaliser", "calculer", "valider", "experimenter", "livrer",
         "capitaliser")
STEP_TITLES = {"comprendre": "1. Comprendre", "acquerir": "2. Acquérir", "organiser": "3. Organiser",
               "formaliser": "4. Formaliser", "calculer": "5. Calculer", "valider": "6. Valider",
               "experimenter": "7. Expérimenter", "livrer": "8. Livrer", "capitaliser": "9. Capitaliser"}
KINDS = ("technique", "erreur", "reussite", "piege")
OUTCOMES = ("marche", "echoue", "neutre")
CONTEXT_FIELDS = ("modality", "problem_type", "metric", "platform", "size")
WEIGHTS = {"modality": 4, "problem_type": 3, "metric": 1, "platform": 2, "size": 1}
QUERY_WEIGHT = 12  # above the whole context weight (11)
SCOPES = ("projet", "modalite", "general")
MARKS = {"marche": "✅", "echoue": "❌", "neutre": "➖"}
KIND_MARKS = {"erreur": "⚠", "piege": "🪤", "reussite": "⭐"}
PROGRESS_DIR = "progress"
SEED_FILE = Path(__file__).with_name("seed_lessons.json")
SEED_SOURCE = "ml-forge (fondation)"
_ALIASES = {"acquérir": "acquerir", "expérimenter": "experimenter", "réussite": "reussite", "piège": "piege",
            "échoue": "echoue", "modalité": "modalite", "général": "general", "baseline": "experimenter",
            "profiler les données": "formaliser", "formaliser et valider": "valider", "création": "organiser"}


class LessonError(ForgeError):
    """A lesson is incomplete or inconsistent."""


def _norm(value: str) -> str:
    value = str(value).strip().lower()
    return _ALIASES.get(value, value)


def _key_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


# --- context --------------------------------------------------------------------------------

def size_bucket(rows: int | None) -> str:
    if not rows:
        return "*"
    return "<1k" if rows < 1_000 else "1k-10k" if rows < 10_000 else "10k-100k" if rows < 100_000 else ">100k"


def project_context(project) -> dict:
    """What characterises a project for the lessons: known as early as possible, '*' otherwise."""
    from forge.intake import competition_card, fact_value

    card = competition_card(project)
    context = {"modality": "*", "problem_type": fact_value(card, "problem_type") or "*",
               "metric": fact_value(card, "metric") or "*", "platform": card.get("platform") or "*", "size": "*"}
    try:
        from forge.runner import dataset_hash, latest_profile, latest_protocol

        protocol = latest_protocol(project)
    except Exception:  # a project without protocol yet
        protocol = None
    if protocol is not None:
        task = protocol.task
        context["problem_type"], context["metric"] = task.problem_type, task.metric
        context["modality"] = modality_of(task.media_dirs)
        profile = latest_profile(project, dataset_hash(project, task)) or {}
        context["size"] = size_bucket(profile.get("rows"))
    return context


def modality_of(media_dirs) -> str:
    """tabulaire, audio, image, tableaux or multimodal, from a sample of the media files."""
    from itertools import islice

    from forge.media import ARRAY_SUFFIXES, AUDIO_SUFFIXES, IMAGE_SUFFIXES

    if not media_dirs:
        return "tabulaire"
    kinds = set()
    for directory in media_dirs:
        for path in islice(Path(directory).rglob("*"), 2000):
            suffix = path.suffix.lower()
            kinds |= {"audio"} if suffix in AUDIO_SUFFIXES else {"image"} if suffix in IMAGE_SUFFIXES else \
                {"tableaux"} if suffix in ARRAY_SUFFIXES else set()
    return "multimodal" if len(kinds) > 1 else (kinds.pop() if kinds else "tabulaire")


def scoped(context: dict, scope: str) -> dict:
    """Widen a context: the whole project, its modality and problem type only, or any project."""
    scope = _norm(scope)
    if scope not in SCOPES:
        raise LessonError(f"Portée inconnue : {scope} (attendue : {', '.join(SCOPES)}).")
    if scope == "general":
        return {field: "*" for field in CONTEXT_FIELDS}
    if scope == "modalite":
        return {field: context.get(field, "*") if field in {"modality", "problem_type"} else "*"
                for field in CONTEXT_FIELDS}
    return {field: context.get(field, "*") for field in CONTEXT_FIELDS}


def context_key(context: dict) -> str:
    return "|".join(str(context.get(field, "*")) for field in CONTEXT_FIELDS)


# --- recording ---------------------------------------------------------------------------------

def record(workspace: Workspace, step: str, kind: str, title: str, why: str, context: dict,
           outcome: str | None = None, source: str = "agent", project_id: str | None = None,
           experiment_id: str | None = None, delta: float | None = None, render: bool = True) -> dict:
    """Add a lesson, or strengthen the same one (same step, title and context) with new evidence."""
    if source != SEED_SOURCE:
        _seed(workspace)
    step, kind = _norm(step), _norm(kind)
    outcome = _norm(outcome) if outcome else ("marche" if kind == "reussite" else "echoue" if kind == "erreur" else "neutre")
    if step not in STEPS:
        raise LessonError(f"Étape inconnue : {step} (attendue : {', '.join(STEPS)}).")
    if kind not in KINDS:
        raise LessonError(f"Type de leçon inconnu : {kind} (attendu : {', '.join(KINDS)}).")
    if outcome not in OUTCOMES:
        raise LessonError(f"Résultat inconnu : {outcome} (attendu : {', '.join(OUTCOMES)}).")
    title, why = " ".join(str(title).split()), " ".join(str(why).split())
    if not 5 <= len(title) <= 160:
        raise LessonError("Le titre doit faire entre 5 et 160 caractères : une phrase qui nomme la technique ou l'erreur.")
    if len(why) < 15:
        raise LessonError("« why » est obligatoire : explique pourquoi cela marche ou échoue, pour qu'un autre agent "
                          "puisse juger si la leçon s'applique à son cas.")
    context = {field: str(context.get(field) or "*") for field in CONTEXT_FIELDS}
    key, title_key = context_key(context), _key_title(title)
    evidence = {"project": project_id, "experiment": experiment_id, "delta": delta, "at": now_iso()[:10]}
    with session(workspace.global_db) as conn, transaction(conn):
        row = conn.execute("SELECT * FROM lessons WHERE step = ? AND title_key = ? AND context_key = ?",
                           (step, title_key, key)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO lessons (step, kind, outcome, title, title_key, why, context_key, context_json, source,"
                " n_seen, projects_json, evidence_json, mean_delta, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
                (step, kind, outcome, title, title_key, why, key, json.dumps(context, ensure_ascii=False), source,
                 json.dumps([project_id] if project_id else []), json.dumps([evidence]), delta, now_iso(), now_iso()),
            )
        else:
            evidences = json.loads(row["evidence_json"])
            if experiment_id and any(e.get("experiment") == experiment_id and e.get("project") == project_id
                                     for e in evidences):
                return _lesson(row)  # the same evidence is never counted twice
            evidences = (evidences + [evidence])[-50:]
            projects = sorted(set(json.loads(row["projects_json"])) | ({project_id} if project_id else set()))
            deltas = [e["delta"] for e in evidences if e.get("delta") is not None]
            outcomes = [row["outcome"], outcome]
            conn.execute(
                "UPDATE lessons SET n_seen = n_seen + 1, projects_json = ?, evidence_json = ?, mean_delta = ?,"
                " outcome = ?, why = ?, updated_at = ? WHERE id = ?",
                (json.dumps(projects), json.dumps(evidences), sum(deltas) / len(deltas) if deltas else None,
                 outcome if len(set(outcomes)) == 1 else "neutre", why if len(why) > len(row["why"]) else row["why"],
                 now_iso(), row["id"]),
            )
        saved = conn.execute("SELECT * FROM lessons WHERE step = ? AND title_key = ? AND context_key = ?",
                             (step, title_key, key)).fetchone()
    if render:
        render_progress(workspace)
    return _lesson(saved)


def record_from_decision(workspace: Workspace, project, experiment_id: str, status: str, delta: float | None) -> None:
    """ML Forge's own lessons: the measured outcome of each compared experiment, in its context."""
    if status not in {"accepted", "rejected", "not_significant"} or delta is None:
        return
    from forge.experiments import get_experiment
    from forge.knowledge import family_of

    experiment = get_experiment(project.db_path, experiment_id)
    if experiment.card.get("role") in {"baseline", "stability"}:
        return
    outcome = {"accepted": "marche", "rejected": "echoue", "not_significant": "neutre"}[status]
    verdict = {"accepted": "gain accepté par la règle calculée", "rejected": "moins bon que la référence",
               "not_significant": "écart dans le bruit entre plis"}[status]
    record(workspace, "experimenter", "technique", f"{family_of(experiment.card)} — {experiment.method}"[:160],
           f"{verdict} (Δ {delta:+.5f}). Hypothèse : {experiment.hypothesis}", project_context(project),
           outcome, "ml-forge", project.id, experiment_id, delta)


# --- retrieval ---------------------------------------------------------------------------------

def search(workspace: Workspace, context: dict, step: str | None = None, query: str | None = None,
           kind: str | None = None, limit: int = 8) -> list[dict]:
    """The few lessons closest to this context. A lesson for another modality or problem type is
    never returned; '*' (any) always matches; ties are broken by how often a lesson was seen."""
    _seed(workspace)
    with session(workspace.global_db) as conn:
        rows = conn.execute("SELECT * FROM lessons").fetchall()
    return rank(rows, context, step, query, kind, limit)


def rank(rows, context: dict, step: str | None = None, query: str | None = None, kind: str | None = None,
         limit: int = 8) -> list[dict]:
    """Ranking of `search`, on rows already read (the read-only UI reads them itself)."""
    step, kind = (_norm(step) if step else None), (_norm(kind) if kind else None)
    words = set(_key_title(query).split()) if query else set()
    scored = []
    for row in rows:
        if (step and row["step"] != step) or (kind and row["kind"] != kind):
            continue
        lesson_context = json.loads(row["context_json"])
        score, excluded = 0.0, False
        for field in CONTEXT_FIELDS:
            mine, theirs = str(context.get(field, "*")), lesson_context.get(field, "*")
            if theirs == "*" or mine == "*":
                score += WEIGHTS[field] * 0.25
            elif theirs == mine:
                score += WEIGHTS[field]
            elif field in {"modality", "problem_type"}:
                excluded = True  # a lesson on audio regression says little about tabular classification
        if excluded:
            continue
        text = set(_key_title(row["title"] + " " + row["why"]).split())
        score += QUERY_WEIGHT * len(words & text)  # an explicit question outweighs a close context
        rank = {"marche": 2, "neutre": 1, "echoue": 0}[row["outcome"]]
        scored.append((score, row["n_seen"], rank, row["updated_at"], row))
    scored.sort(key=lambda item: item[:4], reverse=True)  # closest, most confirmed, what works, newest
    return [_lesson(row) | {"pertinence": round(score, 2)} for score, _, _, _, row in scored[:limit]]


def _lesson(row) -> dict:
    context = json.loads(row["context_json"])
    return {
        "id": row["id"], "étape": row["step"], "type": row["kind"], "résultat": row["outcome"], "titre": row["title"],
        "pourquoi": row["why"], "contexte": {k: v for k, v in context.items() if v != "*"} or "général",
        "vu": row["n_seen"], "projets": len(json.loads(row["projects_json"])), "delta_moyen": row["mean_delta"],
        "source": row["source"],
    }


def brief(lessons: list[dict]) -> str:
    """Compact text for an agent's context window."""
    if not lessons:
        return "Aucune leçon enregistrée pour ce contexte."
    lines = []
    for lesson in lessons:
        mark = KIND_MARKS.get(lesson["type"]) or MARKS[lesson["résultat"]]
        context = lesson["contexte"] if isinstance(lesson["contexte"], str) else ", ".join(lesson["contexte"].values())
        delta = f", Δ moyen {lesson['delta_moyen']:+.4f}" if lesson["delta_moyen"] is not None else ""
        lines.append(f"{mark} [{lesson['étape']}] {lesson['titre']} — {lesson['pourquoi']} "
                     f"(contexte : {context} ; vu {lesson['vu']} fois{delta})")
    return "\n".join(lines)


# --- rendering: progress/ ------------------------------------------------------------------------

def progress_dir(workspace: Workspace) -> Path:
    return workspace.data_dir.parent / PROGRESS_DIR


def render_progress(workspace: Workspace) -> Path:
    """INDEX.md (short: what exists, where) and one file per modality, split by step."""
    with session(workspace.global_db) as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM lessons ORDER BY step, n_seen DESC, updated_at DESC")]
    folder = progress_dir(workspace)
    folder.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        modality = json.loads(row["context_json"]).get("modality", "*")
        groups.setdefault("general" if modality == "*" else modality, []).append(row)
    index = [
        "# Progression — techniques qui marchent, qui échouent, erreurs et pièges", "",
        "**Ne lisez pas tout.** Lisez ce fichier, puis seulement le fichier de votre modalité et `general.md`, et dans",
        "chacun seulement la section de l'étape en cours. Avec ML Forge : `lessons_search` (MCP) ou",
        "`python -m forge lessons search <projet> --step <étape>` renvoient les leçons les plus proches de votre contexte.", "",
        "Généré par ML Forge à partir du registre des leçons : ne pas modifier à la main",
        "(`lesson_record` ou `python -m forge lessons add …`).", "",
        "| Fichier | Leçons | ✅ marche | ❌ échoue | ⚠ erreurs et pièges |", "|---|---|---|---|---|",
    ]
    for name in sorted(groups, key=lambda n: (n != "general", n)):
        items = groups[name]
        index.append(f"| [{name}.md]({name}.md) | {len(items)} | {sum(i['outcome'] == 'marche' and i['kind'] != 'erreur' for i in items)} | "
                     f"{sum(i['outcome'] == 'echoue' and i['kind'] == 'technique' for i in items)} | "
                     f"{sum(i['kind'] in {'erreur', 'piege'} for i in items)} |")
        (folder / f"{name}.md").write_text(_render_group(name, items), encoding="utf-8")
    (folder / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return folder


def _render_group(name: str, items: list[dict]) -> str:
    title = "Leçons générales (tout projet)" if name == "general" else f"Leçons — modalité {name}"
    lines = [f"# {title}", "", "Une section par étape du plan ; dans chacune, les leçons les plus confirmées d'abord.", ""]
    for step in STEPS:
        selected = [item for item in items if item["step"] == step]
        if not selected:
            continue
        lines += [f"## {STEP_TITLES[step]}", ""]
        for item in selected:
            context = {k: v for k, v in json.loads(item["context_json"]).items() if v != "*" and k != "modality"}
            mark = KIND_MARKS.get(item["kind"]) or MARKS[item["outcome"]]
            delta = f" ; Δ moyen {item['mean_delta']:+.4f}" if item["mean_delta"] is not None else ""
            where = ", ".join(f"{k} {v}" for k, v in context.items()) or "tout contexte"
            lines.append(f"- {mark} **{item['title']}** — {item['why']}  \n"
                         f"  _{item['kind']} · {where} · vu {item['n_seen']} fois dans "
                         f"{len(json.loads(item['projects_json']))} projet(s){delta} · {item['source']}_")
        lines.append("")
    return "\n".join(lines)


# --- lessons shipped with ML Forge -----------------------------------------------------------------

def _seed(workspace: Workspace) -> None:
    """Load once the lessons learnt while building ML Forge (docs/JOURNAL_V3.md)."""
    with session(workspace.global_db) as conn:
        if conn.execute("SELECT 1 FROM lessons WHERE source = ? LIMIT 1", (SEED_SOURCE,)).fetchone():
            return
    if not SEED_FILE.is_file():
        return
    for item in json.loads(SEED_FILE.read_text(encoding="utf-8")):
        context = {field: item.get("context", {}).get(field, "*") for field in CONTEXT_FIELDS}
        record(workspace, item["step"], item["kind"], item["title"], item["why"], context, item.get("outcome"),
               SEED_SOURCE, render=False)
