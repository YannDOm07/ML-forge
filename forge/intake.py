"""Competition intake: read the pages, record cited facts, compute what is still unknown.

A fact is recorded only with a quote that ML Forge finds verbatim in a page it has read
(or in text the user pasted, or in a data file). Nothing is guessed: a heuristic match is
marked 'inferred' and does not count for the critical fields until confirmed.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from forge.actions import open_action, resolve_actions
from forge.db import now_iso, session, transaction
from forge.errors import ForgeError
from forge.experiments import log_event
from forge.kaggle import KaggleClient, files_url, load_credentials, metadata_url
from forge.projects import Project
from forge.task import PROBLEM_TYPES, TaskSpec
from forge.web import NetworkError, Web, WebPage
from forge.workspace import Workspace

MAX_PAGES = 15
RULE_CHOICES = ("allowed", "forbidden", "conditional", "unknown")
STATUS_RANK = {"inferred": 0, "found": 1, "user": 2}
FOLLOW_KEYWORDS = (
    "rule", "data", "evaluat", "overview", "faq", "timeline", "submission", "github", "starter",
    "guideline", "regle", "règle", "donnee", "donnée", "evaluation", "évaluation",
)
DATA_EXTENSIONS = (".csv", ".zip", ".parquet", ".json", ".tsv", ".txt", ".gz", ".npy", ".npz", ".feather")
FOLLOW_HOSTS = ("github.com", "raw.githubusercontent.com")


class IntakeError(ForgeError):
    """A fact cannot be recorded, or the competition cannot be read."""


class IntakeIncomplete(ForgeError):
    """Critical information is missing: the campaign must not start."""


@dataclass(frozen=True)
class Field:
    label: str
    kind: str  # text | int | choice | list
    choices: tuple[str, ...] | None = None
    critical: bool = False


FIELDS: dict[str, Field] = {
    "title": Field("Titre", "text"),
    "competition_type": Field("Type de compétition", "choice", ("file_submission", "code_submission")),
    "problem_type": Field(
        "Type de problème", "choice",
        PROBLEM_TYPES + ("multilabel", "ranking", "segmentation", "detection", "forecasting", "other"), True,
    ),
    "target": Field("Colonne cible", "text", critical=True),
    "id_column": Field("Colonne identifiant", "text"),
    "metric": Field("Métrique officielle", "text", critical=True),
    "submission_format": Field("Format de soumission (fichier exemple)", "text", critical=True),
    "deliverables": Field("Livrables exigés", "list"),
    "rules.external_data": Field("Données externes", "choice", RULE_CHOICES),
    "rules.pretrained_models": Field("Modèles préentraînés", "choice", RULE_CHOICES),
    "rules.data_sharing": Field("Partage des données avec des tiers", "choice", ("allowed", "forbidden", "unknown")),
    "rules.daily_submission_limit": Field("Soumissions par jour", "int"),
    "rules.final_selection_count": Field("Soumissions finales retenues", "int"),
    "rules.compute_limits": Field("Limites de calcul", "text"),
    "input_restrictions": Field("Restrictions d'usage des données", "list"),
    "deadline": Field("Échéance finale", "text"),
    "public_extra_info": Field("Informations publiques supplémentaires", "list"),
}


# --- platform ----------------------------------------------------------------------

@dataclass(frozen=True)
class Platform:
    name: str  # kaggle | zindi | itu | generic
    slug: str | None


def identify_platform(url: str) -> Platform:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if host == "kaggle.com" and len(parts) >= 2 and parts[0] in {"competitions", "c"}:
        return Platform("kaggle", parts[1])
    if host.endswith("zindi.africa") and "competitions" in parts:
        index = parts.index("competitions")
        return Platform("zindi", parts[index + 1] if index + 1 < len(parts) else None)
    if host.endswith("itu.int"):
        return Platform("itu", parts[-1] if parts else None)
    return Platform("generic", parts[-1] if parts else None)


# --- pages and facts ------------------------------------------------------------------

def store_page(project: Project, url: str, origin: str, text: str, title: str = "", status: int = 200,
               links: list | tuple = (), flags: list | tuple = ()) -> None:
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO web_pages (url, origin, status, title, text, links_json, flags_json, content_hash, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (url, origin, status, title, text, json.dumps(list(links), ensure_ascii=False),
             json.dumps(list(flags), ensure_ascii=False), hashlib.sha256(text.encode()).hexdigest(), now_iso()),
        )


def latest_page(project: Project, url: str) -> dict | None:
    with session(project.db_path) as conn:
        row = conn.execute("SELECT * FROM web_pages WHERE url = ? ORDER BY id DESC LIMIT 1", (url,)).fetchone()
    return dict(row) if row else None


def stored_pages(project: Project) -> list[dict]:
    with session(project.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM web_pages WHERE id IN (SELECT MAX(id) FROM web_pages GROUP BY url) ORDER BY id"
        ).fetchall()
    return [dict(row) for row in rows]


def paste_text(project: Project, text: str) -> str:
    """Store text the user copied from a page ML Forge could not read; facts may then cite it."""
    if not text.strip():
        raise IntakeError("Le texte collé est vide.")
    with session(project.db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM web_pages WHERE origin = 'user'").fetchone()[0]
    url = f"user://texte-colle/{count + 1}"
    store_page(project, url, "user", text, title="Texte collé par l'utilisateur")
    return url


def _normalise(text: str) -> str:
    return " ".join(text.replace("’", "'").replace(" ", " ").lower().split())


def record_fact(project: Project, name: str, value, source_url: str, quote: str, recorded_by: str,
                inferred: bool = False) -> dict:
    """Record a fact only if `quote` appears verbatim in the stored source."""
    spec = FIELDS.get(name)
    if spec is None:
        raise IntakeError(f"Champ inconnu : {name} (disponibles : {', '.join(FIELDS)})")
    value = _validate(name, spec, value)
    page = latest_page(project, source_url)
    if page is None:
        raise IntakeError(f"Source non lue : {source_url}. Lisez la page ou collez son texte avant de la citer.")
    if not quote.strip() or _normalise(quote) not in _normalise(page["text"]):
        raise IntakeError(f"Citation introuvable dans {source_url} : le fait « {name} » n'est pas enregistré.")
    status = "inferred" if inferred else ("user" if page["origin"] == "user" else "found")
    with session(project.db_path) as conn, transaction(conn):
        duplicate = conn.execute(
            "SELECT 1 FROM competition_facts WHERE field = ? AND value_json = ? AND source_url = ? AND status = ?",
            (name, json.dumps(value, ensure_ascii=False), source_url, status),
        ).fetchone()
        if not duplicate:
            conn.execute(
                "INSERT INTO competition_facts (field, value_json, status, source_url, quote, recorded_by, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, json.dumps(value, ensure_ascii=False), status, source_url, quote.strip(), recorded_by, now_iso()),
            )
    return {"field": name, "value": value, "status": status, "source_url": source_url, "quote": quote.strip()}


def _validate(name: str, spec: Field, value):
    if spec.kind == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            raise IntakeError(f"{name} : nombre entier attendu, « {value} » reçu.") from None
    value = str(value).strip()
    if not value:
        raise IntakeError(f"{name} : valeur vide.")
    if spec.choices and value not in spec.choices:
        raise IntakeError(f"{name} : « {value} » n'est pas parmi {', '.join(spec.choices)}.")
    return value


# --- the competition card (annex A.1) -------------------------------------------------------

def competition_card(project: Project) -> dict:
    with session(project.db_path) as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM competition_facts ORDER BY id")]
    fields: dict[str, object] = {}
    for name, spec in FIELDS.items():
        facts = [_fact(row) for row in rows if row["field"] == name]
        if spec.kind == "list":
            fields[name] = facts
        else:
            fields[name] = max(facts, key=lambda f: (STATUS_RANK[f["status"]], f["id"])) if facts else None
    unknown = [n for n, s in FIELDS.items() if s.critical and (fields[n] is None or fields[n]["status"] == "inferred")]
    platform = identify_platform(project.competition_url) if project.competition_url else Platform("generic", None)
    return {
        "url": project.competition_url,
        "platform": platform.name,
        "slug": platform.slug,
        "fields": fields,
        "critical_unknowns": unknown,
        "support_level": _support_level(fields),
    }


def _fact(row: dict) -> dict:
    return {key: row[key] for key in ("id", "status", "source_url", "quote")} | {"value": json.loads(row["value_json"])}


def _support_level(fields: dict) -> str:
    problem = fields["problem_type"]
    if problem and problem["value"] in PROBLEM_TYPES:
        return "validated"
    return "experimental" if problem else "unknown"


def fact_value(card: dict, name: str, default=None):
    fact = card["fields"].get(name)
    return fact["value"] if isinstance(fact, dict) else default


def questions(card: dict) -> list[str]:
    asked = []
    for name in card["critical_unknowns"]:
        fact = card["fields"][name]
        label = FIELDS[name].label
        if fact is None:
            asked.append(f"{label} : introuvable. Collez le passage de la page qui le précise (commande paste).")
        else:
            asked.append(f"{label} : « {fact['value']} » déduit de « {fact['quote'][:120]} » ; confirmez ou corrigez.")
    return asked


def assert_ready(project: Project) -> dict:
    card = competition_card(project)
    if card["critical_unknowns"]:
        raise IntakeIncomplete(
            "La campagne ne peut pas démarrer : informations critiques manquantes.\n- " + "\n- ".join(questions(card))
        )
    return card


# --- reading a competition ----------------------------------------------------------------

@dataclass
class IntakeReport:
    platform: Platform
    pages: list[str] = field(default_factory=list)
    flags: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    card: dict = field(default_factory=dict)


def read_competition(workspace: Workspace, project: Project, url: str | None = None, transport=None) -> IntakeReport:
    url = url or project.competition_url
    if not url:
        raise IntakeError("Aucun lien de compétition pour ce projet.")
    platform = identify_platform(url)
    web = Web(workspace, transport, project.id)
    report = IntakeReport(platform)
    for page in _crawl(web, url, workspace.config.get("acquisition", "link_depth"), report.notes):
        store_page(project, page.url, "web", page.text, page.title, page.status, page.links, page.flags)
        report.pages.append(page.url)
        for sentence in page.flags:
            report.flags.append((page.url, sentence))
            log_event(project.db_path, None, "injection", f"Instruction ignorée dans {page.url} : {sentence}")
    if report.pages and len(latest_page(project, report.pages[0])["text"]) < 300:
        report.notes.append("Page principale presque vide (JavaScript ou connexion requise) : collez le texte des règles.")
    if platform.name == "kaggle":
        _read_kaggle(project, web, platform.slug, report.notes)
    log_event(project.db_path, None, "intake", f"{len(report.pages)} page(s) lue(s) depuis {url}")
    extract_facts(project)
    report.card = competition_card(project)
    _ask_user(project, report.card)
    return report


def _crawl(web: Web, start: str, depth: int, notes: list[str]) -> list[WebPage]:
    host = urlparse(start).netloc.lower()
    queue, seen, pages = [(start, 0)], set(), []
    while queue and len(pages) < MAX_PAGES:
        url, level = queue.pop(0)
        url = url.split("#")[0]
        if url in seen:
            continue
        seen.add(url)
        try:
            page = web.read(url, "Lecture de la compétition")
        except NetworkError as exc:
            notes.append(str(exc))
            continue
        if not page.ok:
            notes.append(f"{url} : statut HTTP {page.status}.")
            continue
        pages.append(page)
        if level < depth:
            queue += [(link["url"], level + 1) for link in page.links if _follow(link, host)]
    return pages


def _follow(link: dict, host: str) -> bool:
    target = urlparse(link["url"])
    if target.scheme not in {"http", "https"} or target.path.lower().endswith(DATA_EXTENSIONS):
        return False
    target_host = target.netloc.lower()
    related = target_host == host or target_host.endswith("." + host.removeprefix("www.")) or target_host in FOLLOW_HOSTS
    words = (link["url"] + " " + link.get("text", "")).lower()
    return related and any(keyword in words for keyword in FOLLOW_KEYWORDS)


def _read_kaggle(project: Project, web: Web, slug: str, notes: list[str]) -> None:
    credentials = load_credentials()
    if credentials is None:
        notes.append("Jeton Kaggle absent : les métadonnées officielles (métrique, échéance, fichiers) ne sont pas lues.")
        return
    client = KaggleClient(web, credentials)
    status, competition = client.competition(slug)
    if status == 401:
        notes.append("Jeton Kaggle refusé (401) : vérifiez kaggle.json.")
        return
    if competition:
        url = metadata_url(slug)
        store_page(project, url, "api", json.dumps(competition, indent=1, ensure_ascii=False), competition.get("title", ""))
        _kaggle_facts(project, url, competition)
    status, files = client.files(slug)
    if files:
        url = files_url(slug)
        store_page(project, url, "api", json.dumps(files, indent=1, ensure_ascii=False), "Fichiers de la compétition")
        for entry in files:
            name = str(entry.get("name", ""))
            if "sample" in name.lower():
                record_fact(project, "submission_format", name, url, f'"name": {json.dumps(name)}', "kaggle-api")


def _kaggle_facts(project: Project, url: str, competition: dict) -> None:
    def quote(key):
        return f'"{key}": {json.dumps(competition[key], ensure_ascii=False)}'

    if competition.get("title"):
        record_fact(project, "title", competition["title"], url, quote("title"), "kaggle-api")
    if competition.get("evaluationMetric"):
        metrics = _metrics_in(str(competition["evaluationMetric"]))
        value = metrics[0] if len(metrics) == 1 else competition["evaluationMetric"]
        record_fact(project, "metric", value, url, quote("evaluationMetric"), "kaggle-api")
    if competition.get("deadline"):
        record_fact(project, "deadline", competition["deadline"], url, quote("deadline"), "kaggle-api")
    if isinstance(competition.get("maxDailySubmissions"), int):
        record_fact(project, "rules.daily_submission_limit", competition["maxDailySubmissions"], url,
                    quote("maxDailySubmissions"), "kaggle-api")
    if isinstance(competition.get("isKernelsSubmissionsOnly"), bool):
        kind = "code_submission" if competition["isKernelsSubmissionsOnly"] else "file_submission"
        record_fact(project, "competition_type", kind, url, quote("isKernelsSubmissionsOnly"), "kaggle-api")


def _ask_user(project: Project, card: dict) -> None:
    if card["critical_unknowns"]:
        open_action(project, "intake", "Informations critiques à fournir :\n- " + "\n- ".join(questions(card)))
    else:
        resolve_actions(project, "intake")


# --- deterministic extraction from the text -------------------------------------------------

METRIC_PATTERNS = [
    (r"macro[- ]?(averaged )?f1|f1[- ]macro|macro[- ]f[- ]?score", "f1_macro"),
    (r"\brmsle\b|root mean squared? log(arithmic)? error", "rmsle"),
    (r"\brmse\b|root mean squared? error", "rmse"),
    (r"\bmae\b|mean absolute error", "mae"),
    (r"log[- ]?loss|logarithmic loss|cross[- ]entropy", "log_loss"),
    (r"\b(roc[- ])?auc\b|area under (the )?(roc )?curve", "roc_auc"),
    (r"balanced accuracy", "balanced_accuracy"),
    (r"\baccuracy\b|\bexactitude\b", "accuracy"),
    (r"\br2\b|r-squared|coefficient of determination", "r2"),
    (r"\bf1\b|\bf-?score\b", "f1"),
]
EVALUATION_WORDS = re.compile(r"evaluat|metric|scored|évalu|métrique|leaderboard", re.IGNORECASE)
NEGATION = re.compile(r"\bnot\b|\bno\b|prohibit|forbid|n't|interdit|pas autoris|ne (sont|peuvent) pas", re.IGNORECASE)
PERMISSION = re.compile(r"allow|permit|may use|can use|autoris|permis", re.IGNORECASE)
RULE_TOPICS = {
    "rules.external_data": re.compile(r"external data|outside data|external datasets?|données externes", re.IGNORECASE),
    "rules.pretrained_models": re.compile(r"pre-?trained|préentraîn|pré-entraîn", re.IGNORECASE),
    "rules.data_sharing": re.compile(
        r"(shar(e|ing)|distribut|partag)\w*.{0,60}(data|données)|(data|données).{0,60}(shar|distribut|partag)", re.IGNORECASE
    ),
}
DAILY_LIMIT = re.compile(r"(\d+)\s+(?:submissions?|entries)\s+(?:per|a|each)\s+day|(\d+)\s+soumissions?\s+par\s+jour", re.I)
FINAL_SELECTION = re.compile(r"select\s+(?:up\s+to\s+)?(\d+)\s+(?:final\s+)?submissions|(\d+)\s+soumissions?\s+finales", re.I)
DELIVERABLES = {
    "report": re.compile(r"technical report|rapport technique|write-?up", re.IGNORECASE),
    "notebook": re.compile(r"\bnotebook\b", re.IGNORECASE),
    "inference_script": re.compile(r"inference (script|code)|script d'inférence", re.IGNORECASE),
    "training_code": re.compile(r"source code|training code|code source", re.IGNORECASE),
}
DELIVERY_WORDS = re.compile(r"submit|must|required|deliver|provide|soumettre|fournir|doivent|obligatoire", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 3]


def _metrics_in(text: str) -> list[str]:
    for pattern, name in METRIC_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return [name]
    return []


def extract_facts(project: Project) -> None:
    """Deterministic extraction over every stored page; ambiguous matches stay 'inferred'."""
    metric_hits: list[tuple[str, str, str]] = []
    for page in stored_pages(project):
        if page["origin"] == "data":
            continue
        for sentence in _sentences(page["text"]):
            if EVALUATION_WORDS.search(sentence):
                metric_hits += [(name, page["url"], sentence) for name in _metrics_in(sentence)]
            _rule_facts(project, page["url"], sentence)
            _number_facts(project, page["url"], sentence)
            _deliverable_facts(project, page["url"], sentence)
    ambiguous = len({name for name, _, _ in metric_hits}) > 1
    for name, url, sentence in metric_hits:
        record_fact(project, "metric", name, url, sentence, "extraction", inferred=ambiguous)


def _rule_facts(project: Project, url: str, sentence: str) -> None:
    for name, topic in RULE_TOPICS.items():
        if not topic.search(sentence):
            continue
        if NEGATION.search(sentence):
            value = "forbidden"
        elif PERMISSION.search(sentence) and name != "rules.data_sharing":
            value = "allowed"
        else:
            continue
        record_fact(project, name, value, url, sentence, "extraction", inferred=True)


def _number_facts(project: Project, url: str, sentence: str) -> None:
    for name, pattern in (("rules.daily_submission_limit", DAILY_LIMIT), ("rules.final_selection_count", FINAL_SELECTION)):
        match = pattern.search(sentence)
        if match:
            record_fact(project, name, next(g for g in match.groups() if g), url, sentence, "extraction")


def _deliverable_facts(project: Project, url: str, sentence: str) -> None:
    if not DELIVERY_WORDS.search(sentence):
        return
    for name, pattern in DELIVERABLES.items():
        if pattern.search(sentence):
            record_fact(project, "deliverables", name, url, sentence, "extraction", inferred=True)


# --- facts measured on the data files ------------------------------------------------------

def classify_files(paths: list[Path]) -> dict[str, Path]:
    """Recognise train, test and sample-submission files by name."""
    found: dict[str, Path] = {}
    for path in sorted(paths, key=lambda p: len(p.name)):
        name = path.name.lower()
        if not name.endswith(".csv"):
            continue
        if "sample" in name or "submission" in name:
            found.setdefault("sample", path)
        elif "train" in name:
            found.setdefault("train", path)
        elif "test" in name:
            found.setdefault("test", path)
    return found


def infer_from_data(project: Project, paths: list[Path]) -> None:
    """Facts measured on the files themselves: sample columns, target values."""
    files = classify_files(paths)
    if "sample" not in files:
        return
    sample = files["sample"]
    columns = list(pd.read_csv(sample, nrows=5).columns)
    evidence = f"Fichier {sample.name} — colonnes : {', '.join(map(str, columns))}"
    url = sample.resolve().as_uri()
    store_page(project, url, "data", evidence, sample.name)
    record_fact(project, "submission_format", sample.name, url, evidence, "données")
    record_fact(project, "id_column", columns[0], url, evidence, "données")
    if len(columns) != 2 or "train" not in files:
        return
    target = str(columns[1])
    train_columns = list(pd.read_csv(files["train"], nrows=0).columns)
    if target not in train_columns:
        return
    record_fact(project, "target", target, url, evidence, "données")
    _infer_problem_type(project, files["train"], target)


def _infer_problem_type(project: Project, train: Path, target: str) -> None:
    values = pd.read_csv(train, usecols=[target], low_memory=False)[target].dropna()
    distinct = values.nunique()
    numeric = pd.api.types.is_numeric_dtype(values)
    integral = numeric and bool((values == values.round()).all())
    if distinct == 2:
        problem, sure = "binary", True
    elif numeric and not integral:
        problem, sure = "regression", True
    elif distinct <= 50:
        problem, sure = "multiclass", not numeric
    else:
        problem, sure = "regression", False
    shown = ", ".join(map(str, sorted(values.unique().tolist(), key=str)[:10]))
    evidence = f"Cible {target} dans {train.name} : {distinct} valeur(s) distincte(s), type {values.dtype} ; exemples : {shown}"
    url = train.resolve().as_uri() + "#target"
    store_page(project, url, "data", evidence, train.name)
    record_fact(project, "problem_type", problem, url, evidence, "données", inferred=not sure)


def task_from_card(project: Project, paths: list[Path]) -> TaskSpec:
    """Build the task of a competition whose card is complete."""
    card = assert_ready(project)
    files = classify_files(paths)
    if "train" not in files:
        raise IntakeError("Fichier d'entraînement introuvable parmi les données enregistrées.")
    return TaskSpec(
        problem_type=fact_value(card, "problem_type"),
        target=fact_value(card, "target"),
        metric=fact_value(card, "metric"),
        train_path=str(files["train"]),
        test_path=str(files["test"]) if "test" in files else None,
        id_column=fact_value(card, "id_column"),
    )
