"""EvalAI challenges: eval.ai and the many instances it powers (ITU AI for Good, university labs...).

The challenge page is a JavaScript application whose HTML carries no text. The content lives
in a REST API whose address is declared in the front-end configuration, sometimes on another
domain. ML Forge reads that API and renders each challenge and phase as a citable page.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from forge.web import NetworkError, Web, html_to_text

CHALLENGE_PATH = re.compile(r"/web/challenges/challenge-page/(\d+)")
API_IN_CONFIG = re.compile(r"""["']?API["']?\s*:\s*["'](https?://[^"']+)["']""")
TEXT_FIELDS = ("description", "evaluation_details", "submission_guidelines", "terms_and_conditions")
CHALLENGE_FIELDS = (
    "title", "short_description", "start_date", "end_date", "is_registration_open", "manual_participant_approval",
    "is_docker_based", "submission_time_limit", "github_repository", "list_tags",
)
PHASE_FIELDS = (
    "name", "codename", "start_date", "end_date", "is_public", "is_active", "max_submissions_per_day",
    "max_submissions_per_month", "max_submissions", "allowed_submission_file_types", "leaderboard_public",
)


@dataclass
class ApiPage:
    url: str
    title: str
    text: str
    links: list[dict[str, str]] = field(default_factory=list)
    data: dict = field(default_factory=dict)


def challenge_id(url: str) -> str | None:
    match = CHALLENGE_PATH.search(urlparse(url).path)
    return match.group(1) if match else None


def api_base(web: Web, page_url: str) -> str:
    """The API address declared by the front end, else the conventional `/api/` of the same site."""
    parsed = urlparse(page_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    try:
        config = web.read(origin + "/dist/js/config.min.js", "Configuration EvalAI (adresse de l'API)")
    except NetworkError:
        config = None
    match = API_IN_CONFIG.search(config.text) if config is not None and config.ok else None
    base = match.group(1) if match else origin + "/api/"
    return base if base.endswith("/") else base + "/"


def read_challenge(web: Web, page_url: str, notes: list[str]) -> list[ApiPage]:
    number = challenge_id(page_url)
    if number is None:
        return []
    base = api_base(web, page_url)
    pages = []
    challenge = _get_json(web, f"{base}challenges/challenge/{number}/", notes)
    if isinstance(challenge, dict):
        pages.append(_render(f"{base}challenges/challenge/{number}/", challenge.get("title", ""), challenge,
                             CHALLENGE_FIELDS))
    phases = _get_json(web, f"{base}challenges/challenge/{number}/challenge_phase", notes)
    if isinstance(phases, dict):
        results = [phase for phase in phases.get("results", []) if isinstance(phase, dict)]
        url = f"{base}challenges/challenge/{number}/challenge_phase"
        pages.append(_render_phases(url, results))
    return pages


def _get_json(web: Web, url: str, notes: list[str]):
    try:
        page = web.read(url, "API EvalAI", {"Accept": "application/json"})
    except NetworkError as exc:
        notes.append(str(exc))
        return None
    if not page.ok:
        notes.append(f"{url} : statut HTTP {page.status}.")
        return None
    try:
        return json.loads(page.text)
    except ValueError:
        notes.append(f"{url} : réponse illisible (JSON attendu).")
        return None


def _scalars(data: dict, names: tuple[str, ...]) -> list[str]:
    return [f'"{name}": {json.dumps(data[name], ensure_ascii=False)}' for name in names if data.get(name) not in (None, "")]


def _render(url: str, title: str, data: dict, names: tuple[str, ...]) -> ApiPage:
    lines, links = _scalars(data, names), []
    for name in TEXT_FIELDS:
        if isinstance(data.get(name), str) and data[name].strip():
            _, text, found = html_to_text(data[name], url)
            lines += ["", f"## {name}", text]
            links += found
    repository = data.get("github_repository")
    if isinstance(repository, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        links.append({"url": f"https://github.com/{repository}", "text": "github_repository"})
    return ApiPage(url, title, "\n".join(lines), links, data)


def _render_phases(url: str, phases: list[dict]) -> ApiPage:
    lines, links = [], []
    for phase in phases:
        lines += ["", f"## Phase {phase.get('name', phase.get('id'))}"] + _scalars(phase, PHASE_FIELDS)
        if isinstance(phase.get("description"), str):
            _, text, found = html_to_text(phase["description"], url)
            lines.append(text)
            links += found
    return ApiPage(url, "Phases du challenge", "\n".join(lines).strip(), links, {"results": phases})


def submission_phase(phases: list[dict]) -> dict | None:
    """The public phase that accepts submissions now, else the last public one."""
    public = [phase for phase in phases if phase.get("is_public")]
    active = [phase for phase in public if phase.get("is_active")]
    return (active or public or [None])[-1]
