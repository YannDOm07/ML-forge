"""Data and model repositories cited by a competition: Hugging Face (datasets, models) and GitHub.

Their links carry no file extension, so they are recognised by address, then described through
their public API: files and sizes, licence, access conditions (a gated repository stays the
user's business), README. Nothing is downloaded here; each description becomes a citable page.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import quote, urlparse

from forge.evalai import ApiPage
from forge.web import NetworkError, Web

URL_IN_TEXT = re.compile(r"https?://[^\s<>\"')\]]+")
README_MAX_CHARS = 20_000
LISTED_FILES = 200
SUBMISSION_EXAMPLE = re.compile(r"(sample|demo|example)[_-]?submission", re.IGNORECASE)
_NOT_A_REPO = {"api", "docs", "blog", "spaces", "settings", "login", "join", "pricing", "orgs", "topics", "features"}


@dataclass(frozen=True)
class Repository:
    host: str  # huggingface | github
    kind: str  # datasets | models | code
    id: str  # owner/name

    @property
    def url(self) -> str:
        if self.host == "github":
            return f"https://github.com/{self.id}"
        prefix = "datasets/" if self.kind == "datasets" else ""
        return f"https://huggingface.co/{prefix}{self.id}"

    @property
    def label(self) -> str:
        return {"datasets": "jeu de données", "models": "modèle", "code": "code"}[self.kind]


def parse_repository(url: str) -> Repository | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if host == "github.com" and len(parts) >= 2 and parts[0] not in _NOT_A_REPO:
        return Repository("github", "code", f"{parts[0]}/{parts[1].removesuffix('.git')}")
    if host == "huggingface.co":
        if len(parts) >= 3 and parts[0] == "datasets":
            return Repository("huggingface", "datasets", f"{parts[1]}/{parts[2]}")
        if len(parts) >= 2 and parts[0] not in _NOT_A_REPO and parts[0] != "datasets":
            return Repository("huggingface", "models", f"{parts[0]}/{parts[1]}")
    return None


def repositories_in(pages: list[dict]) -> list[Repository]:
    """Repositories cited by stored pages, in order of appearance, without duplicates."""
    found: list[Repository] = []
    for page in pages:
        urls = [link["url"] for link in json.loads(page.get("links_json") or "[]")]
        urls += URL_IN_TEXT.findall(page.get("text") or "")
        for url in urls:
            repository = parse_repository(url.rstrip(".,;:"))
            if repository and repository not in found:
                found.append(repository)
    return found


def describe(web: Web, repository: Repository, notes: list[str]) -> ApiPage | None:
    try:
        if repository.host == "github":
            return _describe_github(web, repository, notes)
        return _describe_huggingface(web, repository, notes)
    except NetworkError as exc:
        notes.append(f"{repository.url} : {exc}")
        return None


def size_label(size: int) -> str:
    for unit, scale in (("Go", 1e9), ("Mo", 1e6), ("ko", 1e3)):
        if size >= scale:
            return f"{size / scale:.2f} {unit}"
    return f"{size} o"


def _json(web: Web, url: str, purpose: str):
    page = web.read(url, purpose, {"Accept": "application/json"})
    if not page.ok:
        return page.status, None
    try:
        return page.status, json.loads(page.text)
    except ValueError:
        return page.status, None


def _file_lines(files: list[tuple[str, int]], link) -> tuple[list[str], list[dict[str, str]]]:
    total = sum(size for _, size in files)
    lines = [f"Fichiers : {len(files)}, taille totale {size_label(total)}"]
    lines += [f"- {path} ({size_label(size)})" for path, size in files[:LISTED_FILES]]
    if len(files) > LISTED_FILES:
        lines.append(f"- … {len(files) - LISTED_FILES} autres fichiers")
    return lines, [{"url": link(path), "text": path} for path, _ in files]


def _describe_huggingface(web: Web, repository: Repository, notes: list[str]) -> ApiPage | None:
    api = f"https://huggingface.co/api/{repository.kind}/{repository.id}"
    status, info = _json(web, api, f"Description du dépôt Hugging Face {repository.id}")
    if not isinstance(info, dict):
        notes.append(f"{repository.url} : dépôt Hugging Face illisible (statut {status}).")
        return None
    card = info.get("cardData") or {}
    lines = [f"Dépôt Hugging Face ({repository.label}) : {repository.id}", f"Adresse : {repository.url}"]
    lines += [f'"{key}": {json.dumps(info[key], ensure_ascii=False)}'
              for key in ("gated", "private", "disabled", "lastModified") if key in info]
    if card.get("license"):
        lines.append(f'"license": {json.dumps(card["license"], ensure_ascii=False)}')
    for key in ("extra_gated_heading", "extra_gated_description"):
        if card.get(key):
            lines.append(f"{key} : {card[key]}")
    if card.get("extra_gated_fields"):
        lines.append("Champs à remplir pour demander l'accès : " + ", ".join(card["extra_gated_fields"]))
    _, tree = _json(web, f"{api}/tree/main?recursive=true", f"Fichiers du dépôt {repository.id}")
    files = [(entry["path"], int(entry.get("size") or 0)) for entry in tree or []
             if isinstance(entry, dict) and entry.get("type") == "file"]
    prefix = "datasets/" if repository.kind == "datasets" else ""
    file_lines, links = _file_lines(
        files, lambda path: f"https://huggingface.co/{prefix}{repository.id}/resolve/main/{quote(path)}")
    if info.get("gated"):
        notes.append(f"{repository.url} : accès sur demande (gated={info['gated']}) ; la demande appartient à l'utilisateur.")
    return ApiPage(api, f"Dépôt Hugging Face {repository.id}", "\n".join(lines + file_lines), links,
                   {"repository": repository, "files": files, "gated": info.get("gated")})


def _describe_github(web: Web, repository: Repository, notes: list[str]) -> ApiPage | None:
    api = f"https://api.github.com/repos/{repository.id}"
    status, info = _json(web, api, f"Description du dépôt GitHub {repository.id}")
    if not isinstance(info, dict):
        notes.append(f"{repository.url} : dépôt GitHub illisible (statut {status}).")
        return None
    branch = info.get("default_branch") or "HEAD"
    lines = [f"Dépôt GitHub : {repository.id}", f"Adresse : {repository.url}"]
    if info.get("description"):
        lines.append(f'"description": {json.dumps(info["description"], ensure_ascii=False)}')
    license_name = (info.get("license") or {}).get("spdx_id")
    if license_name:
        lines.append(f'"license": {json.dumps(license_name)}')
    _, tree = _json(web, f"{api}/git/trees/{branch}?recursive=1", f"Fichiers du dépôt {repository.id}")
    entries = tree.get("tree", []) if isinstance(tree, dict) else []
    files = [(entry["path"], int(entry.get("size") or 0)) for entry in entries if entry.get("type") == "blob"]
    raw = f"https://raw.githubusercontent.com/{repository.id}/{branch}/"
    file_lines, links = _file_lines(files, lambda path: raw + quote(path))
    lines += file_lines
    readme = next((path for path, _ in files if path.lower() in {"readme.md", "readme.rst", "readme.txt", "readme"}), None)
    if readme:
        page = web.read(raw + quote(readme), f"README du dépôt {repository.id}")
        if page.ok:
            lines += ["", f"## {readme}", page.text[:README_MAX_CHARS]]
            links += [{"url": url, "text": ""} for url in URL_IN_TEXT.findall(page.text)]
    return ApiPage(api, f"Dépôt GitHub {repository.id}", "\n".join(lines), links,
                   {"repository": repository, "files": files, "gated": None})
