"""Minimal client for the public Kaggle API (v1 REST endpoints), on top of `Web`.

The token is read from KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json. It is sent only in an
Authorization header (never forwarded on redirects), and never printed, logged or stored.
[à vérifier avec un vrai jeton] : forme exacte des réponses JSON et code d'erreur
quand le règlement n'est pas accepté (403 attendu).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from forge.web import Download, Web

API = "https://www.kaggle.com/api/v1"


@dataclass(frozen=True)
class KaggleCredentials:
    username: str
    key: str

    def header(self) -> dict[str, str]:
        token = base64.b64encode(f"{self.username}:{self.key}".encode()).decode()
        return {"Authorization": f"Basic {token}"}


def credentials_path() -> Path:
    directory = os.environ.get("KAGGLE_CONFIG_DIR")
    return Path(directory) / "kaggle.json" if directory else Path.home() / ".kaggle" / "kaggle.json"


def load_credentials() -> KaggleCredentials | None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return KaggleCredentials(os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    path = credentials_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return KaggleCredentials(data["username"], data["key"])
    except (OSError, ValueError, KeyError):
        return None


def metadata_url(slug: str) -> str:
    return f"{API}/competitions/list?search={quote(slug)}"


def files_url(slug: str) -> str:
    return f"{API}/competitions/data/list/{quote(slug)}"


def download_url(slug: str) -> str:
    return f"{API}/competitions/data/download-all/{quote(slug)}"


class KaggleClient:
    def __init__(self, web: Web, credentials: KaggleCredentials) -> None:
        self.web = web
        self.credentials = credentials

    def _json(self, url: str, purpose: str) -> tuple[int, object | None, str]:
        page = self.web.read(url, purpose, self.credentials.header())
        if not page.ok:
            return page.status, None, page.text
        try:
            return page.status, json.loads(page.text), page.text
        except ValueError:
            return page.status, None, page.text

    def competition(self, slug: str) -> tuple[int, dict | None]:
        status, data, _ = self._json(metadata_url(slug), "Kaggle : métadonnées de la compétition")
        if not isinstance(data, list):
            return status, None
        for item in data:
            reference = str(item.get("ref", ""))
            if reference.rstrip("/").split("/")[-1] == slug:
                return status, item
        return status, None

    def files(self, slug: str) -> tuple[int, list[dict]]:
        status, data, _ = self._json(files_url(slug), "Kaggle : liste des fichiers")
        if isinstance(data, dict):
            data = data.get("files") or data.get("datasetFiles") or []
        return status, data if isinstance(data, list) else []

    def download_all(self, slug: str, dest: Path, max_bytes: int) -> Download:
        return self.web.download(
            download_url(slug), dest, "Kaggle : données officielles", max_bytes, self.credentials.header()
        )


def file_size(entry: dict) -> int:
    for key in ("totalBytes", "size", "fileSize"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0
