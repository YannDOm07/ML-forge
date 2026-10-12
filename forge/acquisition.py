"""Acquisition of the official data (Kaggle API, public links, user inbox) and of external resources.

Official data are downloaded automatically within the size and disk limits. Whatever needs
the user (token, accepting the rules, a login) becomes a precise message and an open action;
ML Forge then resumes when the command is run again. External data and pretrained models
are never downloaded without the acceptance sheet and an explicit approval (prompt_v2 §7).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from forge.actions import open_action, resolve_actions
from forge.datasources import register_data_source
from forge.db import now_iso, session, transaction
from forge.errors import ForgeError
from forge.guardrails import Action, ApprovalRequest, Approver, project_decision_recorder, require_approval
from forge.intake import DATA_EXTENSIONS, competition_card, identify_platform, infer_from_data, stored_pages
from forge.kaggle import KaggleClient, credentials_path, file_size, load_credentials
from forge.projects import Project
from forge.web import NetworkError, Web
from forge.workspace import Workspace

INBOX = "inbox"


class ForbiddenByRules(ForgeError):
    """The competition rules forbid this resource: it is never proposed."""


@dataclass
class AcquisitionResult:
    status: str  # done | needs_user | refused
    files: list[Path] = field(default_factory=list)
    message: str = ""


def _max_bytes(workspace: Workspace) -> int:
    return int(workspace.config.get("acquisition", "max_download_gb") * 1024**3)


def inbox_dir(project: Project) -> Path:
    return project.path / INBOX


# --- official data ------------------------------------------------------------------------

def acquire_official(workspace: Workspace, project: Project, transport=None) -> AcquisitionResult:
    if not project.competition_url:
        raise ForgeError("Aucun lien de compétition pour ce projet.")
    platform = identify_platform(project.competition_url)
    web = Web(workspace, transport, project.id)
    if platform.name == "kaggle" and platform.slug:
        result = _acquire_kaggle(workspace, project, web, platform.slug)
    else:
        result = _acquire_from_pages(workspace, project, web)
    if result.files:
        _register(project, result.files, "official")
    return result


def _acquire_kaggle(workspace: Workspace, project: Project, web: Web, slug: str) -> AcquisitionResult:
    destination = workspace.data_dir / "kaggle" / slug
    if destination.is_dir() and any(destination.iterdir()):
        return AcquisitionResult("done", _data_files(destination), f"Données déjà présentes dans {destination} (réutilisées).")
    credentials = load_credentials()
    if credentials is None:
        return _needs_user(project, "credentials", (
            f"Jeton Kaggle absent. Sur kaggle.com : Settings > API > Create New Token, puis placez le fichier "
            f"kaggle.json dans {credentials_path().parent}. Ne collez jamais ce jeton dans une conversation. "
            f"Relancez ensuite : python -m forge acquire {project.id}"
        ))
    client = KaggleClient(web, credentials)
    status, files = client.files(slug)
    if status == 401:
        return _needs_user(project, "credentials", "Jeton Kaggle refusé (401) : régénérez kaggle.json puis relancez.")
    total = sum(file_size(entry) for entry in files)
    if total > _max_bytes(workspace):
        return AcquisitionResult("refused", [], (
            f"Données officielles : {total / 1024**3:.2f} Go, au-delà de la limite "
            f"acquisition.max_download_gb ({_max_bytes(workspace) / 1024**3:.2f} Go). Rien n'est téléchargé."
        ))
    archive = workspace.data_dir / "kaggle" / f"{slug}.zip"
    archive.unlink(missing_ok=True)  # a leftover of an interrupted download
    try:
        download = client.download_all(slug, archive, _max_bytes(workspace))
    except NetworkError as exc:
        return AcquisitionResult("refused", [], str(exc))
    if download.status == 403:
        return _needs_user(project, "accept_rules", (
            f"Kaggle refuse le téléchargement (403) : le règlement n'est pas accepté. Ouvrez "
            f"https://www.kaggle.com/competitions/{slug}/rules, connectez-vous et cliquez « Join Competition » / "
            f"« I Understand and Accept ». Relancez ensuite : python -m forge acquire {project.id}"
        ))
    if download.status == 401:
        return _needs_user(project, "credentials", "Jeton Kaggle refusé (401) : régénérez kaggle.json puis relancez.")
    if download.status >= 300:
        return AcquisitionResult("refused", [], f"Téléchargement Kaggle impossible (statut {download.status}).")
    files = _unpack(workspace, archive, destination)
    resolve_actions(project, "accept_rules")
    resolve_actions(project, "credentials")
    return AcquisitionResult("done", files, f"{len(files)} fichier(s) dans {destination}.")


def _acquire_from_pages(workspace: Workspace, project: Project, web: Web) -> AcquisitionResult:
    links = _data_links(project)
    if not links:
        return _needs_inbox(project, [], "Aucun lien public vers les données n'a été trouvé sur les pages lues.")
    downloaded, blocked = [], []
    for url in links:
        destination = workspace.data_dir / "web" / urlparse(url).netloc / _file_name(url)
        if destination.is_file():
            downloaded.append(destination)
            continue
        try:
            result = web.download(url, destination, "Données officielles (lien public)", _max_bytes(workspace))
        except NetworkError as exc:
            return AcquisitionResult("refused", downloaded, str(exc))
        if result.status >= 300 or "html" in result.content_type:
            destination.unlink(missing_ok=True)
            blocked.append(_file_name(url))
        else:
            downloaded.append(destination)
    files = [file for path in downloaded for file in _expand(workspace, path)]
    if blocked:
        result = _needs_inbox(project, blocked, "Ces fichiers exigent une connexion ou une inscription.")
        result.files = files
        return result
    resolve_actions(project, INBOX)
    return AcquisitionResult("done", files, f"{len(files)} fichier(s) téléchargé(s).")


def _data_links(project: Project) -> list[str]:
    links = []
    for page in stored_pages(project):
        if page["origin"] != "web":
            continue
        for link in json.loads(page["links_json"]):
            path = urlparse(link["url"]).path.lower()
            if path.endswith(DATA_EXTENSIONS) and link["url"] not in links:
                links.append(link["url"])
    return links


def _file_name(url: str) -> str:
    return PurePosixPath(unquote(urlparse(url).path)).name or "donnees"


def _needs_user(project: Project, kind: str, message: str) -> AcquisitionResult:
    open_action(project, kind, message)
    return AcquisitionResult("needs_user", [], message)


def _needs_inbox(project: Project, expected: list[str], reason: str) -> AcquisitionResult:
    inbox = inbox_dir(project)
    inbox.mkdir(exist_ok=True)
    wanted = f" Fichiers attendus : {', '.join(expected)}." if expected else ""
    message = (
        f"{reason} Téléchargez les données depuis {project.competition_url} (après connexion et acceptation du "
        f"règlement si nécessaire) et déposez-les dans {inbox}.{wanted} Puis : python -m forge inbox {project.id}"
    )
    return _needs_user(project, INBOX, message)


def ingest_inbox(workspace: Workspace, project: Project) -> AcquisitionResult:
    """Register the files the user dropped in the inbox (archives are unpacked next to them)."""
    inbox = inbox_dir(project)
    if not inbox.is_dir() or not any(inbox.iterdir()):
        return AcquisitionResult("needs_user", [], f"Le dossier {inbox} est vide.")
    files = []
    for path in sorted(inbox.iterdir()):
        if path.is_file():
            files += _expand(workspace, path)
    files = [file for file in files if file.is_file()]
    _register(project, files, "inbox")
    resolve_actions(project, INBOX)
    return AcquisitionResult("done", files, f"{len(files)} fichier(s) enregistré(s) depuis {inbox}.")


def _expand(workspace: Workspace, path: Path) -> list[Path]:
    if zipfile.is_zipfile(path):
        return _unpack(workspace, path, path.with_suffix(""))
    return [path]


def _unpack(workspace: Workspace, archive: Path, destination: Path) -> list[Path]:
    """Extract a zip safely (no path traversal, size checked against the disk reserve), then delete it."""
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or ":" in name:
                raise ForgeError(f"Archive refusée : chemin dangereux « {name} ».")
        Web(workspace).ensure_disk_space(sum(info.file_size for info in bundle.infolist()))
        destination.mkdir(parents=True, exist_ok=True)
        bundle.extractall(destination)
    archive.unlink()  # temporary download; the extracted files are the data
    return _data_files(destination)


def _data_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _register(project: Project, files: list[Path], role: str) -> None:
    for path in files:
        register_data_source(project.db_path, path, role, origin_url=project.competition_url)
    infer_from_data(project, files)


# --- external resources -----------------------------------------------------------------------

@dataclass(frozen=True)
class ExternalResource:
    kind: str  # data | pretrained
    url: str
    provider: str
    license: str
    size_mb: float
    role: str
    risk: str


def external_verdict(project: Project, kind: str) -> tuple[str, str]:
    """('allowed' | 'forbidden' | 'uncertain', justification) from the competition card.

    An inferred 'forbidden' is enough to forbid; an inferred 'allowed' is only 'uncertain'."""
    name = "rules.external_data" if kind == "data" else "rules.pretrained_models"
    fact = competition_card(project)["fields"][name]
    if fact is None or fact["value"] in {"unknown", "conditional"}:
        return "uncertain", "Le règlement ne tranche pas (ou n'a pas été trouvé) : la décision vous revient."
    quote = f"« {fact['quote'][:200]} » ({fact['source_url']})"
    if fact["value"] == "forbidden":
        return "forbidden", quote
    if fact["status"] == "inferred":
        return "uncertain", f"Autorisation déduite, non confirmée : {quote}"
    return "allowed", quote


def request_external(
    workspace: Workspace, project: Project, resource: ExternalResource, approver: Approver, decided_by: str,
    transport=None,
) -> AcquisitionResult:
    if resource.kind not in {"data", "pretrained"}:
        raise ForgeError(f"Type de ressource externe inconnu : {resource.kind}")
    verdict, justification = external_verdict(project, resource.kind)
    if verdict == "forbidden":
        raise ForbiddenByRules(f"Ressource externe interdite par le règlement : {justification}")
    request = ApprovalRequest(
        Action.EXTERNAL_DATA if resource.kind == "data" else Action.PRETRAINED_MODEL,
        resource.url,
        {
            "source": f"{resource.url} ({resource.provider})",
            "licence et taille": f"{resource.license}, {resource.size_mb:.0f} Mo",
            "rôle prévu": resource.role,
            "conformité": f"{ {'allowed': 'autorisé', 'uncertain': 'incertain'}[verdict] } — {justification}",
            "risque": resource.risk,
        },
    )
    require_approval(request, approver, project_decision_recorder(project.db_path, decided_by))
    if resource.size_mb * 1024**2 > _max_bytes(workspace):
        return AcquisitionResult("refused", [], "Ressource plus grande que acquisition.max_download_gb : non téléchargée.")
    destination = workspace.data_dir / "external" / urlparse(resource.url).netloc / _file_name(resource.url)
    download = Web(workspace, transport, project.id).download(
        resource.url, destination, f"Ressource externe acceptée ({resource.kind})", _max_bytes(workspace)
    )
    if download.status >= 300:
        return AcquisitionResult("refused", [], f"Téléchargement refusé par le serveur (statut {download.status}).")
    files = _expand(workspace, destination)
    for path in files:
        register_data_source(project.db_path, path, "external", origin_url=resource.url, license=resource.license)
    with session(project.db_path) as conn, transaction(conn):
        conn.execute(
            "INSERT INTO compliance_log (item, justification, source, at) VALUES (?, ?, ?, ?)",
            (f"Ressource externe ({resource.kind}) : {resource.url}", f"Acceptée par l'utilisateur ; {justification}",
             resource.url, now_iso()),
        )
    return AcquisitionResult("done", files, f"Ressource externe enregistrée : {len(files)} fichier(s).")
