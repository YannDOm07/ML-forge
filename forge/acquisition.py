"""Acquisition of the official data (Kaggle API, data repositories, public links, user inbox) and of
external resources.

Official data are downloaded automatically within the size and disk limits. Whatever needs
the user (token, accepting the rules, a login) becomes a precise message and an open action;
ML Forge then resumes when the command is run again. External data and pretrained models
are never downloaded without the acceptance sheet and an explicit approval (prompt_v2 §7).
"""

from __future__ import annotations

import json
import re
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse

from forge.actions import open_action, resolve_actions
from forge.datasources import register_data_source, register_data_sources
from forge.db import now_iso, session, transaction
from forge.errors import ForgeError
from forge.guardrails import Action, ApprovalRequest, Approver, project_decision_recorder, require_approval
from forge.intake import DATA_EXTENSIONS, competition_card, identify_platform, infer_from_data, stored_pages
from forge.kaggle import KaggleClient, credentials_path, file_size, load_credentials
from forge.projects import Project
from forge.repositories import size_label
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


def _max_bytes(workspace: Workspace, max_gb: float | None = None) -> int:
    """The per-download limit; `max_gb` is a one-off allowance given by the user on the command line."""
    return int((max_gb or workspace.config.get("acquisition", "max_download_gb")) * 1024**3)


def inbox_dir(project: Project) -> Path:
    return project.path / INBOX


# --- official data ------------------------------------------------------------------------

def acquire_official(workspace: Workspace, project: Project, transport=None, source: str | None = None,
                     max_gb: float | None = None) -> AcquisitionResult:
    """Official data: Kaggle API, else the dataset repository the competition cites, else direct links.

    `source` names the repository to use when the automatic choice is wrong; `max_gb` raises the
    size limit for this run only (an explicit decision of the user, logged with the downloads)."""
    if not project.competition_url:
        raise ForgeError("Aucun lien de compétition pour ce projet.")
    platform = identify_platform(project.competition_url)
    web = Web(workspace, transport, project.id)
    if platform.name == "kaggle" and platform.slug:
        result = _acquire_kaggle(workspace, project, web, platform.slug, max_gb)
    else:
        result = (_acquire_from_repository(workspace, project, web, source, max_gb)
                  or _acquire_from_pages(workspace, project, web))
    if result.files:
        _register(project, result.files, "official")
    return result


def _acquire_kaggle(workspace: Workspace, project: Project, web: Web, slug: str,
                    max_gb: float | None = None) -> AcquisitionResult:
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
    if total > _max_bytes(workspace, max_gb):
        return AcquisitionResult("refused", [], (
            f"Données officielles : {total / 1024**3:.2f} Go, au-delà de la limite "
            f"acquisition.max_download_gb ({_max_bytes(workspace, max_gb) / 1024**3:.2f} Go). Rien n'est téléchargé."
        ))
    archive = workspace.data_dir / "kaggle" / f"{slug}.zip"
    archive.unlink(missing_ok=True)  # a leftover of an interrupted download
    try:
        download = client.download_all(slug, archive, _max_bytes(workspace, max_gb))
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


@dataclass
class DataSource:
    url: str
    title: str
    files: list[dict]
    gated: str | None

    @property
    def size(self) -> int:
        return sum(int(entry.get("size") or 0) for entry in self.files)


def dataset_sources(project: Project) -> list[DataSource]:
    """Dataset repositories described during the intake, in the order they were found."""
    sources: dict[str, DataSource] = {}
    for page in stored_pages(project):
        if page["origin"] != "api":
            continue
        for link in json.loads(page["links_json"]):
            if link.get("kind") != "datasets" or "repository" not in link:
                continue
            entry = sources.setdefault(link["repository"], DataSource(
                link["repository"], f"{page['title']} {page['text'][:500]}", [], link.get("gated")))
            if all(known["url"] != link["url"] for known in entry.files):
                entry.files.append(link)
    return list(sources.values())


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]+|\d+", text.lower()) if len(token) > 1}


def official_source(project: Project, sources: list[DataSource], wanted: str | None = None) -> DataSource | None:
    """The repository that best matches the competition title; open access first, then citation order."""
    if wanted:
        return next((source for source in sources if source.url.rstrip("/") == wanted.rstrip("/")), None)
    title = competition_card(project)["fields"]["title"]
    words = _tokens(title["value"]) if title else set()
    ranked = sorted(enumerate(sources), key=lambda item: (
        -len(words & _tokens(item[1].title)), item[1].gated is not None, item[0]))
    return ranked[0][1] if ranked else None


def _acquire_from_repository(workspace: Workspace, project: Project, web: Web, wanted: str | None,
                             max_gb: float | None = None) -> AcquisitionResult | None:
    sources = dataset_sources(project)
    source = official_source(project, sources, wanted)
    if source is None:
        if wanted:
            known = ", ".join(s.url for s in sources) or "aucun"
            return AcquisitionResult("refused", [], f"Dépôt {wanted} non décrit (dépôts connus : {known}).")
        return None
    others = [s.url for s in sources if s is not source]
    choice = f"Jeu de données retenu : {source.url} ({size_label(source.size)})" + (
        f" ; autres dépôts : {', '.join(others)} (option --source pour en choisir un autre)" if others else "")
    if source.gated:
        return _needs_user(project, "access", (
            f"{choice}. Accès sur demande ({source.gated}) : faites la demande sur {source.url}, puis relancez "
            f"python -m forge acquire {project.id}. ML Forge n'utilise jamais un mot de passe trouvé sur une page."))
    limit = _max_bytes(workspace, max_gb)
    small = [entry for entry in source.files if int(entry.get("size") or 0) <= limit]
    large = [entry for entry in source.files if int(entry.get("size") or 0) > limit]
    try:
        web.ensure_disk_space(sum(int(entry.get("size") or 0) for entry in small))
    except NetworkError as exc:
        return AcquisitionResult("refused", [], f"{choice}. {exc}")
    root = workspace.data_dir / _source_dir(source.url)
    downloaded, failed = [], []
    for entry in small:
        destination = root / PurePosixPath(entry["text"])
        if not destination.is_relative_to(root):
            failed.append(entry["text"])
            continue
        if not destination.is_file() and not _already_extracted(destination):
            try:
                result = web.download(entry["url"], destination, f"Données officielles ({source.url})", limit)
            except NetworkError as exc:
                failed.append(f"{entry['text']} ({exc})")
                continue
            if result.status >= 300:
                failed.append(f"{entry['text']} (statut {result.status})")
                continue
        downloaded.append(destination)
    files = [file for path in downloaded for file in _expand(workspace, path)]
    lines = [choice, f"{len(downloaded)} fichier(s) téléchargé(s) dans {root}."]
    if failed:
        lines.append("Échecs : " + " ; ".join(failed))
    if large:
        lines.append(
            f"Au-delà de la limite de {limit / 1024**3:g} Go (acquisition.max_download_gb), non téléchargés : "
            + ", ".join(f"{entry['text']} ({size_label(int(entry['size']))})" for entry in large)
            + f". Pour les obtenir : python -m forge acquire {project.id} --max-gb <taille en Go> "
            "(autorisation pour cette fois), ou augmentez la limite dans la configuration.")
        open_action(project, "large_download", "\n".join(lines))
    resolve_actions(project, "access")
    if not large:
        resolve_actions(project, "large_download")
    # Files already obtained are kept and registered; the campaign waits until the set is complete.
    status = "refused" if failed else "needs_user" if large else "done"
    return AcquisitionResult(status, files, "\n".join(lines))


def _source_dir(url: str) -> Path:
    parsed = urlparse(url)
    parts = [re.sub(r"[^\w.-]", "_", part) for part in parsed.path.split("/") if part]
    return Path(parsed.netloc.split(":")[0], *parts)


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


TAR_SUFFIXES = (".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")
EXTRACTED = ".extrait-par-mlforge"  # written once an archive is fully extracted (the archive is then deleted)


def _tar_destination(path: Path) -> Path | None:
    for suffix in TAR_SUFFIXES:
        if path.name.lower().endswith(suffix):
            return path.with_name(path.name[: -len(suffix)])
    return None


def _already_extracted(path: Path) -> bool:
    destination = _tar_destination(path)
    return destination is not None and (destination / EXTRACTED).is_file()


def _extracted_files(destination: Path) -> list[Path]:
    return [path for path in _data_files(destination) if path.name != EXTRACTED]


def _expand(workspace: Workspace, path: Path) -> list[Path]:
    if _already_extracted(path):
        return _extracted_files(_tar_destination(path))
    if zipfile.is_zipfile(path):
        return _unpack(workspace, path, path.with_suffix(""))
    if _tar_destination(path) is not None and tarfile.is_tarfile(path):
        return _untar(workspace, path, _tar_destination(path))
    return [path]


def _untar(workspace: Workspace, archive: Path, destination: Path) -> list[Path]:
    """Extract a tar archive with the 'data' filter (no traversal, links or devices), then delete it."""
    try:
        with tarfile.open(archive) as bundle:
            Web(workspace).ensure_disk_space(sum(member.size for member in bundle.getmembers()))
            destination.mkdir(parents=True, exist_ok=True)
            bundle.extractall(destination, filter="data")
    except tarfile.TarError as exc:
        raise ForgeError(f"Archive refusée ({archive.name}) : {exc}") from exc
    (destination / EXTRACTED).write_text(archive.name, encoding="utf-8")
    archive.unlink()  # temporary download; the extracted files are the data
    return _extracted_files(destination)


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
    register_data_sources(project.db_path, files, role, origin_url=project.competition_url)
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
