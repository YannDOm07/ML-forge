"""Serveur web local pour l'interface HTML de ML Forge."""

from __future__ import annotations

import json
import ipaddress
import mimetypes
import re
import socket
import threading
import webbrowser
from html.parser import HTMLParser
from io import BytesIO, StringIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_LINK_BYTES = 10 * 1024 * 1024
TARGET_NAMES = (
    "target", "label", "class", "outcome", "y", "survived", "saleprice",
    "price", "churn", "species", "diagnosis", "loan_status", "is_fraud",
)


class _CompetitionPageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.description = ""
        self._in_title = False
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta" and values.get("name", "").lower() in {"description", "og:description"}:
            self.description = values.get("content") or self.description
        if tag == "a" and values.get("href"):
            self.links.append((values["href"], values.get("download") or ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if not value or self._ignored_depth:
            return
        self.text_parts.append(value)
        if self._in_title:
            self.title_parts.append(value)


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Utilisez une URL publique commençant par http:// ou https://.")
    if parsed.username or parsed.password:
        raise ValueError("Les URL contenant des identifiants ne sont pas acceptées.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError as exc:
        raise ValueError("Le domaine du lien est introuvable depuis cet ordinateur.") from exc
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Pour des raisons de sécurité, ML Forge n'accède qu'aux adresses publiques.")


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_public_url(url: str, max_bytes: int = MAX_LINK_BYTES) -> tuple[bytes, str, str]:
    _validate_public_url(url)
    request = Request(url, headers={"User-Agent": "ML-Forge-Local/1.0", "Accept": "text/html,text/csv,*/*"})
    opener = build_opener(_PublicRedirectHandler())
    try:
        with opener.open(request, timeout=12) as response:
            content = response.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise ValueError("La page ou le fichier dépasse la limite de 10 Mo.")
            return content, response.headers.get_content_type(), response.geturl()
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError("Le site demande une connexion ou refuse l'accès. Les pages privées ne sont pas importées.") from exc
        raise ValueError(f"Le site a refusé le lien (HTTP {exc.code}).") from exc
    except URLError as exc:
        raise ValueError(f"Impossible de récupérer ce lien : {exc.reason}") from exc


def _infer_target(columns: list[str]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_"): column for column in columns}
    for name in TARGET_NAMES:
        if name in normalized:
            return normalized[name]
    return None


def _import_link(manager, url: str, fetcher=_fetch_public_url) -> dict[str, Any]:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Collez une URL publique commençant par http:// ou https://.")

    content, content_type, final_url = fetcher(url)
    page_title = ""
    page_text = ""
    page_description = ""
    candidates: list[tuple[int, str]] = []
    data_content: bytes | None = None
    data_url = final_url
    filename = Path(urlparse(final_url).path).name or "dataset.csv"

    if content_type in {"text/csv", "application/csv", "application/vnd.ms-excel"} or Path(urlparse(final_url).path).suffix.lower() in {".csv", ".tsv"}:
        data_content = content
    elif content_type in {"text/html", "application/xhtml+xml"} or content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        parser = _CompetitionPageParser()
        parser.feed(content.decode("utf-8", errors="replace"))
        page_title = " ".join(parser.title_parts).strip()
        page_text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
        page_description = parser.description.strip()
        for href, download_name in parser.links:
            candidate = urljoin(final_url, href)
            candidate_name = download_name or Path(urlparse(candidate).path).name
            if Path(urlparse(candidate).path).suffix.lower() not in {".csv", ".tsv"}:
                continue
            lowered = candidate_name.lower()
            score = 5 if "train" in lowered else 3 if any(word in lowered for word in ("dataset", "data")) else 0
            if any(word in lowered for word in ("test", "sample", "submission", "solution")):
                score -= 10
            candidates.append((score, candidate))
        if not candidates:
            raise ValueError("La page est accessible, mais aucun lien CSV direct n'a été trouvé. Les liens de téléchargement protégés ou générés par script ne sont pas encore pris en charge.")
        candidates.sort(key=lambda item: item[0], reverse=True)
        fetch_errors = []
        for _, candidate_url in candidates[:5]:
            try:
                candidate_content, candidate_type, resolved_url = fetcher(candidate_url)
                if candidate_type not in {"text/csv", "application/csv", "application/vnd.ms-excel", "application/octet-stream"} and Path(urlparse(resolved_url).path).suffix.lower() not in {".csv", ".tsv"}:
                    continue
                candidate_frame = pd.read_csv(BytesIO(candidate_content), sep="\t" if Path(urlparse(resolved_url).path).suffix.lower() == ".tsv" else None, engine="python")
                if candidate_frame.empty or len(candidate_frame.columns) < 2:
                    continue
                data_content, data_url = candidate_content, resolved_url
                filename = Path(urlparse(resolved_url).path).name or "dataset.csv"
                break
            except Exception as exc:
                fetch_errors.append(str(exc))
        if data_content is None:
            raise ValueError("Aucun dataset CSV exploitable n'a été récupéré depuis cette page. " + (fetch_errors[0] if fetch_errors else ""))
    else:
        raise ValueError("Ce lien ne pointe ni vers une page web lisible ni vers un fichier CSV/TSV direct.")

    try:
        separator = "\t" if Path(filename).suffix.lower() == ".tsv" else None
        frame = pd.read_csv(BytesIO(data_content), sep=separator, engine="python")
    except Exception as exc:
        raise ValueError(f"Le fichier lié n'est pas un CSV/TSV lisible : {exc}") from exc
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("Le dataset doit contenir des lignes et au moins deux colonnes.")

    target_column = _infer_target([str(column) for column in frame.columns])
    project_name = (page_title or Path(filename).stem or parsed_url.hostname or "Projet importé").strip()[:100]
    source_notes = page_text[:10000]
    description = f"Importé depuis : {url}\nDataset : {data_url}"
    if source_notes:
        description += f"\nConsignes et description extraites : {source_notes}"
    elif page_title:
        description += f"\nPage : {page_title}"
    project_id = manager.create_project(name=project_name, description=description)
    if not project_id:
        raise RuntimeError("La création automatique du projet a échoué.")
    project = manager.open_project(project_id)
    if project is None:
        raise RuntimeError("Impossible d'ouvrir le projet créé.")
    dataset_path = project.dir / filename
    if dataset_path.suffix.lower() not in {".csv", ".tsv"}:
        dataset_path = project.dir / "dataset.csv"
    dataset_path.write_bytes(data_content)
    brief_lines = [
        f"# Import depuis un lien : {project_name}",
        "",
        f"- Page source : {url}",
        f"- Dataset récupéré : {data_url}",
        f"- Fichier local : {dataset_path.name}",
        f"- Colonne cible détectée : {target_column or 'à confirmer'}",
        f"- Lignes : {len(frame)}",
        f"- Colonnes : {len(frame.columns)}",
        "",
        "## Description de la page",
        page_description or "Aucune description meta fournie.",
        "",
        "## Texte public extrait",
        source_notes or "Aucun texte HTML extrait.",
        "",
        "Le texte ci-dessus est archivé comme source ; il n'est pas exécuté comme code.",
    ]
    (project.dir / "source_brief.md").write_text("\n".join(brief_lines), encoding="utf-8")
    updates = {
        "dataset_path": str(dataset_path),
        "dataset_hash": manager._hash_file(str(dataset_path)),
        "target_column": target_column,
    }
    project.db.update("projects", updates, {"id": project_id})
    manager.global_db.update("projects", updates, {"id": project_id})
    project.metadata.update(updates)
    from ml_forge.engines.data_engine import DataEngine

    summary = DataEngine().inspect_dataset(frame, target_column=target_column)

    result: dict[str, Any] = {
        "project_id": project_id,
        "project_name": project_name,
        "source_url": url,
        "dataset_url": data_url,
        "filename": dataset_path.name,
        "target_column": target_column,
        "summary": _json_value(summary),
        "status": "project_created",
        "page_description": page_description if content_type in {"text/html", "application/xhtml+xml"} else "",
    }
    if target_column:
        try:
            experiment = project.run_baseline_experiment(
                str(dataset_path),
                target_column,
                hypothesis=f"Évaluer une baseline à partir du lien fourni : {url}",
                method="Baseline automatique depuis un lien public",
            )
            result["experiment"] = _json_value(experiment)
            result["status"] = "baseline_complete"
        except Exception as exc:
            result["status"] = "project_created_target_inferred"
            result["warning"] = f"Le projet et le dataset sont prêts, mais la baseline n'a pas pu démarrer : {exc}"
    else:
        result["status"] = "project_created_target_unknown"
        result["warning"] = "Projet créé et dataset inspecté. Aucune colonne cible évidente n'a été trouvée ; ML Forge ne devine pas une cible au risque d'évaluer le mauvais objectif."
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def create_server(host: str = "127.0.0.1", port: int = 8765, link_fetcher=_fetch_public_url) -> ThreadingHTTPServer:
    manager_holder: dict[str, Any] = {}

    def get_manager():
        if "manager" not in manager_holder:
            from ml_forge.core.project_manager import ProjectManager

            manager_holder["manager"] = ProjectManager()
        return manager_holder["manager"]

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                raise ValueError("Requête vide ou supérieure à la limite de 10 Mo.")
            try:
                payload = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("Le corps JSON est invalide.") from exc
            if not isinstance(payload, dict):
                raise ValueError("Un objet JSON est attendu.")
            return payload

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/projects":
                from ml_forge.core.config import config
                from ml_forge.storage.database import Database

                projects_dir = config.projects_dir
                global_db = Database(projects_dir / "projects.db")
                projects = []
                records = global_db.query("SELECT * FROM projects WHERE archived = 0 ORDER BY updated_at DESC")
                for record in records:
                    project_dir = projects_dir / record["id"]
                    if not (project_dir / "project.db").is_file():
                        continue
                    project_db = Database(project_dir / "project.db")
                    experiments = project_db.query(
                        "SELECT * FROM experiments WHERE project_id = ? ORDER BY created_at DESC",
                        (record["id"],),
                    )
                    for experiment in experiments:
                        experiment["metrics"] = {
                            row["metric_name"]: row["metric_value"]
                            for row in project_db.query(
                                "SELECT metric_name, metric_value FROM experiment_metrics WHERE experiment_id = ?",
                                (experiment["id"],),
                            )
                        }
                    projects.append({**record, "experiments": experiments})
                self._send_json(200, {"projects": _json_value(projects)})
                return

            relative_path = "index.html" if parsed.path in {"/", "/index.html"} else unquote(parsed.path).lstrip("/")
            requested = (WEB_DIR / relative_path).resolve()
            if WEB_DIR.resolve() not in requested.parents and requested != WEB_DIR.resolve():
                self.send_error(404)
                return
            if not requested.is_file():
                self.send_error(404)
                return
            body = requested.read_bytes()
            content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                path_parts = [part for part in urlparse(self.path).path.split("/") if part]
                if path_parts == ["api", "projects"]:
                    self._create_project(payload)
                    return
                if path_parts == ["api", "import-link"]:
                    result = _import_link(get_manager(), str(payload.get("url", "")).strip(), fetcher=link_fetcher)
                    self._send_json(201, result)
                    return
                if len(path_parts) == 4 and path_parts[:2] == ["api", "projects"] and path_parts[3] == "run":
                    self._run_experiment(path_parts[2], payload)
                    return
                self._send_json(404, {"error": "Route introuvable."})
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def _create_project(self, payload: dict[str, Any]) -> None:
            manager = get_manager()
            name = str(payload.get("name", "")).strip()
            csv_content = payload.get("csv_content")
            filename = str(payload.get("filename", "dataset.csv"))
            target_column = str(payload.get("target_column", "")).strip()
            description = str(payload.get("description", "")).strip()
            if not name or not isinstance(csv_content, str) or not csv_content.strip():
                raise ValueError("Le nom du projet et le contenu CSV sont obligatoires.")
            try:
                frame = pd.read_csv(pd.io.common.StringIO(csv_content))
            except Exception as exc:
                raise ValueError(f"CSV illisible : {exc}") from exc
            if frame.empty or len(frame.columns) < 2:
                raise ValueError("Le CSV doit contenir des lignes et au moins deux colonnes.")
            if target_column not in frame.columns:
                raise ValueError("Choisissez une colonne cible présente dans le CSV.")

            project_id = manager.create_project(name=name, description=description)
            if not project_id:
                raise RuntimeError("La création du projet a échoué.")
            project = manager.open_project(project_id)
            if project is None:
                raise RuntimeError("Impossible d'ouvrir le projet créé.")
            dataset_path = project.dir / "dataset.csv"
            dataset_path.write_text(csv_content, encoding="utf-8-sig")
            dataset_hash = manager._hash_file(str(dataset_path))
            updates = {
                "dataset_path": str(dataset_path),
                "dataset_hash": dataset_hash,
                "target_column": target_column,
            }
            project.db.update("projects", updates, {"id": project_id})
            manager.global_db.update("projects", updates, {"id": project_id})
            from ml_forge.engines.data_engine import DataEngine

            summary = DataEngine().inspect_dataset(frame, target_column=target_column)
            self._send_json(201, {
                "project_id": project_id,
                "filename": Path(filename).name,
                "summary": _json_value(summary),
            })

        def _run_experiment(self, project_id: str, payload: dict[str, Any]) -> None:
            manager = get_manager()
            project = manager.open_project(project_id)
            if project is None:
                self._send_json(404, {"error": "Projet introuvable."})
                return
            dataset_path = project.get_dataset_path()
            target_column = project.get_target_column()
            if not dataset_path or not target_column:
                raise ValueError("Ce projet ne possède pas de dataset et de cible configurés.")
            result = project.run_baseline_experiment(
                dataset_path,
                target_column,
                hypothesis=str(payload.get("hypothesis", "Baseline de référence")),
                method="Baseline ML Forge depuis l'interface web",
            )
            self._send_json(200, {"result": _json_value(result)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    server = create_server()
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"
    print(f"ML Forge Web est disponible sur {url} (Ctrl+C pour arrêter).")
    threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur ML Forge Web.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()