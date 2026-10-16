"""ML Forge UI server — stdlib only, read-only, local binding."""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import urllib.parse
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from forge.config import load_config
from forge.workspace import Workspace, open_workspace
from forge.ui import queries
from forge.ui import viewmodel


# ---------- Configuration ----------

DEFAULT_PORT = 8765
BIND_HOST = "127.0.0.1"
POLL_INTERVAL_MS = 5000

UI_DIR = Path(__file__).parent / "static"


# ---------- Security headers ----------

CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'"
)

SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()",
}


# ---------- Request Handler ----------

class UIHandler(BaseHTTPRequestHandler):
    workspace: Workspace
    project_id: str | None

    def __init__(self, *args, workspace: Workspace, project_id: str | None, **kwargs):
        self.workspace = workspace
        self.project_id = project_id
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        # Quiet by default
        pass

    def _check_host(self) -> bool:
        """Verify Host header against DNS rebinding."""
        host = self.headers.get("Host", "")
        if not _host_is_local(host):
            return False
        if host.startswith("["):
            suffix = host[host.find("]") + 1:]
            port = suffix[1:] if suffix else None
        elif ":" in host:
            port = host.rsplit(":", 1)[1]
        else:
            port = None
        if port is None:
            return True
        return port.isdigit() and int(port) == self.server.server_port

    def _send_json(self, data: Any, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

    def _send_html(self, content: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _send_file(self, path: Path, status: int = 200) -> None:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        self.send_response(status)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        with path.open("rb") as f:
            self.wfile.write(f.read())

    def _send_error(self, status: int, message: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"))

    def do_GET(self) -> None:
        if not self._check_host():
            self._send_error(403, "Host non autorisé")
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Static files
        if path.startswith("/static/"):
            file_path = UI_DIR / path[len("/static/"):]
            if file_path.is_file() and not file_path.name.startswith("."):
                self._send_file(file_path)
            else:
                self._send_error(404, "Fichier non trouvé")
            return

        # API routes
        if path == "/api/projects":
            self._handle_projects()
            return

        if path == "/api/state":
            from forge.ui.viewmodel import build_viewmodel

            project_id = urllib.parse.parse_qs(parsed.query).get("project_id", [None])[0]
            try:
                self._send_json(build_viewmodel(self.workspace, project_id or self.project_id))
            except ValueError as exc:
                self._send_error(404, str(exc))
            return

        if path == "/api/project/overview":
            self._handle_overview()
            return

        if path == "/api/project/campaign":
            self._handle_campaign()
            return

        if path == "/api/project/experiments":
            self._handle_experiments()
            return

        if path.startswith("/api/project/experiment/"):
            exp_id = path.split("/")[-1]
            self._handle_experiment_detail(exp_id)
            return

        if path == "/api/project/hypotheses":
            self._handle_hypotheses()
            return

        if path == "/api/project/data":
            self._handle_data()
            return

        if path == "/api/project/validation":
            self._handle_validation()
            return

        if path == "/api/project/deliverables":
            self._handle_deliverables()
            return

        if path == "/api/project/memory":
            self._handle_memory()
            return

        if path == "/api/project/log":
            self._handle_log()
            return

        # Root -> index.html
        if path == "/" or path == "":
            index_path = UI_DIR / "index.html"
            if index_path.is_file():
                self._send_file(index_path)
            else:
                self._send_error(500, "index.html manquant")
            return

        self._send_error(404, "Route non trouvée")

    def do_POST(self) -> None:
        if not self._check_host():
            self._send_error(403, "Host non autorisé")
            return
        self._send_error(405, "Méthode non autorisée")

    # ---------- API Handlers ----------

    def _handle_projects(self) -> None:
        projects = queries.list_projects(self.workspace.global_db)
        self._send_json({"projects": projects, "current": self.project_id})

    def _get_project_db(self) -> Path | None:
        if not self.project_id:
            return None
        project = queries.get_project(self.workspace.global_db, self.project_id)
        if not project:
            return None
        return self.workspace.projects_dir / self.project_id / "project.db"

    def _handle_overview(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        project = queries.get_project(self.workspace.global_db, self.project_id)
        campaign = queries.get_active_campaign(project_db)
        protocol = queries.get_latest_protocol(project_db)
        profile = None
        if protocol:
            spec = json.loads(protocol["spec_json"])
            data_hash = spec.get("data_hash")
            if data_hash:
                profile = queries.get_latest_data_profile(project_db, data_hash)

        card_raw = queries.get_competition_facts(project_db)
        card = self._build_card(card_raw)

        experiments = queries.get_experiments(project_db)
        decisions = queries.get_experiment_decisions(project_db)
        dec_map = {d["experiment_id"]: d for d in decisions}

        best = None
        for exp in experiments:
            if dec_map.get(exp["id"], {}).get("status") in ("accepted", "baseline"):
                best = {"id": exp["id"], "score": dec_map[exp["id"]]["metric_mean"],
                        "delta": dec_map[exp["id"]].get("delta")}
                break

        recent = []
        for exp in experiments[-5:]:
            dec = dec_map.get(exp["id"])
            recent.append({
                "id": exp["id"],
                "method": exp["method"],
                "hypothesis": exp["hypothesis"],
                "status": exp["status"],
                "decision": dec["status"] if dec else None,
                "score": dec["metric_mean"] if dec else None,
                "delta": dec.get("delta") if dec else None,
                "created_at": exp["created_at"],
            })

        rejected = []
        for exp in reversed(experiments):
            dec = dec_map.get(exp["id"])
            if dec and dec["status"] in ("rejected", "not_significant"):
                rejected.append({
                    "id": exp["id"],
                    "method": exp["method"],
                    "hypothesis": exp["hypothesis"][:160],
                    "decision": dec["status"],
                    "score": dec["metric_mean"],
                    "delta": dec.get("delta"),
                })
            if len(rejected) >= 10:
                break

        actions = queries.get_user_actions(project_db)
        approvals = queries.get_approval_requests(project_db)
        memory = queries.get_knowledge_items(self.workspace.global_db)

        data = viewmodel.adapt_overview(
            project, campaign, protocol, profile, card,
            best, recent, rejected, actions, approvals, {"summary": f"{len(memory)} leçons"}
        )
        self._send_json(data)

    def _handle_campaign(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        project = queries.get_project(self.workspace.global_db, self.project_id)
        campaign = queries.get_active_campaign(project_db)
        protocol = queries.get_latest_protocol(project_db)
        card_raw = queries.get_competition_facts(project_db)
        card = self._build_card(card_raw)

        data = viewmodel.adapt_campaign(campaign, project, protocol, card, self.workspace.config.values)
        self._send_json(data)

    def _handle_experiments(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        experiments = queries.get_experiments(project_db)
        decisions = queries.get_experiment_decisions(project_db)
        protocol = queries.get_latest_protocol(project_db)

        data = viewmodel.adapt_experiments(experiments, decisions, protocol["id"] if protocol else None)
        self._send_json(data)

    def _handle_experiment_detail(self, exp_id: str) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        exp = queries.get_experiment(project_db, exp_id)
        if not exp:
            self._send_error(404, "Expérience non trouvée")
            return

        dec = queries.get_experiment_decision(project_db, exp_id)
        metrics = queries.get_experiment_metrics(project_db, exp_id)
        protocol = queries.get_latest_protocol(project_db)

        data = viewmodel.adapt_experiment_detail(exp, dec, metrics, protocol)
        self._send_json(data)

    def _handle_hypotheses(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        experiments = queries.get_experiments(project_db)
        decisions = queries.get_experiment_decisions(project_db)

        data = viewmodel.adapt_hypotheses(experiments, decisions)
        self._send_json(data)

    def _handle_data(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        card_raw = queries.get_competition_facts(project_db)
        card = self._build_card(card_raw)
        data_sources = queries.get_data_sources(project_db)
        protocol = queries.get_latest_protocol(project_db)
        profile = None
        if protocol:
            spec = json.loads(protocol["spec_json"])
            data_hash = spec.get("data_hash")
            if data_hash:
                profile = queries.get_latest_data_profile(project_db, data_hash)

        data = viewmodel.adapt_data(str(project_db), card, data_sources, profile)
        self._send_json(data)

    def _handle_validation(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        protocol = queries.get_latest_protocol(project_db)
        experiments = queries.get_experiments(project_db)
        decisions = queries.get_experiment_decisions(project_db)
        all_metrics = queries.get_all_experiment_metrics(project_db)

        data = viewmodel.adapt_validation(protocol, experiments, decisions, all_metrics)
        self._send_json(data)

    def _handle_deliverables(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        project = queries.get_project(self.workspace.global_db, self.project_id)
        card_raw = queries.get_competition_facts(project_db)
        card = self._build_card(card_raw)
        submissions = queries.get_submissions(project_db)

        data = viewmodel.adapt_deliverables(card, submissions, project["path"])
        self._send_json(data)

    def _handle_memory(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        knowledge = queries.get_knowledge_items(self.workspace.global_db)
        evidence = queries.get_knowledge_evidence(self.workspace.global_db, self.project_id)
        calibration = queries.get_estimate_calibration(self.workspace.global_db, self.project_id)

        data = viewmodel.adapt_memory(knowledge, evidence, calibration)
        self._send_json(data)

    def _handle_log(self) -> None:
        project_db = self._get_project_db()
        if not project_db:
            self._send_error(404, "Projet non trouvé")
            return

        events = queries.get_execution_events(project_db)
        network = queries.get_network_log(self.workspace.global_db, self.project_id)
        compliance = queries.get_compliance_log(project_db)
        config_hist = queries.get_configuration_history(self.workspace.global_db)

        data = viewmodel.adapt_log(events, network, compliance, config_hist)
        self._send_json(data)

    # ---------- Helpers ----------

    def _build_card(self, facts: list[dict]) -> dict:
        """Reconstruct competition card from facts."""
        from forge.intake import FIELDS, STATUS_RANK

        fields = {}
        for name, spec in FIELDS.items():
            field_facts = [f for f in facts if f["field"] == name]
            if spec.kind == "list":
                fields[name] = [{"value": json.loads(f["value_json"]), "status": f["status"],
                                 "source_url": f["source_url"], "quote": f["quote"]}
                                for f in field_facts]
            else:
                if field_facts:
                    best = max(field_facts, key=lambda f: (STATUS_RANK.get(f["status"], 0), f["id"]))
                    fields[name] = {"value": json.loads(best["value_json"]), "status": best["status"],
                                    "source_url": best["source_url"], "quote": best["quote"]}
                else:
                    fields[name] = None

        from forge.intake import identify_platform
        platform = identify_platform(self.workspace.config.values.get("project", {}).get("competition_url", ""))

        return {
            "platform": platform.name,
            "slug": platform.slug,
            "fields": fields,
            "critical_unknowns": [n for n, s in FIELDS.items() if s.critical and
                                   (fields[n] is None or fields[n]["status"] == "inferred")],
            "support_level": "validated" if fields.get("problem_type", {}).get("value") in ("binary", "multiclass", "regression") else "experimental",
        }


# ---------- Server factory ----------

def create_handler(workspace: Workspace, project_id: str | None):
    class Handler(UIHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, workspace=workspace, project_id=project_id, **kwargs)
    return Handler


def _handler(workspace: Workspace):
    return create_handler(workspace, None)


def _host_is_local(host: str | None) -> bool:
    if not host:
        return False
    hostname = host
    if host.startswith("["):
        close = host.find("]")
        if close < 0:
            return False
        hostname = host[:close + 1]
        suffix = host[close + 1:]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return False
        port = suffix[1:] if suffix else None
        valid_hostname = hostname.lower() == "[::1]"
    else:
        if ":" in host:
            hostname, port = host.rsplit(":", 1)
            if not port.isdigit():
                return False
        else:
            port = None
        valid_hostname = hostname.lower() in ("localhost", "127.0.0.1")
    return valid_hostname and (port is None or 1 <= int(port) <= 65535)


def run_server(workspace: Workspace, project_id: str | None, port: int = DEFAULT_PORT) -> None:
    handler = create_handler(workspace, project_id)
    server = HTTPServer((BIND_HOST, port), handler)
    print(f"ML Forge UI → http://{BIND_HOST}:{port}")
    print(f"Projet: {project_id or '(sélection manuelle)'}")
    print(f"Data root: {workspace.root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
        server.shutdown()


def serve(workspace: Workspace, project_id: str | None, port: int = DEFAULT_PORT) -> None:
    run_server(workspace, project_id, port)


# ---------- CLI Entry ----------

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="forge ui", description="Interface web locale ML Forge")
    parser.add_argument("--project", help="ID du projet à afficher")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port d'écoute (défaut: {DEFAULT_PORT})")
    parser.add_argument("--config", type=Path, help="Fichier de configuration ML Forge")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    workspace = open_workspace(config)

    # If no project specified, use the first active one
    project_id = args.project
    if not project_id:
        projects = queries.list_projects(workspace.global_db)
        if projects:
            project_id = projects[0]["id"]
            print(f"Projet par défaut: {project_id}")

    run_server(workspace, project_id, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())