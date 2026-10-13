"""Web dashboard for ML Forge (FastAPI + HTMX)."""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from forge import __version__
from forge.config import load_config
from forge.workspace import Workspace, open_workspace
from forge.projects import list_projects, open_project, create_project
from forge.experiments import list_experiments, get_experiment
from forge.report import decisions_by_experiment
from forge.campaign import active_campaign
from forge.intake import competition_card
from forge.runner import latest_protocol, current_best
from forge.actions import open_actions
from forge.datasources import check_data_sources


APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

workspace: Workspace | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global workspace
    config_path = os.environ.get("MLFORGE_CONFIG")
    config = load_config(Path(config_path) if config_path else None)
    workspace = open_workspace(config)
    yield
    workspace = None


app = FastAPI(
    title="ML Forge Dashboard",
    version=__version__,
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _require_workspace() -> Workspace:
    if workspace is None:
        raise HTTPException(500, "Workspace not initialized")
    return workspace


# ---------- API Models ----------

class ProjectCreate(BaseModel):
    name: str
    url: str | None = None


class ExperimentResponse(BaseModel):
    id: str
    status: str
    method: str
    hypothesis: str
    metric_mean: float | None = None
    metric_std: float | None = None
    delta: float | None = None
    decision: str | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    status: str
    competition_url: str | None
    created_at: str
    experiment_counts: dict[str, int]
    data_sources: list[dict]
    open_actions: list[dict]
    active_campaign: dict | None = None


# ---------- API Endpoints ----------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    ws = _require_workspace()
    projects = list_projects(ws)
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "projects": projects, "version": __version__},
    )


@app.get("/api/projects")
async def api_projects():
    ws = _require_workspace()
    projects = list_projects(ws, include_archived=False)
    return [{"id": p.id, "name": p.name, "status": p.status, "url": p.competition_url, "created": p.created_at} for p in projects]


@app.get("/api/projects/{project_id}", response_model=ProjectResponse)
async def api_project(project_id: str):
    ws = _require_workspace()
    project = open_project(ws, project_id)

    experiments = list_experiments(project.db_path)
    counts: dict[str, int] = {}
    for exp in experiments:
        counts[exp.status] = counts.get(exp.status, 0) + 1

    decisions = decisions_by_experiment(project)
    sources = check_data_sources(project.db_path)
    actions = open_actions(project)
    campaign = active_campaign(project)

    return {
        "id": project.id,
        "name": project.name,
        "status": project.status,
        "competition_url": project.competition_url,
        "created_at": project.created_at,
        "experiment_counts": counts,
        "data_sources": [
            {"path": s.path, "state": s.state, "bytes": s.bytes, "rows": s.rows, "cols": s.cols}
            for s in sources
        ],
        "open_actions": actions,
        "active_campaign": campaign,
    }


@app.get("/api/projects/{project_id}/experiments")
async def api_experiments(project_id: str):
    ws = _require_workspace()
    project = open_project(ws, project_id)
    experiments = list_experiments(project.db_path)
    decisions = decisions_by_experiment(project)

    result = []
    for exp in experiments:
        dec = decisions.get(exp.id)
        result.append({
            "id": exp.id,
            "status": exp.status,
            "method": exp.method,
            "hypothesis": exp.hypothesis,
            "metric_mean": dec["metric_mean"] if dec else None,
            "metric_std": dec["metric_std"] if dec else None,
            "delta": dec["delta"] if dec else None,
            "decision": dec["status"] if dec else None,
        })
    return result


@app.get("/api/projects/{project_id}/card")
async def api_card(project_id: str):
    ws = _require_workspace()
    project = open_project(ws, project_id)
    return competition_card(project)


@app.get("/api/projects/{project_id}/protocol")
async def api_protocol(project_id: str):
    ws = _require_workspace()
    project = open_project(ws, project_id)
    protocol = latest_protocol(project)
    if not protocol:
        raise HTTPException(404, "No protocol yet")
    return {
        "id": protocol.id,
        "task": protocol.task,
        "splits": protocol.splits,
        "cv": protocol.cv,
        "metric": protocol.task.metric,
    }


@app.get("/api/projects/{project_id}/best")
async def api_best(project_id: str):
    ws = _require_workspace()
    project = open_project(ws, project_id)
    protocol = latest_protocol(project)
    if not protocol:
        raise HTTPException(404, "No protocol yet")
    best_id = current_best(project, protocol.id)
    if not best_id:
        raise HTTPException(404, "No best candidate")
    exp = get_experiment(project.db_path, best_id)
    return {
        "experiment_id": best_id,
        "method": exp.method,
        "hypothesis": exp.hypothesis,
    }


@app.post("/api/projects")
async def api_create_project(data: ProjectCreate):
    ws = _require_workspace()
    project = create_project(ws, data.name, data.url)
    return {"id": project.id, "name": project.name, "path": str(project.path)}


@app.post("/api/projects/{project_id}/campaign/start")
async def api_start_campaign(project_id: str, driver: str = Query("builtin"), interactive: bool = False):
    ws = _require_workspace()
    project = open_project(ws, project_id)

    from forge.campaign import BuiltinDriver, OpencodeDriver, run_campaign
    drv = OpencodeDriver() if driver == "opencode" else BuiltinDriver()
    mode = "interactive" if interactive else "unattended"
    campaign = run_campaign(ws, project, drv, mode)
    return campaign


@app.post("/api/projects/{project_id}/campaign/stop")
async def api_stop_campaign(project_id: str):
    ws = _require_workspace()
    project = open_project(ws, project_id)
    from forge.campaign import request_stop
    request_stop(project)
    return {"message": "Stop requested"}


# ---------- CLI Entry ----------

def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = sys.argv[1:] if argv is None else argv
    host = "127.0.0.1"
    port = 8080
    reload = False

    i = 0
    while i < len(args):
        if args[i] == "--host":
            host = args[i + 1]
            i += 2
        elif args[i] == "--port":
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--reload":
            reload = True
            i += 1
        else:
            i += 1

    config_path = os.environ.get("MLFORGE_CONFIG")
    config = load_config(Path(config_path) if config_path else None)
    ws = open_workspace(config)

    print(f"ML Forge Dashboard → http://{host}:{port}")
    print(f"Data root: {ws.root}")

    uvicorn.run("forge.web_server:app", host=host, port=port, reload=reload, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())