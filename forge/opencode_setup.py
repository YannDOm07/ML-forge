"""opencode configuration: ML Forge as a local MCP server, and restrictive permissions.

In unattended mode `opencode run --auto` approves whatever is not explicitly denied, so every
capability ML Forge does not need is set to "deny" rather than "ask" (docs/ARCHITECTURE.md §6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from forge.config import APP_DIR
from forge.errors import ForgeError
from forge.workspace import Workspace

OPENCODE_FILE = "opencode.json"


def _mcp_server(workspace: Workspace, python: str | None, docker_image: str | None) -> dict:
    if docker_image:  # ML Forge runs in its container; the data live in the mlforge-data volume
        kaggle = Path.home() / ".kaggle"
        command = ["docker", "run", "-i", "--rm", "-v", "mlforge-data:/data",
                   "-v", f"{kaggle}:/home/forge/.kaggle:ro", docker_image, "mcp"]
        return {"type": "local", "command": command, "enabled": True, "timeout": 60000}
    return {
        "type": "local",
        "command": [python or sys.executable, "-m", "forge.mcp_server"],
        "enabled": True,
        "timeout": 30000,
        "environment": {
            "MLFORGE_CONFIG": str(workspace.config.source),
            "PYTHONPATH": str(APP_DIR),
            "PYTHONIOENCODING": "utf-8",
        },
    }


def opencode_config(workspace: Workspace, python: str | None = None, docker_image: str | None = None) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        "instructions": ["AGENTS.md"],
        "mcp": {"mlforge": _mcp_server(workspace, python, docker_image)},
        "permission": {
            "edit": "deny",
            "bash": "deny",
            "webfetch": "deny",
            "websearch": "allow",
            "task": "deny",
            "external_directory": "deny",
            "doom_loop": "deny",
            "read": {"*": "allow", "*.env": "deny", "**/kaggle.json": "deny", "**/.kaggle/**": "deny"},
        },
    }


def write_opencode_config(workspace: Workspace, directory: Path = APP_DIR, docker_image: str | None = None) -> Path:
    path = directory / OPENCODE_FILE
    if path.exists():
        raise ForgeError(f"{path} existe déjà : il n'est pas écrasé. Supprimez-le vous-même pour le régénérer.")
    config = opencode_config(workspace, docker_image=docker_image)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
