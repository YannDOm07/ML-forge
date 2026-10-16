# OpenCode + ML Forge

OpenCode supplies the conversation and uses whichever model is currently selected in OpenCode. ML Forge exposes its local project and ML operations as MCP tools; it does not call Ollama directly.

The project-root `opencode.json` starts the MCP server using this checkout's Python environment. OpenCode must be launched with `C:\DEV\ML FORGE` as its working directory. After changing the MCP configuration, restart OpenCode and check the connection with:

```powershell
opencode mcp list
```

Available tools:

- `list_projects` and `get_project`
- `create_project`
- `inspect_dataset`
- `run_baseline_experiment`
- `list_experiments`

Example requests in the OpenCode chat:

> List my ML Forge projects and tell me their IDs.

> Inspect `C:\data\train.csv` for project `<project_id>`, using `target` as the target column. Summarize the risks and recommend next steps.

> Run a baseline experiment for project `<project_id>` on `C:\data\train.csv`, with `target` as the target column, then report the saved metrics.

OpenCode decides when to call the tools. The tools operate on local files and project databases. The MCP server does not expose project deletion.

Install or restore the MCP dependency with:

```powershell
C:\DEV\ML FORGE\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
