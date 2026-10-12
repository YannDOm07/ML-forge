"""MCP server (real JSON-RPC over stdio), test 26 (minimal sending rule), approvals out of band."""

import json
import os
import subprocess
import sys

import pytest

from fake_web import FakeTransport, html, raw
from forge.acquisition import ExternalResource, request_external
from forge.approvals import ApprovalPending, answer_request, pending_requests, stored_approver
from forge.config import APP_DIR
from forge.datasources import register_data_source
from forge.demo import make_synthetic_competition
from forge.errors import ApprovalRequired
from forge.intake import paste_text, read_competition, record_fact
from forge.mcp_server import Server
from forge.opencode_setup import opencode_config, write_opencode_config
from forge.projects import create_project
from forge.runner import define_protocol, profile_data
from forge.task import TaskSpec


def rpc(process, message):
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()
    return json.loads(process.stdout.readline())


@pytest.fixture
def server_process(config_path):
    env = {**os.environ, "MLFORGE_CONFIG": str(config_path), "PYTHONPATH": str(APP_DIR), "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        [sys.executable, "-m", "forge.mcp_server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", cwd=APP_DIR, env=env,
    )
    yield process
    process.stdin.close()
    process.wait(timeout=30)


def test_mcp_protocol_over_stdio(server_process):
    init = rpc(server_process, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "test", "version": "1"}}})
    assert init["result"]["serverInfo"]["name"] == "mlforge"
    assert init["result"]["protocolVersion"] == "2025-06-18"
    server_process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")

    tools = {t["name"] for t in rpc(server_process, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]}
    assert {"campaign_next", "experiment_submit", "experiment_run", "fact_record", "external_request"} <= tools
    assert not any("approve" in name for name in tools)  # the LLM can never approve

    created = rpc(server_process, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "project_create",
                                              "arguments": {"name": "Flows", "url": "https://example.org/c"}}})
    project_id = json.loads(created["result"]["content"][0]["text"])["id"]
    following = rpc(server_process, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                     "params": {"name": "campaign_next", "arguments": {"project_id": project_id}}})
    assert json.loads(following["result"]["content"][0]["text"])["étape"] == "intake"

    unknown = rpc(server_process, {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "rm_rf"}})
    assert unknown["result"]["isError"]
    missing = rpc(server_process, {"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                                   "params": {"name": "experiment_run", "arguments": {"project_id": project_id}}})
    assert missing["result"]["isError"] and "experiment_id" in missing["result"]["content"][0]["text"]
    error = rpc(server_process, {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                 "params": {"name": "campaign_next", "arguments": {"project_id": "inexistant"}}})
    assert error["result"]["isError"] and "Projet inconnu" in error["result"]["content"][0]["text"]
    assert rpc(server_process, {"jsonrpc": "2.0", "id": 8, "method": "nope"})["error"]["code"] == -32601


@pytest.fixture
def data_project(workspace, tmp_path):
    paths = make_synthetic_competition(tmp_path / "comp", n_train=120, n_test=30)
    project = create_project(workspace, "Données", "https://example.org/c")
    for name in ("train", "test"):
        register_data_source(project.db_path, paths[name], "official")
    task = TaskSpec("binary", "target", "roc_auc", str(paths["train"]), str(paths["test"]), id_column="id")
    define_protocol(workspace, project, task)
    profile_data(workspace, project, task)
    return project


def call(server, name, **arguments):
    result = server.call({"name": name, "arguments": arguments})
    return result["content"][0]["text"], result["isError"]


def test_no_raw_rows_unless_sharing_is_explicitly_allowed(workspace, data_project):
    server = Server(workspace)
    text, _ = call(server, "data_sample", project_id=data_project.id, rows=100)
    assert json.loads(text)["lignes"] == []

    source = paste_text(data_project, "Participants may share the competition data with third parties.")
    record_fact(data_project, "rules.data_sharing", "allowed", source,
                "Participants may share the competition data with third parties.", "user")
    text, _ = call(server, "data_sample", project_id=data_project.id, rows=100)
    assert len(json.loads(text)["lignes"]) == 20  # capped by llm.max_raw_rows

    profile, error = call(server, "data_profile", project_id=data_project.id)
    assert not error and "rows" in profile


def test_inferred_permission_does_not_unlock_rows(workspace, data_project):
    url = "https://example.org/c"
    read_competition(workspace, data_project, transport=FakeTransport({url: html("<p>Sharing data is allowed.</p>")}))
    text, _ = call(Server(workspace), "data_sample", project_id=data_project.id, rows=5)
    assert json.loads(text)["lignes"] == []


def test_outputs_are_bounded_and_framed(workspace, data_project):
    url = "https://example.org/c"
    long_page = "<p>" + "Lorem ipsum dolor sit amet. " * 2000 + "</p>"
    read_competition(workspace, data_project, transport=FakeTransport({url: html(long_page)}))
    server = Server(workspace)
    server.max_chars = 2000
    text, _ = call(server, "page_read", project_id=data_project.id, url=url)
    assert len(text) <= 2000
    assert text.startswith("<<CONTENU EXTERNE") and "offset=" in text
    history, _ = call(server, "experiment_history", project_id=data_project.id)
    assert len(history) <= 2000


def test_approval_is_given_out_of_band_by_the_user(workspace):
    project = create_project(workspace, "Accord", "https://example.org/c")
    read_competition(workspace, project, transport=FakeTransport({"https://example.org/c": html("<p>Hello.</p>")}))
    resource = ExternalResource("data", "https://open.example/geo.csv", "Open Geo", "CC-BY-4.0", 1, "géographie", "faible")
    transport = FakeTransport({resource.url: raw("a,b\n1,2\n")})
    with pytest.raises(ApprovalPending, match="python -m forge approve"):
        request_external(workspace, project, resource, stored_approver(project), "agent", transport)
    assert transport.calls == []
    [pending] = pending_requests(project)
    answer_request(project, pending["id"], approved=True)
    assert request_external(workspace, project, resource, stored_approver(project), "agent", transport).status == "done"

    other = ExternalResource("data", "https://open.example/other.csv", "X", "?", 1, "r", "élevé")
    with pytest.raises(ApprovalPending):
        request_external(workspace, project, other, stored_approver(project), "agent", transport)
    answer_request(project, pending_requests(project)[0]["id"], approved=False)
    with pytest.raises(ApprovalRequired):
        request_external(workspace, project, other, stored_approver(project), "agent", transport)


def test_opencode_configuration_denies_what_is_not_needed(workspace, tmp_path):
    config = opencode_config(workspace)
    permissions = config["permission"]
    assert {permissions[k] for k in ("edit", "bash", "webfetch", "task", "external_directory")} == {"deny"}
    assert permissions["read"]["**/kaggle.json"] == "deny"
    assert config["mcp"]["mlforge"]["command"][-1] == "forge.mcp_server"
    path = write_opencode_config(workspace, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["mcp"]["mlforge"]["type"] == "local"
    with pytest.raises(Exception, match="pas écrasé"):
        write_opencode_config(workspace, tmp_path)
    docker = opencode_config(workspace, docker_image="mlforge:1.0.0")["mcp"]["mlforge"]["command"]
    assert docker[:3] == ["docker", "run", "-i"] and docker[-2:] == ["mlforge:1.0.0", "mcp"]
    assert "mlforge-data:/data" in docker
