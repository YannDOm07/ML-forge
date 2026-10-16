"""Tests d'intégration de l'interface web locale."""

import json
import threading
from http.client import HTTPConnection

from ml_forge.core.config import config
from ml_forge.web_app import create_server


def test_web_flow_creates_project_and_runs_baseline(tmp_path):
    original_projects_dir = config.projects_dir
    config.projects_dir = tmp_path / "projects"
    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_address[1])

    try:
        csv_content = "feature,target\n" + "\n".join(
            f"{index},{1 if index >= 10 else 0}" for index in range(20)
        )
        connection.request(
            "POST",
            "/api/projects",
            body=json.dumps({
                "name": "Projet web",
                "description": "Test de bout en bout",
                "filename": "dataset.csv",
                "csv_content": csv_content,
                "target_column": "target",
            }),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        created = json.loads(response.read())
        assert response.status == 201
        assert created["summary"]["rows"] == 20

        connection.request(
            "POST",
            f"/api/projects/{created['project_id']}/run",
            body=json.dumps({"hypothesis": "Établir une baseline."}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())["result"]
        assert response.status == 200
        assert result["model"] == "LogisticRegression"
        assert "accuracy" in result["metrics"]

        connection.request("GET", "/api/projects")
        response = connection.getresponse()
        projects = json.loads(response.read())["projects"]
        assert response.status == 200
        assert projects[0]["experiments"][0]["status"] == "completed"
        assert "accuracy" in projects[0]["experiments"][0]["metrics"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
        config.projects_dir = original_projects_dir


def test_web_server_serves_html():
    server = create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_address[1])

    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert "ML FORGE" in body
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_link_import_creates_project_and_runs_detected_target(tmp_path):
    original_projects_dir = config.projects_dir
    config.projects_dir = tmp_path / "projects"
    page_url = "https://competition.example/challenge"
    dataset_url = "https://competition.example/files/train.csv"
    responses = {
        page_url: (
            b'<html><head><title>Challenge ML</title><meta name="description" content="Predict the label"></head>'
            b'<body>Competition instructions and evaluation rules<a href="/files/train.csv">Training data</a></body></html>',
            "text/html",
            page_url,
        ),
        dataset_url: (
            ("feature,label\n" + "\n".join(f"{index},{int(index > 9)}" for index in range(20))).encode(),
            "text/csv",
            dataset_url,
        ),
    }
    server = create_server(port=0, link_fetcher=lambda url: responses[url])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_address[1])

    try:
        connection.request(
            "POST",
            "/api/import-link",
            body=json.dumps({"url": page_url}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        imported = json.loads(response.read())
        assert response.status == 201, imported
        assert imported["project_name"] == "Challenge ML"
        assert imported["target_column"] == "label"
        assert imported["status"] == "baseline_complete"
        assert imported["experiment"]["metrics"]["accuracy"] >= 0
        brief_path = tmp_path / "projects" / imported["project_id"] / "source_brief.md"
        assert brief_path.is_file()
        assert "Competition instructions" in brief_path.read_text(encoding="utf-8")
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
        config.projects_dir = original_projects_dir


def test_link_import_does_not_guess_unknown_target_or_accept_file_url(tmp_path):
    original_projects_dir = config.projects_dir
    config.projects_dir = tmp_path / "projects"
    csv_url = "https://data.example/project.csv"
    server = create_server(
        port=0,
        link_fetcher=lambda _: (
            b"feature,decision_code\n1,4\n2,7\n3,4\n4,7\n5,4\n6,7\n7,4\n8,7\n9,4\n10,7\n",
            "text/csv",
            csv_url,
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_address[1])

    try:
        connection.request(
            "POST",
            "/api/import-link",
            body=json.dumps({"url": csv_url}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        imported = json.loads(response.read())
        assert response.status == 201
        assert imported["status"] == "project_created_target_unknown"
        assert imported["target_column"] is None

        connection.request(
            "POST",
            "/api/import-link",
            body=json.dumps({"url": "file:///etc/passwd"}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.status == 400
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()
        config.projects_dir = original_projects_dir