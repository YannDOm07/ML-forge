import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from forge.projects import create_project
from forge.ui.server import _handler, _host_is_local


def request(server, path, method="GET", host="127.0.0.1"):
    url = f"http://127.0.0.1:{server.server_port}{path}"
    req = urllib.request.Request(url, method=method, headers={"Host": host})
    try:
        return urllib.request.urlopen(req, timeout=3)
    except urllib.error.HTTPError as exc:
        return exc


def test_host_validation():
    assert _host_is_local("localhost")
    assert _host_is_local("127.0.0.1:8765")
    assert not _host_is_local("localhost.evil.example")
    assert not _host_is_local("127.0.0.1:99999")
    assert not _host_is_local(None)


def test_read_only_local_http_api_and_security_headers(workspace):
    project = create_project(workspace, "UI")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(workspace))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = request(server, "/")
        assert response.status == 200
        assert "default-src 'none'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"

        response = request(server, f"/api/state?project_id={project.id}")
        data = json.loads(response.read())
        assert data["selected_project"] == project.id
        assert "experiments" in data

        assert request(server, "/api/state", host="localhost.evil").status == 403
        assert request(server, "/api/state", method="POST").status == 405
        assert request(server, "/not-found").status == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
