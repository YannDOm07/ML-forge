"""Recorded HTTP responses for tests: no test touches the network."""

import io
import zipfile
from pathlib import Path

from forge.web import HttpResponse, NetworkError


def html(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse("", status, {"content-type": "text/html; charset=utf-8"}, body.encode("utf-8"))


def raw(body: bytes | str, status: int = 200, content_type: str = "text/csv") -> HttpResponse:
    data = body.encode("utf-8") if isinstance(body, str) else body
    return HttpResponse("", status, {"content-type": content_type}, data)


def json_response(body: str, status: int = 200) -> HttpResponse:
    return raw(body, status, "application/json")


def zip_of(files: dict[str, Path | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, content in files.items():
            bundle.writestr(name, content.read_bytes() if isinstance(content, Path) else content)
    return buffer.getvalue()


def kaggle_routes(competition: dict, url: str, slug: str, metric: str, download: HttpResponse | None = None) -> dict:
    """A Kaggle competition as seen by ML Forge: JavaScript page, API metadata, file list, zip."""
    import json

    from forge.kaggle import download_url, files_url, metadata_url

    files = [{"name": p.name, "totalBytes": p.stat().st_size} for p in competition.values()]
    metadata = [{"ref": url, "title": slug, "evaluationMetric": metric, "maxDailySubmissions": 5,
                 "isKernelsSubmissionsOnly": False}]
    archive = raw(zip_of({p.name: p for p in competition.values()}), content_type="application/zip")
    return {
        url: html("<div id='app'></div>"),
        metadata_url(slug): raw(json.dumps(metadata), content_type="application/json"),
        files_url(slug): raw(json.dumps(files), content_type="application/json"),
        download_url(slug): download or archive,
    }


class FakeTransport:
    def __init__(self, routes: dict[str, HttpResponse]):
        self.routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def _response(self, url: str) -> HttpResponse:
        response = self.routes.get(url, raw("introuvable", 404, "text/plain"))
        return HttpResponse(url, response.status, response.headers, response.body)

    def get(self, url, headers, max_bytes):
        self.calls.append(("GET", url, dict(headers)))
        response = self._response(url)
        if len(response.body) > max_bytes:
            raise NetworkError("trop grand")
        return response

    def download(self, url, headers, dest, max_bytes):
        self.calls.append(("DOWNLOAD", url, dict(headers)))
        response = self._response(url)
        if len(response.body) > max_bytes:
            raise NetworkError("trop grand")
        if 200 <= response.status < 300:
            dest.write_bytes(response.body)
        return HttpResponse(url, response.status, response.headers, b"" if response.status < 300 else response.body)
