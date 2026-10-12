"""Tests 11 (no network from the sandbox) and 12 (time and memory limits, clean stop)."""

import os
import textwrap

import pytest

from forge.sandbox import SandboxLimits, run_in_sandbox

LIMITS = SandboxLimits(timeout_s=60, memory_mb=1024)


def run(tmp_path, code, limits=LIMITS, read_paths=()):
    workdir = tmp_path / "work"
    workdir.mkdir(exist_ok=True)
    script = workdir / "script.py"
    script.write_text(textwrap.dedent(code), encoding="utf-8")
    return run_in_sandbox(script, workdir, list(read_paths), limits)


def test_allowed_script_runs_and_writes_in_its_workdir(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    result = run(tmp_path, f"""
        from pathlib import Path
        text = Path(r"{data}").read_text()
        Path("out.txt").write_text(text + "ok")
        print("fini")
    """, read_paths=[data])
    assert result.status == "ok", result.stderr
    assert "fini" in result.stdout
    assert (tmp_path / "work" / "out.txt").read_text() == "a\n1\nok"


@pytest.mark.parametrize("code", [
    "import socket; socket.create_connection(('127.0.0.1', 9), timeout=1)",
    "import socket; socket.getaddrinfo('example.org', 443)",
    "import urllib.request; urllib.request.urlopen('https://example.org', timeout=2)",
])
def test_network_is_blocked(tmp_path, code):
    result = run(tmp_path, code)
    assert result.status == "blocked"
    assert "réseau" in result.stderr


def test_process_creation_is_blocked(tmp_path):
    result = run(tmp_path, "import subprocess; subprocess.run(['cmd', '/c', 'echo evade'])")
    assert result.status == "blocked"
    assert "evade" not in result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Job Object Windows")
def test_job_object_blocks_children_even_without_audit_event(tmp_path):
    result = run(tmp_path, """
        import _winapi
        _winapi.CreateProcess(None, 'cmd /c echo evade', None, None, False, 0, None, None, None)
    """)
    assert not result.ok
    assert "evade" not in result.stdout


def test_write_outside_workdir_is_blocked(tmp_path):
    target = tmp_path / "outside.txt"
    result = run(tmp_path, f"open(r'{target}', 'w').write('x')")
    assert result.status == "blocked"
    assert not target.exists()


def test_reading_unlisted_files_is_blocked(tmp_path):
    secret = tmp_path / "kaggle.json"
    secret.write_text('{"key": "secret"}', encoding="utf-8")
    result = run(tmp_path, f"print(open(r'{secret}').read())")
    assert result.status == "blocked"
    assert "secret" not in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="/proc Linux")
def test_system_info_is_readable_but_not_the_parent_environment(tmp_path):
    allowed = run(tmp_path, "import os; print(len(open('/proc/cpuinfo').read()) > 0, os.cpu_count())")
    assert allowed.status == "ok", allowed.stderr
    result = run(tmp_path, "import os; print(open(f'/proc/{os.getppid()}/environ').read())")
    assert result.status == "blocked"


def test_secrets_are_not_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_KEY", "ne-doit-pas-fuiter")
    result = run(tmp_path, "import os; print(sorted(os.environ))")
    assert result.status == "ok", result.stderr
    assert "KAGGLE_KEY" not in result.stdout
    assert "ne-doit-pas-fuiter" not in result.stdout


def test_timeout_stops_the_process(tmp_path):
    result = run(tmp_path, "while True: pass", SandboxLimits(timeout_s=2, memory_mb=1024))
    assert result.status == "timeout"
    assert result.duration_s < 30


def test_memory_limit_stops_the_allocation(tmp_path):
    result = run(tmp_path, "data = bytearray(3 * 1024**3)", SandboxLimits(timeout_s=60, memory_mb=400))
    assert result.status == "memory_exceeded"


def test_script_error_is_reported(tmp_path):
    result = run(tmp_path, "raise ValueError('bug du script')")
    assert result.status == "error"
    assert "bug du script" in result.stderr
