"""Run a script in a confined child process: time and memory limits, guarded I/O, no network."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from forge.config import APP_DIR
from forge.sandbox_boot import MARKER

BOOT = Path(__file__).with_name("sandbox_boot.py")
OUTPUT_TAIL = 4_000
_KEPT_ENV = ("SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")


@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float
    memory_mb: int


@dataclass(frozen=True)
class SandboxResult:
    status: str  # ok | error | timeout | memory_exceeded | blocked
    returncode: int | None
    stdout: str
    stderr: str
    duration_s: float
    peak_memory_mb: float | None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def run_in_sandbox(
    script: Path, workdir: Path, read_paths: Sequence[Path], limits: SandboxLimits, python: str = ""
) -> SandboxResult:
    """Execute `script` with `workdir` as the only writable directory."""
    tmp = workdir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    policy = {
        "workdir": str(workdir),
        "script": str(script),
        "write_roots": [str(workdir)],
        "read_roots": [str(path) for path in read_paths] + [str(APP_DIR / "forge")],
        "python_path": [str(APP_DIR)],
    }
    executable, venv_env = _interpreter(python or sys.executable)
    started = time.monotonic()
    process = subprocess.Popen(
        [executable, "-I", "-X", "utf8", str(BOOT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workdir,
        env={**_child_env(workdir, tmp), **venv_env},
        text=True,
        encoding="utf-8",
        errors="replace",
        **_posix_limits(limits),
    )
    job = _WindowsJob(limits.memory_mb) if os.name == "nt" else None
    try:
        if job:
            job.assign(process)
        return _wait(process, policy, limits, job, started)
    finally:
        if job:
            job.close()


def _interpreter(python: str) -> tuple[str, dict[str, str]]:
    """On Windows a venv's python.exe is a launcher that spawns the base interpreter, which
    the single-process Job Object forbids. Start the base interpreter directly instead and
    point it at the venv, as the launcher itself does."""
    path = Path(python)
    config = path.parent.parent / "pyvenv.cfg"
    if os.name != "nt" or not config.is_file():
        return python, {}
    values = dict(
        line.split("=", 1) for line in config.read_text(encoding="utf-8").splitlines() if "=" in line
    )
    values = {key.strip(): value.strip() for key, value in values.items()}
    base = values.get("executable") or str(Path(values["home"]) / "python.exe")
    return base, {"__PYVENV_LAUNCHER__": str(path)}


def _child_env(workdir: Path, tmp: Path) -> dict[str, str]:
    """Minimal environment: no tokens, no user profile, temporary files inside the workdir."""
    env = {key: os.environ[key] for key in _KEPT_ENV if key in os.environ}
    for key in ("TEMP", "TMP", "TMPDIR"):
        env[key] = str(tmp)
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        env[key] = str(workdir)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _posix_limits(limits: SandboxLimits) -> dict:
    if os.name == "nt":
        return {}

    def apply() -> None:
        import resource

        size = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (size, size))

    return {"preexec_fn": apply}


def _wait(process, policy: dict, limits: SandboxLimits, job, started: float) -> SandboxResult:
    timed_out = False
    try:
        stdout, stderr = process.communicate(json.dumps(policy) + "\n", timeout=limits.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        if job:
            job.terminate()
        process.kill()
        stdout, stderr = process.communicate()
    peak = job.peak_memory_mb() if job else None
    status = _status(process.returncode, stderr, timed_out)
    return SandboxResult(
        status=status,
        returncode=process.returncode,
        stdout=stdout[-OUTPUT_TAIL:],
        stderr=stderr[-OUTPUT_TAIL:],
        duration_s=round(time.monotonic() - started, 3),
        peak_memory_mb=peak,
    )


def _status(returncode: int | None, stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "ok"
    if MARKER in stderr:
        return "blocked"
    if "MemoryError" in stderr or "ArrayMemoryError" in stderr or "bad_alloc" in stderr:
        return "memory_exceeded"
    return "error"


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsJob:
    """Job Object: memory cap, a single process (no children), killed when closed."""

    _EXTENDED_LIMIT_INFORMATION = 9
    _LIMIT_ACTIVE_PROCESS = 0x0008
    _LIMIT_JOB_MEMORY = 0x0200
    _LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self, memory_mb: int) -> None:
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        self._handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW a échoué")
        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = (
            self._LIMIT_ACTIVE_PROCESS | self._LIMIT_JOB_MEMORY | self._LIMIT_KILL_ON_JOB_CLOSE
        )
        info.BasicLimitInformation.ActiveProcessLimit = 1
        info.JobMemoryLimit = memory_mb * 1024 * 1024
        if not self._kernel32.SetInformationJobObject(
            ctypes.c_void_p(self._handle), self._EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject a échoué")

    def assign(self, process: subprocess.Popen) -> None:
        if not self._kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(self._handle), ctypes.c_void_p(int(process._handle))
        ):
            process.kill()
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject a échoué")

    def peak_memory_mb(self) -> float | None:
        info = _ExtendedLimit()
        ok = self._kernel32.QueryInformationJobObject(
            ctypes.c_void_p(self._handle), self._EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info), None,
        )
        return round(info.PeakJobMemoryUsed / 1024**2, 1) if ok else None

    def terminate(self) -> None:
        self._kernel32.TerminateJobObject(ctypes.c_void_p(self._handle), 1)

    def close(self) -> None:
        self._kernel32.CloseHandle(ctypes.c_void_p(self._handle))
