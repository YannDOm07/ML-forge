"""Hardware detection and process checks, without third-party dependencies."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_gb: float


@dataclass(frozen=True)
class Hardware:
    cpu_threads: int
    ram_gb: float | None
    free_disk_gb: float
    gpus: list[Gpu] = field(default_factory=list)


def detect_hardware(disk_path: Path) -> Hardware:
    return Hardware(
        cpu_threads=os.cpu_count() or 1,
        ram_gb=_total_ram_gb(),
        free_disk_gb=round(shutil.disk_usage(_existing_parent(disk_path)).free / 1024**3, 1),
        gpus=_nvidia_gpus(),
    )


def _existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _total_ram_gb() -> float | None:
    if os.name == "nt":
        return _windows_total_ram_gb()
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _windows_total_ram_gb() -> float | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.ullTotalPhys / 1024**3, 1)


def _nvidia_gpus() -> list[Gpu]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in output.strip().splitlines():
        name, _, memory_mib = line.rpartition(",")
        try:
            gpus.append(Gpu(name.strip(), round(float(memory_mib) / 1024, 1)))
        except ValueError:
            continue
    return gpus


def hardware_warnings(hardware: Hardware, memory_limit_gb: float, min_free_disk_gb: float) -> list[str]:
    warnings = []
    if hardware.ram_gb is not None and memory_limit_gb > 0.75 * hardware.ram_gb:
        warnings.append(
            f"sandbox.memory_limit_gb = {memory_limit_gb} dépasse 75 % de la RAM ({hardware.ram_gb} Go) : "
            "risque de ralentir tout le PC."
        )
    if hardware.free_disk_gb < min_free_disk_gb:
        warnings.append(
            f"Espace libre ({hardware.free_disk_gb} Go) inférieur au minimum configuré ({min_free_disk_gb} Go) : "
            "aucun téléchargement ne sera lancé."
        )
    return warnings


def process_alive(pid: int, created: int | None = None) -> bool:
    """True if `pid` runs and, when `created` is given, is the same process (not a reused PID)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        alive = _windows_process_alive(pid)
    else:
        alive = _posix_process_alive(pid)
    if not alive or created is None:
        return alive
    current = process_creation_time(pid)
    return current is None or current == created


def _posix_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _open_process(pid: int):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    return kernel32, kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)


def _windows_process_alive(pid: int) -> bool:
    # os.kill(pid, 0) would terminate the process on Windows: query it instead.
    kernel32, handle = _open_process(pid)
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = ctypes.c_ulong()
        alive = kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return bool(alive) and code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def process_creation_time(pid: int) -> int | None:
    """Opaque creation timestamp of a process, used to tell a reused PID apart."""
    if os.name != "nt":
        try:
            return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
        except (OSError, IndexError, ValueError):
            return None
    kernel32, handle = _open_process(pid)
    if not handle:
        return None
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        ok = kernel32.GetProcessTimes(ctypes.c_void_p(handle), *(ctypes.byref(t) for t in times))
        return times[0].value if ok else None
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


ML_MODULES = ("numpy.random", "scipy.special", "scipy.optimize", "pandas", "sklearn.ensemble", "sklearn.metrics")
OPTIONAL_ML_MODULES = ("lightgbm", "xgboost", "catboost", "optuna")


def check_ml_stack(python: str) -> dict[str, str]:
    """Import each ML module in a fresh interpreter: 'ok', 'absent' or the error (e.g. blocked DLL)."""
    report = {}
    for module in ML_MODULES + OPTIONAL_ML_MODULES:
        command = [python, "-c", f"import {module}"]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            report[module] = f"erreur : {exc}"
            continue
        if done.returncode == 0:
            report[module] = "ok"
        elif "ModuleNotFoundError" in done.stderr:
            report[module] = "absent"
        else:
            report[module] = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "échec"
    return report


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def keep_awake(enabled: bool = True) -> Iterator[None]:
    """Prevent Windows from sleeping while a campaign runs (no administrator rights needed)."""
    active = enabled and os.name == "nt"
    if active:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        if active:
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


def prudent_settings(hardware: Hardware) -> dict[str, dict[str, float]]:
    """Defaults adapted to the machine: the sandbox may use at most 60 % of the RAM (10 GB cap)."""
    if hardware.ram_gb is None:
        return {}
    return {"sandbox": {"memory_limit_gb": round(max(0.5, min(10.0, hardware.ram_gb * 0.6)), 1)}}
