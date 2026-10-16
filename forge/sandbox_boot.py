"""Entry point of the sandboxed child process.

It waits for the parent to finish confinement (Job Object), reads its policy on stdin,
installs an audit hook that blocks network access, process creation and file access
outside the allowed directories, then runs the experiment script.

Threat model: protects against mistakes of generated code, not against a determined
attacker (native code and ctypes can bypass audit hooks). See docs/ARCHITECTURE.md.
"""

import json
import os
import runpy
import sys
import sysconfig

MARKER = "[forge-sandbox]"

BLOCKED_EVENTS = {
    "socket.connect": "réseau",
    "socket.bind": "réseau",
    "socket.getaddrinfo": "réseau",
    "socket.gethostbyname": "réseau",
    "socket.gethostbyaddr": "réseau",
    "socket.sendto": "réseau",
    "subprocess.Popen": "création de processus",
    "os.system": "création de processus",
    "os.exec": "création de processus",
    "os.spawn": "création de processus",
    "os.posix_spawn": "création de processus",
    "os.startfile": "création de processus",
    "os.fork": "création de processus",
    "webbrowser.open": "navigateur",
}
PATH_WRITE_EVENTS = {
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.rmdir": (0,),
    "os.mkdir": (0,),
    "os.chmod": (0,),
    "os.truncate": (0,),
    "os.symlink": (0, 1),
    "os.link": (0, 1),
    "shutil.rmtree": (0,),
    "shutil.copyfile": (1,),
}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
DEVICE_NAMES = {"nul", "/dev/null", os.devnull.lower()}


def _normalise(path) -> str | None:
    if isinstance(path, int) or path is None:
        return None
    return os.path.normcase(os.path.realpath(os.fsdecode(path)))


def _inside(path: str, roots: list[str]) -> bool:
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _is_write(mode, flags) -> bool:
    if isinstance(mode, str):
        return any(char in mode for char in "wax+")
    return isinstance(flags, int) and bool(flags & WRITE_FLAGS)


def _python_roots() -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    roots |= {path for path in sysconfig.get_paths().values() if path}
    return [_normalise(root) for root in roots]


def _system_info_roots() -> list[str]:
    """Linux files that libraries read to size thread pools (joblib, OpenBLAS): CPU, memory, cgroups,
    and this process's own /proc entry. The rest of /proc stays closed: another process's
    environment may hold a token."""
    if os.name == "nt":
        return []
    roots = ["/sys/fs/cgroup", "/sys/devices/system/cpu", "/proc/cpuinfo", "/proc/meminfo", "/proc/stat",
             f"/proc/{os.getpid()}"]
    return [_normalise(root) for root in roots]


def install_guards(policy: dict) -> None:
    write_roots = [_normalise(path) for path in policy["write_roots"]]
    read_roots = (write_roots + [_normalise(path) for path in policy["read_roots"]] + _python_roots()
                  + _system_info_roots())

    def deny(kind: str, detail: object) -> None:
        raise PermissionError(f"{MARKER} accès bloqué : {kind} ({detail})")

    def hook(event: str, args: tuple) -> None:
        if event in BLOCKED_EVENTS:
            deny(BLOCKED_EVENTS[event], event)
        elif event == "open":
            path = _normalise(args[0])
            if path is None or path in DEVICE_NAMES or os.path.basename(path) in DEVICE_NAMES:
                return
            writing = _is_write(args[1], args[2] if len(args) > 2 else None)
            if not _inside(path, write_roots if writing else read_roots):
                deny("écriture hors du dossier d'expérience" if writing else "lecture non autorisée", path)
        elif event in PATH_WRITE_EVENTS:
            for index in PATH_WRITE_EVENTS[event]:
                path = _normalise(args[index]) if index < len(args) else None
                if path is not None and not _inside(path, write_roots):
                    deny("modification hors du dossier d'expérience", path)

    sys.addaudithook(hook)


def main() -> None:
    policy = json.loads(sys.stdin.readline())  # sent only once the parent has confined us
    sys.dont_write_bytecode = True
    sys.path[:0] = policy["python_path"]
    os.chdir(policy["workdir"])
    install_guards(policy)
    sys.argv = [policy["script"]]
    runpy.run_path(policy["script"], run_name="__main__")


if __name__ == "__main__":
    main()
