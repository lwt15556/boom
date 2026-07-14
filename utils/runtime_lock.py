from __future__ import annotations

import ctypes
import os
from pathlib import Path

from config import BASE_DIR


RUNTIME_DIR = BASE_DIR / "_debug" / "runtime"
MAIN_PID_FILE = RUNTIME_DIR / "main.pid"


class AlreadyRunningError(RuntimeError):
    """Raised when another main.py process is already registered as running."""


def current_pid() -> int:
    return os.getpid()


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False

    if os.name == "nt":
        return _is_windows_pid_running(pid)

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_pid(pid_file: Path = MAIN_PID_FILE) -> int | None:
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        return int(text)
    except ValueError:
        return None


def write_pid(pid_file: Path = MAIN_PID_FILE, pid: int | None = None) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid if pid is not None else current_pid()), encoding="utf-8")


def remove_pid(pid_file: Path = MAIN_PID_FILE, pid: int | None = None) -> None:
    existing = read_pid(pid_file)
    if pid is not None and existing not in {None, pid}:
        return
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def acquire_main_lock(pid_file: Path = MAIN_PID_FILE) -> int:
    pid = current_pid()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(str(pid_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = read_pid(pid_file)
        if existing is not None and is_pid_running(existing):
            raise AlreadyRunningError(f"main.py 已在运行，PID={existing}")

        remove_pid(pid_file)
        fd = os.open(str(pid_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    with os.fdopen(fd, "w", encoding="utf-8") as file:
        file.write(str(pid))
    return pid


def get_main_process() -> tuple[int, bool] | None:
    pid = read_pid(MAIN_PID_FILE)
    if pid is None:
        return None
    return pid, is_pid_running(pid)


def _is_windows_pid_running(pid: int) -> bool:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    SYNCHRONIZE = 0x00100000
    STILL_ACTIVE = 259

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, int(pid))
    if not handle:
        return False

    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


__all__ = [
    "AlreadyRunningError",
    "MAIN_PID_FILE",
    "RUNTIME_DIR",
    "acquire_main_lock",
    "get_main_process",
    "is_pid_running",
    "read_pid",
    "remove_pid",
    "write_pid",
]
