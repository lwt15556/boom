from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import BinaryIO

from config import BASE_DIR


RUNTIME_DIR = BASE_DIR / "_debug" / "runtime"
MAIN_PID_FILE = RUNTIME_DIR / "main.pid"
_WINDOWS_LOCK_OFFSET = 4096
_HELD_LOCKS: dict[Path, BinaryIO] = {}


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
    except (OSError, UnicodeError):
        return None

    try:
        return int(text)
    except ValueError:
        return None


def write_pid(pid_file: Path = MAIN_PID_FILE, pid: int | None = None) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid if pid is not None else current_pid()), encoding="utf-8")


def remove_pid(pid_file: Path = MAIN_PID_FILE, pid: int | None = None) -> None:
    key = _lock_key(pid_file)
    held_handle = _HELD_LOCKS.get(key)
    if held_handle is not None:
        _clear_pid_if_owned(held_handle, pid)
        return

    if not pid_file.exists():
        return

    handle = _try_acquire_file_lock(pid_file)
    if handle is None:
        return
    try:
        _clear_pid_if_owned(handle, pid)
    finally:
        _release_file_lock(handle)
        handle.close()


def acquire_main_lock(pid_file: Path = MAIN_PID_FILE) -> int:
    pid = current_pid()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    key = _lock_key(pid_file)
    if key in _HELD_LOCKS:
        existing = read_pid(pid_file)
        raise AlreadyRunningError(_already_running_message(existing))

    handle = _try_acquire_file_lock(pid_file)
    if handle is None:
        raise AlreadyRunningError(_already_running_message(read_pid(pid_file)))

    try:
        _write_pid_to_handle(handle, pid)
    except Exception:
        _release_file_lock(handle)
        handle.close()
        raise
    _HELD_LOCKS[key] = handle
    return pid


def release_main_lock(pid_file: Path = MAIN_PID_FILE, pid: int | None = None) -> None:
    handle = _HELD_LOCKS.pop(_lock_key(pid_file), None)
    if handle is None:
        remove_pid(pid_file=pid_file, pid=pid)
        return

    try:
        _clear_pid_if_owned(handle, pid)
    finally:
        _release_file_lock(handle)
        handle.close()


def get_main_process(pid_file: Path = MAIN_PID_FILE) -> tuple[int | None, bool] | None:
    if not pid_file.exists():
        return None

    handle = _try_acquire_file_lock(pid_file)
    if handle is None:
        return read_pid(pid_file), True

    try:
        pid = _read_pid_from_handle(handle)
    finally:
        _release_file_lock(handle)
        handle.close()
    if pid is None:
        return None
    return pid, False


def _lock_key(pid_file: Path) -> Path:
    return pid_file.resolve(strict=False)


def _already_running_message(pid: int | None) -> str:
    if pid is None:
        return "main.py 已在运行（PID 暂不可读）"
    return f"main.py 已在运行，PID={pid}"


def _try_acquire_file_lock(pid_file: Path) -> BinaryIO | None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(pid_file), os.O_RDWR | os.O_CREAT)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        _acquire_file_lock(handle)
    except OSError:
        handle.close()
        return None
    return handle


def _acquire_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_pid_from_handle(handle: BinaryIO) -> int | None:
    handle.seek(0)
    try:
        text = handle.read(64).decode("utf-8").strip()
        return int(text)
    except (UnicodeError, ValueError):
        return None


def _write_pid_to_handle(handle: BinaryIO, pid: int) -> None:
    handle.seek(0)
    handle.truncate()
    handle.write(str(pid).encode("ascii"))
    handle.flush()
    os.fsync(handle.fileno())


def _clear_pid_if_owned(handle: BinaryIO, pid: int | None) -> None:
    existing = _read_pid_from_handle(handle)
    if pid is not None and existing not in {None, pid}:
        return
    handle.seek(0)
    handle.truncate()
    handle.flush()


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
    "release_main_lock",
    "remove_pid",
    "write_pid",
]
