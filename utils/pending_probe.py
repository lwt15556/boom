from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BASE_DIR


PENDING_PROBE_FILE = BASE_DIR / "_debug" / "runtime" / "pending_probe.json"


def read_pending_probe(path: Path = PENDING_PROBE_FILE) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        return _unknown_pending_probe(f"read failed: {exc}")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _unknown_pending_probe(f"invalid json: {exc}")
    if not isinstance(payload, dict):
        return _unknown_pending_probe(
            f"expected object, got {type(payload).__name__}"
        )
    return payload


def has_pending_probe(path: Path = PENDING_PROBE_FILE) -> bool:
    return read_pending_probe(path) is not None


def write_pending_probe(
    *,
    mode: str,
    level: int,
    cell: tuple[int, int],
    index: int,
    phase: str,
    path: Path = PENDING_PROBE_FILE,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "mode": str(mode),
        "level": int(level),
        "cell": [int(cell[0]), int(cell[1])],
        "index": int(index),
        "phase": str(phase),
        "created_at": now,
        "updated_at": now,
    }
    _write_payload(path, payload)


def update_pending_probe(
    *,
    phase: str | None = None,
    path: Path = PENDING_PROBE_FILE,
    **updates: Any,
) -> bool:
    payload = read_pending_probe(path)
    if payload is None:
        return False
    if phase is not None:
        payload["phase"] = str(phase)
    payload.update(updates)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_payload(path, payload)
    return True


def clear_pending_probe(path: Path = PENDING_PROBE_FILE) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _unknown_pending_probe(reason: str) -> dict[str, Any]:
    return {
        "mode": "unknown",
        "phase": "INTERRUPTED",
        "state_unknown": True,
        "error": str(reason),
    }


__all__ = [
    "PENDING_PROBE_FILE",
    "clear_pending_probe",
    "has_pending_probe",
    "read_pending_probe",
    "update_pending_probe",
    "write_pending_probe",
]
