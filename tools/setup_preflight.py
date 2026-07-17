from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ADB_EXE, ADB_SERIAL


@dataclass(frozen=True)
class AdbPreflightResult:
    ready: bool
    message: str


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _run_command(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _device_states(output: str) -> dict[str, str]:
    states: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] != "List":
            states[fields[0]] = fields[1]
    return states


def _detected_devices_text(states: dict[str, str]) -> str:
    if not states:
        return "未检测到任何 ADB 设备"
    return "、".join(f"{serial} ({state})" for serial, state in states.items())


def prepare_adb(
    adb_path: str | Path,
    serial: str,
    *,
    runner: CommandRunner = _run_command,
    sleep: Callable[[float], None] = time.sleep,
    reconnect_attempts: int = 15,
) -> AdbPreflightResult:
    adb = str(adb_path)
    runner([adb, "connect", serial], timeout=20)
    devices = runner([adb, "devices"], timeout=20)
    states = _device_states(devices.stdout)
    state = states.get(serial)

    if state != "device":
        detected = _detected_devices_text(states)
        if state == "unauthorized":
            reason = "目标设备状态为 unauthorized，请在模拟器中允许 ADB 调试"
        elif state == "offline":
            reason = "目标设备状态为 offline，请重启模拟器的 ADB 或模拟器"
        else:
            reason = f"未连接到目标设备 {serial}"
        return AdbPreflightResult(
            False,
            f"{reason}。检测结果：{detected}。如果模拟器使用其他地址，请修改 config.py 中的 ADB_SERIAL。",
        )

    root = runner([adb, "-s", serial, "root"], timeout=20)
    connected = False
    for attempt in range(max(1, reconnect_attempts)):
        runner([adb, "connect", serial], timeout=20)
        state_result = runner([adb, "-s", serial, "get-state"], timeout=20)
        if state_result.returncode == 0 and state_result.stdout.strip() == "device":
            connected = True
            break
        if attempt + 1 < reconnect_attempts:
            sleep(1.0)

    if not connected:
        detail = root.stderr.strip() or root.stdout.strip() or "adb root 后设备没有重新连接"
        return AdbPreflightResult(False, f"目标设备在 adb root 后未恢复连接：{detail}")

    uid = runner([adb, "-s", serial, "shell", "id", "-u"], timeout=20)
    uid_value = uid.stdout.strip()
    if uid.returncode != 0 or uid_value != "0":
        detail = uid_value or uid.stderr.strip() or root.stderr.strip() or "未知"
        return AdbPreflightResult(
            False,
            f"目标设备已连接，但 Root 验证失败，id -u 输出为 {detail}。请在模拟器设置中开启 Root。",
        )

    return AdbPreflightResult(True, f"模拟器 {serial} 已连接，Root 验证通过（id=0）。")


def main() -> int:
    result = prepare_adb(ADB_EXE, ADB_SERIAL)
    print(result.message)
    return 0 if result.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
