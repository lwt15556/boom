from __future__ import annotations

# ruff: noqa: E402 - this script supports direct execution outside the package.

import json
import os
import re
import subprocess
import sys
from collections import deque
from datetime import datetime
from math import isqrt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import (
    QObject,
    QLockFile,
    QProcess,
    QProcessEnvironment,
    QPointF,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen, QPolygonF, QTextCursor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import ADB_EXE, ADB_SERIAL, GAME_PACKAGE_NAME, LOG_FILE, RED_SCOUT_MAX_COUNT
from utils.adb_control import AdbController
from utils.pending_probe import clear_pending_probe, has_pending_probe
from utils.runtime_lock import MAIN_PID_FILE, get_main_process, is_pid_running, remove_pid


MAIN_SCRIPT = PROJECT_ROOT / "main.py"
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
RUN_STDOUT = PROJECT_ROOT / "run_stdout.log"
RUN_STDERR = PROJECT_ROOT / "run_stderr.log"
STATUS_FILE = PROJECT_ROOT / "_debug" / "runtime" / "status.json"
PANEL_LOCK_FILE = PROJECT_ROOT / "_debug" / "runtime" / "control_panel.lock"

NETWORK_BLOCK_SETTLE_SECONDS = 0.2
APP_STOP_TIMEOUT_SECONDS = 5.0
APP_STOP_POLL_SECONDS = 0.1
POST_FORCE_STOP_GUARD_SECONDS = 0.5

MAX_LOG_LINES = 1600
MAX_LOG_BYTES = 1_200_000
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NOISY_LOG_PATTERNS = (
    "截图已保存",
    "正在等待模板",
    "等待模板",
    "screenshot saved",
    "waiting for template",
    "鎴浘宸蹭繚瀛",
    "绛夊緟妯℃澘",
)
MOJIBAKE_MARKERS = (
    "锛",
    "銆",
    "杩",
    "绗",
    "鍏",
    "娼",
    "鏍",
    "缃",
    "绋",
    "鎺",
    "鍚",
    "宸",
)

PHASE_NAMES = {
    "red_scout_preflight": "红色侦察准备",
    "red_scout_capture": "红色侦察取证",
    "red_scout_discard": "红色侦察丢弃请求",
    "red_scout_verify_ammo": "红色侦察核验弹药",
    "stale_probe_recovery": "中断请求安全清理",
    "stale_probe_recovered": "中断请求已安全清理",
    "blue_attack": "蓝色攻击",
    "starting": "正在启动",
    "level_loading": "关卡初始化",
    "level_complete": "关卡已完成",
    "advance_level": "进入下一关",
    "enter_activity": "进入活动",
    "strategy_scan": "智能寻路",
    "supplemental_recheck": "命中线索优先补炸",
    "grid_scan": "逐格扫描",
    "fallback_scan": "保守扫描",
    "stopped": "已停止",
    "complete": "已完成",
}
RESULT_NAMES = {
    "hit": "命中",
    "miss": "未命中",
    "level_complete": "关卡已完成",
    "unknown": "待确认",
    "hit_and_level_complete": "命中并完成",
    "scout_valid": "侦察结果已累计",
    "scout_invalid": "侦察结果无效",
    "supplemental_recheck_pending": "正在补炸侦察未命中格",
}
BOARD_STATE_NAMES = {
    "scout_miss": "侦察未命中",
    "scout_hit": "侦察命中",
    "unknown": "未探测",
    "miss": "未命中",
    "hit": "已命中",
    "ship": "完整潜艇",
    "blocked": "安全区",
}
BOARD_STATE_COLORS = {
    "scout_miss": QColor("#aab7be"),
    "scout_hit": QColor("#d9822b"),
    "unknown": QColor("#d9e7ed"),
    "miss": QColor("#718793"),
    "hit": QColor("#d34f4f"),
    "ship": QColor("#17845c"),
    "blocked": QColor("#eef1f3"),
}

PROBE_MODE_NAMES = {"blue_only": "仅蓝色炮弹", "red_scout": "红色侦察 + 蓝色攻击"}


def format_probe_mode(value: object) -> str:
    return PROBE_MODE_NAMES.get(str(value), PROBE_MODE_NAMES["blue_only"])


def format_red_scout_progress(
    *,
    current: object,
    total: object,
    valid: object,
    complete_six: object,
) -> str:
    current_value = safe_int(current)
    total_value = safe_int(total)
    if current_value is None or total_value is None:
        return "--"
    valid_value = max(0, safe_int(valid) or 0)
    complete_value = max(0, safe_int(complete_six) or 0)
    return (
        f"{current_value} / {total_value} · "
        f"有效 {valid_value} · 完整六格 {complete_value}"
    )


def build_main_environment(mode: object, red_count: object) -> dict[str, str]:
    normalized = str(mode) if str(mode) in PROBE_MODE_NAMES else "blue_only"
    try:
        count = int(red_count)
    except (TypeError, ValueError):
        count = 2
    return {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "BBMA_PROBE_MODE": normalized, "BBMA_RED_SCOUT_COUNT": str(max(1, min(RED_SCOUT_MAX_COUNT, count)))}


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_command(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def adb_executable() -> str:
    bundled = Path(ADB_EXE)
    return str(bundled) if bundled.exists() else "adb"


def stop_pid(pid: int) -> None:
    if os.name == "nt":
        result = run_command(["taskkill", "/PID", str(pid), "/F"], timeout=20)
        if result.returncode != 0 and is_pid_running(pid):
            detail = result.stderr.strip() or result.stdout.strip() or f"停止 PID {pid} 失败"
            raise RuntimeError(detail)
        return
    os.kill(pid, 15)


def _restore_network_rules(adb: AdbController) -> None:
    # Keep at least one blocking rule active until the final restore step.
    adb.disable_weak_network(GAME_PACKAGE_NAME)
    adb.disable_reject_network(GAME_PACKAGE_NAME)


def _block_network_for_safe_stop(adb: AdbController) -> None:
    adb.enable_weak_network(GAME_PACKAGE_NAME)
    adb.enable_reject_network(GAME_PACKAGE_NAME)
    adb.delay(NETWORK_BLOCK_SETTLE_SECONDS)


def _force_stop_game_while_blocked(adb: AdbController) -> None:
    adb.close_app(GAME_PACKAGE_NAME)
    if not adb.wait_until_app_stopped(
        GAME_PACKAGE_NAME,
        timeout=APP_STOP_TIMEOUT_SECONDS,
        poll_interval=APP_STOP_POLL_SECONDS,
    ):
        raise RuntimeError("游戏进程未完全退出，已保留 DROP/REJECT 断网，禁止恢复网络")
    adb.delay(POST_FORCE_STOP_GUARD_SECONDS)


def _runtime_status_is_offline() -> bool:
    status = read_runtime_status()
    network = str(status.get("network", ""))
    return has_pending_probe() or any(
        marker in network for marker in ("断网", "DROP", "REJECT", "fail_closed")
    )


def _clear_runtime_status() -> None:
    try:
        STATUS_FILE.unlink()
    except FileNotFoundError:
        pass


def restore_network() -> str:
    pid_state = get_main_process()
    if pid_state is not None and pid_state[1]:
        raise RuntimeError(
            f"主程序运行中（PID={pid_state[0]}），禁止直接恢复网络；"
            "请使用“停止程序”安全丢弃待处理请求"
        )

    adb = AdbController(ADB_SERIAL)
    adb.ensure_root_shell()
    stale_offline_request = _runtime_status_is_offline()
    if stale_offline_request:
        _block_network_for_safe_stop(adb)
        _force_stop_game_while_blocked(adb)
        clear_pending_probe()
    _restore_network_rules(adb)
    if stale_offline_request:
        _clear_runtime_status()
        return "已丢弃中断请求并恢复网络：游戏已安全关闭"
    return "网络已恢复：REJECT 和 DROP 规则均已关闭"


def stop_program() -> str:
    pid_state = get_main_process()
    if pid_state is None:
        restore_network()
        return "没有发现主程序 PID，网络已恢复"

    pid, running = pid_state
    if running:
        if pid is None:
            raise RuntimeError("主程序锁仍被占用，但 PID 暂不可读；为避免误停其他进程，已中止操作")
        adb = AdbController(ADB_SERIAL)
        adb.ensure_root_shell()
        _block_network_for_safe_stop(adb)
        stop_pid(pid)
        _force_stop_game_while_blocked(adb)
        remove_pid(pid=pid)
        clear_pending_probe()
        _restore_network_rules(adb)
        _clear_runtime_status()
        return f"已停止主程序 PID={pid}，已丢弃待处理请求并恢复网络"

    remove_pid(pid=pid)
    restore_network()
    return f"已清理过期 PID 文件（PID={pid}），网络已恢复"


def check_adb() -> str:
    adb_path = adb_executable()
    devices = run_command([adb_path, "devices"], timeout=20)
    root = run_command([adb_path, "-s", ADB_SERIAL, "shell", "id", "-u"], timeout=20)
    root_value = root.stdout.strip()
    root_text = "root (id=0)" if root_value == "0" else root_value or root.stderr.strip() or "未知"
    connected = f"{ADB_SERIAL}\tdevice" in devices.stdout
    connection_text = "已连接" if connected else "未检测到目标设备"

    return "\n".join(
        [
            f"模拟器：{connection_text}",
            f"目标设备：{ADB_SERIAL}",
            f"Root 状态：{root_text}",
            f"游戏包名：{GAME_PACKAGE_NAME}",
            "",
            "adb devices 输出：",
            devices.stdout.strip() or devices.stderr.strip() or "（无输出）",
        ]
    )


def should_show_log_line(line: str, show_detail: bool) -> bool:
    if show_detail:
        return True
    if " DEBUG " in f" {line} ":
        return False
    return not any(pattern in line for pattern in NOISY_LOG_PATTERNS)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS) + text.count("\ufffd") * 4


def repair_mojibake(text: str) -> str:
    if mojibake_score(text) < 2:
        return text

    best = text
    best_score = mojibake_score(text)
    for encoding in ("gb18030", "gbk"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        candidate_score = mojibake_score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def decode_log_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        break
    else:
        text = data.decode("utf-8", errors="replace")

    clean = strip_ansi(text)
    return "\n".join(repair_mojibake(line) for line in clean.splitlines())


def read_log_tail(path: Path) -> tuple[bytes, int]:
    size = path.stat().st_size
    with path.open("rb") as file:
        start = max(0, size - MAX_LOG_BYTES)
        if start:
            file.seek(start)
            file.readline()
        data = file.read()
    return data, size


def read_runtime_status() -> dict[str, object]:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_phase(value: object) -> str:
    text = str(value or "--")
    return PHASE_NAMES.get(text, repair_mojibake(text))


def format_result(value: object) -> str:
    text = str(value or "")
    return RESULT_NAMES.get(text, repair_mojibake(text) if text else "--")


def format_reason(reason: object) -> str:
    text = str(reason or "")
    if text == "victory_banner_during_reentry":
        return "重新进入时检测到胜利"
    match = re.fullmatch(r"hit_votes_(\d+)", text)
    if match:
        return f"{match.group(1)} 帧确认"
    match = re.fullmatch(r"strong_single_score_([0-9.]+)", text)
    if match:
        return f"强特征 {match.group(1)}"
    match = re.fullmatch(r"near_hit_frames_(\d+)", text)
    if match:
        return f"{match.group(1)} 帧近似命中"
    match = re.fullmatch(r"hit_votes_(\d+)_near_(\d+)", text)
    if match:
        return f"命中帧 {match.group(1)}，近似帧 {match.group(2)}"
    return repair_mojibake(text)


def format_cell(value: object, total_cells: object) -> str:
    index = safe_int(value)
    total = safe_int(total_cells)
    if index is None:
        return str(value or "--")
    if total is None or total <= 0:
        return f"#{index}"
    size = isqrt(total)
    if size * size != total:
        return f"#{index}"
    row, col = divmod(index, size)
    return f"#{index}（第 {row + 1} 行，第 {col + 1} 列）"


class SonarBoardWidget(QWidget):
    """Responsive isometric board used by the runtime monitor."""

    def __init__(self):
        super().__init__()
        self._states: list[list[str]] = []
        self._current_index: int | None = None
        self._cell_polygons: list[tuple[int, int, QPolygonF]] = []
        self.setMouseTracking(True)
        self.setMinimumSize(360, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("潜艇探测棋盘")

    def sizeHint(self) -> QSize:
        return QSize(560, 230)

    def set_board(self, states: object, current_index: object = None) -> None:
        normalized: list[list[str]] = []
        if isinstance(states, list) and states:
            size = len(states)
            if all(isinstance(row, list) and len(row) == size for row in states):
                normalized = [
                    [
                        state if state in BOARD_STATE_NAMES else "unknown"
                        for state in row
                    ]
                    for row in states
                ]

        current = safe_int(current_index)
        if normalized and current is not None:
            current = current if 0 <= current < len(normalized) ** 2 else None
        else:
            current = None

        if normalized == self._states and current == self._current_index:
            return
        self._states = normalized
        self._current_index = current
        self.update()

    def state_counts(self) -> dict[str, int]:
        counts = {state: 0 for state in BOARD_STATE_NAMES}
        for row in self._states:
            for state in row:
                counts[state] += 1
        return counts

    @property
    def board_size(self) -> int:
        return len(self._states)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._cell_polygons = []

        if not self._states:
            painter.setPen(QColor("#74818b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待棋盘状态")
            return

        size = len(self._states)
        padding_x = 20.0
        padding_y = 12.0
        available_width = max(1.0, self.width() - padding_x * 2)
        available_height = max(1.0, self.height() - padding_y * 2)
        tile_width = min(available_width / size, available_height * 2.0 / size)
        tile_height = tile_width / 2.0
        board_height = tile_height * size
        origin_x = self.width() / 2.0
        origin_y = (self.height() - board_height) / 2.0

        for diagonal in range(size * 2 - 1):
            for row in range(size):
                col = diagonal - row
                if not 0 <= col < size:
                    continue

                center_x = origin_x + (col - row) * tile_width / 2.0
                center_y = origin_y + (row + col) * tile_height / 2.0 + tile_height / 2.0
                polygon = QPolygonF(
                    [
                        QPointF(center_x, center_y - tile_height / 2.0),
                        QPointF(center_x + tile_width / 2.0, center_y),
                        QPointF(center_x, center_y + tile_height / 2.0),
                        QPointF(center_x - tile_width / 2.0, center_y),
                    ]
                )
                state = self._states[row][col]
                painter.setBrush(QBrush(BOARD_STATE_COLORS[state]))
                painter.setPen(QPen(QColor("#ffffff"), 1.2))
                painter.drawPolygon(polygon)

                marker_size = max(2.0, tile_height * 0.16)
                if state == "miss":
                    painter.setPen(QPen(QColor("#eef4f6"), max(1.2, marker_size * 0.55)))
                    painter.drawLine(
                        QPointF(center_x - marker_size, center_y - marker_size / 2.0),
                        QPointF(center_x + marker_size, center_y + marker_size / 2.0),
                    )
                    painter.drawLine(
                        QPointF(center_x + marker_size, center_y - marker_size / 2.0),
                        QPointF(center_x - marker_size, center_y + marker_size / 2.0),
                    )
                elif state == "scout_miss":
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(QColor("#ffffff"), max(1.2, marker_size * 0.45)))
                    painter.drawEllipse(QPointF(center_x, center_y), marker_size, marker_size)
                elif state == "scout_hit":
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#ffffff"))
                    painter.drawPolygon(QPolygonF([QPointF(center_x, center_y - marker_size), QPointF(center_x + marker_size, center_y), QPointF(center_x, center_y + marker_size), QPointF(center_x - marker_size, center_y)]))
                elif state == "hit":
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#ffffff"))
                    painter.drawEllipse(QPointF(center_x, center_y), marker_size, marker_size)
                elif state == "ship":
                    painter.setPen(
                        QPen(
                            QColor("#ffffff"),
                            max(2.0, marker_size * 0.9),
                            Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap,
                        )
                    )
                    painter.drawLine(
                        QPointF(center_x - marker_size * 1.35, center_y),
                        QPointF(center_x + marker_size * 1.35, center_y),
                    )
                elif state == "blocked":
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#c7d0d5"))
                    painter.drawEllipse(
                        QPointF(center_x, center_y),
                        max(1.0, marker_size * 0.45),
                        max(1.0, marker_size * 0.45),
                    )

                index = row * size + col
                if index == self._current_index:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(QColor("#e0a21a"), max(2.0, tile_height * 0.11)))
                    painter.drawPolygon(polygon)

                self._cell_polygons.append((row, col, polygon))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        for row, col, polygon in reversed(self._cell_polygons):
            if not polygon.containsPoint(event.position(), Qt.FillRule.OddEvenFill):
                continue
            state = self._states[row][col]
            size = len(self._states)
            index = row * size + col
            current = " · 当前目标" if index == self._current_index else ""
            self.setToolTip(
                f"第 {row + 1} 行，第 {col + 1} 列 · {BOARD_STATE_NAMES[state]}{current}"
            )
            return
        self.setToolTip("")

    def leaveEvent(self, event) -> None:
        self.setToolTip("")
        super().leaveEvent(event)


class Worker(QObject):
    finished = pyqtSignal(bool, str)

    def __init__(self, action: str):
        super().__init__()
        self.action = action

    def run(self) -> None:
        try:
            if self.action == "stop":
                self.finished.emit(True, stop_program())
            elif self.action == "restore_network":
                self.finished.emit(True, restore_network())
            elif self.action == "check_adb":
                self.finished.emit(True, check_adb())
            else:
                raise ValueError(f"不支持的操作：{self.action}")
        except Exception as exc:
            self.finished.emit(False, repair_mojibake(str(exc)))


class ControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BBMA 运行控制台")
        self.resize(1240, 810)
        self.setMinimumSize(1020, 680)

        self.thread: QThread | None = None
        self.worker: Worker | None = None
        self.main_pid: int | None = None
        self.pending_pid: int | None = None
        self.last_running_pid: int | None = None
        self.was_running = False
        self.runtime_status: dict[str, object] = {}
        self.network_status = "未检测"
        self.last_recent_signature = ""
        self.last_runtime_render_signature: str | None = None
        self.last_process_render_signature: tuple[bool, int | None] | None = None

        self.log_lines: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.current_log_path: Path | None = None
        self.last_log_size = 0

        self._create_widgets()
        self._build_ui()
        self._apply_style()
        self._connect_signals()

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        self.append_operation("控制台已启动")
        self.reload_log()
        self.update_status(initial=True)

    def _create_widgets(self) -> None:
        self.status_badge = QLabel("已停止")
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setMinimumWidth(92)
        self.status_badge.setFixedHeight(30)

        self.last_update_label = QLabel("状态更新时间：--")
        self.last_update_label.setObjectName("mutedText")

        self.start_button = QPushButton("启动程序")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.stop_button = QPushButton("停止程序")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.restore_button = QPushButton("恢复网络")
        self.restore_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.check_button = QPushButton("检查模拟器")
        self.check_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
        for button in (self.start_button, self.stop_button, self.restore_button, self.check_button):
            button.setMinimumHeight(38)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.probe_mode_combo = QComboBox()
        for value, label in PROBE_MODE_NAMES.items():
            self.probe_mode_combo.addItem(label, value)
        self.red_scout_count = QSpinBox()
        self.red_scout_count.setRange(1, RED_SCOUT_MAX_COUNT)
        self.red_scout_count.setValue(2)
        self.red_scout_count.setSuffix(" 次/关")
        self.probe_mode_value = QLabel("--")
        self.red_scout_progress_value = QLabel("--")

        self.pid_value = QLabel("--")
        self.adb_value = QLabel(ADB_SERIAL)
        self.network_value = QLabel("未检测")
        self.phase_value = QLabel("--")
        self.level_value = QLabel("--")
        self.current_cell_value = QLabel("--")
        self.last_result_value = QLabel("--")
        self.completed_ships_value = QLabel("--")
        self.visual_mapping_value = QLabel("--")

        self.shot_progress = self._new_progress_bar("shotProgress")
        self.hit_progress = self._new_progress_bar("hitProgress")
        self.ship_progress = self._new_progress_bar("shipProgress")

        self.board_widget = SonarBoardWidget()
        self.board_level_label = QLabel("等待任务")
        self.board_level_label.setObjectName("boardTitle")
        self.board_summary_label = QLabel("未探测 --  ·  未命中 --\n已命中 --  ·  完整潜艇 --")
        self.board_summary_label.setObjectName("mutedText")
        self.board_summary_label.setWordWrap(True)

        self.recent_table = QTableWidget(0, 4)
        self.recent_table.setHorizontalHeaderLabels(["时间", "关卡 / 格子", "结果", "判断依据"])
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.recent_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.recent_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.recent_table.setAlternatingRowColors(True)
        self.recent_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self.recent_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.recent_table.setFixedHeight(150)

        self.operation_view = QPlainTextEdit()
        self.operation_view.setReadOnly(True)
        self.operation_view.setMaximumBlockCount(250)
        self.operation_view.setFixedHeight(105)
        self.operation_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

        self.log_source_label = QLabel("来源：暂无日志")
        self.log_source_label.setObjectName("mutedText")
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItem("关键日志", False)
        self.log_filter_combo.addItem("全部日志", True)
        self.log_search = QLineEdit()
        self.log_search.setPlaceholderText("搜索日志")
        self.log_search.setClearButtonEnabled(True)
        self.log_search.setMinimumWidth(210)
        self.auto_scroll_combo = QComboBox()
        self.auto_scroll_combo.addItem("自动滚动", True)
        self.auto_scroll_combo.addItem("保持位置", False)
        self.clear_log_button = QPushButton("清空视图")
        self.clear_log_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        self.open_log_button = QPushButton("打开日志")
        self.open_log_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setPlaceholderText("暂无日志输出")

    @staticmethod
    def _new_progress_bar(object_name: str) -> QProgressBar:
        bar = QProgressBar()
        bar.setObjectName(object_name)
        bar.setRange(0, 1)
        bar.setValue(0)
        bar.setFormat("--")
        bar.setFixedHeight(20)
        bar.setTextVisible(True)
        return bar

    @staticmethod
    def _legend_item(label: str, color: str, *, outlined: bool = False) -> QWidget:
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(7)
        swatch = QFrame()
        swatch.setFixedSize(14, 14)
        border = "2px solid #e0a21a" if outlined else "1px solid #b8c3ca"
        swatch.setStyleSheet(
            f"background:{color}; border:{border}; border-radius:2px;"
        )
        layout.addWidget(swatch)
        layout.addWidget(QLabel(label))
        layout.addStretch()
        return item

    def _build_ui(self) -> None:
        header = QFrame()
        header.setObjectName("headerBand")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 13, 18, 13)
        title_column = QVBoxLayout()
        title = QLabel("BBMA 运行控制台")
        title.setObjectName("appTitle")
        subtitle = QLabel("潜艇探测自动化 · 模拟器控制与实时监控")
        subtitle.setObjectName("mutedText")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column)
        header_layout.addStretch()
        update_column = QVBoxLayout()
        update_column.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        update_column.addWidget(self.status_badge, alignment=Qt.AlignmentFlag.AlignRight)
        update_column.addWidget(self.last_update_label, alignment=Qt.AlignmentFlag.AlignRight)
        header_layout.addLayout(update_column)

        action_box = QGroupBox("运行控制")
        action_layout = QGridLayout(action_box)
        action_layout.setHorizontalSpacing(10)
        action_layout.setVerticalSpacing(10)
        action_layout.addWidget(self.probe_mode_combo, 0, 0, 1, 2)
        action_layout.addWidget(self.red_scout_count, 1, 0, 1, 2)
        action_layout.addWidget(self.start_button, 2, 0)
        action_layout.addWidget(self.stop_button, 2, 1)
        action_layout.addWidget(self.restore_button, 3, 0)
        action_layout.addWidget(self.check_button, 3, 1)

        progress_box = QGroupBox("任务进度")
        progress_layout = QGridLayout(progress_box)
        progress_layout.setColumnStretch(1, 1)
        progress_layout.addWidget(QLabel("已探测格子"), 0, 0)
        progress_layout.addWidget(self.shot_progress, 0, 1)
        progress_layout.addWidget(QLabel("确认命中"), 1, 0)
        progress_layout.addWidget(self.hit_progress, 1, 1)
        progress_layout.addWidget(QLabel("完成潜艇"), 2, 0)
        progress_layout.addWidget(self.ship_progress, 2, 1)

        state_box = QGroupBox("当前状态")
        state_layout = QGridLayout(state_box)
        state_layout.setColumnStretch(1, 1)
        state_layout.setColumnStretch(3, 1)
        rows = (
            ("PID", self.pid_value),
            ("Probe mode", self.probe_mode_value),
            ("Red scout", self.red_scout_progress_value),
            ("ADB", self.adb_value),
            ("网络", self.network_value),
            ("阶段", self.phase_value),
            ("当前关卡", self.level_value),
            ("当前格子", self.current_cell_value),
            ("最近结果", self.last_result_value),
            ("已完成潜艇", self.completed_ships_value),
            ("视觉坐标", self.visual_mapping_value),
        )
        state_box.setMinimumHeight(178)
        row_count = (len(rows) + 1) // 2
        for row in range(row_count):
            for column_offset in range(2):
                source_index = row + column_offset * row_count
                if source_index >= len(rows):
                    continue
                caption, value = rows[source_index]
                caption_label = QLabel(caption)
                caption_label.setObjectName("fieldCaption")
                value.setObjectName("fieldValue")
                value.setWordWrap(True)
                base_column = column_offset * 2
                state_layout.addWidget(
                    caption_label,
                    row,
                    base_column,
                    alignment=Qt.AlignmentFlag.AlignTop,
                )
                state_layout.addWidget(value, row, base_column + 1)

        recent_panel = QWidget()
        recent_layout = QVBoxLayout(recent_panel)
        recent_layout.setContentsMargins(6, 8, 6, 6)
        recent_layout.addWidget(self.recent_table)
        operation_panel = QWidget()
        operation_layout = QVBoxLayout(operation_panel)
        operation_layout.setContentsMargins(6, 8, 6, 6)
        operation_layout.addWidget(self.operation_view)
        activity_tabs = QTabWidget()
        activity_tabs.addTab(recent_panel, "最近探测")
        activity_tabs.addTab(operation_panel, "操作记录")
        activity_tabs.setMinimumHeight(188)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        left_layout.addWidget(action_box)
        left_layout.addWidget(progress_box)
        left_layout.addWidget(state_box)
        left_layout.addWidget(activity_tabs)
        left_layout.addStretch()
        left_panel = QWidget()
        left_panel.setLayout(left_layout)
        left_panel.setMinimumWidth(365)
        left_panel.setMaximumWidth(470)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(480)

        board_box = QGroupBox("炸潜艇棋盘")
        board_box.setMinimumHeight(218)
        board_layout = QHBoxLayout(board_box)
        board_layout.setSpacing(14)
        board_layout.addWidget(self.board_widget, 1)
        board_info = QWidget()
        board_info.setMinimumWidth(220)
        board_info.setMaximumWidth(270)
        board_info_layout = QVBoxLayout(board_info)
        board_info_layout.setContentsMargins(0, 4, 0, 4)
        board_info_layout.setSpacing(4)
        board_info_layout.addWidget(self.board_level_label)
        board_info_layout.addWidget(self.board_summary_label)
        board_info_layout.addSpacing(4)
        legend_title = QLabel("图例")
        legend_title.setObjectName("fieldCaption")
        board_info_layout.addWidget(legend_title)
        legend_widget = QWidget()
        legend_layout = QGridLayout(legend_widget)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setHorizontalSpacing(8)
        legend_layout.setVerticalSpacing(1)
        legend_items = (
            ("未探测", "#d9e7ed", False),
            ("未命中", "#718793", False),
            ("已命中", "#d34f4f", False),
            ("完整潜艇", "#17845c", False),
            ("安全区", "#eef1f3", False),
            ("侦察未命中", "#aab7be", False),
            ("侦察命中", "#d9822b", False),
            ("当前目标", "#ffffff", True),
        )
        for index, (label, color, outlined) in enumerate(legend_items):
            legend_layout.addWidget(
                self._legend_item(label, color, outlined=outlined),
                index // 2,
                index % 2,
            )
        board_info_layout.addWidget(legend_widget)
        board_info_layout.addStretch()
        board_layout.addWidget(board_info)

        log_box = QGroupBox("实时日志")
        log_box.setMinimumHeight(250)
        log_layout = QVBoxLayout(log_box)
        log_toolbar = QHBoxLayout()
        log_toolbar.setSpacing(8)
        log_toolbar.addWidget(self.log_source_label)
        log_toolbar.addStretch()
        log_toolbar.addWidget(self.log_filter_combo)
        log_toolbar.addWidget(self.log_search)
        log_toolbar.addWidget(self.auto_scroll_combo)
        log_toolbar.addWidget(self.clear_log_button)
        log_toolbar.addWidget(self.open_log_button)
        log_layout.addLayout(log_toolbar)
        log_layout.addWidget(self.log_view)

        monitor_splitter = QSplitter(Qt.Orientation.Vertical)
        monitor_splitter.addWidget(board_box)
        monitor_splitter.addWidget(log_box)
        monitor_splitter.setChildrenCollapsible(False)
        monitor_splitter.setStretchFactor(0, 0)
        monitor_splitter.setStretchFactor(1, 1)
        monitor_splitter.setSizes([250, 460])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(monitor_splitter)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([410, 810])

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)
        root_layout.addWidget(header)
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage(f"设备：{ADB_SERIAL}    PID 文件：{MAIN_PID_FILE}")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f3f5f7;
                color: #202a33;
                font-family: "Microsoft YaHei UI", "Segoe UI";
                font-size: 10pt;
            }
            QFrame#headerBand {
                background: #ffffff;
                border: 1px solid #d9dee3;
                border-radius: 6px;
            }
            QLabel#appTitle {
                font-size: 17pt;
                font-weight: 700;
                color: #182129;
            }
            QLabel#mutedText, QLabel#fieldCaption {
                color: #697782;
            }
            QLabel#fieldCaption {
                min-width: 72px;
            }
            QLabel#fieldValue {
                color: #202a33;
                font-weight: 600;
            }
            QLabel#boardTitle {
                color: #1d2a32;
                font-size: 12pt;
                font-weight: 700;
            }
            SonarBoardWidget {
                background: #f8fbfc;
                border: 1px solid #d7e0e5;
                border-radius: 4px;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d9dee3;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 11px;
                padding: 0 5px;
                color: #35424c;
                background: #f3f5f7;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #cbd3da;
                border-radius: 4px;
                padding: 7px 11px;
                color: #25313a;
            }
            QPushButton:hover {
                background: #f0f4f7;
                border-color: #91a1ad;
            }
            QPushButton:pressed {
                background: #e6ebef;
            }
            QPushButton:disabled {
                color: #a1aab1;
                background: #f2f4f5;
                border-color: #e0e4e7;
            }
            QPushButton#primaryButton {
                background: #187653;
                border-color: #187653;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background: #126244;
            }
            QPushButton#dangerButton {
                color: #a63342;
                border-color: #d5a5ac;
                font-weight: 700;
            }
            QPushButton#dangerButton:hover {
                background: #fbf0f2;
                border-color: #bd6975;
            }
            QProgressBar {
                border: 1px solid #d4dbe0;
                border-radius: 3px;
                background: #edf0f2;
                color: #26323a;
                text-align: center;
                font-size: 9pt;
                font-weight: 600;
            }
            QProgressBar#shotProgress::chunk { background: #3979a8; }
            QProgressBar#hitProgress::chunk { background: #16845c; }
            QProgressBar#shipProgress::chunk { background: #c97822; }
            QPlainTextEdit, QLineEdit, QComboBox, QTableWidget {
                background: #ffffff;
                border: 1px solid #cfd6dc;
                border-radius: 4px;
                selection-background-color: #cfe4f3;
                selection-color: #15212a;
            }
            QLineEdit, QComboBox {
                min-height: 30px;
                padding: 0 7px;
            }
            QPlainTextEdit {
                padding: 7px;
            }
            QPlainTextEdit#logView {
                font-family: "Cascadia Mono", Consolas, "Microsoft YaHei UI";
                font-size: 9.5pt;
                background: #fbfcfd;
                color: #1f2a32;
            }
            QHeaderView::section {
                background: #eef2f4;
                color: #4b5963;
                border: none;
                border-bottom: 1px solid #d6dde2;
                padding: 6px;
                font-weight: 700;
            }
            QTableWidget {
                gridline-color: #e1e6ea;
                alternate-background-color: #f8fafb;
            }
            QTabWidget::pane {
                background: #ffffff;
                border: 1px solid #d9dee3;
                border-radius: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background: #e9edf0;
                color: #596772;
                border: 1px solid #d1d8dd;
                padding: 7px 14px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #24313a;
                font-weight: 700;
                border-bottom-color: #ffffff;
            }
            QSplitter::handle {
                background: #dfe4e8;
                width: 5px;
                height: 5px;
                margin: 2px 4px;
            }
            QStatusBar {
                background: #e9edf0;
                color: #5b6872;
                border-top: 1px solid #d6dce1;
            }
            """
        )
        self.log_view.setObjectName("logView")
        self.operation_view.setStyleSheet(
            'font-family: "Cascadia Mono", Consolas, "Microsoft YaHei UI"; font-size: 9pt;'
        )

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_program)
        self.stop_button.clicked.connect(lambda: self.run_worker("stop"))
        self.restore_button.clicked.connect(lambda: self.run_worker("restore_network"))
        self.check_button.clicked.connect(lambda: self.run_worker("check_adb"))
        self.open_log_button.clicked.connect(self.open_log_file)
        self.clear_log_button.clicked.connect(self.clear_log_view)
        self.log_filter_combo.currentIndexChanged.connect(self.render_log)
        self.log_search.textChanged.connect(self.render_log)
        self.probe_mode_combo.currentIndexChanged.connect(self.update_controls)

    def start_program(self) -> None:
        pid_state = get_main_process()
        if pid_state is not None and pid_state[1]:
            pid, _running = pid_state
            QMessageBox.information(self, "程序已运行", f"检测到 main.py 已经在运行，PID={pid}")
            self.update_status()
            return

        if pid_state is not None:
            remove_pid(pid=pid_state[0])

        if not PYTHON_EXE.exists():
            QMessageBox.warning(self, "缺少运行环境", f"找不到 Python：{PYTHON_EXE}")
            return

        for path in (RUN_STDOUT, RUN_STDERR, STATUS_FILE):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                QMessageBox.warning(self, "无法清理旧文件", f"{path.name}：{exc}")
                return

        process = QProcess()
        process.setProgram(str(PYTHON_EXE))
        process.setArguments([str(MAIN_SCRIPT)])
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setStandardOutputFile(str(RUN_STDOUT))
        process.setStandardErrorFile(str(RUN_STDERR))
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in build_main_environment(self.probe_mode_combo.currentData(), self.red_scout_count.value()).items():
            environment.insert(key, value)
        process.setProcessEnvironment(environment)
        started, pid = process.startDetached()
        if not started:
            QMessageBox.warning(self, "启动失败", "无法启动 main.py，请检查运行环境和日志文件权限")
            return

        self.pending_pid = int(pid)
        self.runtime_status = {}
        self.log_lines.clear()
        self.current_log_path = RUN_STDERR
        self.last_log_size = 0
        self.log_view.clear()
        self.append_operation(f"已启动 main.py，进程 PID={pid}")
        self.update_status()

    def run_worker(self, action: str) -> None:
        if self.thread is not None:
            QMessageBox.information(self, "操作进行中", "请等待当前后台操作完成")
            return

        self.append_operation(f"开始执行：{self.action_label(action)}")
        self.thread = QThread(self)
        self.worker = Worker(action)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(
            lambda success, message: self.on_worker_finished(action, success, message)
        )
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()
        self.update_controls()

    def on_worker_finished(self, action: str, success: bool, message: str) -> None:
        if action in {"restore_network", "stop"} and success:
            self.network_status = "已连接"
        self.append_operation(message)
        if not success:
            QMessageBox.warning(self, self.action_label(action), message)
        self.update_status()

    def on_thread_finished(self) -> None:
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self.update_controls()

    def tick(self) -> None:
        self.update_status()
        self.append_new_log()

    def _resolve_running_process(self) -> tuple[int | None, bool]:
        pid_state = get_main_process()
        if pid_state is not None:
            pid, running = pid_state
            if running:
                self.pending_pid = None
                return pid, True
            remove_pid(pid=pid)

        if self.pending_pid is not None and is_pid_running(self.pending_pid):
            return self.pending_pid, True
        self.pending_pid = None
        return None, False

    def is_program_running(self) -> bool:
        _pid, running = self._resolve_running_process()
        return running

    def _runtime_status_needs_render(self, *, initial: bool = False) -> bool:
        signature = json.dumps(
            {
                "runtime_status": self.runtime_status,
                "network_status": self.network_status,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if not initial and signature == self.last_runtime_render_signature:
            return False
        self.last_runtime_render_signature = signature
        return True

    def update_status(self, initial: bool = False) -> None:
        previous_pid = self.last_running_pid
        self.main_pid, running = self._resolve_running_process()

        latest_status = read_runtime_status()
        if latest_status:
            self.runtime_status = latest_status
        elif not STATUS_FILE.exists():
            self.runtime_status = {}

        if running and not self.was_running and not initial:
            self.append_operation(f"检测到主程序正在运行，PID={self.main_pid}")
        elif not running and self.was_running and not initial:
            self.append_operation(f"main.py 已退出，上一进程 PID={previous_pid or '--'}")
        if running:
            self.last_running_pid = self.main_pid
        self.was_running = running

        process_signature = (running, self.main_pid)
        if initial or process_signature != self.last_process_render_signature:
            self._set_status_badge(running)
            self.pid_value.setText(str(self.main_pid) if self.main_pid is not None else "--")
            self.last_process_render_signature = process_signature

        if not self._runtime_status_needs_render(initial=initial):
            self.update_controls(running=running)
            return

        phase = self.runtime_status.get("phase", "--")
        level = self.runtime_status.get("level", "--")
        current_cell = self.runtime_status.get("current_cell", "--")
        shots_done = self.runtime_status.get("shots_done", "--")
        total_cells = self.runtime_status.get("total_cells", "--")
        hits = self.runtime_status.get("hits", "--")
        total_ship_cells = self.runtime_status.get("total_ship_cells", "--")
        confirmed_ships = self.runtime_status.get("confirmed_ships", "--")
        total_ships = self.runtime_status.get("total_ships", "--")
        last_result = self.runtime_status.get("last_result", "")
        network = repair_mojibake(str(self.runtime_status.get("network", self.network_status)))
        completed_lengths = self.runtime_status.get("sidebar_completed_lengths", [])
        mapped_hits = safe_int(self.runtime_status.get("mapped_visual_hits"))
        visual_hits = safe_int(self.runtime_status.get("initial_visual_hits"))
        unmapped_hits = safe_int(self.runtime_status.get("unmapped_visual_hits"))
        board_states = self.runtime_status.get("board_states", [])
        self.probe_mode_value.setText(format_probe_mode(self.runtime_status.get("probe_mode", "blue_only")))
        red_current = self.runtime_status.get("red_scout_current")
        red_total = self.runtime_status.get("red_scout_total")
        self.red_scout_progress_value.setText(
            format_red_scout_progress(
                current=red_current,
                total=red_total,
                valid=self.runtime_status.get("red_scout_valid"),
                complete_six=self.runtime_status.get("red_scout_complete_six"),
            )
        )

        self.network_value.setText(network)
        self._style_network_value(network)
        self.phase_value.setText(format_phase(phase))
        self.level_value.setText(str(level))
        self.current_cell_value.setText(format_cell(current_cell, total_cells))
        self.last_result_value.setText(format_result(last_result))
        if isinstance(completed_lengths, list) and completed_lengths:
            self.completed_ships_value.setText(", ".join(str(item) for item in completed_lengths))
        else:
            self.completed_ships_value.setText("--")
        if mapped_hits is not None and visual_hits is not None:
            extra = f"，未定位 {unmapped_hits}" if unmapped_hits else ""
            self.visual_mapping_value.setText(f"{mapped_hits}/{visual_hits}{extra}")
        else:
            self.visual_mapping_value.setText("--")

        self._set_progress(self.shot_progress, shots_done, total_cells)
        self._set_progress(self.hit_progress, hits, total_ship_cells)
        self._set_progress(self.ship_progress, confirmed_ships, total_ships)
        self.update_board_status(level, board_states, current_cell)
        self.update_recent_results()
        self.update_controls(running=running)

        updated_at = str(self.runtime_status.get("updated_at", ""))
        if updated_at:
            try:
                updated_text = datetime.fromisoformat(updated_at).strftime("%H:%M:%S")
            except ValueError:
                updated_text = updated_at
        else:
            updated_text = "--"
        self.last_update_label.setText(f"状态更新时间：{updated_text}")

    def update_board_status(
        self,
        level: object,
        board_states: object,
        current_cell: object,
    ) -> None:
        self.board_widget.set_board(board_states, current_cell)
        if self.board_widget.board_size == 0:
            self.board_level_label.setText("等待任务")
            self.board_summary_label.setText("未探测 --  ·  未命中 --\n已命中 --  ·  完整潜艇 --\n侦察未命中 --  ·  侦察命中 --")
            return

        size = self.board_widget.board_size
        current = safe_int(current_cell)
        current_text = f"当前目标 #{current}" if current is not None else "当前目标 --"
        self.board_level_label.setText(f"第 {level} 关 · {size} × {size}")
        counts = self.board_widget.state_counts()
        self.board_summary_label.setText(
            f"{current_text}\n"
            f"未探测 {counts['unknown']}  ·  未命中 {counts['miss']}\n"
            f"已命中 {counts['hit']}  ·  完整潜艇 {counts['ship']}\n"
            f"安全区 {counts['blocked']}\n"
            f"侦察未命中 {counts['scout_miss']}  ·  侦察命中 {counts['scout_hit']}"
        )

    def _set_status_badge(self, running: bool) -> None:
        if running:
            self.status_badge.setText("运行中")
            self.status_badge.setStyleSheet(
                "background:#dff3e9; color:#11623f; border:1px solid #9fd2ba; "
                "border-radius:4px; font-weight:700;"
            )
        else:
            self.status_badge.setText("已停止")
            self.status_badge.setStyleSheet(
                "background:#edf0f2; color:#5b6872; border:1px solid #d1d8dd; "
                "border-radius:4px; font-weight:700;"
            )

    def _style_network_value(self, network: str) -> None:
        if "断网" in network or "DROP" in network or "REJECT" in network:
            self.network_value.setStyleSheet("color:#b45f13; font-weight:700;")
        elif "连接" in network or "恢复" in network:
            self.network_value.setStyleSheet("color:#187653; font-weight:700;")
        else:
            self.network_value.setStyleSheet("color:#687680; font-weight:600;")

    @staticmethod
    def _set_progress(bar: QProgressBar, value: object, total: object) -> None:
        current = safe_int(value)
        maximum = safe_int(total)
        if current is None or maximum is None or maximum <= 0:
            bar.setRange(0, 1)
            bar.setValue(0)
            bar.setFormat("--")
            return
        current = max(0, min(current, maximum))
        bar.setRange(0, maximum)
        bar.setValue(current)
        bar.setFormat(f"{current} / {maximum}")

    def update_controls(self, running: bool | None = None) -> None:
        if running is None:
            _pid, running = self._resolve_running_process()
        busy = self.thread is not None
        self.start_button.setEnabled(not busy and not running)
        self.stop_button.setEnabled(not busy and running)
        self.restore_button.setEnabled(not busy)
        self.check_button.setEnabled(not busy)
        self.probe_mode_combo.setEnabled(not busy and not running)
        self.red_scout_count.setEnabled(not busy and not running and self.probe_mode_combo.currentData() == "red_scout")

    def update_recent_results(self) -> None:
        recent = self.runtime_status.get("recent_results", [])
        if not isinstance(recent, list):
            recent = []
        signature = json.dumps(recent[-5:], ensure_ascii=False, sort_keys=True)
        if signature == self.last_recent_signature:
            return
        self.last_recent_signature = signature

        rows = [item for item in recent[-5:] if isinstance(item, dict)]
        self.recent_table.setRowCount(len(rows))
        for row, item in enumerate(rows):
            result_key = str(item.get("result", ""))
            result_text = format_result(result_key)
            values = (
                str(item.get("time", "--")),
                f"L{item.get('level', '--')}  #{item.get('cell', '--')}",
                result_text,
                format_reason(item.get("reason", "")),
            )
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                if column == 2:
                    if result_key in {"hit", "hit_and_level_complete"}:
                        table_item.setForeground(QColor("#187653"))
                    elif result_key == "unknown":
                        table_item.setForeground(QColor("#b45f13"))
                    else:
                        table_item.setForeground(QColor("#5f6d77"))
                self.recent_table.setItem(row, column, table_item)

    def active_log_path(self) -> Path | None:
        if self.pending_pid is not None or self.is_program_running():
            return RUN_STDERR
        for path in (RUN_STDERR, LOG_FILE, RUN_STDOUT):
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    def reload_log(self) -> None:
        path = self.active_log_path()
        self.current_log_path = path
        self.log_lines.clear()
        self.last_log_size = 0

        if path is None or not path.exists():
            self.log_source_label.setText("来源：暂无日志")
            self.render_log()
            return

        try:
            data, size = read_log_tail(path)
        except OSError:
            self.log_source_label.setText(f"来源：{path.name}（读取失败）")
            self.render_log()
            return

        self.last_log_size = size
        self.log_lines.extend(decode_log_bytes(data).splitlines()[-MAX_LOG_LINES:])
        self.log_source_label.setText(f"来源：{path.name}")
        self.render_log()

    def append_new_log(self) -> None:
        path = self.active_log_path()
        if path != self.current_log_path:
            self.reload_log()
            return
        if path is None or not path.exists():
            return

        try:
            size = path.stat().st_size
            if size < self.last_log_size:
                self.reload_log()
                return
            if size == self.last_log_size:
                return
            with path.open("rb") as file:
                file.seek(self.last_log_size)
                data = file.read()
            self.last_log_size = size
        except OSError:
            return

        lines = decode_log_bytes(data).splitlines()
        if lines:
            self.log_lines.extend(lines)
            self.append_rendered_log_lines(lines)

    def _filter_log_lines(self, lines) -> list[str]:
        show_detail = bool(self.log_filter_combo.currentData())
        query = self.log_search.text().strip().casefold()
        return [
            line
            for line in lines
            if should_show_log_line(line, show_detail)
            and (not query or query in line.casefold())
        ]

    def append_rendered_log_lines(self, lines) -> None:
        filtered = self._filter_log_lines(lines)
        if not filtered:
            return
        self.log_view.appendPlainText("\n".join(filtered))
        if bool(self.auto_scroll_combo.currentData()):
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def render_log(self) -> None:
        filtered = self._filter_log_lines(self.log_lines)
        self.log_view.setPlainText("\n".join(filtered[-MAX_LOG_LINES:]))
        if bool(self.auto_scroll_combo.currentData()):
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def clear_log_view(self) -> None:
        self.log_lines.clear()
        self.log_view.clear()
        if self.current_log_path is not None and self.current_log_path.exists():
            try:
                self.last_log_size = self.current_log_path.stat().st_size
            except OSError:
                pass
        self.append_operation("已清空日志视图，日志文件未删除")

    def open_log_file(self) -> None:
        target = self.active_log_path()
        if target is None or not target.exists():
            QMessageBox.information(self, "日志不存在", "当前没有可打开的日志文件")
            return
        if os.name == "nt":
            os.startfile(target)
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def append_operation(self, message: str) -> None:
        self.operation_view.appendPlainText(f"{now_text()}  {repair_mojibake(message)}")
        self.operation_view.moveCursor(QTextCursor.MoveOperation.End)

    @staticmethod
    def action_label(action: str) -> str:
        return {
            "stop": "停止程序",
            "restore_network": "恢复网络",
            "check_adb": "检查模拟器",
        }.get(action, action)

    def closeEvent(self, event) -> None:
        if self.thread is not None:
            QMessageBox.information(self, "后台操作进行中", "请等待当前操作完成后再关闭控制台")
            event.ignore()
            return

        if self.is_program_running():
            reply = QMessageBox.question(
                self,
                "主程序仍在运行",
                "main.py 仍在运行。是否停止程序并恢复网络后再关闭控制台？\n\n"
                "选择“否”只关闭控制台，主程序会继续运行。",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.append_operation(stop_program())
                except Exception as exc:
                    QMessageBox.warning(self, "停止失败", repair_mojibake(str(exc)))
                    event.ignore()
                    return
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("BBMA Control Panel")
    app.setStyle("Fusion")

    PANEL_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    instance_lock = QLockFile(str(PANEL_LOCK_FILE))
    instance_lock.setStaleLockTime(0)
    if not instance_lock.tryLock(100):
        QMessageBox.information(None, "控制台已打开", "已有一个 BBMA 控制台正在运行")
        return 0

    window = ControlPanel()
    window._instance_lock = instance_lock
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
