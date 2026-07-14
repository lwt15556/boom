import atexit
import json
import os
import signal
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from config import (
    AUTO_DETECT_LEVEL,
    DEFAULT_LEVEL,
    GAME_PACKAGE_NAME,
    LEVEL_GRID_SIZES,
    LEVEL_REFERENCE_DIR,
    MAX_LEVEL,
    OUTPUT_DIR,
    REQUIRE_CONFIDENT_LEVEL_DETECTION,
    SCREENSHOT_DIR,
    SUBMARINES,
    TEMPLATE_DIR,
    USE_SAVED_POINTS,
)
from save_points.points import read_saved_points, read_saved_quad
from utils import AdbController, MatchResult, find_template, get_logger
from utils.diamond_centers import detect_diamond_centers
from utils.diamond_hit import classify_diamond_hit
from utils.hit_map import save_hit_map_image
from utils.image_match import find_template_multi_scale
from utils.level_recognition import recognize_level_from_screenshot
from utils.level_title_recognition import recognize_level_title
from utils.progress import (
    SearchProgress,
    fixed_progress_bar,
    format_elapsed,
    update_fixed_progress,
)
from utils.probe_protocol import (
    ProbeNotReadyError,
    ProbePhase,
    ProbeProtocolError,
    ProbeTransaction,
)
from utils.pending_probe import (
    clear_pending_probe,
    read_pending_probe,
    update_pending_probe,
    write_pending_probe,
)
from utils.runtime_lock import AlreadyRunningError, acquire_main_lock, remove_pid
from utils.sidebar_progress import (
    SidebarProgress,
    calculate_visible_hit_count,
    detect_partial_wreck_cells,
    detect_sidebar_progress,
    merge_confirmed_hit_count,
    newly_completed_lengths,
    progressive_hit_count,
    resolve_completed_ship_cells,
)
from utils.submarine_strategy import Cell, SubmarineStrategy, get_configured_submarines
from utils.wreck_detection import (
    VISIBLE_WRECK_TEMPLATES,
    detect_visible_wreck_cells,
    red_hit_marker_visible,
    visible_wreck_static_detected,
    wreck_template_visible,
)
from utils.red_scout import (
    AmmoFingerprint, ProbeMode, RedFootprint, RedScoutAnalyzer, RedScoutResult,
    RedScoutSettings, RedScoutPlanner, ammo_fingerprint_matches, build_ammo_fingerprint,
    load_red_scout_settings, locate_red_bomb_button, red_bomb_selected,
)

logger = get_logger(__name__)
adb = AdbController()

ACTIVITY_BUTTON_TEMPLATE = TEMPLATE_DIR / "activity_button.png"
LOGIN_TEMPLATE = TEMPLATE_DIR / "login.png"
QUIT_ACTIVITY_TEMPLATE = TEMPLATE_DIR / "quit_activity.png"
RETRY_TEMPLATE = TEMPLATE_DIR / "retry.png"
CONNECTION_INTERRUPTED_TEMPLATE = TEMPLATE_DIR / "connection_interrupted.png"
CONNECTION_RETRY_TEMPLATE = TEMPLATE_DIR / "connection_retry.png"
VICTORY_BANNER_TEMPLATE = TEMPLATE_DIR / "victory_banner.png"
RETRY_TEMPLATE_SCALES = (0.85, 0.95, 1.0, 1.05, 1.15)
RETRY_TEMPLATE_LOOSE_THRESHOLD = 0.72
CONNECTION_TEMPLATE_SCALES = (0.9, 1.0, 1.1)
CONNECTION_DIALOG_THRESHOLD = 0.78
CONNECTION_RETRY_THRESHOLD = 0.74
VICTORY_TEMPLATE_SCALES = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.65, 1.8)
VICTORY_BANNER_THRESHOLD = 0.80
VICTORY_WAIT_AFTER_HIT_SECONDS = 10.0
VICTORY_WAIT_BEFORE_LEVEL_SECONDS = 3.0
VICTORY_SKIP_SETTLE_SECONDS = 2.0
LEVEL_ADVANCE_RETRIES = 3
HIT_RESULT_FRAME_DELAYS = (1.0, 0.35, 0.45, 0.55)
SUSPECT_HIT_EXTRA_FRAME_DELAYS = (0.45, 0.55, 0.65)
MIN_HIT_RESULT_VOTES = 2
SUSPECT_HIT_SCORE_THRESHOLD = 0.78
STRONG_SINGLE_HIT_SCORE = 0.90
NEAR_HIT_SCORE_THRESHOLD = 0.52
NEAR_HIT_MIN_CHANGED_RATIO = 0.08
NEAR_HIT_MIN_CENTER_GRAY_RATIO = 0.065
NEAR_HIT_MIN_COMPONENT_RATIO = 0.020
NEAR_HIT_MIN_S_DROP = 4.0
NEAR_HIT_MIN_FRAMES = 3
FAST_POLL_INTERVAL_SECONDS = 0.25
PROBE_DROP_SETTLE_SECONDS = 0.2
MISS_REJECT_SETTLE_SECONDS = 0.2
APP_STOP_TIMEOUT_SECONDS = 5.0
APP_STOP_POLL_SECONDS = 0.1
POST_FORCE_STOP_GUARD_SECONDS = 0.5
REOPEN_GAME_SETTLE_SECONDS = 0.4
LOGIN_WAIT_AFTER_REOPEN_SECONDS = 14.0
ACTIVITY_BUTTON_WAIT_SECONDS = 8.0
POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS = 25.0
ACTIVITY_DETAIL_WAIT_SECONDS = 8.0
ACTIVITY_EXIT_WAIT_SECONDS = 1.0
ACTIVITY_EXIT_STABLE_FRAMES = 2
ACTIVITY_EXIT_CLICK_ATTEMPTS = 5
STATUS_REPLACE_RETRIES = 5
STATUS_REPLACE_RETRY_SECONDS = 0.05

ACTIVITY_DETAIL_POINT = (1205, 644)
ACTIVITY_LIST_SWIPE = (1000, 660, 1000, 180)
SCREEN_CONTINUE_POINT = (640, 360)
RUN_DEBUG_DIR = SCREENSHOT_DIR / "run_debug"
PROBE_SAMPLE_DIR = SCREENSHOT_DIR / "probes"
RUNTIME_DIR = SCREENSHOT_DIR.parent / "runtime"
STATUS_FILE = RUNTIME_DIR / "status.json"
LEVEL_STATE_FILE = RUNTIME_DIR / "level_state.json"

_weak_network_cleanup_done = False
_active_probe: "ProbeTransaction | None" = None
_runtime_status: dict[str, object] = {}
_network_fail_closed_reason: str | None = None


class RedScoutSafetyError(RuntimeError):
    pass


class ProbeResult(str, Enum):
    MISS = "miss"
    HIT = "hit"
    HIT_AND_LEVEL_COMPLETE = "hit_and_level_complete"
    LEVEL_COMPLETE = "level_complete"
    UNKNOWN = "unknown"


def _probe_result_is_hit(result: ProbeResult) -> bool:
    return result in {ProbeResult.HIT, ProbeResult.HIT_AND_LEVEL_COMPLETE}


def _probe_result_completed_level(result: ProbeResult) -> bool:
    return result in {
        ProbeResult.HIT_AND_LEVEL_COMPLETE,
        ProbeResult.LEVEL_COMPLETE,
    }


def build_runtime_board_states(strategy: object, grid_size: int) -> list[list[str]]:
    """Build a stable JSON-friendly board snapshot for the control panel."""
    getter = getattr(strategy, "get_cell_states", None)
    if callable(getter):
        states = getter()
        if (
            isinstance(states, list)
            and len(states) == grid_size
            and all(isinstance(row, list) and len(row) == grid_size for row in states)
        ):
            return states

    states = [["unknown" for _col in range(grid_size)] for _row in range(grid_size)]
    for row, col in getattr(strategy, "blocked_cells", set()):
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "blocked"
    for (row, col), hit in getattr(strategy, "shots", {}).items():
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "hit" if hit else "miss"
    return states


def build_red_scout_board_states(
    grid_size: int,
    *,
    hits: set[Cell],
    misses: set[Cell],
    initial_hits: set[Cell] | None = None,
) -> list[list[str]]:
    states = [["unknown" for _col in range(grid_size)] for _row in range(grid_size)]
    for row, col in misses - hits:
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "scout_miss"
    for row, col in hits:
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "scout_hit"
    for row, col in initial_hits or set():
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "hit"
    return states


def merge_red_scout_observations(
    hits: set[Cell],
    misses: set[Cell],
    result: RedScoutResult,
) -> None:
    incoming_hits = set(result.hit_cells)
    incoming_misses = set(result.miss_cells)
    conflicts = (incoming_hits & misses) | (incoming_misses & hits)
    if conflicts:
        logger.warning(
            "red scout observations overlap previous attempts at %s; keeping hit evidence",
            sorted(conflicts),
        )

    hits.update(incoming_hits)
    misses.difference_update(incoming_hits)
    misses.update(incoming_misses - hits)


def write_runtime_status(**updates: object) -> None:
    """Write lightweight machine-readable status for the control panel."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _runtime_status.update(updates)
    _runtime_status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temp_path = STATUS_FILE.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(_runtime_status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for attempt in range(STATUS_REPLACE_RETRIES):
        try:
            temp_path.replace(STATUS_FILE)
        except PermissionError as exc:
            if attempt + 1 >= STATUS_REPLACE_RETRIES:
                logger.warning("runtime status is locked; skipping update: %s", exc)
                try:
                    temp_path.unlink()
                except OSError:
                    pass
                return
            sleep(STATUS_REPLACE_RETRY_SECONDS)
        else:
            return


def append_recent_probe_result(
    *,
    level: int,
    index: int,
    result: ProbeResult,
    reason: str,
) -> None:
    recent = list(_runtime_status.get("recent_results", []))
    recent.append(
        {
            "level": level,
            "cell": index,
            "result": result.value,
            "reason": reason,
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )
    write_runtime_status(recent_results=recent[-5:])


def load_level_state() -> dict:
    try:
        return json.loads(LEVEL_STATE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"levels": {}}


def get_state_profile() -> str | None:
    profile = os.environ.get("BBMA_PROFILE", "").strip()
    return profile or None


def load_saved_level_shots(level: int, grid_size: int) -> dict[Cell, bool]:
    profile = get_state_profile()
    if profile is None:
        return {}

    state = load_level_state()
    profile_state = state.get("profiles", {}).get(profile, {})
    level_state = profile_state.get("levels", {}).get(str(level), {})
    if int(level_state.get("grid_size", 0) or 0) != int(grid_size):
        return {}

    shots: dict[Cell, bool] = {}
    for item in level_state.get("shots", []):
        try:
            row, col = item["cell"]
            cell = (int(row), int(col))
            hit = bool(item["hit"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size:
            shots[cell] = hit
    return shots


def save_level_shots(level: int, grid_size: int, shots: Mapping[Cell, bool]) -> None:
    profile = get_state_profile()
    if profile is None:
        return

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    state = load_level_state()
    profiles = state.setdefault("profiles", {})
    profile_state = profiles.setdefault(profile, {})
    profile_state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    levels = profile_state.setdefault("levels", {})
    levels[str(level)] = {
        "grid_size": grid_size,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "shots": [
            {
                "cell": [row, col],
                "hit": bool(hit),
            }
            for (row, col), hit in sorted(shots.items())
        ],
    }
    temp_path = LEVEL_STATE_FILE.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(LEVEL_STATE_FILE)


def _has_pending_probe_request() -> bool:
    return _active_probe is not None and _active_probe.request_may_be_pending


def _create_probe_sample_dir(level: int, cell: Cell, index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    row, col = cell
    sample_dir = PROBE_SAMPLE_DIR / f"level_{level}_cell_{index}_r{row}_c{col}_{timestamp}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def _write_probe_status(sample_dir: Path, stage: str, **extra) -> None:
    payload = {
        "stage": stage,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    (sample_dir / "status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _hit_result_to_dict(result) -> dict:
    return {
        "state": result.state,
        "confidence": float(getattr(result, "confidence", 0.0)),
        "score": float(result.score),
        "rough_center": list(result.rough_center),
        "refined_center": list(result.refined_center),
        "changed_ratio": float(result.changed_ratio),
        "center_gray_ratio": float(result.center_gray_ratio),
        "ring_gray_ratio": float(getattr(result, "ring_gray_ratio", 0.0)),
        "gray_excess": float(result.gray_excess),
        "component_ratio": float(result.component_ratio),
        "s_center": float(getattr(result, "s_center", 0.0)),
        "s_ring": float(getattr(result, "s_ring", 0.0)),
        "s_drop": float(result.s_drop),
        "edge_density": float(result.edge_density),
        "evidence_vetoed": bool(getattr(result, "evidence_vetoed", False)),
    }


def _save_probe_result_json(
    sample_dir: Path,
    *,
    level: int,
    cell: Cell,
    index: int,
    point: tuple[int, int],
    hit: bool,
    hit_votes: int,
    frames: list[dict],
    suspect_extra_checked: bool,
    decision_reason: str = "",
) -> None:
    payload = {
        "level": level,
        "cell": list(cell),
        "index": index,
        "point": list(point),
        "decision": "hit" if hit else "miss",
        "hit_votes": hit_votes,
        "decision_reason": decision_reason,
        "frame_count": len(frames),
        "min_hit_votes": MIN_HIT_RESULT_VOTES,
        "suspect_extra_checked": suspect_extra_checked,
        "frames": frames,
    }
    (sample_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_near_hit_frame(result) -> bool:
    if bool(getattr(result, "evidence_vetoed", False)):
        return False
    return (
        result.score >= NEAR_HIT_SCORE_THRESHOLD
        and result.changed_ratio >= NEAR_HIT_MIN_CHANGED_RATIO
        and result.center_gray_ratio >= NEAR_HIT_MIN_CENTER_GRAY_RATIO
        and result.component_ratio >= NEAR_HIT_MIN_COMPONENT_RATIO
        and result.s_drop >= NEAR_HIT_MIN_S_DROP
    )


def _is_suspect_hit_frame(result) -> bool:
    return result.state == "hit" or result.score >= SUSPECT_HIT_SCORE_THRESHOLD or _is_near_hit_frame(result)


def decide_hit_from_frames(hit_results: list) -> tuple[bool, str]:
    hit_votes = sum(1 for result in hit_results if result.state == "hit")
    if hit_votes >= MIN_HIT_RESULT_VOTES:
        return True, f"hit_votes_{hit_votes}"

    strong_hits = [
        result
        for result in hit_results
        if result.state == "hit"
        and (
            result.score >= STRONG_SINGLE_HIT_SCORE
            or result.confidence >= 0.92
        )
    ]
    if strong_hits:
        return True, f"strong_single_score_{max(result.score for result in strong_hits):.3f}"

    near_hits = [result for result in hit_results if _is_near_hit_frame(result)]
    if len(near_hits) >= NEAR_HIT_MIN_FRAMES:
        return True, f"near_hit_frames_{len(near_hits)}"

    return False, f"hit_votes_{hit_votes}_near_{len(near_hits)}"


def apply_wreck_template_confirmation(after_img: np.ndarray, point: tuple[int, int], result) -> bool:
    if red_hit_marker_visible(after_img, point):
        result.state = "hit"
        result.score = max(float(result.score), 0.98)
        result.confidence = max(float(result.confidence), 0.98)
        return True

    if not visible_wreck_static_detected(after_img, point):
        return False

    result.state = "hit"
    result.score = max(float(result.score), 0.94)
    result.confidence = max(float(result.confidence), 0.95)
    return True


def apply_sidebar_completion_confirmation(
    before_img: np.ndarray,
    after_img: np.ndarray,
    submarine_lengths: Sequence[int],
    result,
) -> tuple[bool, SidebarProgress | None, tuple[int, ...]]:
    before_progress = detect_sidebar_progress(before_img, submarine_lengths)
    after_progress = detect_sidebar_progress(after_img, submarine_lengths)
    newly_completed = newly_completed_lengths(before_progress, after_progress)
    if not newly_completed:
        return False, after_progress, ()

    result.state = "hit"
    result.score = max(float(result.score), 0.99)
    result.confidence = max(float(result.confidence), 0.99)
    return True, after_progress, newly_completed


def enforce_positive_hit_evidence(
    result,
    *,
    wreck_hit: bool,
    sidebar_hit: bool,
) -> bool:
    """Reject a dynamic hit unless a new wreck/red marker or sidebar change supports it."""
    if result.state != "hit" or wreck_hit or sidebar_hit:
        return False

    result.evidence_vetoed = True
    result.state = "miss"
    result.score = min(float(result.score), SUSPECT_HIT_SCORE_THRESHOLD - 0.01)
    result.confidence = max(float(result.confidence), 1.0 - float(result.score))
    return True


def enable_weak_network(second: float = 0) -> None:
    """弢启游戏弱网，并按霢等待网络状生效"""
    adb.enable_weak_network(GAME_PACKAGE_NAME)
    write_runtime_status(network="断网中")
    if second > 0:
        sleep(second)


def disable_weak_network(second: float = 0) -> None:
    """安全关闭游戏弱网；存在待丢弃请求时拒绝恢复网络"""
    if _has_pending_probe_request():
        transaction = _active_probe
        raise ProbeProtocolError(
            "pending probe request may still exist; refuse to disable DROP weak network "
            f"cell={transaction.cell if transaction else None} "
            f"phase={transaction.phase.name if transaction else None}"
        )
    adb.disable_weak_network(GAME_PACKAGE_NAME)
    write_runtime_status(network="已连接")
    if second > 0:
        sleep(second)


def cleanup_weak_network(reason: str = "脚本退出") -> None:
    """仅在不存在待发探测请求时关闭 DROP 弱网。"""
    global _weak_network_cleanup_done
    if _weak_network_cleanup_done:
        return
    if _network_fail_closed_reason is not None:
        logger.critical("network cleanup refused: %s", _network_fail_closed_reason)
        return

    if _has_pending_probe_request():
        transaction = _active_probe
        logger.critical(
            "%s，但格子 %s 的探测处于 %s；为避免暂存请求补发，保留 DROP 弱网",
            reason,
            transaction.cell if transaction else None,
            transaction.phase.name if transaction else None,
        )
        return

    try:
        logger.info("%s, disabling weak network", reason)
        disable_weak_network()
    except Exception as exc:
        logger.error("关闭弱网失败: %s", exc)
    else:
        _weak_network_cleanup_done = True


def latch_network_fail_closed(reason: str) -> None:
    global _network_fail_closed_reason
    _network_fail_closed_reason = str(reason)
    write_runtime_status(network="fail_closed", network_fail_closed_reason=_network_fail_closed_reason)


def _capture_red_ammo_state():
    frames = [adb.read_screenshot(RUN_DEBUG_DIR / f"red_ammo_{i}.png") for i in range(3)]
    match = locate_red_bomb_button(frames[0])
    fingerprint = build_ammo_fingerprint(frames, match) if match is not None else None
    if match is None or fingerprint is None:
        raise RedScoutSafetyError("red bomb button or ammo fingerprint unavailable")
    return frames[0], fingerprint, match


def _select_red_bomb(match: MatchResult) -> bool:
    adb.click(*match.center)
    adb.delay(0.25)
    return red_bomb_selected(adb.read_screenshot(RUN_DEBUG_DIR / "red_selected.png"), match)


def _capture_red_result_frames():
    return [
        adb.delay(frame_delay).read_screenshot(
            RUN_DEBUG_DIR / f"red_result_{frame_index}.png"
        )
        for frame_index, frame_delay in enumerate(HIT_RESULT_FRAME_DELAYS)
    ]


def _verify_red_ammo_unchanged(before_fingerprint: AmmoFingerprint) -> None:
    write_runtime_status(phase="red_scout_verify_ammo")
    after_state = _capture_red_ammo_state()
    if not ammo_fingerprint_matches(before_fingerprint, after_state[1]):
        _stop_and_latch_red_safety_failure("red ammo fingerprint mismatch")
    clear_pending_probe()


def _wait_until_activity_detail_closed(
    timeout: float = ACTIVITY_EXIT_WAIT_SECONDS,
) -> bool:
    logger.info("waiting up to %.1f seconds for activity detail to close", timeout)
    start_time = monotonic()
    absent_frames = 0
    while monotonic() - start_time < timeout:
        screenshot = adb.read_screenshot()
        if isinstance(screenshot, np.ndarray):
            detail_open = find_template(screenshot, QUIT_ACTIVITY_TEMPLATE) is not None
            absent_frames = 0 if detail_open else absent_frames + 1
            if absent_frames >= ACTIVITY_EXIT_STABLE_FRAMES:
                logger.info("activity detail exit confirmed; starting offline re-entry")
                return True
        else:
            absent_frames = 0
        sleep(FAST_POLL_INTERVAL_SECONDS)

    logger.warning("activity detail did not close within %.1f seconds", timeout)
    return False


def _exit_activity_after_probe_click(
    debug_path: Path,
    *,
    use_system_back: bool = False,
) -> None:
    adb.delay(0.3)
    if use_system_back:
        for attempt in range(1, ACTIVITY_EXIT_CLICK_ATTEMPTS + 1):
            attempt_path = (
                debug_path
                if attempt == 1
                else debug_path.with_name(
                    f"{debug_path.stem}_retry_{attempt - 1}{debug_path.suffix}"
                )
            )
            if attempt > 1:
                logger.warning(
                    "system back did not leave the red scout activity; retrying after "
                    "the attack animation (%s/%s)",
                    attempt,
                    ACTIVITY_EXIT_CLICK_ATTEMPTS,
                )
            adb.read_screenshot(attempt_path)
            adb.back()
            if _wait_until_activity_detail_closed():
                return

        raise ProbeProtocolError(
            "system back did not exit the red scout activity after repeated attempts; "
            "pending request state is unknown"
        )

    for attempt in range(1, ACTIVITY_EXIT_CLICK_ATTEMPTS + 1):
        attempt_path = (
            debug_path
            if attempt == 1
            else debug_path.with_name(
                f"{debug_path.stem}_retry_{attempt - 1}{debug_path.suffix}"
            )
        )
        if attempt > 1:
            logger.warning(
                "activity exit click was ignored; retrying after the attack animation "
                "(%s/%s)",
                attempt,
                ACTIVITY_EXIT_CLICK_ATTEMPTS,
            )

        if not click_template(QUIT_ACTIVITY_TEMPLATE, attempt_path):
            if attempt == 1:
                raise ProbeProtocolError(
                    "probe click could not exit the activity; pending request state is unknown"
                )
            if _wait_until_activity_detail_closed():
                return
            continue

        if _wait_until_activity_detail_closed():
            return

    raise ProbeProtocolError(
        "activity exit was not confirmed after repeated clicks; "
        "pending request state is unknown"
    )


def _reenter_activity_for_probe_result() -> bool:
    return enter_activity(re_enter=True, max_retries=1) is True


def _analyze_red_result(
    before_image,
    after_images,
    click_points,
    grid_size,
    center_cell,
):
    return RedScoutAnalyzer().analyze(
        before_image=before_image,
        after_images=after_images,
        click_points=click_points,
        grid_size=grid_size,
        center_cell=center_cell,
        excluded_cells=set(),
        learned_footprint=None,
    )


def _stop_and_latch_red_safety_failure(reason: str) -> None:
    global _network_fail_closed_reason
    first_reason = str(reason)
    # Latch before any safety operation can fail; cleanup must never restore it.
    if _network_fail_closed_reason is None:
        _network_fail_closed_reason = first_reason
        try:
            write_runtime_status(
                network="fail_closed",
                network_fail_closed_reason=_network_fail_closed_reason,
            )
        except Exception as exc:
            logger.error("could not persist red fail-closed latch: %s", exc)

    stopped = False
    operations = (
        ("enable reject network", lambda: adb.enable_reject_network(GAME_PACKAGE_NAME)),
        ("close app", lambda: adb.close_app(GAME_PACKAGE_NAME)),
        ("wait for app stop", lambda: adb.wait_until_app_stopped(
            GAME_PACKAGE_NAME,
            timeout=APP_STOP_TIMEOUT_SECONDS,
            poll_interval=APP_STOP_POLL_SECONDS,
        )),
        ("post-stop delay", lambda: adb.delay(POST_FORCE_STOP_GUARD_SECONDS)),
    )
    for name, operation in operations:
        try:
            result = operation()
            if name == "wait for app stop":
                stopped = bool(result)
        except Exception as exc:
            logger.error("red safety stop operation failed (%s): %s", name, exc)
    final_reason = first_reason
    if not stopped:
        final_reason = f"{final_reason}; process did not exit"
    raise RedScoutSafetyError(final_reason)


def recover_interrupted_probe_at_startup() -> bool:
    pending = read_pending_probe()
    if pending is None:
        return False

    logger.critical(
        "detected interrupted %s probe at phase=%s; blocking network and force-stopping "
        "the game before normal startup",
        pending.get("mode", "unknown"),
        pending.get("phase", "unknown"),
    )
    adb.enable_weak_network(GAME_PACKAGE_NAME)
    adb.enable_reject_network(GAME_PACKAGE_NAME)
    write_runtime_status(
        phase="stale_probe_recovery",
        network="断网中",
        stale_probe=pending,
    )
    adb.delay(PROBE_DROP_SETTLE_SECONDS)
    adb.close_app(GAME_PACKAGE_NAME)
    if not adb.wait_until_app_stopped(
        GAME_PACKAGE_NAME,
        timeout=APP_STOP_TIMEOUT_SECONDS,
        poll_interval=APP_STOP_POLL_SECONDS,
    ):
        latch_network_fail_closed("interrupted probe recovery could not stop the game")
        raise RedScoutSafetyError(
            "interrupted probe recovery could not stop the game; network remains blocked"
        )
    adb.delay(POST_FORCE_STOP_GUARD_SECONDS)
    clear_pending_probe()
    write_runtime_status(
        phase="stale_probe_recovered",
        running=False,
        network="断网中",
        stale_probe={},
    )
    return True


def _execute_red_scout_transaction(
    level: int,
    center_cell: Cell,
    point: tuple[int, int],
    index: int,
    grid_size: int,
    all_click_points: Sequence[tuple[int, int]],
):
    global _active_probe
    transaction = None
    grid_clicked = False
    pending_marker_written = False
    try:
        write_runtime_status(phase="red_scout_preflight", level=level)
        enable_weak_network(PROBE_DROP_SETTLE_SECONDS)
        isolation = adb.verify_app_network_isolated(GAME_PACKAGE_NAME)
        if not bool(getattr(isolation, "safe", False)):
            _stop_and_latch_red_safety_failure(
                getattr(isolation, "detail", "network isolation unsafe")
            )
        before_image, before_fingerprint, match = _capture_red_ammo_state()
        transaction = ProbeTransaction(level, center_cell, index)
        _active_probe = transaction
        transaction.advance(ProbePhase.REQUEST_PENDING)
        if not _select_red_bomb(match):
            raise RedScoutSafetyError("red bomb selection not confirmed")
        write_pending_probe(
            mode=ProbeMode.RED_SCOUT.value,
            level=level,
            cell=center_cell,
            index=index,
            phase=ProbePhase.REQUEST_PENDING.name,
        )
        pending_marker_written = True
        # Once the input command is issued, conservatively assume the request exists
        # even if adb reports an error while returning the command result.
        grid_clicked = True
        adb.click(*point)
        _exit_activity_after_probe_click(
            RUN_DEBUG_DIR / "red_debug_back.png",
            use_system_back=True,
        )
        if _reenter_activity_for_probe_result():
            transaction.advance(ProbePhase.RESULT_VISIBLE)
            transaction.advance(ProbePhase.RESULT_RECORDED)
            update_pending_probe(phase=ProbePhase.RESULT_RECORDED.name, local_victory=True)
            logger.warning(
                "red scout displayed a local victory; discarding it and continuing with blue "
                "attacks because the red request must never be committed"
            )
            _discard_pending_request_and_prepare_next_probe(transaction)
            _verify_red_ammo_unchanged(before_fingerprint)
            pending_marker_written = False
            _active_probe = None
            return RedScoutResult(
                center_cell=center_cell,
                affected_cells=frozenset(),
                hit_cells=frozenset(),
                miss_cells=frozenset(),
                unknown_cells=frozenset(),
                footprint=None,
                valid=False,
                confidence_by_cell={},
                level_completed=False,
            )
        transaction.advance(ProbePhase.RESULT_VISIBLE)
        update_pending_probe(phase=ProbePhase.RESULT_VISIBLE.name)
        write_runtime_status(phase="red_scout_capture", level=level)
        after_images = _capture_red_result_frames()
        transaction.advance(ProbePhase.RESULT_RECORDED)
        update_pending_probe(phase=ProbePhase.RESULT_RECORDED.name)
        analysis = _analyze_red_result(
            before_image,
            after_images,
            all_click_points,
            grid_size,
            center_cell,
        )
        write_runtime_status(phase="red_scout_discard", level=level)
        _discard_pending_request_and_prepare_next_probe(transaction)
        if (
            transaction.phase is not ProbePhase.COMPLETE
            or not getattr(transaction, "red_request_discarded", False)
        ):
            _stop_and_latch_red_safety_failure(
                f"red discard contract violated: phase={transaction.phase.name}"
            )
        _verify_red_ammo_unchanged(before_fingerprint)
        pending_marker_written = False
        _active_probe = None
        return analysis
    except Exception as exc:
        if grid_clicked:
            try:
                update_pending_probe(phase="INTERRUPTED", error=str(exc))
            except Exception as marker_exc:
                logger.error("could not update interrupted red probe marker: %s", marker_exc)
            if isinstance(exc, RedScoutSafetyError) and _network_fail_closed_reason is not None:
                raise
            _stop_and_latch_red_safety_failure(
                f"red scout transaction interrupted: {exc}"
            )
        if pending_marker_written:
            clear_pending_probe()
        _active_probe = None
        raise


def cleanup_reject_network(reason: str = "脚本退出") -> None:
    """关闭游戏 REJECT 断网残留，避免影响本次或下次运行。"""
    try:
        logger.info("%s，正在清理 REJECT 断网", reason)
        adb.disable_reject_network(GAME_PACKAGE_NAME)
    except Exception as exc:
        logger.error("清理 REJECT 断网失败: %s", exc)


def handle_exit_signal(signum: int, _frame) -> None:
    """收到逢出信号时先关闭弱网再逢出"""
    cleanup_weak_network(f"收到退出信号 {signum}")
    raise SystemExit(128 + signum)


def register_exit_cleanup() -> None:
    """注册脚本退出清理，尽量避免弱网规则残留。"""
    atexit.register(cleanup_weak_network)
    for signame in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, signame, None)
        if signum is not None:
            signal.signal(signum, handle_exit_signal)


def enter_activity(
    re_enter: bool = False,
    max_retries: int = 5,
    *,
    activity_button_timeout: float | None = None,
) -> bool:
    """进入活动详情页。

    ``re_enter=False`` 用于没有待验证请求的普通进入，允许重启恢复。
    ``re_enter=True`` 用于点击后的第二次进入，此时 DROP 下可能仍有暂存请求，
    任何失败都必须立即中止，不能复用会关闭弱网的普通恢复流程。
    刚登录后的活动入口加载较慢，可通过 ``activity_button_timeout`` 延长轮询。
    """
    if max_retries <= 0:
        raise ValueError(f"max_retries 必须大于 0: {max_retries}")
    button_timeout = (
        ACTIVITY_BUTTON_WAIT_SECONDS
        if activity_button_timeout is None
        else float(activity_button_timeout)
    )
    if button_timeout <= 0:
        raise ValueError(f"activity_button_timeout 必须大于 0: {button_timeout}")

    last_failure = "进入活动失败"
    level_completed = False
    for attempt in range(1, max_retries + 1):
        adb.delay(0.2)
        screenshot = adb.read_screenshot()
        detail_open = (
            isinstance(screenshot, np.ndarray)
            and find_template(screenshot, QUIT_ACTIVITY_TEMPLATE) is not None
        )
        if detail_open and not re_enter:
            logger.info("already in activity detail; fast path")
            return level_completed
        if detail_open:
            logger.warning(
                "fresh activity re-entry requested while the old detail view is still visible; "
                "waiting for the activity button"
            )
        if not re_enter:
            if handle_victory_prompt(timeout=0.0, screenshot=screenshot):
                level_completed = True
                logger.info("victory banner skipped before entering activity")
                continue

        res = wait_until_occur(ACTIVITY_BUTTON_TEMPLATE, timeout=button_timeout)
        if res is None:
            last_failure = "activity button not found"
            if re_enter:
                raise ProbeProtocolError(
                    f"第二次进入活动失败: {last_failure}; keep DROP weak network and stop probing"
                )
            logger.warning(
                "%s，无法进入活动界面，正在重试 (%s/%s)",
                last_failure,
                attempt,
                max_retries,
            )
            _restart_game_for_activity_retry()
            button_timeout = POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS
            continue

        adb.click(*res.center)  # 点击活动按钮进入活动界面
        if not re_enter:
            enable_weak_network(0.2)
            adb.delay(0.4).swipe(*ACTIVITY_LIST_SWIPE)  # 首次进入霢要展示全部项
            adb.delay(0.2).swipe(*ACTIVITY_LIST_SWIPE)

        adb.delay(0.35).click(*ACTIVITY_DETAIL_POINT)
        if wait_until_occur(QUIT_ACTIVITY_TEMPLATE, timeout=ACTIVITY_DETAIL_WAIT_SECONDS) is not None:
            return level_completed

        recovery = recover_activity_detail_timeout(re_enter=re_enter)
        if recovery == "ready":
            return level_completed
        if recovery == "level_complete":
            level_completed = True
            if re_enter:
                return True
            continue
        if recovery == "retry":
            continue

        last_failure = "进入活动详情界面失败"
        if re_enter:
            raise ProbeProtocolError(
                f"第二次进入活动失败: {last_failure}; keep DROP weak network and stop probing"
            )
        logger.warning(
            "%s，正在重试进入活动 (%s/%s)",
            last_failure,
            attempt,
            max_retries,
        )
        _restart_game_for_activity_retry()
        button_timeout = POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS

    message = f"{last_failure}，已达到最大重试次数 {max_retries}"
    logger.error(message)
    raise RuntimeError(message)


def recover_activity_detail_timeout(re_enter: bool) -> str:
    """Return 'ready', 'retry', or 'unhandled' after an activity-detail timeout."""
    screenshot = adb.read_screenshot()
    if find_template(screenshot, QUIT_ACTIVITY_TEMPLATE) is not None:
        logger.info("activity detail was detected after timeout; continuing")
        return "ready"

    if handle_victory_prompt(
        timeout=0.0,
        screenshot=screenshot,
        restore_network=not re_enter,
    ):
        logger.info("victory banner handled after activity-detail timeout; retrying entry")
        return "level_complete"

    if re_enter:
        return "unhandled"

    try:
        if handle_connection_interrupted_prompt(timeout=6.0):
            logger.info("connection dialog handled after activity-detail timeout; retrying entry")
            return "retry"
    except ProbeProtocolError as exc:
        logger.warning("connection dialog recovery after activity-detail timeout failed: %s", exc)

    return "unhandled"


def _restart_game_for_activity_retry() -> None:
    """在没有待验证请求的普通进入阶段重启游戏"""
    if _has_pending_probe_request():
        raise ProbeProtocolError("存在待发送探测请求，禁止通过重启游戏恢复活动入口")

    adb.close_app(GAME_PACKAGE_NAME)
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    disable_weak_network()
    adb.delay(1.5).open_app(GAME_PACKAGE_NAME)
    login_img = wait_until_occur(LOGIN_TEMPLATE, timeout=30)
    if login_img is None:
        logger.warning("restarted game but login button was not found; continuing")
        return
    adb.click(*login_img.center)  # 点击登录按钮


def get_level_grid_size(level: int) -> int:
    """读取指定关卡的菱形网格边长"""
    if level not in LEVEL_GRID_SIZES:
        raise ValueError(f"未配置第 {level} 关的网格边长")
    return LEVEL_GRID_SIZES[level]


def reset_runtime_level_status(level: int) -> None:
    """Publish a clean board immediately when a new level becomes active."""
    grid_size = get_level_grid_size(level)
    submarines = get_configured_submarines(level, SUBMARINES) or ()
    write_runtime_status(
        phase="level_loading",
        level=level,
        current_cell="--",
        shots_done=0,
        total_cells=grid_size * grid_size,
        hits=0,
        total_ship_cells=sum(submarines),
        confirmed_ships=0,
        total_ships=len(submarines),
        sidebar_completed_cells=0,
        sidebar_completed_lengths=[],
        sidebar_newly_completed_lengths=[],
        initial_visual_hits=0,
        mapped_visual_hits=0,
        unmapped_visual_hits=0,
        board_size=grid_size,
        board_states=[
            ["unknown" for _col in range(grid_size)]
            for _row in range(grid_size)
        ],
        recent_results=[],
        last_result="",
        red_scout_current=0,
        red_scout_total=0,
    )


def get_click_points(
    level: int, grid_img: np.ndarray
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """按配置读取人工点位，失败时回逢到自动识别"""
    grid_size = get_level_grid_size(level)

    if USE_SAVED_POINTS:
        try:
            saved_points = read_saved_points(level, expected_n=grid_size)
            saved_quad = read_saved_quad(level)
        except Exception as exc:
            logger.warning("failed to read saved points for level %s; falling back to auto detection: %s", level, exc)
        else:
            if saved_points is not None and saved_quad is not None:
                logger.info("level %s uses saved calibration points: %s", level, len(saved_points))
                return saved_points, saved_quad
            logger.warning("第 %s 关人工点位不存在或数量不正确，回退自动识别", level)

    grid_result = detect_diamond_centers(grid_img, grid_size)
    logger.info("level %s uses auto-detected points: %s", level, len(grid_result.points))
    return grid_result.points, grid_result.global_quad


def handle_game_level(
    level: int,
    hit_map: list[list[int]],
    run_started_at: float | None = None,
    settings: RedScoutSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """处理单个关卡：有潜艇配置时策略点，缺少配置时回逐格扫描"""
    adb.delay(1.5)
    grid_img = adb.read_screenshot()
    click_points, grid_quad = get_click_points(level, grid_img)
    grid_size = get_level_grid_size(level)

    submarines = get_configured_submarines(level, SUBMARINES)
    visible_hits: set[Cell] = set()
    initial_visual_hits: set[Cell] = set()
    completed_visual_hits: set[Cell] = set()
    sidebar_progress: SidebarProgress | None = None
    partial_wreck_cells: set[Cell] | None = None
    initial_visual_hit_count: int | None = None
    if submarines is not None:
        detected_sidebar_progress = detect_sidebar_progress(grid_img, submarines)
        if detected_sidebar_progress is not None and detected_sidebar_progress.valid:
            sidebar_progress = detected_sidebar_progress
            logger.info(
                "level %s sidebar progress: completed_lengths=%s completed_cells=%s",
                level,
                list(sidebar_progress.completed_lengths),
                sidebar_progress.completed_cells,
            )
        else:
            logger.warning("level %s sidebar progress was not confidently recognized", level)
        visible_hits = detect_visible_wreck_cells(grid_img, click_points, grid_size)
        max_visible_hits = sum(submarines)
        if len(visible_hits) > max_visible_hits:
            logger.warning(
                "level %s visible wreck review ignored suspicious result: %s/%s cells",
                level,
                len(visible_hits),
                grid_size * grid_size,
            )
            visible_hits = set()
        elif visible_hits:
            logger.info("level %s visible wreck review found %s hit cells", level, len(visible_hits))

        partial_wreck_cells = detect_partial_wreck_cells(
            grid_img,
            click_points,
            grid_size=grid_size,
            template_paths=VISIBLE_WRECK_TEMPLATES,
        )
        partial_cells = set(partial_wreck_cells or set())
        completed_candidates = set(visible_hits) - partial_cells
        if sidebar_progress is not None:
            completed_resolution = resolve_completed_ship_cells(
                completed_candidates,
                sidebar_progress.completed_lengths,
                grid_size=grid_size,
            )
            completed_visual_hits = set(completed_resolution.cells)
            logger.info(
                "level %s completed ship geometry: placements=%s unresolved=%s discarded=%s",
                level,
                [list(placement) for placement in completed_resolution.placements],
                list(completed_resolution.unresolved_lengths),
                sorted(completed_resolution.discarded_cells),
            )
        else:
            completed_visual_hits = completed_candidates

        initial_visual_hits = partial_cells | completed_visual_hits

        max_visible_hits = sum(submarines)
        if len(initial_visual_hits) > max_visible_hits:
            logger.warning(
                "level %s visual hit coordinates are suspicious: %s/%s; discarding partial wreck coordinates",
                level,
                len(initial_visual_hits),
                max_visible_hits,
            )
            initial_visual_hits = set(completed_visual_hits)

        for row, col in initial_visual_hits:
            hit_map[row][col] = 1

        if sidebar_progress is not None and partial_wreck_cells is not None:
            initial_visual_hit_count = calculate_visible_hit_count(
                sidebar_progress,
                partial_wreck_count=len(partial_wreck_cells),
                fallback_hit_count=len(visible_hits),
            )
            logger.info(
                "level %s exact visual hit count: completed_cells=%s partial_wrecks=%s total=%s",
                level,
                sidebar_progress.completed_cells,
                len(partial_wreck_cells),
                initial_visual_hit_count,
            )
        else:
            initial_visual_hit_count = len(initial_visual_hits)
            logger.warning(
                "level %s exact visual count unavailable; falling back to visible hit cells=%s",
                level,
                initial_visual_hit_count,
            )

        logger.info(
            "level %s visual hit coordinates: mapped=%s exact_count=%s unmapped=%s",
            level,
            len(initial_visual_hits),
            initial_visual_hit_count,
            max(0, int(initial_visual_hit_count or 0) - len(initial_visual_hits)),
        )

    if submarines is None:
        message = f"第 {level} 关缺少潜艇长度配置，回退逐格扫描"
        logger.warning(message)
        _scan_level_by_grid_order(
            level,
            hit_map,
            click_points,
            run_started_at=run_started_at,
        )
        completed = False
    else:
        completed = _run_red_scout_and_blue_strategy(
            level,
            hit_map,
            click_points,
            submarines,
            run_started_at=run_started_at,
            settings=settings or RedScoutSettings(),
            initial_hits=initial_visual_hits,
            initial_sidebar_progress=sidebar_progress,
            initial_visual_hit_count=initial_visual_hit_count,
            initial_completed_visual_hits=completed_visual_hits,
            initial_completed_lengths=(
                sidebar_progress.completed_lengths
                if sidebar_progress is not None and sidebar_progress.valid
                else ()
            ),
        )

    return grid_img, grid_quad, completed


def _scan_level_by_grid_order(
    level: int,
    hit_map: list[list[int]],
    click_points: list[tuple[int, int]],
    skip_cells: set[Cell] | None = None,
    run_started_at: float | None = None,
    result_callback: Callable[[Cell, ProbeResult], None] | None = None,
    probe_metadata_callback: Callable[[Cell, ProbeResult, Mapping[str, object]], None] | None = None,
    stop_when: Callable[[ProbeResult], bool] | None = None,
    prioritize_from_hits: bool = False,
) -> int:
    """按行优先顺序逐格探测，可跳过策略阶段已获得真实反馈的格子"""
    grid_size = get_level_grid_size(level)
    if skip_cells is None:
        skip_cells = set()
    targets = [
        (index, point, (index // grid_size, index % grid_size))
        for index, point in enumerate(click_points)
        if (index // grid_size, index % grid_size) not in skip_cells
    ]
    if prioritize_from_hits:
        targets = _prioritize_fallback_targets(targets, hit_map, grid_size)
    if not targets:
        logger.info("level %s grid scan has no remaining targets", level)
        return 0

    progress = SearchProgress(
        level=level,
        max_probes=len(targets),
        started_at=run_started_at if run_started_at is not None else monotonic(),
    )
    with fixed_progress_bar(
        total=len(targets),
        description=f"Level {level} grid scan",
        unit="cell",
    ) as bar:
        update_fixed_progress(
            bar,
            0,
            progress.grid_postfix(
                completed=0,
                total=len(targets),
                now=monotonic(),
            ),
        )
        scanned = 0
        for index, point, cell in targets:
            if cell in skip_cells:
                continue

            scanned += 1
            write_runtime_status(
                phase="grid_scan",
                level=level,
                current_cell=index,
            )
            probe_metadata: dict[str, object] = {}
            probe_result = _probe_cell(
                level,
                hit_map,
                cell,
                point,
                index,
                probe_metadata=probe_metadata,
            )
            if result_callback is not None:
                result_callback(cell, probe_result)
            if (
                probe_metadata_callback is not None
                and not _probe_result_completed_level(probe_result)
            ):
                probe_metadata_callback(cell, probe_result, probe_metadata)
            update_fixed_progress(
                bar,
                current=scanned,
                postfix=progress.grid_postfix(
                    completed=scanned,
                    total=len(targets),
                    now=monotonic(),
                ),
            )
            if _probe_result_completed_level(probe_result):
                logger.info(
                    "level %s grid scan stopped because a delayed victory banner completed the level",
                    level,
                )
                break
            if stop_when is not None and stop_when(probe_result):
                logger.info("level %s grid scan stopped early because completion condition was met", level)
                break
    return scanned


def _prioritize_fallback_targets(
    targets: list[tuple[int, tuple[int, int], Cell]],
    hit_map: list[list[int]],
    grid_size: int,
) -> list[tuple[int, tuple[int, int], Cell]]:
    hit_cells = [
        (row, col)
        for row, values in enumerate(hit_map)
        for col, value in enumerate(values)
        if value
    ]
    center = (grid_size - 1) / 2

    def score(target: tuple[int, tuple[int, int], Cell]) -> tuple[float, float, int]:
        index, _point, cell = target
        row, col = cell
        if hit_cells:
            nearest_hit = min(abs(row - hit_row) + abs(col - hit_col) for hit_row, hit_col in hit_cells)
        else:
            nearest_hit = 0
        center_distance = abs(row - center) + abs(col - center)
        return (nearest_hit, center_distance, index)

    return sorted(targets, key=score)


def _scan_level_by_strategy(
    level: int,
    hit_map: list[list[int]],
    click_points: list[tuple[int, int]],
    submarines: list[int],
    run_started_at: float | None = None,
    initial_hits: set[Cell] | None = None,
    initial_sidebar_progress: SidebarProgress | None = None,
    initial_visual_hit_count: int | None = None,
    initial_completed_visual_hits: set[Cell] | None = None,
    initial_completed_lengths: Sequence[int] | None = None,
    initial_scout_hits: set[Cell] | None = None,
    initial_scout_misses: set[Cell] | None = None,
) -> bool:
    """使用潜艇策略选择探测格；策略无法完成时回逢扫描剩余未探测格"""
    grid_size = get_level_grid_size(level)
    strategy = SubmarineStrategy(grid_size, submarines)
    saved_shots = load_saved_level_shots(level, grid_size)
    if saved_shots:
        logger.info(
            "level %s restored %s saved shots for profile %s",
            level,
            len(saved_shots),
            get_state_profile(),
        )
        for cell, hit in saved_shots.items():
            strategy.report_result(cell, hit)
            if hit:
                row, col = cell
                hit_map[row][col] = 1

    for cell in initial_hits or set():
        if cell not in strategy.shots:
            strategy.report_result(cell, True)
    initial_real_observation_count = len(strategy.shots)
    if initial_scout_hits or initial_scout_misses:
        strategy.report_scout_results(
            hits=initial_scout_hits or set(), misses=initial_scout_misses or set()
        )
    if initial_completed_lengths:
        located_initial, unlocated_initial = strategy.reconcile_completed_lengths(
            initial_completed_lengths,
            observed_completed_cells=initial_completed_visual_hits or set(),
        )
        if located_initial or unlocated_initial:
            logger.info(
                "level %s restored completed submarines from visual state: located=%s unlocated=%s",
                level,
                list(located_initial),
                list(unlocated_initial),
            )
    if strategy.shots:
        save_level_shots(level, grid_size, strategy.shots)
    initial_hit_cells = sum(1 for shot_hit in strategy.shots.values() if shot_hit)
    sidebar_progress = (
        initial_sidebar_progress
        if initial_sidebar_progress is not None and initial_sidebar_progress.valid
        else None
    )
    if initial_visual_hit_count is None:
        initial_display_hit_cells = merge_confirmed_hit_count(initial_hit_cells, sidebar_progress)
    else:
        initial_display_hit_cells = max(0, int(initial_visual_hit_count))
    initial_display_hit_cells = min(sum(submarines), initial_display_hit_cells)

    def accounted_completed_lengths() -> list[int]:
        getter = getattr(strategy, "get_accounted_completed_lengths", None)
        if callable(getter):
            return list(getter())
        return [ship.length for ship in strategy.get_confirmed_ships()]

    initial_confirmed_lengths = accounted_completed_lengths()
    mapped_visual_hits = len(initial_hits or set())
    unmapped_visual_hits = max(0, initial_display_hit_cells - mapped_visual_hits)
    max_attempts = grid_size * grid_size
    attempts = 0
    progress = SearchProgress(
        level=level,
        max_probes=max_attempts,
        total_ship_cells=sum(submarines),
        total_ships=len(submarines),
        started_at=run_started_at if run_started_at is not None else monotonic(),
    )
    write_runtime_status(
        phase="strategy_scan",
        level=level,
        current_cell="--",
        shots_done=0,
        total_cells=grid_size * grid_size,
        hits=initial_display_hit_cells,
        total_ship_cells=sum(submarines),
        confirmed_ships=len(initial_confirmed_lengths),
        total_ships=len(submarines),
        sidebar_completed_cells=sidebar_progress.completed_cells if sidebar_progress is not None else 0,
        sidebar_completed_lengths=(
            list(sidebar_progress.completed_lengths) if sidebar_progress is not None else []
        ),
        initial_visual_hits=initial_display_hit_cells,
        mapped_visual_hits=mapped_visual_hits,
        unmapped_visual_hits=unmapped_visual_hits,
        board_size=grid_size,
        board_states=build_runtime_board_states(strategy, grid_size),
        last_result="",
    )

    with fixed_progress_bar(
        total=sum(submarines),
        description=f"Level {level} strategy scan",
        unit="cell",
    ) as bar:
        logger.info(
            "level %s strategy enabled: grid=%s submarines=%s",
            level,
            grid_size,
            submarines,
        )
        update_fixed_progress(
            bar,
            initial_display_hit_cells,
            progress.strategy_postfix(
                attempts=0,
                confirmed_lengths=initial_confirmed_lengths,
                remaining_lengths=(
                    list(strategy.remaining.elements())
                    if hasattr(strategy.remaining, "elements")
                    else list(strategy.remaining)
                ),
                now=monotonic(),
            ),
        )

        while not strategy.done and attempts < max_attempts:
            cell = strategy.choose_next_cell()
            if cell is None:
                logger.warning("第 %s 关策略已无可选方格，提前结束", level)
                break

            row, col = cell
            index = row * grid_size + col
            write_runtime_status(
                phase="strategy_scan",
                level=level,
                current_cell=index,
            )
            probe_metadata: dict[str, object] = {}
            probe_result = _probe_cell(
                level,
                hit_map,
                cell,
                click_points[index],
                index,
                probe_metadata=probe_metadata,
            )
            level_completed = _probe_result_completed_level(probe_result)
            hit = _probe_result_is_hit(probe_result)
            if level_completed and not hit:
                write_runtime_status(
                    phase="level_complete",
                    level=level,
                    current_cell="--",
                    last_result=probe_result.value,
                )
                logger.info(
                    "level %s completed during recovery before cell %s; old-level probe was not recorded",
                    level,
                    index,
                )
                return True
            attempts += 1
            strategy.report_result(cell, hit)
            newly_completed_lengths = tuple(
                int(length)
                for length in probe_metadata.get("sidebar_newly_completed_lengths", ())
            )
            sidebar_completed_lengths = tuple(
                int(length)
                for length in probe_metadata.get("sidebar_completed_lengths", ())
            )
            if not sidebar_completed_lengths and newly_completed_lengths:
                sidebar_completed_lengths = tuple(accounted_completed_lengths()) + newly_completed_lengths
            if hit and sidebar_completed_lengths:
                located, unlocated = strategy.reconcile_completed_lengths(
                    sidebar_completed_lengths,
                    anchor=cell,
                    observed_completed_cells={cell} if hit else set(),
                )
                if located or unlocated:
                    logger.info(
                        "level %s reconciled completed submarines from sidebar: cell=%s located=%s unlocated=%s",
                        level,
                        cell,
                        list(located),
                        list(unlocated),
                    )
            save_level_shots(level, grid_size, strategy.shots)
            confirmed_lengths = accounted_completed_lengths()
            hit_cells = initial_hit_cells + (len(strategy.shots) - initial_real_observation_count)
            display_hit_cells = progressive_hit_count(
                initial_visual_hit_count=initial_display_hit_cells,
                initial_strategy_hit_count=initial_hit_cells,
                current_strategy_hit_count=hit_cells,
            )
            display_hit_cells = min(sum(submarines), display_hit_cells)
            write_runtime_status(
                phase="level_complete" if level_completed else "strategy_scan",
                level=level,
                current_cell="--" if level_completed else index,
                shots_done=len(strategy.shots),
                total_cells=grid_size * grid_size,
                hits=display_hit_cells,
                total_ship_cells=sum(submarines),
                confirmed_ships=len(confirmed_lengths),
                total_ships=len(submarines),
                sidebar_newly_completed_lengths=list(newly_completed_lengths),
                board_size=grid_size,
                board_states=build_runtime_board_states(strategy, grid_size),
                last_result=probe_result.value,
            )
            update_fixed_progress(
                bar,
                display_hit_cells,
                progress.strategy_postfix(
                    attempts=attempts,
                    confirmed_lengths=confirmed_lengths,
                    remaining_lengths=list(strategy.remaining.elements()),
                    now=monotonic(),
                ),
            )
            if level_completed:
                logger.info(
                    "level %s completed by the hit at cell %s; final hit recorded before progression",
                    level,
                    index,
                )
                return True

        if strategy.done:
            logger.info("level %s strategy confirmed all submarines, attempts=%s", level, attempts)
        else:
            logger.warning(
                "level %s strategy did not confirm all submarines; falling back to grid scan",
                level,
            )

    if not strategy.done:
        fallback_level_complete = False

        def report_fallback_result(cell: Cell, probe_result: ProbeResult) -> None:
            nonlocal fallback_level_complete
            level_completed = _probe_result_completed_level(probe_result)
            hit = _probe_result_is_hit(probe_result)
            if level_completed and not hit:
                fallback_level_complete = True
                write_runtime_status(
                    phase="level_complete",
                    level=level,
                    current_cell="--",
                    last_result=probe_result.value,
                )
                return
            strategy.report_result(cell, hit)
            save_level_shots(level, grid_size, strategy.shots)
            confirmed_lengths = accounted_completed_lengths()
            hit_cells = initial_hit_cells + (len(strategy.shots) - initial_real_observation_count)
            display_hit_cells = progressive_hit_count(
                initial_visual_hit_count=initial_display_hit_cells,
                initial_strategy_hit_count=initial_hit_cells,
                current_strategy_hit_count=hit_cells,
            )
            display_hit_cells = min(sum(submarines), display_hit_cells)
            write_runtime_status(
                phase="level_complete" if level_completed else "fallback_scan",
                level=level,
                current_cell="--" if level_completed else cell[0] * grid_size + cell[1],
                shots_done=len(strategy.shots),
                total_cells=grid_size * grid_size,
                hits=display_hit_cells,
                total_ship_cells=sum(submarines),
                confirmed_ships=len(confirmed_lengths),
                total_ships=len(submarines),
                board_size=grid_size,
                board_states=build_runtime_board_states(strategy, grid_size),
                last_result=probe_result.value,
            )
            if level_completed:
                fallback_level_complete = True
        fallback_skip_cells = set(strategy.shots) | set(strategy.blocked_cells) | set(initial_scout_misses or set())

        def apply_fallback_probe_metadata(
            cell: Cell,
            probe_result: ProbeResult,
            probe_metadata: Mapping[str, object],
        ) -> None:
            newly_completed = tuple(
                int(length)
                for length in probe_metadata.get("sidebar_newly_completed_lengths", ())
            )
            completed_lengths = tuple(
                int(length)
                for length in probe_metadata.get("sidebar_completed_lengths", ())
            )
            if not completed_lengths and newly_completed:
                completed_lengths = tuple(accounted_completed_lengths()) + newly_completed
            if _probe_result_is_hit(probe_result) and completed_lengths:
                located, unlocated = strategy.reconcile_completed_lengths(
                    completed_lengths,
                    anchor=cell,
                    observed_completed_cells={cell}
                    if _probe_result_is_hit(probe_result)
                    else set(),
                )
                if located or unlocated:
                    logger.info(
                        "level %s fallback reconciled completed submarines: cell=%s located=%s unlocated=%s",
                        level,
                        cell,
                        list(located),
                        list(unlocated),
                    )
                    write_runtime_status(
                        confirmed_ships=len(accounted_completed_lengths()),
                        sidebar_newly_completed_lengths=list(newly_completed),
                        board_size=grid_size,
                        board_states=build_runtime_board_states(strategy, grid_size),
                    )
            fallback_skip_cells.update(strategy.blocked_cells)
            save_level_shots(level, grid_size, strategy.shots)

        known_cells = set(strategy.shots)
        blocked_unshot = len(strategy.blocked_cells - known_cells)
        if blocked_unshot:
            logger.warning(
                "level %s strategy blocked %s unshot cells; entering conservative fallback scan",
                level,
                blocked_unshot,
            )
        scanned = _scan_level_by_grid_order(
            level,
            hit_map,
            click_points,
            skip_cells=fallback_skip_cells,
            run_started_at=run_started_at,
            result_callback=report_fallback_result,
            probe_metadata_callback=apply_fallback_probe_metadata,
            stop_when=lambda result: fallback_level_complete or strategy.done,
            prioritize_from_hits=True,
        )
        if fallback_level_complete or strategy.done:
            logger.info(
                "level %s fallback scan confirmed all submarines after %s extra probes",
                level,
                scanned,
            )
            return True
        logger.warning(
            "level %s is not confirmed complete after fallback scan; shots=%s blocked=%s scanned=%s",
            level,
            len(strategy.shots),
            len(strategy.blocked_cells),
            scanned,
        )
        return False

    return True


def _run_red_scout_and_blue_strategy(
    level: int, hit_map: list[list[int]], click_points: list[tuple[int, int]],
    submarines: list[int], initial_hits: set[Cell], settings: RedScoutSettings,
    run_started_at: float | None = None, **scan_kwargs: object,
) -> bool:
    if settings.mode is ProbeMode.BLUE_ONLY:
        return _scan_level_by_strategy(level, hit_map, click_points, submarines,
                                        run_started_at=run_started_at,
                                        initial_hits=initial_hits, **scan_kwargs)
    grid_size = get_level_grid_size(level)
    planner = RedScoutPlanner(grid_size)
    footprint = None
    covered: set[Cell] = set()
    hits: set[Cell] = set()
    misses: set[Cell] = set()
    attempts_completed = 0
    for _ in range(settings.count):
        center = planner.choose_center(footprint, known_cells=hits | misses,
                                       covered_cells=covered, cell_scores={})
        if center is None:
            break
        index = center[0] * grid_size + center[1]
        result = _execute_red_scout_transaction(
            level,
            center,
            click_points[index],
            index,
            grid_size,
            click_points,
        )
        attempts_completed += 1
        logger.info(
            "red scout %s/%s center=%s affected=%s hits=%s misses=%s unknown=%s valid=%s",
            attempts_completed,
            settings.count,
            center,
            sorted(result.affected_cells),
            sorted(result.hit_cells),
            sorted(result.miss_cells),
            sorted(result.unknown_cells),
            result.valid,
        )
        if result.level_completed:
            write_runtime_status(
                phase="level_complete",
                level=level,
                current_cell="--",
                red_scout_current=attempts_completed,
                red_scout_total=settings.count,
                board_size=grid_size,
                board_states=build_red_scout_board_states(
                    grid_size,
                    hits=hits,
                    misses=misses,
                    initial_hits=initial_hits,
                ),
                last_result="level_complete",
            )
            return True
        # A result can be unsuitable for learning a reusable footprint while still
        # containing reliable per-cell hit/miss evidence. Keep that evidence on the
        # cumulative board instead of discarding the whole scout attempt.
        if result.affected_cells:
            merge_red_scout_observations(hits, misses, result)
            covered.update(result.affected_cells)
        if result.valid:
            # The first valid footprint is the approved shape for every later attempt.
            if footprint is None:
                footprint = result.footprint
        write_runtime_status(
            phase="red_scout_capture",
            level=level,
            current_cell=index,
            red_scout_current=attempts_completed,
            red_scout_total=settings.count,
            board_size=grid_size,
            board_states=build_red_scout_board_states(
                grid_size,
                hits=hits,
                misses=misses,
                initial_hits=initial_hits,
            ),
            last_result="scout_valid" if result.valid else "scout_invalid",
        )
    write_runtime_status(phase="blue_attack", level=level,
                         red_scout_current=attempts_completed,
                         red_scout_total=settings.count,
                         current_cell="--",
                         board_size=grid_size,
                         board_states=build_red_scout_board_states(
                             grid_size,
                             hits=hits,
                             misses=misses,
                             initial_hits=initial_hits,
                         ))
    return _scan_level_by_strategy(level, hit_map, click_points, submarines,
                                   run_started_at=run_started_at, initial_hits=initial_hits,
                                   initial_scout_hits=hits, initial_scout_misses=misses,
                                   **scan_kwargs)


def _probe_cell(
    level: int,
    hit_map: list[list[int]],
    cell: Cell,
    point: tuple[int, int],
    index: int,
    probe_metadata: dict[str, object] | None = None,
) -> ProbeResult:
    """准备页面并执行一次完整探测；点击前异常只重试当前格"""
    max_preflight_retries = 3
    max_unknown_retries = 2
    for unknown_attempt in range(1, max_unknown_retries + 1):
        for attempt in range(1, max_preflight_retries + 1):
            try:
                if probe_metadata is not None:
                    probe_metadata.clear()
                    result = _execute_probe_transaction(
                        level,
                        hit_map,
                        cell,
                        point,
                        index,
                        probe_metadata=probe_metadata,
                    )
                else:
                    result = _execute_probe_transaction(level, hit_map, cell, point, index)
                if result != ProbeResult.UNKNOWN:
                    return result
                break
            except ProbeNotReadyError as exc:
                if attempt >= max_preflight_retries:
                    raise ProbeProtocolError(
                        f"cell {cell} was not ready before click after {max_preflight_retries} retries"
                    ) from exc
                logger.warning(
                    "cell %s was not ready before click; retrying same cell (%s/%s): %s",
                    cell,
                    attempt,
                    max_preflight_retries,
                    exc,
                )
                if enter_activity() is True:
                    logger.info(
                        "level %s completed while recovering before cell %s; stop probing old level",
                        level,
                        index,
                    )
                    return ProbeResult.LEVEL_COMPLETE

        if unknown_attempt < max_unknown_retries:
            logger.warning(
                "cell %s result was UNKNOWN; retrying same cell (%s/%s)",
                cell,
                unknown_attempt,
                max_unknown_retries,
            )
            continue
        raise ProbeProtocolError(
            f"cell {cell} result stayed UNKNOWN after {max_unknown_retries} retries"
        )

    raise AssertionError("探测重试循环意外结束")


def _execute_probe_transaction(
    level: int,
    hit_map: list[list[int]],
    cell: Cell,
    point: tuple[int, int],
    index: int,
    probe_metadata: dict[str, object] | None = None,
) -> ProbeResult:
    """按固定的 DROP/二次进入/REJECT/登录顺序执行单格探测事务。"""
    global _active_probe

    if probe_metadata is not None:
        probe_metadata.clear()

    if _active_probe is not None:
        raise ProbeProtocolError(
            f"上一轮探测尚未结束，禁止开始格子 {cell}: "
            f"cell={_active_probe.cell} phase={_active_probe.phase.name}"
        )

    if wait_until_occur(QUIT_ACTIVITY_TEMPLATE, timeout=6) is None:
        raise ProbeNotReadyError("当前不在活动详情界面")

    # Activity-entry recovery may return through an already-open fast path after
    # a committed hit. Enforce DROP here so no target click can bypass isolation.
    enable_weak_network(PROBE_DROP_SETTLE_SECONDS)

    transaction = ProbeTransaction(level=level, cell=cell, index=index)
    _active_probe = transaction
    x, y = point
    sample_dir: Path | None = None

    try:
        sample_dir = _create_probe_sample_dir(level, cell, index)
        _write_probe_status(
            sample_dir,
            "started",
            level=level,
            cell=list(cell),
            index=index,
            point=list(point),
            phase=transaction.phase.name,
        )
        before_img = adb.read_screenshot(sample_dir / "before.png")
        before_wreck_visible = (
            red_hit_marker_visible(before_img, (x, y))
            or visible_wreck_static_detected(before_img, (x, y))
        )
        _write_probe_status(sample_dir, "before_captured", phase=transaction.phase.name)

        # 点击命令一旦发出，就保守地认为客户端可能已经暂存验证请求。
        transaction.advance(ProbePhase.REQUEST_PENDING)
        _write_probe_status(sample_dir, "request_pending", phase=transaction.phase.name)
        adb.click(x, y)
        _exit_activity_after_probe_click(RUN_DEBUG_DIR / "debug_quit1.png")
        _write_probe_status(sample_dir, "activity_exited", phase=transaction.phase.name)
        if _reenter_activity_for_probe_result():
            transaction.advance(ProbePhase.RESULT_VISIBLE)
            _write_probe_status(
                sample_dir,
                "victory_detected",
                phase=transaction.phase.name,
            )
            transaction.advance(ProbePhase.RESULT_RECORDED)
            transaction.hit = True
            row, col = cell
            hit_map[row][col] = 1
            _write_probe_status(
                sample_dir,
                "result_recorded",
                phase=transaction.phase.name,
                decision=ProbeResult.HIT_AND_LEVEL_COMPLETE.value,
            )
            logger.info(
                "local victory appeared after blue probe at cell %s; recording the final hit "
                "and restoring network to commit it",
                cell,
            )
            write_runtime_status(
                phase="level_complete",
                level=level,
                current_cell="--",
                last_result=ProbeResult.HIT_AND_LEVEL_COMPLETE.value,
            )
            _commit_hit_request_and_prepare_next_probe(transaction)
            _write_probe_status(
                sample_dir,
                "complete",
                phase=transaction.phase.name,
                decision=ProbeResult.HIT_AND_LEVEL_COMPLETE.value,
            )
            append_recent_probe_result(
                level=level,
                index=index,
                result=ProbeResult.HIT_AND_LEVEL_COMPLETE,
                reason="local_victory_confirms_final_hit",
            )
            if probe_metadata is not None:
                probe_metadata["level_completed"] = True
            return ProbeResult.HIT_AND_LEVEL_COMPLETE
        _write_probe_status(sample_dir, "activity_reentered", phase=transaction.phase.name)
        submarines = get_configured_submarines(level, SUBMARINES) or []
        hit_results = []
        frame_records = []
        latest_sidebar_progress: SidebarProgress | None = None
        sidebar_newly_completed: tuple[int, ...] = ()
        victory_frame_detected = False
        for frame_index, frame_delay in enumerate(HIT_RESULT_FRAME_DELAYS, start=1):
            screenshot_path = sample_dir / f"after_{frame_index}.png"
            after_img = adb.delay(frame_delay).read_screenshot(screenshot_path)
            result = classify_diamond_hit(before_img, after_img, (x, y))
            victory_hit = find_victory_banner(after_img) is not None
            if victory_hit:
                if not victory_frame_detected:
                    logger.info(
                        "victory banner appeared while capturing blue probe cell %s; "
                        "treating the pending probe as the final hit",
                        cell,
                    )
                victory_frame_detected = True
                result.state = "hit"
                result.score = max(float(result.score), 1.0)
                result.confidence = max(float(result.confidence), 1.0)
            template_hit = apply_wreck_template_confirmation(after_img, (x, y), result)
            sidebar_hit = False
            frame_sidebar_progress: SidebarProgress | None = None
            frame_newly_completed: tuple[int, ...] = ()
            if submarines:
                sidebar_hit, frame_sidebar_progress, frame_newly_completed = (
                    apply_sidebar_completion_confirmation(
                        before_img,
                        after_img,
                        submarines,
                        result,
                    )
                )
                if frame_sidebar_progress is not None and frame_sidebar_progress.valid:
                    latest_sidebar_progress = frame_sidebar_progress
                if frame_newly_completed:
                    sidebar_newly_completed = frame_newly_completed
            new_wreck_hit = template_hit and not before_wreck_visible
            dynamic_hit_vetoed = enforce_positive_hit_evidence(
                result,
                wreck_hit=new_wreck_hit,
                sidebar_hit=sidebar_hit or victory_hit,
            )
            hit_results.append(result)
            frame_records.append(
                {
                    "frame": frame_index,
                    "delay": frame_delay,
                    "path": str(screenshot_path),
                    "template_hit": template_hit,
                    "new_wreck_hit": new_wreck_hit,
                    "dynamic_hit_vetoed": dynamic_hit_vetoed,
                    "sidebar_hit": sidebar_hit,
                    "victory_banner": victory_hit,
                    "sidebar_completed_lengths": (
                        list(frame_sidebar_progress.completed_lengths)
                        if frame_sidebar_progress is not None and frame_sidebar_progress.valid
                        else []
                    ),
                    "sidebar_newly_completed_lengths": list(frame_newly_completed),
                    "result": _hit_result_to_dict(result),
                }
            )
            _write_probe_status(
                sample_dir,
                "frame_captured",
                phase=transaction.phase.name,
                frame=frame_index,
                state=result.state,
                score=float(result.score),
            )
        transaction.advance(ProbePhase.RESULT_VISIBLE)
        _write_probe_status(sample_dir, "result_visible", phase=transaction.phase.name)

        hit_votes = sum(1 for result in hit_results if result.state == "hit")
        best_result = max(hit_results, key=lambda result: result.score)
        suspect_extra_checked = False
        if hit_votes < MIN_HIT_RESULT_VOTES and any(_is_suspect_hit_frame(result) for result in hit_results):
            suspect_extra_checked = True
            logger.info(
                "suspect hit cell=%s index=%s votes=%s/%s best_score=%.3f; collecting extra frames",
                cell,
                index,
                hit_votes,
                len(hit_results),
                best_result.score,
            )
            for extra_index, frame_delay in enumerate(
                SUSPECT_HIT_EXTRA_FRAME_DELAYS,
                start=len(hit_results) + 1,
            ):
                screenshot_path = sample_dir / f"after_{extra_index}.png"
                after_img = adb.delay(frame_delay).read_screenshot(screenshot_path)
                result = classify_diamond_hit(before_img, after_img, (x, y))
                victory_hit = find_victory_banner(after_img) is not None
                if victory_hit:
                    if not victory_frame_detected:
                        logger.info(
                            "victory banner appeared while capturing blue probe cell %s; "
                            "treating the pending probe as the final hit",
                            cell,
                        )
                    victory_frame_detected = True
                    result.state = "hit"
                    result.score = max(float(result.score), 1.0)
                    result.confidence = max(float(result.confidence), 1.0)
                template_hit = apply_wreck_template_confirmation(after_img, (x, y), result)
                sidebar_hit = False
                frame_sidebar_progress = None
                frame_newly_completed = ()
                if submarines:
                    sidebar_hit, frame_sidebar_progress, frame_newly_completed = (
                        apply_sidebar_completion_confirmation(
                            before_img,
                            after_img,
                            submarines,
                            result,
                        )
                    )
                    if frame_sidebar_progress is not None and frame_sidebar_progress.valid:
                        latest_sidebar_progress = frame_sidebar_progress
                    if frame_newly_completed:
                        sidebar_newly_completed = frame_newly_completed
                new_wreck_hit = template_hit and not before_wreck_visible
                dynamic_hit_vetoed = enforce_positive_hit_evidence(
                    result,
                    wreck_hit=new_wreck_hit,
                    sidebar_hit=sidebar_hit or victory_hit,
                )
                hit_results.append(result)
                frame_records.append(
                    {
                        "frame": extra_index,
                        "delay": frame_delay,
                        "path": str(screenshot_path),
                        "template_hit": template_hit,
                        "new_wreck_hit": new_wreck_hit,
                        "dynamic_hit_vetoed": dynamic_hit_vetoed,
                        "sidebar_hit": sidebar_hit,
                        "victory_banner": victory_hit,
                        "sidebar_completed_lengths": (
                            list(frame_sidebar_progress.completed_lengths)
                            if frame_sidebar_progress is not None and frame_sidebar_progress.valid
                            else []
                        ),
                        "sidebar_newly_completed_lengths": list(frame_newly_completed),
                        "result": _hit_result_to_dict(result),
                    }
                )
                _write_probe_status(
                    sample_dir,
                    "extra_frame_captured",
                    phase=transaction.phase.name,
                    frame=extra_index,
                    state=result.state,
                    score=float(result.score),
                )
            hit_votes = sum(1 for result in hit_results if result.state == "hit")
            best_result = max(hit_results, key=lambda result: result.score)
        if latest_sidebar_progress is not None:
            write_runtime_status(
                sidebar_completed_cells=latest_sidebar_progress.completed_cells,
                sidebar_completed_lengths=list(latest_sidebar_progress.completed_lengths),
            )
        if sidebar_newly_completed:
            logger.info(
                "sidebar confirms newly completed submarines at cell %s: lengths=%s completed_cells=%s",
                cell,
                list(sidebar_newly_completed),
                latest_sidebar_progress.completed_cells if latest_sidebar_progress is not None else "--",
            )
        first_result = hit_results[0]
        if victory_frame_detected:
            hit, decision_reason = True, "victory_banner_frame"
        else:
            hit, decision_reason = decide_hit_from_frames(hit_results)
        logger.info(
            "hit check cell=%s index=%s votes=%s/%s states=%s scores=%s changed=%s "
            "best_gray=%.3f best_excess=%.3f best_component=%.3f best_s_drop=%.1f best_edge=%.3f "
            "center=%s refined=%s decision=%s",
            cell,
            index,
            hit_votes,
            len(hit_results),
            "/".join(result.state for result in hit_results),
            "/".join(f"{result.score:.3f}" for result in hit_results),
            "/".join(f"{result.changed_ratio:.3f}" for result in hit_results),
            best_result.center_gray_ratio,
            best_result.gray_excess,
            best_result.component_ratio,
            best_result.s_drop,
            best_result.edge_density,
            first_result.rough_center,
            best_result.refined_center,
            decision_reason,
        )
        _save_probe_result_json(
            sample_dir,
            level=level,
            cell=cell,
            index=index,
            point=point,
            hit=hit,
            hit_votes=hit_votes,
            frames=frame_records,
            suspect_extra_checked=suspect_extra_checked,
            decision_reason=decision_reason,
        )
        transaction.hit = hit
        transaction.advance(ProbePhase.RESULT_RECORDED)
        _write_probe_status(
            sample_dir,
            "result_recorded",
            phase=transaction.phase.name,
            decision="hit" if hit else "miss",
            hit_votes=hit_votes,
        )

        if hit:
            row, col = cell
            hit_map[row][col] = 1
            logger.info("level %s cell %s result: hit", level, index)
            level_complete = _commit_hit_request_and_prepare_next_probe(transaction)
            probe_result = (
                ProbeResult.HIT_AND_LEVEL_COMPLETE
                if level_complete or victory_frame_detected
                else ProbeResult.HIT
            )
        elif suspect_extra_checked or hit_votes == 1 or any(_is_near_hit_frame(result) for result in hit_results):
            logger.warning(
                "level %s cell %s result: unknown (%s); discarding request and retrying",
                level,
                index,
                decision_reason,
            )
            level_complete = _discard_pending_request_and_prepare_next_probe(transaction)
            probe_result = (
                ProbeResult.LEVEL_COMPLETE
                if level_complete
                else ProbeResult.UNKNOWN
            )
        else:
            logger.info("level %s cell %s result: miss", level, index)
            level_complete = _discard_pending_request_and_prepare_next_probe(transaction)
            probe_result = (
                ProbeResult.LEVEL_COMPLETE
                if level_complete
                else ProbeResult.MISS
            )

        _write_probe_status(
            sample_dir,
            "complete",
            phase=transaction.phase.name,
            decision=probe_result.value,
        )
        append_recent_probe_result(
            level=level,
            index=index,
            result=probe_result,
            reason=decision_reason,
        )
        if probe_metadata is not None:
            probe_metadata.update(
                sidebar_newly_completed_lengths=tuple(sidebar_newly_completed),
                sidebar_completed_lengths=(
                    tuple(latest_sidebar_progress.completed_lengths)
                    if latest_sidebar_progress is not None and latest_sidebar_progress.valid
                    else ()
                ),
                sidebar_completed_cells=(
                    latest_sidebar_progress.completed_cells
                    if latest_sidebar_progress is not None and latest_sidebar_progress.valid
                    else 0
                ),
            )
        return probe_result
    except Exception as exc:
        if sample_dir is not None:
            _write_probe_status(
                sample_dir,
                "interrupted",
                phase=transaction.phase.name,
                error=repr(exc),
            )
        raise
    finally:
        if transaction.phase in {ProbePhase.PREPARING, ProbePhase.COMPLETE}:
            _active_probe = None
        elif transaction.request_may_be_pending:
            logger.critical(
                "cell %s probe interrupted at %s; pending request may remain; keep DROP weak network",

                transaction.cell,
                transaction.phase.name,
            )


def _commit_hit_request_and_prepare_next_probe(
    transaction: ProbeTransaction,
) -> bool:
    """Restore network immediately on hit so the pending request is submitted."""
    transaction.advance(ProbePhase.REQUEST_COMMITTED)
    logger.info("hit detected; restoring network immediately to submit the pending request")
    transaction.advance(ProbePhase.LOGIN_RECOVERING)
    level_complete = restart_process() is True
    transaction.advance(ProbePhase.COMPLETE)
    return level_complete


def _discard_pending_request_and_prepare_next_probe(
    transaction: ProbeTransaction,
) -> bool:
    """Force-stop the game while offline so a pending request cannot be retried."""
    adb.enable_reject_network(GAME_PACKAGE_NAME)
    write_runtime_status(network="断网中")
    adb.delay(MISS_REJECT_SETTLE_SECONDS)
    logger.info("discarding pending probe request; force-stopping game before restoring network")
    adb.close_app(GAME_PACKAGE_NAME)
    if not adb.wait_until_app_stopped(
        GAME_PACKAGE_NAME,
        timeout=APP_STOP_TIMEOUT_SECONDS,
        poll_interval=APP_STOP_POLL_SECONDS,
    ):
        raise ProbeProtocolError(
            "游戏进程未完全退出；为避免未命中请求补发，保留断网并中止探测"
        )
    adb.delay(POST_FORCE_STOP_GUARD_SECONDS)

    transaction.advance(ProbePhase.REQUEST_DISCARDED)
    transaction.red_request_discarded = True
    update_pending_probe(
        phase=ProbePhase.REQUEST_DISCARDED.name,
        request_discarded=True,
    )
    transaction.advance(ProbePhase.LOGIN_RECOVERING)

    level_complete = restart_process(reopen_game=True, app_already_closed=True) is True
    transaction.advance(ProbePhase.COMPLETE)
    return level_complete


def restart_process(reopen_game: bool = False, app_already_closed: bool = False) -> bool:
    """在请求确认丢弃后恢复网络登录，并进入下一轮探测页靃69"""
    if reopen_game:
        logger.info("pending probe request discarded; reopening game before next probe")
        if not app_already_closed:
            adb.close_app(GAME_PACKAGE_NAME)
        adb.disable_reject_network(GAME_PACKAGE_NAME)
        disable_weak_network()
        adb.delay(REOPEN_GAME_SETTLE_SECONDS).open_app(GAME_PACKAGE_NAME)
        login_img = wait_until_occur(LOGIN_TEMPLATE, timeout=LOGIN_WAIT_AFTER_REOPEN_SECONDS)
        if login_img is not None:
            adb.click(*login_img.center)
        else:
            logger.warning("reopened game but login button was not found; continuing to activity entry")
        return enter_activity(
            activity_button_timeout=POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        ) is True

    disable_weak_network()
    level_complete = handle_victory_prompt(timeout=VICTORY_WAIT_AFTER_HIT_SECONDS)
    recovered_level_complete = enter_activity() is True
    return level_complete or recovered_level_complete


def find_victory_banner(screenshot: np.ndarray) -> MatchResult | None:
    """Detect the victory banner in a screenshot."""
    if not isinstance(screenshot, np.ndarray):
        return None

    victory = find_template(
        screenshot,
        VICTORY_BANNER_TEMPLATE,
        threshold=VICTORY_BANNER_THRESHOLD,
    )
    if victory is not None:
        return victory

    return find_template_multi_scale(
        screenshot,
        VICTORY_BANNER_TEMPLATE,
        scales=VICTORY_TEMPLATE_SCALES,
        threshold=VICTORY_BANNER_THRESHOLD,
    )


def handle_victory_prompt(
    timeout: float = 4.0,
    screenshot: np.ndarray | None = None,
    *,
    restore_network: bool = True,
) -> bool:
    """Skip the victory banner after a committed hit, if it appears."""
    victory = find_victory_banner(screenshot) if screenshot is not None else None
    if victory is None:
        if timeout > 0:
            logger.info("waiting up to %.1f seconds for victory banner", timeout)
        victory = wait_until_victory_banner(timeout=timeout)
    if victory is None:
        return False

    if restore_network:
        logger.info("victory banner detected; restoring network and tapping screen to continue")
        adb.disable_reject_network(GAME_PACKAGE_NAME)
        disable_weak_network()
    else:
        logger.info("victory banner detected while probe request is pending; keeping network isolated")
    adb.click(*SCREEN_CONTINUE_POINT)
    adb.delay(VICTORY_SKIP_SETTLE_SECONDS)
    return True


def handle_connection_interrupted_prompt(timeout: float = 20.0) -> bool:
    """Detect the connection-interrupted dialog, reconnect, and click retry."""
    dialog = wait_until_connection_interrupted_dialog(timeout=min(4.0, float(timeout)))
    if dialog is None:
        return False

    logger.info("connection-interrupted dialog detected; reconnecting and clicking retry")
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    retry = wait_until_retry_button(timeout=max(0.0, float(timeout) - 4.0))
    if retry is None:
        raise ProbeProtocolError("connection-interrupted dialog found, but retry button was not found")

    adb.delay(0.8).click(*retry.center)
    return True


def wait_until_victory_banner(timeout: float = 4.0) -> MatchResult | None:
    """Wait briefly for the victory banner shown after the final submarine is hit."""
    deadline = monotonic() + max(0.0, float(timeout))
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        victory = find_victory_banner(screenshot)
        if victory is not None:
            return victory
        sleep(0.3)
    return None


def wait_until_connection_interrupted_dialog(timeout: float = 20.0) -> MatchResult | None:
    """Wait for the larger connection-interrupted dialog."""
    exact_wait = min(3.0, max(0.0, float(timeout)))
    dialog = wait_until_occur(CONNECTION_INTERRUPTED_TEMPLATE, timeout=exact_wait)
    if dialog is not None:
        return dialog

    deadline = monotonic() + max(0.0, float(timeout) - exact_wait)
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        dialog = find_template_multi_scale(
            screenshot,
            CONNECTION_INTERRUPTED_TEMPLATE,
            scales=CONNECTION_TEMPLATE_SCALES,
            threshold=CONNECTION_DIALOG_THRESHOLD,
        )
        if dialog is not None:
            return dialog
        sleep(FAST_POLL_INTERVAL_SECONDS)
    return None


def wait_until_retry_button(timeout: float = 20.0) -> MatchResult | None:
    """Wait for the current connection dialog retry button or the legacy retry button."""
    exact_wait = min(3.0, max(0.0, float(timeout)))
    retry = wait_until_occur(CONNECTION_RETRY_TEMPLATE, timeout=exact_wait)
    if retry is not None:
        return retry

    legacy_wait = min(5.0, max(0.0, float(timeout) - exact_wait))
    retry = wait_until_occur(RETRY_TEMPLATE, timeout=legacy_wait)
    if retry is not None:
        return retry

    deadline = monotonic() + max(0.0, float(timeout) - exact_wait - legacy_wait)
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        retry = find_template_multi_scale(
            screenshot,
            CONNECTION_RETRY_TEMPLATE,
            scales=RETRY_TEMPLATE_SCALES,
            threshold=CONNECTION_RETRY_THRESHOLD,
        )
        if retry is None:
            retry = find_template_multi_scale(
                screenshot,
                RETRY_TEMPLATE,
                scales=RETRY_TEMPLATE_SCALES,
                threshold=RETRY_TEMPLATE_LOOSE_THRESHOLD,
            )
        if retry is not None:
            return retry
        sleep(FAST_POLL_INTERVAL_SECONDS)
    return None


def wait_until_retry_prompt(timeout: float = 20.0) -> MatchResult | None:
    """Wait for the retry prompt using the consolidated retry-button helper."""
    retry = wait_until_retry_button(timeout=timeout)
    if retry is None:
        logger.warning("retry button wait timed out (%s seconds)", timeout)
    return retry


def wait_until_occur(
    template_path: str | Path,
    timeout: float = 30.0,
) -> MatchResult | None:
    """等待直到指定模板出现，返回匹配结果或 None（超时）"""
    logger.info("正在等待模板 '%s' 出现，超时时间 %s 秒", template_path, timeout)
    start_time = monotonic()
    while monotonic() - start_time < timeout:
        screenshot = adb.read_screenshot()
        match_result = find_template(screenshot, template_path)
        if match_result is not None:
            return match_result
        sleep(FAST_POLL_INTERVAL_SECONDS)
    logger.warning("等待模板 '%s' 超时 (%s 秒)", template_path, timeout)
    return None


def click_template(
    template_path: str | Path,
    screenshot_path: str | Path | None = None,
    threshold: float = 0.85,
) -> bool:
    """查找模板并点击中心点，找不到时返回 False。"""
    img = adb.read_screenshot(screenshot_path)
    match_result = find_template(img, template_path, threshold=threshold)
    if match_result is None:
        return False

    adb.delay(0.5).click(*match_result.center)
    return True


def resolve_current_level(
    screenshot: np.ndarray,
    fallback_level: int = DEFAULT_LEVEL,
    fallback_is_manual: bool = False,
) -> int:
    """Detect the current level from the activity page, or use the fallback."""
    if not AUTO_DETECT_LEVEL:
        logger.info("level auto detection disabled; using fallback level %s", fallback_level)
        return fallback_level

    title_result = recognize_level_title(
        screenshot,
        reference_dir=LEVEL_REFERENCE_DIR,
    )
    if title_result is not None:
        logger.info(
            "level title detection: best=%s score=%.3f second=%s score=%.3f confident=%s",
            title_result.level,
            title_result.score,
            title_result.second_level,
            title_result.second_score,
            title_result.confident,
        )
        if title_result.confident and title_result.level in LEVEL_GRID_SIZES:
            return title_result.level
        if title_result.confident:
            logger.warning(
                "level title detection returned unsupported level %s; falling back to image detection",
                title_result.level,
            )
    else:
        logger.info("level title detection: title number not readable in current screenshot")

    result = recognize_level_from_screenshot(
        screenshot,
        reference_dir=LEVEL_REFERENCE_DIR,
        candidate_levels=LEVEL_GRID_SIZES.keys(),
    )
    if result is None:
        logger.warning("level auto detection found no reference images; using fallback level %s", fallback_level)
        return fallback_level

    logger.info(
        "level auto detection: best=%s score=%.3f second=%s score=%.3f confident=%s",
        result.level,
        result.score,
        result.second_level,
        result.second_score,
        result.confident,
    )
    if result.confident:
        return result.level

    if REQUIRE_CONFIDENT_LEVEL_DETECTION and not fallback_is_manual:
        raise RuntimeError(
            "level auto detection is uncertain; stop before probing to avoid wasting bombs "
            f"(detected={result.level} score={result.score:.3f}, "
            f"second={result.second_level} score={result.second_score:.3f})"
        )

    logger.warning(
        "level auto detection is uncertain; using fallback level %s instead of detected level %s",
        fallback_level,
        result.level,
    )
    return fallback_level


def resolve_current_level_from_device(
    fallback_level: int = DEFAULT_LEVEL,
    fallback_is_manual: bool = False,
    attempts: int = 8,
) -> int:
    """Take several screenshots until the level title is stable enough to read."""
    if attempts <= 0:
        raise ValueError(f"attempts must be positive: {attempts}")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        adb.delay(1.0)
        screenshot = adb.read_screenshot()
        if handle_victory_prompt(
            timeout=VICTORY_WAIT_BEFORE_LEVEL_SECONDS,
            screenshot=screenshot,
        ):
            logger.info(
                "level detection attempt %s/%s skipped a victory banner",
                attempt,
                attempts,
            )
            continue
        try:
            return resolve_current_level(
                screenshot,
                fallback_level=fallback_level,
                fallback_is_manual=fallback_is_manual,
            )
        except RuntimeError as exc:
            last_error = exc
            logger.warning(
                "level detection attempt %s/%s was uncertain: %s",
                attempt,
                attempts,
                exc,
            )

    if last_error is not None:
        raise last_error
    return fallback_level


def resolve_next_level_with_retries(
    current_level: int,
    fallback_level: int,
) -> int | None:
    for attempt in range(1, LEVEL_ADVANCE_RETRIES + 1):
        logger.info(
            "checking next level after level %s (%s/%s)",
            current_level,
            attempt,
            LEVEL_ADVANCE_RETRIES,
        )
        write_runtime_status(
            phase="advance_level",
            level=current_level,
            current_cell="--",
            board_size=0,
            board_states=[],
            last_result=f"advance_attempt_{attempt}",
        )
        try:
            next_level = resolve_current_level_from_device(
                fallback_level=fallback_level,
                fallback_is_manual=False,
            )
        except Exception as exc:
            logger.warning(
                "failed to resolve next level after level %s on attempt %s/%s: %s",
                current_level,
                attempt,
                LEVEL_ADVANCE_RETRIES,
                exc,
            )
            next_level = current_level

        if next_level > current_level:
            return next_level

        logger.warning(
            "next level did not advance beyond %s on attempt %s/%s (detected=%s)",
            current_level,
            attempt,
            LEVEL_ADVANCE_RETRIES,
            next_level,
        )
        adb.click(*SCREEN_CONTINUE_POINT)
        adb.delay(1.5)
        try:
            enter_activity()
        except Exception as exc:
            logger.warning("retrying next-level transition failed to enter activity: %s", exc)

    return None


def main(level: int | None = None) -> Path | None:
    """执行指定关卡的辑探测并输出命中图"""
    run_started_at = monotonic()
    fallback_is_manual = level is not None
    fallback_level = DEFAULT_LEVEL if level is None else int(level)
    last_out_path: Path | None = None
    settings = load_red_scout_settings()
    try:
        write_runtime_status(
            running=True,
            phase="starting",
            level=fallback_level,
            current_cell="--",
            shots_done=0,
            total_cells=0,
            hits=0,
            total_ship_cells=0,
            last_result="",
            profile=get_state_profile() or "",
            probe_mode=settings.mode.value,
            red_scout_total=settings.count,
        )
        disable_weak_network()

        screenshot = adb.read_screenshot()
        if handle_victory_prompt(timeout=0.0, screenshot=screenshot):
            screenshot = adb.delay(1.0).read_screenshot()

        already_in_activity_detail = find_template(screenshot, QUIT_ACTIVITY_TEMPLATE) is not None
        if already_in_activity_detail:
            logger.info("current screen is already the activity detail; skipping activity entry")
        elif find_template(screenshot, ACTIVITY_BUTTON_TEMPLATE) is None:
            logger.error("当前不在海岛主界面，无法启动脚本")
            return None

        if not already_in_activity_detail:
            enter_activity()
        current_level = resolve_current_level_from_device(
            fallback_level=fallback_level,
            fallback_is_manual=fallback_is_manual,
        )
        while current_level <= MAX_LEVEL:
            grid_size = get_level_grid_size(current_level)
            reset_runtime_level_status(current_level)
            hit_map = [[0] * grid_size for _ in range(grid_size)]
            base_img, quad, level_completed = handle_game_level(
                current_level,
                hit_map,
                run_started_at=run_started_at,
                settings=settings,
            )
            out_path = OUTPUT_DIR / f"hit_map_level_{current_level}.png"
            save_hit_map_image(base_img, quad, hit_map, out_path)
            logger.info("hit map: %s", hit_map)
            logger.info("hit map image saved: %s", out_path)
            last_out_path = out_path

            if not level_completed:
                logger.warning(
                    "level %s stopped because submarines were not fully confirmed; not advancing to next level",
                    current_level,
                )
                break

            if current_level >= MAX_LEVEL:
                logger.info("reached max level %s; stopping", MAX_LEVEL)
                break

            next_fallback_level = min(current_level + 1, MAX_LEVEL)
            logger.info(
                "level %s finished; trying to continue to next level (fallback=%s)",
                current_level,
                next_fallback_level,
            )
            next_level = resolve_next_level_with_retries(
                current_level=current_level,
                fallback_level=next_fallback_level,
            )
            if next_level is None:
                logger.warning(
                    "next level detection did not advance beyond %s after retries; stopping progression",
                    current_level,
                )
                break

            current_level = next_level

        return last_out_path
    finally:
        write_runtime_status(running=False, phase="stopped")
        logger.info("脚本总运行时间：%s", format_elapsed(monotonic() - run_started_at))


if __name__ == "__main__":
    main_pid: int | None = None
    register_exit_cleanup()
    try:
        main_pid = acquire_main_lock()
        write_runtime_status(pid=main_pid)
        logger.info("main.py 启动，PID=%s", main_pid)
        adb.ensure_root_shell()
        if recover_interrupted_probe_at_startup():
            raise RedScoutSafetyError(
                "检测到上次中断的探测请求，已在断网状态下安全关闭游戏；请重新启动程序"
            )
        cleanup_reject_network("main startup")
        main()
    except AlreadyRunningError as exc:
        logger.error("%s", exc)
        raise SystemExit(2)
    except RedScoutSafetyError as exc:
        logger.critical("%s", exc)
        raise SystemExit(3)
    finally:
        cleanup_weak_network("main finished")
        cleanup_reject_network("main finished")
        if main_pid is not None:
            remove_pid(pid=main_pid)

