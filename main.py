import atexit
import json
import os
import signal
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Event
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
    MAX_PROBE_SAMPLE_DIRS,
    MAX_RED_SCOUT_SAMPLE_DIRS,
    MAX_SCREENSHOT_STORAGE_BYTES,
    OUTPUT_DIR,
    REQUIRE_CONFIDENT_LEVEL_DETECTION,
    SCREENSHOT_DIR,
    SUBMARINES,
    TEMPLATE_DIR,
    USE_SAVED_POINTS,
)
from save_points.points import read_saved_points, read_saved_quad
from utils import AdbController, MatchResult, find_template, get_logger
from utils.adaptive_frames import (
    ADAPTIVE_HIT_MIN_FRAMES,
    can_stop_after_stable_hit_frames,
)
from utils.diamond_centers import detect_diamond_centers
from utils.diamond_hit import classify_diamond_hit
from utils.frame_stability import (
    analyze_stable_hit,
    register_translation,
    stable_hit_is_suspect,
)
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
from utils.runtime_lock import AlreadyRunningError, acquire_main_lock, release_main_lock
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
from utils.submarine_strategy import (
    Cell,
    Placement,
    SubmarineStrategy,
    get_configured_submarines,
)
from utils.wreck_detection import (
    COMPLETED_SHIP_BODY_MIN_SCORE,
    completed_ship_body_score,
    detect_completed_submarine_candidate_cells,
    detect_red_submarine_marker_cells,
    VISIBLE_WRECK_TEMPLATES,
    detect_visible_wreck_cells,
    grid_cell_polygon,
    red_hit_marker_visible,
    red_submarine_marker_visible,
    visible_wreck_static_detected,
)
from utils.red_scout import (
    AmmoFingerprint, ProbeMode, RedFootprint as RedFootprint, RedScoutAnalyzer, RedScoutResult,
    RedScoutSettings, RedScoutPlanner, ammo_fingerprint_matches, build_ammo_fingerprint,
    load_red_scout_settings, locate_red_bomb_button, red_bomb_selected,
)

logger = get_logger(__name__)
adb = AdbController()

ACTIVITY_BUTTON_TEMPLATE = TEMPLATE_DIR / "activity_button.png"
LOGIN_TEMPLATE = TEMPLATE_DIR / "login.png"
QUIT_ACTIVITY_TEMPLATE = TEMPLATE_DIR / "quit_activity.png"
ACTIVITY_QUIT_ROI_REFERENCE_SIZE = (100, 100)
SCREEN_REFERENCE_SIZE = (1280, 720)
RETRY_TEMPLATE = TEMPLATE_DIR / "retry.png"
CONNECTION_INTERRUPTED_PANEL_TEMPLATE = TEMPLATE_DIR / "connection_interrupted_panel.png"
CONNECTION_RETRY_TEMPLATE = TEMPLATE_DIR / "connection_retry.png"
VICTORY_BANNER_TEMPLATE = TEMPLATE_DIR / "victory_banner.png"
BLUE_BOMB_ZERO_TEMPLATE = TEMPLATE_DIR / "blue_bomb_zero.png"
RETRY_TEMPLATE_SCALES = (0.85, 0.95, 1.0, 1.05, 1.15)
RETRY_TEMPLATE_LOOSE_THRESHOLD = 0.72
CONNECTION_PROMPT_SCALES = (1.0,)
CONNECTION_DIALOG_THRESHOLD = 0.95
CONNECTION_RETRY_THRESHOLD = 0.95
CONNECTION_DIALOG_SEARCH_REGION = (0.18, 0.20, 0.82, 0.80)
CONNECTION_RETRY_SEARCH_REGION = (0.18, 0.45, 0.50, 0.78)
CONNECTION_RETRY_RELATIVE_CENTER = (0.10, 0.81)
VICTORY_TEMPLATE_SCALES = (0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.65, 1.8)
VICTORY_BANNER_THRESHOLD = 0.80
VICTORY_SEARCH_REGION = (0.18, 0.06, 0.82, 0.70)
VICTORY_WAIT_AFTER_HIT_SECONDS = 10.0
VICTORY_WAIT_AFTER_CONFIRMED_INCOMPLETE_SECONDS = 2.0
VICTORY_WAIT_BEFORE_LEVEL_SECONDS = 3.0
VICTORY_SKIP_SETTLE_SECONDS = 2.0
LEVEL_ADVANCE_RETRIES = 3
HIT_RESULT_FRAME_DELAYS = (1.0, 0.35, 0.45, 0.55)
# Red scout results only need three frames: misses require three consistent
# votes, while hits still require at least two votes in the analyzer.
RED_SCOUT_RESULT_FRAME_DELAYS = (0.55, 0.15, 0.20)
# A red result is analysed only when at least two result frames survive the
# transition-frame filter.  Keeping this threshold aligned with the analyzer's
# vote requirement prevents a single potentially stale frame from becoming a
# false positive while retaining the existing three-frame capture schedule.
RED_SCOUT_MIN_ANALYSIS_FRAMES = 2
# Online blue shots already have a confirmed target from the red scout. Keep
# the same number of evidence frames, but sample the result sooner so the next
# confirmed target is not delayed by the generic offline-probe timing.
ONLINE_SCOUT_HIT_FRAME_DELAYS = (0.55, 0.20, 0.28, 0.40)
ONLINE_SCOUT_STABLE_HIT_MIN_FRAMES = 3
# Once a red-scout batch has established and verified blue mode, subsequent
# confirmed targets can use a shorter two-frame evidence window. Unstable
# evidence automatically falls back to the normal schedule.
ONLINE_SCOUT_REUSED_HIT_FRAME_DELAYS = (0.40, 0.18)
ONLINE_SCOUT_REUSED_STABLE_HIT_MIN_FRAMES = 2
# A red marker can prove a completed hull even when a fast batch tap is
# ignored. Retry the still-unopened hull cell with the verified single-target
# path before allowing the complete placement to become immutable.
COMPLETED_SHIP_PENDING_BLUE_RETRIES = 2
# A red-scout result can contain more than one confirmed hit.  Those cells are
# already known targets, so selecting the blue projectile and waiting for a
# full result window for every cell only adds latency.  Batch mode keeps one
# short input window and performs the evidence pass after all taps.
ONLINE_SCOUT_BATCH_ENABLED = True
ONLINE_SCOUT_BATCH_CLICK_INTERVAL_SECONDS = 0.25
ONLINE_SCOUT_BATCH_FRAME_DELAYS = (0.55, 0.22, 0.32)
ADAPTIVE_HIT_FRAMES_ENABLED = True
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
ACTIVITY_REENTRY_INITIAL_DELAY_SECONDS = 0.05
ACTIVITY_REENTRY_POLL_INTERVAL_SECONDS = 0.08
PROBE_DROP_SETTLE_SECONDS = 0.2
MISS_CONNECTION_DIALOG_WAIT_SECONDS = 15.0
MISS_RETRY_BUTTON_WAIT_SECONDS = 4.0
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
ONLINE_SCOUT_NETWORK_SETTLE_SECONDS = 0.3
ONLINE_SCOUT_BLUE_SELECT_SETTLE_SECONDS = 0.25
ONLINE_SCOUT_BLUE_SELECT_FAST_SETTLE_SECONDS = 0.1
ONLINE_SCOUT_BLUE_SELECT_RETRY_SECONDS = 0.15
STATUS_REPLACE_RETRIES = 5
STATUS_REPLACE_RETRY_SECONDS = 0.05

ACTIVITY_DETAIL_POINT = (1205, 644)
ACTIVITY_LIST_SWIPE = (1000, 660, 1000, 180)
# Keep the victory-screen continue tap outside the diamond board.  The former
# screen-center point could become a real cell after the next level loaded.
SCREEN_CONTINUE_POINT = (40, 120)
BLUE_BOMB_POINT = (1120, 660)
BLUE_BOMB_ZERO_SEARCH_REGION = (1115, 650, 1175, 710)
BLUE_BOMB_ZERO_THRESHOLD = 0.92
RUN_DEBUG_DIR = SCREENSHOT_DIR / "run_debug"
PROBE_SAMPLE_DIR = SCREENSHOT_DIR / "probes"
RED_SCOUT_SAMPLE_DIR = SCREENSHOT_DIR.parent / "red_scout_samples"
RUNTIME_DIR = SCREENSHOT_DIR.parent / "runtime"
STATUS_FILE = RUNTIME_DIR / "status.json"
LEVEL_STATE_FILE = RUNTIME_DIR / "level_state.json"

_weak_network_cleanup_done = False
_active_probe: "ProbeTransaction | None" = None
_runtime_status: dict[str, object] = {}
_network_fail_closed_reason: str | None = None


class RedScoutSafetyError(RuntimeError):
    pass


class BlueAmmoDepletedError(RuntimeError):
    """Raised before a blue-bomb action when the visible count is zero."""


class DiscardRecoveryError(ProbeProtocolError):
    """The request is isolated and discarded, but the in-client retry flow stalled."""


class ProbeResult(str, Enum):
    MISS = "miss"
    HIT = "hit"
    HIT_AND_LEVEL_COMPLETE = "hit_and_level_complete"
    LEVEL_COMPLETE = "level_complete"
    UNKNOWN = "unknown"


@dataclass
class OnlineScoutBatchResult:
    """Results collected after a group of red-scout-confirmed blue taps."""

    results: dict[Cell, ProbeResult]
    metadata: dict[Cell, dict[str, object]]
    clicked_cells: tuple[Cell, ...] = ()
    level_completed: bool = False
    stopped_reason: str | None = None


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
    initial_misses: set[Cell] | None = None,
) -> list[list[str]]:
    states = [["unknown" for _col in range(grid_size)] for _row in range(grid_size)]
    for row, col in misses - hits:
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "scout_miss"
    for row, col in hits:
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "scout_hit"
    for row, col in initial_misses or set():
        if 0 <= row < grid_size and 0 <= col < grid_size:
            states[row][col] = "miss"
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


def _find_l_shaped_hit_block(
    hit_cells: set[Cell] | frozenset[Cell],
) -> tuple[Cell, ...] | None:
    """Return three hits occupying an impossible L shape in a 2x2 block."""
    hits = set(hit_cells)
    candidate_origins = {
        (row + row_offset, col + col_offset)
        for row, col in hits
        for row_offset in (-1, 0)
        for col_offset in (-1, 0)
    }
    for row, col in sorted(candidate_origins):
        block = {
            (row, col),
            (row, col + 1),
            (row + 1, col),
            (row + 1, col + 1),
        }
        occupied = block & hits
        if len(occupied) == 3:
            return tuple(sorted(occupied))
    return None


def _online_hit_evidence_score(metadata: Mapping[str, object]) -> tuple[int, float, int]:
    stable_state = str(metadata.get("stable_state", "unknown"))
    stable_rank = {"miss": -1, "unknown": 0, "hit": 1}.get(stable_state, 0)
    try:
        hit_votes = max(0, int(metadata.get("hit_votes", 0)))
        frame_count = max(1, int(metadata.get("frame_count", 0)))
    except (TypeError, ValueError):
        hit_votes = 0
        frame_count = 1
    return stable_rank, hit_votes / frame_count, hit_votes


def _resolve_false_hit_in_l_shape(
    l_shaped_block: Sequence[Cell],
    evidence_by_cell: Mapping[Cell, Mapping[str, object]],
) -> Cell | None:
    block = set(l_shaped_block)

    # For a screen-down-right ship, the raised red flag can color the cell
    # directly above its upper endpoint.  In grid coordinates this produces
    # one cell on the upper row and the real adjacent pair on the lower row.
    rows: dict[int, set[Cell]] = {}
    for cell in block:
        rows.setdefault(cell[0], set()).add(cell)
    if len(rows) == 2:
        upper_row, lower_row = sorted(rows)
        upper_cells = rows[upper_row]
        lower_cells = rows[lower_row]
        lower_cols = sorted(col for _, col in lower_cells)
        if lower_row != upper_row + 1:
            lower_cols = []
        if (
            len(upper_cells) == 1
            and len(lower_cells) == 2
            and len(lower_cols) == 2
            and lower_cols[1] == lower_cols[0] + 1
            and next(iter(upper_cells))[1] in lower_cols
        ):
            return next(iter(upper_cells))
        if (
            len(upper_cells) == 2
            and len(lower_cells) == 1
            and len(lower_cols) == 1
            and sorted(col for _, col in upper_cells)[1]
            == sorted(col for _, col in upper_cells)[0] + 1
            and next(iter(lower_cells))[1] in {
                sorted(col for _, col in upper_cells)[0],
                sorted(col for _, col in upper_cells)[1],
            }
        ):
            lower_col = next(iter(lower_cells))[1]
            false_candidates = [
                cell for cell in upper_cells if cell[1] != lower_col
            ]
            if len(false_candidates) == 1:
                return false_candidates[0]

    removable: list[Cell] = []
    for candidate in sorted(block):
        remaining = block - {candidate}
        if len(remaining) != 2:
            continue
        first, second = sorted(remaining)
        if first[0] == second[0] or first[1] == second[1]:
            removable.append(candidate)
    if len(removable) != 2:
        return None

    ranked = sorted(
        (
            _online_hit_evidence_score(evidence_by_cell.get(cell, {})),
            cell,
        )
        for cell in removable
    )
    if ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _find_flag_overlap_l_shape(
    cells: set[Cell] | frozenset[Cell],
    *,
    ignored_false_cells: set[Cell] | frozenset[Cell] = frozenset(),
) -> tuple[Cell, frozenset[Cell]] | None:
    """Find supported raised-flag L orientations and their ship cells."""
    cell_set = set(cells)
    for row in sorted({cell[0] for cell in cell_set}):
        cols = sorted(col for cell_row, col in cell_set if cell_row == row)
        for start in range(len(cols) - 2):
            triple_cols = cols[start : start + 3]
            if triple_cols != list(range(triple_cols[0], triple_cols[0] + 3)):
                continue
            lower_triple = frozenset((row, col) for col in triple_cols)
            for false_cell in (
                (row - 2, triple_cols[0]),
                (row - 2, triple_cols[-1]),
            ):
                if false_cell in cell_set and false_cell not in ignored_false_cells:
                    return false_cell, lower_triple
    for col in sorted({cell[1] for cell in cell_set}):
        rows = sorted(row for row, cell_col in cell_set if cell_col == col)
        for start in range(len(rows) - 2):
            triple_rows = rows[start : start + 3]
            if triple_rows != list(range(triple_rows[0], triple_rows[0] + 3)):
                continue
            lower_triple = frozenset((row, col) for row in triple_rows)
            for false_cell in (
                (triple_rows[0], col - 2),
                (triple_rows[0], col + 2),
            ):
                if false_cell in cell_set and false_cell not in ignored_false_cells:
                    return false_cell, lower_triple

    by_row: dict[int, set[Cell]] = {}
    for row, col in cell_set:
        by_row.setdefault(row, set()).add((row, col))
    for upper_row, upper_cells in sorted(by_row.items()):
        lower_cells = by_row.get(upper_row + 1, set())
        upper_cols = sorted(col for _row, col in upper_cells)
        lower_cols = sorted(col for _row, col in lower_cells)
        if len(upper_cells) == 1 and len(lower_cells) == 2:
            if (
                lower_cols[1] == lower_cols[0] + 1
                and next(iter(upper_cells))[1] in lower_cols
            ):
                upper = next(iter(upper_cells))
                if upper not in ignored_false_cells:
                    return upper, frozenset(lower_cells)
        elif len(upper_cells) == 2 and len(lower_cells) == 1:
            lower_col = lower_cols[0]
            if (
                upper_cols[1] == upper_cols[0] + 1
                and lower_col in upper_cols
            ):
                false_cell = next(
                    cell for cell in upper_cells if cell[1] != lower_col
                )
                if false_cell in ignored_false_cells:
                    continue
                return false_cell, frozenset(
                    {
                        (upper_row, lower_col),
                        (upper_row + 1, lower_col),
                    }
                )
        lower_edge_cells = by_row.get(upper_row + 2, set())
        lower_edge_cols = sorted(col for _row, col in lower_edge_cells)
        if len(upper_cells) == 1 and len(lower_edge_cells) >= 3:
            for start in range(len(lower_edge_cols) - 2):
                lower_triple_cols = lower_edge_cols[start : start + 3]
                if lower_triple_cols != list(
                    range(lower_triple_cols[0], lower_triple_cols[0] + 3)
                ):
                    continue
                upper = next(iter(upper_cells))
                if upper[1] in {lower_triple_cols[0], lower_triple_cols[-1]}:
                    lower_triple = frozenset(
                        {
                            (upper_row + 2, col)
                            for col in lower_triple_cols
                        }
                    )
                    if upper not in ignored_false_cells:
                        return upper, lower_triple
    return None


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
    if _active_probe is not None and _active_probe.request_may_be_pending:
        return True

    persisted = read_pending_probe()
    if persisted is None:
        return False
    return str(persisted.get("phase", "")).upper() in {
        ProbePhase.REQUEST_PENDING.name,
        ProbePhase.RESULT_VISIBLE.name,
        ProbePhase.RESULT_RECORDED.name,
        "INTERRUPTED",
    }


def _create_probe_sample_dir(
    level: int,
    cell: Cell,
    index: int,
    *,
    prune_retention: bool = True,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    row, col = cell
    sample_dir = PROBE_SAMPLE_DIR / f"level_{level}_cell_{index}_r{row}_c{col}_{timestamp}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if prune_retention:
        _prune_probe_sample_dirs()
        _prune_screenshot_storage(protected_paths=(sample_dir,))
    return sample_dir


def _create_red_scout_sample_dir(
    level: int,
    center: Cell,
    index: int,
    attempt: int,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    row, col = center
    sample_dir = RED_SCOUT_SAMPLE_DIR / (
        f"level_{level}_attempt_{attempt:02d}_cell_{index}_"
        f"r{row}_c{col}_{timestamp}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    _prune_red_scout_sample_dirs()
    _prune_screenshot_storage(protected_paths=(sample_dir,))
    return sample_dir


def _managed_screenshot_sample_dirs() -> list[tuple[int, int, Path]]:
    managed: list[tuple[int, int, Path]] = []
    roots = (
        (PROBE_SAMPLE_DIR, lambda name: name.startswith("level_") and "_cell_" in name),
        (
            RED_SCOUT_SAMPLE_DIR,
            lambda name: name.startswith("level_") and "_attempt_" in name,
        ),
    )
    for directory, matches in roots:
        try:
            root = directory.resolve(strict=False)
            children = tuple(directory.iterdir())
        except (FileNotFoundError, OSError):
            continue
        for path in children:
            try:
                if (
                    path.is_symlink()
                    or not path.is_dir()
                    or not matches(path.name)
                    or path.resolve(strict=False).parent != root
                ):
                    continue
                entries = tuple(path.iterdir())
                if any(entry.is_symlink() or not entry.is_file() for entry in entries):
                    continue
                size = sum(entry.stat().st_size for entry in entries)
                managed.append((path.stat().st_mtime_ns, size, path))
            except OSError:
                continue
    return managed


def _run_debug_storage_bytes() -> int:
    try:
        return sum(
            path.stat().st_size
            for path in RUN_DEBUG_DIR.iterdir()
            if path.is_file() and not path.is_symlink()
        )
    except (FileNotFoundError, OSError):
        return 0


def _prune_screenshot_storage(
    max_bytes: int = MAX_SCREENSHOT_STORAGE_BYTES,
    *,
    protected_paths: Sequence[Path] = (),
) -> None:
    if max_bytes < 1:
        return
    protected = {
        Path(path).resolve(strict=False)
        for path in protected_paths
    }
    managed = _managed_screenshot_sample_dirs()
    total_bytes = _run_debug_storage_bytes() + sum(size for _mtime, size, _path in managed)
    if total_bytes <= max_bytes:
        return

    removed = 0
    for _mtime, size, path in sorted(managed, key=lambda item: item[0]):
        if total_bytes <= max_bytes:
            break
        if path.resolve(strict=False) in protected:
            continue
        try:
            entries = tuple(path.iterdir())
            if any(entry.is_symlink() or not entry.is_file() for entry in entries):
                continue
            for entry in entries:
                entry.unlink()
            path.rmdir()
            total_bytes -= size
            removed += 1
        except OSError as exc:
            logger.warning("failed to prune screenshot storage directory %s: %s", path, exc)
    if removed:
        logger.info(
            "screenshot storage retention removed %s directories; remaining=%.1f MB limit=%.1f MB",
            removed,
            total_bytes / (1024 * 1024),
            max_bytes / (1024 * 1024),
        )


def _compact_successful_red_scout_images(sample_dir: Path) -> None:
    keep = {"selected.png"}
    for prefix in ("before", "after", "verify"):
        paths = sorted(sample_dir.glob(f"{prefix}_*.png"))
        if paths:
            keep.add(paths[(len(paths) - 1) // 2].name)
    for path in sample_dir.glob("*.png"):
        if path.name not in keep and path.is_file() and not path.is_symlink():
            path.unlink()


def _prune_red_scout_sample_dirs(
    max_directories: int = MAX_RED_SCOUT_SAMPLE_DIRS,
) -> None:
    if max_directories < 1:
        return

    try:
        root = RED_SCOUT_SAMPLE_DIR.resolve(strict=False)
        children = tuple(RED_SCOUT_SAMPLE_DIR.iterdir())
    except (FileNotFoundError, OSError):
        return

    managed: list[tuple[int, Path]] = []
    for path in children:
        try:
            if (
                path.is_symlink()
                or not path.is_dir()
                or not path.name.startswith("level_")
                or "_attempt_" not in path.name
                or path.resolve(strict=False).parent != root
            ):
                continue
            managed.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue

    managed.sort(key=lambda item: item[0], reverse=True)
    removed = 0
    for _mtime, path in managed[max_directories:]:
        try:
            entries = tuple(path.iterdir())
            if any(entry.is_symlink() or not entry.is_file() for entry in entries):
                logger.warning(
                    "red scout sample retention skipped unsafe directory: %s",
                    path,
                )
                continue
            for entry in entries:
                entry.unlink()
            path.rmdir()
            removed += 1
        except OSError as exc:
            logger.warning("failed to prune red scout sample directory %s: %s", path, exc)
    if removed:
        logger.info(
            "red scout sample retention removed %s old directories; keeping newest %s",
            removed,
            max_directories,
        )


def _red_scout_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _red_scout_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_red_scout_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def _write_red_scout_analysis(
    sample_dir: Path,
    result: RedScoutResult,
    *,
    level: int,
    index: int,
    attempt: int,
) -> None:
    complete_six = _red_scout_result_is_complete_six(result)
    payload = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "level": int(level),
        "attempt": int(attempt),
        "index": int(index),
        "center": list(result.center_cell),
        "valid": bool(result.valid),
        "complete_six": complete_six,
        "invalid_reason": result.invalid_reason,
        "affected": [list(cell) for cell in sorted(result.affected_cells)],
        "hits": [list(cell) for cell in sorted(result.hit_cells)],
        "misses": [list(cell) for cell in sorted(result.miss_cells)],
        "unknown": [list(cell) for cell in sorted(result.unknown_cells)],
        "confidence": [
            {
                "cell": list(cell),
                "value": float(result.confidence_by_cell[cell]),
            }
            for cell in sorted(result.confidence_by_cell)
        ],
        "diagnostics": _red_scout_json_value(result.diagnostics),
    }
    output_path = sample_dir / "analysis.json"
    temp_path = output_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(output_path)


def _prune_probe_sample_dirs(
    max_directories: int = MAX_PROBE_SAMPLE_DIRS,
) -> None:
    if max_directories < 1:
        return

    try:
        root = PROBE_SAMPLE_DIR.resolve(strict=False)
        children = tuple(PROBE_SAMPLE_DIR.iterdir())
    except (FileNotFoundError, OSError):
        return

    managed: list[tuple[int, Path]] = []
    for path in children:
        try:
            if (
                path.is_symlink()
                or not path.is_dir()
                or not path.name.startswith("level_")
                or path.resolve(strict=False).parent != root
            ):
                continue
            managed.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue

    managed.sort(key=lambda item: item[0], reverse=True)
    removed = 0
    for _mtime, path in managed[max_directories:]:
        try:
            entries = tuple(path.iterdir())
            if any(entry.is_symlink() or not entry.is_file() for entry in entries):
                logger.warning("probe sample retention skipped unsafe directory: %s", path)
                continue
            for entry in entries:
                entry.unlink()
            path.rmdir()
            removed += 1
        except OSError as exc:
            logger.warning("failed to prune old probe sample directory %s: %s", path, exc)
    if removed:
        logger.info(
            "probe sample retention removed %s old directories; keeping newest %s",
            removed,
            max_directories,
        )


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
    evidence_kind = getattr(result, "evidence_kind", None)
    if not evidence_kind:
        evidence_kind = "dynamic_attack_hit" if result.state == "hit" else "unknown"
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
        "evidence_kind": str(evidence_kind),
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
    adaptive_frames_stopped: bool = False,
    result_unknown: bool = False,
    stable_analysis: Mapping[str, object] | None = None,
) -> None:
    decision = "unknown" if result_unknown else ("hit" if hit else "miss")
    evidence_counts: Counter[str] = Counter()
    for frame in frames:
        frame_result = frame.get("result", {}) if isinstance(frame, Mapping) else {}
        if isinstance(frame_result, Mapping):
            kind = str(frame_result.get("evidence_kind", "unknown"))
            evidence_counts[kind] += 1
    primary_evidence = (
        evidence_counts.most_common(1)[0][0] if evidence_counts else "unknown"
    )
    payload = {
        "level": level,
        "cell": list(cell),
        "index": index,
        "point": list(point),
        "decision": decision,
        "hit_votes": hit_votes,
        "decision_reason": decision_reason,
        "evidence_kind": primary_evidence,
        "evidence_kinds": dict(evidence_counts),
        "frame_count": len(frames),
        "min_hit_votes": MIN_HIT_RESULT_VOTES,
        "suspect_extra_checked": suspect_extra_checked,
        "adaptive_frames_stopped": adaptive_frames_stopped,
        "frames": frames,
    }
    if stable_analysis is not None:
        payload["stable_analysis"] = dict(stable_analysis)
    (sample_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _should_preserve_all_probe_images(
    frame_records: Sequence[Mapping[str, object]],
    *,
    suspect_extra_checked: bool,
    victory_detected: bool,
    result_unknown: bool,
) -> bool:
    del frame_records, suspect_extra_checked, victory_detected
    return bool(result_unknown)


def _persist_probe_debug_images(
    sample_dir: Path,
    before_capture,
    frame_captures: Sequence[tuple[Path, object]],
    frame_records: list[dict],
    *,
    preserve_all: bool,
) -> None:
    if before_capture is not None:
        before_capture.save(sample_dir / "before.png")

    for record in frame_records:
        record["saved"] = False
    if not frame_captures:
        return

    if preserve_all or len(frame_captures) != len(frame_records):
        selected = set(range(len(frame_captures)))
    else:
        best_index = max(
            range(len(frame_records)),
            key=lambda index: float(
                frame_records[index].get("result", {}).get("score", 0.0)
            ),
        )
        selected = {best_index}

    for capture_index, (path, capture) in enumerate(frame_captures):
        if capture_index not in selected:
            continue
        capture.save(path)
        if capture_index < len(frame_records):
            frame_records[capture_index]["saved"] = True


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


def _is_strong_hit_frame(result) -> bool:
    """Return whether one frame already meets the strong-hit decision gate."""
    return (
        result.state == "hit"
        and (
            result.score >= STRONG_SINGLE_HIT_SCORE
            or result.confidence >= 0.92
        )
        and not bool(getattr(result, "evidence_vetoed", False))
    )


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


def _analyze_stable_probe_frames(
    before_img: np.ndarray,
    frame_captures: Sequence[tuple[Path, object]],
    point: tuple[int, int],
):
    after_images = [
        capture.image
        for _path, capture in frame_captures
        if isinstance(getattr(capture, "image", None), np.ndarray)
    ]
    if len(after_images) < 3:
        return None
    try:
        return analyze_stable_hit(before_img, after_images, point)
    except Exception as exc:
        logger.warning("stable probe analysis was unavailable: %s", exc)
        return None


def _stable_analysis_to_dict(analysis) -> dict[str, object] | None:
    if analysis is None:
        return None
    return {
        "suspect": stable_hit_is_suspect(analysis),
        "result": _hit_result_to_dict(analysis.result),
        "motion": {
            "inner_ratio": float(analysis.motion.inner_ratio),
            "outer_ratio": float(analysis.motion.outer_ratio),
            "contrast": float(analysis.motion.contrast),
        },
        "registrations": [
            {
                "dx": float(item.dx),
                "dy": float(item.dy),
                "response": float(item.response),
                "accepted": bool(item.accepted),
            }
            for item in analysis.registrations
        ],
    }


def _visible_wreck_for_hit_state(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    red_marker_cells: set[Cell] | None = None,
    cell: Cell | None = None,
    cell_polygon: np.ndarray | None = None,
    require_strong_body: bool = False,
) -> bool:
    """Return real static-wreck evidence, excluding submarine decorations.

    The red object above a surfaced submarine is a persistent decoration.  It
    must never be treated as an already-confirmed hit: callers use this helper
    before a blue shot and a ``True`` result is committed to the strategy as a
    hit without firing.  Only the gray/white wreck body is valid evidence here.
    """
    # Prefer the global marker assignment when available.  Falling back to
    # the local detector preserves compatibility for callers without a grid.
    if red_marker_cells is not None and cell is not None:
        if cell in red_marker_cells:
            return False
    elif red_submarine_marker_visible(image, point):
        return False
    visible = visible_wreck_static_detected(
        image,
        point,
        # Global ownership has already excluded the one true marker cell;
        # do not let the wide local marker radius suppress its neighbor.
        ignore_submarine_marker=(red_marker_cells is not None and cell is not None),
        cell_polygon=cell_polygon,
    )
    if not visible or not require_strong_body:
        return visible
    body_score = completed_ship_body_score(
        image,
        point,
        cell_polygon=cell_polygon,
    )
    if body_score < COMPLETED_SHIP_BODY_MIN_SCORE:
        logger.info(
            "ignoring loose already-visible wreck evidence for cell %s: "
            "body_score=%.3f threshold=%.3f; continuing with the blue tap",
            cell,
            body_score,
            COMPLETED_SHIP_BODY_MIN_SCORE,
        )
        return False
    return True


def apply_wreck_template_confirmation(after_img: np.ndarray, point: tuple[int, int], result) -> bool:
    if not visible_wreck_static_detected(after_img, point):
        return False

    result.state = "hit"
    result.score = max(float(result.score), 0.94)
    result.confidence = max(float(result.confidence), 0.95)
    result.evidence_kind = "static_wreck_hit"
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


def _trusted_completed_cells_from_probe_metadata(
    probe_metadata: Mapping[str, object],
    click_points: Sequence[tuple[int, int]],
    *,
    grid_size: int,
    anchor: Cell | None = None,
    preferred_cells: set[Cell] | frozenset[Cell] = frozenset(),
) -> set[Cell]:
    screenshot = probe_metadata.get("sidebar_completion_screenshot")
    completed_lengths = tuple(
        int(length)
        for length in probe_metadata.get("sidebar_completed_lengths", ())
        if int(length) > 0
    )
    if (
        not isinstance(screenshot, np.ndarray)
        or screenshot.ndim != 3
        or not completed_lengths
        or len(click_points) != grid_size * grid_size
    ):
        return set()

    candidates = detect_completed_submarine_candidate_cells(
        screenshot,
        list(click_points),
        grid_size,
    )
    if not candidates:
        return set()

    preferred = set(preferred_cells)
    if anchor is not None:
        preferred.add(anchor)
    candidates = set(candidates)
    for length in completed_lengths:
        if length < 3:
            continue
        for row in range(grid_size):
            for start_col in range(grid_size - length + 1):
                placement = {
                    (row, col)
                    for col in range(start_col, start_col + length)
                }
                if (
                    placement & preferred
                    and (row, start_col) in candidates
                    and (row, start_col + length - 1) in candidates
                ):
                    candidates.update(placement)
        for col in range(grid_size):
            for start_row in range(grid_size - length + 1):
                placement = {
                    (row, col)
                    for row in range(start_row, start_row + length)
                }
                if (
                    placement & preferred
                    and (start_row, col) in candidates
                    and (start_row + length - 1, col) in candidates
                ):
                    candidates.update(placement)

    resolution = resolve_completed_ship_cells(
        candidates,
        completed_lengths,
        grid_size=grid_size,
        preferred_cells=(
            preferred
        ),
    )
    trusted = set(resolution.cells)
    logger.info(
        "live completed ship geometry: placements=%s unresolved=%s discarded=%s",
        [list(placement) for placement in resolution.placements],
        list(resolution.unresolved_lengths),
        sorted(resolution.discarded_cells),
    )
    return trusted


def _merge_completed_visual_snapshot(
    previous_cells: set[Cell] | frozenset[Cell],
    latest_cells: set[Cell] | frozenset[Cell],
    *,
    completed_lengths: Sequence[int],
    authoritative_cells: set[Cell] | frozenset[Cell] = frozenset(),
) -> set[Cell]:
    previous = set(previous_cells)
    latest = set(latest_cells)
    authoritative = set(authoritative_cells)
    expected_cells = sum(
        int(length)
        for length in completed_lengths
        if int(length) > 0
    )
    merged = latest if expected_cells > 0 and len(latest) == expected_cells else previous
    if not authoritative:
        return merged

    conflicting = {
        cell
        for cell in merged - authoritative
        if any(
            max(abs(cell[0] - row), abs(cell[1] - col)) <= 1
            for row, col in authoritative
        )
    }
    return (merged - conflicting) | authoritative


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
    """开启游戏弱网，并按需等待网络规则生效。"""
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


def _capture_red_ammo_state(
    sample_dir: Path | None = None,
    *,
    prefix: str = "red_ammo",
    include_frames: bool = False,
):
    frames = [
        adb.read_screenshot(
            (
                sample_dir / f"{prefix}_{i}.png"
                if sample_dir is not None
                else RUN_DEBUG_DIR / f"red_ammo_{i}.png"
            )
        )
        for i in range(3)
    ]
    match = locate_red_bomb_button(frames[0])
    fingerprint = build_ammo_fingerprint(frames, match) if match is not None else None
    if match is None or fingerprint is None:
        raise RedScoutSafetyError("red bomb button or ammo fingerprint unavailable")
    return (frames if include_frames else frames[0]), fingerprint, match


def _select_red_bomb(
    match: MatchResult,
    output_path: Path | None = None,
) -> bool:
    adb.click(*match.center)
    adb.delay(0.25)
    return red_bomb_selected(
        adb.read_screenshot(output_path or RUN_DEBUG_DIR / "red_selected.png"),
        match,
    )


def _capture_red_result_frames(sample_dir: Path | None = None):
    return [
        adb.delay(frame_delay).read_screenshot(
            (
                sample_dir / f"after_{frame_index}.png"
                if sample_dir is not None
                else RUN_DEBUG_DIR / f"red_result_{frame_index}.png"
            )
        )
        for frame_index, frame_delay in enumerate(RED_SCOUT_RESULT_FRAME_DELAYS)
    ]


def _verify_red_ammo_unchanged(
    before_fingerprint: AmmoFingerprint,
    sample_dir: Path | None = None,
) -> None:
    write_runtime_status(phase="red_scout_verify_ammo")
    after_state = _capture_red_ammo_state(sample_dir=sample_dir, prefix="verify")
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
            detail_open = _activity_quit_button_visible(screenshot)
            absent_frames = 0 if detail_open else absent_frames + 1
            if absent_frames >= ACTIVITY_EXIT_STABLE_FRAMES:
                logger.info("activity detail exit confirmed; starting offline re-entry")
                return True
        else:
            absent_frames = 0
        sleep(FAST_POLL_INTERVAL_SECONDS)

    if absent_frames:
        screenshot = adb.read_screenshot()
        if (
            isinstance(screenshot, np.ndarray)
            and not _activity_quit_button_visible(screenshot)
        ):
            logger.info(
                "activity detail exit confirmed by final frame after timeout"
            )
            return True

    logger.warning("activity detail did not close within %.1f seconds", timeout)
    return False


def _activity_quit_button_visible(screenshot: np.ndarray) -> bool:
    if not isinstance(screenshot, np.ndarray) or screenshot.ndim < 2:
        return False
    height, width = screenshot.shape[:2]
    reference_width, reference_height = SCREEN_REFERENCE_SIZE
    roi_width, roi_height = ACTIVITY_QUIT_ROI_REFERENCE_SIZE
    x2 = min(width, max(1, round(width * roi_width / reference_width)))
    y2 = min(height, max(1, round(height * roi_height / reference_height)))
    return (
        find_template(screenshot[:y2, :x2], QUIT_ACTIVITY_TEMPLATE)
        is not None
    )


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
    excluded_cells: Sequence[Cell] | set[Cell] | frozenset[Cell] | None = None,
    learned_footprint: RedFootprint | None = None,
    submarine_lengths: Sequence[int] = (),
):
    return RedScoutAnalyzer().analyze(
        before_image=before_image,
        after_images=after_images,
        click_points=click_points,
        grid_size=grid_size,
        center_cell=center_cell,
        excluded_cells=set() if excluded_cells is None else excluded_cells,
        learned_footprint=learned_footprint,
        submarine_lengths=submarine_lengths,
    )


def _red_scout_result_is_complete_six(result: RedScoutResult) -> bool:
    return (
        result.valid
        and len(result.affected_cells) == 6
        and not result.unknown_cells
        and result.affected_cells == result.hit_cells | result.miss_cells
    )


def _red_scout_result_quality(result: RedScoutResult) -> tuple[int, int, int, int]:
    classified = len(result.hit_cells | result.miss_cells)
    return (
        int(_red_scout_result_is_complete_six(result)),
        int(result.valid),
        classified,
        len(result.affected_cells) - len(result.unknown_cells),
    )


def _red_result_frame_transition_reasons(
    frame: object,
    *,
    reference: np.ndarray,
) -> tuple[str, ...]:
    """Return conservative reasons for excluding one captured result frame.

    Red result capture intentionally overlaps the REJECT/dialog wait.  A frame
    containing that dialog (or a victory banner) must not contribute cell
    votes, because the overlay can make many unrelated grid cells appear to
    change.  This helper only identifies explicit UI transitions and malformed
    frames; it does not infer board state and never touches network state.
    """
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] not in (3, 4)
        or frame.size == 0
    ):
        return ("invalid_screenshot",)
    if frame.shape != reference.shape:
        return ("shape_mismatch",)

    reasons: list[str] = []
    # Template matching is deliberately isolated behind exception guards.  A
    # missing/corrupt optional template should leave the original fail-closed
    # analyzer path intact rather than changing the transaction outcome.
    try:
        if find_connection_interrupted_dialog(frame) is not None:
            reasons.append("connection_interrupted_dialog")
    except Exception as exc:
        logger.debug("red result dialog transition check skipped: %s", exc)
    try:
        if find_victory_banner(frame) is not None:
            reasons.append("victory_banner")
    except Exception as exc:
        logger.debug("red result victory transition check skipped: %s", exc)
    return tuple(reasons)


def _filter_red_result_transition_frames(
    before_image: np.ndarray,
    after_images: Sequence[object],
) -> tuple[Sequence[object], Mapping[str, object]]:
    """Filter explicit UI-transition frames before red-result analysis.

    The capture schedule and network protocol remain unchanged.  Filtering is
    applied only when at least ``RED_SCOUT_MIN_ANALYSIS_FRAMES`` clean frames
    remain; otherwise the original frames are returned so the analyzer's
    existing fail-closed checks decide the result.  Returning the raw frames in
    that case also preserves evidence for diagnostics and avoids silently
    turning an under-sampled result into a valid one.
    """
    try:
        raw_frames = tuple(after_images)
    except TypeError:
        return after_images, {
            "captured_count": 0,
            "kept_indices": (),
            "discarded_frames": (),
            "filter_applied": False,
            "reason": "non_iterable_frames",
        }

    kept: list[object] = []
    kept_indices: list[int] = []
    discarded: list[dict[str, object]] = []
    for index, frame in enumerate(raw_frames):
        reasons = _red_result_frame_transition_reasons(
            frame,
            reference=before_image,
        )
        if reasons:
            discarded.append({"index": index, "reasons": reasons})
            continue
        kept.append(frame)
        kept_indices.append(index)

    diagnostics: dict[str, object] = {
        "captured_count": len(raw_frames),
        "kept_indices": tuple(kept_indices),
        "discarded_frames": tuple(discarded),
        "filter_applied": False,
    }
    if not discarded:
        return after_images, diagnostics

    if len(kept) < RED_SCOUT_MIN_ANALYSIS_FRAMES:
        diagnostics["reason"] = "insufficient_stable_frames"
        logger.warning(
            "red scout result frame filter found %s transition frame(s), but only "
            "%s clean frame(s) remain; keeping raw frames for fail-closed analysis",
            len(discarded),
            len(kept),
        )
        return after_images, diagnostics

    diagnostics["filter_applied"] = True
    logger.info(
        "red scout result frame filter discarded indices=%s reasons=%s",
        tuple(item["index"] for item in discarded),
        tuple(item["reasons"] for item in discarded),
    )
    return tuple(kept), diagnostics


def _attach_red_capture_diagnostics(
    result: RedScoutResult,
    capture_diagnostics: Mapping[str, object],
) -> RedScoutResult:
    """Add capture filtering details without mutating an existing result."""
    discarded = capture_diagnostics.get("discarded_frames")
    if not isinstance(result, RedScoutResult) or not discarded:
        return result
    # Test doubles and alternate callers may intentionally provide opaque
    # frame objects.  There is no useful transition evidence in that case, so
    # preserve the original result object and its identity.
    if not capture_diagnostics.get("filter_applied") and all(
        "invalid_screenshot" in tuple(item.get("reasons", ()))
        for item in discarded
        if isinstance(item, Mapping)
    ):
        return result
    diagnostics = dict(result.diagnostics)
    diagnostics["capture_frame_filter"] = dict(capture_diagnostics)
    return replace(result, diagnostics=diagnostics)


def _analyze_red_result_with_baseline_consensus(
    *,
    before_images: Sequence[object],
    after_images: Sequence[object],
    click_points: Sequence[tuple[int, int]],
    grid_size: int,
    center_cell: Cell,
    excluded_cells: Sequence[Cell] | set[Cell] | frozenset[Cell] | None = None,
    learned_footprint: RedFootprint | None = None,
    submarine_lengths: Sequence[int] = (),
) -> RedScoutResult:
    baselines = tuple(
        baseline
        for baseline in before_images
        if isinstance(baseline, np.ndarray) and baseline.ndim == 3
    )
    if not baselines:
        raise ValueError("red scout analysis requires at least one baseline frame")
    if any(baseline.shape != baselines[0].shape for baseline in baselines[1:]):
        raise ValueError("red scout baseline frames must have matching shapes")

    analysis_after_images, capture_diagnostics = (
        _filter_red_result_transition_frames(baselines[0], after_images)
    )

    def analyze(baseline: object) -> RedScoutResult:
        return _analyze_red_result(
            baseline,
            analysis_after_images,
            click_points,
            grid_size,
            center_cell,
            excluded_cells=excluded_cells,
            learned_footprint=learned_footprint,
            submarine_lengths=submarine_lengths,
        )

    median_baseline = np.median(np.stack(baselines, axis=0), axis=0).astype(
        baselines[0].dtype
    )
    primary = analyze(median_baseline)
    uncertain_reasons = {
        "preflight_failed",
        "before_hit_detection_failed",
        "stable_hit_detection_failed",
        "evidence_collection_failed",
        "result_classification_failed",
        "insufficient_changed_cells",
        "ambiguous_result",
    }
    needs_fallback = bool(primary.unknown_cells) or (
        not primary.valid and primary.invalid_reason in uncertain_reasons
    )
    if not needs_fallback or len(baselines) == 1:
        return _attach_red_capture_diagnostics(primary, capture_diagnostics)

    ranked_baselines = sorted(
        baselines,
        key=lambda baseline: float(
            np.mean(
                np.abs(
                    baseline.astype(np.int16)
                    - median_baseline.astype(np.int16)
                )
            )
        ),
    )
    fallback = next(
        (
            baseline
            for baseline in ranked_baselines
            if not np.array_equal(baseline, median_baseline)
        ),
        None,
    )
    if fallback is None:
        return _attach_red_capture_diagnostics(primary, capture_diagnostics)

    try:
        alternative = analyze(fallback)
    except Exception as exc:
        logger.warning(
            "red scout fallback baseline analysis failed; keeping median result: %s",
            exc,
        )
        return _attach_red_capture_diagnostics(primary, capture_diagnostics)

    if _red_scout_result_quality(alternative) > _red_scout_result_quality(primary):
        logger.info(
            "red scout fallback baseline recovered a stronger result: affected=%s "
            "hits=%s misses=%s",
            sorted(alternative.affected_cells),
            sorted(alternative.hit_cells),
            sorted(alternative.miss_cells),
        )
        return _attach_red_capture_diagnostics(alternative, capture_diagnostics)
    return _attach_red_capture_diagnostics(primary, capture_diagnostics)


def _stop_and_latch_safety_failure(
    reason: str,
    error_type: type[RuntimeError],
) -> None:
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
            logger.error("network safety stop operation failed (%s): %s", name, exc)
    final_reason = first_reason
    if not stopped:
        final_reason = f"{final_reason}; process did not exit"
    raise error_type(final_reason)


def _stop_and_latch_red_safety_failure(reason: str) -> None:
    _stop_and_latch_safety_failure(reason, RedScoutSafetyError)


def _stop_and_latch_blue_safety_failure(reason: str) -> None:
    _stop_and_latch_safety_failure(reason, ProbeProtocolError)


def _verify_network_isolated_or_fail_closed(*, red_scout: bool) -> None:
    mode_label = "red scout" if red_scout else "blue probe"
    try:
        isolation = adb.verify_app_network_isolated(GAME_PACKAGE_NAME)
    except Exception as exc:
        reason = f"{mode_label} network isolation verification failed: {exc}"
    else:
        if bool(getattr(isolation, "safe", False)):
            return
        reason = str(getattr(isolation, "detail", "network isolation unsafe"))

    if red_scout:
        _stop_and_latch_red_safety_failure(reason)
    else:
        _stop_and_latch_blue_safety_failure(reason)


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
    excluded_cells: Sequence[Cell] | set[Cell] | frozenset[Cell] = (),
    learned_footprint: RedFootprint | None = None,
    submarine_lengths: Sequence[int] = (),
    attempt: int | None = None,
):
    global _active_probe
    transaction = None
    grid_clicked = False
    pending_marker_written = False
    sample_dir: Path | None = None
    sample_failed = False
    analysis_executor: ThreadPoolExecutor | None = None
    analysis_future = None
    capture_executor: ThreadPoolExecutor | None = None
    capture_future = None
    if attempt is not None:
        try:
            sample_dir = _create_red_scout_sample_dir(
                level,
                center_cell,
                index,
                attempt,
            )
        except OSError as exc:
            logger.warning("could not create red scout sample directory: %s", exc)
    try:
        write_runtime_status(phase="red_scout_preflight", level=level)
        enable_weak_network(PROBE_DROP_SETTLE_SECONDS)
        _verify_network_isolated_or_fail_closed(red_scout=True)
        before_capture, before_fingerprint, match = _capture_red_ammo_state(
            sample_dir=sample_dir,
            prefix="before",
            include_frames=True,
        )
        before_images = (
            list(before_capture)
            if isinstance(before_capture, (list, tuple))
            else [before_capture]
        )
        transaction = ProbeTransaction(level, center_cell, index)
        _active_probe = transaction
        transaction.advance(ProbePhase.REQUEST_PENDING)
        if not _select_red_bomb(
            match,
            output_path=(sample_dir / "selected.png" if sample_dir is not None else None),
        ):
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
            (
                sample_dir / "exit_attempt.png"
                if sample_dir is not None
                else RUN_DEBUG_DIR / "red_debug_back.png"
            ),
            use_system_back=True,
        )
        if _reenter_activity_for_probe_result():
            transaction.advance(ProbePhase.RESULT_VISIBLE)
            transaction.advance(ProbePhase.RESULT_RECORDED)
            update_pending_probe(phase=ProbePhase.RESULT_RECORDED.name, local_victory=True)
            local_victory_result = RedScoutResult(
                center_cell=center_cell,
                affected_cells=frozenset(),
                hit_cells=frozenset(),
                miss_cells=frozenset(),
                unknown_cells=frozenset(),
                footprint=None,
                valid=False,
                confidence_by_cell={},
                level_completed=False,
                invalid_reason="local_victory_screen",
                diagnostics={"stage": "local_victory"},
            )
            if sample_dir is not None and attempt is not None:
                try:
                    _write_red_scout_analysis(
                        sample_dir,
                        local_victory_result,
                        level=level,
                        index=index,
                        attempt=attempt,
                    )
                except OSError as exc:
                    logger.warning("could not write red scout analysis: %s", exc)
            logger.warning(
                "red scout displayed a local victory; discarding it and continuing with blue "
                "attacks because the red request must never be committed"
            )
            _discard_pending_request_and_prepare_next_probe(transaction)
            _verify_red_ammo_unchanged(before_fingerprint, sample_dir=sample_dir)
            pending_marker_written = False
            _active_probe = None
            return local_victory_result
        transaction.advance(ProbePhase.RESULT_VISIBLE)
        update_pending_probe(phase=ProbePhase.RESULT_VISIBLE.name)
        write_runtime_status(phase="red_scout_capture", level=level)
        # Start result capture first. Let it run for a short head start, then
        # enable REJECT so the connection-dialog wait overlaps the remaining
        # screenshot delays.
        capture_started = Event()

        def capture_red_result_frames() -> list[np.ndarray]:
            capture_started.set()
            return _capture_red_result_frames(sample_dir=sample_dir)

        capture_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="red-scout-capture",
        )
        capture_future = capture_executor.submit(capture_red_result_frames)
        # Ensure the worker has begun before waiting for the complete capture.
        # This is effectively free in production and keeps ordering deterministic
        # in tests with an immediate mocked capture.
        capture_started.wait(timeout=0.1)
        # Keep the connection-interrupted dialog out of every result frame:
        # collect all three frames before enabling REJECT.
        if capture_future is not None:
            capture_future.result()
        adb.enable_reject_network(GAME_PACKAGE_NAME)
        transaction.red_reject_enabled = True
        transaction.advance(ProbePhase.RESULT_RECORDED)
        update_pending_probe(phase=ProbePhase.RESULT_RECORDED.name)

        analysis_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="red-scout-analysis",
        )

        def analyze_captured_red_result() -> RedScoutResult:
            assert capture_future is not None
            captured_after_images = capture_future.result()
            # A victory banner can appear during the result animation before
            # the re-entry probe observes it. Do not feed that transition frame
            # to the red-scout analyzer; propagate the same local-victory
            # signal used by the re-entry path so remaining scouts are stopped.
            if any(
                find_victory_banner(frame) is not None
                for frame in captured_after_images
                if isinstance(frame, np.ndarray)
            ):
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
                    invalid_reason="local_victory_screen",
                    diagnostics={"stage": "result_frame_victory"},
                )
            return _analyze_red_result_with_baseline_consensus(
                before_images=before_images,
                after_images=captured_after_images,
                click_points=all_click_points,
                grid_size=grid_size,
                center_cell=center_cell,
                excluded_cells=excluded_cells,
                learned_footprint=learned_footprint,
                submarine_lengths=submarine_lengths,
            )

        analysis_future = analysis_executor.submit(
            analyze_captured_red_result,
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
        _verify_red_ammo_unchanged(before_fingerprint, sample_dir=sample_dir)
        pending_marker_written = False
        if capture_future is not None:
            capture_future.result()
        analysis = analysis_future.result()
        if capture_executor is not None:
            capture_executor.shutdown(wait=True)
            capture_executor = None
            capture_future = None
        analysis_executor.shutdown(wait=True)
        analysis_executor = None
        analysis_future = None
        if sample_dir is not None and attempt is not None:
            try:
                _write_red_scout_analysis(
                    sample_dir,
                    analysis,
                    level=level,
                    index=index,
                    attempt=attempt,
                )
                if analysis.valid and not analysis.unknown_cells and not analysis.invalid_reason:
                    _compact_successful_red_scout_images(sample_dir)
            except OSError as exc:
                logger.warning("could not write red scout analysis: %s", exc)
        _active_probe = None
        return analysis
    except Exception as exc:
        sample_failed = True
        if analysis_future is not None:
            analysis_future.cancel()
        if analysis_executor is not None:
            analysis_executor.shutdown(wait=False, cancel_futures=True)
        discard_completed = bool(
            transaction is not None
            and transaction.phase is ProbePhase.COMPLETE
            and getattr(transaction, "red_request_discarded", False)
            and not pending_marker_written
        )
        if discard_completed:
            logger.error(
                "red scout analysis failed after the red request was safely discarded: %s",
                exc,
            )
            _active_probe = None
            raise
        if grid_clicked:
            try:
                update_pending_probe(phase="INTERRUPTED", error=str(exc))
            except Exception as marker_exc:
                logger.error("could not update interrupted red probe marker: %s", marker_exc)
            if isinstance(exc, RedScoutSafetyError) and _network_fail_closed_reason is not None:
                raise
            if (
                isinstance(exc, DiscardRecoveryError)
                and transaction is not None
                and transaction.phase is ProbePhase.REQUEST_DISCARDED
                and bool(getattr(transaction, "red_request_discarded", False))
            ):
                reason = f"red scout discard recovery stalled: {exc}"
                logger.critical(
                    "%s; keeping DROP/REJECT and leaving the game process running",
                    reason,
                )
                raise RedScoutSafetyError(reason) from exc
            _stop_and_latch_red_safety_failure(
                f"red scout transaction interrupted: {exc}"
            )
        if pending_marker_written:
            clear_pending_probe()
        _active_probe = None
        raise
    finally:
        if capture_executor is not None:
            capture_executor.shutdown(wait=True)
        if sample_dir is not None:
            protected = (
                (sample_dir,)
                if sample_failed
                or (transaction is not None and transaction.request_may_be_pending)
                else ()
            )
            _prune_screenshot_storage(protected_paths=protected)


def cleanup_reject_network(reason: str = "脚本退出") -> None:
    """关闭游戏 REJECT 断网残留，避免影响本次或下次运行。"""
    if _network_fail_closed_reason is not None:
        logger.critical("REJECT cleanup refused: %s", _network_fail_closed_reason)
        return
    if _has_pending_probe_request():
        transaction = _active_probe
        logger.critical(
            "%s，但格子 %s 的探测仍可能待提交；保留 REJECT 断网",
            reason,
            transaction.cell if transaction else None,
        )
        return
    try:
        logger.info("%s，正在清理 REJECT 断网", reason)
        adb.disable_reject_network(GAME_PACKAGE_NAME)
    except Exception as exc:
        logger.error("清理 REJECT 断网失败: %s", exc)


def handle_exit_signal(signum: int, _frame) -> None:
    """收到退出信号时先执行安全清理，再退出进程。"""
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
    prepare_activity_list: bool | None = None,
) -> bool:
    """进入活动详情页。

    ``re_enter=False`` 用于没有待验证请求的普通进入，允许重启恢复。
    ``re_enter=True`` 用于点击后的第二次进入，此时 DROP 下可能仍有暂存请求，
    任何失败都必须立即中止，不能复用会关闭弱网的普通恢复流程。
    刚登录后的活动入口加载较慢，可通过 ``activity_button_timeout`` 延长轮询。
    客户端完整重载后可用 ``prepare_activity_list=True`` 恢复首次进入的列表位置。
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
    should_prepare_activity_list = (
        not re_enter
        if prepare_activity_list is None
        else bool(prepare_activity_list)
    )

    last_failure = "进入活动失败"
    level_completed = False
    for attempt in range(1, max_retries + 1):
        adb.delay(
            ACTIVITY_REENTRY_INITIAL_DELAY_SECONDS if re_enter else 0.2
        )
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

        res = wait_until_occur(
            ACTIVITY_BUTTON_TEMPLATE,
            timeout=button_timeout,
            poll_interval=(
                ACTIVITY_REENTRY_POLL_INTERVAL_SECONDS
                if re_enter
                else FAST_POLL_INTERVAL_SECONDS
            ),
        )
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
        if should_prepare_activity_list:
            adb.delay(0.4).swipe(*ACTIVITY_LIST_SWIPE)  # 首次进入需要展示全部项
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
        red_scout_valid=0,
        red_scout_complete_six=0,
    )


def _grid_calibration_error(
    click_points: Sequence[tuple[int, int]],
    quad: np.ndarray,
    image: np.ndarray,
    grid_size: int,
) -> str | None:
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
        return "screenshot is invalid"
    if len(click_points) != grid_size * grid_size:
        return f"expected {grid_size * grid_size} points, got {len(click_points)}"

    try:
        normalized_quad = np.asarray(quad, dtype=np.float32)
    except (TypeError, ValueError):
        return "quad is not numeric"
    if normalized_quad.shape != (4, 2) or not np.isfinite(normalized_quad).all():
        return "quad must contain four finite points"

    height, width = image.shape[:2]
    if any(
        x < 0 or x >= width or y < 0 or y >= height
        for x, y in normalized_quad
    ):
        return "quad extends outside the screenshot"
    contour = normalized_quad.reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        return "quad is not convex"
    minimum_area = max(100.0, float(width * height) * 0.01)
    if abs(float(cv2.contourArea(contour))) < minimum_area:
        return "quad area is too small"

    normalized_points: list[tuple[int, int]] = []
    for raw_point in click_points:
        try:
            raw_x, raw_y = raw_point
            x = float(raw_x)
            y = float(raw_y)
        except (TypeError, ValueError):
            return f"invalid click point: {raw_point!r}"
        if not np.isfinite(x) or not np.isfinite(y):
            return f"non-finite click point: {raw_point!r}"
        if not 0 <= x < width or not 0 <= y < height:
            return f"click point is outside the screenshot: {raw_point!r}"
        if cv2.pointPolygonTest(contour, (x, y), False) < 0:
            return f"click point is outside the grid quad: {raw_point!r}"
        normalized_points.append((int(round(x)), int(round(y))))

    if len(set(normalized_points)) != len(normalized_points):
        return "click points contain duplicates"
    return None


def get_click_points(
    level: int, grid_img: np.ndarray
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """按配置读取人工点位，失败时回退到自动识别。"""
    grid_size = get_level_grid_size(level)

    if USE_SAVED_POINTS:
        try:
            saved_points = read_saved_points(level, expected_n=grid_size)
            saved_quad = read_saved_quad(level)
        except Exception as exc:
            logger.warning("failed to read saved points for level %s; falling back to auto detection: %s", level, exc)
        else:
            if saved_points is not None and saved_quad is not None:
                calibration_error = _grid_calibration_error(
                    saved_points,
                    saved_quad,
                    grid_img,
                    grid_size,
                )
                if calibration_error is None:
                    logger.info("level %s uses saved calibration points: %s", level, len(saved_points))
                    return saved_points, saved_quad
                logger.warning(
                    "level %s saved calibration is unsafe; falling back to auto detection: %s",
                    level,
                    calibration_error,
                )
            logger.warning("第 %s 关人工点位不存在或数量不正确，回退自动识别", level)

    grid_result = detect_diamond_centers(grid_img, grid_size)
    calibration_error = _grid_calibration_error(
        grid_result.points,
        grid_result.global_quad,
        grid_img,
        grid_size,
    )
    if calibration_error is not None:
        raise RuntimeError(
            f"unsafe grid calibration for level {level}; refusing to probe: {calibration_error}"
        )
    logger.info("level %s uses auto-detected points: %s", level, len(grid_result.points))
    return grid_result.points, grid_result.global_quad


def handle_game_level(
    level: int,
    hit_map: list[list[int]],
    run_started_at: float | None = None,
    settings: RedScoutSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """处理单个关卡：有潜艇配置时使用策略，缺少配置时逐格扫描。"""
    adb.delay(1.5)
    grid_img = adb.read_screenshot()
    click_points, grid_quad = get_click_points(level, grid_img)
    grid_size = get_level_grid_size(level)

    submarines = get_configured_submarines(level, SUBMARINES)
    visible_hits: set[Cell] = set()
    initial_visual_hits: set[Cell] = set()
    completed_visual_hits: set[Cell] = set()
    red_marker_completed_cells: set[Cell] = set()
    sidebar_progress: SidebarProgress | None = None
    partial_wreck_cells: set[Cell] | None = None
    authoritative_completed_placements: tuple[Placement, ...] = ()
    marker_completed_lengths: tuple[int, ...] = ()
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
        completed_anchor_candidates = detect_completed_submarine_candidate_cells(
            grid_img,
            click_points,
            grid_size,
        )
        if completed_anchor_candidates:
            logger.info(
                "level %s completed ship anchor review found %s candidate cells",
                level,
                len(completed_anchor_candidates),
            )
        completed_candidates = (
            completed_anchor_candidates
            if completed_anchor_candidates
            else set(visible_hits) - partial_cells
        )
        if sidebar_progress is not None:
            completed_resolution = resolve_completed_ship_cells(
                completed_candidates,
                sidebar_progress.completed_lengths,
                grid_size=grid_size,
            )
            completed_visual_hits = set(completed_resolution.cells)
            if completed_anchor_candidates:
                red_marker_completed_cells = {
                    cell
                    for placement in completed_resolution.placements
                    if set(placement) & set(completed_anchor_candidates)
                    for cell in placement
                }
            authoritative_completed_placements = tuple(
                Placement(
                    length=len(cells),
                    direction=(
                        "H"
                        if len({row for row, _ in cells}) == 1
                        else "V"
                    ),
                    cells=tuple(cells),
                )
                for cells in completed_resolution.placements
            )
            logger.info(
                "level %s completed ship geometry: placements=%s unresolved=%s discarded=%s",
                level,
                [list(placement) for placement in completed_resolution.placements],
                list(completed_resolution.unresolved_lengths),
                sorted(completed_resolution.discarded_cells),
            )
        else:
            # A red submarine component is itself a completion signal.  When
            # the sidebar is unavailable, resolve the nearby hull candidates
            # against the configured fleet lengths instead of treating every
            # candidate as an ordinary hit.  The red component/anchor is never
            # inserted as a hit coordinate by this resolution.
            marker_resolution = resolve_completed_ship_cells(
                completed_anchor_candidates,
                submarines,
                grid_size=grid_size,
            )
            completed_visual_hits = set(marker_resolution.cells)
            red_marker_completed_cells = set(completed_visual_hits)
            marker_completed_lengths = tuple(
                len(cells) for cells in marker_resolution.placements
            )
            authoritative_completed_placements = tuple(
                Placement(
                    length=len(cells),
                    direction=(
                        "H"
                        if len({row for row, _ in cells}) == 1
                        else "V"
                    ),
                    cells=tuple(cells),
                )
                for cells in marker_resolution.placements
            )
            if marker_resolution.placements:
                logger.info(
                    "level %s red completion markers resolved without sidebar: placements=%s",
                    level,
                    [list(cells) for cells in marker_resolution.placements],
                )

        initial_visual_hits = (
            partial_cells
            | completed_visual_hits
            | (
                (set(visible_hits) - partial_cells - completed_visual_hits)
                if sidebar_progress is None
                else set()
            )
        )
        fleet_visual_hits: set[Cell] = set()
        # Ordinary wreck/hit pixels are not sufficient evidence of a complete
        # submarine.  A straight run that happens to match a fleet length
        # must stay ``hit`` unless the sidebar or a red submarine component
        # independently confirms completion.  Otherwise four adjacent wrecks
        # (such as level-10 row 6, columns 2-5) can be painted green.
        if visible_hits and (sidebar_progress is not None or completed_anchor_candidates):
            fleet_resolution = resolve_completed_ship_cells(
                set(visible_hits),
                submarines,
                grid_size=grid_size,
            )
            fleet_visual_hits = set(fleet_resolution.cells)
            if fleet_visual_hits:
                logger.info(
                    "level %s visible fleet geometry: placements=%s unresolved=%s discarded=%s",
                    level,
                    [list(placement) for placement in fleet_resolution.placements],
                    list(fleet_resolution.unresolved_lengths),
                    sorted(fleet_resolution.discarded_cells),
                )
            if len(fleet_visual_hits) > len(initial_visual_hits):
                logger.info(
                    "level %s visible fleet geometry restored %s additional hit cells",
                    level,
                    len(fleet_visual_hits - initial_visual_hits),
                )
                initial_visual_hits = fleet_visual_hits
        elif visible_hits:
            logger.info(
                "level %s keeping %s ordinary visible hits provisional; "
                "no sidebar/red-marker completion evidence",
                level,
                len(visible_hits),
            )

        max_visible_hits = sum(submarines)
        if len(initial_visual_hits) > max_visible_hits:
            logger.warning(
                "level %s visual hit coordinates are suspicious: %s/%s; "
                "falling back to geometry-constrained coordinates",
                level,
                len(initial_visual_hits),
                max_visible_hits,
            )
            if 0 < len(fleet_visual_hits) <= max_visible_hits:
                initial_visual_hits = fleet_visual_hits
            else:
                initial_visual_hits = set(completed_visual_hits)

        for row, col in initial_visual_hits:
            hit_map[row][col] = 1

        if sidebar_progress is not None and partial_wreck_cells is not None:
            initial_visual_hit_count = calculate_visible_hit_count(
                sidebar_progress,
                partial_wreck_count=len(partial_wreck_cells),
                fallback_hit_count=len(visible_hits),
            )
            initial_visual_hit_count = max(initial_visual_hit_count, len(initial_visual_hits))
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
            initial_red_marker_completed_cells=red_marker_completed_cells,
            initial_authoritative_completed_visual_hits=(
                completed_visual_hits
                if (
                    sidebar_progress is not None
                    and sidebar_progress.valid
                )
                or authoritative_completed_placements
                else set()
            ),
            initial_authoritative_completed_placements=(
                authoritative_completed_placements
                if authoritative_completed_placements
                else ()
            ),
            initial_completed_lengths=(
                sidebar_progress.completed_lengths
                if sidebar_progress is not None and sidebar_progress.valid
                else marker_completed_lengths
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
    initial_misses: set[Cell] | None = None,
    initial_sidebar_progress: SidebarProgress | None = None,
    initial_visual_hit_count: int | None = None,
    initial_completed_visual_hits: set[Cell] | None = None,
    initial_red_marker_completed_cells: set[Cell] | None = None,
    initial_authoritative_completed_visual_hits: set[Cell] | None = None,
    initial_authoritative_completed_placements: Sequence[Placement | Sequence[Cell]] | None = None,
    initial_completed_lengths: Sequence[int] | None = None,
    initial_scout_hits: set[Cell] | None = None,
    initial_scout_misses: set[Cell] | None = None,
    commit_scout_hits_online: bool = False,
) -> bool:
    """使用潜艇策略选择探测格；策略无法完成时回退扫描剩余格。"""
    grid_size = get_level_grid_size(level)
    strategy = SubmarineStrategy(grid_size, submarines)
    authoritative_placements = tuple(initial_authoritative_completed_placements or ())
    restore_placements = getattr(strategy, "restore_confirmed_placements", None)
    if authoritative_placements and callable(restore_placements):
        restored = restore_placements(authoritative_placements)
        if restored:
            logger.info(
                "level %s restored authoritative completed placements: %s",
                level,
                [list(ship.cells) for ship in restored],
            )
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

    real_initial_hits = set(initial_hits or set())
    # Completed cells confirmed by the initial sidebar/geometry pass are
    # carried through the final scan as durable facts.  They remain included
    # in the normal hit set for strategy accounting, but are kept separately
    # so later visual observations cannot downgrade them.
    if initial_authoritative_completed_visual_hits:
        real_initial_hits.update(initial_authoritative_completed_visual_hits)
    for placement in authoritative_placements:
        if isinstance(placement, Placement):
            real_initial_hits.update(placement.cells)
        else:
            real_initial_hits.update(tuple(tuple(cell) for cell in placement))
    real_initial_misses = set(initial_misses or set()) - real_initial_hits
    for cell in real_initial_hits:
        if cell not in strategy.shots:
            strategy.report_result(cell, True)
    for cell in real_initial_misses:
        if cell not in strategy.shots:
            strategy.report_result(cell, False)
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
    # Only an explicitly supplied visual count can represent occupied cells
    # without coordinates. When the count is synthesized from strategy state,
    # subtracting it again from batch capacity would double-count known hits.
    unmapped_visual_hits = (
        max(0, initial_display_hit_cells - mapped_visual_hits)
        if initial_visual_hit_count is not None
        else 0
    )
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
        supplemental_rechecks_done=0,
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

        supplemental_rechecked: set[Cell] = set()
        supplemental_attempts = 0

        def run_supplemental_neighbor_rechecks() -> bool:
            nonlocal attempts, supplemental_attempts

            getter = getattr(
                strategy,
                "get_priority_scout_miss_recheck_targets",
                None,
            )
            if not callable(getter):
                getter = getattr(
                    strategy,
                    "get_isolated_hit_scout_miss_neighbors_for_recheck",
                    None,
                )
            if not callable(getter):
                return False

            while attempts < max_attempts:
                candidates = [
                    cell
                    for cell in getter(supplemental_rechecked)
                    if cell not in supplemental_rechecked
                ]
                if not candidates:
                    return False

                for cell in candidates:
                    if cell in supplemental_rechecked or attempts >= max_attempts:
                        continue
                    supplemental_rechecked.add(cell)
                    supplemental_attempts += 1
                    row, col = cell
                    index = row * grid_size + col
                    logger.info(
                        "level %s high-priority scout-miss recheck from hit evidence: "
                        "cell=%s index=%s",
                        level,
                        cell,
                        index,
                    )
                    current_hit_cells = sum(
                        1 for shot_hit in strategy.shots.values() if shot_hit
                    )
                    current_display_hit_cells = progressive_hit_count(
                        initial_visual_hit_count=initial_display_hit_cells,
                        initial_strategy_hit_count=initial_hit_cells,
                        current_strategy_hit_count=current_hit_cells,
                    )
                    write_runtime_status(
                        phase="supplemental_recheck",
                        level=level,
                        current_cell=index,
                        shots_done=len(strategy.shots),
                        total_cells=grid_size * grid_size,
                        hits=min(sum(submarines), current_display_hit_cells),
                        total_ship_cells=sum(submarines),
                        supplemental_rechecks_done=supplemental_attempts,
                        board_size=grid_size,
                        board_states=build_runtime_board_states(strategy, grid_size),
                        last_result="supplemental_recheck_pending",
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
                            supplemental_rechecks_done=supplemental_attempts,
                            last_result=probe_result.value,
                        )
                        return True

                    attempts += 1
                    if hit:
                        strategy.blocked_cells.discard(cell)
                    strategy.report_result(cell, hit)
                    if hit:
                        hit_map[row][col] = 1
                    else:
                        logger.info(
                            "level %s priority scout-miss recheck cell=%s result=%s",
                            level,
                            cell,
                            probe_result.value,
                        )

                    newly_completed_lengths = tuple(
                        int(length)
                        for length in probe_metadata.get("sidebar_newly_completed_lengths", ())
                    )
                    sidebar_completed_lengths = tuple(
                        int(length)
                        for length in probe_metadata.get("sidebar_completed_lengths", ())
                    )
                    if not sidebar_completed_lengths and newly_completed_lengths:
                        sidebar_completed_lengths = (
                            tuple(accounted_completed_lengths()) + newly_completed_lengths
                        )
                    reconcile = getattr(strategy, "reconcile_completed_lengths", None)
                    if hit and sidebar_completed_lengths and callable(reconcile):
                        trusted_completed_cells = _trusted_completed_cells_from_probe_metadata(
                            probe_metadata,
                            click_points,
                            grid_size=grid_size,
                            anchor=cell,
                        )
                        reconcile(
                            sidebar_completed_lengths,
                            anchor=cell,
                            observed_completed_cells=trusted_completed_cells,
                        )

                    save_level_shots(level, grid_size, strategy.shots)
                    confirmed_lengths = accounted_completed_lengths()
                    hit_cells = sum(1 for shot_hit in strategy.shots.values() if shot_hit)
                    display_hit_cells = progressive_hit_count(
                        initial_visual_hit_count=initial_display_hit_cells,
                        initial_strategy_hit_count=initial_hit_cells,
                        current_strategy_hit_count=hit_cells,
                    )
                    display_hit_cells = min(sum(submarines), display_hit_cells)
                    write_runtime_status(
                        phase=(
                            "level_complete"
                            if level_completed
                            else "supplemental_recheck"
                        ),
                        level=level,
                        current_cell="--" if level_completed else index,
                        shots_done=len(strategy.shots),
                        total_cells=grid_size * grid_size,
                        hits=display_hit_cells,
                        total_ship_cells=sum(submarines),
                        supplemental_rechecks_done=supplemental_attempts,
                        confirmed_ships=len(confirmed_lengths),
                        total_ships=len(submarines),
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
                        return True
            return False

        def apply_strategy_probe_result(
            cell: Cell,
            index: int,
            probe_result: ProbeResult,
            probe_metadata: Mapping[str, object],
            *,
            direct_scout_hit: bool,
        ) -> bool:
            nonlocal attempts

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
            row, col = cell
            if hit:
                hit_map[row][col] = 1

            newly_completed_lengths = tuple(
                int(length)
                for length in probe_metadata.get("sidebar_newly_completed_lengths", ())
            )
            sidebar_completed_lengths = tuple(
                int(length)
                for length in probe_metadata.get("sidebar_completed_lengths", ())
            )
            if not sidebar_completed_lengths and newly_completed_lengths:
                sidebar_completed_lengths = (
                    tuple(accounted_completed_lengths()) + newly_completed_lengths
                )
            if hit and sidebar_completed_lengths:
                trusted_completed_cells = _trusted_completed_cells_from_probe_metadata(
                    probe_metadata,
                    click_points,
                    grid_size=grid_size,
                    anchor=cell,
                )
                located, unlocated = strategy.reconcile_completed_lengths(
                    sidebar_completed_lengths,
                    anchor=cell,
                    observed_completed_cells=trusted_completed_cells,
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
            hit_cells = sum(1 for shot_hit in strategy.shots.values() if shot_hit)
            display_hit_cells = progressive_hit_count(
                initial_visual_hit_count=initial_display_hit_cells,
                initial_strategy_hit_count=initial_hit_cells,
                current_strategy_hit_count=hit_cells,
            )
            display_hit_cells = min(sum(submarines), display_hit_cells)
            write_runtime_status(
                phase=(
                    "level_complete"
                    if level_completed
                    else "blue_online_scout_hits"
                    if direct_scout_hit
                    else "strategy_scan"
                ),
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
            return False

        while not strategy.done and attempts < max_attempts:
            cell = strategy.choose_next_cell()
            if cell is None:
                # Unknown cells are the normal-search priority. Only once the
                # strategy has exhausted them do we spend blue bombs verifying
                # red-scout misses that are adjacent to hit evidence.
                if run_supplemental_neighbor_rechecks():
                    return True
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
            direct_scout_hit = (
                commit_scout_hits_online
                and cell in strategy.get_scout_hit_cells()
            )
            if direct_scout_hit:
                write_runtime_status(
                    phase="blue_online_scout_hits",
                    level=level,
                    current_cell=index,
                )
                pending_scout_cells = sorted(strategy.get_scout_hit_cells())
                batch_points = [
                    click_points[item[0] * grid_size + item[1]]
                    for item in pending_scout_cells
                ]
                known_strategy_hits = {
                    (row, col)
                    for row, values in enumerate(hit_map)
                    for col, value in enumerate(values)
                    if bool(value)
                }
                known_strategy_hits.update(
                    cell
                    for cell, shot_hit in strategy.shots.items()
                    if bool(shot_hit)
                )
                batch_capacity = max(
                    0,
                    sum(int(length) for length in submarines)
                    - len(known_strategy_hits)
                    - unmapped_visual_hits,
                )
                use_scout_batch = bool(
                    ONLINE_SCOUT_BATCH_ENABLED
                    and len(pending_scout_cells) > 1
                    and len(set(batch_points)) == len(batch_points)
                    and len(pending_scout_cells) <= batch_capacity
                )
                if use_scout_batch:
                    batch_outcome = _execute_online_scout_hit_batch(
                        level=level,
                        hit_map=hit_map,
                        targets=[
                            (
                                item,
                                click_points[item[0] * grid_size + item[1]],
                                item[0] * grid_size + item[1],
                            )
                            for item in pending_scout_cells
                        ],
                        submarines=submarines,
                        activity_ready=True,
                        unmapped_visual_hits=unmapped_visual_hits,
                    )
                    for batch_cell in pending_scout_cells:
                        batch_metadata = dict(batch_outcome.metadata.get(batch_cell, {}))
                        batch_result = batch_outcome.results.get(
                            batch_cell,
                            ProbeResult.UNKNOWN,
                        )
                        if batch_result is ProbeResult.UNKNOWN:
                            raise ProbeProtocolError(
                                f"online scout-hit batch result for cell {batch_cell} is unknown; "
                                "refusing to retry it"
                            )
                        if apply_strategy_probe_result(
                            batch_cell,
                            batch_cell[0] * grid_size + batch_cell[1],
                            batch_result,
                            batch_metadata,
                            direct_scout_hit=True,
                        ):
                            return True
                    continue
                probe_result = _execute_online_scout_hit(
                    level=level,
                    hit_map=hit_map,
                    cell=cell,
                    point=click_points[index],
                    click_points=click_points,
                    index=index,
                    submarines=submarines,
                    probe_metadata=probe_metadata,
                )
            else:
                probe_result = _probe_cell(
                    level,
                    hit_map,
                    cell,
                    click_points[index],
                    index,
                    probe_metadata=probe_metadata,
                )
            if apply_strategy_probe_result(
                cell,
                index,
                probe_result,
                probe_metadata,
                direct_scout_hit=direct_scout_hit,
            ):
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
            hit_cells = sum(1 for shot_hit in strategy.shots.values() if shot_hit)
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
                trusted_completed_cells = _trusted_completed_cells_from_probe_metadata(
                    probe_metadata,
                    click_points,
                    grid_size=grid_size,
                    anchor=cell,
                )
                located, unlocated = strategy.reconcile_completed_lengths(
                    completed_lengths,
                    anchor=cell,
                    observed_completed_cells=trusted_completed_cells,
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
    scout_hits: set[Cell] = set()
    scout_misses: set[Cell] = set()
    committed_hits: set[Cell] = set()
    committed_misses: set[Cell] = set()
    direct_attempted_cells: set[Cell] = set()
    # Red-scout geometry can identify the whole hull before a blue request has
    # committed any cell in it.  Keep that provenance separate so monitoring
    # does not render inferred, not-yet-bombed cells as authoritative ships.
    visual_only_completed_cells: set[Cell] = set()
    initial_real_hits = set(initial_hits)
    initial_misses = set(scan_kwargs.get("initial_misses") or set())
    attempted_centers: set[Cell] = set()
    attempts_completed = 0
    valid_attempts = 0
    complete_six_attempts = 0
    online_sidebar_completed_lengths: tuple[int, ...] = ()
    online_completed_visual_hits = set(
        scan_kwargs.get("initial_completed_visual_hits") or set()
    )
    initial_completed_lengths = tuple(
        int(length)
        for length in (scan_kwargs.get("initial_completed_lengths") or ())
    )
    initial_completed_visual_hits = set(
        scan_kwargs.get("initial_completed_visual_hits") or set()
    )
    visual_only_completed_cells.update(
        initial_completed_visual_hits - initial_real_hits
    )
    raw_initial_visual_count = scan_kwargs.get("initial_visual_hit_count")
    if raw_initial_visual_count is None:
        unmapped_initial_visual_hits = 0
    else:
        try:
            initial_visual_count = max(0, int(raw_initial_visual_count))
        except (TypeError, ValueError):
            initial_visual_count = 0
        mapped_initial_visual_hits = len(
            initial_real_hits | initial_completed_visual_hits
        )
        unmapped_initial_visual_hits = max(
            0,
            initial_visual_count - mapped_initial_visual_hits,
        )
        if unmapped_initial_visual_hits:
            logger.warning(
                "level %s has %s initial visual submarine cells without coordinates; "
                "reserving that capacity before any blue batch",
                level,
                unmapped_initial_visual_hits,
            )
    online_hit_evidence: dict[Cell, Mapping[str, object]] = {}
    authoritative_completed_visual_hits: set[Cell] = set(
        scan_kwargs.get("initial_authoritative_completed_visual_hits") or set()
    )
    # Cells whose complete-submarine state was established by the red
    # component are immutable facts.  Keep this provenance separate so later
    # noisy visual frames cannot turn them into ordinary hits.
    red_marker_completed_cells: set[Cell] = set(
        scan_kwargs.get("initial_red_marker_completed_cells") or set()
    )
    authoritative_completed_placements: list[Placement] = []
    for raw_placement in scan_kwargs.get(
        "initial_authoritative_completed_placements"
    ) or ():
        if isinstance(raw_placement, Placement):
            placement = raw_placement
        else:
            cells = tuple(tuple(cell) for cell in raw_placement)
            if not cells:
                continue
            placement = Placement(
                length=len(cells),
                direction=(
                    "H"
                    if len({row for row, _ in cells}) == 1
                    else "V"
                ),
                cells=cells,
            )
        if placement.cells not in {item.cells for item in authoritative_completed_placements}:
            authoritative_completed_placements.append(placement)
            authoritative_completed_visual_hits.update(placement.cells)
    discarded_flag_cells: set[Cell] = set()
    # Once a legal, contiguous completed-submarine placement is confirmed,
    # its cells are immutable for the remainder of this level.  Keep this
    # separate from provisional visual-hit sets so later noisy frames or L
    # shape cleanup cannot downgrade a confirmed ship cell.
    # Initial visual placements remain provisional until the current level's
    # red-scout/geometry pass confirms them; this allows the explicit L-shape
    # flag correction to remove a false upper cell before locking a ship.
    locked_completed_ship_cells: set[Cell] = set()
    # A red marker proves the complete hull geometry, but cells without an
    # existing wreck or committed blue hit still need a real blue tap.  Keep
    # those cells out of the monotonic hit restoration until the tap is
    # visibly confirmed.
    pending_completed_ship_cells: set[Cell] = set()
    # A confirmed complete submarine guarantees that every neighboring cell
    # is water.  Keep that perimeter as a durable invariant so red-scout
    # animation noise or a later visual snapshot cannot turn it into a hit.
    completed_ship_safety_cells: set[Cell] = set()

    def completed_placement_safety_area(placement: Placement) -> set[Cell]:
        ship_cells = set(placement.cells)
        safety: set[Cell] = set()
        for row, col in ship_cells:
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    neighbor = (row + row_offset, col + col_offset)
                    if (
                        neighbor not in ship_cells
                        and 0 <= neighbor[0] < grid_size
                        and 0 <= neighbor[1] < grid_size
                    ):
                        safety.add(neighbor)
        return safety

    def enforce_completed_ship_safety_area() -> None:
        if not completed_ship_safety_cells:
            return
        conflicting_hits = completed_ship_safety_cells & (
            initial_real_hits
            | committed_hits
            | scout_hits
            | initial_completed_visual_hits
            | online_completed_visual_hits
            | authoritative_completed_visual_hits
            | red_marker_completed_cells
        )
        if conflicting_hits:
            logger.warning(
                "clearing impossible hit evidence from completed-submarine "
                "safety area: %s",
                sorted(conflicting_hits),
            )
        for hit_set in (
            initial_real_hits,
            committed_hits,
            scout_hits,
            initial_completed_visual_hits,
            online_completed_visual_hits,
            authoritative_completed_visual_hits,
            red_marker_completed_cells,
        ):
            hit_set.difference_update(completed_ship_safety_cells)
        pending_completed_ship_cells.difference_update(
            completed_ship_safety_cells
        )
        visual_only_completed_cells.difference_update(
            completed_ship_safety_cells
        )
        # The perimeter is stronger than a visual miss: the game rules prove
        # it is safe once the completed placement is known.  Persist it as a
        # normal miss so the red planner, blue strategy and final scan all
        # refuse to target it.
        initial_misses.update(completed_ship_safety_cells)
        committed_misses.update(completed_ship_safety_cells)
        scout_misses.difference_update(completed_ship_safety_cells)
        for row, col in completed_ship_safety_cells:
            if 0 <= row < len(hit_map) and 0 <= col < len(hit_map[row]):
                hit_map[row][col] = 0

    def refresh_completed_ship_safety_area() -> None:
        all_ship_cells = {
            cell
            for placement in authoritative_completed_placements
            for cell in placement.cells
        }
        safety = {
            cell
            for placement in authoritative_completed_placements
            for cell in completed_placement_safety_area(placement)
        }
        # This subtraction is defensive.  Legal completed placements never
        # touch, but a ship cell must not be erased if conflicting visual
        # geometry briefly reaches this stage.
        safety.difference_update(all_ship_cells)
        completed_ship_safety_cells.clear()
        completed_ship_safety_cells.update(safety)
        enforce_completed_ship_safety_area()

    def restore_locked_completed_ship_cells() -> None:
        enforce_completed_ship_safety_area()
        if not locked_completed_ship_cells:
            return
        confirmed_locked_cells = (
            locked_completed_ship_cells - pending_completed_ship_cells
        )
        for hit_set in (
            initial_real_hits,
            committed_hits,
            initial_completed_visual_hits,
            online_completed_visual_hits,
            authoritative_completed_visual_hits,
            red_marker_completed_cells,
        ):
            hit_set.update(confirmed_locked_cells)
        for miss_set in (initial_misses, committed_misses, scout_misses):
            miss_set.difference_update(locked_completed_ship_cells)
        discarded_flag_cells.difference_update(locked_completed_ship_cells)
        visual_only_completed_cells.difference_update(confirmed_locked_cells)
        visual_only_completed_cells.update(pending_completed_ship_cells)
        scout_hits.update(pending_completed_ship_cells)
        for row, col in confirmed_locked_cells:
            if 0 <= row < len(hit_map) and 0 <= col < len(hit_map[row]):
                hit_map[row][col] = 1

    def prune_discarded_authoritative_placements() -> None:
        """Drop a locked placement if one of its cells is proven flag noise.

        The 2x2 L rule is allowed to override every completion source.  A
        placement that contains the removed upper cell is therefore no longer
        safe to restore as a whole; the surviving lower evidence can be
        resolved again against the fleet lengths on the next pass.
        """
        if not discarded_flag_cells or not authoritative_completed_placements:
            return
        kept = [
            placement
            for placement in authoritative_completed_placements
            if (
                set(placement.cells) <= locked_completed_ship_cells
                or not (set(placement.cells) & discarded_flag_cells)
            )
        ]
        if len(kept) != len(authoritative_completed_placements):
            logger.warning(
                "discarding authoritative completed placement touched by a flag artifact: %s",
                [
                    list(placement.cells)
                    for placement in authoritative_completed_placements
                    if set(placement.cells) & discarded_flag_cells
                ],
            )
            authoritative_completed_placements[:] = kept
            refresh_completed_ship_safety_area()

    blue_bomb_ready = False
    # _execute_red_scout_transaction restores connectivity before returning, so
    # all blue targets produced by the same red scout share this online state.
    online_network_ready = False

    def normalize_flag_overlap_state(
        scope: set[Cell] | frozenset[Cell] | None = None,
        *,
        snapshot_cells: set[Cell] | frozenset[Cell] | None = None,
    ) -> None:
        """Remove a flag artifact while keeping red-scout snapshots isolated.

        The red result can expose the false upper cell as an already-known hit
        (or as a completed visual cell) while reporting only the lower ship
        cells as new hits.  ``snapshot_cells`` identifies that same result
        footprint.  A candidate from an older, unrelated result is ignored
        unless its upper cell is present in this footprint; this preserves the
        cross-frame protection for real committed hits.
        """
        current_scope = set(scope or ())
        current_snapshot = (
            set(snapshot_cells)
            if snapshot_cells is not None
            else set(current_scope)
        )
        ignored_false_cells = set(discarded_flag_cells)
        visual_evidence = (
            initial_completed_visual_hits
            | online_completed_visual_hits
            | authoritative_completed_visual_hits
        )
        historical_evidence = initial_real_hits | committed_hits | scout_hits
        while True:
            if scope is not None:
                snapshot_evidence = historical_evidence & current_snapshot
                snapshot_evidence |= visual_evidence & current_snapshot
                hit_union = set(current_scope) | snapshot_evidence | visual_evidence
            else:
                hit_union = historical_evidence | visual_evidence
            match = _find_flag_overlap_l_shape(
                hit_union,
                ignored_false_cells=ignored_false_cells,
            )
            if match is None:
                return
            upper, lower_pair = match
            # A supported 2x2 L is the explicit raised-flag exception to the
            # normal monotonic rule.  This exception intentionally applies to
            # every source of evidence, including a red-component-backed
            # completed cell: in a 2x2 L the upper cell is the raised flag and
            # must be removed before blue targeting.
            if scope is not None:
                # A lower hit from this result may correct an upper cell that
                # was already known in the same red footprint.  If neither
                # side is present in that footprint, this is a cross-frame
                # coincidence and must remain untouched.
                candidate = {upper} | set(lower_pair)
                if not (
                    candidate.issubset(current_snapshot)
                    or candidate.issubset(visual_evidence)
                    or candidate.issubset(current_scope)
                ):
                    ignored_false_cells.add(upper)
                    continue
            upper_was_visual = upper in (
                initial_completed_visual_hits
                | online_completed_visual_hits
                | authoritative_completed_visual_hits
            )
            upper_was_committed = upper in (initial_real_hits | committed_hits)
            for hit_set in (
                initial_real_hits,
                committed_hits,
                scout_hits,
                initial_completed_visual_hits,
                online_completed_visual_hits,
                authoritative_completed_visual_hits,
                red_marker_completed_cells,
            ):
                hit_set.discard(upper)
            # ``scope`` can be an immutable snapshot; remember removals so the
            # same artifact is not rediscovered on the next iteration.
            ignored_false_cells.add(upper)
            discarded_flag_cells.add(upper)
            visual_only_completed_cells.discard(upper)
            if upper_was_visual:
                online_completed_visual_hits.update(lower_pair)
                authoritative_completed_visual_hits.update(lower_pair)
            if upper_was_committed:
                committed_misses.add(upper)
                initial_misses.add(upper)
            else:
                scout_misses.add(upper)
                # Keep the discarded flag out of the later blue strategy too;
                # it is known visual noise, not an unexplored cell.
                initial_misses.add(upper)
            if 0 <= upper[0] < len(hit_map) and 0 <= upper[1] < len(hit_map[upper[0]]):
                hit_map[upper[0]][upper[1]] = 0
            logger.info(
                "normalizing L-shaped submarine evidence: removing upper cell %s; "
                "keeping lower pair %s (L-shape rule overrides completion lock)",
                upper,
                sorted(lower_pair),
            )

    def normalize_current_2x2_l_shape(
        result_cells: set[Cell] | frozenset[Cell],
        *,
        snapshot_cells: set[Cell] | frozenset[Cell] | None = None,
    ) -> None:
        """Force-remove an upper flag cell for either 2x2 L orientation.

        Known cells are included so a red footprint can correct an upper cell
        that was classified before this attempt.  The footprint guard prevents
        unrelated results from being joined into a false L.
        """
        current_cells = set(result_cells)
        current_snapshot = (
            set(snapshot_cells)
            if snapshot_cells is not None
            else set(current_cells)
        )
        visual_evidence = (
            initial_completed_visual_hits
            | online_completed_visual_hits
            | authoritative_completed_visual_hits
        )
        historical_evidence = initial_real_hits | committed_hits | scout_hits
        # A red result may omit cells that were already known, but its affected
        # footprint tells us which historical cells came from the same snapshot.
        # Keep a complete visual snapshot available for the legacy case where
        # the analyzer reports only the newly changed lower cells. Initial hits
        # need the same exception because the analyzer excludes known cells;
        # later committed hits deliberately remain snapshot-scoped.
        evidence = (
            current_cells
            | (historical_evidence & current_snapshot)
            | (visual_evidence & current_snapshot)
            | visual_evidence
            | initial_real_hits
        )
        def find_supported_block() -> tuple[tuple[Cell, ...], Cell] | None:
            candidate_origins = {
                (row + row_offset, col + col_offset)
                for row, col in evidence
                for row_offset in (-1, 0)
                for col_offset in (-1, 0)
            }
            for row, col in sorted(candidate_origins):
                block = {
                    (row, col),
                    (row, col + 1),
                    (row + 1, col),
                    (row + 1, col + 1),
                }
                occupied = block & evidence
                if len(occupied) != 3 or not (occupied & current_cells):
                    continue
                block_tuple = tuple(sorted(occupied))
                upper = _resolve_false_hit_in_l_shape(block_tuple, {})
                if upper is None:
                    continue
                # A raised red flag is an explicit exception to the normal
                # completed-cell lock.  The L-shape rule has priority for all
                # evidence sources, including red-component-backed completion.
                # Only a cell from the initial visual snapshot or the current
                # red result may be corrected.  A cell learned by an earlier
                # scout/blue shot is historical evidence and must not be
                # joined with this result to manufacture an L shape.
                upper_is_visual = upper in initial_completed_visual_hits
                upper_is_initial_hit = upper in initial_real_hits
                upper_is_scout_hit = upper in scout_hits
                lower_cells = set(block_tuple) - {upper}
                lower_cells_from_current = lower_cells.issubset(current_cells)
                if (
                    upper not in current_cells
                    and upper not in current_snapshot
                    and not (
                        (
                            upper_is_visual
                            and (
                                set(block_tuple).issubset(initial_completed_visual_hits)
                                or lower_cells.issubset(current_cells)
                                # A completed visual cell can be the raised
                                # upper flag while the aligned lower pair is
                                # split between this scout result and a
                                # previously committed/visual cell.  This is
                                # still a local, supported 2x2 shape because
                                # the current snapshot contributes at least
                                # one of the lower cells.
                                or (
                                    bool(lower_cells & current_cells)
                                    and lower_cells.issubset(
                                        visual_evidence | historical_evidence
                                    )
                                )
                            )
                        )
                        or (
                            upper_is_initial_hit
                            and lower_cells_from_current
                        )
                        or (
                            upper_is_scout_hit
                            and lower_cells.issubset(
                                visual_evidence | current_cells
                            )
                        )
                    )
                ):
                    continue
                # A later red-scout result must not manufacture an L by
                # combining two fresh lower hits with an upper cell from a
                # previously locked completed placement.  The L exception is
                # still honored when the current snapshot explicitly contains
                # that upper cell (or during the initial-board normalization).
                # Without this provenance boundary, a real placement such as
                # ``[(3, 6), (4, 6)]`` can be discarded repeatedly when a later
                # scout reports only ``[(4, 6), (4, 7)]``.
                if (
                    upper not in current_snapshot
                    and any(
                        upper in placement.cells
                        for placement in authoritative_completed_placements
                    )
                ):
                    continue
                if upper not in current_cells and not (
                    (set(block_tuple) - {upper}) & current_cells
                ):
                    continue
                return block_tuple, upper
            return None

        while True:
            supported = find_supported_block()
            if supported is None:
                return
            block, upper = supported
            lower = set(block) - {upper}
            upper_was_visual = upper in visual_evidence
            upper_was_known = upper in (initial_real_hits | committed_hits)
            for hit_set in (
                initial_real_hits,
                committed_hits,
                scout_hits,
                initial_completed_visual_hits,
                online_completed_visual_hits,
                authoritative_completed_visual_hits,
                red_marker_completed_cells,
            ):
                hit_set.discard(upper)
            if upper_was_visual:
                initial_completed_visual_hits.update(lower)
                online_completed_visual_hits.update(lower)
                authoritative_completed_visual_hits.update(lower)
            committed_misses.discard(upper)
            initial_misses.add(upper)
            if not upper_was_known:
                scout_misses.add(upper)
            else:
                scout_misses.discard(upper)
            if 0 <= upper[0] < len(hit_map) and 0 <= upper[1] < len(hit_map[upper[0]]):
                hit_map[upper[0]][upper[1]] = 0
            discarded_flag_cells.add(upper)
            visual_only_completed_cells.discard(upper)
            evidence.discard(upper)
            evidence.update(lower)
            logger.info(
                "normalizing 2x2 L-shaped submarine evidence before blue attack: "
                "removing upper cell %s; keeping lower cells %s "
                "(L-shape rule overrides completion lock)",
                upper,
                sorted(lower),
            )

    # Initial visual ships are already geometry-resolved, but a raised flag can
    # still add one false upper cell to an otherwise valid 2x2 L.  Normalize
    # that shape before choosing the first red-scout center or any blue target.
    # The helper only removes a cell when the remaining two cells form a
    # straight adjacent pair, so isolated visual hits and unrelated ships are
    # left untouched.
    initial_geometry_scope = (
        initial_real_hits
        | initial_completed_visual_hits
        | online_completed_visual_hits
        | authoritative_completed_visual_hits
        | red_marker_completed_cells
    )
    if initial_geometry_scope:
        normalize_current_2x2_l_shape(
            initial_geometry_scope,
            snapshot_cells=initial_geometry_scope,
        )
        restore_locked_completed_ship_cells()
        prune_discarded_authoritative_placements()
    refresh_completed_ship_safety_area()

    # Initial completed geometry has already been resolved from one board
    # snapshot.  Re-running the L heuristic across all resolved ships can join
    # unrelated submarines, so only new red-scout snapshots are normalized.

    def retain_authoritative_completed_placements(
        candidate_cells: set[Cell] | frozenset[Cell],
        completed_lengths: Sequence[int],
    ) -> None:
        """Add only new, non-conflicting complete placements to the durable lock.

        A later animation frame can expose only a prefix of a ship (for
        example, a confirmed length-5 line may be resolved as length 4).  Such
        a candidate overlaps an existing placement and is intentionally
        rejected.  Disjoint placements are still accepted, including repeated
        fleet lengths, subject to the sidebar's per-length count.
        """
        if not candidate_cells or not completed_lengths:
            return
        limits = Counter(
            int(length)
            for length in completed_lengths
            if int(length) > 0
        )
        existing_cells = {
            cell
            for placement in authoritative_completed_placements
            for cell in placement.cells
        }
        existing_keys = {
            placement.cells for placement in authoritative_completed_placements
        }
        existing_counts = Counter(
            placement.length for placement in authoritative_completed_placements
        )
        remaining_lengths = list(limits.elements())
        for placement in authoritative_completed_placements:
            try:
                remaining_lengths.remove(placement.length)
            except ValueError:
                continue
        if not remaining_lengths:
            return
        remaining_candidates = set(candidate_cells) - existing_cells
        if not remaining_candidates:
            return
        resolution = resolve_completed_ship_cells(
            remaining_candidates,
            remaining_lengths,
            grid_size=grid_size,
            preferred_cells=remaining_candidates,
        )

        for cells in resolution.placements:
            normalized_cells = tuple(cells)
            if normalized_cells in existing_keys:
                continue
            length = len(normalized_cells)
            if existing_counts[length] >= limits.get(length, 0):
                continue
            overlap = set(normalized_cells) & existing_cells
            if overlap:
                logger.warning(
                    "rejecting conflicting completed placement %s; "
                    "authoritative cells=%s overlap=%s",
                    list(normalized_cells),
                    [list(item.cells) for item in authoritative_completed_placements],
                    sorted(overlap),
                )
                continue
            placement = Placement(
                length=length,
                direction=(
                    "H"
                    if len({row for row, _ in normalized_cells}) == 1
                    else "V"
                ),
                cells=normalized_cells,
            )
            authoritative_completed_placements.append(placement)
            existing_keys.add(placement.cells)
            existing_cells.update(placement.cells)
            existing_counts[length] += 1
            # A placement resolved from an online completion snapshot is just
            # as authoritative as one resolved directly from the red scout.
            # Lock every hull cell immediately so later animation frames or
            # stale probe results cannot downgrade it to miss/unknown.
            locked_completed_ship_cells.update(normalized_cells)
            authoritative_completed_visual_hits.update(placement.cells)
            logger.info(
                "locking completed submarine placement: length=%s cells=%s",
                length,
                list(placement.cells),
            )
        refresh_completed_ship_safety_area()

    def retain_red_scout_completed_diagnostics(result: RedScoutResult) -> None:
        """Commit sidebar-backed ship geometry learned by the red result.

        The red analyzer can prove completion before a new blue target is
        available.  In particular, every cell may already have been committed
        by earlier scouts.  Preserve the resolved placement directly instead
        of waiting for another blue probe to copy the same sidebar evidence
        into the durable state.
        """
        nonlocal online_sidebar_completed_lengths
        diagnostics = result.diagnostics
        if not result.valid or not isinstance(diagnostics, Mapping):
            return
        if diagnostics.get("completed_ship_failure") is not None:
            return
        try:
            completed_lengths = tuple(
                int(length)
                for length in diagnostics.get("completed_lengths", ())
                if int(length) > 0
            )
        except (TypeError, ValueError):
            return
        raw_placements = diagnostics.get("resolved_ship_placements", ())
        if not completed_lengths or not isinstance(raw_placements, Sequence):
            return

        allowed_lengths = Counter(completed_lengths)
        accepted_cells: set[Cell] = set()
        existing_keys = {placement.cells for placement in authoritative_completed_placements}
        existing_cells = {
            cell
            for placement in authoritative_completed_placements
            for cell in placement.cells
        }
        for raw_cells in raw_placements:
            try:
                cells = tuple(sorted(tuple(cell) for cell in raw_cells))
            except (TypeError, ValueError):
                continue
            length = len(cells)
            if not cells or allowed_lengths[length] <= 0:
                continue
            rows = {row for row, _col in cells}
            cols = {col for _row, col in cells}
            horizontal = len(rows) == 1 and [col for _row, col in cells] == list(
                range(cells[0][1], cells[0][1] + length)
            )
            vertical = len(cols) == 1 and [row for row, _col in cells] == list(
                range(cells[0][0], cells[0][0] + length)
            )
            if not (horizontal or vertical):
                continue
            if any(not (0 <= row < grid_size and 0 <= col < grid_size) for row, col in cells):
                continue
            if set(cells) & discarded_flag_cells:
                continue
            if cells not in existing_keys and set(cells) & existing_cells:
                continue
            allowed_lengths[length] -= 1
            accepted_cells.update(cells)
            if cells in existing_keys:
                continue
            placement = Placement(
                length=length,
                direction="H" if horizontal else "V",
                cells=cells,
            )
            authoritative_completed_placements.append(placement)
            existing_keys.add(cells)
            existing_cells.update(cells)
            pending_completed_ship_cells.update(
                set(cells) - initial_real_hits - committed_hits
            )
            locked_completed_ship_cells.update(cells)
            logger.info(
                "locking red-scout-completed submarine placement: length=%s cells=%s",
                length,
                list(cells),
            )

        if not accepted_cells:
            return
        refresh_completed_ship_safety_area()
        visual_only_completed_cells.update(
            pending_completed_ship_cells & accepted_cells
        )
        red_marker_completed_cells.update(accepted_cells)
        initial_completed_visual_hits.update(accepted_cells)
        online_completed_visual_hits.update(accepted_cells)
        authoritative_completed_visual_hits.update(accepted_cells)

        cumulative_lengths = list(
            online_sidebar_completed_lengths or initial_completed_lengths
        )
        remaining_fleet = list(submarines)
        for length in cumulative_lengths:
            try:
                remaining_fleet.remove(length)
            except ValueError:
                continue
        for length in completed_lengths:
            if length in remaining_fleet:
                cumulative_lengths.append(length)
                remaining_fleet.remove(length)
        online_sidebar_completed_lengths = tuple(cumulative_lengths)

    def current_red_state_strategy() -> SubmarineStrategy:
        state_strategy = SubmarineStrategy(grid_size, submarines)
        restore_placements = getattr(
            state_strategy,
            "restore_confirmed_placements",
            None,
        )
        if authoritative_completed_placements and callable(restore_placements):
            restore_placements(authoritative_completed_placements)
        real_hits = initial_real_hits | committed_hits
        for cell in real_hits:
            state_strategy.report_result(cell, True)
        for cell in committed_misses - real_hits:
            state_strategy.report_result(cell, False)
        for cell in initial_misses - real_hits - committed_misses:
            state_strategy.report_result(cell, False)
        if scout_hits or scout_misses:
            state_strategy.report_scout_results(
                hits=scout_hits - real_hits,
                misses=scout_misses - real_hits - committed_misses,
            )
        completed_lengths = (
            online_sidebar_completed_lengths
            if online_sidebar_completed_lengths
            else initial_completed_lengths
        )
        completed_visual_hits = (
            online_completed_visual_hits
            if online_sidebar_completed_lengths
            else initial_completed_visual_hits
        )
        if completed_lengths:
            state_strategy.reconcile_completed_lengths(
                completed_lengths,
                observed_completed_cells=completed_visual_hits,
            )
        return state_strategy

    def current_board_states() -> list[list[str]]:
        states = build_runtime_board_states(
            current_red_state_strategy(),
            grid_size,
        )
        for row, col in visual_only_completed_cells - discarded_flag_cells:
            if 0 <= row < grid_size and 0 <= col < grid_size:
                if states[row][col] == "ship":
                    states[row][col] = "scout_hit"
        return states

    def current_forbidden_red_centers() -> set[Cell]:
        states = current_red_state_strategy().get_cell_states()
        return {
            (row, col)
            for row in range(grid_size)
            for col in range(grid_size)
            if states[row][col] != "unknown"
        }

    def current_display_hit_count() -> int:
        initial_visual_count = scan_kwargs.get("initial_visual_hit_count")
        base_count = (
            len(initial_real_hits)
            if initial_visual_count is None
            else max(0, int(initial_visual_count))
        )
        return min(sum(submarines), base_count + len(committed_hits))

    for _ in range(settings.count):
        # Keep a snapshot of visual completion evidence from before this red
        # result is processed.  Cells learned as part of the current result
        # must remain eligible for a blue tap; only cells that were already
        # visually completed before this attempt may be skipped.
        pre_result_visual_hits = set(
            initial_completed_visual_hits
            | online_completed_visual_hits
            | authoritative_completed_visual_hits
        )
        pre_result_real_hits = set(initial_real_hits | committed_hits)
        # Keep a snapshot of cells that were already known misses before this
        # red result.  A red footprint may report the same cell again; that
        # repeated evidence must not be forwarded as a new scout miss.
        pre_result_real_misses = set(initial_misses | committed_misses)
        discarded_before_result = set(discarded_flag_cells)
        forbidden_centers = current_forbidden_red_centers()
        center = planner.choose_center(
            footprint,
            known_cells=forbidden_centers,
            covered_cells=covered,
            cell_scores={},
            excluded_centers=attempted_centers,
        )
        if center is None:
            break
        explored_cells = forbidden_centers | covered
        if center in explored_cells:
            raise RedScoutSafetyError(
                "red scout planner selected an already explored or otherwise "
                f"non-unknown center: {center}"
            )
        if center in attempted_centers:
            raise RedScoutSafetyError(
                f"red scout planner repeated an already used center: {center}"
            )
        attempted_centers.add(center)
        index = center[0] * grid_size + center[1]
        result = _execute_red_scout_transaction(
            level,
            center,
            click_points[index],
            index,
            grid_size,
            click_points,
            excluded_cells=explored_cells,
            learned_footprint=footprint,
            submarine_lengths=submarines,
            attempt=attempts_completed + 1,
        )
        blue_bomb_ready = False
        online_network_ready = False
        attempts_completed += 1
        if result.valid:
            valid_attempts += 1
        complete_six = (
            result.valid
            and len(result.affected_cells) == 6
            and not result.unknown_cells
            and result.affected_cells == result.hit_cells | result.miss_cells
        )
        if complete_six:
            complete_six_attempts += 1
        logger.info(
            "red scout %s/%s center=%s affected=%s hits=%s misses=%s unknown=%s "
            "valid=%s complete_six=%s invalid_reason=%s",
            attempts_completed,
            settings.count,
            center,
            sorted(result.affected_cells),
            sorted(result.hit_cells),
            sorted(result.miss_cells),
            sorted(result.unknown_cells),
            result.valid,
            complete_six,
            result.invalid_reason,
        )
        if result.level_completed:
            write_runtime_status(
                phase="level_complete",
                level=level,
                current_cell="--",
                red_scout_current=attempts_completed,
                red_scout_total=settings.count,
                red_scout_valid=valid_attempts,
                red_scout_complete_six=complete_six_attempts,
                board_size=grid_size,
                board_states=current_board_states(),
                last_result="level_complete",
            )
            return True
        if result.invalid_reason == "local_victory_screen":
            # The red request was safely discarded by the transaction, but the
            # victory screen means no remaining red-scout attempts are useful.
            # Continue with the normal blue attack strategy instead.
            logger.info(
                "level %s red scout reached a victory screen after %s/%s attempts; "
                "stopping remaining red scouts and switching to blue attack",
                level,
                attempts_completed,
                settings.count,
            )
            break
        # A result can be unsuitable for learning a reusable footprint while still
        # containing reliable per-cell hit/miss evidence. Keep that evidence on the
        # cumulative board instead of discarding the whole scout attempt.
        if result.affected_cells:
            merge_red_scout_observations(scout_hits, scout_misses, result)
            covered.update(result.affected_cells)
            normalize_current_2x2_l_shape(
                result.hit_cells,
                snapshot_cells=result.affected_cells,
            )
            normalize_flag_overlap_state(
                result.hit_cells,
                snapshot_cells=result.affected_cells,
            )
            prune_discarded_authoritative_placements()
            retain_red_scout_completed_diagnostics(result)
            # Completed-ship diagnostics can add the two real lower cells of
            # an L-shape after the initial normalization pass.  Re-run the
            # correction before constructing any blue-target list so the
            # raised upper flag is removed without spending a blue shot.
            normalize_current_2x2_l_shape(
                result.hit_cells,
                snapshot_cells=result.affected_cells,
            )
            restore_locked_completed_ship_cells()
            prune_discarded_authoritative_placements()
        if result.valid:
            # The first valid footprint is the approved shape for every later attempt.
            if footprint is None:
                footprint = result.footprint

        # Drop only evidence that was already authoritative before this
        # attempt.  Cells first learned by the current red result must remain
        # eligible for the blue follow-up (unless L-shape cleanup removed
        # them), even when completion diagnostics promoted them to a locked
        # placement earlier in this pass.
        scout_hits.difference_update(pre_result_real_hits)
        scout_misses.difference_update(pre_result_real_misses)
        write_runtime_status(
            phase="red_scout_capture",
            level=level,
            current_cell=index,
            red_scout_current=attempts_completed,
            red_scout_total=settings.count,
            red_scout_valid=valid_attempts,
            red_scout_complete_six=complete_six_attempts,
            board_size=grid_size,
            board_states=current_board_states(),
            last_result="scout_valid" if result.valid else "scout_invalid",
        )

        excluded_direct_cells = (
            pre_result_real_hits
            | committed_misses
            | direct_attempted_cells
        )
        # Use the filtered cumulative set so a flag-overlap cell removed above
        # cannot still be selected from the raw result for a blue shot.
        new_scout_hits = sorted(
            (set(result.hit_cells) & scout_hits)
            - excluded_direct_cells
            - discarded_flag_cells
            - pre_result_visual_hits
        )
        # A red marker can reveal the complete hull while one of its cells is
        # still unopened in the committed board.  That cell is mandatory even
        # if ordinary result filtering removed it after geometry reconciliation.
        new_scout_hits = sorted(
            set(new_scout_hits)
            | (
                pending_completed_ship_cells
                - direct_attempted_cells
                - discarded_flag_cells
            )
        )
        # L-shape correction can move the lower pair into the completed-visual
        # set while processing this result. Those cells are an explicit safety
        # exception; ordinary completed-submarine scout hits must still be
        # clicked even when they are also present in a visual completion set.
        if discarded_flag_cells - discarded_before_result:
            new_scout_hits = [
                cell
                for cell in new_scout_hits
                if cell not in online_completed_visual_hits
            ]

        # A complete submarine can be learned from the red result before its
        # cells become eligible as ordinary scout hits.  In that case the
        # normal filter above intentionally removes every cell, but we still
        # need one blue shot to submit the scout-confirmed submarine hit.  Use
        # one deterministic representative and remember it through
        # ``direct_attempted_cells`` so later red scouts do not repeat it.
        if complete_six and not new_scout_hits and result.hit_cells:
            # The placement may be normalized before it is copied into one of
            # the completed-visual sets, so use the current red result as the
            # source of truth for this first representative shot.  Exclude
            # every upper flag removed by the pre-blue L normalization.
            representative_candidates = (
                set(result.hit_cells)
                - discarded_flag_cells
                - direct_attempted_cells
                - pre_result_real_hits
            )
            if representative_candidates:
                representative = min(representative_candidates)
                new_scout_hits = [representative]
                logger.info(
                    "complete submarine red hit was filtered from ordinary targets; "
                    "using representative blue target cell=%s",
                    representative,
                )

        def commit_online_scout_result(
            cell: Cell,
            direct_index: int,
            probe_result: ProbeResult,
            probe_metadata: Mapping[str, object],
        ) -> bool:
            nonlocal online_sidebar_completed_lengths
            nonlocal online_completed_visual_hits

            pending_completed_confirmation = (
                cell in pending_completed_ship_cells
            )
            restore_locked_completed_ship_cells()
            online_hit_evidence[cell] = dict(probe_metadata)
            level_completed = _probe_result_completed_level(probe_result)
            hit = _probe_result_is_hit(probe_result)

            # Completed submarines are monotonic facts.  A later shared batch
            # frame can be affected by another tap or a completion animation,
            # so never allow that observation to downgrade an authoritative
            # completed cell to MISS/UNKNOWN.  The cell is already excluded
            # from future target selection; this guard protects stale or
            # overlapping results from the current batch.
            if (
                not pending_completed_confirmation
                and cell in (
                    authoritative_completed_visual_hits
                    | red_marker_completed_cells
                )
            ):
                probe_result = ProbeResult.HIT
                probe_metadata = dict(probe_metadata)
                probe_metadata["completed_locked"] = True
                probe_metadata["decision_reason"] = "authoritative_completed_locked"
                hit = True

            if probe_result is ProbeResult.UNKNOWN:
                raise ProbeProtocolError(
                    f"online scout-hit result for cell {cell} is unknown; refusing to retry it"
                )
            if level_completed and not hit:
                write_runtime_status(
                    phase="level_complete",
                    level=level,
                    current_cell="--",
                    red_scout_current=attempts_completed,
                    red_scout_total=settings.count,
                    red_scout_valid=valid_attempts,
                    red_scout_complete_six=complete_six_attempts,
                    board_size=grid_size,
                    board_states=current_board_states(),
                    last_result=probe_result.value,
                )
                return True

            completed_lengths = tuple(
                int(length)
                for length in probe_metadata.get("sidebar_completed_lengths", ())
            )
            if completed_lengths:
                online_sidebar_completed_lengths = completed_lengths
                latest_completed_visual_hits = (
                    _trusted_completed_cells_from_probe_metadata(
                        probe_metadata,
                        click_points,
                        grid_size=grid_size,
                        anchor=cell,
                        preferred_cells=online_completed_visual_hits,
                    )
                )
                # Apply the mandatory red-scout L cleanup before recording
                # this snapshot as red-component-backed completion.  This
                # prevents a raised red item from promoting the upper cell of
                # a 2x2 L into an immutable completed placement.
                if latest_completed_visual_hits:
                    local_snapshot = {
                        candidate
                        for candidate in latest_completed_visual_hits
                        if max(
                            abs(candidate[0] - cell[0]),
                            abs(candidate[1] - cell[1]),
                        ) <= 1
                    }
                    normalize_current_2x2_l_shape(
                        local_snapshot,
                        snapshot_cells=local_snapshot,
                    )
                    latest_completed_visual_hits.difference_update(discarded_flag_cells)
                    prune_discarded_authoritative_placements()
                red_marker_completed_cells.update(latest_completed_visual_hits)
                retain_authoritative_completed_placements(
                    latest_completed_visual_hits,
                    completed_lengths,
                )
                online_completed_visual_hits = _merge_completed_visual_snapshot(
                    online_completed_visual_hits,
                    latest_completed_visual_hits,
                    completed_lengths=completed_lengths,
                    authoritative_cells=authoritative_completed_visual_hits,
                )
                completed_cells = set(latest_completed_visual_hits) - discarded_flag_cells
                visual_only_completed_cells.update(
                    completed_cells
                    - initial_real_hits
                    - committed_hits
                    - committed_misses
                    - {cell}
                )
                if completed_cells:
                    initial_real_hits.update(completed_cells)
                    committed_hits.update(completed_cells)
                    initial_misses.difference_update(completed_cells)
                    committed_misses.difference_update(completed_cells)
                    scout_misses.difference_update(completed_cells)
                    # From this point on, completed cells are immutable.  Keep
                    # the authoritative set separate from provisional visual
                    # snapshots so a later snapshot cannot remove them.
                    authoritative_completed_visual_hits.update(completed_cells)
                for discarded_cell in discarded_flag_cells:
                    initial_real_hits.discard(discarded_cell)
                    committed_hits.discard(discarded_cell)
                    red_marker_completed_cells.discard(discarded_cell)
                    online_completed_visual_hits.discard(discarded_cell)
                    authoritative_completed_visual_hits.discard(discarded_cell)

            if hit:
                pending_completed_ship_cells.discard(cell)
                proposed_hits = initial_real_hits | committed_hits | {cell}
                l_shaped_block = _find_l_shaped_hit_block(proposed_hits)
                if l_shaped_block is not None and cell in l_shaped_block:
                    false_cell = _resolve_false_hit_in_l_shape(
                        l_shaped_block,
                        online_hit_evidence,
                    )
                    if false_cell is not None:
                        # The explicit 2x2 L rule has priority over every
                        # ordinary and completion lock.  Remove the raised
                        # upper cell from all state/provenance sets before it
                        # can consume another blue shot.
                        corrected_ship = set(l_shaped_block) - {false_cell}
                        logger.warning(
                            "correcting impossible L-shaped hits %s from online evidence; "
                            "keeping straight ship cells %s and discarding false hit %s "
                            "(L-shape rule overrides completion lock)",
                            list(l_shaped_block),
                            sorted(corrected_ship),
                            false_cell,
                        )
                        initial_real_hits.discard(false_cell)
                        committed_hits.discard(false_cell)
                        red_marker_completed_cells.discard(false_cell)
                        initial_real_hits.update(corrected_ship)
                        committed_hits.update(corrected_ship)
                        committed_misses.add(false_cell)
                        discarded_flag_cells.add(false_cell)
                        scout_hits.discard(false_cell)
                        scout_misses.discard(false_cell)
                        if online_completed_visual_hits & set(l_shaped_block):
                            online_completed_visual_hits.difference_update(l_shaped_block)
                            online_completed_visual_hits.update(corrected_ship)
                            authoritative_completed_visual_hits.update(corrected_ship)
                        prune_discarded_authoritative_placements()
                    else:
                        raise ProbeProtocolError(
                            "online scout-hit result creates an impossible L-shaped hit block "
                            "that cannot be resolved from online hit evidence: "
                            f"{list(l_shaped_block)}"
                        )
                if cell not in committed_misses:
                    committed_hits.add(cell)
                    committed_misses.discard(cell)
                visual_only_completed_cells.discard(cell)
            elif probe_result is ProbeResult.MISS:
                if pending_completed_confirmation:
                    raise ProbeProtocolError(
                        "red-marker-completed submarine cell remained unopened "
                        f"after verified blue retries: {cell}"
                    )
                committed_misses.add(cell)
                committed_hits.discard(cell)
            else:
                raise ProbeProtocolError(
                    f"unexpected online scout-hit result for cell {cell}: {probe_result!r}"
                )

            scout_hits.discard(cell)
            scout_misses.discard(cell)
            restore_locked_completed_ship_cells()
            write_runtime_status(
                phase="level_complete" if level_completed else "blue_online_scout_hits",
                level=level,
                current_cell="--" if level_completed else direct_index,
                red_scout_current=attempts_completed,
                red_scout_total=settings.count,
                red_scout_valid=valid_attempts,
                red_scout_complete_six=complete_six_attempts,
                hits=current_display_hit_count(),
                total_ship_cells=sum(submarines),
                board_size=grid_size,
                board_states=current_board_states(),
                last_result=probe_result.value,
            )
            return level_completed

        batch_outcome: OnlineScoutBatchResult | None = None
        known_batch_hits = {
            (row, col)
            for row, values in enumerate(hit_map)
            for col, value in enumerate(values)
            if bool(value)
        }
        known_batch_hits.update(
            initial_real_hits
            | committed_hits
            | online_completed_visual_hits
            | authoritative_completed_visual_hits
            | (scout_hits - set(new_scout_hits))
        )
        batch_enabled_for_result = bool(
            ONLINE_SCOUT_BATCH_ENABLED
            and len(new_scout_hits) > 1
            and len({click_points[cell[0] * grid_size + cell[1]] for cell in new_scout_hits})
            == len(new_scout_hits)
            and len({cell[0] * grid_size + cell[1] for cell in new_scout_hits})
            == len(new_scout_hits)
            and len(new_scout_hits)
            <= max(
                0,
                sum(int(length) for length in submarines)
                - len(known_batch_hits)
                - unmapped_initial_visual_hits,
            )
        )
        if batch_enabled_for_result:
            batch_targets = [
                (cell, click_points[cell[0] * grid_size + cell[1]], cell[0] * grid_size + cell[1])
                for cell in new_scout_hits
            ]
            batch_outcome = _execute_online_scout_hit_batch(
                level=level,
                hit_map=hit_map,
                targets=batch_targets,
                click_points=click_points,
                submarines=submarines,
                activity_ready=True,
                blue_bomb_ready=blue_bomb_ready,
                network_ready=online_network_ready,
                unmapped_visual_hits=unmapped_initial_visual_hits,
            )
            for batch_cell, batch_metadata in batch_outcome.metadata.items():
                online_hit_evidence[batch_cell] = dict(batch_metadata)
            blue_bomb_ready = bool(
                any(
                    bool(metadata.get("blue_bomb_ready"))
                    for metadata in batch_outcome.metadata.values()
                )
                or blue_bomb_ready
            )
            online_network_ready = bool(
                any(
                    bool(metadata.get("network_ready"))
                    for metadata in batch_outcome.metadata.values()
                )
                or online_network_ready
            )

        pending_batch_error: ProbeProtocolError | None = None
        for cell in new_scout_hits:
            direct_attempted_cells.add(cell)
            row, col = cell
            direct_index = row * grid_size + col
            probe_metadata: dict[str, object] = {}
            write_runtime_status(
                phase="blue_online_scout_hits",
                level=level,
                current_cell=direct_index,
                red_scout_current=attempts_completed,
                red_scout_total=settings.count,
                red_scout_valid=valid_attempts,
                red_scout_complete_six=complete_six_attempts,
                board_size=grid_size,
                board_states=current_board_states(),
            )
            if batch_outcome is not None:
                probe_metadata.update(batch_outcome.metadata.get(cell, {}))
                probe_result = batch_outcome.results.get(cell, ProbeResult.UNKNOWN)
                blue_bomb_ready = bool(
                    probe_metadata.get("blue_bomb_ready", blue_bomb_ready)
                )
                online_network_ready = bool(
                    probe_metadata.get("network_ready", online_network_ready)
                )
            else:
                probe_result = _execute_online_scout_hit(
                    level=level,
                    hit_map=hit_map,
                    cell=cell,
                    point=click_points[direct_index],
                    click_points=click_points,
                    index=direct_index,
                    submarines=submarines,
                    probe_metadata=probe_metadata,
                    activity_ready=True,
                    blue_bomb_ready=blue_bomb_ready,
                    network_ready=online_network_ready,
                    fast_batch=True,
                )
                blue_bomb_ready = bool(probe_metadata.get("blue_bomb_ready", False))
                online_network_ready = bool(
                    probe_metadata.get("network_ready", online_network_ready)
                )

            if (
                cell in pending_completed_ship_cells
                and not _probe_result_is_hit(probe_result)
                and not _probe_result_completed_level(probe_result)
            ):
                first_probe_result = probe_result
                for retry_number in range(
                    1,
                    COMPLETED_SHIP_PENDING_BLUE_RETRIES + 1,
                ):
                    logger.warning(
                        "red-marker-completed submarine cell %s is still unopened "
                        "after blue result=%s; retrying with verified single-target "
                        "path (%s/%s)",
                        cell,
                        probe_result.value,
                        retry_number,
                        COMPLETED_SHIP_PENDING_BLUE_RETRIES,
                    )
                    retry_metadata: dict[str, object] = {}
                    probe_result = _execute_online_scout_hit(
                        level=level,
                        hit_map=hit_map,
                        cell=cell,
                        point=click_points[direct_index],
                        click_points=click_points,
                        index=direct_index,
                        submarines=submarines,
                        probe_metadata=retry_metadata,
                        activity_ready=True,
                        # Re-select and verify the blue projectile.  The fast
                        # batch state is exactly what may have caused the
                        # original tap to be ignored.
                        blue_bomb_ready=False,
                        network_ready=online_network_ready,
                        fast_batch=False,
                    )
                    retry_metadata["completed_pending_retry"] = retry_number
                    retry_metadata["first_probe_result"] = first_probe_result.value
                    probe_metadata = retry_metadata
                    blue_bomb_ready = bool(
                        retry_metadata.get("blue_bomb_ready", False)
                    )
                    online_network_ready = bool(
                        retry_metadata.get(
                            "network_ready",
                            online_network_ready,
                        )
                    )
                    if (
                        _probe_result_is_hit(probe_result)
                        or _probe_result_completed_level(probe_result)
                    ):
                        break

            try:
                level_completed = commit_online_scout_result(
                    cell,
                    direct_index,
                    probe_result,
                    probe_metadata,
                )
            except ProbeProtocolError as exc:
                if batch_outcome is None:
                    raise
                pending_batch_error = pending_batch_error or exc
                continue
            if level_completed and pending_batch_error is None:
                return True

        if pending_batch_error is not None:
            raise pending_batch_error

        # Retention scans walk every sample directory and can take about a
        # second on a populated debug folder. A blue batch has already finished
        # all of its requests here, so prune once instead of on every target.
        if new_scout_hits:
            _prune_probe_sample_dirs()
            _prune_screenshot_storage()

    write_runtime_status(phase="blue_attack", level=level,
                         red_scout_current=attempts_completed,
                         red_scout_total=settings.count,
                         red_scout_valid=valid_attempts,
                         red_scout_complete_six=complete_six_attempts,
                         current_cell="--",
                         board_size=grid_size,
                         board_states=current_board_states())

    final_scan_kwargs = dict(scan_kwargs)
    initial_visual_count = final_scan_kwargs.get("initial_visual_hit_count")
    if initial_visual_count is not None:
        final_scan_kwargs["initial_visual_hit_count"] = min(
            sum(submarines),
            max(0, int(initial_visual_count)) + len(committed_hits),
        )
    if online_sidebar_completed_lengths:
        remaining_lengths = list(submarines)
        for length in online_sidebar_completed_lengths:
            if length in remaining_lengths:
                remaining_lengths.remove(length)
        final_scan_kwargs["initial_sidebar_progress"] = SidebarProgress(
            active_lengths=tuple(remaining_lengths),
            completed_lengths=online_sidebar_completed_lengths,
        )
        final_scan_kwargs["initial_completed_lengths"] = online_sidebar_completed_lengths
        final_scan_kwargs["initial_completed_visual_hits"] = (
            (online_completed_visual_hits | committed_hits) - discarded_flag_cells
        )
        final_scan_kwargs["initial_authoritative_completed_visual_hits"] = (
            authoritative_completed_visual_hits - discarded_flag_cells
        )
        final_scan_kwargs["initial_authoritative_completed_placements"] = tuple(
            authoritative_completed_placements
        )
    else:
        final_scan_kwargs["initial_completed_visual_hits"] = (
            initial_completed_visual_hits - discarded_flag_cells
        )
        final_scan_kwargs["initial_authoritative_completed_visual_hits"] = (
            authoritative_completed_visual_hits - discarded_flag_cells
        )
        final_scan_kwargs["initial_authoritative_completed_placements"] = tuple(
            authoritative_completed_placements
        )
    final_scan_kwargs.update(
        initial_hits=(
            initial_real_hits
            | committed_hits
            | online_completed_visual_hits
            | authoritative_completed_visual_hits
        ) - discarded_flag_cells,
        initial_misses=initial_misses | committed_misses,
        # Propagate the mutable provenance set after L-shape normalization;
        # otherwise an upper cell removed from a red-marker lock would be
        # reintroduced into the final strategy as an immutable completion.
        initial_red_marker_completed_cells=(
            red_marker_completed_cells - discarded_flag_cells
        ),
        initial_scout_hits=scout_hits,
        initial_scout_misses=scout_misses,
        commit_scout_hits_online=True,
    )
    return _scan_level_by_strategy(
        level,
        hit_map,
        click_points,
        submarines,
        run_started_at=run_started_at,
        **final_scan_kwargs,
    )


def _sidebar_confirms_all_submarines(
    progress: SidebarProgress | None,
    submarines: Sequence[int],
) -> bool:
    return bool(
        progress is not None
        and progress.valid
        and sorted(progress.completed_lengths) == sorted(int(length) for length in submarines)
    )


def _victory_wait_timeout_for_sidebar_samples(
    samples: Sequence[SidebarProgress | None],
    submarines: Sequence[int],
    *,
    required_frames: int | None = None,
) -> float:
    required_frames = len(HIT_RESULT_FRAME_DELAYS) if required_frames is None else required_frames
    progress = _consistent_sidebar_progress(
        samples,
        submarines,
        required_frames=required_frames,
    )
    if progress is None or not progress.active_lengths:
        return VICTORY_WAIT_AFTER_HIT_SECONDS
    return VICTORY_WAIT_AFTER_CONFIRMED_INCOMPLETE_SECONDS


def _consistent_sidebar_progress(
    samples: Sequence[SidebarProgress | None],
    submarines: Sequence[int],
    *,
    required_frames: int,
) -> SidebarProgress | None:
    required_frames = max(1, int(required_frames))
    expected_fleet = tuple(sorted((int(length) for length in submarines), reverse=True))
    if not expected_fleet or len(samples) < required_frames:
        return None

    signatures: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for progress in samples:
        if progress is None or not progress.valid:
            return None
        active = tuple(sorted((int(length) for length in progress.active_lengths), reverse=True))
        completed = tuple(
            sorted((int(length) for length in progress.completed_lengths), reverse=True)
        )
        if tuple(sorted(active + completed, reverse=True)) != expected_fleet:
            return None
        signatures.append((active, completed))

    if any(signature != signatures[0] for signature in signatures[1:]):
        return None
    return samples[0]


def _can_stop_probe_frames_early(
    frame_records: Sequence[Mapping[str, object]],
    sidebar_samples: Sequence[SidebarProgress | None],
    submarines: Sequence[int],
) -> bool:
    if not ADAPTIVE_HIT_FRAMES_ENABLED:
        return False
    if not can_stop_after_stable_hit_frames(frame_records):
        return False
    progress = _consistent_sidebar_progress(
        sidebar_samples,
        submarines,
        required_frames=ADAPTIVE_HIT_MIN_FRAMES,
    )
    return progress is not None and bool(progress.active_lengths)


def _can_stop_online_scout_frames_early(
    frame_records: Sequence[Mapping[str, object]],
    *,
    min_frames: int | None = None,
) -> bool:
    """Allow a confirmed red-scout hit to finish online sampling early.

    Online scout hits already have an independent red-scout signal. Requiring
    two consecutive positive frames with template evidence keeps the blue
    request guarded while avoiding a fourth ADB screenshot when the sidebar
    detector is unavailable. The first frame may still be a transition frame
    (the common pattern is miss, hit, hit, hit). A victory frame never qualifies
    for this shortcut; it must use the normal completion path so the banner can
    be handled before the next request.
    """
    if not ADAPTIVE_HIT_FRAMES_ENABLED:
        return False
    required_frames = max(
        1,
        int(
            ONLINE_SCOUT_STABLE_HIT_MIN_FRAMES
            if min_frames is None
            else min_frames
        ),
    )
    if len(frame_records) < required_frames:
        return False
    stable_frames = frame_records[:required_frames]
    if any(bool(record.get("victory_banner")) for record in stable_frames):
        return False
    consecutive_frames = stable_frames[-2:]
    if len(consecutive_frames) < 2:
        return False
    for record in consecutive_frames:
        if record.get("template_hit") is not True:
            return False
        if record.get("dynamic_hit_vetoed") is not False:
            return False
        result = record.get("result")
        if not isinstance(result, Mapping):
            return False
        if result.get("state") != "hit" or result.get("evidence_vetoed") is not False:
            return False
    return True


def _select_blue_bomb_for_online_scout(
    sample_dir: Path,
    selection_screen: np.ndarray,
    *,
    fast: bool,
    check_ammo: bool = True,
) -> np.ndarray:
    if check_ammo:
        _raise_if_blue_ammo_depleted(selection_screen)
    adb.click(*BLUE_BOMB_POINT)
    settle_seconds = (
        ONLINE_SCOUT_BLUE_SELECT_FAST_SETTLE_SECONDS
        if fast
        else ONLINE_SCOUT_BLUE_SELECT_SETTLE_SECONDS
    )
    # The post-click frame is still needed as the before-image for hit
    # comparison, but projectile selection is trusted after the click. Do not
    # spend another screenshot/template pass checking the red button state.
    return adb.delay(settle_seconds).read_screenshot(sample_dir / "before.png")


def _execute_online_scout_hit_batch(
    level: int,
    hit_map: list[list[int]],
    targets: Sequence[tuple[Cell, tuple[int, int], int]] | None = None,
    submarines: Sequence[int] = (),
    *,
    cells: Sequence[Cell] | None = None,
    click_points: Sequence[tuple[int, int]] | None = None,
    indices: Sequence[int] | None = None,
    activity_ready: bool = False,
    blue_bomb_ready: bool = False,
    network_ready: bool = False,
    unmapped_visual_hits: int = 0,
) -> OnlineScoutBatchResult:
    """Fire several red-scout-confirmed targets, then analyse one frame set."""
    if targets is None:
        if cells is None or click_points is None:
            raise ProbeProtocolError(
                "online scout-hit batch requires targets or cells plus click_points"
            )
        grid_size = get_level_grid_size(level)
        normalized_indices = (
            [int(index) for index in indices]
            if indices is not None
            else [int(cell[0]) * grid_size + int(cell[1]) for cell in cells]
        )
        if len(cells) != len(click_points) or len(cells) != len(normalized_indices):
            raise ProbeProtocolError(
                "online scout-hit batch cells, click_points, and indices must have equal lengths"
            )
        targets = [
            (cell, point, index)
            for cell, point, index in zip(cells, click_points, normalized_indices)
        ]
    if not targets:
        return OnlineScoutBatchResult(results={}, metadata={})
    grid_size = get_level_grid_size(level)
    if _active_probe is not None:
        raise ProbeProtocolError(
            "cannot start online scout-hit batch while another probe is active"
        )

    normalized_targets: list[tuple[Cell, tuple[int, int], int]] = []
    seen_cells: set[Cell] = set()
    seen_indices: set[int] = set()
    seen_points: set[tuple[int, int]] = set()
    for target in targets:
        try:
            cell, point, index = target
            cell = (int(cell[0]), int(cell[1]))
            point = (int(point[0]), int(point[1]))
            index = int(index)
        except (TypeError, ValueError, IndexError):
            raise ProbeProtocolError(f"invalid online scout-hit batch target: {target!r}")
        if cell in seen_cells or index in seen_indices or point in seen_points:
            raise ProbeProtocolError(
                "online scout-hit batch contains duplicate cell, index, or point: "
                f"cell={cell} index={index} point={point}"
            )
        if (
            cell[0] < 0
            or cell[0] >= len(hit_map)
            or cell[1] < 0
            or cell[1] >= len(hit_map[cell[0]])
        ):
            raise ProbeProtocolError(
                f"online scout-hit batch cell is outside the hit map: {cell}"
            )
        seen_cells.add(cell)
        seen_indices.add(index)
        seen_points.add(point)
        normalized_targets.append((cell, point, index))

    result = OnlineScoutBatchResult(results={}, metadata={})
    sample_dirs: dict[Cell, Path | None] = {}
    before_visible: dict[Cell, bool] = {}

    def safe_status(sample_dir: Path | None, stage: str, **extra: object) -> None:
        if sample_dir is None:
            return
        try:
            _write_probe_status(sample_dir, stage, **extra)
        except OSError as exc:
            logger.warning("could not write batch probe status %s: %s", sample_dir, exc)

    def mark_level_complete(cells: Sequence[Cell], reason: str) -> OnlineScoutBatchResult:
        for cell in cells:
            result.results[cell] = ProbeResult.LEVEL_COMPLETE
            result.metadata[cell] = {
                "batch": True,
                "online_committed": False,
                "blue_bomb_ready": bool(blue_bomb_ready),
                "network_ready": True,
                "level_completed": True,
                "decision_reason": reason,
                "stable_state": "level_complete",
                "hit_votes": 0,
                "frame_count": 0,
            }
        result.level_completed = True
        result.stopped_reason = reason
        return result

    if network_ready:
        logger.info("reusing connected network state for scout-hit batch")
    else:
        adb.disable_reject_network(GAME_PACKAGE_NAME)
        disable_weak_network()
        if not activity_ready:
            adb.delay(ONLINE_SCOUT_NETWORK_SETTLE_SECONDS)

    initial_screen = adb.read_screenshot()
    red_marker_cells = detect_red_submarine_marker_cells(
        initial_screen,
        click_points or [],
        grid_size,
    ) if click_points is not None else set()
    # Do not send taps while a stale connection dialog is already covering the
    # activity.  Treat matcher failures as unsafe as well: a false negative here
    # would make the subsequent blind batch click the dialog instead of the
    # board, and there is no per-tap result frame to recover that mistake.
    try:
        initial_connection_dialog = find_connection_interrupted_dialog(initial_screen)
    except Exception as exc:
        raise ProbeProtocolError(
            "online scout-hit batch connection dialog detection failed before taps"
        ) from exc
    if initial_connection_dialog is not None:
        raise ProbeProtocolError(
            "online scout-hit batch found a connection dialog before taps; refusing to click"
        )

    if handle_victory_prompt(timeout=0.0, screenshot=initial_screen):
        return mark_level_complete([item[0] for item in normalized_targets], "initial_victory")

    initial_sidebar_progress = detect_sidebar_progress(initial_screen, submarines)
    if submarines and _sidebar_confirms_all_submarines(initial_sidebar_progress, submarines):
        handle_victory_prompt(timeout=VICTORY_WAIT_AFTER_HIT_SECONDS)
        return mark_level_complete(
            [item[0] for item in normalized_targets],
            "sidebar_already_complete",
        )

    if not activity_ready:
        detail_open = (
            isinstance(initial_screen, np.ndarray)
            and find_template(initial_screen, QUIT_ACTIVITY_TEMPLATE) is not None
        )
        if not detail_open and wait_until_occur(QUIT_ACTIVITY_TEMPLATE, timeout=2.0) is None:
            if enter_activity() is True:
                return mark_level_complete(
                    [item[0] for item in normalized_targets],
                    "activity_entry_reported_complete",
                )
            raise ProbeNotReadyError("online scout-hit batch could not reach activity detail")

    for cell, point, index in normalized_targets:
        try:
            sample_dir = _create_probe_sample_dir(
                level,
                cell,
                index,
                prune_retention=False,
            )
        except OSError as exc:
            logger.warning("could not create batch probe sample directory: %s", exc)
            sample_dir = None
        sample_dirs[cell] = sample_dir
        safe_status(
            sample_dir,
            "batch_started",
            level=level,
            cell=list(cell),
            index=index,
            point=list(point),
        )
        polygon = grid_cell_polygon(
            list(click_points or ()),
            index,
            grid_size,
        )
        before_visible[cell] = _visible_wreck_for_hit_state(
            initial_screen,
            point,
            red_marker_cells=red_marker_cells,
            cell=cell,
            cell_polygon=polygon,
            require_strong_body=True,
        )
        if before_visible[cell]:
            row, col = cell
            hit_map[row][col] = 1

    to_click = [item for item in normalized_targets if not before_visible[item[0]]]
    for cell, _point, index in normalized_targets:
        if not before_visible[cell]:
            continue
        row, col = cell
        hit_map[row][col] = 1
        result.results[cell] = ProbeResult.HIT
        result.metadata[cell] = {
            "batch": True,
            "online_committed": False,
            "already_visible": True,
            "blue_bomb_ready": bool(blue_bomb_ready),
            "network_ready": True,
            "stable_state": "hit",
            "hit_votes": MIN_HIT_RESULT_VOTES,
            "frame_count": MIN_HIT_RESULT_VOTES,
            "decision_reason": "already_visible",
            "level_completed": False,
        }
        safe_status(sample_dirs[cell], "complete", decision=ProbeResult.HIT.value)
        append_recent_probe_result(
            level=level,
            index=index,
            result=ProbeResult.HIT,
            reason="online_scout_batch_already_visible",
        )
    if not to_click:
        return result

    known_hits = sum(
        1
        for row in hit_map
        for value in row
        if bool(value)
    )
    remaining_capacity = max(
        0,
        sum(int(length) for length in submarines)
        - known_hits
        - max(0, int(unmapped_visual_hits)),
    )
    # An empty fleet configuration is used by a few low-level callers and does
    # not provide enough information to calculate a safe capacity.  Enforce the
    # guard whenever the configured fleet is available, but do not turn the
    # convenience form of this helper into an unconditional no-op.
    if submarines and len(to_click) > remaining_capacity:
        raise ProbeProtocolError(
            "online scout-hit batch exceeds the remaining submarine-cell capacity; "
            "refusing to risk taps after a possible victory"
        )

    selection_screen = initial_screen
    if not blue_bomb_ready:
        _raise_if_blue_ammo_depleted(selection_screen)
        first_sample_dir = sample_dirs[to_click[0][0]] or RUN_DEBUG_DIR
        selection_screen = _select_blue_bomb_for_online_scout(
            first_sample_dir,
            selection_screen,
            fast=True,
            check_ammo=False,
        )
        blue_bomb_ready = True
    else:
        _raise_if_blue_ammo_depleted(selection_screen)

    if not to_click:
        return result

    clicked_cells: list[Cell] = []
    for position, (cell, point, _index) in enumerate(to_click):
        logger.info(
            "online scout-hit batch tap %s/%s: cell=%s point=%s",
            position + 1,
            len(to_click),
            cell,
            point,
        )
        clicked_cells.append(cell)
        # Record the cell before issuing the input command.  ADB can raise
        # after dispatching a touch, so the conservative state must survive
        # both command failures and later screenshot failures.
        result.clicked_cells = tuple(clicked_cells)
        try:
            adb.click(*point)
        except Exception as exc:
            raise ProbeProtocolError(
                f"online scout-hit batch click failed after {len(clicked_cells)} taps; "
                "the committed cells will not be retried"
            ) from exc
        if position + 1 < len(to_click):
            try:
                adb.delay(ONLINE_SCOUT_BATCH_CLICK_INTERVAL_SECONDS)
            except Exception as exc:
                raise ProbeProtocolError(
                    f"online scout-hit batch click interval failed after "
                    f"{len(clicked_cells)} taps; the committed cells will not be retried"
                ) from exc
    result.clicked_cells = tuple(clicked_cells)
    clicked_target_cells = set(clicked_cells)
    clicked_target_items = [
        item for item in to_click if item[0] in clicked_target_cells
    ]
    frame_captures: list[tuple[Path, object]] = []
    clean_frame_captures: list[tuple[Path, object]] = []
    frame_results: dict[Cell, list[object]] = {cell: [] for cell in clicked_cells}
    frame_records: dict[Cell, list[dict[str, object]]] = {
        cell: [] for cell in clicked_cells
    }
    sidebar_progress_samples: list[SidebarProgress | None] = []
    latest_sidebar_progress: SidebarProgress | None = None
    sidebar_newly_completed: tuple[int, ...] = ()
    sidebar_completion_screenshot: np.ndarray | None = None
    sidebar_evidence_cell: Cell | None = None
    sidebar_completed_cells: set[Cell] = set()
    victory_detected = False
    victory_screenshot: np.ndarray | None = None
    connection_overlay_detected = False
    victory_cell = clicked_target_items[-1][0] if clicked_target_items else None

    for frame_index, frame_delay in enumerate(ONLINE_SCOUT_BATCH_FRAME_DELAYS, start=1):
        try:
            capture = adb.delay(frame_delay).capture_screenshot()
        except Exception as exc:
            raise ProbeProtocolError(
                "online scout-hit batch result screenshot failed after the blue taps; "
                "the committed cells will not be retried"
            ) from exc
        screenshot_path = RUN_DEBUG_DIR / f"online_batch_after_{frame_index}.png"
        frame_captures.append((screenshot_path, capture))
        after_img = getattr(capture, "image", None)
        if not isinstance(after_img, np.ndarray):
            raise ProbeProtocolError("online scout-hit batch returned an invalid screenshot")

        # Check the connection dialog before victory or hit classification.  A
        # dialog overlay can contain visual fragments that match both templates
        # and must never be allowed to manufacture a successful result.
        try:
            connection_overlay = find_connection_interrupted_dialog(after_img) is not None
        except Exception as exc:
            connection_overlay_detected = True
            raise ProbeProtocolError(
                "online scout-hit batch connection dialog detection failed after taps; "
                "the committed cells will not be retried"
            ) from exc
        if connection_overlay:
            connection_overlay_detected = True
            logger.error("connection dialog appeared during online scout-hit batch")
            break

        try:
            victory_hit = find_victory_banner(after_img) is not None
        except Exception:
            victory_hit = False
        victory_detected = victory_detected or victory_hit
        if victory_hit:
            victory_screenshot = after_img

        clean_frame_captures.append((screenshot_path, capture))

        # Register the complete result frame against the pre-click board
        # before any per-cell dynamic/static classification.  Overlay and
        # victory checks intentionally stay on the raw frame because those
        # templates belong to the screen, not the grid.
        try:
            grid_img, registration = register_translation(
                selection_screen,
                after_img,
                max_translation=8.0,
                min_response=0.08,
            )
        except Exception as exc:
            logger.warning("grid frame registration failed; using raw frame: %s", exc)
            grid_img = after_img
            registration = None

        current_frame_results: dict[Cell, object] = {}
        current_template_hits: dict[Cell, bool] = {}
        for cell, point, _index in clicked_target_items:
            try:
                classified = classify_diamond_hit(selection_screen, grid_img, point)
            except Exception as exc:
                logger.error("batch hit classification failed for cell %s: %s", cell, exc)
                raise
            if victory_hit and cell == victory_cell:
                classified.state = "hit"
                classified.score = max(float(classified.score), 1.0)
                classified.confidence = max(float(classified.confidence), 1.0)
            template_hit = apply_wreck_template_confirmation(grid_img, point, classified)
            if not template_hit:
                classified.evidence_kind = (
                    "dynamic_attack_hit" if classified.state == "hit" else "unknown"
                )
            current_frame_results[cell] = classified
            current_template_hits[cell] = template_hit

        frame_sidebar_progress: SidebarProgress | None = None
        frame_newly_completed: tuple[int, ...] = ()
        sidebar_hit = False
        frame_sidebar_evidence_cell: Cell | None = None
        if submarines and current_frame_results:
            positive_candidates = [
                (
                    position,
                    cell,
                    current_template_hits[cell],
                    current_frame_results[cell],
                )
                for position, (cell, _point, _index) in enumerate(clicked_target_items)
                if getattr(current_frame_results[cell], "state", None) == "hit"
            ]
            if positive_candidates:
                (
                    _position,
                    frame_sidebar_evidence_cell,
                    _template_hit,
                    representative,
                ) = max(
                    positive_candidates,
                    key=lambda item: (
                        int(item[2]),
                        float(getattr(item[3], "score", 0.0)),
                        float(getattr(item[3], "confidence", 0.0)),
                        item[0],
                    ),
                )
            else:
                # The sidebar detector only needs an object to promote while it
                # compares before/after progress.  Do not let that promotion be
                # used as per-cell hit evidence when every visual classification
                # is a miss.
                representative = current_frame_results[clicked_target_items[-1][0]]
            # Sidebar completion is batch-level evidence.  The confirmation
            # helper promotes its result argument in place, so isolate that
            # mutation from the per-cell visual classification; otherwise a
            # sidebar change caused by an earlier tap can turn a later miss
            # into a false hit.
            sidebar_result = copy(representative)
            sidebar_hit, frame_sidebar_progress, frame_newly_completed = (
                apply_sidebar_completion_confirmation(
                    selection_screen,
                    after_img,
                    submarines,
                    sidebar_result,
                )
            )
            if frame_sidebar_progress is not None and frame_sidebar_progress.valid:
                latest_sidebar_progress = frame_sidebar_progress
            if frame_newly_completed:
                sidebar_newly_completed = frame_newly_completed
                sidebar_completion_screenshot = after_img
                if frame_sidebar_evidence_cell is not None:
                    sidebar_evidence_cell = frame_sidebar_evidence_cell
            if frame_newly_completed and frame_sidebar_progress is not None:
                # A batch can complete more than one submarine while all
                # per-cell animations are still moving.  Resolve the finished
                # placements from the same screenshot and use that structural
                # evidence for only the clicked cells it actually supports.
                # ``targets=`` is the normal caller form, so use the points
                # carried by the normalized target tuples rather than the
                # optional legacy ``click_points`` argument.
                completion_points = (
                    list(click_points)
                    if click_points is not None
                    else [point for _cell, point, _index in normalized_targets]
                )
                completion_candidates = detect_completed_submarine_candidate_cells(
                    after_img,
                    completion_points,
                    grid_size,
                )
                if completion_candidates:
                    completion_resolution = resolve_completed_ship_cells(
                        completion_candidates,
                        frame_sidebar_progress.completed_lengths,
                        grid_size=grid_size,
                        preferred_cells=set(clicked_target_cells),
                    )
                    sidebar_completed_cells.update(
                        set(completion_resolution.cells) & clicked_target_cells
                    )
                for completed_cell in sidebar_completed_cells:
                    classified = current_frame_results.get(completed_cell)
                    if classified is not None:
                        classified.state = "hit"
                        classified.score = max(float(classified.score), 0.99)
                        classified.confidence = max(float(classified.confidence), 0.99)
                        classified.evidence_kind = "completed_submarine"
            elif sidebar_hit and frame_sidebar_evidence_cell is not None:
                current_frame_results[frame_sidebar_evidence_cell].evidence_kind = (
                    "completed_submarine"
                )
        sidebar_progress_samples.append(frame_sidebar_progress)

        for cell, point, _index in clicked_target_items:
            classified = current_frame_results[cell]
            dynamic_hit_vetoed = enforce_positive_hit_evidence(
                classified,
                wreck_hit=current_template_hits[cell],
                sidebar_hit=(
                    (sidebar_hit and cell == frame_sidebar_evidence_cell)
                    or (victory_hit and cell == victory_cell)
                    or cell in sidebar_completed_cells
                ),
            )
            frame_results[cell].append(classified)
            frame_records[cell].append(
                {
                    "frame": frame_index,
                    "delay": frame_delay,
                    "path": str(sample_dirs[cell] / f"after_{frame_index}.png")
                    if sample_dirs[cell] is not None
                    else str(screenshot_path),
                    "batch": True,
                    "template_hit": current_template_hits[cell],
                    "dynamic_hit_vetoed": dynamic_hit_vetoed,
                    "sidebar_hit": (
                        (sidebar_hit and cell == frame_sidebar_evidence_cell)
                        or cell in sidebar_completed_cells
                    ),
                    "victory_banner": victory_hit and cell == victory_cell,
                    "connection_overlay": connection_overlay,
                    "grid_registration": (
                        {
                            "dx": float(registration.dx),
                            "dy": float(registration.dy),
                            "response": float(registration.response),
                            "accepted": bool(registration.accepted),
                        }
                        if registration is not None
                        else None
                    ),
                    "sidebar_completed_lengths": (
                        list(frame_sidebar_progress.completed_lengths)
                        if frame_sidebar_progress is not None and frame_sidebar_progress.valid
                        else []
                    ),
                    "sidebar_newly_completed_lengths": list(frame_newly_completed),
                    "result": _hit_result_to_dict(classified),
                }
            )
        if victory_hit:
            break

    if not frame_captures:
        raise ProbeProtocolError("online scout-hit batch captured no result frames")

    shared_capture_records = list(clean_frame_captures)
    unknown_cells: list[Cell] = []
    for cell, _point, index in normalized_targets:
        if cell in result.results:
            continue
        results_for_cell = frame_results.get(cell, [])
        records_for_cell = frame_records.get(cell, [])
        insufficient_after_connection_dialog = bool(
            connection_overlay_detected
            and len(clean_frame_captures) < MIN_HIT_RESULT_VOTES
        )
        if not results_for_cell or insufficient_after_connection_dialog:
            result.results[cell] = ProbeResult.UNKNOWN
            result.metadata[cell] = {
                "batch": True,
                "online_committed": cell in clicked_cells,
                "blue_bomb_ready": bool(blue_bomb_ready),
                "network_ready": True,
                "hit_votes": 0,
                "frame_count": 0,
                "stable_state": "unknown",
                "decision_reason": (
                    "connection_overlay_insufficient_frames"
                    if insufficient_after_connection_dialog
                    else "no_result_frames"
                ),
                "connection_overlay_detected": connection_overlay_detected,
                "clean_frame_count": len(clean_frame_captures),
                "captured_frame_count": len(frame_captures),
                "level_completed": False,
            }
            unknown_cells.append(cell)
            safe_status(
                sample_dirs[cell],
                "complete",
                decision=ProbeResult.UNKNOWN.value,
                reason=(
                    "connection_overlay_insufficient_frames"
                    if insufficient_after_connection_dialog
                    else "no_result_frames"
                ),
                batch=True,
            )
            append_recent_probe_result(
                level=level,
                index=index,
                result=ProbeResult.UNKNOWN,
                reason=(
                    "online_scout_batch_connection_overlay_insufficient_frames"
                    if insufficient_after_connection_dialog
                    else "online_scout_batch_no_result_frames"
                ),
            )
            continue

        stable_analysis = _analyze_stable_probe_frames(
            selection_screen,
            shared_capture_records,
            next(item[1] for item in normalized_targets if item[0] == cell),
        )
        stable_suspect = stable_analysis is not None and stable_hit_is_suspect(stable_analysis)
        if victory_detected and cell == victory_cell:
            hit, decision_reason = True, "victory_banner_frame"
        else:
            hit, decision_reason = decide_hit_from_frames(results_for_cell)
        hit_votes = sum(1 for item in results_for_cell if item.state == "hit")
        uncertain = not hit and (
            hit_votes == 1
            or any(_is_suspect_hit_frame(item) for item in results_for_cell)
            or stable_suspect
        )
        sample_dir = sample_dirs[cell]
        if sample_dir is not None:
            try:
                _persist_probe_debug_images(
                    sample_dir,
                    None,
                    [
                        (
                            sample_dir / f"after_{frame_index}.png",
                            capture,
                        )
                        for frame_index, (_path, capture) in enumerate(
                            frame_captures,
                            start=1,
                        )
                    ],
                    records_for_cell,
                    preserve_all=uncertain,
                )
                _save_probe_result_json(
                    sample_dir,
                    level=level,
                    cell=cell,
                    index=index,
                    point=next(item[1] for item in normalized_targets if item[0] == cell),
                    hit=hit,
                    hit_votes=hit_votes,
                    frames=records_for_cell,
                    suspect_extra_checked=False,
                    decision_reason=decision_reason,
                    result_unknown=uncertain,
                    stable_analysis=_stable_analysis_to_dict(stable_analysis),
                )
            except OSError as exc:
                logger.warning("could not persist online batch evidence for %s: %s", cell, exc)

        if uncertain:
            result.results[cell] = ProbeResult.UNKNOWN
            unknown_cells.append(cell)
            stable_state = "unknown"
        elif hit:
            row, col = cell
            hit_map[row][col] = 1
            result.results[cell] = ProbeResult.HIT
            stable_state = "hit"
        else:
            row, col = cell
            hit_map[row][col] = 0
            result.results[cell] = ProbeResult.MISS
            stable_state = "miss"
        result.metadata[cell] = {
            "batch": True,
            "online_committed": True,
            "blue_bomb_ready": bool(blue_bomb_ready),
            "network_ready": True,
            "hit_votes": hit_votes,
            "frame_count": len(results_for_cell),
            "stable_state": stable_state,
            "decision_reason": decision_reason,
            "connection_overlay_detected": connection_overlay_detected,
            "clean_frame_count": len(clean_frame_captures),
            "captured_frame_count": len(frame_captures),
            "sidebar_newly_completed_lengths": (
                tuple(sidebar_newly_completed)
                if cell in sidebar_completed_cells or cell == sidebar_evidence_cell
                else ()
            ),
            "sidebar_completed_lengths": (
                tuple(latest_sidebar_progress.completed_lengths)
                if (
                    cell in sidebar_completed_cells or cell == sidebar_evidence_cell
                ) and latest_sidebar_progress is not None
                and latest_sidebar_progress.valid
                else ()
            ),
            "sidebar_completed_cells": (
                latest_sidebar_progress.completed_cells
                if (
                    cell in sidebar_completed_cells or cell == sidebar_evidence_cell
                ) and latest_sidebar_progress is not None
                and latest_sidebar_progress.valid
                else 0
            ),
            "sidebar_completion_screenshot": (
                sidebar_completion_screenshot
                if cell in sidebar_completed_cells or cell == sidebar_evidence_cell
                else None
            ),
            "level_completed": False,
        }
        safe_status(
            sample_dir,
            "complete",
            decision=(ProbeResult.UNKNOWN.value if uncertain else result.results[cell].value),
            reason=decision_reason,
            hit_votes=hit_votes,
            batch=True,
        )
        append_recent_probe_result(
            level=level,
            index=index,
            result=result.results[cell],
            reason=f"online_scout_batch_{decision_reason}",
        )

    if victory_detected and clicked_cells:
        # The victory overlay can cover the earlier cells, so their shared
        # frame analysis may be UNKNOWN even though the level is authoritative
        # complete. A victory banner is stronger evidence than those polluted
        # per-cell classifications; resolve unknown cells as hits so the outer
        # commit loop cannot retry an already dispatched target. The final tap
        # is the only one carrying the level-complete result.
        for cell in clicked_cells:
            if result.results.get(cell) is not ProbeResult.UNKNOWN:
                continue
            result.results[cell] = ProbeResult.HIT
            metadata = result.metadata.setdefault(cell, {})
            metadata.update(
                stable_state="hit",
                decision_reason="victory_banner_batch",
                level_completed=False,
                victory_banner=True,
            )
        unknown_cells.clear()
        final_cell = clicked_cells[-1]
        result.results[final_cell] = ProbeResult.HIT_AND_LEVEL_COMPLETE
        result.metadata.setdefault(final_cell, {}).update(
            level_completed=True,
            decision_reason="victory_banner",
            victory_banner=True,
        )
        result.level_completed = True
        result.stopped_reason = "victory_banner"
        if victory_screenshot is not None:
            # Match the single-target online path: consume the victory prompt
            # before the caller starts the base/reconnect transition.
            handle_victory_prompt(timeout=0.0, screenshot=victory_screenshot)
    elif (
        latest_sidebar_progress is not None
        and _sidebar_confirms_all_submarines(latest_sidebar_progress, submarines)
        and clicked_cells
        and _probe_result_is_hit(
            result.results.get(clicked_cells[-1], ProbeResult.UNKNOWN)
        )
    ):
        final_cell = clicked_cells[-1]
        result.results[final_cell] = ProbeResult.HIT_AND_LEVEL_COMPLETE
        result.metadata[final_cell]["level_completed"] = True
        result.level_completed = True
        result.stopped_reason = "sidebar_complete"
    if unknown_cells and not victory_detected:
        result.stopped_reason = "unknown_result"
        result.level_completed = False
    if result.level_completed:
        for cell in clicked_cells:
            if cell in result.metadata:
                result.metadata[cell]["level_completed"] = cell == clicked_cells[-1]
    return result


def _blue_bomb_zero_visible(screenshot: np.ndarray) -> bool:
    """Return true only when the blue-bomb counter explicitly matches 0."""
    if not isinstance(screenshot, np.ndarray) or screenshot.ndim != 3:
        return False
    if not BLUE_BOMB_ZERO_TEMPLATE.exists():
        return False
    left, top, right, bottom = BLUE_BOMB_ZERO_SEARCH_REGION
    height, width = screenshot.shape[:2]
    left = max(0, min(width, left))
    top = max(0, min(height, top))
    right = max(left, min(width, right))
    bottom = max(top, min(height, bottom))
    roi = screenshot[top:bottom, left:right]
    if roi.size == 0:
        return False
    return (
        find_template_multi_scale(
            roi,
            BLUE_BOMB_ZERO_TEMPLATE,
            scales=(0.9, 1.0, 1.1),
            threshold=BLUE_BOMB_ZERO_THRESHOLD,
        )
        is not None
    )


def _raise_if_blue_ammo_depleted(initial_screen: np.ndarray | None = None) -> None:
    """Return to base and stop before any blue-bomb click when count is 0."""
    if initial_screen is not None and not isinstance(initial_screen, np.ndarray):
        return
    first_screen = initial_screen if isinstance(initial_screen, np.ndarray) else adb.read_screenshot()
    if not _blue_bomb_zero_visible(first_screen):
        return
    second_screen = adb.delay(0.12).read_screenshot()
    if not _blue_bomb_zero_visible(second_screen):
        return
    _return_to_base_after_blue_ammo_depleted()
    write_runtime_status(
        phase="blue_ammo_depleted",
        current_cell="--",
        last_result="blue_ammo_zero",
    )
    raise BlueAmmoDepletedError(
        "blue bomb count is 0; returned to base and stopped before the next blue-bomb action"
    )


def _return_to_base_after_blue_ammo_depleted() -> None:
    """Use the normal reconnect dialog to leave the activity at zero blue ammo."""
    if _has_pending_probe_request():
        raise ProbeProtocolError(
            "blue ammo is 0 but a probe request is pending; refuse to reconnect before it is resolved"
        )

    logger.info("blue ammo is 0; opening the reconnect dialog before returning to base")
    enable_weak_network(PROBE_DROP_SETTLE_SECONDS)
    _verify_network_isolated_or_fail_closed(red_scout=False)
    adb.enable_reject_network(GAME_PACKAGE_NAME)
    write_runtime_status(network="DROP+REJECT 断网中")

    dialog = wait_until_connection_interrupted_dialog(
        timeout=MISS_CONNECTION_DIALOG_WAIT_SECONDS,
    )
    if dialog is None:
        latch_network_fail_closed("blue ammo is 0 but connection-interrupted dialog did not appear")
        raise BlueAmmoDepletedError(
            "blue bomb count is 0; connection dialog did not appear, keeping network isolated"
        )

    retry = wait_until_retry_button(timeout=MISS_RETRY_BUTTON_WAIT_SECONDS)
    if retry is None:
        latch_network_fail_closed("blue ammo is 0 but reconnect retry button did not appear")
        raise BlueAmmoDepletedError(
            "blue bomb count is 0; retry button did not appear, keeping network isolated"
        )

    disable_weak_network()
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    logger.info("blue ammo is 0; clicking reconnect retry and waiting for the base screen")
    adb.click(*retry.center)
    if wait_until_occur(
        ACTIVITY_BUTTON_TEMPLATE,
        timeout=POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        poll_interval=ACTIVITY_REENTRY_POLL_INTERVAL_SECONDS,
    ) is None:
        raise BlueAmmoDepletedError(
            "blue bomb count is 0; retry was clicked but the base activity icon did not appear"
        )
    logger.info("blue ammo is 0; base screen confirmed, stopping automation")


def _execute_online_scout_hit(
    *,
    level: int,
    hit_map: list[list[int]],
    cell: Cell,
    point: tuple[int, int],
    click_points: Sequence[tuple[int, int]] | None = None,
    index: int,
    submarines: Sequence[int],
    probe_metadata: dict[str, object] | None = None,
    activity_ready: bool = False,
    blue_bomb_ready: bool = False,
    network_ready: bool = False,
    fast_batch: bool = False,
) -> ProbeResult:
    """Commit one scout-confirmed blue hit online without the offline replay flow."""
    # Only a subsequent target in the same red-scout batch may use the
    # shortened confirmation path. The first target must establish blue-mode
    # state and keep the original evidence window.
    reused_batch_blue_selection = bool(blue_bomb_ready or fast_batch)
    if probe_metadata is not None:
        probe_metadata.clear()
    if _active_probe is not None:
        raise ProbeProtocolError(
            f"cannot commit online scout hit while probe {getattr(_active_probe, 'cell', None)} is active"
        )

    if network_ready:
        logger.info("reusing connected network state for scout-hit cell %s", cell)
    else:
        adb.disable_reject_network(GAME_PACKAGE_NAME)
        disable_weak_network()
        if not activity_ready:
            adb.delay(ONLINE_SCOUT_NETWORK_SETTLE_SECONDS)
    write_runtime_status(
        phase="blue_online_scout_hits",
        level=level,
        current_cell=index,
        network="已连接",
    )

    initial_screen = adb.read_screenshot()
    if handle_victory_prompt(timeout=0.0, screenshot=initial_screen):
        if probe_metadata is not None:
            probe_metadata["level_completed"] = True
        return ProbeResult.LEVEL_COMPLETE
    initial_sidebar_progress = detect_sidebar_progress(initial_screen, submarines)
    if _sidebar_confirms_all_submarines(initial_sidebar_progress, submarines):
        logger.info(
            "sidebar already shows every submarine complete before online scout cell %s; "
            "waiting for the victory screen instead of firing",
            cell,
        )
        handle_victory_prompt(timeout=VICTORY_WAIT_AFTER_HIT_SECONDS)
        if probe_metadata is not None:
            probe_metadata.update(
                level_completed=True,
                sidebar_completed_lengths=tuple(initial_sidebar_progress.completed_lengths),
                sidebar_completed_cells=initial_sidebar_progress.completed_cells,
            )
        return ProbeResult.LEVEL_COMPLETE

    detail_open = (
        isinstance(initial_screen, np.ndarray)
        and find_template(initial_screen, QUIT_ACTIVITY_TEMPLATE) is not None
    )
    # The caller only sets activity_ready after a confirmed detail-page entry.
    # Within one red-scout batch the page is not left between blue targets, so
    # trust that state and avoid a full-screen template match on every target.
    fast_activity_path = bool(activity_ready)
    if not fast_activity_path and not detail_open and wait_until_occur(
        QUIT_ACTIVITY_TEMPLATE, timeout=2.0
    ) is None:
        fast_activity_path = False
        if enter_activity() is True:
            if probe_metadata is not None:
                probe_metadata["level_completed"] = True
            return ProbeResult.LEVEL_COMPLETE
        adb.disable_reject_network(GAME_PACKAGE_NAME)
        disable_weak_network()
        adb.delay(ONLINE_SCOUT_NETWORK_SETTLE_SECONDS)
        if wait_until_occur(QUIT_ACTIVITY_TEMPLATE, timeout=6.0) is None:
            raise ProbeNotReadyError("online scout-hit commit could not reach activity detail")

    sample_dir = _create_probe_sample_dir(
        level,
        cell,
        index,
        prune_retention=False,
    )
    _write_probe_status(
        sample_dir,
        "online_scout_started",
        level=level,
        cell=list(cell),
        index=index,
        point=list(point),
    )

    selection_screen = initial_screen if fast_activity_path else adb.read_screenshot()
    marker_cells = detect_red_submarine_marker_cells(
        selection_screen,
        list(click_points or ()),
        get_level_grid_size(level),
    ) if click_points is not None else None
    cell_polygon = grid_cell_polygon(
        list(click_points or ()),
        index,
        get_level_grid_size(level),
    )
    already_visible = _visible_wreck_for_hit_state(
        selection_screen,
        point,
        red_marker_cells=marker_cells,
        cell=cell,
        cell_polygon=cell_polygon,
        require_strong_body=True,
    )
    if already_visible:
        before_img = selection_screen
    else:
        # A red-scout batch can contain several confirmed cells. The game keeps
        # the selected projectile after a shot, so reuse that state when the
        # current frame still proves the red projectile is not selected. If the
        # state is ambiguous, fall back to the original verified switch path.
        reused_blue_selection = bool(blue_bomb_ready)
        if reused_blue_selection:
            logger.info("reusing verified blue bomb selection for scout-hit cell %s", cell)
            before_img = selection_screen
        else:
            before_img = _select_blue_bomb_for_online_scout(
                sample_dir,
                selection_screen,
                fast=fast_activity_path,
            )
            blue_bomb_ready = True

    before_wreck_visible = already_visible or _visible_wreck_for_hit_state(
        before_img,
        point,
        red_marker_cells=marker_cells,
        cell=cell,
        cell_polygon=cell_polygon,
        require_strong_body=True,
    )
    if before_wreck_visible:
        row, col = cell
        hit_map[row][col] = 1
        logger.info(
            "scout-hit cell %s is already visible; recording it without firing another blue bomb",
            cell,
        )
        _write_probe_status(
            sample_dir,
            "complete",
            decision=ProbeResult.HIT.value,
            reason="already_visible",
        )
        append_recent_probe_result(
            level=level,
            index=index,
            result=ProbeResult.HIT,
            reason="online_scout_already_visible",
        )
        if probe_metadata is not None:
            probe_metadata.update(
                online_committed=False,
                already_visible=True,
                hit_votes=MIN_HIT_RESULT_VOTES,
                frame_count=MIN_HIT_RESULT_VOTES,
                stable_state="hit",
                decision_reason="already_visible",
            )
        return ProbeResult.HIT

    logger.info(
        "committing scout-confirmed hit online: level=%s cell=%s index=%s",
        level,
        cell,
        index,
    )
    _raise_if_blue_ammo_depleted(before_img)
    adb.click(*point)

    hit_results = []
    frame_records = []
    frame_captures: list[tuple[Path, object]] = []
    latest_sidebar_progress: SidebarProgress | None = None
    sidebar_progress_samples: list[SidebarProgress | None] = []
    sidebar_newly_completed: tuple[int, ...] = ()
    sidebar_completion_screenshot: np.ndarray | None = None
    victory_screenshot: np.ndarray | None = None

    def capture_online_frame(frame_index: int, frame_delay: float) -> None:
        nonlocal latest_sidebar_progress
        nonlocal sidebar_newly_completed
        nonlocal sidebar_completion_screenshot
        nonlocal victory_screenshot
        screenshot_path = sample_dir / f"after_{frame_index}.png"
        frame_capture = adb.delay(frame_delay).capture_screenshot()
        frame_captures.append((screenshot_path, frame_capture))
        after_img = frame_capture.image
        try:
            aligned_after, _registration = register_translation(
                before_img, after_img, max_translation=8.0, min_response=0.08
            )
            result = classify_diamond_hit(before_img, aligned_after, point)
        except Exception:
            for captured_path, captured_frame in frame_captures:
                captured_frame.save(captured_path)
            raise
        victory_hit = find_victory_banner(after_img) is not None
        if victory_hit:
            victory_screenshot = after_img
            result.state = "hit"
            result.score = max(float(result.score), 1.0)
            result.confidence = max(float(result.confidence), 1.0)

        template_hit = apply_wreck_template_confirmation(aligned_after, point, result)
        if not template_hit:
            result.evidence_kind = "dynamic_attack_hit" if result.state == "hit" else "unknown"
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
                sidebar_completion_screenshot = after_img
        sidebar_progress_samples.append(frame_sidebar_progress)

        dynamic_hit_vetoed = enforce_positive_hit_evidence(
            result,
            wreck_hit=template_hit,
            sidebar_hit=sidebar_hit or victory_hit,
        )
        hit_results.append(result)
        frame_records.append(
            {
                "frame": frame_index,
                "delay": frame_delay,
                "path": str(screenshot_path),
                "template_hit": template_hit,
                "dynamic_hit_vetoed": dynamic_hit_vetoed,
                "sidebar_hit": sidebar_hit,
                "sidebar_completed_lengths": (
                    list(frame_sidebar_progress.completed_lengths)
                    if frame_sidebar_progress is not None and frame_sidebar_progress.valid
                    else []
                ),
                "sidebar_newly_completed_lengths": list(frame_newly_completed),
                "victory_banner": victory_hit,
                "result": _hit_result_to_dict(result),
            }
        )
        _write_probe_status(
            sample_dir,
            "online_frame_captured",
            frame=frame_index,
            state=result.state,
            score=float(result.score),
        )

    adaptive_frames_stopped = False
    frame_schedule = (
        ONLINE_SCOUT_REUSED_HIT_FRAME_DELAYS
        if reused_batch_blue_selection
        else ONLINE_SCOUT_HIT_FRAME_DELAYS
    )
    for frame_index, frame_delay in enumerate(frame_schedule, start=1):
        capture_online_frame(frame_index, frame_delay)
        sidebar_stable = _can_stop_probe_frames_early(
            frame_records,
            sidebar_progress_samples,
            submarines,
        )
        # A reused blue selection has already been verified by the preceding
        # target in this red-scout batch, so two consecutive positive frames
        # are sufficient. The first online shot keeps the original schedule.
        online_hit_stable = reused_batch_blue_selection and _can_stop_online_scout_frames_early(
            frame_records,
            min_frames=ONLINE_SCOUT_REUSED_STABLE_HIT_MIN_FRAMES,
        )
        fast_strong_hit = (
            reused_batch_blue_selection
            and _is_strong_hit_frame(hit_results[-1])
        )
        if sidebar_stable or online_hit_stable or fast_strong_hit:
            adaptive_frames_stopped = True
            logger.info(
                "online scout hit stabilized after %s frames%s; skipping the remaining result frame",
                len(hit_results),
                (
                    " with sidebar evidence"
                    if sidebar_stable
                    else " with strong single-frame evidence"
                    if fast_strong_hit
                    else " with red-scout evidence"
                ),
            )
            break

    if (
        reused_batch_blue_selection
        and not adaptive_frames_stopped
        and len(frame_records) < len(ONLINE_SCOUT_HIT_FRAME_DELAYS)
    ):
        logger.info(
            "online scout hit quick confirmation was inconclusive after %s frames; "
            "falling back to the normal evidence schedule",
            len(frame_records),
        )
        for frame_index, frame_delay in enumerate(
            ONLINE_SCOUT_HIT_FRAME_DELAYS[len(frame_records):],
            start=len(frame_records) + 1,
        ):
            capture_online_frame(frame_index, frame_delay)

    hit_votes = sum(1 for result in hit_results if result.state == "hit")
    suspect_extra_checked = False
    if (
        victory_screenshot is None
        and hit_votes < MIN_HIT_RESULT_VOTES
        and not any(_is_strong_hit_frame(result) for result in hit_results)
        and any(_is_suspect_hit_frame(result) for result in hit_results)
    ):
        suspect_extra_checked = True
        logger.info(
            "online scout-hit cell=%s index=%s is uncertain after %s frames; "
            "collecting extra evidence without firing again",
            cell,
            index,
            len(hit_results),
        )
        for extra_index, frame_delay in enumerate(
            SUSPECT_HIT_EXTRA_FRAME_DELAYS,
            start=len(hit_results) + 1,
        ):
            capture_online_frame(extra_index, frame_delay)
        hit_votes = sum(1 for result in hit_results if result.state == "hit")

    stable_analysis = _analyze_stable_probe_frames(before_img, frame_captures, point)
    stable_suspect = (
        stable_analysis is not None and stable_hit_is_suspect(stable_analysis)
    )
    if victory_screenshot is not None:
        hit, decision_reason = True, "victory_banner_frame"
    else:
        hit, decision_reason = decide_hit_from_frames(hit_results)
    uncertain = not hit and (
        hit_votes == 1
        or any(_is_suspect_hit_frame(result) for result in hit_results)
        or stable_suspect
    )
    preserve_all_images = _should_preserve_all_probe_images(
        frame_records,
        suspect_extra_checked=suspect_extra_checked,
        victory_detected=victory_screenshot is not None,
        result_unknown=uncertain,
    )
    _persist_probe_debug_images(
        sample_dir,
        None,
        frame_captures,
        frame_records,
        preserve_all=preserve_all_images,
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
        adaptive_frames_stopped=adaptive_frames_stopped,
        result_unknown=uncertain,
        stable_analysis=_stable_analysis_to_dict(stable_analysis),
    )

    if uncertain:
        _write_probe_status(
            sample_dir,
            "complete",
            decision=ProbeResult.UNKNOWN.value,
            reason=decision_reason,
        )
        append_recent_probe_result(
            level=level,
            index=index,
            result=ProbeResult.UNKNOWN,
            reason=f"online_scout_{decision_reason}",
        )
        raise ProbeProtocolError(
            f"online scout-hit result for cell {cell} is uncertain; the blue request is already "
            "committed, so the cell will not be clicked again"
        )

    if hit:
        row, col = cell
        hit_map[row][col] = 1
        level_completed = victory_screenshot is not None or _sidebar_confirms_all_submarines(
            latest_sidebar_progress,
            submarines,
        )
        if victory_screenshot is not None:
            handle_victory_prompt(timeout=0.0, screenshot=victory_screenshot)
        elif level_completed:
            handle_victory_prompt(timeout=VICTORY_WAIT_AFTER_HIT_SECONDS)
        probe_result = (
            ProbeResult.HIT_AND_LEVEL_COMPLETE
            if level_completed
            else ProbeResult.HIT
        )
        logger.info(
            "online scout-hit result: level=%s cell=%s result=%s reason=%s",
            level,
            cell,
            probe_result.value,
            decision_reason,
        )
    else:
        row, col = cell
        hit_map[row][col] = 0
        probe_result = ProbeResult.MISS
        logger.warning(
            "scout-hit cell %s was a false positive; the online blue shot was committed as a miss",
            cell,
        )

    if latest_sidebar_progress is not None:
        write_runtime_status(
            sidebar_completed_cells=latest_sidebar_progress.completed_cells,
            sidebar_completed_lengths=list(latest_sidebar_progress.completed_lengths),
        )
    if probe_metadata is not None:
        probe_metadata.update(
            online_committed=True,
            blue_bomb_ready=blue_bomb_ready,
            network_ready=True,
            hit_votes=hit_votes,
            frame_count=len(hit_results),
            stable_state=(
                str(stable_analysis.result.state)
                if stable_analysis is not None
                else "unknown"
            ),
            decision_reason=decision_reason,
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
            sidebar_completion_screenshot=sidebar_completion_screenshot,
            level_completed=_probe_result_completed_level(probe_result),
        )
    _write_probe_status(
        sample_dir,
        "complete",
        decision=probe_result.value,
        reason=decision_reason,
        hit_votes=hit_votes,
    )
    append_recent_probe_result(
        level=level,
        index=index,
        result=probe_result,
        reason=f"online_scout_{decision_reason}",
    )
    return probe_result


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
    _verify_network_isolated_or_fail_closed(red_scout=False)

    transaction = ProbeTransaction(level=level, cell=cell, index=index)
    _active_probe = transaction
    x, y = point
    sample_dir: Path | None = None
    sample_failed = False
    before_capture = None
    frame_captures: list[tuple[Path, object]] = []
    frame_records: list[dict] = []

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
        before_capture = adb.capture_screenshot()
        before_img = before_capture.image
        _raise_if_blue_ammo_depleted(before_img)
        before_wreck_visible = visible_wreck_static_detected(before_img, (x, y))
        _write_probe_status(sample_dir, "before_captured", phase=transaction.phase.name)

        # 点击命令一旦发出，就保守地认为客户端可能已经暂存验证请求。
        transaction.advance(ProbePhase.REQUEST_PENDING)
        _write_probe_status(sample_dir, "request_pending", phase=transaction.phase.name)
        write_pending_probe(
            mode="blue_probe",
            level=level,
            cell=cell,
            index=index,
            phase=transaction.phase.name,
        )
        adb.click(x, y)
        _exit_activity_after_probe_click(
            RUN_DEBUG_DIR / "debug_quit1.png",
            use_system_back=True,
        )
        _write_probe_status(sample_dir, "activity_exited", phase=transaction.phase.name)
        if _reenter_activity_for_probe_result():
            transaction.advance(ProbePhase.RESULT_VISIBLE)
            update_pending_probe(phase=transaction.phase.name)
            _write_probe_status(
                sample_dir,
                "victory_detected",
                phase=transaction.phase.name,
            )
            transaction.advance(ProbePhase.RESULT_RECORDED)
            update_pending_probe(phase=transaction.phase.name)
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
            _persist_probe_debug_images(
                sample_dir,
                before_capture,
                frame_captures,
                frame_records,
                preserve_all=False,
            )
            _commit_hit_request_and_prepare_next_probe(transaction)
            clear_pending_probe()
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
        latest_sidebar_progress: SidebarProgress | None = None
        sidebar_progress_samples: list[SidebarProgress | None] = []
        sidebar_newly_completed: tuple[int, ...] = ()
        sidebar_completion_screenshot: np.ndarray | None = None
        victory_frame_detected = False
        adaptive_frames_stopped = False
        for frame_index, frame_delay in enumerate(HIT_RESULT_FRAME_DELAYS, start=1):
            screenshot_path = sample_dir / f"after_{frame_index}.png"
            frame_capture = adb.delay(frame_delay).capture_screenshot()
            frame_captures.append((screenshot_path, frame_capture))
            after_img = frame_capture.image
            try:
                aligned_after, _registration = register_translation(
                    before_img, after_img, max_translation=8.0, min_response=0.08
                )
            except Exception:
                aligned_after = after_img
            result = classify_diamond_hit(before_img, aligned_after, (x, y))
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
            template_hit = apply_wreck_template_confirmation(aligned_after, (x, y), result)
            if not template_hit:
                result.evidence_kind = "dynamic_attack_hit" if result.state == "hit" else "unknown"
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
                    sidebar_completion_screenshot = after_img
            sidebar_progress_samples.append(frame_sidebar_progress)
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
            if _can_stop_probe_frames_early(
                frame_records,
                sidebar_progress_samples,
                submarines,
            ):
                adaptive_frames_stopped = True
                logger.info(
                    "stable hit and sidebar evidence confirmed after %s frames; "
                    "skipping the remaining result frame",
                    len(hit_results),
                )
                break
            _write_probe_status(
                sample_dir,
                "frame_captured",
                phase=transaction.phase.name,
                frame=frame_index,
                state=result.state,
                score=float(result.score),
            )
        transaction.advance(ProbePhase.RESULT_VISIBLE)
        update_pending_probe(phase=transaction.phase.name)
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
                frame_capture = adb.delay(frame_delay).capture_screenshot()
                frame_captures.append((screenshot_path, frame_capture))
                after_img = frame_capture.image
                try:
                    aligned_after, _registration = register_translation(
                        before_img, after_img, max_translation=8.0, min_response=0.08
                    )
                except Exception:
                    aligned_after = after_img
                result = classify_diamond_hit(before_img, aligned_after, (x, y))
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
                template_hit = apply_wreck_template_confirmation(aligned_after, (x, y), result)
                if not template_hit:
                    result.evidence_kind = "dynamic_attack_hit" if result.state == "hit" else "unknown"
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
                        sidebar_completion_screenshot = after_img
                sidebar_progress_samples.append(frame_sidebar_progress)
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
        stable_analysis = _analyze_stable_probe_frames(
            before_img,
            frame_captures,
            (x, y),
        )
        stable_suspect = (
            stable_analysis is not None and stable_hit_is_suspect(stable_analysis)
        )
        first_result = hit_results[0]
        if victory_frame_detected:
            hit, decision_reason = True, "victory_banner_frame"
        else:
            hit, decision_reason = decide_hit_from_frames(hit_results)
        result_unknown = not hit and (
            suspect_extra_checked
            or hit_votes == 1
            or any(_is_near_hit_frame(result) for result in hit_results)
            or stable_suspect
        )
        preserve_all_images = _should_preserve_all_probe_images(
            frame_records,
            suspect_extra_checked=suspect_extra_checked,
            victory_detected=victory_frame_detected,
            result_unknown=result_unknown,
        )
        _persist_probe_debug_images(
            sample_dir,
            before_capture,
            frame_captures,
            frame_records,
            preserve_all=preserve_all_images,
        )
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
            adaptive_frames_stopped=adaptive_frames_stopped,
            result_unknown=result_unknown,
            stable_analysis=_stable_analysis_to_dict(stable_analysis),
        )
        transaction.hit = hit
        transaction.advance(ProbePhase.RESULT_RECORDED)
        update_pending_probe(phase=transaction.phase.name)
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
            victory_wait_timeout = VICTORY_WAIT_AFTER_HIT_SECONDS
            if not victory_frame_detected:
                victory_wait_timeout = _victory_wait_timeout_for_sidebar_samples(
                    sidebar_progress_samples,
                    submarines,
                    required_frames=(
                        ADAPTIVE_HIT_MIN_FRAMES
                        if adaptive_frames_stopped
                        else len(HIT_RESULT_FRAME_DELAYS)
                    ),
                )
            if victory_wait_timeout < VICTORY_WAIT_AFTER_HIT_SECONDS:
                logger.info(
                    "consistent sidebar frames confirm unfinished submarines; "
                    "limiting victory wait to %.1f seconds",
                    victory_wait_timeout,
                )
            level_complete = _commit_hit_request_and_prepare_next_probe(
                transaction,
                victory_wait_timeout=victory_wait_timeout,
            )
            probe_result = (
                ProbeResult.HIT_AND_LEVEL_COMPLETE
                if level_complete or victory_frame_detected
                else ProbeResult.HIT
            )
        elif result_unknown:
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

        clear_pending_probe()

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
                sidebar_completion_screenshot=sidebar_completion_screenshot,
            )
        return probe_result
    except Exception as exc:
        sample_failed = True
        if sample_dir is not None:
            try:
                _persist_probe_debug_images(
                    sample_dir,
                    before_capture,
                    frame_captures,
                    frame_records,
                    preserve_all=True,
                )
            except OSError as save_exc:
                logger.warning("failed to preserve interrupted probe images: %s", save_exc)
            _write_probe_status(
                sample_dir,
                "interrupted",
                phase=transaction.phase.name,
                error=repr(exc),
            )
        raise
    finally:
        if sample_dir is not None:
            protected = (
                (sample_dir,)
                if sample_failed or transaction.request_may_be_pending
                else ()
            )
            _prune_screenshot_storage(protected_paths=protected)
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
    *,
    victory_wait_timeout: float = VICTORY_WAIT_AFTER_HIT_SECONDS,
) -> bool:
    """Restore network immediately on hit so the pending request is submitted."""
    transaction.advance(ProbePhase.REQUEST_COMMITTED)
    update_pending_probe(phase=transaction.phase.name, request_committed=True)
    logger.info("hit detected; restoring network immediately to submit the pending request")
    transaction.advance(ProbePhase.LOGIN_RECOVERING)
    level_complete = restart_process(victory_wait_timeout=victory_wait_timeout) is True
    transaction.advance(ProbePhase.COMPLETE)
    return level_complete


def _discard_pending_request_and_prepare_next_probe(
    transaction: ProbeTransaction,
) -> bool:
    """Reject the offline connection, then reconnect through the retry prompt."""
    logger.info(
        "miss detected; enabling REJECT to trigger the connection-interrupted dialog"
    )
    # Red-scout flow may have enabled REJECT already to overlap result-frame
    # capture with the connection-dialog wait. Keep the operation idempotent
    # and preserve the original ordering for ordinary blue probes.
    if not bool(getattr(transaction, "red_reject_enabled", False)):
        adb.enable_reject_network(GAME_PACKAGE_NAME)
    write_runtime_status(network="DROP+REJECT 断网中")
    transaction.advance(ProbePhase.REQUEST_DISCARDED)
    transaction.red_request_discarded = True
    update_pending_probe(
        phase=ProbePhase.REQUEST_DISCARDED.name,
        request_discarded=True,
    )

    dialog = wait_until_connection_interrupted_dialog(
        timeout=MISS_CONNECTION_DIALOG_WAIT_SECONDS,
    )
    if dialog is None:
        reason = "未检测到连接中断弹窗；保留 DROP/REJECT、保持游戏运行并停止自动探测"
        latch_network_fail_closed(reason)
        raise DiscardRecoveryError(reason)

    retry = wait_until_retry_button(timeout=MISS_RETRY_BUTTON_WAIT_SECONDS)
    if retry is None:
        reason = "连接中断弹窗已出现，但未检测到重试按钮；保留 DROP/REJECT、保持游戏运行并停止自动探测"
        latch_network_fail_closed(reason)
        raise DiscardRecoveryError(reason)

    transaction.advance(ProbePhase.LOGIN_RECOVERING)
    logger.info(
        "retry button confirmed; restoring DROP and REJECT before clicking"
    )
    disable_weak_network()
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    logger.info(
        "clicking retry button with network restored: center=%s score=%.3f",
        retry.center,
        float(getattr(retry, "score", 0.0)),
    )
    adb.click(*retry.center)
    level_complete = enter_activity(
        re_enter=True,
        max_retries=1,
        prepare_activity_list=True,
        activity_button_timeout=POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
    ) is True
    transaction.advance(ProbePhase.COMPLETE)
    return level_complete


def restart_process(
    reopen_game: bool = False,
    app_already_closed: bool = False,
    *,
    victory_wait_timeout: float = VICTORY_WAIT_AFTER_HIT_SECONDS,
) -> bool:
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
    level_complete = handle_victory_prompt(timeout=victory_wait_timeout)
    recovered_level_complete = enter_activity() is True
    return level_complete or recovered_level_complete


def find_victory_banner(
    screenshot: np.ndarray,
    *,
    full_screen: bool = False,
) -> MatchResult | None:
    """Detect the victory banner in a screenshot."""
    if not isinstance(screenshot, np.ndarray):
        return None

    search_image = screenshot
    offset_x = 0
    offset_y = 0
    if not full_screen:
        height, width = screenshot.shape[:2]
        left, top, right, bottom = VICTORY_SEARCH_REGION
        x1 = max(0, min(width, int(round(width * left))))
        y1 = max(0, min(height, int(round(height * top))))
        x2 = max(x1, min(width, int(round(width * right))))
        y2 = max(y1, min(height, int(round(height * bottom))))
        if x2 <= x1 or y2 <= y1:
            return None
        search_image = screenshot[y1:y2, x1:x2]
        offset_x = x1
        offset_y = y1

    victory = find_template_multi_scale(
        search_image,
        VICTORY_BANNER_TEMPLATE,
        scales=VICTORY_TEMPLATE_SCALES,
        threshold=VICTORY_BANNER_THRESHOLD,
    )
    if victory is None or (offset_x == 0 and offset_y == 0):
        return victory

    return MatchResult(
        template_path=victory.template_path,
        top_left=(victory.top_left[0] + offset_x, victory.top_left[1] + offset_y),
        bottom_right=(
            victory.bottom_right[0] + offset_x,
            victory.bottom_right[1] + offset_y,
        ),
        center=(victory.center[0] + offset_x, victory.center[1] + offset_y),
        score=victory.score,
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
        if _has_pending_probe_request():
            raise ProbeProtocolError("存在待提交探测请求，禁止在胜利界面恢复网络")
        logger.info("victory banner detected; restoring network and tapping screen to continue")
        disable_weak_network()
        adb.disable_reject_network(GAME_PACKAGE_NAME)
    else:
        logger.info("victory banner detected while probe request is pending; keeping network isolated")
    adb.click(*SCREEN_CONTINUE_POINT)
    adb.delay(VICTORY_SKIP_SETTLE_SECONDS)
    return True


def handle_connection_interrupted_prompt(timeout: float = 20.0) -> bool:
    """Detect the connection-interrupted dialog, reconnect, and click retry."""
    if _has_pending_probe_request():
        raise ProbeProtocolError("存在待提交探测请求，禁止通过连接弹窗恢复网络")

    dialog = wait_until_connection_interrupted_dialog(timeout=min(4.0, float(timeout)))
    if dialog is None:
        return False

    logger.info("connection-interrupted dialog detected; reconnecting and clicking retry")
    disable_weak_network()
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    retry = wait_until_retry_button(timeout=max(0.0, float(timeout) - 4.0))
    if retry is None:
        raise ProbeProtocolError("connection-interrupted dialog found, but retry button was not found")

    adb.delay(0.8).click(*retry.center)
    return True


def _reconnect_to_base_and_reenter_activity_after_victory() -> bool:
    """通过一次完整的断网重连，把胜利后的旧详情页清理干净。"""
    if _has_pending_probe_request():
        raise ProbeProtocolError("胜利后切换下一关时仍有待提交探测请求，禁止重连")

    logger.info(
        "victory transition: enabling DROP+REJECT to return to the base before next level"
    )
    enable_weak_network(PROBE_DROP_SETTLE_SECONDS)
    _verify_network_isolated_or_fail_closed(red_scout=False)
    adb.enable_reject_network(GAME_PACKAGE_NAME)
    write_runtime_status(network="DROP+REJECT 断网中", phase="victory_reconnect")

    dialog = wait_until_connection_interrupted_dialog(
        timeout=MISS_CONNECTION_DIALOG_WAIT_SECONDS,
    )
    if dialog is None:
        reason = "胜利后重连未检测到连接中断弹窗；保持断网并停止下一关切换"
        latch_network_fail_closed(reason)
        raise ProbeProtocolError(reason)

    retry = wait_until_retry_button(timeout=MISS_RETRY_BUTTON_WAIT_SECONDS)
    if retry is None:
        reason = "胜利后连接中断弹窗未检测到重试按钮；保持断网并停止下一关切换"
        latch_network_fail_closed(reason)
        raise ProbeProtocolError(reason)

    disable_weak_network()
    adb.disable_reject_network(GAME_PACKAGE_NAME)
    logger.info(
        "victory transition: base reconnect confirmed; clicking retry center=%s",
        retry.center,
    )
    adb.click(*retry.center)
    if wait_until_occur(
        ACTIVITY_BUTTON_TEMPLATE,
        timeout=POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        poll_interval=ACTIVITY_REENTRY_POLL_INTERVAL_SECONDS,
    ) is None:
        reason = "胜利后重试已点击，但未确认回到主基地活动入口"
        latch_network_fail_closed(reason)
        raise ProbeProtocolError(reason)

    logger.info("victory transition: base screen confirmed; reopening activity list")
    enter_activity(
        prepare_activity_list=True,
        activity_button_timeout=POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
    )
    return True


def wait_until_victory_banner(timeout: float = 4.0) -> MatchResult | None:
    """Wait briefly for the victory banner shown after the final submarine is hit."""
    deadline = monotonic() + max(0.0, float(timeout))
    last_screenshot: np.ndarray | None = None
    while monotonic() < deadline:
        last_screenshot = adb.read_screenshot()
        victory = find_victory_banner(last_screenshot)
        if victory is not None:
            return victory
        sleep(0.3)
    if last_screenshot is not None:
        return find_victory_banner(last_screenshot, full_screen=True)
    return None


def _crop_normalized_region(
    screenshot: np.ndarray,
    region: tuple[float, float, float, float],
) -> tuple[np.ndarray, int, int] | None:
    if not isinstance(screenshot, np.ndarray) or screenshot.ndim < 2:
        return None
    height, width = screenshot.shape[:2]
    min_x, min_y, max_x, max_y = region
    left = max(0, min(width, int(round(width * min_x))))
    top = max(0, min(height, int(round(height * min_y))))
    right = max(left, min(width, int(round(width * max_x))))
    bottom = max(top, min(height, int(round(height * max_y))))
    if right <= left or bottom <= top:
        return None
    return screenshot[top:bottom, left:right], left, top


def _offset_match(match: MatchResult | None, left: int, top: int) -> MatchResult | None:
    if match is None:
        return None
    return MatchResult(
        template_path=match.template_path,
        top_left=(match.top_left[0] + left, match.top_left[1] + top),
        bottom_right=(match.bottom_right[0] + left, match.bottom_right[1] + top),
        center=(match.center[0] + left, match.center[1] + top),
        score=match.score,
    )


def find_connection_interrupted_dialog(screenshot: np.ndarray) -> MatchResult | None:
    """Match the complete connection prompt only inside the central screen area."""
    cropped = _crop_normalized_region(screenshot, CONNECTION_DIALOG_SEARCH_REGION)
    if cropped is None:
        return None
    roi, left, top = cropped
    dialog = find_template_multi_scale(
        roi,
        CONNECTION_INTERRUPTED_PANEL_TEMPLATE,
        scales=CONNECTION_PROMPT_SCALES,
        threshold=CONNECTION_DIALOG_THRESHOLD,
    )
    return _offset_match(dialog, left, top)


def find_connection_retry_button(
    screenshot: np.ndarray,
    *,
    require_dialog: bool = True,
) -> MatchResult | None:
    """Find or derive the retry control only after confirming the complete dialog."""
    if require_dialog:
        dialog = find_connection_interrupted_dialog(screenshot)
        if dialog is None:
            return None
        width = dialog.bottom_right[0] - dialog.top_left[0]
        height = dialog.bottom_right[1] - dialog.top_left[1]
        center = (
            dialog.top_left[0] + round(width * CONNECTION_RETRY_RELATIVE_CENTER[0]),
            dialog.top_left[1] + round(height * CONNECTION_RETRY_RELATIVE_CENTER[1]),
        )
        return MatchResult(
            template_path=CONNECTION_RETRY_TEMPLATE,
            top_left=center,
            bottom_right=center,
            center=center,
            score=dialog.score,
        )

    cropped = _crop_normalized_region(screenshot, CONNECTION_RETRY_SEARCH_REGION)
    if cropped is None:
        return None
    roi, left, top = cropped
    retry = find_template_multi_scale(
        roi,
        CONNECTION_RETRY_TEMPLATE,
        scales=CONNECTION_PROMPT_SCALES,
        threshold=CONNECTION_RETRY_THRESHOLD,
    )
    if retry is None:
        retry = find_template_multi_scale(
            roi,
            RETRY_TEMPLATE,
            scales=RETRY_TEMPLATE_SCALES,
            threshold=RETRY_TEMPLATE_LOOSE_THRESHOLD,
        )
    return _offset_match(retry, left, top)


def wait_until_connection_interrupted_dialog(timeout: float = 20.0) -> MatchResult | None:
    """Wait for a connection-interrupted dialog in the center of the screen."""
    deadline = monotonic() + max(0.0, float(timeout))
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        dialog = find_connection_interrupted_dialog(screenshot)
        if dialog is not None:
            return dialog
        sleep(FAST_POLL_INTERVAL_SECONDS)
    return None


def wait_until_retry_button(timeout: float = 20.0) -> MatchResult | None:
    """Wait until one frame contains both the dialog and its retry button."""
    deadline = monotonic() + max(0.0, float(timeout))
    while monotonic() < deadline:
        screenshot = adb.read_screenshot()
        retry = find_connection_retry_button(screenshot, require_dialog=True)
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
    *,
    poll_interval: float = FAST_POLL_INTERVAL_SECONDS,
) -> MatchResult | None:
    """等待直到指定模板出现，返回匹配结果或 None（超时）"""
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be positive: {poll_interval}")
    logger.info("正在等待模板 '%s' 出现，超时时间 %s 秒", template_path, timeout)
    start_time = monotonic()
    while monotonic() - start_time < timeout:
        screenshot = adb.read_screenshot()
        match_result = find_template(screenshot, template_path)
        if match_result is not None:
            return match_result
        sleep(poll_interval)
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
    # Always leave the completed level through the base/reconnect flow before
    # reading the next level. This prevents a stale victory/detail transition
    # frame from being treated as the next board and avoids opening a cell.
    if not _reconnect_to_base_and_reenter_activity_after_victory():
        logger.warning(
            "victory transition returned without a confirmed activity detail; "
            "refusing to probe the next level"
        )
        return None

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
        # The activity list can keep the completed level selected for a few
        # seconds while the victory transition settles.  Re-running the full
        # app reconnect path immediately costs nearly a minute and can land on
        # the login screen even though the app is healthy.  First poll the
        # current screen briefly for a genuine level advance.
        if attempt < LEVEL_ADVANCE_RETRIES:
            settle_deadline = monotonic() + min(
                6.0,
                max(1.0, float(POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS) / 4.0),
            )
            while monotonic() < settle_deadline:
                screenshot = adb.read_screenshot()
                try:
                    settled_level = resolve_current_level(
                        screenshot,
                        fallback_level=fallback_level,
                        fallback_is_manual=False,
                    )
                except Exception:
                    settled_level = current_level
                if settled_level > current_level:
                    logger.info(
                        "next level advanced during transition settle: %s",
                        settled_level,
                    )
                    return settled_level
                sleep(FAST_POLL_INTERVAL_SECONDS)
            logger.info(
                "next level still not visible after transition settle; "
                "performing one guarded base reconnect"
            )
        try:
            if not _reconnect_to_base_and_reenter_activity_after_victory():
                logger.warning(
                    "retrying next-level transition did not confirm activity detail; "
                    "refusing to probe the grid"
                )
                return None
        except Exception as exc:
            logger.warning(
                "retrying next-level transition failed through base reconnect: %s",
                exc,
            )
            return None

    return None


def _process_memory_usage_mb() -> tuple[float | None, float | None]:
    if os.name != "nt":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            return None, None
        unit = 1024 * 1024
        return counters.WorkingSetSize / unit, counters.PrivateUsage / unit
    except (AttributeError, OSError, ValueError):
        return None, None


def _log_level_memory(level: int) -> None:
    working_set, private = _process_memory_usage_mb()
    if working_set is None or private is None:
        logger.info("level %s memory: unavailable", level)
        return
    working_set = round(working_set, 1)
    private = round(private, 1)
    logger.info(
        "level %s memory: working_set=%.1f MB private=%.1f MB",
        level,
        working_set,
        private,
    )
    write_runtime_status(
        memory_working_set_mb=working_set,
        memory_private_mb=private,
    )


def main(level: int | None = None) -> Path | None:
    """执行自动探测流程并输出各关命中图。"""
    run_started_at = monotonic()
    run_started_text = datetime.now().isoformat(timespec="seconds")
    fallback_is_manual = level is not None
    fallback_level = DEFAULT_LEVEL if level is None else int(level)
    last_out_path: Path | None = None
    settings = load_red_scout_settings()
    try:
        write_runtime_status(
            running=True,
            started_at=run_started_text,
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
            red_scout_valid=0,
            red_scout_complete_six=0,
        )
        _prune_probe_sample_dirs()
        _prune_red_scout_sample_dirs()
        _prune_screenshot_storage()
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
            _log_level_memory(current_level)
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


def run_main_entrypoint() -> int:
    main_pid: int | None = None
    try:
        main_pid = acquire_main_lock()
        register_exit_cleanup()
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
        return 2
    except BlueAmmoDepletedError as exc:
        logger.warning("%s", exc)
        return 0
    except RedScoutSafetyError as exc:
        logger.critical("%s", exc)
        return 3
    finally:
        if main_pid is not None:
            cleanup_weak_network("main finished")
            cleanup_reject_network("main finished")
            release_main_lock(pid=main_pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main_entrypoint())
