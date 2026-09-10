from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path
from statistics import median
from types import MappingProxyType

import cv2
import numpy as np

from config import RED_SCOUT_DEFAULT_COUNT, RED_SCOUT_MAX_COUNT, TEMPLATE_DIR
from utils.diamond_hit import DiamondHitConfig, classify_diamond_hit, make_diamond_mask
from utils.image_match import MatchResult, find_template_multi_scale
from utils.sidebar_progress import (
    detect_sidebar_progress,
    newly_completed_lengths,
    resolve_completed_ship_cells,
    resolve_completed_ship_cells_by_anchors,
)
from utils.wreck_detection import (
    COMPLETED_SHIP_BODY_MIN_SCORE,
    completed_ship_body_score,
    detect_completed_submarine_candidate_cells,
    detect_red_submarine_marker_cells,
    detect_visible_wreck_cells,
    red_hit_marker_visible,
    red_submarine_marker_visible,
    grid_cell_polygon,
)


Cell = tuple[int, int]

RED_BOMB_TEMPLATE: Path = TEMPLATE_DIR / "red_bomb_button.png"
RED_BOMB_TEMPLATE_SCALES = (0.85, 0.95, 1.0, 1.05, 1.15)
RED_BOMB_TEMPLATE_THRESHOLD = 0.72
RED_BOMB_BUTTON_REFERENCE_SIZE = (1280, 720)
RED_BOMB_BUTTON_REFERENCE_BOUNDS = (1173, 619, 1256, 699)
RED_BOMB_SELECTION_MIN_EDGE_RATIO = 0.25
RED_BOMB_SELECTION_MIN_AVERAGE_RATIO = 0.30
FIRST_FOOTPRINT_CHANGE_THRESHOLD = 0.72
LEARNED_FOOTPRINT_CHANGE_THRESHOLD = 0.45
MINIMUM_FRAME_VOTES = 2
RED_SCOUT_RESULT_CELL_COUNT = 6
RED_SCOUT_MISS_MIN_CHANGE = 0.88
RED_SCOUT_MISS_MIN_VOTES = 3
RED_SCOUT_MISS_FALLBACK_MIN_CHANGE = 0.60
COMPLETED_SHIP_ENDPOINT_MIN_MARGIN = 0.08
# A surfaced ship's end cell is often partially occluded by the wake.  Keep
# the normal body threshold for ordinary candidates, but allow a slightly
# weaker endpoint when the sidebar has independently confirmed completion.
COMPLETED_SHIP_ENDPOINT_MIN_SCORE = 0.20
# A surfaced hull can project into an adjacent row/column in the isometric
# grid.  When the sidebar has independently confirmed completion, use the
# body score to recover the legal straight placement instead of treating the
# projection as an irregular ship.  These guards keep the fallback fail-closed.
COMPLETED_SHIP_GEOMETRY_MIN_MEAN_SCORE = 0.30
COMPLETED_SHIP_GEOMETRY_MIN_STRONG_CELLS = 3
COMPLETED_SHIP_GEOMETRY_MIN_SUPPORT_CELLS = 2
COMPLETED_SHIP_GEOMETRY_MIN_MARGIN = 0.03

# Result captures can straddle the connection-dialog animation.  These
# thresholds are intentionally conservative: a frame is discarded only when
# it is a clear outlier relative to the other captured frames.
RED_SCOUT_TRANSITION_DOWNSAMPLE = (32, 18)
RED_SCOUT_TRANSITION_PIXEL_DIFF = 24.0
RED_SCOUT_TRANSITION_MIN_DISTANCE = 0.10
RED_SCOUT_TRANSITION_MIN_CHANGED_RATIO = 0.35
RED_SCOUT_TRANSITION_OUTLIER_FACTOR = 2.5


def _infer_completed_ship_endpoints(
    body_candidates: set[Cell],
    *,
    unresolved_lengths: Sequence[int],
    grid_size: int,
    after_images: Sequence[np.ndarray],
    points_by_cell: Mapping[Cell, tuple[int, int]],
    minimum_score: float = COMPLETED_SHIP_ENDPOINT_MIN_SCORE,
) -> set[Cell]:
    inferred: set[Cell] = set()

    try:
        endpoint_min_score = float(minimum_score)
    except (TypeError, ValueError):
        return inferred
    if not np.isfinite(endpoint_min_score) or not 0.0 <= endpoint_min_score <= 1.0:
        return inferred

    def maximal_runs(values: set[int]) -> list[tuple[int, ...]]:
        runs: list[tuple[int, ...]] = []
        pending: list[int] = []
        for value in sorted(values):
            if pending and value != pending[-1] + 1:
                runs.append(tuple(pending))
                pending = []
            pending.append(value)
        if pending:
            runs.append(tuple(pending))
        return runs

    for raw_length in unresolved_lengths:
        length = int(raw_length)
        if length < 2 or length > grid_size:
            continue

        endpoint_groups: list[tuple[Cell, ...]] = []
        for row in range(grid_size):
            columns = {col for candidate_row, col in body_candidates if candidate_row == row}
            for run in maximal_runs(columns):
                if len(run) != length - 1:
                    continue
                endpoints = tuple(
                    cell
                    for cell in ((row, run[0] - 1), (row, run[-1] + 1))
                    if 0 <= cell[1] < grid_size and cell not in body_candidates
                )
                if endpoints:
                    endpoint_groups.append(endpoints)
        for col in range(grid_size):
            rows = {row for row, candidate_col in body_candidates if candidate_col == col}
            for run in maximal_runs(rows):
                if len(run) != length - 1:
                    continue
                endpoints = tuple(
                    cell
                    for cell in ((run[0] - 1, col), (run[-1] + 1, col))
                    if 0 <= cell[0] < grid_size and cell not in body_candidates
                )
                if endpoints:
                    endpoint_groups.append(endpoints)

        for endpoints in endpoint_groups:
            scored = []
            for cell in endpoints:
                point = points_by_cell.get(cell)
                if point is None:
                    continue
                polygon = grid_cell_polygon(
                    [points_by_cell[(r, c)] for r in range(grid_size) for c in range(grid_size)],
                    cell[0] * grid_size + cell[1],
                    grid_size,
                )
                scores = []
                for image in after_images:
                    try:
                        score = completed_ship_body_score(
                            image,
                            point,
                            cell_polygon=polygon,
                        )
                    except TypeError:
                        score = completed_ship_body_score(image, point)
                    scores.append(float(score))
                if scores:
                    scored.append((float(median(scores)), cell))
            if not scored:
                continue
            scored.sort(reverse=True)
            best_score, best_cell = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if (
                best_score >= endpoint_min_score
                and best_score - second_score >= COMPLETED_SHIP_ENDPOINT_MIN_MARGIN
            ):
                inferred.add(best_cell)
    return inferred


def _infer_completed_ship_body_placements(
    body_candidates: set[Cell],
    *,
    unresolved_lengths: Sequence[int],
    grid_size: int,
    after_images: Sequence[np.ndarray],
    points_by_cell: Mapping[Cell, tuple[int, int]],
    blocked_cells: set[Cell] | frozenset[Cell] = frozenset(),
) -> tuple[set[Cell], tuple[tuple[Cell, ...], ...]]:
    """Recover straight ship placements from a stable surfaced-hull projection.

    The red marker and isometric hull can straddle two visual rows, producing
    an L-shaped candidate set even though the game ship is straight.  Sidebar
    completion is required by the caller; this helper only accepts a placement
    when multiple candidate cells support it and the complete body has a clear
    score margin over the next legal placement.
    """
    inferred_cells: set[Cell] = set()
    inferred_placements: list[tuple[Cell, ...]] = []
    occupied = set(blocked_cells)
    candidates = set(body_candidates)
    try:
        images = tuple(after_images)
    except TypeError:
        return inferred_cells, tuple(inferred_placements)
    if not images:
        return inferred_cells, tuple(inferred_placements)

    for raw_length in unresolved_lengths:
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            continue
        if length < 3 or length > grid_size:
            continue

        scored: list[tuple[float, int, int, tuple[Cell, ...]]] = []

        def consider(cells: tuple[Cell, ...]) -> None:
            nonlocal scored
            placement_set = set(cells)
            if any(
                max(abs(row - used_row), abs(col - used_col)) <= 1
                for row, col in placement_set
                for used_row, used_col in occupied
            ):
                return
            support_count = len(placement_set & candidates)
            if support_count < COMPLETED_SHIP_GEOMETRY_MIN_SUPPORT_CELLS:
                return
            scores: list[float] = []
            for cell in cells:
                point = points_by_cell.get(cell)
                if point is None:
                    return
                polygon = grid_cell_polygon(
                    [points_by_cell[(r, c)] for r in range(grid_size) for c in range(grid_size)],
                    cell[0] * grid_size + cell[1],
                    grid_size,
                )
                per_frame = []
                for image in images:
                    try:
                        score = completed_ship_body_score(
                            image,
                            point,
                            cell_polygon=polygon,
                        )
                    except TypeError:
                        score = completed_ship_body_score(image, point)
                    per_frame.append(float(score))
                if not per_frame:
                    return
                scores.append(float(median(per_frame)))
            strong_count = sum(
                score >= COMPLETED_SHIP_BODY_MIN_SCORE for score in scores
            )
            if strong_count < max(
                COMPLETED_SHIP_GEOMETRY_MIN_STRONG_CELLS,
                length - 1,
            ):
                return
            mean_score = sum(scores) / float(length)
            if mean_score < COMPLETED_SHIP_GEOMETRY_MIN_MEAN_SCORE:
                return
            scored.append((mean_score, strong_count, support_count, cells))

        for row in range(grid_size):
            for start_col in range(grid_size - length + 1):
                consider(tuple((row, start_col + offset) for offset in range(length)))
        for col in range(grid_size):
            for start_row in range(grid_size - length + 1):
                consider(tuple((start_row + offset, col) for offset in range(length)))

        if not scored:
            continue
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if len(scored) > 1 and best[0] - second_score < COMPLETED_SHIP_GEOMETRY_MIN_MARGIN:
            continue
        placement = best[3]
        inferred_placements.append(placement)
        inferred_cells.update(placement)
        occupied.update(placement)

    return inferred_cells, tuple(inferred_placements)


class ProbeMode(str, Enum):
    BLUE_ONLY = "blue_only"
    RED_SCOUT = "red_scout"


@dataclass(frozen=True)
class RedScoutSettings:
    mode: ProbeMode = ProbeMode.BLUE_ONLY
    count: int = RED_SCOUT_DEFAULT_COUNT


@dataclass(frozen=True)
class RedFootprint:
    offsets: frozenset[Cell]


@dataclass(frozen=True)
class RedScoutResult:
    center_cell: Cell
    affected_cells: frozenset[Cell]
    hit_cells: frozenset[Cell]
    miss_cells: frozenset[Cell]
    unknown_cells: frozenset[Cell]
    footprint: RedFootprint | None
    valid: bool
    confidence_by_cell: Mapping[Cell, float]
    level_completed: bool = False
    invalid_reason: str | None = None
    diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_by_cell",
            MappingProxyType(dict(self.confidence_by_cell)),
        )
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics or {})),
        )


@dataclass(frozen=True)
class _CompletedShipEvidence:
    new_hit_cells: frozenset[Cell]
    ship_cells: frozenset[Cell]
    perimeter_cells: frozenset[Cell]


@dataclass(frozen=True)
class AmmoFingerprint:
    shape: tuple[int, int]
    packed_mask: bytes
    foreground_pixels: int


def load_red_scout_settings(
    environment: Mapping[str, str] | None = None,
) -> RedScoutSettings:
    values = os.environ if environment is None else environment
    raw_mode = str(values.get("BBMA_PROBE_MODE", ProbeMode.BLUE_ONLY.value)).strip()
    try:
        mode = ProbeMode(raw_mode)
    except ValueError:
        return RedScoutSettings()

    raw_count = str(
        values.get("BBMA_RED_SCOUT_COUNT", str(RED_SCOUT_DEFAULT_COUNT))
    ).strip()
    try:
        count = int(raw_count)
    except ValueError:
        count = RED_SCOUT_DEFAULT_COUNT
    if not 1 <= count <= RED_SCOUT_MAX_COUNT:
        count = RED_SCOUT_DEFAULT_COUNT
    return RedScoutSettings(mode=mode, count=count)


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _valid_screenshot(image: object) -> bool:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        return False
    if image.ndim == 2:
        return image.shape[0] > 0 and image.shape[1] > 0
    return (
        image.ndim == 3
        and image.shape[2] in (3, 4)
        and image.shape[0] > 0
        and image.shape[1] > 0
    )


def _normalize_pair(value: object) -> Cell | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 2:
        return None
    row, col = value
    if not _is_integer(row) or not _is_integer(col):
        return None
    return (row, col)


def _inside_grid(cell: Cell, grid_size: int) -> bool:
    row, col = cell
    return 0 <= row < grid_size and 0 <= col < grid_size


def _default_hit_detector(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    ignore_submarine_marker: bool = False,
) -> bool:
    # Match the visible wreck to the requested cell instead of searching the
    # whole crop for a template that can also occur in ordinary water tiles.
    # A red component attached to a surfaced submarine marks that submarine as
    # complete, but the component itself is not hit evidence; completed-ship
    # geometry handles the real hull cells separately.
    # Red submarine decorations are never positive hit evidence.  Keep this
    # guard independent of the caller's marker-ownership mode so the red
    # object cannot become a scout hit through the diamond classifier.
    if red_hit_marker_visible(image, point):
        return False
    if not ignore_submarine_marker and red_submarine_marker_visible(image, point):
        return False
    try:
        result = classify_diamond_hit(
            image,
            image,
            point,
            config=DiamondHitConfig(search_radius=2),
        )
    except Exception:
        return False
    return str(getattr(result, "state", "")).strip().lower() == "hit"


def _prefilter_candidates_by_change_upper_bound(
    *,
    before_image: np.ndarray,
    after_images: Sequence[np.ndarray],
    points_by_cell: Mapping[Cell, tuple[int, int]],
    candidates: set[Cell],
    minimum_change_threshold: float,
) -> set[Cell] | None:
    if not candidates:
        return set()
    if (
        not np.isfinite(minimum_change_threshold)
        or not 0.0 <= minimum_change_threshold <= 1.0
        or before_image.ndim != 3
        or before_image.shape[2] != 3
    ):
        return None

    frames = tuple(after_images)
    if any(
        frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[:2] != before_image.shape[:2]
        for frame in frames
    ):
        return None

    config = DiamondHitConfig()
    half_width = int(np.ceil(config.diamond_w * config.inner_scale / 2.0))
    half_height = int(np.ceil(config.diamond_h * config.inner_scale / 2.0))
    kernel = (
        make_diamond_mask(
            (half_height * 2 + 1, half_width * 2 + 1),
            (half_width, half_height),
            config.diamond_w,
            config.diamond_h,
            scale=config.inner_scale,
        )
        > 0
    ).astype(np.float32)

    # The convolution gives an upper bound for every center the exact classifier
    # may choose during refinement. Falling below the threshold here is conclusive.
    try:
        before_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
        ones = np.ones(before_gray.shape, dtype=np.float32)
        area_map = cv2.filter2D(
            ones,
            -1,
            kernel,
            anchor=(half_width, half_height),
            borderType=cv2.BORDER_CONSTANT,
        )
        upper_bound_maps: list[np.ndarray] = []
        for frame in frames:
            after_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            changed = (
                cv2.absdiff(before_gray, after_gray) >= config.diff_threshold
            ).astype(np.float32)
            changed_count = cv2.filter2D(
                changed,
                -1,
                kernel,
                anchor=(half_width, half_height),
                borderType=cv2.BORDER_CONSTANT,
            )
            upper_bound_maps.append(
                np.divide(
                    changed_count,
                    np.maximum(area_map, 1.0),
                    dtype=np.float32,
                )
            )
    except (cv2.error, TypeError, ValueError):
        return None

    height, width = before_gray.shape
    filtered: set[Cell] = set()
    for cell in sorted(candidates):
        point = points_by_cell.get(cell)
        if point is None:
            return None
        x, y = point
        if not 0 <= x < width or not 0 <= y < height:
            return None
        x1 = max(0, x - config.search_radius)
        x2 = min(width, x + config.search_radius + 1)
        y1 = max(0, y - config.search_radius)
        y2 = min(height, y + config.search_radius + 1)
        frame_upper_bounds = [
            float(np.max(change_map[y1:y2, x1:x2]))
            for change_map in upper_bound_maps
        ]
        if median(frame_upper_bounds) >= minimum_change_threshold - 1e-6:
            filtered.add(cell)
    return filtered


def _frame_luma_signature(image: object) -> np.ndarray | None:
    """Return a compact luminance signature used to spot transition frames.

    The signature deliberately keeps absolute luminance.  A connection dialog
    dims the complete scene, while ordinary water animation changes only a
    small fraction of the down-sampled pixels.  Invalid images are ignored by
    the caller rather than raising from the diagnostic path.
    """
    if not _valid_screenshot(image):
        return None
    try:
        if image.ndim == 2:
            gray = image
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        width, height = RED_SCOUT_TRANSITION_DOWNSAMPLE
        if width <= 0 or height <= 0:
            return None
        return cv2.resize(
            gray,
            (int(width), int(height)),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
    except (cv2.error, TypeError, ValueError):
        return None


def _frame_pair_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, float] | None:
    """Return (normalized distance, changed-pixel ratio) for two frames."""
    first_signature = _frame_luma_signature(first)
    second_signature = _frame_luma_signature(second)
    if first_signature is None or second_signature is None:
        return None
    if first_signature.shape != second_signature.shape:
        return None
    delta = np.abs(first_signature - second_signature)
    return (
        float(np.mean(delta) / 255.0),
        float(np.mean(delta >= RED_SCOUT_TRANSITION_PIXEL_DIFF)),
    )


def _looks_like_transition_overlay(image: np.ndarray) -> bool:
    """Detect a full-screen dim overlay without requiring a template match.

    The connection-interrupted and victory prompts dim the game scene.  A
    variance guard prevents synthetic/blank test images (and a black loading
    frame) from being treated as a dialog.
    """
    if not _valid_screenshot(image):
        return False
    try:
        signature = _frame_luma_signature(image)
        # Keep a small variance floor so blank/loading frames are ignored,
        # while still recognizing a genuinely textured scene after a strong
        # dimming overlay (which can have variance below 10 after downsampling).
        if signature is None or float(np.std(signature)) < 8.0:
            return False
        dark_ratio = float(np.mean(signature < 80.0))
        bright_ratio = float(np.mean(signature > 180.0))
        # A real dimmed game screen still has texture throughout the central
        # play area.  This guard excludes the mostly-black synthetic/sidebar
        # frames used by tests and avoids treating a loading placeholder as a
        # connection dialog.
        central = signature[
            int(signature.shape[0] * 0.20) : int(signature.shape[0] * 0.80),
            int(signature.shape[1] * 0.20) : int(signature.shape[1] * 0.80),
        ]
        central_nonzero_ratio = float(np.mean(central > 5.0))
        return (
            dark_ratio >= 0.90
            and bright_ratio <= 0.15
            and central_nonzero_ratio >= 0.80
        )
    except (TypeError, ValueError):
        return False


def _find_transition_frame_indices(
    frames: Sequence[np.ndarray],
) -> set[int]:
    """Find clear outlier/overlay frames in a result capture.

    A real result can differ substantially from the pre-click image, so the
    decision is made against *other result frames*, not against the baseline.
    At least two mutually consistent frames are required before an outlier is
    removed.  This keeps animated-but-consistent captures fail-closed.
    """
    snapshots = tuple(frames)
    if not snapshots:
        return set()

    discarded = {
        index
        for index, frame in enumerate(snapshots)
        if _looks_like_transition_overlay(frame)
    }
    if len(snapshots) < 3:
        return discarded

    pair_metrics: dict[tuple[int, int], tuple[float, float]] = {}
    for first_index in range(len(snapshots)):
        for second_index in range(first_index + 1, len(snapshots)):
            metrics = _frame_pair_metrics(
                snapshots[first_index],
                snapshots[second_index],
            )
            if metrics is None:
                return discarded
            pair_metrics[(first_index, second_index)] = metrics

    nearest_distance: list[float] = []
    nearest_changed: list[float] = []
    for index in range(len(snapshots)):
        neighbours = [
            (pair_metrics[(min(index, other), max(index, other))], other)
            for other in range(len(snapshots))
            if other != index
        ]
        (distance, changed_ratio), _other = min(
            neighbours,
            key=lambda item: item[0][0],
        )
        nearest_distance.append(distance)
        nearest_changed.append(changed_ratio)

    # Exclude the largest nearest-neighbour distance when estimating the
    # stable cluster.  If fewer than two frames remain in that cluster, no
    # automatic outlier removal is safe.
    ordered_distances = sorted(nearest_distance)
    stable_reference = float(median(ordered_distances[:-1]))
    stable_cutoff = max(
        RED_SCOUT_TRANSITION_MIN_DISTANCE,
        stable_reference * RED_SCOUT_TRANSITION_OUTLIER_FACTOR,
    )
    stable_indices = {
        index
        for index, distance in enumerate(nearest_distance)
        if distance < stable_cutoff
    }
    if len(stable_indices) < MINIMUM_FRAME_VOTES:
        return discarded

    for index, distance in enumerate(nearest_distance):
        if (
            distance >= stable_cutoff
            and nearest_changed[index] >= RED_SCOUT_TRANSITION_MIN_CHANGED_RATIO
        ):
            discarded.add(index)
    return discarded


def _filter_transition_frames(
    frames: Sequence[np.ndarray],
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    """Drop only confidently transient result frames.

    Returning an empty frame tuple signals that fewer than the required two
    stable frames remain; the analyzer then returns an invalid result instead
    of silently accepting a single animation frame.
    """
    snapshots = tuple(frames)
    discarded = _find_transition_frame_indices(snapshots)
    if not discarded:
        return snapshots, ()
    retained = tuple(
        frame
        for index, frame in enumerate(snapshots)
        if index not in discarded
    )
    if len(retained) < MINIMUM_FRAME_VOTES:
        return (), tuple(sorted(discarded))
    return retained, tuple(sorted(discarded))


def _required_strong_miss_votes(frame_count: int) -> int:
    """Return the minimum vote count for a stable miss decision.

    Captures normally contain three frames.  When one transition frame is
    discarded, two good frames remain and both must agree; never lower the
    requirement below the global two-frame evidence floor.
    """
    try:
        count = int(frame_count)
    except (TypeError, ValueError):
        return RED_SCOUT_MISS_MIN_VOTES
    if count <= 0:
        return RED_SCOUT_MISS_MIN_VOTES
    return max(
        MINIMUM_FRAME_VOTES,
        min(RED_SCOUT_MISS_MIN_VOTES, count),
    )


def _consistent_strong_miss_cells(
    *,
    median_change_by_cell: Mapping[Cell, float],
    states_by_cell: Mapping[Cell, tuple[str, ...]],
    frame_count: int,
    excluded_cells: set[Cell] | frozenset[Cell] = frozenset(),
) -> set[Cell]:
    """Select strong misses supported consistently across result frames.

    A single-frame ``miss`` caused by an explosion or page redraw must not
    become part of the six-cell footprint.  In addition to the change
    threshold, require enough miss votes and reject any contradictory hit vote.
    """
    required_votes = _required_strong_miss_votes(frame_count)
    excluded = set(excluded_cells)
    selected: set[Cell] = set()
    for cell, changed_ratio in median_change_by_cell.items():
        if cell in excluded:
            continue
        try:
            ratio = float(changed_ratio)
        except (TypeError, ValueError):
            continue
        states = tuple(str(state).strip().lower() for state in states_by_cell.get(cell, ()))
        miss_votes = states.count("miss")
        hit_votes = states.count("hit")
        if (
            np.isfinite(ratio)
            and ratio >= RED_SCOUT_MISS_MIN_CHANGE
            and miss_votes >= required_votes
            and hit_votes == 0
        ):
            selected.add(cell)
    return selected


class RedScoutAnalyzer:
    def __init__(
        self,
        classifier: Callable[..., object] = classify_diamond_hit,
        hit_detector: Callable[[np.ndarray, tuple[int, int]], bool] = (
            _default_hit_detector
        ),
    ) -> None:
        if not callable(classifier):
            raise TypeError("classifier must be callable")
        if not callable(hit_detector):
            raise TypeError("hit_detector must be callable")
        self._classifier = classifier
        self._hit_detector = hit_detector
        self._marker_points: tuple[tuple[int, int], ...] = ()
        self._marker_grid_size = 0

    def _detect_hit(self, image: np.ndarray, point: tuple[int, int]) -> bool:
        """Run hit detection with global red-marker ownership when possible."""
        if self._hit_detector is not _default_hit_detector:
            return bool(self._hit_detector(image, point))
        if not self._marker_points or self._marker_grid_size <= 0:
            return _default_hit_detector(image, point)
        marker_cells = detect_red_submarine_marker_cells(
            image,
            list(self._marker_points),
            self._marker_grid_size,
        )
        nearest_cell = min(
            (
                (index // self._marker_grid_size, index % self._marker_grid_size)
                for index in range(len(self._marker_points))
            ),
            key=lambda cell: (
                (self._marker_points[cell[0] * self._marker_grid_size + cell[1]][0] - point[0]) ** 2
                + (self._marker_points[cell[0] * self._marker_grid_size + cell[1]][1] - point[1]) ** 2
            ),
        )
        return _default_hit_detector(
            image,
            point,
            ignore_submarine_marker=nearest_cell not in marker_cells,
        )

    def analyze(
        self,
        before_image: np.ndarray,
        after_images: Sequence[np.ndarray],
        grid_size: int,
        click_points: Sequence[tuple[int, int]],
        center_cell: Cell,
        excluded_cells: Sequence[Cell] | set[Cell] | frozenset[Cell] = (),
        learned_footprint: RedFootprint | None = None,
        submarine_lengths: Sequence[int] = (),
    ) -> RedScoutResult:
        normalized_center = _normalize_pair(center_cell)
        result_center = normalized_center if normalized_center is not None else (0, 0)
        preflight = self._preflight(
            before_image=before_image,
            after_images=after_images,
            grid_size=grid_size,
            click_points=click_points,
            center_cell=normalized_center,
            excluded_cells=excluded_cells,
            learned_footprint=learned_footprint,
        )
        if preflight is None:
            return self._invalid_result(
                result_center,
                reason="preflight_failed",
                diagnostics={"stage": "preflight"},
            )

        raw_frames, points_by_cell, excluded, learned_offsets = preflight
        self._marker_points = tuple((int(x), int(y)) for x, y in click_points)
        self._marker_grid_size = int(grid_size)
        frames, transition_frame_indices = _filter_transition_frames(raw_frames)
        diagnostics: dict[str, object] = {
            "stage": "candidate_detection",
            "center": result_center,
            "excluded_cells": tuple(sorted(excluded)),
            "raw_frame_count": len(raw_frames),
            "frame_count": len(frames),
            "transition_frame_indices": transition_frame_indices,
            "learned_footprint": (
                tuple(sorted(learned_offsets))
                if learned_offsets is not None
                else ()
            ),
        }
        if len(frames) < MINIMUM_FRAME_VOTES:
            diagnostics["stage"] = "transition_frame_filter"
            diagnostics["retained_frame_count"] = len(frames)
            return self._invalid_result(
                result_center,
                reason="transition_frames_insufficient",
                diagnostics=diagnostics,
            )
        # A learned footprint is a planning hint, not a fixed description of
        # every later red-bomb result. The affected cells can vary with the
        # target position, so every unknown cell must be considered by analysis.
        candidates = {
            cell
            for cell in points_by_cell
            if cell not in excluded
        }

        minimum_change_threshold = min(
            FIRST_FOOTPRINT_CHANGE_THRESHOLD
            if learned_offsets is None
            else LEARNED_FOOTPRINT_CHANGE_THRESHOLD,
            RED_SCOUT_MISS_FALLBACK_MIN_CHANGE,
        )
        before_visible = self._before_visible_hit_cells(
            before_image=before_image,
            points_by_cell=points_by_cell,
            candidates=candidates,
        )
        if before_visible is None:
            diagnostics["stage"] = "before_visible_hits"
            return self._invalid_result(
                result_center,
                reason="before_hit_detection_failed",
                diagnostics=diagnostics,
            )
        diagnostics["before_visible"] = tuple(sorted(before_visible))
        raw_stable_result_hits = self._stable_visible_hit_cells(
            after_images=frames,
            points_by_cell=points_by_cell,
            candidates=candidates - before_visible,
        )
        if raw_stable_result_hits is None:
            diagnostics["stage"] = "stable_result_hits"
            return self._invalid_result(
                result_center,
                reason="stable_hit_detection_failed",
                diagnostics=diagnostics,
            )
        diagnostics["raw_stable_hits"] = tuple(sorted(raw_stable_result_hits))
        evidence = self._collect_evidence(
            before_image=before_image,
            after_images=frames,
            points_by_cell=points_by_cell,
            candidates=candidates,
            minimum_change_threshold=minimum_change_threshold,
            mandatory_candidates=raw_stable_result_hits,
        )
        if evidence is None:
            diagnostics["stage"] = "cell_evidence"
            return self._invalid_result(
                result_center,
                reason="evidence_collection_failed",
                diagnostics=diagnostics,
            )
        median_change_by_cell, states_by_cell = evidence
        diagnostics["cell_evidence"] = tuple(
            {
                "cell": cell,
                "median_change": float(median_change_by_cell[cell]),
                "states": states_by_cell[cell],
            }
            for cell in sorted(median_change_by_cell)
        )
        completed_diagnostics: dict[str, object] = {}
        completed_ship = self._completed_ship_evidence(
            before_image=before_image,
            after_images=frames,
            submarine_lengths=submarine_lengths,
            before_visible=before_visible,
            raw_stable_result_hits=raw_stable_result_hits,
            grid_size=grid_size,
            points_by_cell=points_by_cell,
            eligible_cells=candidates,
            diagnostics=completed_diagnostics,
        )
        diagnostics.update(completed_diagnostics)
        completed_visual_zone: set[Cell] = set()
        independent_stable_hits = set(raw_stable_result_hits)
        if completed_ship is not None:
            completed_visual_zone = set(
                completed_ship.ship_cells | completed_ship.perimeter_cells
            )
            independent_stable_hits.difference_update(completed_visual_zone)

        stable_result_hits = self._collapse_completed_submarine_hits(
            independent_stable_hits,
            center_cell=result_center,
            confidence_by_cell=median_change_by_cell,
        )
        if completed_ship is not None:
            stable_result_hits.update(completed_ship.new_hit_cells)
            authoritative_states = tuple("hit" for _ in frames)
            for cell in completed_ship.new_hit_cells:
                median_change_by_cell.setdefault(cell, 1.0)
                states_by_cell.setdefault(cell, authoritative_states)
        diagnostics["resolved_ship_hits"] = tuple(sorted(stable_result_hits))

        strong_result_misses = _consistent_strong_miss_cells(
            median_change_by_cell=median_change_by_cell,
            states_by_cell=states_by_cell,
            frame_count=len(frames),
            excluded_cells=(
                raw_stable_result_hits
                | completed_visual_zone
            ),
        )
        diagnostics["strong_miss_required_votes"] = _required_strong_miss_votes(
            len(frames)
        )
        diagnostics["strong_misses"] = tuple(sorted(strong_result_misses))
        affected = stable_result_hits | strong_result_misses
        if before_visible:
            affected = affected - before_visible
        # Remove the raised-flag cell before enforcing the six-cell red-bomb
        # limit.  Without this early pass, the flag artifact itself can make a
        # genuine six-cell result look like seven affected cells and invalidate
        # the whole scout attempt.
        early_flag_cells = self._find_down_right_flag_overlap_cells(
            stable_result_hits,
            affected,
        )
        if early_flag_cells:
            affected.difference_update(early_flag_cells)
            stable_result_hits.difference_update(early_flag_cells)
            strong_result_misses.difference_update(early_flag_cells)
            diagnostics["down_right_flag_discarded"] = tuple(
                sorted(early_flag_cells)
            )
            diagnostics["down_right_flag_forced_misses"] = tuple(
                sorted(early_flag_cells)
            )
        diagnostics["affected_before_limit"] = tuple(sorted(affected))
        if len(affected) > RED_SCOUT_RESULT_CELL_COUNT:
            if completed_ship is None:
                diagnostics["stage"] = "limit_strong_cells"
                # A noisy red result must not erase a stable hit.  The six
                # cell cap limits miss/perimeter evidence only; every hit
                # supported by all result frames remains available for the
                # blue confirmation path.  Returning a non-valid result is
                # intentional: the footprint is rejected, while its hit
                # cells are still merged into the board transaction.
                retained_hits = frozenset(stable_result_hits)
                retained_confidence = {
                    cell: median_change_by_cell[cell]
                    for cell in retained_hits
                    if cell in median_change_by_cell
                }
                diagnostics["retained_hits_after_limit"] = tuple(
                    sorted(retained_hits)
                )
                diagnostics["final_hits"] = tuple(sorted(retained_hits))
                diagnostics["final_misses"] = ()
                diagnostics["final_unknown"] = ()
                return RedScoutResult(
                    center_cell=result_center,
                    affected_cells=retained_hits,
                    hit_cells=retained_hits,
                    miss_cells=frozenset(),
                    unknown_cells=frozenset(),
                    footprint=None,
                    valid=False,
                    confidence_by_cell=retained_confidence,
                    invalid_reason="too_many_strong_cells",
                    diagnostics=diagnostics,
                )
            # Surfacing a completed ship changes its entire body, so authoritative
            # ship hits may legitimately outnumber the bomb's six result cells.
            remaining_slots = max(
                0,
                RED_SCOUT_RESULT_CELL_COUNT - len(stable_result_hits),
            )
            selected_misses = set(
                sorted(
                    strong_result_misses,
                    key=lambda cell: (
                        -median_change_by_cell[cell],
                        cell[0],
                        cell[1],
                    ),
                )[:remaining_slots]
            )
            diagnostics["trimmed_strong_misses"] = tuple(
                sorted(strong_result_misses - selected_misses)
            )
            strong_result_misses = selected_misses
            affected = stable_result_hits | strong_result_misses

        consistent_misses: list[Cell] = []
        if len(affected) < RED_SCOUT_RESULT_CELL_COUNT:
            consistent_misses = sorted(
                (
                    cell
                    for cell, changed_ratio in median_change_by_cell.items()
                    if (
                        cell not in affected
                        and cell not in before_visible
                        and cell not in raw_stable_result_hits
                        and cell not in completed_visual_zone
                        and changed_ratio >= RED_SCOUT_MISS_FALLBACK_MIN_CHANGE
                        and states_by_cell[cell]
                        and states_by_cell[cell].count("miss")
                        == len(states_by_cell[cell])
                    )
                ),
                key=lambda cell: (
                    -median_change_by_cell[cell],
                    cell[0],
                    cell[1],
                ),
            )
            missing_count = RED_SCOUT_RESULT_CELL_COUNT - len(affected)
            affected.update(consistent_misses[:missing_count])
        diagnostics["moderate_misses"] = tuple(consistent_misses)

        safe_perimeter_misses: list[Cell] = []
        if (
            completed_ship is not None
            and len(affected) < RED_SCOUT_RESULT_CELL_COUNT
        ):
            safe_perimeter_misses = sorted(
                (
                    cell
                    for cell in completed_ship.perimeter_cells
                    if (
                        cell in median_change_by_cell
                        and cell not in before_visible
                        and cell not in raw_stable_result_hits
                        and median_change_by_cell[cell]
                        >= RED_SCOUT_MISS_FALLBACK_MIN_CHANGE
                        and states_by_cell[cell]
                        and states_by_cell[cell].count("miss")
                        == len(states_by_cell[cell])
                    )
                ),
                key=lambda cell: (
                    -median_change_by_cell[cell],
                    cell[0],
                    cell[1],
                ),
            )
            missing_count = RED_SCOUT_RESULT_CELL_COUNT - len(affected)
            affected.update(safe_perimeter_misses[:missing_count])
        diagnostics["completed_perimeter_candidates"] = tuple(
            safe_perimeter_misses
        )
        diagnostics["final_affected"] = tuple(sorted(affected))

        if learned_offsets is None:
            valid = len(affected) >= 2
            footprint = (
                RedFootprint(
                    offsets=frozenset(
                        (
                            row - result_center[0],
                            col - result_center[1],
                        )
                        for row, col in affected
                    )
                )
                if valid
                else None
            )
        else:
            valid = bool(affected)
            footprint = learned_footprint

        classified = self._classify_affected_cells(
            after_images=frames,
            points_by_cell=points_by_cell,
            affected=affected,
            states_by_cell=states_by_cell,
        )
        if classified is None:
            diagnostics["stage"] = "classify_affected_cells"
            return self._invalid_result(
                result_center,
                reason="result_classification_failed",
                diagnostics=diagnostics,
            )
        hit_cells, miss_cells, unknown_cells = classified
        # The discarded flag cell is known to be a visual false positive. Keep
        # it as a miss observation for the board even though it is excluded
        # from the six-cell footprint used by the planner.
        miss_cells.update(early_flag_cells)
        unknown_cells.difference_update(early_flag_cells)
        if completed_ship is not None:
            authoritative_hits = set(completed_ship.new_hit_cells) & affected
            hit_cells.update(authoritative_hits)
            miss_cells.difference_update(authoritative_hits)
            unknown_cells.difference_update(authoritative_hits)

        # Perspective overlap can make two cells above a real diagonal pair
        # look like hits.  A submarine cannot occupy a diagonal, so when the
        # result is an exact top-left -> bottom-right run, keep the lower pair
        # only if it has independent hit evidence and discard the upper noise.
        diagonal_discarded = self._filter_diagonal_overlap_hits(
            hit_cells=hit_cells,
            affected=affected,
            confidence_by_cell=median_change_by_cell,
            states_by_cell=states_by_cell,
        )
        if diagonal_discarded:
            diagnostics["diagonal_overlap_discarded"] = tuple(
                sorted(diagonal_discarded)
            )
        flag_discarded = self._filter_down_right_flag_overlap_hits(
            hit_cells=hit_cells,
            miss_cells=miss_cells,
            unknown_cells=unknown_cells,
        )
        if flag_discarded:
            diagnostics["down_right_flag_discarded"] = tuple(
                sorted(flag_discarded)
            )

        confidence_by_cell = {
            cell: median_change_by_cell[cell]
            for cell in sorted(affected)
        }
        diagnostics.update(
            {
                "stage": "complete" if valid else "insufficient_changes",
                "final_hits": tuple(sorted(hit_cells)),
                "final_misses": tuple(sorted(miss_cells)),
                "final_unknown": tuple(sorted(unknown_cells)),
            }
        )
        return RedScoutResult(
            center_cell=result_center,
            affected_cells=frozenset(affected),
            hit_cells=frozenset(hit_cells),
            miss_cells=frozenset(miss_cells),
            unknown_cells=frozenset(unknown_cells),
            footprint=footprint,
            valid=valid,
            confidence_by_cell=MappingProxyType(dict(confidence_by_cell)),
            invalid_reason=None if valid else "insufficient_changed_cells",
            diagnostics=diagnostics,
        )

    @staticmethod
    def _filter_diagonal_overlap_hits(
        *,
        hit_cells: set[Cell],
        affected: set[Cell],
        confidence_by_cell: Mapping[Cell, float],
        states_by_cell: Mapping[Cell, tuple[str, ...]],
    ) -> set[Cell]:
        """Drop upper false hits from the known diagonal-overlap layout.

        The filter is deliberately narrow: it requires at least four hits on
        one exact r-c diagonal and two contiguous lower cells with stronger,
        multi-frame hit evidence.  Other layouts are left untouched.
        """
        if len(hit_cells) < 4:
            return set()
        groups: dict[int, list[Cell]] = {}
        for cell in hit_cells:
            row, col = cell
            groups.setdefault(row - col, []).append(cell)
        for cells in groups.values():
            ordered = sorted(cells)
            if len(ordered) < 4:
                continue
            lower = ordered[-2:]
            if not (
                lower[1][0] == lower[0][0] + 1
                and lower[1][1] == lower[0][1] + 1
            ):
                continue
            upper = ordered[:-2]
            lower_votes = [
                states_by_cell.get(cell, ()).count("hit") for cell in lower
            ]
            upper_votes = [
                states_by_cell.get(cell, ()).count("hit") for cell in upper
            ]
            lower_confidence = min(
                float(confidence_by_cell.get(cell, 0.0)) for cell in lower
            )
            upper_confidence = max(
                float(confidence_by_cell.get(cell, 0.0)) for cell in upper
            )
            if (
                min(lower_votes) >= 2
                and max(upper_votes, default=0) <= min(lower_votes)
                and lower_confidence >= upper_confidence
            ):
                discarded = set(upper)
                hit_cells.difference_update(discarded)
                affected.difference_update(discarded)
                return discarded
        return set()

    @staticmethod
    def _filter_down_right_flag_overlap_hits(
        *,
        hit_cells: set[Cell],
        miss_cells: set[Cell],
        unknown_cells: set[Cell],
    ) -> set[Cell]:
        """Convert the raised-flag cell above a down-right ship to a miss.

        A screen-down-right submarine can produce an L-shaped visual result:
        one cell on the upper row (the raised red flag) and two adjacent cells
        on the row below (the actual hull).  Since ships are strictly straight,
        the upper cell cannot be a third submarine cell.  Apply this narrowly
        to any matching three-hit subset before blue targets are queued.
        """
        if len(hit_cells) < 3:
            return set()
        by_row: dict[int, set[Cell]] = {}
        for cell in hit_cells:
            by_row.setdefault(cell[0], set()).add(cell)
        for upper_row, upper_cells in sorted(by_row.items()):
            if len(upper_cells) != 1:
                continue
            lower_cells = by_row.get(upper_row + 1, set())
            if len(lower_cells) < 2:
                continue
            lower_cols = sorted(col for _row, col in lower_cells)
            for left_col in lower_cols:
                pair = {(upper_row + 1, left_col), (upper_row + 1, left_col + 1)}
                if pair.issubset(hit_cells) and next(iter(upper_cells))[1] in {
                    left_col,
                    left_col + 1,
                }:
                    false_cell = next(iter(upper_cells))
                    hit_cells.discard(false_cell)
                    miss_cells.add(false_cell)
                    unknown_cells.discard(false_cell)
                    return {false_cell}
        return set()

    @staticmethod
    def _find_down_right_flag_overlap_cells(
        hit_cells: set[Cell],
        affected: set[Cell],
    ) -> set[Cell]:
        """Find upper flag cells before the affected-cell limit is enforced."""
        if len(hit_cells) < 2:
            return set()
        discarded: set[Cell] = set()
        for upper_row, upper_col in sorted(affected):
            upper = (upper_row, upper_col)
            # The raised flag can appear above either end of the two-cell
            # horizontal hull. Check both possible pair anchors.
            for left_col in (upper_col - 1, upper_col):
                pair = {
                    (upper_row + 1, left_col),
                    (upper_row + 1, left_col + 1),
                }
                if pair.issubset(hit_cells):
                    discarded.add(upper)
                    break
        return discarded

    @staticmethod
    def _completed_ship_evidence(
        *,
        before_image: np.ndarray,
        after_images: tuple[np.ndarray, ...],
        submarine_lengths: Sequence[int],
        before_visible: set[Cell],
        raw_stable_result_hits: set[Cell],
        grid_size: int,
        points_by_cell: Mapping[Cell, tuple[int, int]],
        eligible_cells: set[Cell],
        diagnostics: dict[str, object] | None = None,
    ) -> _CompletedShipEvidence | None:
        details = diagnostics if diagnostics is not None else {}
        details.update(
            {
                "completed_sidebar_votes": (),
                "completed_lengths": (),
                "resolved_ship_placements": (),
                "completed_perimeter": (),
                "completed_body_candidates": (),
                "completed_ship_failure": None,
            }
        )
        try:
            lengths = tuple(int(length) for length in submarine_lengths)
        except (TypeError, ValueError):
            details["completed_ship_failure"] = "invalid_submarine_lengths"
            return None
        if not lengths or any(length <= 0 for length in lengths):
            details["completed_ship_failure"] = "submarine_lengths_unavailable"
            return None

        before_progress = detect_sidebar_progress(before_image, lengths)
        if before_progress is None or not before_progress.valid:
            details["completed_ship_failure"] = "before_sidebar_unavailable"
            return None

        completion_votes: Counter[tuple[int, ...]] = Counter()
        for after_image in after_images:
            after_progress = detect_sidebar_progress(after_image, lengths)
            completed = newly_completed_lengths(before_progress, after_progress)
            if completed:
                completion_votes[completed] += 1
        details["completed_sidebar_votes"] = tuple(
            {
                "lengths": completed_lengths,
                "votes": int(votes),
            }
            for completed_lengths, votes in sorted(completion_votes.items())
        )
        if not completion_votes:
            details["completed_ship_failure"] = "no_sidebar_completion"
            return None

        completed_lengths, votes = min(
            completion_votes.items(),
            key=lambda item: (-item[1], item[0]),
        )
        details["completed_lengths"] = completed_lengths
        if votes < MINIMUM_FRAME_VOTES:
            details["completed_ship_failure"] = "insufficient_sidebar_votes"
            return None

        click_points = [
            points_by_cell[(row, col)]
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        stable_body_candidates: set[Cell] = set()
        stable_red_anchor_cells: set[Cell] = set()
        try:
            before_body_candidates = detect_completed_submarine_candidate_cells(
                before_image,
                click_points,
                grid_size,
            )
            body_votes: Counter[Cell] = Counter()
            for after_image in after_images:
                body_votes.update(
                    detect_completed_submarine_candidate_cells(
                        after_image,
                        click_points,
                        grid_size,
                    )
                )
            stable_body_candidates = {
                cell
                for cell, candidate_votes in body_votes.items()
                if candidate_votes >= MINIMUM_FRAME_VOTES
                and cell not in before_body_candidates
            }
            before_red_anchors = detect_red_submarine_marker_cells(
                before_image,
                click_points,
                grid_size,
            )
            anchor_votes: Counter[Cell] = Counter()
            for after_image in after_images:
                anchor_votes.update(
                    detect_red_submarine_marker_cells(
                        after_image,
                        click_points,
                        grid_size,
                    )
                )
            stable_red_anchor_cells = {
                cell
                for cell, candidate_votes in anchor_votes.items()
                if candidate_votes >= MINIMUM_FRAME_VOTES
                and cell not in before_red_anchors
            }
        except Exception:
            details["completed_body_detection_failed"] = True
        details["completed_body_candidates"] = tuple(
            sorted(stable_body_candidates)
        )
        details["completed_red_anchor_cells"] = tuple(
            sorted(stable_red_anchor_cells)
        )
        details["inferred_ship_body_placements"] = ()
        details["completed_body_overrides"] = tuple(
            sorted(stable_body_candidates - eligible_cells)
        )

        # A sidebar color change says that some submarine completed, but it
        # does not locate that submarine.  Only a stable red surfaced-submarine
        # marker and its nearby hull evidence may promote coordinates to a
        # complete placement.  Without the red object, keep ordinary wrecks as
        # provisional hits so an unrelated old hit line cannot steal the newly
        # completed fleet length.
        if not stable_body_candidates:
            details["completed_ship_failure"] = "missing_red_body_evidence"
            return None

        # Resolve all newly completed ships from the red-marked hull evidence
        # before allowing raw hit coordinates to participate.  In a frame that
        # surfaces multiple ships, a short exact run can otherwise greedily
        # consume a hit cell that belongs to a longer ship and make the longer
        # placement impossible to recover.
        geometry_candidates = set(stable_body_candidates)
        anchor_binding_expected = (
            len(completed_lengths) > 1
            and len(stable_red_anchor_cells) == len(completed_lengths)
        )
        anchor_resolution = None

        def try_anchor_resolution(candidates: set[Cell]):
            if not anchor_binding_expected:
                return None
            candidate_resolution = resolve_completed_ship_cells_by_anchors(
                candidates,
                stable_red_anchor_cells,
                completed_lengths,
                grid_size=grid_size,
                preferred_cells=raw_stable_result_hits - before_visible,
                # Do not silently replace a tied/failed anchor assignment with
                # a global geometry guess; that can swap two completed lengths.
                fallback_to_global=False,
            )
            if (
                candidate_resolution.unresolved_lengths
                or len(candidate_resolution.placements) != len(completed_lengths)
            ):
                return None
            sorted_anchors = tuple(sorted(stable_red_anchor_cells))
            if not all(
                any(
                    max(abs(row - anchor[0]), abs(col - anchor[1])) <= 1
                    for row, col in placement
                )
                for anchor, placement in zip(
                    sorted_anchors,
                    candidate_resolution.placements,
                    strict=True,
                )
            ):
                return None
            return candidate_resolution

        if geometry_candidates:
            inferred_geometry_cells, inferred_geometry_placements = (
                _infer_completed_ship_body_placements(
                    geometry_candidates | raw_stable_result_hits,
                    unresolved_lengths=completed_lengths,
                    grid_size=grid_size,
                    after_images=after_images,
                    points_by_cell=points_by_cell,
                )
            )
            if inferred_geometry_cells:
                details["inferred_ship_body_placements"] = tuple(
                    tuple(sorted(placement))
                    for placement in inferred_geometry_placements
                )
                geometry_candidates.update(inferred_geometry_cells)

            anchor_resolution = try_anchor_resolution(geometry_candidates)
            if anchor_resolution is not None:
                resolution = anchor_resolution
                details["completed_resolution_mode"] = "red_anchor_length_binding"
            else:
                resolution = resolve_completed_ship_cells(
                    geometry_candidates,
                    completed_lengths,
                    grid_size=grid_size,
                    preferred_cells=raw_stable_result_hits - before_visible,
                )
        else:
            # Preserve the legacy perimeter/spill filtering when the marker
            # detector misses a frame.  The caller still requires an exact
            # straight placement; no geometry can be inferred from an empty
            # candidate set.
            raw_resolution = resolve_completed_ship_cells(
                before_visible | raw_stable_result_hits,
                completed_lengths,
                grid_size=grid_size,
                preferred_cells=raw_stable_result_hits - before_visible,
            )
            # A plain straight run of changed cells is still ordinary hit
            # evidence.  Without an independently detected red component or
            # gray hull body, do not promote that run to a completed ship just
            # because the sidebar color classifier changed for one frame.
            raw_exact_completion = any(
                set(placement).issubset(raw_stable_result_hits)
                for placement in raw_resolution.placements
            )
            if raw_exact_completion:
                details["completed_ship_failure"] = "missing_red_body_evidence"
                resolution = resolve_completed_ship_cells(
                    set(),
                    completed_lengths,
                    grid_size=grid_size,
                )
            else:
                resolution = raw_resolution
        inferred_endpoints: set[Cell] = set()
        if resolution.unresolved_lengths:
            inferred_endpoints = _infer_completed_ship_endpoints(
                stable_body_candidates,
                unresolved_lengths=resolution.unresolved_lengths,
                grid_size=grid_size,
                after_images=after_images,
                points_by_cell=points_by_cell,
                # The sidebar completion vote is an independent confirmation
                # that justifies the relaxed endpoint threshold.  The helper
                # still requires a contiguous length-1 run and a clear score
                # margin before adding either endpoint.
                minimum_score=COMPLETED_SHIP_ENDPOINT_MIN_SCORE,
            )
            if inferred_endpoints:
                geometry_candidates.update(inferred_endpoints)
                anchor_resolution = try_anchor_resolution(geometry_candidates)
                if anchor_resolution is not None:
                    resolution = anchor_resolution
                    details["completed_resolution_mode"] = "red_anchor_length_binding"
                else:
                    resolution = resolve_completed_ship_cells(
                        geometry_candidates,
                        completed_lengths,
                        grid_size=grid_size,
                        preferred_cells=(
                            raw_stable_result_hits
                            - before_visible
                            | inferred_endpoints
                        ),
                    )
        if resolution.unresolved_lengths:
            inferred_body_cells, inferred_body_placements = (
                _infer_completed_ship_body_placements(
                    stable_body_candidates,
                    unresolved_lengths=resolution.unresolved_lengths,
                    grid_size=grid_size,
                    after_images=after_images,
                    points_by_cell=points_by_cell,
                    blocked_cells=resolution.cells,
                )
            )
            if inferred_body_cells:
                details["inferred_ship_body_placements"] = tuple(
                    tuple(sorted(placement))
                    for placement in inferred_body_placements
                )
                geometry_candidates.update(inferred_body_cells)
                anchor_resolution = try_anchor_resolution(geometry_candidates)
                if anchor_resolution is not None:
                    resolution = anchor_resolution
                    details["completed_resolution_mode"] = "red_anchor_length_binding"
                else:
                    resolution = resolve_completed_ship_cells(
                        geometry_candidates,
                        completed_lengths,
                        grid_size=grid_size,
                        preferred_cells=(
                            raw_stable_result_hits
                            - before_visible
                            | inferred_body_cells
                        ),
                    )
        details["inferred_ship_endpoints"] = tuple(sorted(inferred_endpoints))
        details["resolved_ship_placements"] = resolution.placements
        details["unresolved_ship_lengths"] = resolution.unresolved_lengths
        details["discarded_ship_cells"] = tuple(sorted(resolution.discarded_cells))
        if anchor_binding_expected and anchor_resolution is None:
            details["completed_ship_failure"] = (
                "multi_completion_geometry_ambiguous"
            )
            details["ambiguous_completed_cells"] = tuple(
                sorted(resolution.cells or geometry_candidates)
            )
            # Do not leave the rejected global guess in diagnostics.  Keeping
            # it visible makes downstream recovery code look as if a valid
            # placement exists even though this result is deliberately failed.
            details["resolved_ship_placements"] = ()
            details["unresolved_ship_lengths"] = completed_lengths
            details["discarded_ship_cells"] = tuple(sorted(geometry_candidates))
            return None
        if resolution.unresolved_lengths:
            if details.get("completed_ship_failure") is None:
                details["completed_ship_failure"] = "ship_geometry_unresolved"
            return None

        # A completed sidebar entry proves that a submarine was finished, but
        # it does not make every visually plausible hull extension a confirmed
        # board cell.  The isometric model can span neighbouring diamonds.
        # Require all but at most one cell of every placement to be directly
        # present in the stable body mask from this red request.  A surfaced
        # 10x10 submarine can legitimately expose only its middle two cells,
        # though: the remaining hull is covered by the isometric projection
        # while the sidebar and red flag still identify the completed ship.
        # Permit that specific case only when a stable red anchor is adjacent
        # to the inferred straight placement and at least two body cells are
        # directly supported.  Without an anchor, keep the strict rule so a
        # transient two-cell wreck cannot manufacture a full ship.
        under_supported: list[tuple[Cell, ...]] = []
        relaxed_partial_support = False
        for placement in resolution.placements:
            support_count = len(set(placement) & stable_body_candidates)
            if support_count >= len(placement) - 1:
                continue
            anchor_bound = any(
                max(abs(row - anchor_row), abs(col - anchor_col)) <= 1
                for anchor_row, anchor_col in stable_red_anchor_cells
                for row, col in placement
            )
            minimum_partial_support = max(2, len(placement) - 2)
            if anchor_bound and support_count >= minimum_partial_support:
                relaxed_partial_support = True
                continue
            under_supported.append(placement)
        under_supported_placements = tuple(under_supported)
        if relaxed_partial_support and not under_supported_placements:
            details["completed_resolution_mode"] = (
                "red_anchor_partial_body_support"
            )
        if under_supported_placements:
            details["completed_ship_failure"] = "insufficient_direct_body_support"
            details["ambiguous_completed_cells"] = tuple(
                sorted({cell for placement in under_supported_placements for cell in placement})
            )
            details["resolved_ship_placements"] = ()
            details["unresolved_ship_lengths"] = completed_lengths
            details["discarded_ship_cells"] = tuple(sorted(geometry_candidates))
            # Do not infer the missing hull cells, but keep the body pixels
            # that were independently stable in every frame.  The sidebar
            # completion and red marker prove this is a surfaced submarine;
            # these direct cells must be offered to the isolated blue
            # confirmation path instead of being reclassified as misses by
            # the generic pixel classifier below.
            supported_cells = {
                cell
                for placement in under_supported_placements
                for cell in placement
                if cell in stable_body_candidates
            }
            direct_body_hits = supported_cells - before_visible
            if not direct_body_hits:
                return None
            direct_perimeter: set[Cell] = set()
            for row, col in direct_body_hits:
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        neighbor = (row + row_offset, col + col_offset)
                        if (
                            neighbor not in direct_body_hits
                            and _inside_grid(neighbor, grid_size)
                        ):
                            direct_perimeter.add(neighbor)
            details["partial_completed_body_hits"] = tuple(
                sorted(direct_body_hits)
            )
            return _CompletedShipEvidence(
                new_hit_cells=frozenset(direct_body_hits),
                ship_cells=frozenset(direct_body_hits),
                perimeter_cells=frozenset(direct_perimeter),
            )

        # When more than one submarine completes in the same red-scout frame,
        # a global candidate set can legally contain a short run from one
        # hull and a spill/occlusion fragment from the other.  If any resolved
        # placement relies on cells that were not stable body evidence, its
        # length-to-hull association is ambiguous.  Do not promote that
        # assignment to authoritative completed cells; the caller will retain
        # the real red-hit evidence and let blue attacks confirm the cells.
        if len(completed_lengths) > 1:
            inferred_only_cells = set(resolution.cells) - stable_body_candidates
            if inferred_only_cells and details.get("completed_resolution_mode") != "red_anchor_length_binding":
                details["completed_ship_failure"] = (
                    "multi_completion_geometry_ambiguous"
                )
                details["ambiguous_completed_cells"] = tuple(
                    sorted(inferred_only_cells)
                )
                return None

        new_hit_cells = set(resolution.cells) - before_visible
        if not new_hit_cells:
            details["completed_ship_failure"] = "no_new_completed_ship_cells"
            return None

        perimeter_cells: set[Cell] = set()
        for row, col in resolution.cells:
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    neighbor = (row + row_offset, col + col_offset)
                    if (
                        neighbor not in resolution.cells
                        and _inside_grid(neighbor, grid_size)
                    ):
                        perimeter_cells.add(neighbor)

        details["completed_perimeter"] = tuple(sorted(perimeter_cells))
        details["completed_ship_failure"] = None

        return _CompletedShipEvidence(
            new_hit_cells=frozenset(new_hit_cells),
            ship_cells=resolution.cells,
            perimeter_cells=frozenset(perimeter_cells),
        )

    def _preflight(
        self,
        *,
        before_image: object,
        after_images: Sequence[np.ndarray],
        grid_size: int,
        click_points: Sequence[tuple[int, int]],
        center_cell: Cell | None,
        excluded_cells: Sequence[Cell] | set[Cell] | frozenset[Cell],
        learned_footprint: RedFootprint | None,
    ) -> tuple[
        tuple[np.ndarray, ...],
        dict[Cell, tuple[int, int]],
        frozenset[Cell],
        frozenset[Cell] | None,
    ] | None:
        if not _is_integer(grid_size) or grid_size <= 0:
            return None
        if center_cell is None or not _inside_grid(center_cell, grid_size):
            return None
        if not _valid_screenshot(before_image):
            return None

        try:
            frames = tuple(after_images)
            raw_points = tuple(click_points)
            raw_excluded = tuple(excluded_cells)
        except TypeError:
            return None
        if len(frames) < MINIMUM_FRAME_VOTES:
            return None
        if any(not _valid_screenshot(frame) for frame in frames):
            return None
        if len(raw_points) != grid_size * grid_size:
            return None

        normalized_points = tuple(_normalize_pair(point) for point in raw_points)
        if any(point is None for point in normalized_points):
            return None
        points_by_cell = {
            (index // grid_size, index % grid_size): point
            for index, point in enumerate(normalized_points)
            if point is not None
        }

        normalized_excluded = tuple(
            _normalize_pair(cell)
            for cell in raw_excluded
        )
        if any(cell is None for cell in normalized_excluded):
            return None
        excluded = frozenset(
            cell
            for cell in normalized_excluded
            if cell is not None
        )

        learned_offsets: frozenset[Cell] | None = None
        if learned_footprint is not None:
            if not isinstance(learned_footprint, RedFootprint):
                return None
            try:
                raw_offsets = tuple(learned_footprint.offsets)
            except TypeError:
                return None
            normalized_offsets = tuple(
                _normalize_pair(offset)
                for offset in raw_offsets
            )
            if any(offset is None for offset in normalized_offsets):
                return None
            learned_offsets = frozenset(
                offset
                for offset in normalized_offsets
                if offset is not None
            )

        return frames, points_by_cell, excluded, learned_offsets

    def _collect_evidence(
        self,
        *,
        before_image: np.ndarray,
        after_images: tuple[np.ndarray, ...],
        points_by_cell: Mapping[Cell, tuple[int, int]],
        candidates: set[Cell],
        minimum_change_threshold: float,
        mandatory_candidates: frozenset[Cell] | set[Cell] = frozenset(),
    ) -> tuple[dict[Cell, float], dict[Cell, tuple[str, ...]]] | None:
        if self._classifier is classify_diamond_hit:
            filtered = _prefilter_candidates_by_change_upper_bound(
                before_image=before_image,
                after_images=after_images,
                points_by_cell=points_by_cell,
                candidates=candidates,
                minimum_change_threshold=minimum_change_threshold,
            )
            if filtered is not None:
                candidates = filtered | mandatory_candidates

        median_change_by_cell: dict[Cell, float] = {}
        states_by_cell: dict[Cell, tuple[str, ...]] = {}
        for cell in sorted(candidates):
            point = points_by_cell[cell]
            changes: list[float] = []
            states: list[str] = []
            for after_image in after_images:
                try:
                    result = self._classifier(before_image, after_image, point)
                    changed_ratio = float(result.changed_ratio)
                    state = str(result.state).strip().lower()
                except Exception:
                    return None
                if not np.isfinite(changed_ratio) or not 0.0 <= changed_ratio <= 1.0:
                    return None
                changes.append(changed_ratio)
                states.append(state)
            median_change_by_cell[cell] = float(median(changes))
            states_by_cell[cell] = tuple(states)
        return median_change_by_cell, states_by_cell

    def _stable_visible_hit_cells(
        self,
        *,
        after_images: tuple[np.ndarray, ...],
        points_by_cell: Mapping[Cell, tuple[int, int]],
        candidates: set[Cell],
    ) -> set[Cell] | None:
        visible: set[Cell] = set()
        for cell in sorted(candidates):
            detector_votes = 0
            for after_image in after_images:
                try:
                    detector_votes += bool(
                        self._detect_hit(after_image, points_by_cell[cell])
                    )
                except Exception:
                    return None
            if detector_votes >= MINIMUM_FRAME_VOTES:
                visible.add(cell)
        return visible

    @staticmethod
    def _collapse_completed_submarine_hits(
        hit_cells: set[Cell],
        *,
        center_cell: Cell,
        confidence_by_cell: Mapping[Cell, float],
    ) -> set[Cell]:
        remaining = set(hit_cells)
        collapsed: set[Cell] = set()
        while remaining:
            first = min(remaining)
            remaining.remove(first)
            component = {first}
            pending = [first]
            while pending:
                row, col = pending.pop()
                for neighbor in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        pending.append(neighbor)
            if len(component) == 1:
                collapsed.update(component)
                continue
            if center_cell in component:
                collapsed.add(center_cell)
                continue
            collapsed.add(
                min(
                    component,
                    key=lambda cell: (
                        -float(confidence_by_cell.get(cell, 0.0)),
                        cell[0],
                        cell[1],
                    ),
                )
            )
        return collapsed

    def _before_visible_hit_cells(
        self,
        *,
        before_image: np.ndarray,
        points_by_cell: Mapping[Cell, tuple[int, int]],
        candidates: set[Cell],
    ) -> set[Cell] | None:
        visible: set[Cell] = set()

        # The dynamic diamond classifier is deliberately strict and can miss
        # a stationary gray wreck when the water animation changes its local
        # contrast.  If that same wreck becomes clearer in the result frames,
        # treating it as a new red-scout hit incorrectly attaches an old board
        # fact to the current six-cell footprint.  Reuse the static wreck
        # detector on the pre-click frame and union both sources.  Keep custom
        # test/integration detectors isolated from this production-only path.
        if (
            self._hit_detector is _default_hit_detector
            and self._marker_grid_size > 0
            and len(self._marker_points)
            == self._marker_grid_size * self._marker_grid_size
        ):
            try:
                visible.update(
                    detect_visible_wreck_cells(
                        before_image,
                        list(self._marker_points),
                        self._marker_grid_size,
                    )
                    & candidates
                )
            except Exception:
                return None
        for cell in sorted(candidates):
            if cell in visible:
                continue
            try:
                if self._detect_hit(before_image, points_by_cell[cell]):
                    visible.add(cell)
            except Exception:
                return None
        return visible

    def _classify_affected_cells(
        self,
        *,
        after_images: tuple[np.ndarray, ...],
        points_by_cell: Mapping[Cell, tuple[int, int]],
        affected: set[Cell],
        states_by_cell: Mapping[Cell, tuple[str, ...]],
    ) -> tuple[set[Cell], set[Cell], set[Cell]] | None:
        hit_cells: set[Cell] = set()
        miss_cells: set[Cell] = set()
        unknown_cells: set[Cell] = set()
        for cell in sorted(affected):
            point = points_by_cell[cell]
            detector_votes = 0
            for after_image in after_images:
                try:
                    detector_votes += bool(self._detect_hit(after_image, point))
                except Exception:
                    return None

            miss_votes = states_by_cell[cell].count("miss")
            if detector_votes >= MINIMUM_FRAME_VOTES:
                hit_cells.add(cell)
            elif detector_votes == 0 and miss_votes >= MINIMUM_FRAME_VOTES:
                miss_cells.add(cell)
            else:
                unknown_cells.add(cell)
        return hit_cells, miss_cells, unknown_cells

    @staticmethod
    def _invalid_result(
        center_cell: Cell,
        *,
        reason: str = "analysis_failed",
        diagnostics: Mapping[str, object] | None = None,
    ) -> RedScoutResult:
        return RedScoutResult(
            center_cell=center_cell,
            affected_cells=frozenset(),
            hit_cells=frozenset(),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=None,
            valid=False,
            confidence_by_cell=MappingProxyType({}),
            invalid_reason=reason,
            diagnostics=diagnostics,
        )


class RedScoutPlanner:
    def __init__(self, grid_size: int) -> None:
        if not _is_integer(grid_size) or grid_size <= 0:
            raise ValueError("grid_size must be a positive integer")
        self.grid_size = grid_size

    def choose_center(
        self,
        footprint: RedFootprint | None,
        covered_cells: Sequence[Cell] | set[Cell] | frozenset[Cell] = (),
        known_cells: Sequence[Cell] | set[Cell] | frozenset[Cell] = (),
        cell_scores: Mapping[Cell, float] | None = None,
        excluded_centers: Sequence[Cell] | set[Cell] | frozenset[Cell] = (),
    ) -> Cell | None:
        excluded = self._snapshot_cells(excluded_centers)
        covered = self._snapshot_cells(covered_cells)
        known = self._snapshot_cells(known_cells)
        if excluded is None or covered is None or known is None:
            return None
        blocked_centers = excluded | covered | known
        if footprint is None:
            return self._choose_untried_center(blocked_centers)
        if not isinstance(footprint, RedFootprint):
            return None

        offsets = self._snapshot_cells(footprint.offsets)
        if offsets is None or not offsets:
            return None

        if cell_scores is None:
            scores: Mapping[Cell, float] = {}
        else:
            try:
                scores = dict(cell_scores)
            except Exception:
                return None
        best_center: Cell | None = None
        best_score = float("-inf")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) in blocked_centers:
                    continue
                projected = {
                    (row + row_offset, col + col_offset)
                    for row_offset, col_offset in offsets
                    if _inside_grid(
                        (row + row_offset, col + col_offset),
                        self.grid_size,
                    )
                }
                if not projected:
                    continue

                new_unknown = projected - known - covered
                if not new_unknown:
                    continue
                clipped_offsets = len(offsets) - len(projected)
                overlap_cells = len(projected & covered)
                placement_score = sum(
                    self._cell_score(scores, cell)
                    for cell in sorted(projected)
                )
                score = (
                    len(new_unknown) * 100.0
                    + placement_score
                    - overlap_cells * 25.0
                    - clipped_offsets * 40.0
                )
                if score > best_score:
                    best_score = score
                    best_center = (row, col)
        return best_center

    def _choose_untried_center(self, excluded: frozenset[Cell]) -> Cell | None:
        candidates = [
            (row, col)
            for row in range(self.grid_size)
            for col in range(self.grid_size)
            if (row, col) not in excluded
        ]
        if not candidates:
            return None

        center = self.grid_size // 2
        preferred = (center, center)
        if not excluded and preferred in candidates:
            return preferred

        board_center = (self.grid_size - 1) / 2

        def spread_score(cell: Cell) -> tuple[float, float, int, int]:
            row, col = cell
            nearest_attempt = min(
                abs(row - old_row) + abs(col - old_col)
                for old_row, old_col in excluded
            )
            edge_spread = abs(row - board_center) + abs(col - board_center)
            return (nearest_attempt, edge_spread, -row, -col)

        return max(candidates, key=spread_score)

    @staticmethod
    def _snapshot_cells(cells: object) -> frozenset[Cell] | None:
        try:
            raw_cells = tuple(cells)  # type: ignore[arg-type]
        except TypeError:
            return None
        normalized = tuple(_normalize_pair(cell) for cell in raw_cells)
        if any(cell is None for cell in normalized):
            return None
        return frozenset(cell for cell in normalized if cell is not None)

    @staticmethod
    def _cell_score(cell_scores: Mapping[Cell, float], cell: Cell) -> float:
        try:
            score = float(cell_scores.get(cell, 0.0))
        except (TypeError, ValueError):
            return 0.0
        return score if np.isfinite(score) else 0.0


def _red_bomb_button_bounds(image: np.ndarray) -> tuple[int, int, int, int] | None:
    if not _valid_screenshot(image):
        return None

    image_height, image_width = image.shape[:2]
    reference_width, reference_height = RED_BOMB_BUTTON_REFERENCE_SIZE
    reference_x1, reference_y1, reference_x2, reference_y2 = (
        RED_BOMB_BUTTON_REFERENCE_BOUNDS
    )
    x1 = min(max(round(image_width * reference_x1 / reference_width), 0), image_width)
    y1 = min(max(round(image_height * reference_y1 / reference_height), 0), image_height)
    x2 = min(max(round(image_width * reference_x2 / reference_width), 0), image_width)
    y2 = min(max(round(image_height * reference_y2 / reference_height), 0), image_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def locate_red_bomb_button(image: np.ndarray) -> MatchResult | None:
    if not _valid_screenshot(image):
        return None

    bounds = _red_bomb_button_bounds(image)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    match = find_template_multi_scale(
        image[y1:y2, x1:x2],
        RED_BOMB_TEMPLATE,
        scales=RED_BOMB_TEMPLATE_SCALES,
        threshold=RED_BOMB_TEMPLATE_THRESHOLD,
        shape_weight=0.0,
    )
    try:
        score = float(match.score) if match is not None else None
    except (TypeError, ValueError):
        return None
    if score is None or not np.isfinite(score):
        return None

    return MatchResult(
        template_path=match.template_path,
        top_left=(x1, y1),
        bottom_right=(x2, y2),
        center=((x1 + x2) // 2, (y1 + y2) // 2),
        score=match.score,
    )


def _ammo_mask(image: np.ndarray, match: MatchResult) -> np.ndarray | None:
    if (
        not _valid_screenshot(image)
        or not isinstance(match, MatchResult)
    ):
        return None

    image_height, image_width = image.shape[:2]
    x1, y1 = match.top_left
    x2, y2 = match.bottom_right
    button_width = x2 - x1
    button_height = y2 - y1
    if button_width <= 0 or button_height <= 0:
        return None

    ammo_x1 = x1 + int(button_width * 0.78)
    ammo_y1 = y1 + int(button_height * 0.70)
    crop_x1 = min(max(ammo_x1, 0), image_width)
    crop_y1 = min(max(ammo_y1, 0), image_height)
    crop_x2 = min(max(x2, 0), image_width)
    crop_y2 = min(max(y2, 0), image_height)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.ndim == 2:
        hsv = cv2.cvtColor(crop, cv2.COLOR_GRAY2HSV)
    else:
        hsv = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)
    white = ((saturation <= 70) & (value >= 175)).astype(np.uint8)

    # The selected-state outline enters this corner crop from these two edges.
    edge_width = min(6, white.shape[0], white.shape[1])
    white[-edge_width:, :] = 0
    white[:, -edge_width:] = 0
    return cv2.resize(white, (24, 24), interpolation=cv2.INTER_NEAREST)


def build_ammo_fingerprint(
    frames: Sequence[np.ndarray],
    match: MatchResult,
) -> AmmoFingerprint | None:
    if not isinstance(frames, Sequence) or len(frames) < 3:
        return None

    masks = [_ammo_mask(frame, match) for frame in frames[:3]]
    if any(mask is None for mask in masks):
        return None

    consensus = sum(mask > 0 for mask in masks) >= 2
    foreground_pixels = int(np.count_nonzero(consensus))
    if foreground_pixels < 3:
        return None

    return AmmoFingerprint(
        shape=(int(consensus.shape[0]), int(consensus.shape[1])),
        packed_mask=np.packbits(consensus.reshape(-1)).tobytes(),
        foreground_pixels=foreground_pixels,
    )


def _unpack_fingerprint(fingerprint: AmmoFingerprint | None) -> np.ndarray | None:
    if not isinstance(fingerprint, AmmoFingerprint):
        return None
    shape = fingerprint.shape
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or not all(_is_integer(value) for value in shape)
        or any(value <= 0 for value in shape)
    ):
        return None
    height, width = (int(value) for value in shape)
    bit_count = height * width
    if not isinstance(fingerprint.packed_mask, (bytes, bytearray, memoryview)):
        return None
    if not _is_integer(fingerprint.foreground_pixels) or fingerprint.foreground_pixels <= 0:
        return None
    try:
        packed = np.frombuffer(fingerprint.packed_mask, dtype=np.uint8)
    except (BufferError, TypeError, ValueError):
        return None
    if packed.size != (bit_count + 7) // 8:
        return None
    try:
        unpacked = np.unpackbits(packed, count=bit_count).reshape((height, width)).astype(bool)
    except (ValueError, TypeError):
        return None
    if int(np.count_nonzero(unpacked)) != int(fingerprint.foreground_pixels):
        return None
    return unpacked


def ammo_fingerprint_matches(
    first: AmmoFingerprint | None,
    second: AmmoFingerprint | None,
    minimum_iou: float = 0.88,
) -> bool:
    if (
        not isinstance(first, AmmoFingerprint)
        or not isinstance(second, AmmoFingerprint)
        or first.shape != second.shape
    ):
        return False
    try:
        minimum_iou = float(minimum_iou)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(minimum_iou) or not 0.0 <= minimum_iou <= 1.0:
        return False

    first_mask = _unpack_fingerprint(first)
    second_mask = _unpack_fingerprint(second)
    if first_mask is None or second_mask is None:
        return False

    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = int(np.count_nonzero(first_mask | second_mask))
    return union > 0 and intersection / union >= minimum_iou


def red_bomb_selected(image: np.ndarray, match: MatchResult) -> bool:
    if (
        not _valid_screenshot(image)
        or not isinstance(match, MatchResult)
    ):
        return False

    image_height, image_width = image.shape[:2]
    x1 = min(max(match.top_left[0], 0), image_width)
    y1 = min(max(match.top_left[1], 0), image_height)
    x2 = min(max(match.bottom_right[0], 0), image_width)
    y2 = min(max(match.bottom_right[1], 0), image_height)
    if x2 <= x1 or y2 <= y1:
        return False

    crop = image[y1:y2, x1:x2]
    if crop.ndim == 2:
        hsv = cv2.cvtColor(crop, cv2.COLOR_GRAY2HSV)
    else:
        hsv = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)
    white = (saturation <= 45) & (value >= 205)

    edge_width = max(1, min(6, round(min(white.shape) * 0.04)))
    edge_ratios = (
        float(np.mean(white[:edge_width, :])),
        float(np.mean(white[-edge_width:, :])),
        float(np.mean(white[:, :edge_width])),
        float(np.mean(white[:, -edge_width:])),
    )
    return (
        min(edge_ratios) >= RED_BOMB_SELECTION_MIN_EDGE_RATIO
        and float(np.mean(edge_ratios)) >= RED_BOMB_SELECTION_MIN_AVERAGE_RATIO
    )
