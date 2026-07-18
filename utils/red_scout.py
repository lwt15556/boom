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

from config import RED_SCOUT_MAX_COUNT, TEMPLATE_DIR
from utils.diamond_hit import DiamondHitConfig, classify_diamond_hit, make_diamond_mask
from utils.image_match import MatchResult, find_template_multi_scale
from utils.sidebar_progress import (
    detect_sidebar_progress,
    newly_completed_lengths,
    resolve_completed_ship_cells,
)
from utils.wreck_detection import (
    COMPLETED_SHIP_BODY_MIN_SCORE,
    completed_ship_body_score,
    detect_completed_submarine_candidate_cells,
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


def _infer_completed_ship_endpoints(
    body_candidates: set[Cell],
    *,
    unresolved_lengths: Sequence[int],
    grid_size: int,
    after_images: Sequence[np.ndarray],
    points_by_cell: Mapping[Cell, tuple[int, int]],
) -> set[Cell]:
    inferred: set[Cell] = set()

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
        if length < 3 or length > grid_size:
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
                scores = [
                    float(completed_ship_body_score(image, point))
                    for image in after_images
                ]
                if scores:
                    scored.append((float(median(scores)), cell))
            if not scored:
                continue
            scored.sort(reverse=True)
            best_score, best_cell = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if (
                best_score >= COMPLETED_SHIP_BODY_MIN_SCORE
                and best_score - second_score >= COMPLETED_SHIP_ENDPOINT_MIN_MARGIN
            ):
                inferred.add(best_cell)
    return inferred


class ProbeMode(str, Enum):
    BLUE_ONLY = "blue_only"
    RED_SCOUT = "red_scout"


@dataclass(frozen=True)
class RedScoutSettings:
    mode: ProbeMode = ProbeMode.BLUE_ONLY
    count: int = 2


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

    raw_count = str(values.get("BBMA_RED_SCOUT_COUNT", "2")).strip()
    try:
        count = int(raw_count)
    except ValueError:
        count = 2
    if not 1 <= count <= RED_SCOUT_MAX_COUNT:
        count = 2
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


def _default_hit_detector(image: np.ndarray, point: tuple[int, int]) -> bool:
    # Match the visible wreck to the requested cell instead of searching the
    # whole crop for a template that can also occur in ordinary water tiles.
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

        frames, points_by_cell, excluded, learned_offsets = preflight
        diagnostics: dict[str, object] = {
            "stage": "candidate_detection",
            "center": result_center,
            "excluded_cells": tuple(sorted(excluded)),
            "learned_footprint": (
                tuple(sorted(learned_offsets))
                if learned_offsets is not None
                else ()
            ),
        }
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

        strong_result_misses = {
            cell
            for cell, changed_ratio in median_change_by_cell.items()
            if (
                cell not in raw_stable_result_hits
                and cell not in completed_visual_zone
                and changed_ratio >= RED_SCOUT_MISS_MIN_CHANGE
                and states_by_cell[cell].count("miss") >= RED_SCOUT_MISS_MIN_VOTES
            )
        }
        diagnostics["strong_misses"] = tuple(sorted(strong_result_misses))
        affected = stable_result_hits | strong_result_misses
        if before_visible:
            affected = affected - before_visible
        diagnostics["affected_before_limit"] = tuple(sorted(affected))
        if len(affected) > RED_SCOUT_RESULT_CELL_COUNT:
            if completed_ship is None:
                diagnostics["stage"] = "limit_strong_cells"
                return self._invalid_result(
                    result_center,
                    reason="too_many_strong_cells",
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
        if completed_ship is not None:
            authoritative_hits = set(completed_ship.new_hit_cells) & affected
            hit_cells.update(authoritative_hits)
            miss_cells.difference_update(authoritative_hits)
            unknown_cells.difference_update(authoritative_hits)

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
        except Exception:
            details["completed_body_detection_failed"] = True
        details["completed_body_candidates"] = tuple(
            sorted(stable_body_candidates)
        )
        details["completed_body_overrides"] = tuple(
            sorted(stable_body_candidates - eligible_cells)
        )

        resolution = resolve_completed_ship_cells(
            before_visible | raw_stable_result_hits | stable_body_candidates,
            completed_lengths,
            grid_size=grid_size,
            preferred_cells=raw_stable_result_hits - before_visible,
        )
        inferred_endpoints: set[Cell] = set()
        if resolution.unresolved_lengths:
            inferred_endpoints = _infer_completed_ship_endpoints(
                stable_body_candidates,
                unresolved_lengths=resolution.unresolved_lengths,
                grid_size=grid_size,
                after_images=after_images,
                points_by_cell=points_by_cell,
            )
            if inferred_endpoints:
                resolution = resolve_completed_ship_cells(
                    before_visible
                    | raw_stable_result_hits
                    | stable_body_candidates
                    | inferred_endpoints,
                    completed_lengths,
                    grid_size=grid_size,
                    preferred_cells=(
                        raw_stable_result_hits
                        - before_visible
                        | inferred_endpoints
                    ),
                )
        details["inferred_ship_endpoints"] = tuple(sorted(inferred_endpoints))
        details["resolved_ship_placements"] = resolution.placements
        details["unresolved_ship_lengths"] = resolution.unresolved_lengths
        details["discarded_ship_cells"] = tuple(sorted(resolution.discarded_cells))
        if resolution.unresolved_lengths:
            details["completed_ship_failure"] = "ship_geometry_unresolved"
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
                        self._hit_detector(after_image, points_by_cell[cell])
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
        for cell in sorted(candidates):
            try:
                if self._hit_detector(before_image, points_by_cell[cell]):
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
                    detector_votes += bool(self._hit_detector(after_image, point))
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
