from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720
SIDEBAR_X1 = 20
SIDEBAR_X2 = 145
FIRST_ROW_CENTER_Y = 275
ROW_STEP_Y = 35
ROW_HALF_HEIGHT = 13
MIN_ROW_COLOR_RATIO = 0.15
MIN_DOMINANCE_RATIO = 1.4
PARTIAL_WRECK_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
PARTIAL_WRECK_THRESHOLD = 0.62
PARTIAL_WRECK_CROP_HALF_WIDTH = 65
PARTIAL_WRECK_CROP_HALF_HEIGHT = 50
PARTIAL_WRECK_NMS_DISTANCE = 20


@dataclass(frozen=True)
class SidebarProgress:
    active_lengths: tuple[int, ...] = ()
    completed_lengths: tuple[int, ...] = ()
    unknown_lengths: tuple[int, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.unknown_lengths

    @property
    def completed_cells(self) -> int:
        return sum(self.completed_lengths)


@dataclass(frozen=True)
class CompletedShipResolution:
    cells: frozenset[tuple[int, int]]
    placements: tuple[tuple[tuple[int, int], ...], ...]
    unresolved_lengths: tuple[int, ...]
    discarded_cells: frozenset[tuple[int, int]]


def detect_sidebar_progress(
    image: np.ndarray,
    submarine_lengths: Sequence[int],
) -> SidebarProgress | None:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return None

    lengths = tuple(sorted((int(length) for length in submarine_lengths), reverse=True))
    if not lengths or any(length <= 0 for length in lengths):
        return None

    height, width = image.shape[:2]
    scale_x = width / REFERENCE_WIDTH
    scale_y = height / REFERENCE_HEIGHT
    x1 = max(0, int(round(SIDEBAR_X1 * scale_x)))
    x2 = min(width, int(round(SIDEBAR_X2 * scale_x)))
    half_height = max(2, int(round(ROW_HALF_HEIGHT * scale_y)))
    if x2 <= x1:
        return None

    active: list[int] = []
    completed: list[int] = []
    unknown: list[int] = []
    for row_index, length in enumerate(lengths):
        center_y = int(round((FIRST_ROW_CENTER_Y + row_index * ROW_STEP_Y) * scale_y))
        y1 = max(0, center_y - half_height)
        y2 = min(height, center_y + half_height + 1)
        if y2 <= y1:
            unknown.append(length)
            continue

        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        area = max(1, int(hsv.shape[0] * hsv.shape[1]))
        white_ratio = float(np.count_nonzero((saturation < 75) & (value > 170))) / area
        gray_ratio = float(
            np.count_nonzero(
                (hue >= 93)
                & (hue <= 106)
                & (saturation >= 65)
                & (saturation <= 150)
                & (value >= 82)
                & (value <= 140)
            )
        ) / area

        if white_ratio >= MIN_ROW_COLOR_RATIO and white_ratio >= gray_ratio * MIN_DOMINANCE_RATIO:
            active.append(length)
        elif gray_ratio >= MIN_ROW_COLOR_RATIO and gray_ratio >= white_ratio * MIN_DOMINANCE_RATIO:
            completed.append(length)
        else:
            unknown.append(length)

    return SidebarProgress(
        active_lengths=tuple(active),
        completed_lengths=tuple(completed),
        unknown_lengths=tuple(unknown),
    )


def newly_completed_lengths(
    before: SidebarProgress | None,
    after: SidebarProgress | None,
) -> tuple[int, ...]:
    if before is None or after is None or not before.valid or not after.valid:
        return ()

    newly_completed = Counter(after.completed_lengths) - Counter(before.completed_lengths)
    return tuple(sorted(newly_completed.elements(), reverse=True))


def merge_confirmed_hit_count(
    known_hit_count: int,
    progress: SidebarProgress | None,
) -> int:
    known_hit_count = max(0, int(known_hit_count))
    if progress is None or not progress.valid:
        return known_hit_count
    return max(known_hit_count, progress.completed_cells)


def detect_partial_wreck_cells(
    image: np.ndarray,
    click_points: Sequence[tuple[int, int]],
    *,
    grid_size: int,
    template_paths: Sequence[str | Path],
    threshold: float = PARTIAL_WRECK_THRESHOLD,
) -> set[tuple[int, int]] | None:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return None
    if grid_size <= 0 or not click_points:
        return None

    prepared_templates: list[np.ndarray] = []
    for template_path in template_paths:
        template = cv2.imread(str(template_path))
        if template is None:
            continue
        for scale in PARTIAL_WRECK_SCALES:
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=interpolation,
            )
            prepared_templates.append(resized)
    if not prepared_templates:
        return None

    height, width = image.shape[:2]
    candidates: list[tuple[float, tuple[int, int]]] = []
    for point_x, point_y in click_points:
        x1 = max(0, int(point_x) - PARTIAL_WRECK_CROP_HALF_WIDTH)
        y1 = max(0, int(point_y) - PARTIAL_WRECK_CROP_HALF_HEIGHT)
        x2 = min(width, int(point_x) + PARTIAL_WRECK_CROP_HALF_WIDTH + 1)
        y2 = min(height, int(point_y) + PARTIAL_WRECK_CROP_HALF_HEIGHT + 1)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        best_score = -1.0
        best_center: tuple[int, int] | None = None
        for template in prepared_templates:
            template_height, template_width = template.shape[:2]
            if template_height > crop.shape[0] or template_width > crop.shape[1]:
                continue
            match = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(match)
            if score <= best_score:
                continue
            best_score = float(score)
            best_center = (
                x1 + int(location[0]) + template_width // 2,
                y1 + int(location[1]) + template_height // 2,
            )

        if best_center is not None and best_score >= float(threshold):
            candidates.append((best_score, best_center))

    selected_centers: list[tuple[int, int]] = []
    min_distance_squared = PARTIAL_WRECK_NMS_DISTANCE ** 2
    for _score, center in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(
            (center[0] - selected[0]) ** 2 + (center[1] - selected[1]) ** 2
            < min_distance_squared
            for selected in selected_centers
        ):
            continue
        selected_centers.append(center)

    cells: set[tuple[int, int]] = set()
    for center in selected_centers:
        nearest_index = min(
            range(len(click_points)),
            key=lambda index: (
                (int(click_points[index][0]) - center[0]) ** 2
                + (int(click_points[index][1]) - center[1]) ** 2
            ),
        )
        cells.add((nearest_index // grid_size, nearest_index % grid_size))
    return cells


def calculate_visible_hit_count(
    progress: SidebarProgress | None,
    *,
    partial_wreck_count: int,
    fallback_hit_count: int = 0,
) -> int:
    if progress is None or not progress.valid:
        return max(0, int(fallback_hit_count))
    return progress.completed_cells + max(0, int(partial_wreck_count))


def resolve_completed_ship_cells(
    candidate_cells: set[tuple[int, int]],
    completed_lengths: Sequence[int],
    *,
    grid_size: int,
    preferred_cells: set[tuple[int, int]] | frozenset[tuple[int, int]] = frozenset(),
) -> CompletedShipResolution:
    """Keep only exact straight placements supported by completed-ship pixels."""
    candidates = {
        (int(row), int(col))
        for row, col in candidate_cells
        if 0 <= int(row) < grid_size and 0 <= int(col) < grid_size
    }
    lengths = tuple(sorted((int(length) for length in completed_lengths), reverse=True))
    preferred = {
        (int(row), int(col))
        for row, col in preferred_cells
        if 0 <= int(row) < grid_size and 0 <= int(col) < grid_size
    }

    def placement_extension_count(
        placement: tuple[tuple[int, int], ...],
    ) -> int:
        if len(placement) < 2:
            return 0
        first_row, first_col = placement[0]
        last_row, last_col = placement[-1]
        if first_row == last_row:
            neighbors = (
                (first_row, first_col - 1),
                (last_row, last_col + 1),
            )
        else:
            neighbors = (
                (first_row - 1, first_col),
                (last_row + 1, last_col),
            )
        return sum(neighbor in candidates for neighbor in neighbors)

    def supported_placements(length: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        if length <= 0 or length > grid_size:
            return ()

        exact: set[tuple[tuple[int, int], ...]] = set()
        for row in range(grid_size):
            for start_col in range(grid_size - length + 1):
                placement = tuple((row, col) for col in range(start_col, start_col + length))
                if set(placement).issubset(candidates):
                    exact.add(placement)
        for col in range(grid_size):
            for start_row in range(grid_size - length + 1):
                placement = tuple((row, col) for row in range(start_row, start_row + length))
                if set(placement).issubset(candidates):
                    exact.add(placement)
        return tuple(sorted(exact))

    placements_by_length = {
        length: supported_placements(length)
        for length in set(lengths)
    }

    def placement_fits(
        placement: tuple[tuple[int, int], ...],
        used: frozenset[tuple[int, int]],
    ) -> bool:
        return not any(
            max(abs(row - used_row), abs(col - used_col)) <= 1
            for row, col in placement
            for used_row, used_col in used
        )

    @lru_cache(maxsize=None)
    def solve(
        index: int,
        used: frozenset[tuple[int, int]],
    ) -> tuple[
        int,
        int,
        int,
        int,
        tuple[tuple[tuple[int, int], ...], ...],
        tuple[int, ...],
    ]:
        if index >= len(lengths):
            return 0, 0, 0, 0, (), ()

        length = lengths[index]
        skipped = solve(index + 1, used)
        best = (
            skipped[0],
            skipped[1],
            skipped[2],
            skipped[3],
            skipped[4],
            (length,) + skipped[5],
        )

        for placement in placements_by_length.get(length, ()):
            if not placement_fits(placement, used):
                continue
            next_used = used | frozenset(placement)
            remainder = solve(index + 1, next_used)
            candidate = (
                length + remainder[0],
                1 + remainder[1],
                len(set(placement) & preferred) + remainder[2],
                remainder[3] - placement_extension_count(placement),
                (placement,) + remainder[4],
                remainder[5],
            )
            candidate_score = candidate[:4]
            best_score = best[:4]
            if candidate_score > best_score or (
                candidate_score == best_score
                and (candidate[4], candidate[5]) < (best[4], best[5])
            ):
                best = candidate
        return best

    (
        _resolved_cells,
        _resolved_count,
        _preferred_count,
        _negative_extensions,
        placements,
        unresolved,
    ) = solve(
        0,
        frozenset(),
    )
    used = {
        cell
        for placement in placements
        for cell in placement
    }

    return CompletedShipResolution(
        cells=frozenset(used),
        placements=placements,
        unresolved_lengths=unresolved,
        discarded_cells=frozenset(candidates - used),
    )


def progressive_hit_count(
    *,
    initial_visual_hit_count: int,
    initial_strategy_hit_count: int,
    current_strategy_hit_count: int,
) -> int:
    new_strategy_hits = max(
        0,
        int(current_strategy_hit_count) - int(initial_strategy_hit_count),
    )
    return max(0, int(initial_visual_hit_count)) + new_strategy_hits


__all__ = [
    "CompletedShipResolution",
    "SidebarProgress",
    "calculate_visible_hit_count",
    "detect_partial_wreck_cells",
    "detect_sidebar_progress",
    "merge_confirmed_hit_count",
    "newly_completed_lengths",
    "progressive_hit_count",
    "resolve_completed_ship_cells",
]
