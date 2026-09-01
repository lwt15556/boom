from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from config import TEMPLATE_DIR
from utils.diamond_hit import DiamondHitConfig, classify_diamond_hit
from utils.image_match import find_template_multi_scale
from utils.submarine_strategy import Cell

SUBMARINE_HIT_WRECK_TEMPLATE = TEMPLATE_DIR / "submarine_hit_wreck.png"
RED_HIT_MARKER_TEMPLATE = TEMPLATE_DIR / "red_hit_marker.png"
RED_SUBMARINE_COMPONENT_TEMPLATES = tuple(
    TEMPLATE_DIR / f"red_submarine_component_{index}.png"
    for index in range(1, 6)
)
VISIBLE_WRECK_TEMPLATES = tuple(
    TEMPLATE_DIR / f"visible_wreck_{index}.png"
    for index in range(1, 4)
)
WRECK_TEMPLATE_THRESHOLD = 0.965
WRECK_TEMPLATE_SCALES = (0.75, 0.9, 1.0, 1.1, 1.25)
WRECK_TEMPLATE_MAX_CENTER_OFFSET = 14
WRECK_TEMPLATE_MASK_S_MAX = 100
WRECK_TEMPLATE_MASK_V_MIN = 150
WRECK_TEMPLATE_MASK_ELLIPSE_SCALE = 0.55
RED_HIT_MARKER_TEMPLATE_THRESHOLD = 0.82
RED_HIT_MARKER_TEMPLATE_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
RED_HIT_MARKER_MAX_CENTER_OFFSET = 18
RED_HIT_MARKER_MIN_AREA = 80
RED_HIT_MARKER_MIN_WIDTH = 12
RED_HIT_MARKER_MIN_HEIGHT = 8
# The flag can sit near a hull edge rather than at the calibrated diamond
# center.  Perspective screenshots can place it roughly 50px from the
# nearest calibrated point, so use a radius that covers that offset while
# remaining local to the same tile neighbourhood.
RED_SUBMARINE_MARKER_MAX_POINT_DISTANCE = 56
COMPLETED_SHIP_BODY_MIN_SCORE = 0.25
# Diagonal cells next to a red marker are commonly water or a neighbouring
# wreck.  Only admit such a cell when its hull evidence is substantially
# stronger than the normal body threshold.
COMPLETED_SHIP_DIAGONAL_BODY_MIN_SCORE = 0.48
# A red component can sit on an endpoint while the hull extends up to two
# tiles away along its orientation.
COMPLETED_SHIP_ANCHOR_MAX_CELL_DISTANCE = 2
COMPLETED_SHIP_MARKER_MAX_POINT_DISTANCE_FACTOR = 1.3
# The flag's anti-aliased red component grows between game frames.  Keep the
# component-size guard, but allow the ~300-400px variants seen at 1280x720.
COMPLETED_SHIP_MARKER_MAX_AREA = 420
# A template must belong to one diamond rather than merely touching its
# centre.  Sampling the footprint keeps detections on cell borders from being
# duplicated into neighbouring cells.
WRECK_TEMPLATE_MIN_CELL_COVERAGE = 0.70
# The level title and countdown are drawn over the upper isometric diamonds.
# Their fixed screen-space footprint contains the same bright gray/white and
# occasional red pixels used by the wreck detectors.  Keep the footprint
# normalized so it follows both 1280x720 captures and resized diagnostics.
ACTIVITY_TITLE_OVERLAY_LEFT = 0.40
ACTIVITY_TITLE_OVERLAY_RIGHT = 0.60
ACTIVITY_TITLE_OVERLAY_BOTTOM = 0.13


def _inside_activity_title_overlay(
    image: np.ndarray,
    point: tuple[int, int],
) -> bool:
    height, width = image.shape[:2]
    x, y = point
    return (
        int(round(width * ACTIVITY_TITLE_OVERLAY_LEFT)) <= int(x)
        <= int(round(width * ACTIVITY_TITLE_OVERLAY_RIGHT))
        and 0 <= int(y) <= int(round(height * ACTIVITY_TITLE_OVERLAY_BOTTOM))
    )


def _exclude_activity_title_overlay(
    mask: np.ndarray,
    *,
    image_shape: tuple[int, ...],
    roi_left: int,
    roi_top: int,
) -> None:
    """Remove title/countdown pixels from a cell-local evidence mask."""
    image_height, image_width = image_shape[:2]
    overlay_left = int(round(image_width * ACTIVITY_TITLE_OVERLAY_LEFT))
    overlay_right = int(round(image_width * ACTIVITY_TITLE_OVERLAY_RIGHT))
    overlay_bottom = int(round(image_height * ACTIVITY_TITLE_OVERLAY_BOTTOM))
    local_left = max(0, overlay_left - int(roi_left))
    local_right = min(mask.shape[1], overlay_right - int(roi_left) + 1)
    local_bottom = min(mask.shape[0], overlay_bottom - int(roi_top) + 1)
    if local_left < local_right and local_bottom > 0:
        mask[:local_bottom, local_left:local_right] = 0


@lru_cache(maxsize=1)
def _red_submarine_component_bounds() -> tuple[int, int, int]:
    """Read the supplied component templates as red-shape calibration."""
    areas: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    for path in RED_SUBMARINE_COMPONENT_TEMPLATES:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_mask = (
            ((hue <= 12) | (hue >= 168))
            & (saturation >= 90)
            & (value >= 90)
        ).astype(np.uint8) * 255
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            red_mask,
            connectivity=8,
        )
        for label_index in range(1, count):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            if area < 20 or width < 4 or height < 4:
                continue
            areas.append(area)
            widths.append(width)
            heights.append(height)
    if not areas:
        return COMPLETED_SHIP_MARKER_MAX_AREA, 28, 28
    return max(areas), max(widths), max(heights)


@lru_cache(maxsize=1)
def _load_masked_wreck_templates() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for path in (SUBMARINE_HIT_WRECK_TEMPLATE, *VISIBLE_WRECK_TEMPLATES):
        template = cv2.imread(str(path))
        if template is None:
            continue

        height, width = template.shape[:2]
        hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        _hue, saturation, value = cv2.split(hsv)
        yy, xx = np.ogrid[:height, :width]
        ellipse = (
            (
                (xx - (width - 1) / 2.0)
                / (width * 0.5 * WRECK_TEMPLATE_MASK_ELLIPSE_SCALE)
            ) ** 2
            + (
                (yy - (height - 1) / 2.0)
                / (height * 0.5 * WRECK_TEMPLATE_MASK_ELLIPSE_SCALE)
            ) ** 2
            <= 1.0
        )
        mask = (
            (saturation <= WRECK_TEMPLATE_MASK_S_MAX)
            & (value >= WRECK_TEMPLATE_MASK_V_MIN)
            & ellipse
        ).astype(np.uint8) * 255
        if np.count_nonzero(mask) >= 30:
            prepared.append((template, mask))
    return tuple(prepared)


def wreck_template_visible(
    image: np.ndarray,
    point: tuple[int, int],
    threshold: float = WRECK_TEMPLATE_THRESHOLD,
    *,
    cell_polygon: np.ndarray | None = None,
) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    templates = _load_masked_wreck_templates()
    if not templates:
        return False

    x, y = point
    image_height, image_width = image.shape[:2]
    for template, mask in templates:
        template_height, template_width = template.shape[:2]
        x1 = max(0, int(x) - template_width // 2 - WRECK_TEMPLATE_MAX_CENTER_OFFSET)
        y1 = max(0, int(y) - template_height // 2 - WRECK_TEMPLATE_MAX_CENTER_OFFSET)
        x2 = min(
            image_width,
            int(x) + template_width // 2 + WRECK_TEMPLATE_MAX_CENTER_OFFSET + 1,
        )
        y2 = min(
            image_height,
            int(y) + template_height // 2 + WRECK_TEMPLATE_MAX_CENTER_OFFSET + 1,
        )
        crop = image[y1:y2, x1:x2]
        if crop.shape[0] < template_height or crop.shape[1] < template_width:
            continue

        try:
            scores = cv2.matchTemplate(
                crop,
                template,
                cv2.TM_SQDIFF_NORMED,
                mask=mask,
            )
        except cv2.error:
            continue

        score_rows, score_cols = scores.shape
        candidate_x = x1 + np.arange(score_cols) + template_width // 2
        candidate_y = y1 + np.arange(score_rows) + template_height // 2
        valid_centers = (
            np.abs(candidate_y[:, None] - int(y))
            <= WRECK_TEMPLATE_MAX_CENTER_OFFSET
        ) & (
            np.abs(candidate_x[None, :] - int(x))
            <= WRECK_TEMPLATE_MAX_CENTER_OFFSET
        )
        valid_scores = scores[valid_centers & np.isfinite(scores)]
        if valid_scores.size and 1.0 - float(np.min(valid_scores)) >= threshold:
            # Reject a match whose template footprint crosses into a
            # neighbouring diamond.  The center-only check is insufficient for
            # large wreck templates near a cell edge.
            if cell_polygon is None:
                return True
            min_score_location = np.unravel_index(
                int(np.nanargmin(np.where(valid_centers, scores, np.nan))),
                scores.shape,
            )
            match_x = x1 + int(min_score_location[1]) + template_width // 2
            match_y = y1 + int(min_score_location[0]) + template_height // 2
            polygon = np.asarray(cell_polygon, dtype=np.float32)
            if cv2.pointPolygonTest(polygon, (float(match_x), float(match_y)), False) < 0:
                continue
            # Check the complete template footprint.  A small regular sample
            # is sufficient here and avoids allocating a full-frame mask for
            # every template/scale candidate.
            half_w = max(1.0, template_width * 0.5)
            half_h = max(1.0, template_height * 0.5)
            inside = 0
            total = 0
            for fraction_y in np.linspace(-1.0, 1.0, 7):
                for fraction_x in np.linspace(-1.0, 1.0, 7):
                    sample = (
                        float(match_x) + half_w * float(fraction_x),
                        float(match_y) + half_h * float(fraction_y),
                    )
                    total += 1
                    inside += int(cv2.pointPolygonTest(polygon, sample, False) >= 0)
            if total and inside / total >= WRECK_TEMPLATE_MIN_CELL_COVERAGE:
                return True
    return False


def red_hit_marker_visible(image: np.ndarray, point: tuple[int, int]) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    if not _red_hit_marker_color_visible(image, point):
        return False
    if RED_HIT_MARKER_TEMPLATE.exists():
        return red_hit_marker_template_visible(image, point)
    return True


def red_submarine_marker_visible(
    image: np.ndarray,
    point: tuple[int, int],
) -> bool:
    """Return whether a red submarine decoration is attached to this cell.

    Surfaced submarines carry a small red component above or beside the hull.
    It is a persistent visual decoration, not a shot result.  The component
    can be too far from the calibrated cell center for the legacy template
    helper, so use the nearest red connected component instead.  This helper
    is deliberately independent from ``red_hit_marker_visible``: the latter
    remains an exact hit-marker API for callers that need that distinction,
    while this function is used as a negative guard for static hit detection.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False
    try:
        x, y = (int(point[0]), int(point[1]))
    except (TypeError, ValueError, IndexError):
        return False

    height, width = image.shape[:2]
    crop_half_width = 70
    crop_half_height = 65
    x1 = max(0, x - crop_half_width)
    y1 = max(0, y - crop_half_height)
    x2 = min(width, x + crop_half_width + 1)
    y2 = min(height, y + crop_half_height + 1)
    if x2 <= x1 or y2 <= y1:
        return False

    try:
        hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        red_mask = (
            ((hue <= 12) | (hue >= 168))
            & (saturation >= 90)
            & (value >= 90)
        ).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            red_mask,
            connectivity=8,
        )
    except (cv2.error, TypeError, ValueError):
        return False

    local_x = float(x - x1)
    local_y = float(y - y1)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        component_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if not (
            area >= RED_HIT_MARKER_MIN_AREA
            and component_width >= RED_HIT_MARKER_MIN_WIDTH
            and component_height >= RED_HIT_MARKER_MIN_HEIGHT
        ):
            continue
        center_x, center_y = centroids[label_index]
        distance = float(
            np.hypot(center_x - local_x, center_y - local_y)
        )
        if distance <= RED_SUBMARINE_MARKER_MAX_POINT_DISTANCE:
            return True
    return False


def _red_hit_marker_color_visible(
    image: np.ndarray,
    point: tuple[int, int],
) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    x, y = point
    h, w = image.shape[:2]
    crop_w = 150
    crop_h = 120
    x1 = max(0, int(x - crop_w // 2))
    y1 = max(0, int(y - crop_h // 2))
    x2 = min(w, int(x + crop_w // 2))
    y2 = min(h, int(y + crop_h // 2))
    if x2 <= x1 or y2 <= y1:
        return False

    crop = image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    local_center = (int(x - x1), int(y - y1))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_mask = (
        ((hue <= 12) | (hue >= 168))
        & (saturation >= 90)
        & (value >= 90)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    for index in range(1, num_labels):
        area = int(stats[index, cv2.CC_STAT_AREA])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        offset_x = float(centroids[index][0] - local_center[0])
        offset_y = float(centroids[index][1] - local_center[1])
        if (
            area >= RED_HIT_MARKER_MIN_AREA
            and width >= RED_HIT_MARKER_MIN_WIDTH
            and height >= RED_HIT_MARKER_MIN_HEIGHT
            and abs(offset_x) <= 32
            and -35 <= offset_y <= -4
        ):
            return True
    return False


def red_hit_marker_template_visible(
    image: np.ndarray,
    point: tuple[int, int],
    threshold: float = RED_HIT_MARKER_TEMPLATE_THRESHOLD,
) -> bool:
    if not RED_HIT_MARKER_TEMPLATE.exists():
        return False
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    x, y = point
    h, w = image.shape[:2]
    crop_w = 160
    crop_h = 130
    x1 = max(0, int(x - crop_w // 2))
    y1 = max(0, int(y - crop_h // 2))
    x2 = min(w, int(x + crop_w // 2))
    y2 = min(h, int(y + crop_h // 2))
    if x2 <= x1 or y2 <= y1:
        return False

    crop = image[y1:y2, x1:x2]
    match = find_template_multi_scale(
        crop,
        RED_HIT_MARKER_TEMPLATE,
        scales=RED_HIT_MARKER_TEMPLATE_SCALES,
        threshold=threshold,
    )
    if match is None:
        return False

    match_center = (x1 + match.center[0], y1 + match.center[1])
    return (
        abs(match_center[0] - x) <= RED_HIT_MARKER_MAX_CENTER_OFFSET
        and abs(match_center[1] - y) <= RED_HIT_MARKER_MAX_CENTER_OFFSET
    )


def detect_visible_wreck_cells(
    screenshot: np.ndarray,
    click_points: list[tuple[int, int]],
    grid_size: int,
) -> set[Cell]:
    if not isinstance(screenshot, np.ndarray) or screenshot.ndim != 3:
        return set()
    if grid_size <= 0 or len(click_points) != grid_size * grid_size:
        return set()

    # Assign each red submarine component to its single nearest grid cell
    # before per-cell wreck detection.  The red-component guard uses a wider
    # local radius for perspective offsets; applying it independently to every
    # cell would suppress real wrecks adjacent to a surfaced submarine.
    red_marker_cells = _detect_completed_ship_anchor_cells(
        screenshot,
        [(int(x), int(y)) for x, y in click_points],
        grid_size,
    )
    def cell_polygon(index: int) -> np.ndarray:
        row, col = divmod(index, grid_size)
        p = np.asarray(click_points[index], dtype=np.float32)
        if col + 1 < grid_size:
            right = np.asarray(click_points[index + 1], dtype=np.float32) - p
        else:
            right = p - np.asarray(click_points[index - 1], dtype=np.float32)
        if row + 1 < grid_size:
            down = np.asarray(click_points[index + grid_size], dtype=np.float32) - p
        else:
            down = p - np.asarray(click_points[index - grid_size], dtype=np.float32)
        return np.asarray(
            [
                p - right * 0.5 - down * 0.5,
                p + right * 0.5 - down * 0.5,
                p + right * 0.5 + down * 0.5,
                p - right * 0.5 + down * 0.5,
            ],
            dtype=np.float32,
        )

    hits: set[Cell] = set()
    for index, point in enumerate(click_points):
        cell = (index // grid_size, index % grid_size)
        if cell in red_marker_cells:
            continue
        if visible_wreck_static_detected(
            screenshot,
            point,
            ignore_submarine_marker=True,
            cell_polygon=cell_polygon(index),
        ):
            hits.add(cell)
    return hits


def detect_red_submarine_marker_cells(
    screenshot: np.ndarray,
    click_points: list[tuple[int, int]],
    grid_size: int,
) -> set[Cell]:
    """Assign visible red submarine components to their unique grid cells."""
    if not isinstance(screenshot, np.ndarray) or screenshot.ndim != 3:
        return set()
    if grid_size <= 0 or len(click_points) != grid_size * grid_size:
        return set()
    return _detect_completed_ship_anchor_cells(
        screenshot,
        [(int(x), int(y)) for x, y in click_points],
        grid_size,
    )


def detect_completed_submarine_candidate_cells(
    screenshot: np.ndarray,
    click_points: list[tuple[int, int]],
    grid_size: int,
) -> set[Cell]:
    """Return hull-cell candidates for submarines marked complete in red."""
    if not isinstance(screenshot, np.ndarray) or screenshot.ndim != 3:
        return set()
    if grid_size <= 0 or len(click_points) != grid_size * grid_size:
        return set()

    normalized_points = [(int(x), int(y)) for x, y in click_points]
    anchors = _detect_completed_ship_anchor_cells(screenshot, normalized_points, grid_size)
    if not anchors:
        return set()

    # A red component is definitive completion evidence for the nearby
    # submarine, but it is not a hit coordinate.  Its nearest grid point can
    # be an empty projected cell (for example `(0, 6)` in the level-8 view),
    # so never admit the anchor by itself.  Only gray/white hull evidence may
    # contribute coordinates to the completed-ship geometry.
    candidates: set[Cell] = set()
    body_scores: dict[Cell, float] = {}
    for index, point in enumerate(normalized_points):
        cell = (index // grid_size, index % grid_size)
        body_score = completed_ship_body_score(
            screenshot,
            point,
            cell_polygon=grid_cell_polygon(normalized_points, index, grid_size),
        )

        def _near_anchor(anchor: Cell) -> bool:
            row_delta = abs(cell[0] - anchor[0])
            col_delta = abs(cell[1] - anchor[1])
            # Keep the endpoint allowance along a ship axis.  A diagonal cell
            # is accepted only with much stronger body evidence, preventing a
            # nearby wreck or bright water tile from becoming ship geometry.
            return (
                (row_delta == 0 and col_delta <= COMPLETED_SHIP_ANCHOR_MAX_CELL_DISTANCE)
                or (
                    col_delta == 0
                    and row_delta <= COMPLETED_SHIP_ANCHOR_MAX_CELL_DISTANCE
                )
                or (
                    row_delta == 1
                    and col_delta == 1
                    and body_score >= COMPLETED_SHIP_DIAGONAL_BODY_MIN_SCORE
                )
            )

        if not any(_near_anchor(anchor) for anchor in anchors):
            continue
        if body_score >= COMPLETED_SHIP_BODY_MIN_SCORE:
            candidates.add(cell)
            body_scores[cell] = float(body_score)

    if not candidates:
        return set()

    # The red component often bleeds into an adjacent diamond.  Keeping every
    # nearby gray patch lets the later fleet resolver choose a wrong
    # horizontal/vertical line (for example, the real vertical ship
    # ``(4,4),(5,4),(6,4)`` can be joined with the lower-score red-marker
    # spill at ``(5,2),(5,3)``).  For each marker, retain the strongest
    # contiguous straight run supported by the body scores.  The marker is
    # allowed to sit beside the hull, so the run only needs to pass within one
    # cell of the marker rather than include the marker cell itself.
    selected: set[Cell] = set()
    for anchor in anchors:
        best_key: tuple[float, float, int, int, tuple[Cell, ...]] | None = None
        for orientation in ("H", "V"):
            for length in range(2, grid_size + 1):
                if orientation == "H":
                    for row in range(grid_size):
                        for start_col in range(grid_size - length + 1):
                            placement = tuple(
                                (row, start_col + offset)
                                for offset in range(length)
                            )
                            placement_set = set(placement)
                            if not placement_set <= candidates:
                                continue
                            if not any(
                                max(abs(cell[0] - anchor[0]), abs(cell[1] - anchor[1])) <= 1
                                for cell in placement
                            ):
                                continue
                            scores = [body_scores[cell] for cell in placement]
                            mean_score = sum(scores) / float(length)
                            # Prefer a longer run only when its average body
                            # evidence is effectively tied.  This avoids a
                            # short high-score fragment stealing a complete
                            # longer hull while still rejecting weak spillover.
                            key = (
                                mean_score + 0.03 * length,
                                mean_score,
                                length,
                                sum(score >= COMPLETED_SHIP_BODY_MIN_SCORE for score in scores),
                                tuple(sorted(placement)),
                            )
                            if best_key is None or key > best_key:
                                best_key = key
                else:
                    for col in range(grid_size):
                        for start_row in range(grid_size - length + 1):
                            placement = tuple(
                                (start_row + offset, col)
                                for offset in range(length)
                            )
                            placement_set = set(placement)
                            if not placement_set <= candidates:
                                continue
                            if not any(
                                max(abs(cell[0] - anchor[0]), abs(cell[1] - anchor[1])) <= 1
                                for cell in placement
                            ):
                                continue
                            scores = [body_scores[cell] for cell in placement]
                            mean_score = sum(scores) / float(length)
                            key = (
                                mean_score + 0.03 * length,
                                mean_score,
                                length,
                                sum(score >= COMPLETED_SHIP_BODY_MIN_SCORE for score in scores),
                                tuple(sorted(placement)),
                            )
                            if best_key is None or key > best_key:
                                best_key = key
        if best_key is not None:
            selected.update(best_key[4])

    # Keep the conservative raw set when no complete straight run can be
    # formed.  The caller may still use it for fail-closed diagnostics, but a
    # red marker alone will never manufacture a completed ship.
    return selected or candidates


def _detect_completed_ship_anchor_cells(
    image: np.ndarray,
    click_points: list[tuple[int, int]],
    grid_size: int,
) -> set[Cell]:
    height, width = image.shape[:2]
    xs = [point[0] for point in click_points]
    ys = [point[1] for point in click_points]
    step = _estimate_grid_step(click_points, grid_size)
    margin = max(20, int(round(step * 2.2)))
    x1 = max(0, min(xs) - margin)
    y1 = max(0, min(ys) - margin)
    x2 = min(width, max(xs) + margin)
    y2 = min(height, max(ys) + margin)
    if x2 <= x1 or y2 <= y1:
        return set()

    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_mask = (
        ((hue <= 12) | (hue >= 168))
        & (saturation >= 90)
        & (value >= 90)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    scale = max(0.2, (width / 1280.0) * (height / 720.0))
    linear_scale = max(width / 1280.0, height / 720.0)
    template_max_area, template_max_width, template_max_height = (
        _red_submarine_component_bounds()
    )
    min_area = max(12, int(round(45 * scale)))
    max_area = max(
        180,
        int(round(COMPLETED_SHIP_MARKER_MAX_AREA * scale)),
        int(round(template_max_area * 1.25 * scale)),
    )
    min_side = max(5, int(round(7 * min(width / 1280.0, height / 720.0))))
    max_component_width = max(
        18,
        int(round(max(28, template_max_width) * 1.20 * linear_scale)),
    )
    max_component_height = max(
        18,
        int(round(max(28, template_max_height) * 1.20 * linear_scale)),
    )
    max_point_distance_sq = (step * COMPLETED_SHIP_MARKER_MAX_POINT_DISTANCE_FACTOR) ** 2

    anchors: set[Cell] = set()
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        component_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
        if not (
            min_area <= area <= max_area
            and component_width >= min_side
            and component_height >= min_side
            and component_width <= max_component_width
            and component_height <= max_component_height
        ):
            continue

        center_x = float(x1 + centroids[label_index][0])
        center_y = float(y1 + centroids[label_index][1])
        nearest_index = min(
            range(len(click_points)),
            key=lambda index: (
                (float(click_points[index][0]) - center_x) ** 2
                + (float(click_points[index][1]) - center_y) ** 2
            ),
        )
        nearest_point = click_points[nearest_index]
        distance_sq = (
            (float(nearest_point[0]) - center_x) ** 2
            + (float(nearest_point[1]) - center_y) ** 2
        )
        if distance_sq <= max_point_distance_sq:
            anchors.add((nearest_index // grid_size, nearest_index % grid_size))

    return anchors


def grid_cell_polygon(
    click_points: list[tuple[int, int]],
    index: int,
    grid_size: int,
) -> np.ndarray | None:
    """Return the calibrated diamond polygon for one cell."""
    if grid_size <= 0 or index < 0 or index >= len(click_points):
        return None
    row, col = divmod(index, grid_size)
    p = np.asarray(click_points[index], dtype=np.float32)
    if col + 1 < grid_size:
        right = np.asarray(click_points[index + 1], dtype=np.float32) - p
    elif col > 0:
        right = p - np.asarray(click_points[index - 1], dtype=np.float32)
    else:
        return None
    if row + 1 < grid_size:
        down = np.asarray(click_points[index + grid_size], dtype=np.float32) - p
    elif row > 0:
        down = p - np.asarray(click_points[index - grid_size], dtype=np.float32)
    else:
        return None
    return np.asarray(
        [
            p - right * 0.5 - down * 0.5,
            p + right * 0.5 - down * 0.5,
            p + right * 0.5 + down * 0.5,
            p - right * 0.5 + down * 0.5,
        ],
        dtype=np.float32,
    )


def completed_ship_body_score(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    cell_polygon: np.ndarray | None = None,
) -> float:
    x, y = point
    height, width = image.shape[:2]
    step = max(20, int(round(min(width / 1280.0, height / 720.0) * 48)))
    half_width = max(22, int(round(step * 0.95)))
    half_height = max(16, int(round(step * 0.70)))
    x1 = max(0, int(x) - half_width)
    y1 = max(0, int(y) - half_height)
    x2 = min(width, int(x) + half_width + 1)
    y2 = min(height, int(y) + half_height + 1)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = image[y1:y2, x1:x2]
    if cell_polygon is not None:
        polygon = np.asarray(cell_polygon, dtype=np.float32).copy()
        polygon[:, 0] -= x1
        polygon[:, 1] -= y1
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        # Avoid counting the grid border and neighboring diamonds.  Erode the
        # polygon slightly because anti-aliased edges belong to both cells.
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        if not np.any(mask):
            return 0.0
    else:
        mask = np.full(roi.shape[:2], 255, dtype=np.uint8)

    # A top-row submarine can genuinely sit beneath the title.  Do not reject
    # the whole cell: discard only the UI-covered pixels and score the visible
    # remainder of its diamond.  Sidebar completion plus the red component
    # still provides the independent evidence needed to accept the geometry.
    _exclude_activity_title_overlay(
        mask,
        image_shape=image.shape,
        roi_left=x1,
        roi_top=y1,
    )
    if not np.any(mask):
        return 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)
    denominator = max(1, int(np.count_nonzero(mask)))
    white = (saturation < 65) & (value > 145)
    gray = (saturation < 80) & (value > 100) & (value < 210)
    white &= mask > 0
    gray &= mask > 0
    white_ratio = float(np.count_nonzero(white)) / denominator
    gray_ratio = float(
        np.count_nonzero(gray)
    ) / denominator
    return max(white_ratio, gray_ratio)


def _estimate_grid_step(click_points: list[tuple[int, int]], grid_size: int) -> float:
    distances: list[float] = []
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            x, y = click_points[index]
            if col + 1 < grid_size:
                nx, ny = click_points[index + 1]
                distances.append(float(np.hypot(nx - x, ny - y)))
            if row + 1 < grid_size:
                nx, ny = click_points[index + grid_size]
                distances.append(float(np.hypot(nx - x, ny - y)))
    if not distances:
        return 40.0
    return max(12.0, float(np.median(distances)))


def visible_wreck_static_detected(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    ignore_submarine_marker: bool = False,
    cell_polygon: np.ndarray | None = None,
) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    # A single static frame cannot distinguish the title/countdown strokes
    # from a wreck in the diamonds directly underneath.  Leave these cells
    # unknown so a real blue probe or completed-submarine geometry can decide
    # them; this is safer than silently committing a false existing hit.
    if _inside_activity_title_overlay(image, point):
        return False

    # The red object attached to a submarine is a visual decoration, not a
    # result marker.  Reject it unconditionally before template/classifier
    # checks so a red component can never be promoted to a static hit.
    if red_hit_marker_visible(image, point):
        return False

    # A red effect on a surfaced submarine is a persistent decoration, not
    # evidence that this cell was already confirmed.  Check this before the
    # gray wreck template because the hull itself can still match that mask.
    if not ignore_submarine_marker and red_submarine_marker_visible(image, point):
        return False
    if wreck_template_visible(image, point, cell_polygon=cell_polygon):
        return True
    try:
        # Static review must stay tied to the requested cell.  The default
        # 14px refinement can jump across a tile edge and classify a nearby
        # submarine hull as this water cell (notably level-8 `(8,7)`).
        result = classify_diamond_hit(
            image,
            image,
            point,
            config=DiamondHitConfig(search_radius=2),
        )
    except Exception:
        return False
    if str(getattr(result, "state", "")).strip().lower() != "hit":
        return False
    if cell_polygon is None:
        return True
    refined = getattr(result, "refined_center", point)
    return cv2.pointPolygonTest(
        np.asarray(cell_polygon, dtype=np.float32),
        (float(refined[0]), float(refined[1])),
        False,
    ) >= 0
