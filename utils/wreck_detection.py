from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from config import TEMPLATE_DIR
from utils.diamond_hit import classify_diamond_hit
from utils.image_match import find_template_multi_scale
from utils.submarine_strategy import Cell

SUBMARINE_HIT_WRECK_TEMPLATE = TEMPLATE_DIR / "submarine_hit_wreck.png"
RED_HIT_MARKER_TEMPLATE = TEMPLATE_DIR / "red_hit_marker.png"
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
COMPLETED_SHIP_BODY_MIN_SCORE = 0.25
COMPLETED_SHIP_ANCHOR_MAX_CELL_DISTANCE = 2
COMPLETED_SHIP_MARKER_MAX_POINT_DISTANCE_FACTOR = 1.3


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

    hits: set[Cell] = set()
    for index, point in enumerate(click_points):
        if visible_wreck_static_detected(screenshot, point):
            hits.add((index // grid_size, index % grid_size))
    return hits


def detect_completed_submarine_candidate_cells(
    screenshot: np.ndarray,
    click_points: list[tuple[int, int]],
    grid_size: int,
) -> set[Cell]:
    """Return strict full-submarine candidates anchored by visible red ship markers."""
    if not isinstance(screenshot, np.ndarray) or screenshot.ndim != 3:
        return set()
    if grid_size <= 0 or len(click_points) != grid_size * grid_size:
        return set()

    normalized_points = [(int(x), int(y)) for x, y in click_points]
    anchors = _detect_completed_ship_anchor_cells(screenshot, normalized_points, grid_size)
    if not anchors:
        return set()

    candidates: set[Cell] = set(anchors)
    for index, point in enumerate(normalized_points):
        cell = (index // grid_size, index % grid_size)
        if not any(
            max(abs(cell[0] - anchor[0]), abs(cell[1] - anchor[1]))
            <= COMPLETED_SHIP_ANCHOR_MAX_CELL_DISTANCE
            for anchor in anchors
        ):
            continue
        if _completed_ship_body_score(screenshot, point) >= COMPLETED_SHIP_BODY_MIN_SCORE:
            candidates.add(cell)
    return candidates


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
    min_area = max(12, int(round(45 * scale)))
    max_area = max(180, int(round(260 * scale)))
    min_side = max(5, int(round(7 * min(width / 1280.0, height / 720.0))))
    max_side = max(18, int(round(28 * max(width / 1280.0, height / 720.0))))
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
            and component_width <= max_side
            and component_height <= max_side
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


def _completed_ship_body_score(image: np.ndarray, point: tuple[int, int]) -> float:
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

    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    _hue, saturation, value = cv2.split(hsv)
    white_ratio = float(np.count_nonzero((saturation < 65) & (value > 145))) / max(1, saturation.size)
    gray_ratio = float(
        np.count_nonzero((saturation < 80) & (value > 100) & (value < 210))
    ) / max(1, saturation.size)
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


def visible_wreck_static_detected(image: np.ndarray, point: tuple[int, int]) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    if red_hit_marker_visible(image, point):
        return True
    if wreck_template_visible(image, point):
        return True
    try:
        result = classify_diamond_hit(image, image, point)
    except Exception:
        return False
    return str(getattr(result, "state", "")).strip().lower() == "hit"
