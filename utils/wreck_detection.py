from __future__ import annotations

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
WRECK_TEMPLATE_THRESHOLD = 0.70
WRECK_TEMPLATE_SCALES = (0.75, 0.9, 1.0, 1.1, 1.25)
RED_HIT_MARKER_TEMPLATE_THRESHOLD = 0.82
RED_HIT_MARKER_TEMPLATE_SCALES = (0.8, 0.9, 1.0, 1.1, 1.2)
RED_HIT_MARKER_MAX_CENTER_OFFSET = 18
RED_HIT_MARKER_MIN_AREA = 80
RED_HIT_MARKER_MIN_WIDTH = 12
RED_HIT_MARKER_MIN_HEIGHT = 8


def wreck_template_visible(
    image: np.ndarray,
    point: tuple[int, int],
    threshold: float = WRECK_TEMPLATE_THRESHOLD,
) -> bool:
    if not SUBMARINE_HIT_WRECK_TEMPLATE.exists():
        return False
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
    match = find_template_multi_scale(
        crop,
        SUBMARINE_HIT_WRECK_TEMPLATE,
        scales=WRECK_TEMPLATE_SCALES,
        threshold=threshold,
    )
    return match is not None


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


def visible_wreck_static_detected(image: np.ndarray, point: tuple[int, int]) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    if red_hit_marker_visible(image, point):
        return True
    try:
        result = classify_diamond_hit(image, image, point)
    except Exception:
        return False
    return str(getattr(result, "state", "")).strip().lower() == "hit"
