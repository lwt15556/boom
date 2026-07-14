from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import TEMPLATE_DIR
from utils.image_match import find_template_multi_scale
from utils.submarine_strategy import Cell

SUBMARINE_HIT_WRECK_TEMPLATE = TEMPLATE_DIR / "submarine_hit_wreck.png"
VISIBLE_WRECK_TEMPLATES = tuple(
    TEMPLATE_DIR / f"visible_wreck_{index}.png"
    for index in range(1, 4)
)
WRECK_TEMPLATE_THRESHOLD = 0.70
WRECK_GLOBAL_TEMPLATE_THRESHOLD = 0.90
WRECK_TEMPLATE_SCALES = (0.75, 0.9, 1.0, 1.1, 1.25)
RED_HIT_MARKER_MIN_AREA = 80
RED_HIT_MARKER_MIN_WIDTH = 12
RED_HIT_MARKER_MIN_HEIGHT = 8
FULL_SHIP_MIN_GRAY_RATIO = 0.50
FULL_SHIP_MIN_COMPONENT_AREA = 450
FULL_SHIP_MIN_COMPONENT_WIDTH = 35
FULL_SHIP_MIN_COMPONENT_HEIGHT = 20
FULL_SHIP_MAX_CENTER_OFFSET = 10


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

    x, y = point
    h, w = image.shape[:2]
    crop_w = 120
    crop_h = 96
    x1 = max(0, int(x - crop_w // 2))
    y1 = max(0, int(y - crop_h // 2))
    x2 = min(w, int(x + crop_w // 2))
    y2 = min(h, int(y + crop_h // 2))
    if x2 <= x1 or y2 <= y1:
        return False

    crop = image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    local_center = (int(x - x1), int(y - y1))
    center_mask = np.zeros((ch, cw), dtype=np.uint8)
    pts = np.array(
        [
            [local_center[0], local_center[1] - 16],
            [local_center[0] + 28, local_center[1]],
            [local_center[0], local_center[1] + 16],
            [local_center[0] - 28, local_center[1]],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(center_mask, pts, 255)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(crop)
    _h, s, v = cv2.split(hsv)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    rgb_delta = max_rgb.astype(np.int16) - min_rgb.astype(np.int16)

    gray_white = (
        (s <= 65)
        & (rgb_delta <= 58)
        & (v >= 80)
        & (v <= 245)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gray_white = cv2.morphologyEx(gray_white, cv2.MORPH_OPEN, kernel)
    gray_white = cv2.morphologyEx(gray_white, cv2.MORPH_CLOSE, kernel)

    masked = cv2.bitwise_and(gray_white, gray_white, mask=center_mask)
    area = max(1, int(np.count_nonzero(center_mask)))
    ratio = int(np.count_nonzero(masked)) / area
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(masked, connectivity=8)
    largest = 0
    center_offset = (999.0, 999.0)
    largest_bbox = (0, 0)
    edge_density = 0.0
    mean_value = 999.0
    if num_labels > 1:
        largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest = int(stats[largest_index, cv2.CC_STAT_AREA])
        largest_bbox = (
            int(stats[largest_index, cv2.CC_STAT_WIDTH]),
            int(stats[largest_index, cv2.CC_STAT_HEIGHT]),
        )
        center_offset = (
            float(centroids[largest_index][0] - local_center[0]),
            float(centroids[largest_index][1] - local_center[1]),
        )
        component_mask = (_labels == largest_index).astype(np.uint8) * 255
        edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = int(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=component_mask))) / max(1, largest)
        mean_value = float(np.mean(v[component_mask > 0]))
    component_ratio = largest / area

    full_ship_body = (
        ratio >= FULL_SHIP_MIN_GRAY_RATIO
        and component_ratio >= FULL_SHIP_MIN_GRAY_RATIO
        and largest >= FULL_SHIP_MIN_COMPONENT_AREA
        and largest_bbox[0] >= FULL_SHIP_MIN_COMPONENT_WIDTH
        and largest_bbox[1] >= FULL_SHIP_MIN_COMPONENT_HEIGHT
        and abs(center_offset[0]) <= FULL_SHIP_MAX_CENTER_OFFSET
        and abs(center_offset[1]) <= FULL_SHIP_MAX_CENTER_OFFSET
    )
    wreck_fragment = (
        ratio >= 0.18
        and component_ratio >= 0.18
        and largest >= 220
        and largest_bbox[0] >= 30
        and largest_bbox[1] >= 22
        and edge_density >= 0.23
        and mean_value <= 185.0
        and abs(center_offset[0]) <= 14
        and abs(center_offset[1]) <= 14
    )
    return full_ship_body or wreck_fragment
