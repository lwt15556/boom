from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import cv2
import numpy as np

from config import TEMPLATE_DIR
from utils.diamond_hit import DiamondHitConfig, classify_diamond_hit, make_diamond_mask
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
# On 10x10 boards the title/subtitle overlaps these five upper diamonds in
# the calibrated layout.  Keep this as a cell-level rule: other board sizes
# and other cells must retain the normal static recognition path.
TITLE_OCCLUDED_CELLS_10X10 = frozenset(
    {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
    }
)

# Static recognition is intentionally conservative.  The game renders a
# moving, blue/teal highlight over the upper-right part of the board; its
# pixels can satisfy the gray/bright heuristics used for wrecks.  These
# thresholds are deliberately exposed so replay tooling can tune them without
# changing the recognition flow.
SURFACE_GLARE_CYAN_MIN_RATIO = 0.16
SURFACE_GLARE_BRIGHT_MIN_RATIO = 0.34
SURFACE_GLARE_TEMPORAL_MAD = 3.5
SURFACE_GLARE_WEAK_SHAPE_SCORE = 0.34
WRECK_SHAPE_MIN_SCORE = 0.28


@dataclass(frozen=True)
class SurfaceWaterBaseline:
    """Compact temporal baseline for an unclicked board.

    ``median_gray`` describes the stable water/background appearance while
    ``temporal_mad`` records how much each pixel moved during the baseline
    capture.  Keeping the baseline in grayscale avoids retaining several full
    colour screenshots in the long-running process.
    """

    median_gray: np.ndarray
    temporal_mad: np.ndarray
    frame_count: int


@dataclass(frozen=True)
class WreckShapeMetrics:
    """Shape and colour evidence measured inside one calibrated diamond."""

    center_gray_ratio: float
    ring_gray_ratio: float
    gray_excess: float
    component_ratio: float
    compactness: float
    cyan_ratio: float
    bright_ratio: float
    score: float


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


def build_surface_water_baseline(
    frames: Sequence[np.ndarray],
) -> SurfaceWaterBaseline | None:
    """Build a temporal baseline from consecutive pre-click screenshots.

    Invalid frames and frames with a different size are ignored.  A baseline
    is useful with one frame as a spatial reference, but temporal glare
    detection is enabled only when at least two valid frames are available.
    Returning ``None`` for an entirely invalid input keeps callers fail-closed.
    """

    valid: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for frame in frames:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            continue
        if frame.shape[2] not in (3, 4) or frame.size == 0:
            continue
        if shape is None:
            shape = frame.shape[:2]
        if frame.shape[:2] != shape:
            continue
        if frame.shape[2] == 4:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            except cv2.error:
                continue
        if frame.dtype != np.uint8:
            try:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            except (TypeError, ValueError):
                continue
        valid.append(frame)

    if not valid:
        return None

    try:
        gray_frames = np.stack(
            [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in valid],
            axis=0,
        ).astype(np.float32)
        median_gray = np.median(gray_frames, axis=0).astype(np.uint8)
        # Median absolute deviation is less sensitive than a range to one
        # dropped/transition frame and is inexpensive at 1280x720.
        temporal_mad = np.median(
            np.abs(gray_frames - median_gray.astype(np.float32)),
            axis=0,
        ).astype(np.float32)
    except (cv2.error, TypeError, ValueError):
        return None

    return SurfaceWaterBaseline(
        median_gray=median_gray,
        temporal_mad=temporal_mad,
        frame_count=len(valid),
    )


def is_title_occluded_cell(cell: Cell, grid_size: int) -> bool:
    """Return whether a calibrated board cell is covered by the level title."""
    try:
        normalized = (int(cell[0]), int(cell[1]))
        size = int(grid_size)
    except (TypeError, ValueError, IndexError):
        return False
    return size == 10 and normalized in TITLE_OCCLUDED_CELLS_10X10


def _diamond_evidence_mask(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    cell_polygon: np.ndarray | None = None,
    scale: float = 0.72,
    exclude_activity_title_overlay: bool = True,
) -> np.ndarray | None:
    """Return an eroded interior mask that excludes grid borders/UI text."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.size == 0:
        return None
    height, width = image.shape[:2]
    try:
        x, y = int(point[0]), int(point[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (0 <= x < width and 0 <= y < height):
        return None

    if cell_polygon is not None:
        polygon = np.asarray(cell_polygon, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        # Two erosions remove antialiased borders and the bright grid stroke.
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=2)
    else:
        # The default dimensions match the 1280x720 calibration.  Scale them
        # with the screenshot so replaying a resized image uses the same
        # relative footprint.
        scale_factor = max(0.25, min(width / 1280.0, height / 720.0))
        diamond_w = max(20, int(round(80 * scale_factor)))
        diamond_h = max(16, int(round(56 * scale_factor)))
        local = make_diamond_mask(
            (height, width),
            (x, y),
            diamond_w,
            diamond_h,
            scale=max(0.25, float(scale)),
        )
        mask = cv2.erode(local, np.ones((3, 3), dtype=np.uint8), iterations=1)

    if exclude_activity_title_overlay:
        _exclude_activity_title_overlay(
            mask,
            image_shape=image.shape,
            roi_left=0,
            roi_top=0,
        )
    if not np.any(mask):
        return None
    return mask


def wreck_shape_metrics(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    cell_polygon: np.ndarray | None = None,
    exclude_activity_title_overlay: bool = True,
) -> WreckShapeMetrics:
    """Measure whether a bright/gray region resembles a compact wreck.

    Grid lines and water reflections tend to occupy a broad, low-contrast or
    blue/teal area.  A wreck has a neutral-gray core, a compact connected
    component, and a positive centre-versus-ring contrast.  The result is a
    score only; callers still need dynamic or completion evidence before
    treating a cell as an authoritative hit.
    """

    zero = WreckShapeMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mask = _diamond_evidence_mask(
        image,
        point,
        cell_polygon=cell_polygon,
        exclude_activity_title_overlay=exclude_activity_title_overlay,
    )
    if mask is None:
        return zero

    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        b, g, r = cv2.split(image)
    except (cv2.error, ValueError):
        return zero

    # The centre is a second erosion, not a fixed screen-space rectangle.
    center_mask = cv2.erode(mask, np.ones((7, 7), dtype=np.uint8), iterations=1)
    if not np.any(center_mask):
        center_mask = mask
    ring_mask = cv2.subtract(mask, center_mask)
    neutral_gray = (
        (saturation <= 95)
        & (value >= 90)
        & (value <= 235)
    ).astype(np.uint8) * 255
    center_gray = cv2.bitwise_and(neutral_gray, neutral_gray, mask=center_mask)
    ring_gray = cv2.bitwise_and(neutral_gray, neutral_gray, mask=ring_mask)
    center_area = max(1, int(np.count_nonzero(center_mask)))
    ring_area = max(1, int(np.count_nonzero(ring_mask)))
    center_ratio = float(np.count_nonzero(center_gray)) / center_area
    ring_ratio = float(np.count_nonzero(ring_gray)) / ring_area
    gray_excess = center_ratio - ring_ratio

    labels, _label_map, stats, _centroids = cv2.connectedComponentsWithStats(
        center_gray,
        connectivity=8,
    )
    largest = 0
    largest_bbox_area = 0
    for label_index in range(1, labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area > largest:
            largest = area
            largest_bbox_area = int(
                stats[label_index, cv2.CC_STAT_WIDTH]
                * stats[label_index, cv2.CC_STAT_HEIGHT]
            )
    component_ratio = largest / float(center_area)
    compactness = (
        largest / float(max(1, largest_bbox_area))
        if largest_bbox_area
        else 0.0
    )

    valid_mask = mask > 0
    cyan = (
        (saturation >= 60)
        & (value >= 120)
        & (
            (b.astype(np.int16) - r.astype(np.int16) >= 8)
            | (g.astype(np.int16) - r.astype(np.int16) >= 12)
        )
    )
    bright = value >= 170
    cyan_ratio = float(np.count_nonzero(cyan & valid_mask)) / max(
        1, int(np.count_nonzero(valid_mask))
    )
    bright_ratio = float(np.count_nonzero(bright & valid_mask)) / max(
        1, int(np.count_nonzero(valid_mask))
    )

    def _signal(value_: float, threshold: float, span: float) -> float:
        return max(0.0, min(1.0, (value_ - threshold) / max(span, 1e-6)))

    score = (
        0.34 * _signal(center_ratio, 0.10, 0.22)
        + 0.28 * _signal(max(0.0, gray_excess), 0.025, 0.12)
        + 0.24 * _signal(component_ratio, 0.025, 0.18)
        + 0.14 * compactness
        - 0.24 * _signal(cyan_ratio, 0.08, 0.32)
    )
    return WreckShapeMetrics(
        center_gray_ratio=center_ratio,
        ring_gray_ratio=ring_ratio,
        gray_excess=gray_excess,
        component_ratio=component_ratio,
        compactness=compactness,
        cyan_ratio=cyan_ratio,
        bright_ratio=bright_ratio,
        score=max(0.0, min(1.0, score)),
    )


def surface_glare_score(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    baseline: SurfaceWaterBaseline | None = None,
    cell_polygon: np.ndarray | None = None,
    relative_position: tuple[float, float] | None = None,
    _metrics: WreckShapeMetrics | None = None,
) -> float:
    """Return a normalized score for broad blue/teal surface reflection."""

    # ``surface_reflection_detected`` already computes the shape metrics for
    # its final gate.  The private hand-off avoids running the full-frame HSV
    # and connected-component pass twice for every cell while keeping the
    # public helper convenient for replay tooling.
    metrics = _metrics or wreck_shape_metrics(image, point, cell_polygon=cell_polygon)
    mask = _diamond_evidence_mask(image, point, cell_polygon=cell_polygon)
    if mask is None:
        return 0.0
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        b, g, r = cv2.split(image)
    except (cv2.error, ValueError):
        return 0.0

    valid_mask = mask > 0
    cyan = (
        (saturation >= 60)
        & (value >= 120)
        & (
            (b.astype(np.int16) - r.astype(np.int16) >= 8)
            | (g.astype(np.int16) - r.astype(np.int16) >= 12)
        )
    )
    bright = value >= 170
    cyan_ratio = float(np.count_nonzero(cyan & valid_mask)) / max(
        1, int(np.count_nonzero(valid_mask))
    )
    bright_ratio = float(np.count_nonzero(bright & valid_mask)) / max(
        1, int(np.count_nonzero(valid_mask))
    )

    temporal_score = 0.0
    baseline_residual_score = 0.0
    if (
        baseline is not None
        and baseline.temporal_mad.shape == image.shape[:2]
        and baseline.median_gray.shape == image.shape[:2]
    ):
        if baseline.frame_count >= 2:
            temporal = baseline.temporal_mad[valid_mask]
            if temporal.size:
                temporal_score = max(0.0, min(1.0, float(np.median(temporal)) / 8.0))
        # MAD describes motion *within* the baseline and can be zero when a
        # highlight appears in only one of the sampled frames.  Compare the
        # current cell to the median water image as a second, local signal.
        try:
            current_gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float32)
        except (cv2.error, TypeError, ValueError):
            current_gray = None
        if current_gray is not None:
            residual = np.abs(
                current_gray - baseline.median_gray.astype(np.float32)
            )
            local_residual = residual[valid_mask]
            if local_residual.size:
                residual_p75 = float(np.percentile(local_residual, 75))
                baseline_residual_score = max(
                    0.0,
                    min(1.0, (residual_p75 - 3.0) / 20.0),
                )

    # A reflection is broad and bright relative to the local board water, or
    # varies over time.  Absolute cyan saturation is intentionally given only
    # a small weight because the entire board is blue/teal in normal frames.
    # The optional relative position lets the known upper-right glare band use
    # a slightly more sensitive gate without hard-coding screen coordinates.
    broad_score = max(0.0, min(1.0, 1.0 - metrics.score))
    bright_excess = max(0.0, min(1.0, (bright_ratio - 0.24) / 0.34))
    cyan_excess = max(0.0, min(1.0, (cyan_ratio - 0.62) / 0.32))
    upper_right = bool(
        relative_position is not None
        and float(relative_position[0]) <= 0.55
        and float(relative_position[1]) >= 0.55
    )
    score = (
        0.42 * bright_excess
        + 0.14 * cyan_excess
        + 0.20 * temporal_score
        + 0.12 * baseline_residual_score
        + 0.12 * broad_score
    )
    if upper_right:
        score += 0.10 * bright_excess
    return max(0.0, min(1.0, score))


def surface_reflection_detected(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    baseline: SurfaceWaterBaseline | None = None,
    cell_polygon: np.ndarray | None = None,
    relative_position: tuple[float, float] | None = None,
) -> bool:
    """Return whether the cell is dominated by water reflection/highlight."""

    metrics = wreck_shape_metrics(image, point, cell_polygon=cell_polygon)
    score = surface_glare_score(
        image,
        point,
        baseline=baseline,
        cell_polygon=cell_polygon,
        relative_position=relative_position,
        _metrics=metrics,
    )
    temporal_dynamic = False
    baseline_residual_dynamic = False
    if (
        baseline is not None
        and baseline.temporal_mad.shape == image.shape[:2]
        and baseline.median_gray.shape == image.shape[:2]
    ):
        mask = _diamond_evidence_mask(image, point, cell_polygon=cell_polygon)
        if mask is not None and np.any(mask):
            if baseline.frame_count >= 2:
                temporal_dynamic = (
                    float(np.median(baseline.temporal_mad[mask > 0]))
                    >= SURFACE_GLARE_TEMPORAL_MAD
                )
            try:
                current_gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                ).astype(np.float32)
            except (cv2.error, TypeError, ValueError):
                current_gray = None
            if current_gray is not None:
                residual = np.abs(
                    current_gray - baseline.median_gray.astype(np.float32)
                )
                local_residual = residual[mask > 0]
                baseline_residual_dynamic = bool(
                    local_residual.size
                    and float(np.percentile(local_residual, 75)) >= 10.0
                )

    upper_right = bool(
        relative_position is not None
        and float(relative_position[0]) <= 0.55
        and float(relative_position[1]) >= 0.55
    )
    # The upper-right band is where the supplied screenshots show the broad
    # specular reflection.  Outside that band require stronger temporal or
    # spatial evidence so ordinary blue water is not discarded.
    temporal_reflection = (
        (temporal_dynamic or baseline_residual_dynamic)
        and metrics.score < 0.55
    )
    broad_blue = (
        metrics.cyan_ratio >= SURFACE_GLARE_CYAN_MIN_RATIO
        and (
            temporal_reflection
            or (
                upper_right
                and (
                    metrics.score < SURFACE_GLARE_WEAK_SHAPE_SCORE
                    or metrics.bright_ratio >= SURFACE_GLARE_BRIGHT_MIN_RATIO
                )
            )
        )
    )
    broad_bright = (
        metrics.bright_ratio >= 0.52
        and metrics.score < SURFACE_GLARE_WEAK_SHAPE_SCORE
    )
    return bool(score >= 0.40 and (broad_blue or broad_bright))


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
    *,
    surface_baseline: SurfaceWaterBaseline | None = None,
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
        row, col = divmod(index, grid_size)
        cell = (row, col)
        # The title is drawn over the first five diamonds on 10x10 boards.
        # Leave them unknown during the initial static review; a later blue
        # probe still gets the full dynamic before/after confirmation path.
        if is_title_occluded_cell(cell, grid_size):
            continue
        if cell in red_marker_cells:
            continue
        if visible_wreck_static_detected(
            screenshot,
            point,
            ignore_submarine_marker=True,
            cell_polygon=cell_polygon(index),
            surface_baseline=surface_baseline,
            relative_position=(
                row / max(1, grid_size - 1),
                col / max(1, grid_size - 1),
            ),
            cell=cell,
            grid_size=grid_size,
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
        if distance_sq > max_point_distance_sq:
            continue

        # The flag is a decoration above/beside the surfaced hull.  On the
        # isometric board its nearest calibrated point can therefore be one
        # tile away from the actual submarine (for example, (2,2) versus the
        # hull at (3,2)).  Treat the nearest point as a seed and compare the
        # surrounding cells by hull evidence before binding the marker.  A
        # red component without any usable hull evidence still falls back to
        # the nearest point so marker-only compatibility tests remain stable;
        # callers will keep that anchor provisional until a real layout is
        # found.
        nearest_row, nearest_col = divmod(nearest_index, grid_size)
        candidate_indices: list[int] = []
        for index, point in enumerate(click_points):
            row, col = divmod(index, grid_size)
            if max(abs(row - nearest_row), abs(col - nearest_col)) > 1:
                continue
            candidate_distance_sq = (
                (float(point[0]) - center_x) ** 2
                + (float(point[1]) - center_y) ** 2
            )
            if candidate_distance_sq <= max_point_distance_sq:
                candidate_indices.append(index)

        best_index = nearest_index
        best_key: tuple[float, float, float, int] | None = None
        for index in candidate_indices:
            point = click_points[index]
            candidate_row, candidate_col = divmod(index, grid_size)
            candidate_distance_sq = (
                (float(point[0]) - center_x) ** 2
                + (float(point[1]) - center_y) ** 2
            )
            body_score = completed_ship_body_score(
                image,
                point,
                cell_polygon=grid_cell_polygon(click_points, index, grid_size),
            )
            # Prefer strong hull evidence first, then proximity.  The small
            # proximity tie-break keeps a flag centred on its own hull from
            # jumping to a similarly bright neighbouring cell.
            key = (
                float(body_score),
                -float(candidate_distance_sq),
                -float(
                    abs(candidate_row - nearest_row)
                    + abs(candidate_col - nearest_col)
                ),
                -index,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index

        if best_key is not None and best_key[0] < COMPLETED_SHIP_BODY_MIN_SCORE:
            best_index = nearest_index
        anchors.add((best_index // grid_size, best_index % grid_size))

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
    surface_baseline: SurfaceWaterBaseline | None = None,
    relative_position: tuple[float, float] | None = None,
    filter_surface_reflection: bool = True,
    cell: Cell | None = None,
    grid_size: int | None = None,
    filter_activity_title_overlay: bool = True,
) -> bool:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return False

    # A single static frame cannot distinguish the title/countdown strokes
    # from a wreck in the diamonds directly underneath.  Leave these cells
    # unknown so a real blue probe or completed-submarine geometry can decide
    # them; this is safer than silently committing a false existing hit.
    if filter_activity_title_overlay:
        if cell is not None and grid_size is not None:
            if is_title_occluded_cell(cell, grid_size):
                return False
        elif _inside_activity_title_overlay(image, point):
            # Preserve the legacy point-only guard for callers that do not
            # have board coordinates. Grid-aware callers use the fixed cell
            # rule above, so unrelated 10x10 cells are not affected.
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

    # Reject broad blue/teal specular highlights before template matching.  A
    # reflection can reach the old 0.965 template threshold even though it has
    # no compact neutral hull.  The gate is spatially relative to the board
    # (when supplied) and can also use the multi-frame temporal baseline.
    if filter_surface_reflection:
        if surface_reflection_detected(
            image,
            point,
            baseline=surface_baseline,
            cell_polygon=cell_polygon,
            relative_position=relative_position,
        ):
            return False
    if wreck_template_visible(image, point, cell_polygon=cell_polygon):
        # Template correlation alone is not enough for the static recovery
        # path: a small bright-water patch can match a masked template at a
        # high score.  Require a compact, centre-weighted neutral shape too.
        return wreck_shape_metrics(
            image,
            point,
            cell_polygon=cell_polygon,
            exclude_activity_title_overlay=filter_activity_title_overlay,
        ).score >= WRECK_SHAPE_MIN_SCORE
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
    shape = wreck_shape_metrics(
        image,
        point,
        cell_polygon=cell_polygon,
        exclude_activity_title_overlay=filter_activity_title_overlay,
    )
    if shape.score < WRECK_SHAPE_MIN_SCORE:
        return False
    if cell_polygon is None:
        return True
    refined = getattr(result, "refined_center", point)
    return cv2.pointPolygonTest(
        np.asarray(cell_polygon, dtype=np.float32),
        (float(refined[0]), float(refined[1])),
        False,
    ) >= 0
