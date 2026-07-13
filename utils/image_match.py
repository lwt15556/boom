from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_TEMPLATE_SHAPE_POWER,
    DEFAULT_TEMPLATE_SHAPE_WEIGHT,
)


@dataclass(frozen=True)
class MatchResult:
    template_path: Path
    top_left: tuple[int, int]
    bottom_right: tuple[int, int]
    center: tuple[int, int]
    score: float


def find_template(
    screenshot,
    template_path: str | Path,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    shape_weight: float = DEFAULT_TEMPLATE_SHAPE_WEIGHT,
    shape_power: float = DEFAULT_TEMPLATE_SHAPE_POWER,
) -> MatchResult | None:
    """Match a template against one screenshot."""
    path = Path(template_path)
    template, mask = _read_template(path)
    screenshot = _normalize_screenshot(screenshot)
    shape_weight = _normalize_weight(shape_weight)
    shape_power = max(float(shape_power), 1.0)
    return _match_prepared_template(
        screenshot=screenshot,
        template=template,
        mask=mask,
        template_path=path,
        threshold=threshold,
        shape_weight=shape_weight,
        shape_power=shape_power,
    )


def find_template_multi_scale(
    screenshot,
    template_path: str | Path,
    *,
    scales: Sequence[float],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
    shape_weight: float = DEFAULT_TEMPLATE_SHAPE_WEIGHT,
    shape_power: float = DEFAULT_TEMPLATE_SHAPE_POWER,
) -> MatchResult | None:
    """Match a template across several scales and return the best hit."""
    if not scales:
        raise ValueError("scales cannot be empty")

    path = Path(template_path)
    template, mask = _read_template(path)
    screenshot = _normalize_screenshot(screenshot)
    shape_weight = _normalize_weight(shape_weight)
    shape_power = max(float(shape_power), 1.0)

    best_match: MatchResult | None = None
    best_score = -1.0
    for scale in scales:
        scaled_template, scaled_mask = _resize_template(template, mask, float(scale))
        match = _match_prepared_template(
            screenshot=screenshot,
            template=scaled_template,
            mask=scaled_mask,
            template_path=path,
            threshold=threshold,
            shape_weight=shape_weight,
            shape_power=shape_power,
        )
        if match is not None and match.score > best_score:
            best_match = match
            best_score = match.score

    return best_match


def _match_prepared_template(
    *,
    screenshot,
    template,
    mask,
    template_path: Path,
    threshold: float,
    shape_weight: float,
    shape_power: float,
) -> MatchResult | None:
    if template.shape[0] > screenshot.shape[0] or template.shape[1] > screenshot.shape[1]:
        return None

    if mask is None:
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    else:
        color_result = cv2.matchTemplate(screenshot, template, cv2.TM_CCORR_NORMED, mask=mask)
        shape_quality = _match_alpha_shape(screenshot, template, mask)
        result = _combine_color_and_shape(color_result, shape_quality, shape_weight, shape_power)

    _, max_score, _, max_loc = cv2.minMaxLoc(result)
    if max_score < threshold:
        return None

    template_height, template_width = template.shape[:2]
    top_left = max_loc
    bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
    center = (top_left[0] + template_width // 2, top_left[1] + template_height // 2)
    return MatchResult(
        template_path=template_path,
        top_left=top_left,
        bottom_right=bottom_right,
        center=center,
        score=max_score,
    )


def _read_template(path: Path):
    raw_template = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw_template is None:
        raise FileNotFoundError(f"unable to read template: {path}")

    if raw_template.ndim == 2:
        return cv2.cvtColor(raw_template, cv2.COLOR_GRAY2BGR), None

    if raw_template.shape[2] != 4:
        return raw_template, None

    template = raw_template[:, :, :3]
    alpha = raw_template[:, :, 3]
    x, y, width, height = cv2.boundingRect(alpha)
    if width == 0 or height == 0:
        raise ValueError(f"template alpha channel is empty: {path}")

    return template[y : y + height, x : x + width], alpha[y : y + height, x : x + width]


def _resize_template(template, mask, scale: float):
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"scale must be positive: {scale}")
    if abs(scale - 1.0) < 1e-6:
        return template, mask

    width = max(1, int(round(template.shape[1] * scale)))
    height = max(1, int(round(template.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_template = cv2.resize(template, (width, height), interpolation=interpolation)
    resized_mask = None if mask is None else cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized_template, resized_mask


def _normalize_screenshot(screenshot):
    if screenshot is None:
        raise ValueError("screenshot cannot be None")
    if screenshot.ndim == 2:
        return cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
    if screenshot.ndim == 3 and screenshot.shape[2] == 4:
        return screenshot[:, :, :3]
    return screenshot


def _match_alpha_shape(screenshot, template, mask):
    alpha = (mask > 16).astype("uint8") * 255
    if _is_almost_rect_mask(alpha):
        return None

    kernel = np.ones((3, 3), np.uint8)
    template_edges = cv2.morphologyEx(alpha, cv2.MORPH_GRADIENT, kernel)
    template_edges = cv2.dilate(template_edges, kernel, iterations=1)
    if cv2.countNonZero(template_edges) == 0:
        return None

    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    screenshot_edges = cv2.Canny(screenshot_gray, 50, 150)
    shape_result = cv2.matchTemplate(screenshot_edges, template_edges, cv2.TM_CCORR_NORMED)
    edge_quality = _normalize_shape_score(shape_result)

    silhouette_quality = _match_solid_color_silhouette(screenshot, template, alpha)
    if silhouette_quality is None:
        return edge_quality
    return np.minimum(edge_quality, silhouette_quality)


def _combine_color_and_shape(color_result, shape_quality, shape_weight: float, shape_power: float):
    if shape_quality is None or shape_weight <= 0:
        return color_result

    shape_quality = np.power(shape_quality, shape_power)
    return color_result * ((1.0 - shape_weight) + shape_weight * shape_quality)


def _is_almost_rect_mask(alpha) -> bool:
    fill_ratio = cv2.countNonZero(alpha) / alpha.size
    return fill_ratio >= 0.92


def _match_solid_color_silhouette(screenshot, template, alpha):
    foreground = template[alpha > 0]
    if foreground.size == 0:
        return None

    color_std = float(np.mean(np.std(foreground.reshape(-1, 3), axis=0)))
    if color_std > 35:
        return None

    median_color = np.median(foreground.reshape(-1, 3), axis=0)
    tolerance = max(45.0, color_std * 3.0)
    distance = np.linalg.norm(screenshot.astype(np.float32) - median_color.astype(np.float32), axis=2)
    candidate_mask = (distance <= tolerance).astype("uint8")
    alpha_binary = (alpha > 0).astype("uint8")

    intersection = cv2.matchTemplate(candidate_mask, alpha_binary, cv2.TM_CCORR)
    rect_count = cv2.boxFilter(
        candidate_mask.astype("float32"),
        ddepth=-1,
        ksize=(alpha.shape[1], alpha.shape[0]),
        normalize=False,
        anchor=(0, 0),
    )
    rect_count = rect_count[: intersection.shape[0], : intersection.shape[1]]

    alpha_count = float(alpha_binary.sum())
    union = alpha_count + rect_count - intersection
    silhouette_iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype="float32"),
        where=union > 0,
    )
    return np.clip(silhouette_iou / 0.65, 0.0, 1.0)


def _normalize_shape_score(shape_result):
    shape_result = np.nan_to_num(shape_result, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(shape_result / 0.45, 0.0, 1.0)


def _normalize_weight(weight: float) -> float:
    return min(max(float(weight), 0.0), 1.0)


__all__ = ["MatchResult", "find_template", "find_template_multi_scale"]
