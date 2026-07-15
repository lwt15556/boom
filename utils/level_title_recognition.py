from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class LevelTitleRecognitionResult:
    level: int
    score: float
    second_level: int | None
    second_score: float
    confident: bool


def recognize_level_title(
    screenshot: np.ndarray,
    *,
    reference_dir: str | Path,
    min_score: float = 0.78,
    min_margin: float = 0.04,
) -> LevelTitleRecognitionResult | None:
    """Recognize the level number from the top title text such as "2号海域"."""
    if screenshot is None:
        raise ValueError("screenshot cannot be None")

    target_components = _extract_leading_digit_masks(_prepare_title_roi(screenshot))
    if not target_components:
        return None

    digit_templates = _load_single_digit_templates(str(Path(reference_dir).resolve()))
    if not digit_templates:
        return None

    digit_choices: list[list[tuple[float, str]]] = []
    for component in target_components:
        best_by_digit: dict[str, float] = {}
        for digit, template in digit_templates:
            score = _mask_iou(component, template)
            best_by_digit[digit] = max(best_by_digit.get(digit, -1.0), score)
        choices = [(score, digit) for digit, score in best_by_digit.items()]
        choices.sort(reverse=True)
        digit_choices.append(choices)

    best_digits = [choices[0][1] for choices in digit_choices]
    best_scores = [choices[0][0] for choices in digit_choices]
    best_level = int("".join(best_digits))
    best_score = float(np.mean(best_scores))

    second_candidates: list[tuple[float, int]] = []
    for index, choices in enumerate(digit_choices):
        if len(choices) < 2:
            continue
        alternative_digits = best_digits.copy()
        alternative_digits[index] = choices[1][1]
        alternative_scores = best_scores.copy()
        alternative_scores[index] = choices[1][0]
        second_candidates.append(
            (float(np.mean(alternative_scores)), int("".join(alternative_digits)))
        )

    if second_candidates:
        second_score, second_level = max(second_candidates)
    else:
        second_score, second_level = -1.0, None

    confident = best_score >= min_score and (best_score - second_score) >= min_margin
    return LevelTitleRecognitionResult(
        level=best_level,
        score=best_score,
        second_level=second_level,
        second_score=second_score,
        confident=confident,
    )


def _iter_reference_images(reference_dir: Path) -> list[Path]:
    if not reference_dir.exists():
        return []
    paths = [
        path
        for path in reference_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and path.stem.isdigit()
    ]
    return sorted(paths, key=lambda path: int(path.stem))


@lru_cache(maxsize=8)
def _load_single_digit_templates(reference_dir_text: str) -> tuple[tuple[str, np.ndarray], ...]:
    templates: list[tuple[str, np.ndarray]] = []
    reference_dir = Path(reference_dir_text)
    for number in range(1, 11):
        reference_path = reference_dir / f"{number}.png"
        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            continue
        components = _extract_leading_digit_masks(_prepare_title_roi(reference))
        digits = str(number)
        if len(components) != len(digits):
            continue
        templates.extend(zip(digits, components, strict=True))
    return tuple(templates)


def _prepare_title_roi(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
    else:
        bgr = img

    height, width = bgr.shape[:2]
    x1 = int(width * 0.22)
    x2 = int(width * 0.78)
    y1 = 0
    y2 = int(height * 0.18)
    return bgr[y1:y2, x1:x2].copy()


def _extract_leading_digit_masks(title_roi: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(title_roi, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    components: list[tuple[int, int, int, int]] = []
    roi_h, roi_w = white.shape[:2]
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 30:
            continue
        if h < roi_h * 0.18 or h > roi_h * 0.75:
            continue
        if y > roi_h * 0.58:
            continue
        if y > roi_h * 0.24 or h < roi_h * 0.24:
            continue
        components.append((int(x), int(y), int(w), int(h)))

    if not components:
        return []

    components.sort()
    digit_components = [components[0]]
    for component in components[1:]:
        prev_x, _, prev_w, _ = digit_components[-1]
        gap = component[0] - (prev_x + prev_w)
        if gap > roi_w * 0.07:
            break
        digit_components.append(component)

    masks: list[np.ndarray] = []
    for x, y, w, h in digit_components:
        x1 = max(0, x - 3)
        y1 = max(0, y - 3)
        x2 = min(roi_w, x + w + 3)
        y2 = min(roi_h, y + h + 3)
        crop = white[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        masks.append(cv2.resize(crop, (40, 56), interpolation=cv2.INTER_AREA))
    return masks


def _mask_iou(target: np.ndarray, template: np.ndarray) -> float:
    target = target.astype(np.float32) / 255.0
    template = template.astype(np.float32) / 255.0
    intersection = float(np.minimum(target, template).sum())
    union = float(np.maximum(target, template).sum())
    if union <= 1e-6:
        return -1.0
    return intersection / union


__all__ = ["LevelTitleRecognitionResult", "recognize_level_title"]
