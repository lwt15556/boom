from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    LEVEL_DETECTION_MIN_MARGIN,
    LEVEL_DETECTION_MIN_SCORE,
    LEVEL_REFERENCE_DIR,
)
from utils.diamond_centers import read_image


@dataclass(frozen=True)
class LevelRecognitionResult:
    level: int
    score: float
    second_level: int | None
    second_score: float
    confident: bool


def recognize_level_from_screenshot(
    screenshot: np.ndarray,
    *,
    reference_dir: str | Path = LEVEL_REFERENCE_DIR,
    candidate_levels: Iterable[int] | None = None,
    min_score: float = LEVEL_DETECTION_MIN_SCORE,
    min_margin: float = LEVEL_DETECTION_MIN_MARGIN,
) -> LevelRecognitionResult | None:
    """Recognize the current sonar level by comparing against saved screenshots."""
    if screenshot is None:
        raise ValueError("screenshot cannot be None")

    reference_dir = Path(reference_dir)
    candidates = _candidate_level_set(candidate_levels)
    target = _prepare_level_image(screenshot)
    matches: list[tuple[float, int]] = []

    for reference_path in _iter_reference_images(reference_dir, candidates):
        reference = read_image(reference_path)
        score = _normalized_correlation(target, _prepare_level_image(reference))
        matches.append((score, int(reference_path.stem)))

    if not matches:
        return None

    matches.sort(reverse=True)
    best_score, best_level = matches[0]
    second_score, second_level = matches[1] if len(matches) > 1 else (-1.0, None)
    confident = best_score >= min_score and (best_score - second_score) >= min_margin
    return LevelRecognitionResult(
        level=best_level,
        score=best_score,
        second_level=second_level,
        second_score=second_score,
        confident=confident,
    )


def _candidate_level_set(candidate_levels: Iterable[int] | None) -> set[int] | None:
    if candidate_levels is None:
        return None
    return {int(level) for level in candidate_levels}


def _iter_reference_images(
    reference_dir: Path,
    candidate_levels: set[int] | None,
) -> list[Path]:
    if not reference_dir.exists():
        return []

    paths = [
        path
        for path in reference_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and path.stem.isdigit()
        and (candidate_levels is None or int(path.stem) in candidate_levels)
    ]
    return sorted(paths, key=lambda path: int(path.stem))


def _prepare_level_image(img: np.ndarray) -> np.ndarray:
    """Crop the stable activity area and normalize it for correlation matching."""
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    x1 = int(width * 0.27)
    x2 = int(width * 0.74)
    y1 = int(height * 0.16)
    y2 = int(height * 0.86)
    crop = gray[y1:y2, x1:x2]
    resized = cv2.resize(crop, (240, 200), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (3, 3), 0).astype(np.float32)


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= float(a.mean())
    b -= float(b.mean())
    denominator = float(a.std() * b.std())
    if denominator <= 1e-6:
        return -1.0
    return float(np.mean((a / denominator) * b))


__all__ = ["LevelRecognitionResult", "recognize_level_from_screenshot"]
