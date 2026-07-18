from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import math
from typing import Callable

import cv2
import numpy as np

from utils.diamond_hit import (
    DiamondHitConfig,
    classify_diamond_hit,
    make_diamond_mask,
    ratio_in_mask,
)


@dataclass(frozen=True)
class TranslationRegistration:
    dx: float
    dy: float
    response: float
    accepted: bool


@dataclass(frozen=True)
class LocalMotionMetrics:
    inner_ratio: float
    outer_ratio: float
    contrast: float


@dataclass(frozen=True)
class StableHitAnalysis:
    result: object
    stable_image: np.ndarray
    registrations: tuple[TranslationRegistration, ...]
    motion: LocalMotionMetrics


def _validate_pair(reference: np.ndarray, image: np.ndarray) -> None:
    for name, value in (("reference", reference), ("image", image)):
        if not isinstance(value, np.ndarray) or value.ndim != 3 or value.size == 0:
            raise ValueError(f"{name} must be a non-empty BGR image")
    if reference.shape != image.shape:
        raise ValueError("reference and image must have identical shapes")


def register_translation(
    reference: np.ndarray,
    image: np.ndarray,
    *,
    max_translation: float = 8.0,
    min_response: float = 0.08,
) -> tuple[np.ndarray, TranslationRegistration]:
    _validate_pair(reference, image)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    window = cv2.createHanningWindow(
        (reference_gray.shape[1], reference_gray.shape[0]),
        cv2.CV_32F,
    )
    (dx, dy), response = cv2.phaseCorrelate(reference_gray, image_gray, window)
    finite = np.isfinite((dx, dy, response)).all()
    accepted = bool(
        finite
        and float(response) >= float(min_response)
        and abs(float(dx)) <= float(max_translation)
        and abs(float(dy)) <= float(max_translation)
    )
    registration = TranslationRegistration(
        dx=float(dx) if finite else 0.0,
        dy=float(dy) if finite else 0.0,
        response=float(response) if finite else 0.0,
        accepted=accepted,
    )
    if not accepted:
        return image.copy(), registration

    transform = np.float32([[1, 0, -dx], [0, 1, -dy]])
    aligned = cv2.warpAffine(
        image,
        transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, registration


def build_stable_after_image(
    reference: np.ndarray,
    after_images: Sequence[np.ndarray],
    *,
    max_translation: float = 8.0,
) -> tuple[np.ndarray, tuple[TranslationRegistration, ...]]:
    if not after_images:
        raise ValueError("after_images must contain at least one image")

    aligned: list[np.ndarray] = []
    registrations: list[TranslationRegistration] = []
    for image in after_images:
        registered, registration = register_translation(
            reference,
            image,
            max_translation=max_translation,
        )
        aligned.append(registered)
        registrations.append(registration)

    stable = np.median(np.stack(aligned, axis=0), axis=0)
    return np.rint(stable).astype(reference.dtype), tuple(registrations)


def local_motion_contrast(
    before: np.ndarray,
    after: np.ndarray,
    center: tuple[int, int],
    diamond_w: int,
    diamond_h: int,
    *,
    diff_threshold: int = 18,
    inner_scale: float = 0.72,
    outer_scale: float = 1.25,
) -> LocalMotionMetrics:
    _validate_pair(before, after)
    height, width = before.shape[:2]
    inner = make_diamond_mask(
        (height, width), center, diamond_w, diamond_h, scale=inner_scale
    )
    outer = make_diamond_mask(
        (height, width), center, diamond_w, diamond_h, scale=outer_scale
    )
    outer_ring = cv2.subtract(outer, inner)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    changed = (
        cv2.absdiff(before_gray, after_gray) >= max(0, int(diff_threshold))
    ).astype(np.uint8) * 255
    inner_ratio = ratio_in_mask(changed, inner)
    outer_ratio = ratio_in_mask(changed, outer_ring)
    return LocalMotionMetrics(
        inner_ratio=inner_ratio,
        outer_ratio=outer_ratio,
        contrast=inner_ratio - outer_ratio,
    )


def analyze_stable_hit(
    before: np.ndarray,
    after_images: Sequence[np.ndarray],
    center: tuple[int, int],
    config: DiamondHitConfig | None = None,
    *,
    classifier: Callable[..., object] = classify_diamond_hit,
) -> StableHitAnalysis:
    effective_config = config or DiamondHitConfig()
    stable_image, registrations = build_stable_after_image(before, after_images)
    result = classifier(before, stable_image, center, config=config)
    refined_center = getattr(result, "refined_center", center)
    motion = local_motion_contrast(
        before,
        stable_image,
        refined_center,
        effective_config.diamond_w,
        effective_config.diamond_h,
        diff_threshold=effective_config.diff_threshold,
        inner_scale=effective_config.inner_scale,
    )
    return StableHitAnalysis(
        result=result,
        stable_image=stable_image,
        registrations=registrations,
        motion=motion,
    )


def stable_hit_is_suspect(
    analysis: StableHitAnalysis,
    *,
    min_score: float = 0.72,
    min_inner_ratio: float = 0.08,
    min_motion_contrast: float = 0.025,
) -> bool:
    result = analysis.result
    if getattr(result, "state", None) != "hit" and float(
        getattr(result, "score", 0.0)
    ) < min_score:
        return False
    if analysis.motion.inner_ratio < min_inner_ratio:
        return False
    if analysis.motion.contrast < min_motion_contrast:
        return False
    required = max(2, math.ceil(len(analysis.registrations) / 2))
    accepted = sum(1 for item in analysis.registrations if item.accepted)
    return accepted >= required


__all__ = [
    "LocalMotionMetrics",
    "StableHitAnalysis",
    "TranslationRegistration",
    "build_stable_after_image",
    "analyze_stable_hit",
    "local_motion_contrast",
    "register_translation",
    "stable_hit_is_suspect",
]
