"""Build additional wreck templates from full-board screenshots.

This project uses deterministic OpenCV rules, so this is exemplar generation,
not neural-network training.  Only cells confirmed by both the static detector
and the partial-template detector are used as positive samples.  The resulting
small set of crops is written to ``template/visible_wreck_generated_*.png``;
the runtime loader discovers them automatically.

Usage::

    .venv\\Scripts\\python.exe tools\\train_wreck_images.py
    .venv\\Scripts\\python.exe tools\\train_wreck_images.py --root "残骸识图"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LEVEL_GRID_SIZES
from save_points.points import read_saved_points
from utils.level_title_recognition import recognize_level_title
from utils.sidebar_progress import detect_partial_wreck_cells
from utils.wreck_detection import PARTIAL_WRECK_TEMPLATES, VISIBLE_WRECK_TEMPLATES, detect_visible_wreck_cells
from utils.image_io import write_image_compat


def _recognize_level(image: np.ndarray) -> int | None:
    title = recognize_level_title(
        image,
        reference_dir=PROJECT_ROOT / "save_points" / "imgs",
        min_score=0.60 if image.shape[1] < 1000 else 0.78,
    )
    if title is None or not title.confident:
        return None
    level = int(title.level)
    return level if level in LEVEL_GRID_SIZES else None


def _read_image(path: Path) -> np.ndarray | None:
    """Read paths containing non-ASCII characters on Windows reliably."""
    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
        if payload.size == 0:
            return None
        return cv2.imdecode(payload, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def _scaled_points(level: int, image: np.ndarray) -> list[tuple[int, int]] | None:
    grid_size = LEVEL_GRID_SIZES[level]
    points = read_saved_points(level, expected_n=grid_size)
    if not points:
        return None
    sx = image.shape[1] / 1280.0
    sy = image.shape[0] / 720.0
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in points]


def _positive_cells(image: np.ndarray, points: list[tuple[int, int]], grid_size: int) -> set[tuple[int, int]]:
    visible = detect_visible_wreck_cells(image, points, grid_size)
    partial = detect_partial_wreck_cells(
        image,
        points,
        grid_size=grid_size,
        template_paths=PARTIAL_WRECK_TEMPLATES,
    ) or set()
    # Requiring agreement keeps generated exemplars precision-first.  A
    # single detector can be affected by reflections or a neighbouring hull.
    return set(visible) & set(partial)


def _crop_wreck(image: np.ndarray, point: tuple[int, int]) -> np.ndarray | None:
    """Return a compact crop around neutral bright wreck pixels."""
    x, y = point
    half_w, half_h = 31, 24
    x1, x2 = max(0, x - half_w), min(image.shape[1], x + half_w + 1)
    y1, y2 = max(0, y - half_h), min(image.shape[0], y + half_h + 1)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = ((saturation <= 145) & (value >= 135)).astype(np.uint8) * 255
    # Remove the thin diamond border and isolated specular pixels.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    cx, cy = roi.shape[1] / 2.0, roi.shape[0] / 2.0
    candidates: list[tuple[float, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        bx = float(stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH] / 2)
        by = float(stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT] / 2)
        if area < 18 or area > 900:
            continue
        distance = float(np.hypot(bx - cx, by - cy))
        candidates.append((distance - min(area, 250) * 0.015, index))
    if not candidates:
        return None
    index = min(candidates)[1]
    # Preserve the full calibrated cell footprint.  Tight connected-component
    # crops contain too little water context and match many ordinary tiles.
    # The existing templates are approximately 51x39, so normalize every
    # generated exemplar to that same footprint.
    return cv2.resize(roi, (51, 39), interpolation=cv2.INTER_AREA)


def _signature(crop: np.ndarray) -> str:
    small = cv2.resize(crop, (24, 18), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Quantization makes screenshots from adjacent animation frames collapse
    # to one exemplar while retaining different wreck orientations.
    return hashlib.sha1((gray // 12).tobytes()).hexdigest()


def build(root: Path, output_dir: Path, max_templates: int = 24) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    crops: dict[str, np.ndarray] = {}
    # ``Path.iterdir`` can transiently return an empty iterator for a Chinese
    # directory when launched through PowerShell's legacy code page.  Glob
    # resolves the same directory through the Windows file API reliably.
    paths = sorted(root.glob("*.png")) + sorted(root.glob("*.jpg")) + sorted(root.glob("*.jpeg"))
    for index, path in enumerate(paths, start=1):
        print(f"[wreck-training] {index}/{len(paths)} {path.name}", file=sys.stderr, flush=True)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        image = _read_image(path)
        if image is None:
            samples.append({"file": path.name, "error": "invalid_image"})
            continue
        level = _recognize_level(image)
        points = _scaled_points(level, image) if level is not None else None
        if level is None or points is None:
            samples.append({"file": path.name, "error": "level_or_points_not_recognized"})
            continue
        cells = _positive_cells(image, points, LEVEL_GRID_SIZES[level])
        added = 0
        for row, col in sorted(cells):
            crop = _crop_wreck(image, points[row * LEVEL_GRID_SIZES[level] + col])
            if crop is None:
                continue
            signature = _signature(crop)
            if signature not in crops:
                crops[signature] = crop
                added += 1
        samples.append({"file": path.name, "level": level, "positive_cells": [list(c) for c in sorted(cells)], "new_crops": added})

    # Stable ordering makes regenerated repositories reproducible.
    selected = list(crops.items())[:max_templates]
    generated = []
    for index, (_signature_value, crop) in enumerate(selected, start=1):
        path = output_dir / f"visible_wreck_generated_{index:02d}.png"
        write_image_compat(path, crop)
        generated.append(path.name)
    report = {
        "root": str(root),
        "samples": samples,
        "unique_positive_crops": len(crops),
        "generated_templates": generated,
        "note": "Positive crops require agreement between static and partial detectors; review overlays before lowering thresholds.",
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "残骸识图")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "wreck_templates_review")
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "outputs" / "wreck_template_training.json")
    args = parser.parse_args()
    report = build(args.root, args.output)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
