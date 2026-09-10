"""Snapshot per-cell ``classify_diamond_hit`` output for the static-fallback case.

The static recovery path calls ``classify_diamond_hit(image, image, point)``
(before == after).  This tool records every cell's ``state``, ``score``,
``confidence`` and ``refined_center`` for a few boards so a performance change
can be verified to be numerically identical.  Produces
``outputs/classify_baseline.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LEVEL_GRID_SIZES
from save_points.points import read_saved_points
from utils.diamond_hit import DiamondHitConfig, classify_diamond_hit
from utils.image_io import read_image_compat
from utils.level_title_recognition import recognize_level_title

ROOT = PROJECT_ROOT / "识图" / "训练识图"


def _level(image) -> int:
    image_width = int(getattr(image, "shape", (0, 0))[1]) if hasattr(image, "shape") else 0
    min_score = 0.60 if image_width and image_width < 1000 else 0.78
    title = recognize_level_title(
        image, reference_dir=PROJECT_ROOT / "save_points" / "imgs", min_score=min_score
    )
    if title is not None and title.confident and 1 <= title.level <= 50:
        return title.level
    return 0


def main() -> int:
    rows = []
    for path in sorted(ROOT.glob("*.png"))[:6]:
        image = read_image_compat(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        level = _level(image)
        if level <= 0:
            continue
        grid_size = LEVEL_GRID_SIZES.get(level, 10)
        points = read_saved_points(level, expected_n=grid_size)
        if not points:
            continue
        sx = image.shape[1] / 1280.0
        sy = image.shape[0] / 720.0
        points = [(int(round(x * sx)), int(round(y * sy))) for x, y in points]
        for index, point in enumerate(points):
            row, col = divmod(index, grid_size)
            result = classify_diamond_hit(
                image, image, point, config=DiamondHitConfig(search_radius=2)
            )
            rows.append({
                "file": path.name,
                "cell": [row, col],
                "state": result.state,
                "score": round(float(result.score), 6),
                "confidence": round(float(result.confidence), 6),
                "refined": [int(result.refined_center[0]), int(result.refined_center[1])],
            })
            print(f"[classify-baseline] {path.name} ({row},{col}) {result.state}", file=sys.stderr)
    (PROJECT_ROOT / "outputs" / "classify_baseline.json").write_text(
        json.dumps({"cells": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
