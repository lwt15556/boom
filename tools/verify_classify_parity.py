"""Verify a classify_diamond_hit performance change is numerically identical.

Reads ``outputs/classify_baseline.json`` (captured before the change), re-runs
``classify_diamond_hit(image, image, point, config=DiamondHitConfig(search_radius=2))``
on the same boards/cells, and compares ``state``, ``score``, ``confidence`` and
``refined_center`` exactly.  Prints mismatches and total time.
"""

from __future__ import annotations

import json
import sys
import time
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
CONFIG = DiamondHitConfig(search_radius=2)


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
    baseline = json.loads((PROJECT_ROOT / "outputs" / "classify_baseline.json").read_text(encoding="utf-8"))
    cells = baseline["cells"]

    mismatches = 0
    compared = 0
    total = 0.0
    cache: dict[str, tuple] = {}
    for entry in cells:
        file_name = entry["file"]
        if file_name not in cache:
            path = ROOT / file_name
            image = read_image_compat(path, cv2.IMREAD_COLOR)
            level = _level(image)
            grid_size = LEVEL_GRID_SIZES.get(level, 10)
            points = read_saved_points(level, expected_n=grid_size)
            sx = image.shape[1] / 1280.0
            sy = image.shape[0] / 720.0
            cache[file_name] = (image, grid_size, [
                (int(round(x * sx)), int(round(y * sy))) for x, y in points
            ])
        image, grid_size, points = cache[file_name]
        row, col = entry["cell"]
        point = points[row * grid_size + col]

        start = time.perf_counter()
        res = classify_diamond_hit(image, image, point, config=CONFIG)
        total += time.perf_counter() - start

        compared += 1
        ok = (
            res.state == entry["state"]
            and abs(float(res.score) - float(entry["score"])) < 1e-6
            and abs(float(res.confidence) - float(entry["confidence"])) < 1e-6
            and int(res.refined_center[0]) == int(entry["refined"][0])
            and int(res.refined_center[1]) == int(entry["refined"][1])
        )
        if not ok:
            mismatches += 1
            print(f"[DIFF] {file_name} {entry['cell']}: "
                  f"new=({res.state},{res.score:.6f},{res.confidence:.6f},{res.refined_center}) "
                  f"old=({entry['state']},{entry['score']},{entry['confidence']},{entry['refined']})")

    print(f"\ncells compared: {compared}, mismatches: {mismatches}")
    print(f"classify_diamond_hit total: {total:.2f}s "
          f"({total / max(1, compared) * 1000:.1f} ms/cell)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
