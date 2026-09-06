"""Verify a performance refactor changed no recognition output on the baseline sample.

Reads ``outputs/perf_baseline.json`` (captured before the refactor), re-runs
``detect_visible_wreck_cells`` on the same boards with the current code, and
asserts the recognised visible-wreck cell sets are identical.  Prints total
elapsed time for the refactored run.
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
from utils.image_io import read_image_compat
from utils.wreck_detection import detect_visible_wreck_cells

ROOT = PROJECT_ROOT / "识图" / "训练识图"


def main() -> int:
    baseline_path = PROJECT_ROOT / "outputs" / "perf_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    boards = baseline["boards"]

    total_elapsed = 0.0
    mismatches = 0
    seen = 0
    for entry in boards:
        path = ROOT / entry["file"]
        if not path.exists():
            print(f"[skip] missing {entry['file']}", file=sys.stderr)
            continue
        image = read_image_compat(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        level = int(entry["level"])
        grid_size = LEVEL_GRID_SIZES.get(level, 10)
        points = read_saved_points(level, expected_n=grid_size)
        if not points:
            continue
        sx = image.shape[1] / 1280.0
        sy = image.shape[0] / 720.0
        points = [(int(round(x * sx)), int(round(y * sy))) for x, y in points]

        start = time.perf_counter()
        visible = detect_visible_wreck_cells(image, points, grid_size)
        total_elapsed += time.perf_counter() - start

        expected = {tuple(c) for c in entry["visible"]}
        actual = {tuple(int(v) for v in c) for c in visible}
        seen += 1
        if expected != actual:
            mismatches += 1
            print(f"[DIFF] {entry['file']}: expected={sorted(expected)} actual={sorted(actual)}")

    print(f"\nboards compared: {seen}, mismatches: {mismatches}")
    print(f"refactored detect_visible_wreck_cells total: {total_elapsed:.2f}s "
          f"({total_elapsed / max(1, seen) * 1000:.0f} ms/board)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
