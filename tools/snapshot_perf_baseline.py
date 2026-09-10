"""Snapshot ``detect_visible_wreck_cells`` output before a performance change.

Run this once to produce a reference of recognised visible-wreck cell sets on a
sample of training boards, then re-run the comparison tool after the change.
Produces ``outputs/perf_baseline.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LEVEL_GRID_SIZES, MAX_LEVEL
from save_points.points import read_saved_points
from utils.image_io import read_image_compat
from utils.level_title_recognition import recognize_level_title


def _level(image) -> int:
    image_width = int(getattr(image, "shape", (0, 0))[1]) if hasattr(image, "shape") else 0
    min_score = 0.60 if image_width and image_width < 1000 else 0.78
    title = recognize_level_title(
        image, reference_dir=PROJECT_ROOT / "save_points" / "imgs", min_score=min_score
    )
    if title is not None and title.confident and 1 <= title.level <= MAX_LEVEL:
        return title.level
    return 0


def main() -> int:
    root = PROJECT_ROOT / "识图" / "训练识图"
    from utils.wreck_detection import detect_visible_wreck_cells

    output = []
    for path in sorted(root.glob("*.png"))[:20]:
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
        visible = detect_visible_wreck_cells(image, points, grid_size)
        output.append({"file": path.name, "level": level, "visible": sorted(list(visible))})
        print(f"[perf-baseline] {path.name} -> {len(visible)}", file=sys.stderr)
    json.dump(
        {"boards": output},
        (PROJECT_ROOT / "outputs" / "perf_baseline.json").open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
