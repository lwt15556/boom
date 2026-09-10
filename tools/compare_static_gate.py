"""Compare ``detect_visible_wreck_cells`` output before/after the centre-grey gate.

The runtime helper ``visible_wreck_static_detected`` reads the module-level
``STATIC_WRECK_MIN_CENTER_GRAY_RATIO``.  Setting it to an extreme low value
reproduces the pre-change behaviour (shape score only); setting it to the real
0.35 exercises the new gate.  The same boards and calibration are reused for
both, so the only difference is the gate.

For every board we report how many cells the new gate removes, how many of
those removals fall on red-anchor/completed-ship cells (true-positive loss,
which should be ~zero), and the resulting visible-wreck count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.wreck_detection as wd
from config import LEVEL_GRID_SIZES, MAX_LEVEL, SUBMARINES
from save_points.points import read_saved_points
from utils.image_io import read_image_compat
from utils.level_title_recognition import recognize_level_title
from utils.sidebar_progress import resolve_completed_ship_cells_by_anchors


def _recognize_level(image) -> int:
    image_width = int(getattr(image, "shape", (0, 0))[1]) if hasattr(image, "shape") else 0
    min_score = 0.60 if image_width and image_width < 1000 else 0.78
    title = recognize_level_title(
        image,
        reference_dir=PROJECT_ROOT / "save_points" / "imgs",
        min_score=min_score,
    )
    if title is not None and title.confident and 1 <= title.level <= MAX_LEVEL:
        return title.level
    return 0


def _candidates(image, level) -> tuple[int, list[tuple[int, int]]]:
    grid_size = LEVEL_GRID_SIZES.get(level, 10)
    points = read_saved_points(level, expected_n=grid_size)
    if not points:
        return grid_size, []
    sx = image.shape[1] / 1280.0
    sy = image.shape[0] / 720.0
    return grid_size, [(int(round(x * sx)), int(round(y * sy))) for x, y in points]


def _board(path: Path, threshold: float) -> dict:
    wd.STATIC_WRECK_MIN_CENTER_GRAY_RATIO = threshold
    image = read_image_compat(path, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "unreadable"}
    level = _recognize_level(image)
    if level <= 0:
        return {"error": "no_level"}
    grid_size, points = _candidates(image, level)
    if not points:
        return {"error": "no_points"}
    fleet = SUBMARINES.get(level, ())
    visible = wd.detect_visible_wreck_cells(image, points, grid_size)
    anchors = wd.detect_red_submarine_marker_cells(image, points, grid_size)
    candidates = wd.detect_completed_submarine_candidate_cells(image, points, grid_size)
    completed = set()
    if candidates and anchors:
        try:
            from utils.sidebar_progress import detect_sidebar_progress
            progress = detect_sidebar_progress(image, fleet)
            if progress and progress.completed_lengths:
                resolution = resolve_completed_ship_cells_by_anchors(
                    candidates, anchors, progress.completed_lengths, grid_size=grid_size,
                    preferred_cells=candidates, fallback_to_global=False,
                )
                completed = set(resolution.cells)
        except Exception:
            completed = set()
    return {
        "visible": sorted((list(c) for c in visible)),
        "anchors": sorted((list(c) for c in anchors)),
        "completed": sorted((list(c) for c in completed)),
    }


def main() -> int:
    root = PROJECT_ROOT / "识图" / "训练识图"
    paths = sorted(root.glob("*.png"))[:20]
    old_threshold = -1e9
    new_threshold = 0.35
    rows = []
    totals = {"old_visible": 0, "new_visible": 0, "lost_on_true_positive": 0, "boards": 0}
    for path in paths:
        old = _board(path, old_threshold)
        new = _board(path, new_threshold)
        if "error" in old or "error" in new:
            rows.append({"file": path.name, "error": old.get("error") or new.get("error")})
            continue
        old_set = {tuple(c) for c in old["visible"]}
        new_set = {tuple(c) for c in new["visible"]}
        removed = old_set - new_set
        true_positive_mask = {
            tuple(c) for c in old["anchors"] + old["completed"]
        }
        lost_true = sorted(list(removed & true_positive_mask))
        totals["old_visible"] += len(old_set)
        totals["new_visible"] += len(new_set)
        totals["lost_on_true_positive"] += len(lost_true)
        totals["boards"] += 1
        rows.append({
            "file": path.name,
            "level": new.get("level"),
            "old_visible": len(old_set),
            "new_visible": len(new_set),
            "removed": len(removed),
            "lost_true_positive": lost_true,
        })
    print(json.dumps({"totals": totals, "boards": rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
