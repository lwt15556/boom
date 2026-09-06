"""Diagnose false positives in the static vision recovery path.

The runtime static detector (``detect_visible_wreck_cells``) decides whether a
cell already shows a wreck on a single frame.  Complete-board screenshots have
no per-cell ground truth, so this tool builds best-available weak labels from
the only independent evidence the board exposes:

- ``anchor``: a red completion marker is attached to this cell (the game shows
  a surfaced, completed submarine).
- ``complete``: the cell is part of a straight placement resolved to match the
  sidebar completed lengths.
- ``partial``: a partial-wreck template matched here.
- ``wreck_only``: the static detector flagged the cell but it is not part of a
  completed ship, so it is either a real hit-not-yet-complete wreck or a false
  positive.
- ``other``: every remaining cell (mostly water, but may contain legitimate
  hit-not-complete wrecks that no template matched).

The report shows, per label, the distribution of the shape/colour features the
detector keys on.  The goal is to see how well ``center_gray_ratio``,
``gray_excess``, ``component_ratio``, ``cyan_ratio`` and the aggregate
``wreck_shape_metrics().score`` separate ``complete``/``anchor`` from
``other``, which is where the false-positive risk lives.  No threshold is ever
changed by this tool; it only measures.

Usage:
    .venv\\Scripts\\python.exe tools\\analyse_vision_false_positives.py --json outputs\\vision_fp_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LEVEL_GRID_SIZES, SUBMARINES
from save_points.points import read_saved_points
from utils.image_io import read_image_compat
from utils.level_title_recognition import recognize_level_title
from utils.sidebar_progress import (
    detect_partial_wreck_cells,
    detect_sidebar_progress,
    resolve_completed_ship_cells_by_anchors,
)
from utils.wreck_detection import (
    PARTIAL_WRECK_TEMPLATES,
    detect_completed_submarine_candidate_cells,
    detect_red_submarine_marker_cells,
    detect_visible_wreck_cells,
    grid_cell_polygon,
)
from utils.wreck_detection import wreck_shape_metrics as _full_frame_wreck_shape_metrics


def _fast_wreck_metrics(image, point, cell_polygon):
    """Evaluate wreck_shape_metrics on a local crop.

    The runtime helper converts the whole frame to HSV for every cell, which
    makes board-wide diagnostics ~50x too slow.  The diamond-ratio features are
    computed inside a polygon-derived mask, so cropping to the polygon bounding
    box and translating the polygon/local point preserves the ratios.  This is
    diagnostic only; runtime behaviour is unchanged.
    """
    if cell_polygon is None:
        return _full_frame_wreck_shape_metrics(image, point)
    polygon = np.asarray(cell_polygon, dtype=np.float32)
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    margin = 6
    x1 = max(0, int(np.floor(xs.min())) - margin)
    y1 = max(0, int(np.floor(ys.min())) - margin)
    x2 = min(image.shape[1], int(np.ceil(xs.max())) + margin)
    y2 = min(image.shape[0], int(np.ceil(ys.max())) + margin)
    if x2 <= x1 or y2 <= y1:
        return _full_frame_wreck_shape_metrics(image, point)
    crop = image[y1:y2, x1:x2]
    local_polygon = polygon.copy()
    local_polygon[:, 0] -= x1
    local_polygon[:, 1] -= y1
    local_point = (float(point[0]) - x1, float(point[1]) - y1)
    # Titles only overlap the very top cells, which the runtime already leaves
    # unknown; for the diagnostic we keep the same crop so the median baseline
    # of the board is unaffected by a couple of top-row cells.
    return _full_frame_wreck_shape_metrics(crop, local_point, cell_polygon=local_polygon)


def _default_root() -> Path | None:
    return next(
        (
            path
            for path in PROJECT_ROOT.iterdir()
            if path.is_dir()
            and any(ord(ch) > 127 for ch in path.name)
            and (path / "before.png").exists()
        ),
        None,
    )


def _recognize_level(image: Any, fallback_level: int | None) -> int:
    image_width = int(getattr(image, "shape", (0, 0))[1]) if hasattr(image, "shape") else 0
    min_score = 0.60 if image_width and image_width < 1000 else 0.78
    title = recognize_level_title(
        image,
        reference_dir=PROJECT_ROOT / "save_points" / "imgs",
        min_score=min_score,
    )
    if title is not None and title.confident and 1 <= title.level <= 50:
        return title.level
    return int(fallback_level) if fallback_level is not None else 0


def _percentiles(values: list[float], points: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    return [round(float(np.percentile(arr, p * 100.0)), 4) for p in points]


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    features = [
        "score",
        "center_gray_ratio",
        "gray_excess",
        "component_ratio",
        "compactness",
        "cyan_ratio",
        "bright_ratio",
    ]
    by_label: dict[str, dict[str, float | list[float] | int]] = {}
    for label in ("anchor", "complete", "partial", "wreck_only", "other"):
        cells = [s for s in samples if s["label"] == label]
        bucket: dict[str, Any] = {"count": len(cells)}
        for feature in features:
            bucket[feature] = _percentiles([float(s[feature]) for s in cells])
        by_label[label] = bucket

    other = [s for s in samples if s["label"] == "other"]
    anchor_or_complete = [s for s in samples if s["label"] in ("anchor", "complete")]
    # ``wreck_only`` cells are flagged by the static detector but are not part
    # of any completed ship, so they are the candidates for false positives
    # (water or hit-not-yet-complete wrecks promoted to a static hit).
    wreck_only = [s for s in samples if s["label"] == "wreck_only"]
    summary = {
        "sample_count": len(samples),
        "other_count": len(other),
        "wreck_only_count": len(wreck_only),
        "wreck_only_rate": round(len(wreck_only) / len(samples), 4) if samples else 0.0,
        "anchor_or_complete_count": len(anchor_or_complete),
    }
    return {"summary": summary, "by_label": by_label, "samples": samples}


def _analyse(path: Path, fallback_level: int | None = None) -> dict[str, Any] | None:
    image = read_image_compat(path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    level = _recognize_level(image, fallback_level)
    if level <= 0:
        return None
    grid_size = LEVEL_GRID_SIZES.get(level, 10)
    points = read_saved_points(level, expected_n=grid_size)
    if not points:
        return None
    scale_x = image.shape[1] / 1280.0
    scale_y = image.shape[0] / 720.0
    points = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in points]

    fleet = SUBMARINES.get(level, ())
    progress = detect_sidebar_progress(image, fleet)
    candidates = detect_completed_submarine_candidate_cells(image, points, grid_size)
    anchors = detect_red_submarine_marker_cells(image, points, grid_size)
    wrecks = detect_visible_wreck_cells(image, points, grid_size)
    partial = detect_partial_wreck_cells(
        image, points, grid_size=grid_size, template_paths=PARTIAL_WRECK_TEMPLATES
    )

    completed: set[tuple[int, int]] = set()
    if progress and progress.completed_lengths and candidates and anchors:
        resolution = resolve_completed_ship_cells_by_anchors(
            candidates, anchors, progress.completed_lengths, grid_size=grid_size,
            preferred_cells=candidates, fallback_to_global=False,
        )
        completed = set(resolution.cells)

    cells: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        row, col = divmod(index, grid_size)
        cell = (row, col)
        polygon = grid_cell_polygon(points, index, grid_size)
        metrics = _fast_wreck_metrics(image, point, polygon)
        if cell in anchors:
            label = "anchor"
        elif cell in completed:
            label = "complete"
        elif partial and cell in partial:
            label = "partial"
        elif cell in wrecks:
            label = "wreck_only"
        else:
            label = "other"
        cells.append({
            "cell": list(cell),
            "label": label,
            "score": round(float(metrics.score), 4),
            "center_gray_ratio": round(float(metrics.center_gray_ratio), 4),
            "gray_excess": round(float(metrics.gray_excess), 4),
            "component_ratio": round(float(metrics.component_ratio), 4),
            "compactness": round(float(metrics.compactness), 4),
            "cyan_ratio": round(float(metrics.cyan_ratio), 4),
            "bright_ratio": round(float(metrics.bright_ratio), 4),
        })
    return {"file": path.name, "level": level, "grid_size": grid_size, "cells": cells}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    root = args.root or _default_root()
    if root is None or not root.exists():
        print(json.dumps({"error": "training_folder_not_found"}, ensure_ascii=False))
        return 2

    fallback_levels = {"before.png": 22, "after_1.png": 22, "debug_quit1_retry_1.png": 10}
    all_cells: list[dict[str, Any]] = []
    per_board: list[dict[str, Any]] = []
    paths = sorted(root.glob("*.png"))
    for index, path in enumerate(paths, start=1):
        print(f"[fp-analysis] {index}/{len(paths)} {path.name}", file=sys.stderr, flush=True)
        result = _analyse(path, fallback_levels.get(path.name))
        if result is None:
            continue
        per_board.append({k: v for k, v in result.items() if k != "cells"})
        all_cells.extend(result["cells"])

    report = {
        "root": str(root),
        "board_count": len(per_board),
        "boards": per_board,
        **_aggregate(all_cells),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
