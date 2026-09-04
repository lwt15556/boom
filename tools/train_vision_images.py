"""Calibrate and replay the OpenCV vision rules on local training screenshots.

The project uses deterministic OpenCV rules rather than a learned neural
network.  This tool turns screenshots in the local training folder into a
repeatable report, so new samples can be checked before changing thresholds.

Usage:
    .venv\\Scripts\\python.exe tools\\train_vision_images.py
    .venv\\Scripts\\python.exe tools\\train_vision_images.py --json outputs\\vision_training_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LEVEL_GRID_SIZES, SPECIAL_SUBMARINES, SUBMARINES
from save_points.points import read_saved_points
from utils.level_title_recognition import recognize_level_title
from utils.sidebar_progress import (
    detect_partial_wreck_cells,
    detect_sidebar_progress,
    resolve_completed_ship_cells,
    resolve_completed_ship_cells_by_anchors,
    resolution_has_unique_anchor_support,
)
from utils.wreck_detection import (
    VISIBLE_WRECK_TEMPLATES,
    PARTIAL_WRECK_TEMPLATES,
    detect_completed_submarine_candidate_cells,
    detect_red_submarine_marker_cells,
    detect_visible_wreck_cells,
)


def _default_root() -> Path | None:
    project_root = PROJECT_ROOT
    return next(
        (
            path
            for path in project_root.iterdir()
            if path.is_dir()
            and any(ord(character) > 127 for character in path.name)
            and (path / "before.png").exists()
        ),
        None,
    )


def _cells(value: set[tuple[int, int]] | None) -> list[list[int]]:
    return [list(cell) for cell in sorted(value or set())]


def _resolution_quality(
    placements: list[list[list[int]]],
    completed_lengths: list[int],
    grid_size: int,
) -> dict[str, Any]:
    """Validate the structural guarantees required before writing ship cells."""
    normalized = [
        tuple((int(cell[0]), int(cell[1])) for cell in placement)
        for placement in placements
    ]
    lengths_match = sorted(map(len, normalized), reverse=True) == sorted(
        (int(length) for length in completed_lengths),
        reverse=True,
    )
    straight = True
    contiguous = True
    in_bounds = True
    for placement in normalized:
        if not placement:
            straight = False
            continue
        in_bounds = in_bounds and all(
            0 <= row < grid_size and 0 <= col < grid_size
            for row, col in placement
        )
        rows = {row for row, _ in placement}
        cols = {col for _, col in placement}
        if len(rows) != 1 and len(cols) != 1:
            straight = False
            continue
        ordered = sorted(col for _, col in placement) if len(rows) == 1 else sorted(
            row for row, _ in placement
        )
        contiguous = contiguous and ordered == list(range(ordered[0], ordered[-1] + 1))

    non_adjacent = True
    for index, first in enumerate(normalized):
        for second in normalized[index + 1 :]:
            if any(
                max(abs(row_a - row_b), abs(col_a - col_b)) <= 1
                for row_a, col_a in first
                for row_b, col_b in second
            ):
                non_adjacent = False
                break
        if not non_adjacent:
            break

    return {
        "lengths_match_sidebar": lengths_match,
        "straight": straight,
        "contiguous": contiguous,
        "in_bounds": in_bounds,
        "non_adjacent": non_adjacent,
        "valid": all(
            (lengths_match, straight, contiguous, in_bounds, non_adjacent)
        ),
    }


def _recognize_level(image: Any, path: Path, fallback_level: int | None) -> tuple[int, dict[str, Any]]:
    # Clipboard captures are often downscaled to roughly 760x430. Their title
    # glyphs lose edge contrast after resampling, so use a lower score floor
    # only for these small images while retaining the normal threshold for
    # full-resolution runtime screenshots.
    image_width = int(getattr(image, "shape", (0, 0))[1]) if hasattr(image, "shape") else 0
    min_score = 0.60 if image_width and image_width < 1000 else 0.78
    title = recognize_level_title(
        image,
        reference_dir=PROJECT_ROOT / "save_points" / "imgs",
        min_score=min_score,
    )
    if title is not None and title.confident and 1 <= title.level <= 50:
        return title.level, {
            "level": title.level,
            "score": round(float(title.score), 4),
            "second_level": title.second_level,
            "second_score": round(float(title.second_score), 4),
            "confident": True,
        }
    level = int(fallback_level) if fallback_level is not None else 0
    return level, {
        "level": level,
        "score": round(float(title.score), 4) if title is not None else 0.0,
        "second_level": title.second_level if title is not None else None,
        "second_score": round(float(title.second_score), 4) if title is not None else 0.0,
        "confident": False,
    }


def _analyse(path: Path, fallback_level: int | None = None) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"file": path.name, "error": "invalid_image"}

    level, title = _recognize_level(image, path, fallback_level)
    if level <= 0:
        return {"file": path.name, "error": "level_not_recognized", "title": title}

    grid_size = LEVEL_GRID_SIZES.get(level, 10)
    points = read_saved_points(level, expected_n=grid_size)
    if not points:
        return {
            "file": path.name,
            "level": level,
            "title": title,
            "error": "missing_calibration",
        }
    # One supplied level-7 capture is a 754x422 scaled screenshot.  Reuse the
    # saved 1280x720 calibration after applying the same affine scale.
    scale_x = image.shape[1] / 1280.0
    scale_y = image.shape[0] / 720.0
    points = [
        (int(round(x * scale_x)), int(round(y * scale_y)))
        for x, y in points
    ]

    fleet = SUBMARINES.get(level, SPECIAL_SUBMARINES.get(level, ()))
    progress = detect_sidebar_progress(image, fleet)
    candidates = detect_completed_submarine_candidate_cells(image, points, grid_size)
    anchors = detect_red_submarine_marker_cells(image, points, grid_size)
    wrecks = detect_visible_wreck_cells(image, points, grid_size)
    partial = detect_partial_wreck_cells(
        image,
        points,
        grid_size=grid_size,
        template_paths=PARTIAL_WRECK_TEMPLATES,
    )

    resolution = None
    if progress and progress.completed_lengths and candidates and anchors:
        resolution = resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            progress.completed_lengths,
            grid_size=grid_size,
            preferred_cells=candidates,
            fallback_to_global=False,
        )
        if resolution.unresolved_lengths:
            broad_candidates = detect_completed_submarine_candidate_cells(
                image,
                points,
                grid_size,
                preserve_alternatives=True,
            )
            broad_resolution = resolve_completed_ship_cells_by_anchors(
                broad_candidates or candidates,
                anchors,
                progress.completed_lengths,
                grid_size=grid_size,
                preferred_cells=candidates,
                fallback_to_global=False,
                allow_ambiguous=True,
            )
            if (
                not broad_resolution.unresolved_lengths
                and resolution_has_unique_anchor_support(
                    broad_resolution.placements,
                    anchors,
                )
            ):
                resolution = broad_resolution

            global_resolution = resolve_completed_ship_cells(
                broad_candidates or candidates,
                progress.completed_lengths,
                grid_size=grid_size,
                preferred_cells=candidates,
            )
            if (
                resolution.unresolved_lengths
                and
                len(global_resolution.placements) > len(resolution.placements)
                and resolution_has_unique_anchor_support(
                    global_resolution.placements,
                    anchors,
                )
            ):
                resolution = global_resolution

    report: dict[str, Any] = {
        "file": path.name,
        "level": level,
        "title": title,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "sidebar": {
            "completed_lengths": list(progress.completed_lengths) if progress else [],
            "active_lengths": list(progress.active_lengths) if progress else [],
            "valid": bool(progress and progress.valid),
        },
        "visible_wreck_cells": _cells(wrecks),
        "partial_wreck_cells": _cells(partial),
        "completed_candidates": _cells(candidates),
        "red_anchor_cells": _cells(anchors),
    }
    if resolution is not None:
        resolved_cells = set(resolution.cells)
        placements = [
            [list(cell) for cell in placement]
            for placement in resolution.placements
        ]
        report["wreck_only_cells"] = _cells(set(wrecks) - resolved_cells)
        report["completed_resolution"] = {
            "placements": placements,
            "unresolved_lengths": list(resolution.unresolved_lengths),
            "discarded_cells": _cells(set(resolution.discarded_cells)),
        }
        report["quality"] = _resolution_quality(
            placements,
            list(progress.completed_lengths) if progress else [],
            grid_size,
        )
    else:
        report["wreck_only_cells"] = _cells(wrecks)
        report["quality"] = {
            "lengths_match_sidebar": False,
            "straight": False,
            "contiguous": False,
            "in_bounds": True,
            "non_adjacent": True,
            "valid": False,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="training screenshot folder")
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    root = (args.root or _default_root())
    if root is None or not root.exists():
        print(json.dumps({"error": "training_folder_not_found"}, ensure_ascii=False))
        return 2

    fallback_levels = {
        "before.png": 22,
        "after_1.png": 22,
        "debug_quit1_retry_1.png": 10,
    }
    samples = [
        _analyse(path, fallback_levels.get(path.name))
        for path in sorted(root.glob("*.png"))
    ]
    resolved_samples = [
        sample
        for sample in samples
        if sample.get("quality", {}).get("valid")
    ]
    quality_failures = [
        sample["file"]
        for sample in samples
        if sample.get("sidebar", {}).get("completed_lengths")
        and not sample.get("quality", {}).get("valid")
    ]
    report = {
        "root": str(root),
        "sample_count": len(samples),
        "summary": {
            "resolved_complete_fleet_samples": len(resolved_samples),
            "samples_with_completed_sidebar_evidence": sum(
                bool(sample.get("sidebar", {}).get("completed_lengths"))
                for sample in samples
            ),
            "samples_without_completed_sidebar_evidence": sum(
                not bool(sample.get("sidebar", {}).get("completed_lengths"))
                for sample in samples
            ),
            "quality_failures": quality_failures,
            "all_resolved_geometry_valid": not quality_failures,
        },
        "samples": samples,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
