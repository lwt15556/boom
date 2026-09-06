"""Measure the dynamic before/after hit classifier on recorded probe samples.

Each ``_debug/screenshots/probes`` directory stores a ``before.png``, one or
more ``after_*.png`` frames and a ``result.json`` carrying the runtime
``decision`` for the single clicked cell.  This tool re-runs ``classify_diamond_hit``
on every available after frame and aligns the per-frame features with the
runtime decision, so we can see how strongly the "hit" verdict is backed by a
compact, centre-bright grey wrecks marker.

The key question the dynamic path must answer: a hit shows a *persistent grey
wreck marker* in the clicked cell, while a miss leaves water (whose bright spots
change but stay broad/reflective).  We report the per-frame ``center_gray_ratio``,
``component_ratio``, ``s_drop`` and ``changed_ratio`` distributions separately for
runtime decisions of ``hit`` and ``miss``.  A ``hit`` with a very low
``center_gray_ratio`` is a candidate false positive (shell spent on water).

This tool never changes thresholds; it only measures.  Review-only: result.json
values are trusted as the runtime verdict, not as ground truth.

Usage:
    .venv\\Scripts\\python.exe tools\\evaluate_dynamic_hits.py --json outputs\\dynamic_hit_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.diamond_hit import DiamondHitConfig, classify_diamond_hit
from utils.image_io import read_image_compat


def _percentiles(values, points=(0.05, 0.5, 0.95)):
    if not values:
        return [round(0.0, 4)] * len(points)
    arr = np.asarray(values, dtype=float)
    return [round(float(np.percentile(arr, p * 100.0)), 4) for p in points]


def _aggregate(rows):
    by_decision = defaultdict(list)
    for row in rows:
        by_decision[row["decision"]].append(row)
    features = ("score", "center_gray_ratio", "component_ratio", "s_drop", "changed_ratio")
    out = {}
    for decision, record_list in by_decision.items():
        bucket = {"frame_count": len(record_list)}
        for feature in features:
            bucket[feature] = _percentiles([float(r[feature]) for r in record_list])
        # How many hit verdicts have a weak centre-grey marker?
        weak_centre = sum(
            1 for r in record_list
            if r["decision"] == "hit" and r["center_gray_ratio"] < 0.20
        )
        bucket["hits_with_weak_centre"] = weak_centre
        out[decision] = bucket
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("_debug/screenshots/probes"))
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    rows = []
    for directory in sorted(args.root.glob("*")):
        if not directory.is_dir():
            continue
        result_path = directory / "result.json"
        before_path = directory / "before.png"
        if not result_path.exists() or not before_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        decision = str(result.get("decision", "unknown")).lower()
        point = result.get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        before = read_image_compat(before_path, cv2.IMREAD_COLOR)
        if before is None:
            continue
        point = (int(point[0]), int(point[1]))
        after_paths = sorted(directory.glob("after_*.png"))
        if not after_paths:
            continue
        for after_path in after_paths[:4]:
            after = read_image_compat(after_path, cv2.IMREAD_COLOR)
            if after is None or after.shape[:2] != before.shape[:2]:
                continue
            try:
                hit = classify_diamond_hit(before, after, point, config=DiamondHitConfig())
            except Exception as exc:  # noqa: BLE001 - diagnostics must not stop the sweep
                continue
            rows.append({
                "sample": directory.name,
                "decision": decision,
                "frame": after_path.stem,
                "state": hit.state,
                "score": round(float(hit.score), 4),
                "center_gray_ratio": round(float(hit.center_gray_ratio), 4),
                "component_ratio": round(float(hit.component_ratio), 4),
                "s_drop": round(float(hit.s_drop), 4),
                "changed_ratio": round(float(hit.changed_ratio), 4),
            })

    report = {
        "root": str(args.root),
        "sample_count": len({r["sample"] for r in rows}),
        "frame_count": len(rows),
        "by_decision": _aggregate(rows),
        "frames": rows,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
