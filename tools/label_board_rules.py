"""用规则识别（残骸 / 完整潜艇 / 海浪）给原图训练样本打干净标签。

标签直接来自 ``utils.submarine_detector.classify_cells_by_rules``：整帧灰白连通域
+ 格子菱形归属 + 红色指挥塔判定，逐帧实测 100% 准确（见 ``_debug`` 审计图）。
写入 ``识图/训练样本/<stem>.json`` 的新键 ``labels_rules_3class``（非破坏性，
不覆盖旧标签）：

* ``0`` = 水面 / 海浪
* ``1`` = 完整潜艇（艇身 + 尾迹所在的格子）
* ``2`` = 残骸

用法::

    .venv\\Scripts\\python.exe tools\\label_board_rules.py [--dir 识图/原图训练样本]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from save_points.points import points_from_quad  # noqa: E402
from utils.image_io import read_image_compat  # noqa: E402
from utils.submarine_detector import classify_cells_by_rules  # noqa: E402

DEFAULT_DIR = PROJECT_ROOT / "识图" / "原图训练样本"
LABEL_DIR = PROJECT_ROOT / "识图" / "训练样本"
LABEL_KEY = "labels_rules_3class"


def label_frame(image_path: Path, record: dict) -> tuple[list[list[int]], dict] | None:
    quad = record.get("quad")
    grid_size = record.get("grid_size")
    if not quad or not grid_size:
        return None
    image = read_image_compat(image_path)
    if image is None:
        return None
    n = int(grid_size)
    points = points_from_quad([tuple(map(float, corner)) for corner in quad], n)
    ships, wrecks = classify_cells_by_rules(image, points, n)
    grid = [[0] * n for _ in range(n)]
    for row, col in ships:
        if 0 <= row < n and 0 <= col < n:
            grid[row][col] = 1
    for row, col in wrecks:
        if 0 <= row < n and 0 <= col < n:
            grid[row][col] = 2
    summary = {"ships": len(ships), "wrecks": len(wrecks), "ship_cells": sorted(ships), "wreck_cells": sorted(wrecks)}
    return grid, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="原图目录")
    parser.add_argument("--label-dir", default=str(LABEL_DIR), help="存放 json 的目录")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args(argv)

    image_dir = Path(args.dir)
    label_dir = Path(args.label_dir)
    stats: Counter = Counter()
    written = skipped = 0
    for image_path in sorted(image_dir.glob("*.png")):
        json_path = label_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            skipped += 1
            continue
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped += 1
            continue
        result = label_frame(image_path, record)
        if result is None:
            skipped += 1
            continue
        grid, summary = result
        for row in grid:
            stats.update(row)
        if not args.dry_run:
            record[LABEL_KEY] = grid
            record["rule_sprite_summary"] = summary
            json_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        written += 1

    print(f"打标签帧数: {written}（跳过 {skipped}）")
    print(f"类别分布: 水/海浪 {stats[0]}  潜艇 {stats[1]}  残骸 {stats[2]}")
    total = sum(stats.values()) or 1
    print(f"占比: 水 {stats[0]/total:.1%}  潜艇 {stats[1]/total:.2%}  残骸 {stats[2]/total:.2%}")
    if not args.dry_run:
        print(f"已写入键 {LABEL_KEY} -> {label_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
