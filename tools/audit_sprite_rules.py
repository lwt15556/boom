"""用规则识别跑一张棋盘截图，打印潜艇格 / 残骸格并落一张标注图。

用法::

    .venv\\Scripts\\python.exe tools\\audit_sprite_rules.py <截图.png> [--level N]
    .venv\\Scripts\\python.exe tools\\audit_sprite_rules.py <截图.png> --quad "676,96 1100,355 686,718, 262,352"

不传 ``--level`` 时从文件名里解析 ``level_<N>_...``；不传 ``--quad`` 时读
``save_points`` 里该关卡的四角。输出：

* 每个灰白连通块（面积 / 填充率 / 纹理 / 是否带红塔 / 覆盖了哪些格）；
* ``潜艇格`` 与 ``残骸格`` 两组坐标；
* ``<截图>_rules.png`` 标注图（黄圈 = 潜艇，红圈 = 残骸）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from save_points.points import (  # noqa: E402
    points_from_quad,
    read_level_entry,
    read_saved_quad,
)
from utils.image_io import read_image_compat, write_image_compat  # noqa: E402
from utils.submarine_detector import (  # noqa: E402
    classify_cells_by_rules,
    detect_board_sprites,
)


def _parse_quad(text: str) -> list[tuple[float, float]]:
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(numbers) != 8:
        raise argparse.ArgumentTypeError("quad 需要 8 个数字：上,右,下,左 四个角")
    return [(numbers[i], numbers[i + 1]) for i in range(0, 8, 2)]


def _level_from_name(name: str) -> int | None:
    match = re.search(r"level[_-](\d+)", name)
    return int(match.group(1)) if match else None


def _sample_metadata(image_path: Path) -> dict:
    """读同名的训练样本 JSON（含 quad / grid_size），找不到就返回空。"""
    candidates = [
        image_path.with_suffix(".json"),
        PROJECT_ROOT / "识图" / "训练样本" / f"{image_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("quad"):
                return data
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="规则识别（潜艇 / 残骸）逐格审计")
    parser.add_argument("image", type=Path, help="棋盘截图")
    parser.add_argument("--level", type=int, default=None, help="关卡号（用于读取已保存的四角）")
    parser.add_argument("--quad", type=str, default=None, help="棋盘大菱形四角，格式 'x1,y1 x2,y2 x3,y3 x4,y4'")
    parser.add_argument("--grid", type=int, default=None, help="网格边长，默认取已保存值")
    parser.add_argument("--output", type=Path, default=None, help="标注图输出路径")
    args = parser.parse_args(argv)

    image = read_image_compat(args.image)
    if image is None:
        print(f"读不到图片：{args.image}", file=sys.stderr)
        return 2

    level = args.level if args.level is not None else _level_from_name(args.image.stem)
    entry = read_level_entry(level) if level is not None else None
    if not isinstance(entry, dict):
        entry = {}
    sample = _sample_metadata(args.image)

    quad = _parse_quad(args.quad) if args.quad else None
    if quad is None and sample.get("quad"):
        quad = [tuple(map(float, corner)) for corner in sample["quad"]]
    if quad is None:
        if level is None:
            print("没有 --quad 时必须能从文件名解析出关卡号，或显式传 --level", file=sys.stderr)
            return 2
        saved = read_saved_quad(level)
        if saved is None:
            print(f"save_points 里没有第 {level} 关的四角，请用 --quad 指定", file=sys.stderr)
            return 2
        quad = [tuple(map(float, corner)) for corner in saved]

    grid_size = args.grid
    if grid_size is None:
        for source in (sample, entry):
            if source.get("grid_size"):
                grid_size = int(source["grid_size"])
                break
    if grid_size is None:
        print("拿不到网格边长，请显式传 --grid", file=sys.stderr)
        return 2

    points = points_from_quad(quad, grid_size)
    sprites = detect_board_sprites(image, points, grid_size)
    ships, wrecks = classify_cells_by_rules(image, points, grid_size)

    print(f"图片: {args.image}")
    print(f"关卡: {level}  网格: {grid_size}x{grid_size}  四角: {quad}")
    print(f"灰白连通块: {len(sprites)} 个")
    for sprite in sorted(sprites, key=lambda item: -item.area):
        cells = ", ".join(f"({row},{col}) {coverage:.2f}" for (row, col), coverage in sprite.cells)
        print(
            f"  块{sprite.label:<3} 面积{sprite.area:<5} 填充{sprite.fill_ratio:.2f} "
            f"纹理{sprite.texture:5.1f} 红塔={'是' if sprite.has_red_tower else '否'}  "
            f"格: {cells}"
        )
    print(f"\n潜艇格 {len(ships)}: {sorted(ships)}")
    print(f"残骸格 {len(wrecks)}: {sorted(wrecks)}")

    overlay = image.copy()
    for row, col in sorted(ships):
        x, y = points[row * grid_size + col]
        cv2.circle(overlay, (int(x), int(y)), 13, (0, 255, 255), 3)
    for row, col in sorted(wrecks):
        x, y = points[row * grid_size + col]
        cv2.circle(overlay, (int(x), int(y)), 13, (0, 0, 255), 3)
    out_path = args.output or (
        PROJECT_ROOT / "_debug" / "audit_rules" / f"{args.image.stem}_rules.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_image_compat(out_path, overlay)
    print(f"标注图: {out_path}（黄圈=潜艇，红圈=残骸）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
