"""按四边形裁剪截图，只保留棋盘、裁掉四角水面/周围 UI。

ROI 裁剪点位与 points.json 里的 quad 点位保持一致（同一套四角），
保证裁剪区域和网格点重合。必须显式指定 quad 来源，避免双数据源冲突。

用法示例::

    # 用 points.json 里第 17 关的固定 quad 裁剪（推荐，与网格点一致）
    python tools/crop_red_border.py --level 17 --input screen.png --output cropped.png

    # 显式给定四角（top,right,bottom,left）
    python tools/crop_red_border.py --quad 676,96,1100,355,686,718,262,352 \
        --input screen.png --output cropped.png

    # 明确要求自动检测红色菱形边框（可能与 points.json 固定 quad 不一致）
    python tools/crop_red_border.py --auto --input screen.png --output cropped.png

可选参数:
    --no-mask      只按外接矩形裁剪，四角保留原图（不置 0）
    --draw-quad    额外保存一张画了四角和名称的调试图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from utils.diamond_centers import (
    apply_quad_roi,
    detect_red_border_quad,
    draw_quad,
)
from utils.image_io import read_image_compat, write_image_compat


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="原始截图路径")
    parser.add_argument("--output", required=True, help="裁剪结果保存路径")
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="从 save_points/points.json 读取该关的固定 quad 作为 ROI 裁剪点位",
    )
    parser.add_argument(
        "--quad",
        default=None,
        help="显式四角，top,right,bottom,left 顺序：x1,y1,x2,y2,x3,y3,x4,y4",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动检测红色菱形边框作为 quad（注意：可能与 points.json 固定 quad 不一致）",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="只按外接矩形裁剪，四角保留原图而不置 0",
    )
    parser.add_argument(
        "--draw-quad",
        action="store_true",
        help="另存一张画了四角名称的调试图",
    )
    return parser.parse_args(argv)


def _read_quad_from_level(level: int) -> np.ndarray | None:
    """从 points.json 读取该关的固定 quad（[top,right,bottom,left]）。"""
    from save_points.points import read_saved_quad

    quad = read_saved_quad(level)
    if quad is None or quad.shape != (4, 2):
        return None
    return quad


def _parse_quad(text: str) -> np.ndarray:
    """解析 x1,y1,...,x4,y4 为 Nx2 quad（top,right,bottom,left）。"""
    parts = [int(p) for p in text.replace("，", ",").split(",")]
    if len(parts) != 8:
        raise ValueError("--quad 需要 8 个数字：x1,y1,x2,y2,x3,y3,x4,y4")
    return np.asarray(parts, dtype=np.float32).reshape(4, 2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    image = read_image_compat(args.input)
    if image is None:
        print(f"无法读取图片：{args.input}", file=sys.stderr)
        return 1

    # 三选一指定 quad 来源；不再静默自动检测，避免和 points.json 里的
    # 固定 quad 不一致（双数据源冲突）。
    if args.level is not None:
        quad = _read_quad_from_level(args.level)
        if quad is None:
            print(
                f"points.json 里没有第 {args.level} 关的 quad，裁剪中止。",
                file=sys.stderr,
            )
            return 1
        source = f"level {args.level} (points.json)"
    elif args.quad is not None:
        try:
            quad = _parse_quad(args.quad)
        except ValueError as exc:
            print(f"quad 参数错误：{exc}", file=sys.stderr)
            return 1
        source = "--quad"
    elif args.auto:
        quad = detect_red_border_quad(image)
        if quad is None:
            print("自动检测未找到红色菱形边框，裁剪中止。", file=sys.stderr)
            return 1
        source = "auto-detect"
    else:
        print(
            "必须指定 quad 来源：--level N（用 points.json 固定 quad）、"
            "--quad x1,y1,...,x4,y4（显式四角）或 --auto（自动检测红框）。",
            file=sys.stderr,
        )
        return 1

    names = ["top", "right", "bottom", "left"]
    print(f"quad 来源: {source}")
    for name, point in zip(names, np.round(quad).astype(int), strict=True):
        print(f"{name:8s} {point[0]:5d},{point[1]:5d}")

    cropped, offset_x, offset_y = apply_quad_roi(
        image,
        quad,
        mask=not args.no_mask,
    )

    if not write_image_compat(args.output, cropped):
        print(f"无法保存裁剪图：{args.output}", file=sys.stderr)
        return 1

    print(f"裁剪尺寸 {cropped.shape[1]}x{cropped.shape[0]}，偏移 ({offset_x},{offset_y})")
    print(f"已保存：{args.output}")

    if args.draw_quad:
        overlay = draw_quad(image, quad)
        debug_path = str(Path(args.output).with_name(Path(args.output).stem + "_quad.png"))
        if write_image_compat(debug_path, overlay):
            print(f"已保存：{debug_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
