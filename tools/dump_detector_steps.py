from __future__ import annotations

"""Dump every submarine/debris detector stage to PNG for step-by-step debugging.

Usage::

    # 单张截图
    .venv\\Scripts\\python.exe tools\\dump_detector_steps.py shot.png
    # 整个目录（每个文件一个子目录）
    .venv\\Scripts\\python.exe tools\\dump_detector_steps.py _debug\\screenshots\\probes
    # 指定输出根目录
    .venv\\Scripts\\python.exe tools\\dump_detector_steps.py shot.png --out _debug\\steps
    # 带关卡：额外输出"框 -> 棋盘格子"映射图（需要 save_points/points.json 里有该关）
    .venv\\Scripts\\python.exe tools\\dump_detector_steps.py shot.png --level 9

每个输入图片输出到 ``<out>/<图片名>/``：

    step01_input.png        输入帧
    step02_roi.png          ROI 裁剪（后续坐标都在这个裁剪系里）
    step03_roi_quad.png     全帧上画出棋盘四边形 ROI（角点见 ROI_QUAD）
    step04_roi_mask.png     四边形掩码后的 ROI（涂黑区域不参与检测）
    step05_sauvola.png      Sauvola 局部二值化
    step06_blue_mask.png    HSV 蓝色掩码
    step07_and.png          二值 AND 蓝掩码（海水前景）
    step08_morph_open.png   开运算
    step09_area_filtered.png 面积过滤后的前景（碎片检测输入）
    step10_red_mask.png     红色掩码（红船旗）
    step11_dilated.png      膨胀（船聚类输入）
    step12_centers.png      轮廓质心 + 序号
    step13_clusters.png     聚类判定（红=船 青=网格丢弃 黄=未归类 品红=超大簇）
    step14_erased_ships.png 擦除船体后的掩码
    step15_debris.png       碎片候选（绿=通过 灰=被拒）
    step16_final.png        最终结果叠加
    step17_cells.png        框 -> 格子映射（仅 --level 时）
    report.txt              每一步的像素数、每个簇/每个碎片的判定数值

每一步的阈值都写在 ``report.txt`` 里，改参数前先看这里哪一步开始不对。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LEVEL_GRID_SIZES  # noqa: E402
from save_points.points import read_saved_points, read_saved_quad  # noqa: E402
from utils import submarine_detector as detector  # noqa: E402
from utils.image_io import read_image_compat  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "_debug" / "detector_steps"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def iter_inputs(path: Path) -> list[Path]:
    """返回待处理的图片列表（目录则取一层内的图片）。"""
    if path.is_dir():
        return sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_EXTS
        )
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return []


def _scale_quad(quad, frame_shape, ref=(1280, 720)):
    """把参考系(1280x720)里的四角缩放到当前帧分辨率。"""
    if quad is None:
        return None
    h, w = frame_shape[:2]
    if not w or not h:
        return None
    sx, sy = w / float(ref[0]), h / float(ref[1])
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in quad]


def dump_one(image_path: Path, out_root: Path, level: int | None) -> int:
    """处理一张图，返回写出的文件数量（读图失败返回 0）。"""
    image = read_image_compat(image_path)
    if image is None:
        print(f"  [跳过] 读不到图片: {image_path}")
        return 0

    out_dir = out_root / image_path.stem
    points = None
    grid_size = None
    quad = None
    if level is not None:
        grid_size = LEVEL_GRID_SIZES.get(int(level))
        if grid_size is None:
            print(f"  [警告] 关卡 {level} 没有配置棋盘大小，只输出检测步骤")
        else:
            points = read_saved_points(int(level), expected_n=grid_size)
            quad = _scale_quad(read_saved_quad(int(level)), image.shape)
            if points is None or quad is None:
                print(f"  [警告] save_points 里关卡 {level} 的 {grid_size}x{grid_size} 点位/四角不全，只输出检测步骤")

    if points:
        ship_cells, debris_cells = detector.detect_ship_and_debris_cells(
            image, points, grid_size, debug_dir=out_dir, grid_quad=quad
        )
        print(f"  {image_path.name} -> {out_dir}")
        print(f"    船格={sorted(ship_cells)}  碎片格={sorted(debris_cells)}")
    else:
        ships, debris = detector.detect_ship_and_debris_boxes(image, debug_dir=out_dir, grid_quad=quad)
        print(f"  {image_path.name} -> {out_dir}")
        print(f"    船框={ships}")
        print(f"    碎片框={debris}")

    written = sorted(out_dir.glob("step*.png"))
    print(f"    落盘 {len(written)} 张步骤图 + report.txt")
    return len(written)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="截图文件或目录")
    parser.add_argument("--out", help=f"输出根目录（默认 {DEFAULT_OUT}）")
    parser.add_argument("--level", type=int, help="关卡号：额外输出 框->格子 映射图")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"输入不存在: {input_path}")

    images = iter_inputs(input_path)
    if not images:
        parser.error(f"没有找到图片: {input_path}")

    out_root = Path(args.out) if args.out else DEFAULT_OUT
    print(f"输入 {len(images)} 张，输出到 {out_root}")
    total = 0
    for image_path in images:
        total += dump_one(image_path, out_root, args.level)
    print(f"完成：共 {total} 张步骤图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
