from __future__ import annotations

"""用红旗 + 模板给训练样本重新打标签（非破坏：写入 labels_v2 / labels_v2_3class 字段）。

为什么：现有 labels 是"开局视觉"自动打的，稀疏且噪声大（平均每样本 1~2 格，应有 16~20）。
这里改用两个**独立、可靠**的信号：

  1. 红旗（`red_flag_centroids`）：红旗一定在潜艇上 → 该格必是潜艇格。
  2. 模板匹配（潜艇模板 + 红旗约束 / 残骸模板）：给出潜艇/残骸精灵覆盖的格。

需要**原图**（二值化裁剪里红旗和颜色都没了），所以只能处理有原图配对的样本。

用法：
    .venv\\Scripts\\python.exe tools\\relabel_from_flags.py [--backup]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import audit_label_candidates as A  # noqa: E402

from save_points.points import points_from_quad  # noqa: E402
from config import SUBMARINES  # noqa: E402
from utils.sidebar_progress import resolve_completed_ship_cells  # noqa: E402

ORIGINAL_DIR = PROJECT_ROOT / "识图" / "原图训练样本"
LABEL_DIR = PROJECT_ROOT / "识图" / "训练样本"
BACKUP_DIR = PROJECT_ROOT / "识图" / "训练样本_labels_backup"


def _cell_for_point(px, py, points, n):
    return min(
        (((px - x) ** 2 + (py - y) ** 2, (i // n, i % n)) for i, (x, y) in enumerate(points)),
        key=lambda t: t[0],
    )[1]


def _cells_inside_boxes(boxes, points, n):
    """点击点落在某个框内的格子。"""
    out = set()
    for i, (px, py) in enumerate(points):
        for bx, by, bw, bh in boxes:
            if bx <= px < bx + bw and by <= py < by + bh:
                out.add((i // n, i % n))
                break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backup", action="store_true", help="先把要改的 json 备份到 训练样本_labels_backup/")
    parser.add_argument("--tight", action="store_true",
                        help="收紧标签：只标潜艇逻辑格（几何拟合），默认标精灵覆盖格")
    args = parser.parse_args(argv)

    sub_templates = A.load_templates(A.SUB_TEMPLATE_DIR)
    debris_templates = A.load_templates(A.DEBRIS_TEMPLATE_DIR)
    print(f"潜艇模板 {len(sub_templates)}，残骸模板 {len(debris_templates)}")

    originals = sorted(ORIGINAL_DIR.glob("*.png"))
    done = skipped = 0
    tot_sub = tot_deb = 0
    for op in originals:
        jp = LABEL_DIR / f"{op.stem}.json"
        if not jp.exists():
            skipped += 1
            continue
        try:
            record = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped += 1
            continue
        quad = record.get("quad")
        grid_size = record.get("grid_size")
        if not quad or not grid_size:
            skipped += 1
            continue
        image = A.read_image_compat(op)
        if image is None:
            skipped += 1
            continue
        n = int(grid_size)
        points = points_from_quad([tuple(map(float, p)) for p in quad], n)
        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
        quad_bbox = (min(xs), min(ys), max(xs), max(ys))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        flags = A.red_flag_centroids(image)
        subs = A.submarine_candidates(gray, sub_templates, flags)
        debris = A.debris_candidates(gray, debris_templates, quad, flags)

        sub_cells = _cells_inside_boxes([c["box"] for c in subs], points, n)
        # 红旗所在格一定是潜艇格（即使模板没框住）
        for fx, fy in flags:
            sub_cells.add(_cell_for_point(fx, fy, points, n))
        # 收紧（可选）：精灵框会覆盖到相邻格（过标），用该关潜艇长度把覆盖格收敛成合法直线潜艇。
        # 注意：实测收紧后潜艇类分类反而变差（精灵覆盖格与逻辑格单格外观太像），默认关。
        lengths = SUBMARINES.get(int(record.get("level") or 0))
        if args.tight and lengths and sub_cells:
            try:
                fit = resolve_completed_ship_cells(sub_cells, lengths, grid_size=n)
                fitted = set(fit.cells)
                if fitted:
                    sub_cells = fitted
            except Exception:
                pass
        debris_cells = _cells_inside_boxes([c["box"] for c in debris], points, n) - sub_cells

        labels_v2 = [[0] * n for _ in range(n)]
        for r, c in sub_cells | debris_cells:
            labels_v2[r][c] = 1
        labels_v2_3 = [[0] * n for _ in range(n)]
        for r, c in sub_cells:
            labels_v2_3[r][c] = 1
        for r, c in debris_cells:
            labels_v2_3[r][c] = 2

        if args.backup:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jp, BACKUP_DIR / jp.name)

        record["labels_v2"] = labels_v2
        record["labels_v2_3class"] = labels_v2_3
        jp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        done += 1
        tot_sub += len(sub_cells)
        tot_deb += len(debris_cells)

    print(f"完成 {done} 个样本（跳过 {skipped}）")
    print(f"新标签: 潜艇格合计 {tot_sub}，残骸格合计 {tot_deb}")
    if args.backup:
        print(f"原标签已备份 -> {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
