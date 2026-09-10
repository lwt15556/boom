from __future__ import annotations

"""Generate label-verification overlays for the original training samples.

两类目标：
  潜艇（完整潜艇）：大号精灵模板多尺度匹配 + 只保留"框内含红旗"的候选（水纹没有红旗）
  残骸（碎片）：残骸模板多尺度匹配 + 阈值 + NMS，限制在棋盘范围内

每张图输出一张核对图（原图 + 网格 + 红旗 + 潜艇候选 + 残骸候选 + 建议格），
放到 ``_debug/label_audit/``，并写 ``summary.txt``。现有标签大多是错的，不画在图上。

用法：
    .venv\\Scripts\\python.exe tools\\audit_label_candidates.py [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from save_points.points import points_from_quad  # noqa: E402
from utils import submarine_detector as sd  # noqa: E402
from utils.image_io import read_image_compat, write_image_compat  # noqa: E402

ORIGINAL_DIR = PROJECT_ROOT / "识图" / "原图训练样本"
LABEL_DIR = PROJECT_ROOT / "识图" / "训练样本"
SUB_TEMPLATE_DIR = PROJECT_ROOT / "识图" / "潜艇模板"
DEBRIS_TEMPLATE_DIR = PROJECT_ROOT / "识图" / "残骸模板"
OUT_DIR = PROJECT_ROOT / "_debug" / "label_audit"

UI_RED_ZONES = ((0, 0, 110, 110), (1140, 590, 140, 130))
SUB_MATCH_SCALES = (0.85, 0.95, 1.05, 1.15)
SUB_MATCH_THRESHOLD = 0.50
DEBRIS_MATCH_SCALES = (0.8, 0.9, 1.0, 1.1)
DEBRIS_MATCH_THRESHOLD = 0.66
FLAG_MIN_AREA = 40


def red_flag_centroids(image: np.ndarray) -> list[tuple[int, int]]:
    """棋盘上的红旗中心（排除界面上的红色按钮）。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    mask = (((h <= 12) | (h >= 168)) & (s > 110) & (v > 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < FLAG_MIN_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if any(ux <= bx and uy <= by and bx + bw <= ux + uw and by + bh <= uy + uh
               for ux, uy, uw, uh in UI_RED_ZONES):
            continue
        m = cv2.moments(cnt)
        if m["m00"] < 1e-6:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])))
    return out


def load_templates(directory: Path):
    out = []
    if not directory.exists():
        return out
    for path in sorted(directory.iterdir()):
        img = read_image_compat(path)
        if img is None:
            continue
        out.append((path.name, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return out


def _match_all(gray, templates, scales, threshold):
    raw = []
    for name, tg in templates:
        for scale in scales:
            th, tw = int(tg.shape[0] * scale), int(tg.shape[1] * scale)
            if th < 12 or tw < 12 or th >= gray.shape[0] or tw >= gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, cv2.resize(tg, (tw, th)), cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= threshold)
            for y, x in zip(ys, xs):
                raw.append((float(res[y, x]), int(x), int(y), tw, th, name))
    return raw


def _nms(cands):
    cands = sorted(cands, key=lambda c: c["score"], reverse=True)
    keep = []
    for cand in cands:
        x, y, w, h = cand["box"]
        if all(abs(x - k["box"][0]) > k["box"][2] * 0.5 or abs(y - k["box"][1]) > k["box"][3] * 0.5 for k in keep):
            keep.append(cand)
    return keep


def submarine_candidates(gray, templates, flags):
    """模板匹配 -> 只保留框内(外扩)含红旗的候选 -> NMS。"""
    kept = []
    for score, x, y, w, h, name in _match_all(gray, templates, SUB_MATCH_SCALES, SUB_MATCH_THRESHOLD):
        for fx, fy in flags:
            if x - 10 <= fx <= x + w + 10 and y - 10 <= fy <= y + h + 10:
                kept.append({"score": score, "box": (x, y, w, h), "template": name, "flag": (fx, fy)})
                break
    return _nms(kept)


def debris_candidates(gray, templates, quad_poly, flags):
    """残骸模板匹配，限制在棋盘**菱形内部**（排除菱形外的海浪/边缘），并排除与红旗重叠的（那是潜艇）。"""
    poly = np.asarray(quad_poly, dtype=np.float32).reshape(-1, 2)
    kept = []
    for score, x, y, w, h, name in _match_all(gray, templates, DEBRIS_MATCH_SCALES, DEBRIS_MATCH_THRESHOLD):
        cx, cy = x + w / 2, y + h / 2
        # 菱形外的匹配（右上角高亮海浪等）不算残骸。
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) < 0:
            continue
        if any(x - 8 <= fx <= x + w + 8 and y - 8 <= fy <= y + h + 8 for fx, fy in flags):
            continue
        kept.append({"score": score, "box": (x, y, w, h), "template": name})
    return _nms(kept)


def draw_audit(image, points, grid_size, labels_grid, flags, subs, debris, suggested_cells):
    """核对图：不再画现有标签（大多错误、只造成干扰），只画检测结果。"""
    canvas = image.copy()
    n = int(grid_size)
    for px, py in points:
        cv2.circle(canvas, (int(px), int(py)), 2, (200, 200, 200), -1)
    for (r, c) in suggested_cells:
        px, py = points[r * n + c]
        cv2.circle(canvas, (int(px), int(py)), 5, (0, 255, 0), -1)
    for fx, fy in flags:
        cv2.drawMarker(canvas, (fx, fy), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    for cand in subs:
        x, y, w, h = cand["box"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 128, 0), 2)
        cv2.putText(canvas, f"SUB {cand['score']:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)
    for cand in debris:
        x, y, w, h = cand["box"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(canvas, f"DEB {cand['score']:.2f}", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    args = parser.parse_args(argv)

    sub_templates = load_templates(SUB_TEMPLATE_DIR)
    debris_templates = load_templates(DEBRIS_TEMPLATE_DIR)
    print(f"潜艇模板 {len(sub_templates)} 个，残骸模板 {len(debris_templates)} 个")

    originals = sorted(ORIGINAL_DIR.glob("*.png"))
    if args.limit:
        originals = originals[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    total = 0
    for op in originals:
        jp = LABEL_DIR / f"{op.stem}.json"
        if not jp.exists():
            continue
        record = json.loads(jp.read_text(encoding="utf-8"))
        quad = record.get("quad")
        grid_size = record.get("grid_size")
        labels_grid = record.get("labels")
        if not quad or not grid_size:
            continue
        image = read_image_compat(op)
        if image is None:
            continue
        n = int(grid_size)
        points = points_from_quad([tuple(map(float, p)) for p in quad], n)
        xs = [p[0] for p in quad]; ys = [p[1] for p in quad]
        quad_bbox = (min(xs), min(ys), max(xs), max(ys))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        flags = red_flag_centroids(image)
        subs = submarine_candidates(gray, sub_templates, flags)
        debris = debris_candidates(gray, debris_templates, quad, flags)
        boxes = [c["box"] for c in subs] + [c["box"] for c in debris]
        suggested = sd._assign_boxes_to_cells(boxes, points, n)
        overlay = draw_audit(image, points, n, labels_grid, flags, subs, debris, suggested)
        write_image_compat(OUT_DIR / f"{op.stem}_audit.png", overlay)
        cur = sum(1 for row in (labels_grid or []) for v in row if int(v) == 1)
        lines.append(
            f"{op.stem}: 红旗={len(flags)} 潜艇候选={len(subs)} 残骸候选={len(debris)} "
            f"建议格={len(suggested)} 现有标签={cur}"
        )
        total += 1
        print(lines[-1])
    (OUT_DIR / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n共 {total} 张核对图 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
