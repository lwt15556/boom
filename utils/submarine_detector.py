from __future__ import annotations

"""Detect surfaced submarines and debris (wrecks) in a board screenshot.

Ported from ``detection_algorithm.py``: Sauvola local threshold + HSV blue mask ->
non-water foreground, group blobs with DBSCAN (``min_samples=1`` is single-linkage
on an eps graph), and accept submarines when a cluster carries a red flag and
enough area.  Detected ship bodies are erased from the foreground and the
remaining medium blobs are classified as debris/wreck fragments.  Both are then
mapped onto the board grid cells.

The whole pipeline runs **once** per frame (``detect_ship_and_debris_boxes``);
``detect_submarine_boxes`` / ``detect_debris_boxes`` are thin wrappers kept for
callers that only need one half.

Debugging
---------
Every stage can be dumped as a numbered PNG plus a ``report.txt`` with the numbers
behind each decision.  Either pass ``debug_dir=`` to any entry point, or set the
environment variable ``BBMA_DETECTOR_DEBUG_DIR`` (each run then gets its own
timestamped sub-directory).  The CLI wrapper is::

    .venv\\Scripts\\python.exe tools\\dump_detector_steps.py <image.png>

Dumping is off by default and costs nothing on the production path.

This complements the per-cell vision with a sprite-level detector for the
surface-submarines and wrecks that are visible at the start of a level.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from utils.image_io import write_image_compat

# ===================== 参数配置区 =====================

# ----- 图像预处理参数 -----
# ROI：棋盘大菱形四角（全帧 1280x720 坐标，顺序 = 上、右、下、左）。
# 检测只在这四个角围成的四边形内部进行，避免把棋盘外的海面/边缘当成前景或碎片。
# 由 after_swipe 截图校准；非 1280x720 帧会按比例等比缩放。
ROI_QUAD = ((676, 96), (1100, 355), (686, 718), (262, 352))
# 掩码外边距（像素）：往外扩一点，避免贴边探出的潜艇/碎片被四边形裁掉。0 = 严格按四角。
# 实测 after_swipe 上顶部潜艇探出棋盘左上边缘约 20px，设为 20 可找回而不引入海面假阳性。
ROI_QUAD_MARGIN = 20

# HSV 蓝色掩码范围（用于过滤海水背景）。
H_MIN = 85
H_MAX = 135
S_MIN = 12
S_MAX = 120
V_MIN = 85
V_MAX = 255

# 面积过滤阈值：小于此面积的连通域直接丢弃（单位：像素）。
MIN_AREA = 100

# 形态学开运算核大小（2x2 足够去除微小噪点）。
MORPH_KERNEL_SIZE = 2

# Sauvola 局部二值化参数。
SAUVOLA_WINDOW = 25
SAUVOLA_K = 0.26
SAUVOLA_R = 127.0

# ----- 船检测参数 -----
# 膨胀核大小（用于连接船体断裂，单位：像素）。
DILATE_KERNEL_SIZE = 5
# DBSCAN 聚类半径（同一艘船断裂部分的最大距离，单位：像素）。可调范围约 40~70。
DBSCAN_EPS = 40
# DBSCAN 最小样本数（固定为 1，每个点都视为一个簇）。
DBSCAN_MIN_SAMPLES = 1

# 红色像素和阈值：超过此值且面积 > SHIP_AREA_MIN 则判为船。
SHIP_RED_SUM_THRESH = 40
# 船的最小面积（单位：像素）。
SHIP_AREA_MIN = 50
# 网格丢弃阈值：凸包密实度 >= 此值且红色 < RED_PROTECT_THRESHOLD 则丢弃。
HULL_SOLID_GRID_THRESH = 0.78
# 红色保护阈值：即使密实度高，红色和 > 此值仍保留（防止误丢弃红船）。
RED_PROTECT_THRESHOLD = 100
# 超大背景簇丢弃阈值（单位：像素）。
CLUSTER_AREA_MAX = 20000

# 红色掩码范围（固定，不随蓝掩码参数变化）。
RED_LOWER = (0, 110, 70)
RED_UPPER = (8, 255, 255)

# ----- 碎片检测参数 -----
# 碎片面积范围（单位：像素）。
FRAG_AREA_MIN = 500
FRAG_AREA_MAX = 1200
# 碎片长宽比范围（宽/高）。
FRAG_ASPECT_MIN = 0.8
FRAG_ASPECT_MAX = 2.0
# 矩形度上限（面积/外接矩形面积），排除过于规则的方块（如网格）。
FRAG_RECT_MAX = 0.95
# 凸包密实度上限（面积/凸包面积），排除过于饱满的形状。
FRAG_HULL_SOLID_MAX = 0.95
# 碎片允许的最大红色像素和（避免把带红色船体碎片误判）。
FRAG_RED_MAX = 50

# 参考帧尺寸：ROI_QUAD、patch 半径等全帧坐标都按这个尺寸标定。
REFERENCE_FRAME = (1280, 720)

# ----- 逐格规则识别参数（残骸 / 潜艇 / 海浪） -----
# 1280x720 下格心间距约 44px，格子菱形约 74x46px，格心间距的 1/5 ≈ 9px。
# 残骸 = 格内一块紧凑的灰白实体：开运算后最大连通块面积占格 ≥5% 且填充率 ≥40%。
# 海浪 = 细长软泡沫条纹，开运算后只剩 ≤3%，因此被这条判据挡住。
# 潜艇艇身也是灰白实体，只有红色指挥塔能把它和残骸分开。
_GRAY_BLOB_SAT_MAX = 80.0     # 灰白实体饱和度上限（海浪/海水饱和度更高）
_GRAY_BLOB_VAL_MIN = 100.0    # 灰白实体亮度下限
_GRAY_BLOB_OPEN_RATIO = 0.2   # 开运算核 = 格心间距 * 该比例（滤掉网格线与细泡沫）
_WRECK_BLOB_AREA_MIN = 0.05   # 最大灰白连通块面积 / 格面积
_WRECK_BLOB_FILL_MIN = 0.40   # 最大灰白连通块面积 / 其外接矩形面积
_WRECK_TEXTURE_MIN = 30.0     # 块内平均 |Laplacian|：残骸有碎块和暗影，泡沫很平滑
_RED_TOWER_HUE_MAX = 14       # 红色色相上限（0 附近）
_RED_TOWER_HUE_MIN = 166      # 红色色相下限（180 附近回绕）
_RED_TOWER_SAT_MIN = 110
_RED_TOWER_VAL_MIN = 90
_RED_TOWER_AREA_MIN = 60.0    # 红色指挥塔最小面积（像素）

# 设成目录路径即开启逐步骤落盘；每次调用在该目录下建一个时间戳子目录。
DEBUG_ENV = "BBMA_DETECTOR_DEBUG_DIR"

# 叠加层配色（BGR）。
COLOR_SHIP = (0, 0, 255)          # 红：判定为船
COLOR_DEBRIS = (0, 255, 0)        # 绿：判定为碎片
COLOR_GRID = (255, 255, 0)        # 青：密实度过高被当网格丢弃
COLOR_UNKNOWN = (0, 255, 255)     # 黄：既不是船也不是碎片
COLOR_TOO_BIG = (255, 0, 255)     # 品红：超大背景簇
COLOR_TEXT = (255, 255, 255)      # 白：说明文字


# ===================== 调试落盘 =====================


class _StepDumper:
    """把流水线每一步写成 PNG，并记录每一步的数值。

    ``enabled`` 为 False 时所有方法都是空操作，生产路径零开销。
    """

    def __init__(self, base_dir: str | Path | None = None, *, add_timestamp: bool = False):
        self.enabled = base_dir is not None and str(base_dir).strip() != ""
        self.out_dir: Path | None = None
        if self.enabled:
            base = Path(base_dir)
            if add_timestamp:
                base = base / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.out_dir = base
        self._step = 0
        self._report: list[str] = []
        # 供后续步骤复用/绘制最终叠加层的中间结果。
        self.records: dict[str, object] = {}
        self.roi: np.ndarray | None = None

    def save(self, name: str, image: np.ndarray | None, note: str = "") -> None:
        """写一张 ``stepNN_<name>.png``。"""
        if not self.enabled or image is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._step += 1
        filename = f"step{self._step:02d}_{name}.png"
        write_image_compat(self.out_dir / filename, image)
        self._report.append(f"{filename}  {note}".rstrip())

    def log(self, text: str = "") -> None:
        if self.enabled:
            self._report.append(text)

    def flush(self, extra: dict | None = None) -> None:
        if not self.enabled:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        lines = list(self._report)
        if extra:
            lines.append("")
            lines.append("==== 汇总 ====")
            lines.extend(f"{key}: {value}" for key, value in extra.items())
        (self.out_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_debug_dir(debug_dir) -> tuple[Path | None, bool]:
    """-> (输出目录, 是否加时间戳子目录)。

    ``debug_dir=None`` 时读环境变量 ``BBMA_DETECTOR_DEBUG_DIR``；显式传空字符串关闭。
    """
    if debug_dir is None:
        value = os.environ.get(DEBUG_ENV, "").strip()
        if not value:
            return None, False
        return Path(value), True
    text = str(debug_dir).strip()
    if not text:
        return None, False
    return Path(debug_dir), False


def _make_dumper(debug_dir) -> _StepDumper:
    base, stamp = _resolve_debug_dir(debug_dir)
    return _StepDumper(base, add_timestamp=stamp)


def _text(image: np.ndarray, text: str, org: tuple[int, int], color=COLOR_TEXT, scale: float = 0.5) -> None:
    """带黑色描边的文字，避免压在浅色背景上看不清。"""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _label_origin(x: int, y: int) -> tuple[int, int]:
    """把标签放在框上方，贴到画面顶边时改放框内。"""
    return (int(x), int(y) - 6 if y > 14 else int(y) + 14)


def _draw_centers(dilated: np.ndarray, centers: list[list[float]], labels: np.ndarray | None) -> np.ndarray:
    """在膨胀掩码上标出每个轮廓质心及其聚类编号。"""
    canvas = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    for index, (cx, cy) in enumerate(centers):
        cluster = int(labels[index]) if labels is not None else -1
        color = COLOR_SHIP if cluster < 0 else COLOR_DEBRIS
        point = (int(round(cx)), int(round(cy)))
        cv2.circle(canvas, point, 3, color, -1)
        cv2.circle(canvas, point, 5, (0, 0, 0), 1)
        _text(canvas, f"{index}:C{cluster}", (point[0] + 6, point[1] - 4), color, 0.4)
    return canvas


def _draw_clusters(background: np.ndarray, clusters: list[dict]) -> np.ndarray:
    """按判定结果给每个簇画框并标注特征。"""
    canvas = background.copy()
    palette = {
        "ship": COLOR_SHIP,
        "grid": COLOR_GRID,
        "unclassified": COLOR_UNKNOWN,
        "too_large": COLOR_TOO_BIG,
    }
    for info in clusters:
        color = palette.get(str(info["decision"]), COLOR_TEXT)
        x, y, w, h = info["box"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        _text(
            canvas,
            f"C{info['id']} {info['decision']}",
            _label_origin(x, y),
            color,
            0.5,
        )
        _text(
            canvas,
            f"area={info['area']:.0f} sol={info['solidity']:.2f} red={info['red_sum']:.0f}",
            (x, y + h + 14 if y + h + 14 < canvas.shape[0] else y + h - 6),
            color,
            0.42,
        )
    return canvas


def _draw_debris(background: np.ndarray, records: list[dict]) -> np.ndarray:
    """给每个碎片候选画框：绿=通过，灰=被拒。"""
    canvas = background.copy()
    for record in records:
        x, y, w, h = record["box"]
        color = COLOR_DEBRIS if record["accepted"] else (150, 150, 150)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = "Debris" if record["accepted"] else "reject"
        _text(canvas, label, _label_origin(x, y), color, 0.5)
        if record["reason"]:
            _text(canvas, record["reason"], (x, y + h + 14), color, 0.4)
    return canvas


def _draw_final(
    roi: np.ndarray,
    clusters: list[dict],
    debris: list[dict],
) -> np.ndarray:
    """最终叠加：红=船，绿=碎片，青=网格丢弃，黄=未归类，品红=超大簇。"""
    canvas = _draw_clusters(roi, clusters)
    for record in debris:
        if not record["accepted"]:
            continue
        x, y, w, h = record["box"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_DEBRIS, 2)
        _text(canvas, "Debris", _label_origin(x, y), COLOR_DEBRIS, 0.5)
    ships = sum(1 for info in clusters if info["decision"] == "ship")
    debris_count = sum(1 for record in debris if record["accepted"])
    _text(canvas, f"ships={ships} debris={debris_count}", (8, 22), COLOR_TEXT, 0.7)
    return canvas


def _draw_cell_mapping(
    grid_img: np.ndarray,
    click_points,
    grid_size: int,
    ship_cells: set[tuple[int, int]],
    debris_cells: set[tuple[int, int]],
) -> np.ndarray:
    """把"框 -> 格子"的映射画出来，看清命中/漏掉的是哪一格。"""
    canvas = grid_img.copy()
    for index, (px, py) in enumerate(click_points):
        row, col = divmod(int(index), int(grid_size))
        if (row, col) in ship_cells:
            color, radius = COLOR_SHIP, 5
        elif (row, col) in debris_cells:
            color, radius = COLOR_DEBRIS, 5
        else:
            color, radius = (200, 200, 200), 2
        cv2.circle(canvas, (int(px), int(py)), radius, color, -1)
    return canvas


# ===================== 流水线 =====================


def sauvola_local_threshold(
    gray_img: np.ndarray,
    window_size: int = SAUVOLA_WINDOW,
    k: float = SAUVOLA_K,
) -> np.ndarray:
    """Sauvola 局部自适应二值化。"""
    img_float = gray_img.astype(np.float32)
    mean = cv2.boxFilter(img_float, -1, (window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(img_float ** 2, -1, (window_size, window_size), normalize=True)
    # float32 下 mean_sq - mean**2 可能因舍入变成极小负数，夹到 0 以免 sqrt -> NaN。
    std = np.sqrt(np.maximum(0.0, mean_sq - mean ** 2))
    threshold = mean * (1.0 + k * ((std / SAUVOLA_R) - 1.0))
    return np.where(img_float > threshold, 255, 0).astype(np.uint8)


def preprocess(roi: np.ndarray, dumper: _StepDumper | None = None) -> tuple[np.ndarray, np.ndarray]:
    """预处理：Sauvola 二值化、HSV 蓝色掩码、开运算、面积过滤。

    返回 ``(combined, red_mask)``：``combined`` 是**膨胀前**的前景二值图（碎片
    检测必须用它），``red_mask`` 是红船旗掩码。
    """
    dumper = dumper or _StepDumper()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sauvola_bin = sauvola_local_threshold(gray)
    dumper.save(
        "sauvola",
        sauvola_bin,
        f"Sauvola 局部二值化 window={SAUVOLA_WINDOW} k={SAUVOLA_K} R={SAUVOLA_R} 前景={int((sauvola_bin > 0).sum())}px",
    )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(
        hsv,
        np.array([H_MIN, S_MIN, V_MIN]),
        np.array([H_MAX, S_MAX, V_MAX]),
    )
    dumper.save("blue_mask", blue_mask, f"HSV 蓝掩码 H[{H_MIN},{H_MAX}] S[{S_MIN},{S_MAX}] V[{V_MIN},{V_MAX}] 命中={int((blue_mask > 0).sum())}px")

    combined = cv2.bitwise_and(sauvola_bin, blue_mask)
    dumper.save("and", combined, f"二值 AND 蓝掩码（海水前景）={int((combined > 0).sum())}px")

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    dumper.save("morph_open", combined, f"开运算 {MORPH_KERNEL_SIZE}x{MORPH_KERNEL_SIZE} -> {int((combined > 0).sum())}px")

    filtered = np.zeros_like(combined)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = dropped = 0
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_AREA:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
            kept += 1
        else:
            dropped += 1
    combined = filtered
    dumper.save(
        "area_filtered",
        combined,
        f"面积过滤 MIN_AREA={MIN_AREA}：保留 {kept} 个连通域、丢弃 {dropped} 个 -> {int((combined > 0).sum())}px（碎片检测用这张）",
    )

    red_mask = cv2.inRange(hsv, np.array(RED_LOWER), np.array(RED_UPPER))
    dumper.save(
        "red_mask",
        red_mask,
        f"红色掩码 {RED_LOWER}~{RED_UPPER} 命中={int((red_mask > 0).sum())}px",
    )
    return combined, red_mask


def detect_ships(
    combined_mask: np.ndarray,
    red_mask: np.ndarray,
    dumper: _StepDumper | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    """从前景二值图和红色掩码中识别船。

    返回 ``(ship_boxes, ship_contours)``，坐标均为 **ROI 局部坐标**。
    ``ship_contours`` 是聚类后的船体点集，用于从前景里擦除船体。
    """
    dumper = dumper or _StepDumper()

    kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
    dilated = cv2.dilate(combined_mask, kernel, iterations=1)
    dumper.save("dilated", dilated, f"膨胀 {DILATE_KERNEL_SIZE}x{DILATE_KERNEL_SIZE}（船聚类输入）-> {int((dilated > 0).sum())}px")

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers: list[list[float]] = []
    contour_list: list[np.ndarray] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        moments = cv2.moments(cnt)
        if moments["m00"] < 1e-6:
            continue
        centers.append(
            [
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            ]
        )
        contour_list.append(cnt)

    if not centers:
        dumper.save("centers", cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR), "无有效轮廓，聚类跳过")
        dumper.records["clusters"] = []
        return [], []

    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(np.array(centers)).labels_
    dumper.save(
        "centers",
        _draw_centers(dilated, centers, labels),
        f"轮廓质心 {len(centers)} 个（DBSCAN eps={DBSCAN_EPS} min_samples={DBSCAN_MIN_SAMPLES} -> {len(set(labels))} 簇）",
    )

    ship_boxes: list[tuple[int, int, int, int]] = []
    ship_contours: list[np.ndarray] = []
    clusters: list[dict] = []
    dumper.log("")
    dumper.log("==== 簇特征（ROI 坐标）====")
    for cluster_id in set(labels):
        idx = np.where(labels == cluster_id)[0]
        all_points = np.concatenate([contour_list[i] for i in idx])
        x, y, w, h = cv2.boundingRect(all_points)
        area = cv2.contourArea(all_points)
        hull_area = cv2.contourArea(cv2.convexHull(all_points))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0
        red_sum = float(np.sum(red_mask[y : y + h, x : x + w]))

        decision = "unclassified"
        if area > CLUSTER_AREA_MAX or w <= 0 or h <= 0:
            decision = "too_large"
        elif solidity >= HULL_SOLID_GRID_THRESH and red_sum <= RED_PROTECT_THRESHOLD:
            decision = "grid"
        elif red_sum > SHIP_RED_SUM_THRESH and area > SHIP_AREA_MIN:
            decision = "ship"

        clusters.append(
            {
                "id": int(cluster_id),
                "points": int(len(idx)),
                "box": (int(x), int(y), int(w), int(h)),
                "area": float(area),
                "solidity": float(solidity),
                "red_sum": red_sum,
                "aspect": float(w / h) if h > 0 else 0.0,
                "decision": decision,
            }
        )
        dumper.log(
            f"簇{int(cluster_id):2d} 点数={len(idx):3d} box=({x},{y},{w},{h}) area={area:7.1f} "
            f"sol={solidity:.3f} red={red_sum:7.1f} ar={float(w / h) if h > 0 else 0.0:.2f} -> {decision}"
        )

        if decision == "too_large":
            continue
        if decision == "grid":
            continue
        if decision == "ship":
            ship_boxes.append((x, y, w, h))
            ship_contours.append(all_points)

    background = dumper.roi if dumper.roi is not None else cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    dumper.save("clusters", _draw_clusters(background, clusters), "聚类判定：红=船 青=网格丢弃 黄=未归类 品红=超大簇")
    dumper.records["clusters"] = clusters
    return ship_boxes, ship_contours


def detect_debris(
    fragment_mask: np.ndarray,
    ship_contours: list[np.ndarray],
    red_mask: np.ndarray,
    dumper: _StepDumper | None = None,
) -> list[tuple[int, int, int, int]]:
    """擦除船体后提取碎片，返回碎片框列表（ROI 局部坐标）。

    ``fragment_mask`` 必须是**膨胀前**的前景二值图，否则船体膨胀会把相邻碎片
    连成一片而漏检。
    """
    dumper = dumper or _StepDumper()

    erased = fragment_mask.copy()
    for cnt in ship_contours:
        cv2.drawContours(erased, [cnt], -1, 0, -1)
    dumper.save("erased_ships", erased, f"擦除 {len(ship_contours)} 个船体后的掩码")

    debris_boxes: list[tuple[int, int, int, int]] = []
    records: list[dict] = []
    contours, _ = cv2.findContours(erased, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dumper.log("")
    dumper.log("==== 碎片候选（ROI 坐标）====")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        aspect = w / h
        rect_fill = area / (w * h)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        hull_solid = area / hull_area if hull_area > 1e-6 else 0.0
        red_sum = float(np.sum(red_mask[y : y + h, x : x + w]))

        reason = ""
        if not (FRAG_AREA_MIN <= area <= FRAG_AREA_MAX):
            reason = f"area {area:.0f} ∉ [{FRAG_AREA_MIN},{FRAG_AREA_MAX}]"
        elif not (FRAG_ASPECT_MIN <= aspect <= FRAG_ASPECT_MAX):
            reason = f"aspect {aspect:.2f} ∉ [{FRAG_ASPECT_MIN},{FRAG_ASPECT_MAX}]"
        elif rect_fill > FRAG_RECT_MAX:
            reason = f"rect {rect_fill:.2f} > {FRAG_RECT_MAX}"
        elif hull_solid > FRAG_HULL_SOLID_MAX:
            reason = f"hull {hull_solid:.2f} > {FRAG_HULL_SOLID_MAX}"
        elif red_sum > FRAG_RED_MAX:
            reason = f"red {red_sum:.0f} > {FRAG_RED_MAX}"

        accepted = reason == ""
        records.append(
            {
                "box": (int(x), int(y), int(w), int(h)),
                "area": float(area),
                "aspect": float(aspect),
                "rect_fill": float(rect_fill),
                "hull_solid": float(hull_solid),
                "red_sum": red_sum,
                "accepted": accepted,
                "reason": reason,
            }
        )
        dumper.log(
            f"box=({x},{y},{w},{h}) area={area:7.1f} ar={aspect:.2f} rect={rect_fill:.3f} "
            f"hull={hull_solid:.3f} red={red_sum:7.1f} -> {'通过' if accepted else '拒绝: ' + reason}"
        )
        if accepted:
            debris_boxes.append((x, y, w, h))

    background = dumper.roi if dumper.roi is not None else cv2.cvtColor(erased, cv2.COLOR_GRAY2BGR)
    dumper.save("debris", _draw_debris(background, records), "碎片候选：绿=通过 灰=被拒")
    dumper.records["debris"] = records
    return debris_boxes


def _scaled_quad(frame_shape) -> list[tuple[int, int]]:
    """把 ROI_QUAD 按实际分辨率等比缩放到当前帧的整数角点。"""
    height, width = frame_shape[:2]
    sx = width / float(REFERENCE_FRAME[0]) if width else 1.0
    sy = height / float(REFERENCE_FRAME[1]) if height else 1.0
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in ROI_QUAD]


def _normalize_quad(grid_quad) -> list[tuple[int, int]] | None:
    """把任意外部 4 角输入规范成 ``[(x, y)]`` 整数列表；无效返回 None。"""
    if grid_quad is None:
        return None
    try:
        pts = list(grid_quad)
    except TypeError:
        return None
    if len(pts) < 4:
        return None
    out: list[tuple[int, int]] = []
    for p in pts[:4]:
        try:
            out.append((int(round(float(p[0]))), int(round(float(p[1])))))
        except (TypeError, ValueError, IndexError):
            return None
    return out


def _resolve_quad(grid_img, grid_quad) -> list[tuple[int, int]] | None:
    """确定本帧用的四边形 ROI。

    优先用调用方给的每关 ``grid_quad``（save_points/points.json 的 quad 或自动识别
    的红色边框 quad，本身就处在本帧坐标系里，不再缩放）；取不到时回退到
    ``ROI_QUAD``（按分辨率等比缩放）。返回**当前帧坐标**的角点。
    """
    if grid_img is None or grid_img.ndim < 2:
        return None
    explicit = _normalize_quad(grid_quad)
    if explicit is not None:
        return explicit
    return _scaled_quad(grid_img.shape)


def _roi_box(quad) -> tuple[int, int, int, int]:
    """返回四边形 ROI 的包围盒 ``(x0, y0, x1, y1)``。"""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _run_pipeline(
    grid_img: np.ndarray,
    dumper: _StepDumper,
    quad: list[tuple[int, int]] | None,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """核心流水线（ROI -> 预处理 -> 船 -> 碎片），返回全帧坐标，不负责收尾落盘。"""
    if grid_img is None or grid_img.ndim < 2:
        dumper.log("grid_img 为空，跳过")
        return [], []
    if quad is None or len(quad) < 4:
        dumper.log("没有可用的四边形 ROI，跳过")
        return [], []

    rx0, ry0, rx1, ry1 = _roi_box(quad)
    dumper.save("input", grid_img, f"输入帧 shape={grid_img.shape[:2]} quad={quad}")
    if rx1 <= rx0 or ry1 <= ry0:
        dumper.log(f"ROI 超出画面 ({rx0},{ry0})-({rx1},{ry1})")
        return [], []

    roi = grid_img[ry0:ry1, rx0:rx1]
    dumper.roi = roi
    dumper.save("roi", roi, f"ROI 裁剪 ({rx0},{ry0})-({rx1},{ry1})，后续所有坐标都是这个裁剪系")

    # 只保留四边形内部的像素：把四边形外的海水/边缘涂黑，Sauvola/蓝掩码都不会再碰到它。
    quad_crop = np.array([[[x - rx0, y - ry0] for x, y in quad]], dtype=np.int32)
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, quad_crop, 255)
    margin = int(ROI_QUAD_MARGIN)
    if margin > 0:
        roi_mask = cv2.dilate(roi_mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    roi_masked = cv2.bitwise_and(roi, cv2.merge([roi_mask, roi_mask, roi_mask]))

    if dumper.enabled:
        marked = grid_img.copy()
        polygon_full = np.array([[[x, y] for x, y in quad]], dtype=np.int32)
        cv2.polylines(marked, polygon_full, True, (0, 255, 255), 2)
        dumper.save("roi_quad", marked, "全帧上画出棋盘四边形 ROI（角点见 quad）")
        dumper.log(f"quad(当前帧坐标) = {quad}")
        crop_marked = roi.copy()
        cv2.polylines(crop_marked, quad_crop, True, (0, 255, 255), 2)
        dumper.save("roi_mask", roi_masked, "四边形掩码后的 ROI（涂黑部分不参与检测）")

    combined_mask, red_mask = preprocess(roi_masked, dumper)
    ship_boxes, ship_contours = detect_ships(combined_mask, red_mask, dumper)
    debris_boxes = detect_debris(combined_mask, ship_contours, red_mask, dumper)

    ships = [(x + rx0, y + ry0, w, h) for x, y, w, h in ship_boxes]
    debris = [(x + rx0, y + ry0, w, h) for x, y, w, h in debris_boxes]
    dumper.records["ships"] = ships
    dumper.records["debris_boxes"] = debris
    return ships, debris


def detect_ship_and_debris_boxes(
    grid_img: np.ndarray,
    debug_dir=None,
    grid_quad=None,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """跑一次完整流水线，返回 ``(ship_boxes, debris_boxes)``（全帧坐标）。

    ROI 裁剪、预处理、船检测只做一次，碎片检测复用同一帧的中间结果——调用方
    同时需要船和碎片时必须用这个入口，不要分别调 ``detect_submarine_boxes`` /
    ``detect_debris_boxes``（那会把整条流水线算两遍）。

    ``grid_quad``：该关棋盘的 4 个角点（全帧坐标，如 ``save_points/points.json`` 的
    ``quad`` 或自动识别的棋盘四角），用它当检测区域；不传则回退到 ``ROI_QUAD``。

    ``debug_dir`` 给目录路径时，逐步落盘 PNG + ``report.txt``。
    """
    dumper = _make_dumper(debug_dir)
    quad = _resolve_quad(grid_img, grid_quad)
    ships, debris = _run_pipeline(grid_img, dumper, quad)
    dumper.save(
        "final",
        _draw_final(
            dumper.roi if dumper.roi is not None else np.zeros((1, 1, 3), np.uint8),
            dumper.records.get("clusters", []),
            dumper.records.get("debris", []),
        ),
        f"最终结果：船 {len(ships)} 个、碎片 {len(debris)} 个（这里还是 ROI 坐标）",
    )
    dumper.flush(
        {
            "ships_full_frame": ships,
            "debris_full_frame": debris,
        }
    )
    return ships, debris


def detect_submarine_boxes(grid_img: np.ndarray, debug_dir=None, grid_quad=None) -> list[tuple[int, int, int, int]]:
    """Return a list of ship boxes in full-frame coords, or an empty list."""
    return detect_ship_and_debris_boxes(grid_img, debug_dir=debug_dir, grid_quad=grid_quad)[0]


def detect_debris_boxes(grid_img: np.ndarray, debug_dir=None, grid_quad=None) -> list[tuple[int, int, int, int]]:
    """Return a list of debris/wreck boxes in full-frame coords."""
    return detect_ship_and_debris_boxes(grid_img, debug_dir=debug_dir, grid_quad=grid_quad)[1]


def _map_boxes_to_cells(
    boxes: list[tuple[int, int, int, int]],
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
) -> set[tuple[int, int]]:
    """Map boxes onto grid cells: a cell is marked when its click point is inside a box."""
    if grid_size <= 0 or not click_points or not boxes:
        return set()
    cells: set[tuple[int, int]] = set()
    for index, (px, py) in enumerate(click_points):
        row, col = divmod(int(index), int(grid_size))
        if row >= int(grid_size) or col >= int(grid_size):
            continue
        for bx, by, bw, bh in boxes:
            if bx <= px < bx + bw and by <= py < by + bh:
                cells.add((row, col))
                break
    return cells


def _assign_boxes_to_cells(
    boxes: list[tuple[int, int, int, int]],
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
) -> set[tuple[int, int]]:
    """二分配对：把每个精灵框匹配到离它中心最近的、且不与其它框重复的那一格。

    用 ``scipy.optimize.linear_sum_assignment`` 做最小总距离的 1:1 匹配，避免
    "一艘艇覆盖一片格"（过标）。只有格点落在框（外扩一点）附近才保留，太远的
    匹配丢弃，防止海面杂斑抓走不相干的格。返回被匹配到的格子集合。
    """
    if grid_size <= 0 or not click_points or not boxes:
        return set()
    cells: list[tuple[int, int]] = []
    pts: list[tuple[int, int]] = []
    for index, (px, py) in enumerate(click_points):
        row, col = divmod(int(index), int(grid_size))
        if row >= int(grid_size) or col >= int(grid_size):
            continue
        cells.append((row, col))
        pts.append((int(px), int(py)))
    if not pts:
        return set()

    centers = [(int(bx + bw / 2.0), int(by + bh / 2.0)) for bx, by, bw, bh in boxes]
    cost = np.zeros((len(centers), len(pts)), dtype=np.float64)
    for i, (cx, cy) in enumerate(centers):
        for j, (px, py) in enumerate(pts):
            cost[i, j] = (cx - px) ** 2 + (cy - py) ** 2

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # 没有 scipy 时退回到"格子是否落在某框内"的旧逻辑。
        return _map_boxes_to_cells(boxes, click_points, grid_size)

    rows, cols = linear_sum_assignment(cost)
    matched: set[tuple[int, int]] = set()
    for i, j in zip(rows, cols):
        bx, by, bw, bh = boxes[int(i)]
        px, py = pts[int(j)]
        # 格点必须落在框附近（框中心向外最多半个长边 + 一点余量），否则视为杂斑匹配，丢弃。
        margin = 0.5 * float(max(bw, bh)) + 8.0
        near_x = abs(px - (bx + bw / 2.0)) <= bw / 2.0 + margin
        near_y = abs(py - (by + bh / 2.0)) <= bh / 2.0 + margin
        if near_x and near_y:
            matched.add(cells[int(j)])
    return matched


def detect_submarine_cells(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    debug_dir=None,
    grid_quad=None,
) -> set[tuple[int, int]]:
    """Map detected submarine boxes onto grid cells -> ``{(row, col)}`` (1:1 二分配对)."""
    return _assign_boxes_to_cells(
        detect_submarine_boxes(grid_img, debug_dir=debug_dir, grid_quad=grid_quad),
        click_points,
        grid_size,
    )


def detect_debris_cells(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    debug_dir=None,
    grid_quad=None,
) -> set[tuple[int, int]]:
    """Map detected debris/wreck boxes onto grid cells -> ``{(row, col)}`` (1:1 二分配对)."""
    return _assign_boxes_to_cells(
        detect_debris_boxes(grid_img, debug_dir=debug_dir, grid_quad=grid_quad),
        click_points,
        grid_size,
    )


def detect_ship_and_debris_cells(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    debug_dir=None,
    grid_quad=None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Single-pass variant of the two ``*_cells`` helpers.

    Returns ``(submarine_cells, debris_cells)`` while running the pixel pipeline
    exactly once, instead of once per half.  ``grid_quad`` 给该关棋盘四角作为检测区域。
    """
    dumper = _make_dumper(debug_dir)
    quad = _resolve_quad(grid_img, grid_quad)
    ship_boxes, debris_boxes = _run_pipeline(grid_img, dumper, quad)
    ship_cells = _assign_boxes_to_cells(ship_boxes, click_points, grid_size)
    debris_cells = _assign_boxes_to_cells(debris_boxes, click_points, grid_size)

    dumper.save(
        "final",
        _draw_final(
            dumper.roi if dumper.roi is not None else np.zeros((1, 1, 3), np.uint8),
            dumper.records.get("clusters", []),
            dumper.records.get("debris", []),
        ),
        f"最终结果：船 {len(ship_boxes)} 个、碎片 {len(debris_boxes)} 个（ROI 坐标）",
    )
    if dumper.enabled and click_points and grid_img is not None and grid_img.ndim >= 2:
        dumper.save(
            "cells",
            _draw_cell_mapping(grid_img, click_points, grid_size, ship_cells, debris_cells),
            f"框 -> 格子映射：红=船格 {sorted(ship_cells)} 绿=碎片格 {sorted(debris_cells)}",
        )
    dumper.flush(
        {
            "ships_full_frame": ship_boxes,
            "debris_full_frame": debris_boxes,
            "ship_cells": sorted(ship_cells),
            "debris_cells": sorted(debris_cells),
        }
    )
    return ship_cells, debris_cells


def _cell_patch_features(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    patch_radius: int | None = None,
):
    """Yield ``((row, col), centre_s_mean, centre_gray_std, centre_v_mean)`` for every cell."""
    if grid_size <= 0 or not click_points:
        return
    h, w = grid_img.shape[:2]
    if w <= 0 or h <= 0:
        return
    radius = patch_radius if patch_radius is not None else int(round(30 * (w / float(REFERENCE_FRAME[0]))))
    if radius <= 2:
        return
    hsv = cv2.cvtColor(grid_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for index, (px, py) in enumerate(click_points):
        row, col = divmod(int(index), int(grid_size))
        if row >= int(grid_size) or col >= int(grid_size):
            continue
        cx, cy = int(px), int(py)
        top = max(0, cy - radius)
        bottom = min(h, cy + radius)
        left = max(0, cx - radius)
        right = min(w, cx + radius)
        if bottom <= top or right <= left:
            continue
        s_patch = sat[top:bottom, left:right]
        v_patch = val[top:bottom, left:right]
        g_patch = gray[top:bottom, left:right]
        if s_patch.size < 20:
            continue
        ph, pw = s_patch.shape
        s_center = s_patch[ph // 4 : 3 * ph // 4, pw // 4 : 3 * pw // 4]
        v_center = v_patch[ph // 4 : 3 * ph // 4, pw // 4 : 3 * pw // 4]
        g_center = g_patch[ph // 4 : 3 * ph // 4, pw // 4 : 3 * pw // 4]
        yield (row, col), float(s_center.mean()), float(g_center.std()), float(v_center.mean())


def _cell_polygon(
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    index: int,
    grid_size: int,
) -> np.ndarray | None:
    """Return the four corners of one cell's diamond in frame coordinates."""
    if grid_size <= 0 or index < 0 or index >= len(click_points):
        return None
    row, col = divmod(int(index), int(grid_size))
    p = np.asarray(click_points[index], dtype=np.float32)
    if col + 1 < grid_size:
        right = np.asarray(click_points[index + 1], dtype=np.float32) - p
    elif col > 0:
        right = p - np.asarray(click_points[index - 1], dtype=np.float32)
    else:
        return None
    if row + 1 < grid_size:
        down = np.asarray(click_points[index + grid_size], dtype=np.float32) - p
    elif row > 0:
        down = p - np.asarray(click_points[index - grid_size], dtype=np.float32)
    else:
        return None
    return np.asarray(
        [
            p - right * 0.5 - down * 0.5,
            p + right * 0.5 - down * 0.5,
            p + right * 0.5 + down * 0.5,
            p - right * 0.5 + down * 0.5,
        ],
        dtype=np.float32,
    )


def _cell_step(
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
) -> float:
    """Median distance between adjacent cell centres, in pixels."""
    if grid_size <= 1 or len(click_points) < 2:
        return 44.0
    deltas: list[float] = []
    for index in range(len(click_points)):
        row, col = divmod(int(index), int(grid_size))
        px, py = click_points[index]
        if col + 1 < grid_size:
            qx, qy = click_points[index + 1]
            deltas.append(float(np.hypot(qx - px, qy - py)))
        if row + 1 < grid_size:
            qx, qy = click_points[index + grid_size]
            deltas.append(float(np.hypot(qx - px, qy - py)))
    return float(np.median(deltas)) if deltas else 44.0


def _odd_kernel_size(value: float) -> int:
    size = int(round(float(value)))
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    return size


@dataclass(frozen=True)
class BoardSprite:
    """One frame-level grey sprite (submarine hull or rubble pile).

    ``cells`` maps every cell diamond the sprite covers to its coverage ratio
    (sprite pixels / cell area).  ``has_red_tower`` is True when a red component
    (the submarine conning tower) touches the sprite.
    """

    label: int
    area: int
    fill_ratio: float
    texture: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    cells: tuple[tuple[tuple[int, int], float], ...]
    has_red_tower: bool


def _cell_label_map(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise every cell diamond into a label map plus per-cell pixel counts."""
    h, w = grid_img.shape[:2]
    count = int(grid_size) * int(grid_size)
    label_map = np.zeros((h, w), dtype=np.int32)
    for index in range(count):
        polygon = _cell_polygon(click_points, index, grid_size)
        if polygon is None:
            continue
        cv2.fillPoly(label_map, [np.round(polygon).astype(np.int32)], index + 1)
    areas = np.bincount(label_map.ravel(), minlength=count + 1)
    return label_map, areas


def detect_board_sprites(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    *,
    sat_max: float = _GRAY_BLOB_SAT_MAX,
    val_min: float = _GRAY_BLOB_VAL_MIN,
    open_ksize: int | None = None,
    min_area_ratio: float = 0.03,
    min_fill: float = 0.35,
    hull_reach_steps: float = 2.0,
    texture_threshold: float = _WRECK_TEXTURE_MIN,
) -> list[BoardSprite]:
    """Detect grey sprites on the board as whole-frame connected components.

    Per-cell patches bleed into their neighbours (a cell diamond is only ~74x46
    at 1280x720 while a patch used to be 60x60), which made one submarine look
    like five wrecks.  Working on the full frame instead keeps each sprite
    intact: a submarine hull plus its red conning tower is ONE component, while
    every rubble pile is its own compact component.

    Only components that cover at least ``min_area_ratio`` of a cell diamond and
    are at least ``min_fill`` solid are kept; that removes the white grid lines
    and the thin bright foam streaks of waves.  ``texture`` is the mean absolute
    Laplacian inside the component: a rubble pile is full of hard fragments and
    dark shadows, while foam and hull plating are smooth.  A submarine's own
    hull, wake and detached hull fragments are pulled into the ship by
    ``hull_reach_steps`` before any wreck decision is made.
    """
    sprites: list[BoardSprite] = []
    if not isinstance(grid_img, np.ndarray) or grid_img.ndim < 3:
        return sprites
    if grid_size <= 0 or len(click_points) < grid_size * grid_size:
        return sprites
    h, w = grid_img.shape[:2]
    if h <= 0 or w <= 0:
        return sprites
    label_map, cell_areas = _cell_label_map(grid_img, click_points, grid_size)
    board_mask = (label_map > 0).astype(np.uint8)
    if not board_mask.any():
        return sprites
    kernel_size = _odd_kernel_size(
        open_ksize if open_ksize is not None else _cell_step(click_points, grid_size) * _GRAY_BLOB_OPEN_RATIO
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    hsv = cv2.cvtColor(grid_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0].astype(np.int16)
    grey = ((sat < sat_max) & (val > val_min) & (board_mask > 0)).astype(np.uint8)
    grey_gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    laplacian = np.abs(cv2.Laplacian(grey_gray, cv2.CV_32F))
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
    grey &= board_mask
    if not grey.any():
        return sprites
    red = (
        ((hue < _RED_TOWER_HUE_MAX) | (hue > _RED_TOWER_HUE_MIN))
        & (sat > _RED_TOWER_SAT_MIN)
        & (val > _RED_TOWER_VAL_MIN)
    ).astype(np.uint8)
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel)
    grey &= board_mask
    if not grey.any():
        return sprites
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(grey, 8)
    # 红塔可能被暗部阴影和艇身断开，所以不要求它和艇身像素直接相连：把每个红塔
    # 向外扩一点，落在它下面的那个灰白连通块就是它所属的潜艇。潜艇的艇身还会被
    # 弹痕暗斑切成好几块，因此和"带红塔的块"相距不超过约 1.5 格心间距的灰白块
    # 也算同一艘艇。
    tower_labels: set[int] = set()
    reach = _odd_kernel_size(_cell_step(click_points, grid_size) * 0.7)
    if red.any():
        reach_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))
        red_count, _red_labels, red_stats, _red_centroids = cv2.connectedComponentsWithStats(
            red, 8
        )
        for red_label in range(1, red_count):
            if int(red_stats[red_label, cv2.CC_STAT_AREA]) < _RED_TOWER_AREA_MIN:
                continue
            rx = int(red_stats[red_label, cv2.CC_STAT_LEFT])
            ry = int(red_stats[red_label, cv2.CC_STAT_TOP])
            rw = int(red_stats[red_label, cv2.CC_STAT_WIDTH])
            rh = int(red_stats[red_label, cv2.CC_STAT_HEIGHT])
            pad = reach
            x0 = max(0, rx - pad)
            y0 = max(0, ry - pad)
            x1 = min(w, rx + rw + pad)
            y1 = min(h, ry + rh + pad)
            local_red = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            local_red[
                ry - y0 : ry - y0 + rh, rx - x0 : rx - x0 + rw
            ] = (red[ry : ry + rh, rx : rx + rw] > 0).astype(np.uint8)
            local_red = cv2.dilate(local_red, reach_kernel, iterations=1)
            underneath = labels[y0:y1, x0:x1][local_red > 0]
            underneath = underneath[underneath > 0]
            if underneath.size == 0:
                continue
            tower_labels.add(int(np.bincount(underneath).argmax()))
    if tower_labels:
        # 把和"带红塔的块"挨得足够近的块也算成同一艘潜艇。但只吸收平滑的艇身碎片/
        # 尾迹（纹理低于残骸阈值上限）：一块高纹理的独立残骸堆即使落在红塔附近，也
        # 只是被炸沉的残骸，不是潜艇艇身，不能据此把它从"已确认残骸"里剔除。
        hull_reach = _odd_kernel_size(_cell_step(click_points, grid_size) * hull_reach_steps)
        hull_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hull_reach, hull_reach))
        tower_mask = np.isin(labels, list(tower_labels)).astype(np.uint8)
        grown = cv2.dilate(tower_mask, hull_kernel, iterations=1)
        nearby = np.unique(labels[grown > 0])
        texture_floor = float(texture_threshold) * max(w / float(REFERENCE_FRAME[0]), 0.25)
        absorb_floor = texture_floor * 1.25
        for item in nearby:
            label = int(item)
            if label <= 0:
                continue
            component_texture = float(laplacian[labels == label].mean())
            if component_texture >= absorb_floor:
                continue
            tower_labels.add(label)
    board_cells = int(grid_size) * int(grid_size)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill = area / float(max(1, bw * bh))
        if fill < min_fill:
            continue
        component = labels == label
        covered = label_map[component]
        if covered.size == 0:
            continue
        counts = np.bincount(covered, minlength=board_cells + 1)
        cell_hits: list[tuple[tuple[int, int], float]] = []
        for index in range(1, board_cells + 1):
            cell_area = int(cell_areas[index])
            if cell_area <= 0:
                continue
            coverage = float(counts[index]) / float(cell_area)
            if coverage < min_area_ratio:
                continue
            cell_hits.append((divmod(index - 1, int(grid_size)), coverage))
        if not cell_hits:
            continue
        sprites.append(
            BoardSprite(
                label=label,
                area=area,
                fill_ratio=float(fill),
                texture=float(laplacian[component].mean()),
                bbox=(
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    bw,
                    bh,
                ),
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
                cells=tuple(cell_hits),
                has_red_tower=label in tower_labels,
            )
        )
    return sprites


def classify_cells_by_rules(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    *,
    coverage_threshold: float = 0.10,
    wreck_coverage_threshold: float = 0.10,
    texture_threshold: float = _WRECK_TEXTURE_MIN,
    hull_reach_steps: float = 2.0,
    sat_max: float = _GRAY_BLOB_SAT_MAX,
    val_min: float = _GRAY_BLOB_VAL_MIN,
    open_ksize: int | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Split the board into ``(ship_cells, wreck_cells)`` with the rule criteria.

    * ``ship_cells``: cells covered by a grey sprite that carries a red conning
      tower (a whole submarine, hull plus wake).
    * ``wreck_cells``: cells covered by a compact, textured grey sprite with no
      red tower.

    Waves are excluded by construction: foam streaks are thin, the opening
    removes them, and any leftover is neither compact nor large enough to cover
    a tenth of a cell diamond.  Anything smooth (thick wake foam, hull plating)
    is rejected by the texture floor, and every grey blob within
    ``hull_reach_steps`` cell steps of a red tower belongs to that submarine
    instead of being reported as rubble.
    """
    ships: set[tuple[int, int]] = set()
    wrecks: set[tuple[int, int]] = set()
    if not isinstance(grid_img, np.ndarray) or grid_img.ndim < 3:
        return ships, wrecks
    width = float(grid_img.shape[1])
    texture_floor = float(texture_threshold) * max(width / float(REFERENCE_FRAME[0]), 0.25)
    for sprite in detect_board_sprites(
        grid_img,
        click_points,
        grid_size,
        sat_max=sat_max,
        val_min=val_min,
        open_ksize=open_ksize,
        hull_reach_steps=hull_reach_steps,
    ):
        if sprite.has_red_tower:
            for cell, coverage in sprite.cells:
                if coverage >= coverage_threshold:
                    ships.add(cell)
            continue
        if sprite.texture < texture_floor:
            continue
        for cell, coverage in sprite.cells:
            if coverage >= wreck_coverage_threshold:
                wrecks.add(cell)
    return ships, wrecks


def detect_wreck_cells_by_features(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    blob_area_threshold: float = 0.10,
    blob_fill_threshold: float = 0.35,
    texture_threshold: float = _WRECK_TEXTURE_MIN,
    sat_max: float = _GRAY_BLOB_SAT_MAX,
    val_min: float = _GRAY_BLOB_VAL_MIN,
    open_ksize: int | None = None,
) -> set[tuple[int, int]]:
    """Per-cell static wreck detection from whole-frame grey sprites.

    残骸 = 一块紧凑、带纹理的灰白实体：先按"低饱和度 + 中高亮度"取掩码，再用
    形态学开运算滤掉白色网格线和细长的海浪泡沫，然后按整帧连通域取块。实测
    （1280x720，格心间距 44px）真残骸会占据所在格菱形 10% 以上面积、填充率
    ≥35%、块内平均 |Laplacian| ≥30；高亮海浪只剩细条纹，开运算后连 3% 都占
    不到，潜艇尾迹虽然也是厚泡沫但很平滑，而且离红塔不超过 2 格心间距，
    会先被归到潜艇，两者因此都不会被当成残骸。

    潜艇艇身同样是灰白实体，但它和红色指挥塔连成同一个连通块，这里整块排除，
    由 :func:`classify_cells_by_rules` 的 ship 分支输出。
    """
    _ships, wrecks = classify_cells_by_rules(
        grid_img,
        click_points,
        grid_size,
        coverage_threshold=max(blob_area_threshold, 0.0),
        texture_threshold=texture_threshold,
        sat_max=sat_max,
        val_min=val_min,
        open_ksize=open_ksize,
    )
    return wrecks


def detect_ship_cells_by_red_tower(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    *,
    coverage_threshold: float = 0.10,
    open_ksize: int | None = None,
) -> set[tuple[int, int]]:
    """Per-cell submarine detection: every cell of a red-tower grey sprite."""
    ships, _wrecks = classify_cells_by_rules(
        grid_img,
        click_points,
        grid_size,
        coverage_threshold=coverage_threshold,
        open_ksize=open_ksize,
    )
    return ships


def detect_water_cells_by_features(
    grid_img: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    sat_threshold: float = 150.0,
    texture_threshold: float = 12.0,
    patch_radius: int | None = None,
) -> set[tuple[int, int]]:
    """Per-cell definite-water detection (high saturation, low texture).

    Used as the PRIMARY guard in the wreck review: a cell that looks like pure
    deep water cannot be a wreck, so any template match on such a cell is
    rejected.
    """
    cells: set[tuple[int, int]] = set()
    for cell, s_mean, g_std, _v_mean in _cell_patch_features(
        grid_img, click_points, grid_size, patch_radius
    ):
        if s_mean > sat_threshold and g_std < texture_threshold:
            cells.add(cell)
    return cells


# --------------------------------------------------------------------------
# 残骸 / 潜艇艇身 / 海浪 / 深水 四类地表判定
# --------------------------------------------------------------------------
# 口径很重要：必须以**格心方形 patch**取样。游戏对象都画在格心，patch 覆盖对象
# 本体；若改用整格菱形，周围海水会混进来，残骸的饱和度被拉到和海水一样（~121），
# 灰/蓝之分就被冲掉了。
#
# 主判据是「最大灰白连通域占比」(coverage)——也就是对象的**不透明度**：
#   残骸/艇身是不透明精灵、几乎填满整格   coverage 0.66~0.94
#   海浪泡沫是半透明薄层盖在蓝水上         coverage 0.32~0.39
#   深水几乎没有灰白像素                   coverage 0.01~0.03
# 三者间隔达 0.27，比「饱和度/白沫」这类颜色统计量稳得多（后者的分界只有 0.015）。
#
# 阈值由 level-20/22/25 帧标定（每格中心 patch rad=16~18）：
#   残骸  coverage 0.66~0.68  dark 0.004~0.011  white 0.38~0.41
#   艇身  coverage 0.72~0.94  dark 0.025~0.201  white 0.14~0.44
#   海浪  coverage 0.32~0.39  dark 0.019~0.046  white 0.11~0.20
#   深水  coverage 0.01~0.03  dark 0.000~0.050  white 0.000~0.084
#
# 判别逻辑：先用 coverage 把「不透明实体(残骸/艇身)」与「泡沫/水面」分开；
# 实体内用暗部占比区分残骸与艇壳；泡沫/水面内用白沫占比区分海浪与深水。
SURFACE_PATCH_RADIUS = 16
SURFACE_COVERAGE_MIN = 0.50
# 覆盖率低于此值的候选格几乎没有灰白像素（纯水面）：端点延伸常把这类格并进直排
# （level 22 的 (0,1) 覆盖率 0.00），使真正的「长4」潜艇看起来是「长5」。而暗淡的
# 真艇身端点仍有 0.3+ 的覆盖率（(4,1)=0.38），不会被误删。
SURFACE_COVERAGE_SPURIOUS_MAX = 0.05
SURFACE_COVERAGE_SAT_MAX = 80.0
SURFACE_COVERAGE_VAL_MIN = 100.0
SURFACE_DARK_THRESHOLD = 0.015
SURFACE_WHITE_THRESHOLD = 0.09
SURFACE_DARK_VALUE_MAX = 90.0
SURFACE_WHITE_SAT_MAX = 65.0
SURFACE_WHITE_VALUE_MIN = 175.0
# 整块 patch 太暗（黑帧/过渡帧/UI 遮罩）时无法判断地表：返回「未判定」而不是水，
# 否则会把残骸误剔。真实水面/对象的 patch 平均亮度在 150 以上。
SURFACE_MIN_MEAN_VALUE = 80.0

SURFACE_WRECK = "wreck"
SURFACE_HULL = "hull"
SURFACE_WAVE = "wave"
SURFACE_WATER = "water"


def _largest_component_ratio(mask: np.ndarray) -> float:
    """灰白掩码开运算后，最大连通域占整块 patch 的比例（对象不透明度）。"""
    if mask.size == 0 or not mask.any():
        return 0.0
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    if not opened.any():
        return 0.0
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    largest = 0
    for index in range(1, count):
        largest = max(largest, int(stats[index, cv2.CC_STAT_AREA]))
    return largest / float(mask.size)


def surface_patch_features(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    patch_radius: int = SURFACE_PATCH_RADIUS,
) -> tuple[float, float, float, float] | None:
    """格心方形 patch 的 (饱和度均值, 暗部占比, 白沫占比, 最大灰白连通域占比)。"""
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return None
    height, width = image.shape[:2]
    cx, cy = int(point[0]), int(point[1])
    radius = max(2, int(patch_radius))
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius + 1)
    y2 = min(height, cy + radius + 1)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    if float(value.mean()) < SURFACE_MIN_MEAN_VALUE:
        # 黑帧/过渡帧：无法判断地表，交给调用方当作「未判定」。
        return None
    count = max(1, saturation.size)
    dark_ratio = float(np.count_nonzero(value < SURFACE_DARK_VALUE_MAX)) / count
    white_ratio = (
        float(
            np.count_nonzero(
                (saturation < SURFACE_WHITE_SAT_MAX) & (value > SURFACE_WHITE_VALUE_MIN)
            )
        )
        / count
    )
    gray_mask = (
        (saturation < SURFACE_COVERAGE_SAT_MAX) & (value > SURFACE_COVERAGE_VAL_MIN)
    ).astype(np.uint8)
    coverage_ratio = _largest_component_ratio(gray_mask)
    return float(saturation.mean()), dark_ratio, white_ratio, coverage_ratio


def classify_surface_cell(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    patch_radius: int = SURFACE_PATCH_RADIUS,
) -> str | None:
    """把一个格子判成 wreck / hull / wave / water 之一。

    ``None`` 表示图像或取样无效，调用方应视为「未判定」而不是水。
    """
    features = surface_patch_features(image, point, patch_radius=patch_radius)
    if features is None:
        return None
    _saturation, dark_ratio, white_ratio, coverage_ratio = features
    if coverage_ratio >= SURFACE_COVERAGE_MIN:
        return SURFACE_WRECK if dark_ratio < SURFACE_DARK_THRESHOLD else SURFACE_HULL
    return SURFACE_WAVE if white_ratio > SURFACE_WHITE_THRESHOLD else SURFACE_WATER


def classify_surface_cells(
    image: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    *,
    patch_radius: int = SURFACE_PATCH_RADIUS,
) -> dict[tuple[int, int], str]:
    """对整块棋盘逐格做四类地表判定，返回 {cell: 类别}。"""
    classes: dict[tuple[int, int], str] = {}
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return classes
    if grid_size <= 0 or len(click_points) != grid_size * grid_size:
        return classes
    for index, point in enumerate(click_points):
        row, col = divmod(index, grid_size)
        label = classify_surface_cell(image, (int(point[0]), int(point[1])), patch_radius=patch_radius)
        if label is not None:
            classes[(row, col)] = label
    return classes


def surface_coverage_map(
    image: np.ndarray,
    click_points: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    grid_size: int,
    *,
    patch_radius: int = SURFACE_PATCH_RADIUS,
) -> dict[tuple[int, int], float]:
    """逐格的灰白覆盖率（对象不透明度），用于剔除「几乎没有灰白像素」的假候选。"""
    coverage: dict[tuple[int, int], float] = {}
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        return coverage
    if grid_size <= 0 or len(click_points) != grid_size * grid_size:
        return coverage
    for index, point in enumerate(click_points):
        features = surface_patch_features(
            image,
            (int(point[0]), int(point[1])),
            patch_radius=patch_radius,
        )
        if features is None:
            continue
        row, col = divmod(index, grid_size)
        coverage[(row, col)] = features[3]
    return coverage
