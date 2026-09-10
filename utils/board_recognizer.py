"""Runtime inference for the binarized board-cell recognizer (二值法识图).

与 ``tools/train_board_real.py`` 保持完全一致的特征提取：整幅图 Otsu 反色二值化
→ 裁剪固定棋盘区域 → 用 quad 映射到裁剪坐标系 → 逐格取出 FEATURE_SIZE 二值特征
→ 用 ``board_cell_model.json`` 的 2 层 MLP 预测每格是水(0)还是潜艇(1)。

用法::

    from utils.board_recognizer import load_board_model, classify_board_cells

    model = load_board_model()                      # 无模型时返回 None
    cells = classify_board_cells(rgb_img, quad, grid_size, model)   # -> set[(row, col)]
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from utils.image_io import read_image_compat  # noqa: F401  (shared unicode-path helper)

FEATURE_SIZE = 32
CROP_MARGIN = 12                  # 棋盘 quad 包围盒外围边距
PAD = 6                           # 与 train_board_real 一致

# 特征模式：模型自带，推理按模型决定用哪种特征。
FEATURE_MODE_BINARY = "binary"    # 整帧 Otsu 反色二值化后取 (crop<128)
FEATURE_MODE_GRAY = "gray"        # 原图灰度归一化到 [0,1]（原图训练，信息更丰富）

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "识图" / "board_cell_model.json"


def prepare_frame(img: np.ndarray, mode: str = FEATURE_MODE_BINARY) -> np.ndarray:
    """按特征模式把整帧转成后续裁剪用的图：二值图或灰度图。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if mode == FEATURE_MODE_GRAY:
        return gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def cell_feature(crop: np.ndarray, mode: str = FEATURE_MODE_BINARY) -> np.ndarray:
    """把一个格子裁剪缩放成特征向量（训练与推理必须用同一个 mode）。"""
    box = cv2.resize(crop, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    if mode == FEATURE_MODE_GRAY:
        return (box.astype(np.float32) / 255.0).reshape(-1)
    return (box < 128).astype(np.float32).reshape(-1)


class BoardCellMLP:
    """2 层网络（input -> hidden relu -> softmax），与训练工具同构。

    ``feature_mode`` 决定输入特征，训练脚本写进模型 json，推理时按它自动选择，
    保证训练/推理特征一致。
    """

    def __init__(self, W1, b1, W2, b2, feature_mode: str = FEATURE_MODE_BINARY):
        self.W1 = np.asarray(W1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.W2 = np.asarray(W2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)
        self.feature_mode = (
            feature_mode if feature_mode in {FEATURE_MODE_BINARY, FEATURE_MODE_GRAY} else FEATURE_MODE_BINARY
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        a1 = np.maximum(0.0, X @ self.W1 + self.b1)
        z2 = a1 @ self.W2 + self.b2
        z2 = z2 - z2.max(axis=1, keepdims=True)
        e = np.exp(z2)
        p = e / e.sum(axis=1, keepdims=True)
        return p.argmax(1)


def load_board_model(path: str | Path | None = None) -> BoardCellMLP | None:
    """加载训练好的 ``board_cell_model.json``；文件不存在或格式错误时返回 None。"""
    weight_path = Path(path) if path is not None else DEFAULT_WEIGHTS
    if not weight_path.exists():
        return None
    try:
        import json

        data = json.loads(weight_path.read_text(encoding="utf-8"))
        model = BoardCellMLP(
            data["W1"],
            data["b1"],
            data["W2"],
            data["b2"],
            feature_mode=str(data.get("feature_mode", FEATURE_MODE_BINARY)),
        )
    except (OSError, ValueError, KeyError, TypeError, SyntaxError):
        return None
    if model.W1.shape[0] != FEATURE_SIZE * FEATURE_SIZE:
        return None
    return model


def binarize_full(img: np.ndarray) -> np.ndarray:
    """整幅 Otsu 反色二值化（与 ``_capture_training_sample`` 一致）。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def quad_crop_box(quad, frame_shape, margin: int = CROP_MARGIN) -> tuple[int, int, int, int]:
    """取棋盘 quad 的包围盒（加边距），作为二值图裁剪区域，避免固定区域裁掉棋盘边缘。"""
    height, width = frame_shape[:2]
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    x0 = max(0, int(np.floor(min(xs))) - margin)
    y0 = max(0, int(np.floor(min(ys))) - margin)
    x1 = min(width, int(np.ceil(max(xs))) + margin)
    y1 = min(height, int(np.ceil(max(ys))) + margin)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("quad 包围盒超出图像范围")
    return x0, y0, x1, y1


def cell_centers_perspective(quad_in_crop, n: int) -> tuple[list[tuple[float, float]], np.ndarray]:
    """用与大网格中心点完全一致的透视变换，把 n×n 格中心映射到裁剪坐标系。

    游戏棋盘是透视投影：四角包围盒的均匀均分会让靠边格子的中心错位。
    这里和 ``utils.diamond_centers.centers_from_quad`` 使用同一套
    ``getPerspectiveTransform``，保证"实际位置=棋盘格子位置"。
    """
    src = np.array(
        [[0, 0], [n, 0], [n, n], [0, n]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(
        src,
        np.asarray(quad_in_crop, dtype=np.float32).reshape(4, 2),
    )
    centers: list[tuple[float, float]] = []
    for row in range(n):
        for col in range(n):
            point = np.array([[[col + 0.5, row + 0.5]]], dtype=np.float32)
            q = cv2.perspectiveTransform(point, matrix)[0, 0]
            centers.append((float(q[0]), float(q[1])))
    return centers, matrix


def cell_feature_boxes(
    quad,
    grid_size: int,
    frame_shape,
    pad: int = PAD,
) -> list[tuple[tuple[int, int], tuple[int, int, int, int]]]:
    """逐格特征框（**全帧坐标**），训练与推理共用。

    棋盘是透视投影，所以格子中心走 ``getPerspectiveTransform``；每个格子的特征框
    是以格心为中心、宽 ``棋盘宽/n`` 高 ``棋盘高/n`` 再各边外扩 ``pad`` 的矩形。
    返回 ``[((row, col), (left, top, right, bottom))]``。
    """
    n = int(grid_size)
    if n <= 0:
        return []
    x0, y0, x1, y1 = quad_crop_box(quad, frame_shape)
    quad_crop = np.asarray(quad, dtype=np.float32) - np.array([x0, y0], dtype=np.float32)
    centers, _matrix = cell_centers_perspective(quad_crop, n)
    qx0 = min(p[0] for p in quad_crop)
    qy0 = min(p[1] for p in quad_crop)
    w = max(p[0] for p in quad_crop) - qx0
    h = max(p[1] for p in quad_crop) - qy0
    if w <= 0 or h <= 0:
        return []
    a = w / n
    b = h / n
    height, width = frame_shape[:2]
    boxes: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []
    for index, (cx, cy) in enumerate(centers):
        row, col = divmod(index, n)
        left = int(max(0, cx - a / 2 - pad + x0))
        right = int(min(width, cx + a / 2 + pad + x0))
        top = int(max(0, cy - b / 2 - pad + y0))
        bottom = int(min(height, cy + b / 2 + pad + y0))
        if right <= left or bottom <= top:
            continue
        boxes.append(((row, col), (left, top, right, bottom)))
    return boxes


def extract_cell_features(
    img: np.ndarray,
    quad,
    grid_size: int,
    mode: str = FEATURE_MODE_BINARY,
    pad: int = PAD,
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """整帧按 ``mode`` 预处理后逐格取特征，返回 ``[((row, col), 特征向量)]``。

    训练脚本和线上推理都走这个函数，避免"训练用一套特征、推理用另一套"。
    """
    prepared = prepare_frame(img, mode)
    features: list[tuple[tuple[int, int], np.ndarray]] = []
    for cell, (left, top, right, bottom) in cell_feature_boxes(
        quad, grid_size, prepared.shape, pad
    ):
        box = prepared[top:bottom, left:right]
        if box.size == 0:
            continue
        features.append((cell, cell_feature(box, mode)))
    return features


def classify_board_cells(
    rgb_img: np.ndarray,
    quad,
    grid_size: int,
    model: BoardCellMLP,
) -> set[tuple[int, int]]:
    """对整幅图 binarize→按 quad 包围盒裁剪→透视逐格 MLP 预测，返回"潜艇内容"格子集合。"""
    if model is None:
        return set()
    n = int(grid_size)
    if n <= 0:
        return set()
    return {
        cell
        for cell, cls in _classify_board_cells_raw(rgb_img, quad, n, model)
        if cls in (1, 2)          # 1=潜艇 2=残骸；3=海浪不算内容
    }


def classify_board_cells_classes(
    rgb_img: np.ndarray,
    quad,
    grid_size: int,
    model: BoardCellMLP,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """3 类模型：返回 ``(潜艇格, 残骸格)``；2 类模型时残骸格为空集。"""
    if model is None:
        return set(), set()
    n = int(grid_size)
    if n <= 0:
        return set(), set()
    raw = _classify_board_cells_raw(rgb_img, quad, n, model)
    subs = {cell for cell, cls in raw if cls == 1}
    debris = {cell for cell, cls in raw if cls == 2}
    return subs, debris


def _classify_board_cells_raw(
    rgb_img: np.ndarray,
    quad,
    n: int,
    model: BoardCellMLP,
) -> list[tuple[tuple[int, int], int]]:
    """逐格预测，返回 ``[((row, col), class)]``，class：0=水 / 1=潜艇 / 2=残骸。"""
    mode = getattr(model, "feature_mode", FEATURE_MODE_BINARY)
    features = extract_cell_features(rgb_img, quad, n, mode)
    if not features:
        return []
    cells = [cell for cell, _ in features]
    pred = model.predict(np.stack([feat for _, feat in features]).astype(np.float32))
    return [(cell, int(p)) for cell, p in zip(cells, pred)]


def classify_cells_at_points(
    img: np.ndarray,
    points,
    grid_size: int,
    model: BoardCellMLP,
    pad: int = PAD,
) -> set[tuple[int, int]]:
    """按"点击点=格中心"逐格二值法分类，返回判为"潜艇内容"的格子。

    与 ``classify_board_cells`` 同源特征（整幅 Otsu 反色二值化 + 32×32 特征 + MLP），
    只是用现成的点击点定位格子（红侦察/蓝炮路径里已有 click_points）。
    取格半宽/半高按训练时的 ``a=w/n, b=h/n`` 反推：格子中心坐标的外接范围
    跨 (n-1) 格，故半宽 = (max_x-min_x)/(2*(n-1))，与训练框一致。
    """
    if model is None or not points:
        return set()
    n = int(grid_size)
    if n <= 0:
        return set()

    mode = getattr(model, "feature_mode", FEATURE_MODE_BINARY)
    binary = prepare_frame(img, mode)
    height, width = binary.shape[:2]

    px = [float(p[0]) for p in points]
    py = [float(p[1]) for p in points]
    denom = 2.0 * max(1, n - 1)
    half_w = (max(px) - min(px)) / denom if n > 1 else 42.0
    half_h = (max(py) - min(py)) / denom if n > 1 else 26.0

    feats: list[np.ndarray] = []
    cell_inds: list[tuple[int, int]] = []
    for index, (cx, cy) in enumerate(points):
        row, col = index // n, index % n
        left = int(max(0, cx - half_w - pad))
        right = int(min(width, cx + half_w + pad))
        top = int(max(0, cy - half_h - pad))
        bottom = int(min(height, cy + half_h + pad))
        if right <= left or bottom <= top:
            continue
        box = binary[top:bottom, left:right]
        feats.append(cell_feature(box, mode))
        cell_inds.append((row, col))

    if not feats:
        return set()
    pred = model.predict(np.stack(feats).astype(np.float32))
    return {
        cell
        for cell, p in zip(cell_inds, pred)
        if int(p) in (1, 2)       # 1=潜艇 2=残骸；0=水 3=海浪都不算内容
    }


def binarize_reveal_cells(
    before_img: np.ndarray,
    after_img: np.ndarray,
    points,
    grid_size: int,
    model: BoardCellMLP,
) -> set[tuple[int, int]]:
    """前后帧二值法差分：返回"打点后新出现潜艇内容"的格子（=揭示/命中）。"""
    before_cells = classify_cells_at_points(before_img, points, grid_size, model)
    after_cells = classify_cells_at_points(after_img, points, grid_size, model)
    return after_cells - before_cells


def binarize_cell_is_submarine(
    img: np.ndarray,
    cell_polygon,
    model: BoardCellMLP,
    pad: int = PAD,
) -> bool:
    """用二值法判断"某个格子（由 cell_polygon 给出）是否含潜艇内容"。"""
    if model is None or cell_polygon is None:
        return False
    poly = np.asarray(cell_polygon, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 2:
        return False
    mode = getattr(model, "feature_mode", FEATURE_MODE_BINARY)
    binary = prepare_frame(img, mode)
    height, width = binary.shape[:2]
    x0 = int(max(0, poly[:, 0].min() - pad))
    x1 = int(min(width, poly[:, 0].max() + pad))
    y0 = int(max(0, poly[:, 1].min() - pad))
    y1 = int(min(height, poly[:, 1].max() + pad))
    if x1 <= x0 or y1 <= y0:
        return False
    box = binary[y0:y1, x0:x1]
    feat = cell_feature(box, mode).reshape(1, -1)
    return int(model.predict(feat)[0]) in (1, 2)


def binarize_cell_hit(
    before_img: np.ndarray,
    after_img: np.ndarray,
    cell_polygon,
    model: BoardCellMLP,
) -> bool:
    """前后帧二值法差分（单格）：打点后该格新出现潜艇内容 → 命中。"""
    if model is None or cell_polygon is None:
        return False
    before_sub = binarize_cell_is_submarine(before_img, cell_polygon, model)
    after_sub = binarize_cell_is_submarine(after_img, cell_polygon, model)
    return bool(after_sub and not before_sub)
