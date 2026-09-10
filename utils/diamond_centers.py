import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.image_io import read_image_compat, write_image_compat
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GridDetection:
    """菱形网格外框检测结果，包含调试时需要的中间信息。"""

    quad: np.ndarray
    component_area: float
    component_bbox: tuple[int, int, int, int]
    angle_pos: float
    angle_neg: float
    orientation_mask: np.ndarray
    component_mask: np.ndarray


@dataclass
class DiamondCentersResult:
    """菱形网格中心点检测结果，供 CLI 和调试流程复用。"""

    points: list[tuple[int, int]]
    float_points: list[tuple[float, float]]
    detection: GridDetection
    local_quad: np.ndarray
    global_quad: np.ndarray
    offset_x: int
    offset_y: int


def find_diamond_centers(
    screenshot: np.ndarray,
    n: int,
    roi: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]]:
    """从截图中识别菱形网格中心点，返回可直接点击的整数坐标。"""
    return detect_diamond_centers(screenshot, n, roi).points


def detect_diamond_centers(
    screenshot: np.ndarray,
    n: int,
    roi: tuple[int, int, int, int] | None = None,
) -> DiamondCentersResult:
    """检测菱形网格中心点，并返回外框、角度和调试图等详细信息。"""
    _validate_screenshot(screenshot)
    n = _validate_grid_size(n)

    work_img, offset_x, offset_y = apply_roi(screenshot, roi)
    detection = detect_grid_quad(work_img)
    local_points = centers_from_quad(detection.quad, n)

    global_float_points = [
        (
            x + offset_x,
            y + offset_y,
        )
        for x, y in local_points
    ]
    points = [
        (
            int(round(x)),
            int(round(y)),
        )
        for x, y in global_float_points
    ]

    global_quad = detection.quad.copy()
    global_quad[:, 0] += offset_x
    global_quad[:, 1] += offset_y

    return DiamondCentersResult(
        points=points,
        float_points=global_float_points,
        detection=detection,
        local_quad=detection.quad.copy(),
        global_quad=global_quad,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def read_image(path: str | Path) -> np.ndarray:
    """读取图片文件，兼容中文路径。"""
    img = read_image_compat(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    """保存图片文件，兼容中文路径。"""
    if not write_image_compat(path, img):
        raise RuntimeError(f"无法保存图片：{path}")


def apply_roi(
    img: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, int, int]:
    """裁剪 ROI，并返回裁剪图和原图偏移。"""
    if roi is None:
        return img.copy(), 0, 0

    x, y, width, height = roi
    img_h, img_w = img.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + width)
    y2 = min(img_h, y + height)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI 无效或越界")

    return img[y1:y2, x1:x2].copy(), x1, y1


def angle_diff(a: np.ndarray | float, b: float) -> np.ndarray | float:
    """计算线方向角差，角度周期为 180 度，返回 0 到 90 范围内的差值。"""
    d = (a - b + 90) % 180 - 90
    return np.abs(d)


def make_white_mask(img: np.ndarray, mode: str = "loose") -> np.ndarray:
    """提取白色边框或淡白填充区域。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    channel_a = lab[:, :, 1].astype(np.int16)
    channel_b = lab[:, :, 2].astype(np.int16)

    if mode == "strict":
        mask_hsv = ((s <= 105) & (v >= 115)).astype(np.uint8) * 255
        mask_lab = (
            (lightness >= 125)
            & (np.abs(channel_a - 128) <= 35)
            & (np.abs(channel_b - 128) <= 48)
        ).astype(np.uint8) * 255
    else:
        mask_hsv = ((s <= 130) & (v >= 90)).astype(np.uint8) * 255
        mask_lab = (
            (lightness >= 105)
            & (np.abs(channel_a - 128) <= 42)
            & (np.abs(channel_b - 128) <= 58)
        ).astype(np.uint8) * 255

    mask = cv2.bitwise_or(mask_hsv, mask_lab)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
        iterations=1,
    )

    return mask


def estimate_grid_angles(img: np.ndarray, white_mask: np.ndarray) -> tuple[float, float]:
    """用 Hough 线段估计菱形网格的正斜率和负斜率方向角。"""
    edges = cv2.Canny(white_mask, 50, 150)

    h, w = img.shape[:2]
    scale = math.sqrt((h * w) / (1280 * 720))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(25, int(30 * scale)),
        minLineLength=max(20, int(24 * scale)),
        maxLineGap=max(6, int(10 * scale)),
    )

    pos_angles: list[float] = []
    neg_angles: list[float] = []

    if lines is not None:
        lines = np.asarray(lines).reshape(-1, 4)
        for line in lines:
            x1, y1, x2, y2 = map(float, line)

            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)

            if length < max(18, 20 * scale):
                continue

            angle = math.degrees(math.atan2(dy, dx))

            while angle < -90:
                angle += 180

            while angle >= 90:
                angle -= 180

            if 12 <= abs(angle) <= 78:
                if angle > 0:
                    pos_angles.append(angle)
                else:
                    neg_angles.append(angle)

    pos = float(np.median(pos_angles)) if len(pos_angles) >= 3 else 34.0
    neg = float(np.median(neg_angles)) if len(neg_angles) >= 3 else -34.0

    return pos, neg


def build_orientation_edge_mask(
    img: np.ndarray,
    pos_angle: float,
    neg_angle: float,
    tolerance: float = 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """只保留菱形网格两组斜边方向的边缘。"""
    white = make_white_mask(img, mode="strict")
    edges = cv2.Canny(white, 50, 150)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    grad_angle = np.degrees(np.arctan2(sy, sx))

    # 梯度方向加 90 度得到边缘线方向，并归一到 [-90, 90)。
    line_angle = (grad_angle + 180) % 180 - 90

    keep = (edges > 0) & (
        (angle_diff(line_angle, pos_angle) <= tolerance)
        | (angle_diff(line_angle, neg_angle) <= tolerance)
    )

    oriented = keep.astype(np.uint8) * 255

    h, w = img.shape[:2]
    scale = math.sqrt((h * w) / (1920 * 865))

    kernel_size = max(3, int(round(3 * scale)))

    if kernel_size % 2 == 0:
        kernel_size += 1

    connected = cv2.dilate(
        oriented,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    )

    return oriented, connected


def order_quad_top_right_bottom_left(points: np.ndarray) -> np.ndarray:
    """把四个角点排序为 top、right、bottom、left。"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)

    top_idx = int(np.argmin(pts[:, 1]))
    bottom_idx = int(np.argmax(pts[:, 1]))

    rest = [i for i in range(4) if i not in (top_idx, bottom_idx)]

    if pts[rest[0], 0] < pts[rest[1], 0]:
        left_idx = rest[0]
        right_idx = rest[1]
    else:
        left_idx = rest[1]
        right_idx = rest[0]

    return np.array(
        [
            pts[top_idx],
            pts[right_idx],
            pts[bottom_idx],
            pts[left_idx],
        ],
        dtype=np.float32,
    )


def quad_geometry_score(quad: np.ndarray, img_shape: tuple[int, int]) -> tuple[float, ...] | None:
    """判断四边形是否像外层大菱形，返回越小越好的排序分数。"""
    h, w = img_shape[:2]

    top, right, bottom, left = quad

    if not (top[1] < right[1] < bottom[1]):
        return None

    if not (top[1] < left[1] < bottom[1]):
        return None

    if not (left[0] < top[0] < right[0]):
        return None

    if not (left[0] < bottom[0] < right[0]):
        return None

    vertical_diag = math.hypot(
        float(bottom[0] - top[0]),
        float(bottom[1] - top[1]),
    )

    horizontal_diag = math.hypot(
        float(right[0] - left[0]),
        float(right[1] - left[1]),
    )

    if vertical_diag < min(h, w) * 0.12:
        return None

    if horizontal_diag < min(h, w) * 0.12:
        return None

    aspect = min(vertical_diag, horizontal_diag) / max(vertical_diag, horizontal_diag)

    if not (0.45 <= aspect <= 0.95):
        return None

    area = abs(cv2.contourArea(quad.reshape(-1, 1, 2)))
    image_area = h * w

    if area < image_area * 0.015:
        return None

    if area > image_area * 0.75:
        return None

    middle_y = (top[1] + bottom[1]) * 0.5

    side_middle_error = (
        abs(float(right[1] - middle_y))
        + abs(float(left[1] - middle_y))
    ) / max(vertical_diag, 1.0)

    side_level_error = abs(float(right[1] - left[1])) / max(vertical_diag, 1.0)

    center_x = float(np.mean(quad[:, 0]))

    # 右侧 UI 更容易误判，给过右的候选一点惩罚。
    right_penalty = max(0.0, center_x / max(w, 1) - 0.75)

    return (
        side_middle_error * 2.0 + side_level_error + right_penalty,
        -area * 1e-6,
    )


def approximate_quad_from_component(component_mask: np.ndarray) -> np.ndarray | None:
    """从方向边缘连通组件中提取凸包四角。"""
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)

    perimeter = cv2.arcLength(hull, True)

    for eps_ratio in [
        0.006,
        0.008,
        0.010,
        0.012,
        0.016,
        0.020,
        0.030,
        0.040,
        0.060,
        0.080,
    ]:
        approx = cv2.approxPolyDP(
            hull,
            eps_ratio * perimeter,
            True,
        )

        if len(approx) == 4:
            return order_quad_top_right_bottom_left(approx.reshape(4, 2))

    pts = hull.reshape(-1, 2).astype(np.float32)

    top = pts[np.argmin(pts[:, 1])]
    bottom = pts[np.argmax(pts[:, 1])]
    left = pts[np.argmin(pts[:, 0])]
    right = pts[np.argmax(pts[:, 0])]

    quad = np.array([top, right, bottom, left], dtype=np.float32)

    return order_quad_top_right_bottom_left(quad)


def detect_grid_quad(img: np.ndarray) -> GridDetection:
    """检测外层大菱形四角。"""
    white = make_white_mask(img, mode="strict")

    pos_angle, neg_angle = estimate_grid_angles(img, white)

    raw_oriented, connected_oriented = build_orientation_edge_mask(
        img,
        pos_angle,
        neg_angle,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(connected_oriented)

    h, w = img.shape[:2]

    candidates = []

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        if area < max(500, h * w * 0.002):
            continue

        if bw < w * 0.08 or bh < h * 0.08:
            continue

        component = (labels == label).astype(np.uint8) * 255

        quad = approximate_quad_from_component(component)

        if quad is None:
            continue

        score = quad_geometry_score(quad, img.shape)

        if score is None:
            continue

        final_score = (
            *score,
            -float(area) * 1e-5,
        )

        candidates.append(
            (
                final_score,
                quad,
                float(area),
                (int(x), int(y), int(bw), int(bh)),
                component,
            )
        )

    if not candidates:
        raise RuntimeError(
            "没有找到稳定的大菱形网格。建议检查截图是否为完整棋盘画面，"
            "或先用 tools/crop_red_border.py 裁剪出棋盘区域再重试。"
        )

    candidates.sort(key=lambda item: item[0])

    _, quad, area, bbox, component = candidates[0]

    return GridDetection(
        quad=quad.astype(np.float32),
        component_area=area,
        component_bbox=bbox,
        angle_pos=pos_angle,
        angle_neg=neg_angle,
        orientation_mask=raw_oriented,
        component_mask=component,
    )


RED_BORDER_MIN_AREA_RATIO = 0.0008


def make_red_mask(img: np.ndarray) -> np.ndarray:
    """提取亮红色掩码（游戏红色边框/菱形边框）。"""
    blue = img[:, :, 0].astype(np.int16)
    green = img[:, :, 1].astype(np.int16)
    red = img[:, :, 2].astype(np.int16)
    mask = (red > 150) & (red - green > 80) & (red - blue > 80)
    return mask.astype(np.uint8)


def detect_red_border_quad(
    img: np.ndarray,
    min_area: float | None = None,
) -> np.ndarray | None:
    """检测棋子棋盘外围的红色菱形边框，返回排序好的四角 (top, right, bottom, left)。

    找不到可靠的红色菱形时返回 ``None``，由调用方回退到白线检测。
    """
    _validate_screenshot(img)
    h, w = img.shape[:2]

    floor_area = max(
        300.0,
        float(h) * float(w) * RED_BORDER_MIN_AREA_RATIO,
    )
    min_area = max(floor_area, float(min_area) if min_area else 0.0)

    red = make_red_mask(img)
    closed = cv2.morphologyEx(
        red * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
    )

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        closed,
        8,
    )

    candidates: list[tuple[tuple[float, ...], np.ndarray]] = []

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        if area < min_area:
            continue

        component = (labels == label).astype(np.uint8) * 255
        quad = approximate_quad_from_component(component)

        if quad is None:
            continue

        score = quad_geometry_score(quad, img.shape)

        if score is None:
            # 该组件不是贴合外部大菱形的形状（例如左上角关闭按钮、右下角
            # 弹药图标、棋盘上的小红晶体等），跳过。
            continue

        candidates.append((score, quad))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def apply_quad_roi(
    img: np.ndarray,
    quad: np.ndarray,
    *,
    mask: bool = True,
) -> tuple[np.ndarray, int, int]:
    """把图像裁剪到 ``quad`` 的外接矩形，返回 (裁剪图, 相对原图偏移 x, 偏移 y)。

    ``mask=True`` 时把菱形四角外的像素置 0，只保留红色边框内部（裁掉四角水面）。
    """
    _validate_screenshot(img)
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)

    img_h, img_w = img.shape[:2]
    xs = q[:, 0]
    ys = q[:, 1]

    x0 = max(0, int(np.floor(float(xs.min()))))
    y0 = max(0, int(np.floor(float(ys.min()))))
    x1 = min(img_w, int(np.ceil(float(xs.max()))))
    y1 = min(img_h, int(np.ceil(float(ys.max()))))

    if x1 <= x0 or y1 <= y0:
        raise ValueError("quad ROI 无效或越界")

    cropped = img[y0:y1, x0:x1].copy()

    if mask:
        local = q - np.array([x0, y0], dtype=np.float32)
        polygon = np.round(local).astype(np.int32).reshape(-1, 1, 2)
        mask_img = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillConvexPoly(mask_img, polygon, 255, lineType=cv2.LINE_8)
        cropped[mask_img == 0] = 0

    return cropped, x0, y0


def _quad_centers_result(
    quad: np.ndarray,
    n: int,
) -> DiamondCentersResult:
    """根据给定的菱形外框四角直接生成 N x N 中心点结果。"""
    float_points = centers_from_quad(quad, n)
    points = [
        (int(round(x)), int(round(y)))
        for x, y in float_points
    ]
    return DiamondCentersResult(
        points=points,
        float_points=float_points,
        detection=None,
        local_quad=quad.copy(),
        global_quad=quad.copy(),
        offset_x=0,
        offset_y=0,
    )


def _sane_grid_quad(quad: np.ndarray, img: np.ndarray) -> bool:
    """轻量校验，确保四角能当作网格边界使用，否则回退。"""
    h, w = img.shape[:2]
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(q).all():
        return False
    if (
        q[:, 0].min() < 0
        or q[:, 0].max() >= w
        or q[:, 1].min() < 0
        or q[:, 1].max() >= h
    ):
        return False
    contour = q.reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        return False
    minimum_area = max(100.0, float(w * h) * 0.01)
    if abs(float(cv2.contourArea(contour))) < minimum_area:
        return False
    return True


def detect_grid_points_red_roi(
    img: np.ndarray,
    n: int,
    *,
    fixed_quad: np.ndarray | Sequence[Sequence[float]] | None = None,
    use_red_border: bool = True,
) -> DiamondCentersResult:
    """检测 N x N 网格中心点，按优先级依次尝试：

    1. ``fixed_quad`` —— 调用方提供的固定四角（例如 points.json 里的标定点位）。
       这是唯一权威的网格外框，可保证裁剪、点击点、检测三处使用同一组四角。
    2. 自动检测的红色菱形边框四角。
    3. 常规白线网格检测。

    前两者拿到合法 quad 时直接用 ``centers_from_quad`` 生成中心点（当格子被
    潜艇/泡沫遮挡时，白线检测常会找到过小的错误菱形）。返回的点和 quad 均在
    原图像坐标系内。
    """
    _validate_screenshot(img)
    n = _validate_grid_size(n)

    if fixed_quad is not None:
        quad = np.asarray(fixed_quad, dtype=np.float32).reshape(4, 2)
        if _sane_grid_quad(quad, img):
            logger.info("using fixed quad as the grid boundary")
            return _quad_centers_result(quad, n)
        logger.warning("fixed quad 无效，回退到自动检测")

    if use_red_border:
        quad = detect_red_border_quad(img)
        if quad is not None:
            if _sane_grid_quad(quad, img):
                logger.info("using red-border quad as the grid boundary")
                return _quad_centers_result(quad, n)
            logger.warning("red-border quad 无效，回退到白线检测")

    return detect_diamond_centers(img, n)


def centers_from_quad(quad: np.ndarray, n: int) -> list[tuple[float, float]]:
    """根据外层菱形四角透视投影生成 N x N 中心点。"""
    src = np.array(
        [
            [0, 0],
            [n, 0],
            [n, n],
            [0, n],
        ],
        dtype=np.float32,
    )

    dst = quad.astype(np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)

    points: list[tuple[float, float]] = []

    for i in range(n):
        for j in range(n):
            p = np.array(
                [
                    [
                        [j + 0.5, i + 0.5],
                    ]
                ],
                dtype=np.float32,
            )

            q = cv2.perspectiveTransform(p, matrix)[0, 0]

            points.append(
                (
                    float(q[0]),
                    float(q[1]),
                )
            )

    return points


def draw_quad(
    img: np.ndarray,
    quad: np.ndarray,
    offset_x: int = 0,
    offset_y: int = 0,
) -> np.ndarray:
    """在图片上绘制菱形外框和四角名称。"""
    out = img.copy()

    q = quad.copy()
    q[:, 0] += offset_x
    q[:, 1] += offset_y

    cv2.polylines(
        out,
        [q.astype(np.int32)],
        True,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    names = ["top", "right", "bottom", "left"]

    for name, point in zip(names, q, strict=True):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))

        cv2.circle(
            out,
            (x, y),
            7,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            name,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def _validate_screenshot(screenshot: np.ndarray) -> None:
    """校验截图必须是 OpenCV 可处理的彩色图片。"""
    if not isinstance(screenshot, np.ndarray):
        raise TypeError("screenshot 必须是 OpenCV 图像对象")

    if screenshot.ndim != 3 or screenshot.shape[2] != 3:
        raise ValueError("screenshot 必须是 BGR 彩色图片")

    if screenshot.size == 0:
        raise ValueError("screenshot 不能为空")


def _validate_grid_size(n: int) -> int:
    """校验网格边长必须为正整数。"""
    try:
        value = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"n 必须是正整数: {n}") from exc

    if value <= 0:
        raise ValueError(f"n 必须是正整数: {n}")

    return value
