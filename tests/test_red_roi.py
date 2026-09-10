import unittest
from unittest import mock

import cv2
import numpy as np

from utils.diamond_centers import (
    apply_quad_roi,
    detect_red_border_quad,
    detect_grid_points_red_roi,
)

# BGR 近似海水颜色（深蓝绿）与红色边框颜色。
SEA = (140, 90, 30)
RED = (30, 30, 230)
# 合成菱形外框四角（top, right, bottom, left）。
QUAD = np.array([[640, 80], [1100, 360], [640, 640], [180, 360]], dtype=np.float32)


def _synthetic_board(shape=(720, 1280, 3), red_border=True, red_ui=True):
    """构造一张带红色菱形边框（和少量红色 UI 干扰）的合成截图。"""
    img = np.full(shape, SEA, dtype=np.uint8)
    if red_border:
        pts = QUAD.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, RED, 8, cv2.LINE_AA)
    if red_ui:
        cv2.rectangle(img, (20, 20), (55, 55), RED, -1)  # 左上角红色方块
        cv2.circle(img, (1220, 680), 20, RED, -1)  # 右下角红色圆点
    return img


class DetectRedBorderQuadTest(unittest.TestCase):
    def test_detects_full_diamond_corners(self):
        img = _synthetic_board(red_border=True, red_ui=True)
        detected = detect_red_border_quad(img)
        self.assertIsNotNone(detected)
        np.testing.assert_allclose(detected, QUAD, atol=12)

    def test_rejects_background_without_border(self):
        img = _synthetic_board(red_border=False, red_ui=True)
        self.assertIsNone(detect_red_border_quad(img))


class ApplyQuadRoiTest(unittest.TestCase):
    def test_crop_bounds_and_offset(self):
        img = _synthetic_board()
        crop, ox, oy = apply_quad_roi(img, QUAD, mask=False)
        self.assertEqual((ox, oy), (180, 80))
        self.assertEqual(crop.shape, (640 - 80, 1100 - 180, 3))
        # 未 mask 时四角保留海水颜色。
        self.assertTrue(np.all(crop[0, 0] == SEA))
        self.assertTrue(np.all(crop[-1, -1] == SEA))

    def test_mask_zeroes_outside_corners(self):
        img = _synthetic_board()
        crop, ox, oy = apply_quad_roi(img, QUAD, mask=True)
        self.assertEqual((ox, oy), (180, 80))
        # 菱形四角之外（bbox 四个角）应被置 0。
        self.assertEqual(int(crop[0, 0].sum()), 0)
        self.assertEqual(int(crop[0, -1].sum()), 0)
        self.assertEqual(int(crop[-1, 0].sum()), 0)
        self.assertEqual(int(crop[-1, -1].sum()), 0)
        # 菱形中心区域仍保留海水颜色。
        center = (crop.shape[1] // 2, crop.shape[0] // 2)
        self.assertTrue(np.all(crop[center[1], center[0]] == SEA))


class DetectGridPointsRedRoiTest(unittest.TestCase):
    def test_uses_red_quad_for_centers(self):
        img = _synthetic_board(red_border=True)
        with mock.patch(
            "utils.diamond_centers.detect_red_border_quad",
            return_value=QUAD.copy(),
        ):
            res = detect_grid_points_red_roi(img, 10)
        self.assertEqual(len(res.points), 100)
        self.assertEqual(len(res.float_points), 100)
        np.testing.assert_allclose(res.global_quad, QUAD, atol=1.0)
        contour = QUAD.astype(np.int32).reshape(-1, 1, 2)
        for x, y in res.points:
            self.assertGreaterEqual(cv2.pointPolygonTest(contour, (x, y), False), 0)

    def test_falls_back_to_white_detector_when_no_border(self):
        img = _synthetic_board(red_border=False, red_ui=False)
        with mock.patch(
            "utils.diamond_centers.detect_red_border_quad",
            return_value=None,
        ), mock.patch(
            "utils.diamond_centers.detect_diamond_centers",
            return_value="fallback",
        ) as white:
            result = detect_grid_points_red_roi(img, 5)
        self.assertEqual(result, "fallback")
        white.assert_called_once()

    def test_fixed_quad_takes_priority_over_red_border(self):
        # 传入固定 quad 时直接使用它，不再触发红色边框/白线检测。
        img = _synthetic_board(red_border=True)
        fixed = QUAD.copy()
        with mock.patch(
            "utils.diamond_centers.detect_red_border_quad",
        ) as red, mock.patch(
            "utils.diamond_centers.detect_diamond_centers",
        ) as white:
            res = detect_grid_points_red_roi(img, 10, fixed_quad=fixed)
        self.assertEqual(len(res.points), 100)
        np.testing.assert_allclose(res.global_quad, fixed, atol=1.0)
        red.assert_not_called()
        white.assert_not_called()

    def test_invalid_fixed_quad_falls_back_to_red_border(self):
        # 固定 quad 越界时丢弃它，仍走红色边框检测。
        img = _synthetic_board(red_border=True)
        out_of_bounds = np.array(
            [[500, -50], [1200, 300], [500, 650], [-50, 300]],
            dtype=np.float32,
        )
        with mock.patch(
            "utils.diamond_centers.detect_red_border_quad",
            return_value=QUAD.copy(),
        ):
            res = detect_grid_points_red_roi(img, 10, fixed_quad=out_of_bounds)
        np.testing.assert_allclose(res.global_quad, QUAD, atol=1.0)

    def test_real_red_border_and_calibration_clean(self):
        # 端到端：镜像展示的红框截图应返回全覆盖的中心点并过校准。
        img = _synthetic_board(red_border=True, red_ui=True)
        res = detect_grid_points_red_roi(img, 10)
        self.assertEqual(len(res.points), 100)
        self.assertGreaterEqual(
            abs(float(cv2.contourArea(QUAD.astype(np.int32).reshape(-1, 1, 2)))),
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
