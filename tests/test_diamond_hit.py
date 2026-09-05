import unittest

import cv2
import numpy as np

from utils.diamond_hit import (
    DiamondHitConfig,
    classify_diamond_hit,
    diamond_points,
)


class DiamondHitLabEvidenceTest(unittest.TestCase):
    CENTER = (140, 100)
    CONFIG = DiamondHitConfig(
        diamond_w=80,
        diamond_h=56,
        search_radius=0,
        adaptive_thresholds=False,
    )

    @staticmethod
    def board() -> np.ndarray:
        return np.full((200, 280, 3), (160, 105, 48), dtype=np.uint8)

    def color_target(self, before: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        after = before.copy()
        points = diamond_points(
            self.CENTER,
            self.CONFIG.diamond_w,
            self.CONFIG.diamond_h,
            self.CONFIG.center_scale,
        )
        cv2.fillConvexPoly(after, points, color)
        return after

    def test_local_color_change_has_positive_center_excess(self):
        before = self.board()
        after = self.color_target(before, (15, 175, 220))

        result = classify_diamond_hit(before, after, self.CENTER, self.CONFIG)

        self.assertGreater(result.lab_color_change_ratio, 0.95)
        self.assertGreater(result.lab_color_change_excess, 0.90)

    def test_global_color_shift_has_no_local_color_change_excess(self):
        before = self.board()
        after = np.full_like(before, (15, 175, 220))

        result = classify_diamond_hit(before, after, self.CENTER, self.CONFIG)

        self.assertGreater(result.lab_color_change_ratio, 0.95)
        self.assertAlmostEqual(result.lab_color_change_excess, 0.0, delta=0.01)

    def test_color_change_without_wreck_features_cannot_be_a_hit(self):
        before = self.board()
        after = self.color_target(before, (15, 175, 220))

        result = classify_diamond_hit(before, after, self.CENTER, self.CONFIG)

        self.assertEqual(result.state, "miss")
        self.assertLess(result.center_gray_ratio, self.CONFIG.min_center_gray_ratio)
        self.assertLess(result.component_ratio, self.CONFIG.min_component_ratio)


if __name__ == "__main__":
    unittest.main()
