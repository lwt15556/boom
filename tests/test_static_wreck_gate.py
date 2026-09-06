import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.wreck_detection import (
    STATIC_WRECK_MIN_CENTER_GRAY_RATIO,
    WRECK_SHAPE_MIN_SCORE,
    WreckShapeMetrics,
    static_wreck_shape_accepts,
    wreck_shape_metrics,
)


class StaticWreckGateTest(unittest.TestCase):
    """The static recovery gate must reject broad bright water patches that
    reach the shape-score floor but lack a compact, centre-bright hull core,
    while still keeping real wreck/hull evidence.
    """

    @classmethod
    def _shape(cls, score, center_gray_ratio):
        return WreckShapeMetrics(
            center_gray_ratio=center_gray_ratio,
            ring_gray_ratio=0.0,
            gray_excess=center_gray_ratio - 0.1,
            component_ratio=center_gray_ratio,
            compactness=0.5,
            cyan_ratio=0.1,
            bright_ratio=0.2,
            score=score,
        )

    def test_gate_requires_both_score_and_centre_gray(self):
        # A real hull: high score and a strong centre-bright core.
        self.assertTrue(static_wreck_shape_accepts(self._shape(0.6, 0.5)))
        # A broad bright water patch: high shape score but no centre core.
        self.assertFalse(
            static_wreck_shape_accepts(self._shape(0.6, 0.2)),
        )
        # Low shape score is still rejected regardless of the centre core.
        self.assertFalse(
            static_wreck_shape_accepts(self._shape(0.2, 0.5)),
        )

    def test_real_wreck_keeps_centre_gray_above_gate(self):
        template_path = Path(__file__).parents[1] / "template" / "visible_wreck_1.png"
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(template)
        self.assertGreater(template.shape[0], 0)

        frame = np.full((720, 1280, 3), (55, 82, 105), dtype=np.uint8)
        height, width = template.shape[:2]
        point = (640, 360)
        x = point[0] - width // 2
        y = point[1] - height // 2
        frame[y : y + height, x : x + width] = template

        metrics = wreck_shape_metrics(frame, point)
        self.assertGreaterEqual(metrics.score, WRECK_SHAPE_MIN_SCORE)
        self.assertGreaterEqual(
            metrics.center_gray_ratio,
            STATIC_WRECK_MIN_CENTER_GRAY_RATIO,
        )


if __name__ == "__main__":
    unittest.main()
