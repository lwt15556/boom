import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.wreck_detection import (
    WRECK_SHAPE_MIN_SCORE,
    build_surface_water_baseline,
    surface_glare_score,
    surface_reflection_detected,
    wreck_shape_metrics,
)


class SurfaceGlareDetectionTest(unittest.TestCase):
    """Regression tests for water reflection and shape evidence."""

    FRAME_SHAPE = (720, 1280, 3)
    POINT = (640, 360)

    @classmethod
    def _water_frame(cls, color=(55, 82, 105)):
        return np.full(cls.FRAME_SHAPE, color, dtype=np.uint8)

    @classmethod
    def _cyan_reflection_frame(cls, blue=210, green=180, red=80):
        frame = cls._water_frame()
        # A broad, high-value blue/teal patch fills most of one cell.  It has
        # no neutral compact core, which is the characteristic difference from
        # the gray wreck template.
        cv2.ellipse(
            frame,
            cls.POINT,
            (31, 21),
            0,
            0,
            360,
            (blue, green, red),
            cv2.FILLED,
        )
        return frame

    def test_baseline_uses_only_matching_valid_frames(self):
        first = self._water_frame()
        second = self._water_frame()
        second[350:371, 630:651] = (210, 180, 80)
        mismatched = np.zeros((360, 640, 3), dtype=np.uint8)

        baseline = build_surface_water_baseline(
            [None, first, mismatched, second, "invalid"]
        )

        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.frame_count, 2)
        self.assertEqual(baseline.median_gray.shape, first.shape[:2])
        self.assertEqual(baseline.temporal_mad.shape, first.shape[:2])
        self.assertGreater(float(np.max(baseline.temporal_mad)), 0.0)

    def test_broad_blue_reflection_is_flagged(self):
        frames = [
            self._cyan_reflection_frame(blue=170, green=150, red=65),
            self._cyan_reflection_frame(blue=210, green=180, red=80),
            self._cyan_reflection_frame(blue=190, green=165, red=70),
        ]
        baseline = build_surface_water_baseline(frames)

        metrics = wreck_shape_metrics(frames[-1], self.POINT)
        self.assertLess(metrics.score, WRECK_SHAPE_MIN_SCORE)
        self.assertGreater(metrics.cyan_ratio, 0.5)
        self.assertTrue(
            surface_reflection_detected(
                frames[-1],
                self.POINT,
                baseline=baseline,
            )
        )
        self.assertGreater(surface_glare_score(frames[-1], self.POINT, baseline=baseline), 0.48)

    def test_compact_neutral_wreck_is_not_flagged_as_reflection(self):
        template_path = Path(__file__).parents[1] / "template" / "visible_wreck_1.png"
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(template)

        frame = self._water_frame()
        height, width = template.shape[:2]
        x = self.POINT[0] - width // 2
        y = self.POINT[1] - height // 2
        frame[y : y + height, x : x + width] = template

        metrics = wreck_shape_metrics(frame, self.POINT)
        self.assertGreaterEqual(metrics.score, WRECK_SHAPE_MIN_SCORE)
        self.assertLess(metrics.cyan_ratio, 0.5)
        self.assertFalse(surface_reflection_detected(frame, self.POINT))

    def test_insufficient_baseline_stays_spatial_only(self):
        frame = self._cyan_reflection_frame()
        baseline = build_surface_water_baseline([frame])

        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.frame_count, 1)
        # A single frame must not manufacture temporal evidence.  The colour
        # and shape guards can still identify an obvious broad reflection.
        self.assertEqual(float(np.max(baseline.temporal_mad)), 0.0)
        self.assertTrue(surface_reflection_detected(frame, self.POINT, baseline=baseline))

    def test_current_frame_is_compared_with_median_baseline(self):
        baseline_frame = self._water_frame()
        baseline = build_surface_water_baseline(
            [baseline_frame.copy(), baseline_frame.copy(), baseline_frame.copy()]
        )
        current = self._cyan_reflection_frame()

        self.assertIsNotNone(baseline)
        self.assertEqual(float(np.max(baseline.temporal_mad)), 0.0)
        # Even a perfectly static baseline must contribute evidence when the
        # current frame contains a broad highlight absent from its median.
        self.assertTrue(
            surface_reflection_detected(
                current,
                self.POINT,
                baseline=baseline,
            )
        )

    def test_bright_strip_crossing_cell_boundary_is_still_reflection(self):
        frame = self._water_frame()
        # The strip is deliberately wider than one calibrated cell.  A
        # cross-cell highlight must not become two independent wreck hits.
        cv2.rectangle(frame, (590, 349), (771, 375), (210, 180, 80), cv2.FILLED)

        left = (640, 360)
        right = (720, 360)
        self.assertTrue(surface_reflection_detected(frame, left))
        self.assertTrue(surface_reflection_detected(frame, right))
        self.assertLess(wreck_shape_metrics(frame, left).score, WRECK_SHAPE_MIN_SCORE)
        self.assertLess(wreck_shape_metrics(frame, right).score, WRECK_SHAPE_MIN_SCORE)

    def test_shape_metrics_scale_with_resized_frame(self):
        source = self._cyan_reflection_frame()
        resized = cv2.resize(source, (640, 360), interpolation=cv2.INTER_AREA)

        source_point = self.POINT
        resized_point = (source_point[0] // 2, source_point[1] // 2)
        source_metrics = wreck_shape_metrics(source, source_point)
        resized_metrics = wreck_shape_metrics(resized, resized_point)

        self.assertLess(source_metrics.score, WRECK_SHAPE_MIN_SCORE)
        self.assertLess(resized_metrics.score, WRECK_SHAPE_MIN_SCORE)
        self.assertGreater(resized_metrics.cyan_ratio, 0.5)
        self.assertTrue(surface_reflection_detected(resized, resized_point))


if __name__ == "__main__":
    unittest.main()
