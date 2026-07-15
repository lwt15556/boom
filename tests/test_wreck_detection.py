import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.wreck_detection import (
    red_hit_marker_template_visible,
    red_hit_marker_visible,
    visible_wreck_static_detected,
)


class WreckDetectionTest(unittest.TestCase):
    def test_clean_reference_cell_is_not_a_visible_wreck(self):
        reference_path = Path(__file__).parents[1] / "save_points" / "imgs" / "8.png"
        frame = cv2.imread(str(reference_path))
        point = (665, 318)

        self.assertIsNotNone(frame)
        self.assertFalse(red_hit_marker_template_visible(frame, point))
        self.assertFalse(red_hit_marker_visible(frame, point))
        self.assertFalse(visible_wreck_static_detected(frame, point))

    def test_red_hit_marker_template_matches_real_template(self):
        template = cv2.imread(str(Path("template") / "red_hit_marker.png"))
        self.assertIsNotNone(template)

        frame = np.zeros((360, 360, 3), dtype=np.uint8)
        x, y = 120, 110
        h, w = template.shape[:2]
        frame[y : y + h, x : x + w] = template
        point = (x + w // 2, y + h // 2)

        self.assertTrue(red_hit_marker_template_visible(frame, point))
        self.assertTrue(red_hit_marker_visible(frame, point))
        self.assertTrue(visible_wreck_static_detected(frame, point))

    def test_red_blob_without_marker_template_is_not_treated_as_hit(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.circle(frame, (120, 96), 16, (0, 0, 255), cv2.FILLED)

        point = (120, 120)

        self.assertFalse(red_hit_marker_template_visible(frame, point))
        self.assertFalse(red_hit_marker_visible(frame, point))
        self.assertFalse(visible_wreck_static_detected(frame, point))


if __name__ == "__main__":
    unittest.main()
