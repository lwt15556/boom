import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.image_match import find_template, find_template_multi_scale


class ImageMatchTest(unittest.TestCase):
    def test_multi_scale_matches_resized_retry_template(self):
        template_path = Path("template/retry.png")
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(template)

        scale = 1.15
        resized = cv2.resize(
            template,
            (int(round(template.shape[1] * scale)), int(round(template.shape[0] * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

        screenshot = np.zeros((240, 320, 3), dtype=np.uint8)
        y, x = 80, 120
        screenshot[y : y + resized.shape[0], x : x + resized.shape[1]] = resized

        self.assertIsNone(find_template(screenshot, template_path, threshold=0.9))

        match = find_template_multi_scale(
            screenshot,
            template_path,
            scales=(0.85, 0.95, 1.0, 1.15),
            threshold=0.85,
        )

        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.score, 0.85)
        self.assertLess(abs(match.center[0] - (x + resized.shape[1] // 2)), 3)
        self.assertLess(abs(match.center[1] - (y + resized.shape[0] // 2)), 3)


if __name__ == "__main__":
    unittest.main()
