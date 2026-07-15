import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import utils.image_match as image_match
from utils.image_match import find_template, find_template_multi_scale


class ImageMatchTest(unittest.TestCase):
    def setUp(self):
        image_match._clear_template_caches()

    def tearDown(self):
        image_match._clear_template_caches()

    def test_template_file_is_read_only_once_for_repeated_matches(self):
        template_path = Path("template/retry.png")
        screenshot = np.zeros((240, 320, 3), dtype=np.uint8)

        with patch.object(
            image_match.cv2,
            "imread",
            wraps=image_match.cv2.imread,
        ) as imread:
            find_template(screenshot, template_path)
            find_template(screenshot, template_path)

        self.assertEqual(imread.call_count, 1)

    def test_scaled_templates_are_reused_across_multi_scale_matches(self):
        template_path = Path("template/retry.png")
        screenshot = np.zeros((240, 320, 3), dtype=np.uint8)
        scales = (0.85, 1.0, 1.15)

        with patch.object(
            image_match,
            "_resize_template",
            wraps=image_match._resize_template,
        ) as resize:
            find_template_multi_scale(screenshot, template_path, scales=scales)
            find_template_multi_scale(screenshot, template_path, scales=scales)

        self.assertEqual(resize.call_count, len(scales))

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
