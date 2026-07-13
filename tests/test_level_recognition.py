import unittest
from pathlib import Path

from utils.diamond_centers import read_image
from utils.level_recognition import recognize_level_from_screenshot
from utils.level_title_recognition import recognize_level_title


class LevelRecognitionTest(unittest.TestCase):
    def test_saved_reference_recognizes_its_own_level(self):
        screenshot = read_image(Path("save_points/imgs/1.png"))

        result = recognize_level_from_screenshot(
            screenshot,
            candidate_levels=[1, 2, 3],
            min_score=0.9,
            min_margin=0.05,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.level, 1)
        self.assertTrue(result.confident)
        self.assertGreaterEqual(result.score, 0.99)

    def test_title_recognition_reads_level_number(self):
        screenshot = read_image(Path("save_points/imgs/2.png"))

        result = recognize_level_title(
            screenshot,
            reference_dir=Path("save_points/imgs"),
            min_score=0.9,
            min_margin=0.05,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.level, 2)
        self.assertTrue(result.confident)
        self.assertGreaterEqual(result.score, 0.99)


if __name__ == "__main__":
    unittest.main()
