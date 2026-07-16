import unittest
from pathlib import Path

import cv2
import numpy as np

from config import LEVEL_GRID_SIZES
from save_points.points import read_saved_points
from utils.wreck_detection import (
    detect_completed_submarine_candidate_cells,
    detect_visible_wreck_cells,
    red_hit_marker_template_visible,
    red_hit_marker_visible,
    visible_wreck_static_detected,
    wreck_template_visible,
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

    def test_low_contrast_centered_gray_wreck_is_visible(self):
        template = cv2.imread(str(Path("template") / "visible_wreck_1.png"))
        self.assertIsNotNone(template)

        frame = np.full((240, 240, 3), (110, 125, 130), dtype=np.uint8)
        point = (120, 120)
        height, width = template.shape[:2]
        x = point[0] - width // 2 + 4
        y = point[1] - height // 2 - 4
        frame[y:y + height, x:x + width] = template

        self.assertTrue(wreck_template_visible(frame, point))
        self.assertTrue(visible_wreck_static_detected(frame, point))

    def test_gray_wreck_template_in_neighbor_cell_is_not_visible(self):
        template = cv2.imread(str(Path("template") / "visible_wreck_1.png"))
        self.assertIsNotNone(template)

        frame = np.full((240, 240, 3), (110, 125, 130), dtype=np.uint8)
        point = (120, 120)
        height, width = template.shape[:2]
        x = point[0] - width // 2 + 36
        y = point[1] - height // 2
        frame[y:y + height, x:x + width] = template

        self.assertFalse(wreck_template_visible(frame, point))

    def test_gray_wreck_templates_do_not_match_clean_reference_grids(self):
        reference_dir = Path(__file__).parents[1] / "save_points" / "imgs"
        for level in range(1, 11):
            with self.subTest(level=level):
                frame = cv2.imread(str(reference_dir / f"{level}.png"))
                grid_size = LEVEL_GRID_SIZES[level]
                points = read_saved_points(level, expected_n=grid_size)
                self.assertIsNotNone(frame)
                self.assertIsNotNone(points)
                self.assertEqual(
                    detect_visible_wreck_cells(frame, points, grid_size),
                    set(),
                )

    def test_completed_ship_candidates_are_anchored_by_red_markers(self):
        grid_size = 7
        points = [
            (80 + col * 40, 80 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((420, 420, 3), dtype=np.uint8)
        ship_cells = {
            (0, 1),
            (0, 2),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (6, 1),
            (6, 2),
            (6, 3),
        }
        for row, col in ship_cells:
            x, y = points[row * grid_size + col]
            cv2.ellipse(frame, (x, y), (22, 14), 0, 0, 360, (175, 178, 180), cv2.FILLED)
        for row, col in ((0, 1), (4, 2), (6, 1)):
            x, y = points[row * grid_size + col]
            cv2.circle(frame, (x + 4, y - 4), 6, (0, 0, 255), cv2.FILLED)

        noise_cell = (2, 6)
        x, y = points[noise_cell[0] * grid_size + noise_cell[1]]
        cv2.ellipse(frame, (x, y), (22, 14), 0, 0, 360, (190, 190, 190), cv2.FILLED)

        candidates = detect_completed_submarine_candidate_cells(frame, points, grid_size)

        self.assertTrue(ship_cells.issubset(candidates))
        self.assertNotIn(noise_cell, candidates)


if __name__ == "__main__":
    unittest.main()
