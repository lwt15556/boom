import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from config import LEVEL_GRID_SIZES
from save_points.points import read_saved_points
from utils.wreck_detection import (
    COMPLETED_SHIP_BODY_MIN_SCORE,
    _title_flag_l_hull_pairs,
    completed_ship_body_score,
    detect_completed_submarine_candidate_cells,
    detect_red_submarine_marker_cells,
    detect_visible_wreck_cells,
    is_title_occluded_cell,
    red_hit_marker_template_visible,
    red_hit_marker_visible,
    red_submarine_marker_visible,
    visible_wreck_static_detected,
    wreck_template_visible,
)
from utils.red_scout import _default_hit_detector


class WreckDetectionTest(unittest.TestCase):
    def test_title_flag_l_is_corrected_before_selecting_a_straight_hull(self):
        crop = cv2.imread(str(Path(__file__).parent / "fixtures" / "level15_top_submarine.png"))
        self.assertIsNotNone(crop)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[82:162, 655:775] = crop
        points = read_saved_points(15, expected_n=10)

        # Preserve the raw L until its raised corner has been removed.
        for preserve_alternatives in (False, True):
            with self.subTest(preserve_alternatives=preserve_alternatives):
                cells = detect_completed_submarine_candidate_cells(
                    frame, points, 10,
                    preserve_alternatives=preserve_alternatives,
                )
                self.assertEqual(cells, {(0, 2), (1, 2)})

    def test_title_flag_l_requires_an_unambiguous_marked_two_cell_hull(self):
        scores = {(0, 1): 0.68, (0, 2): 0.49, (1, 2): 0.45}
        for candidate_scores, anchors in (
            (scores, set()),
            (scores, {(0, 2)}),
            (scores | {(1, 1): 0.4}, {(0, 1)}),
            (scores | {(1, 2): 0.31}, {(0, 1)}),
            (scores | {(2, 2): 0.5}, {(0, 1)}),
            (scores | {(0, 0): 0.5, (1, 0): 0.5}, {(0, 1)}),
        ):
            with self.subTest(scores=candidate_scores, anchors=anchors):
                self.assertEqual(_title_flag_l_hull_pairs(candidate_scores, anchors, 10), {})

    def test_title_occlusion_is_limited_to_fixed_cells_on_10x10_board(self):
        self.assertTrue(is_title_occluded_cell((0, 0), 10))
        self.assertTrue(is_title_occluded_cell((1, 1), 10))
        self.assertFalse(is_title_occluded_cell((0, 3), 10))
        self.assertFalse(is_title_occluded_cell((0, 0), 9))

    def test_visible_wreck_scan_skips_only_fixed_title_cells_on_10x10_board(self):
        grid_size = 10
        points = [
            (600 + col * 35, 60 + row * 35)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch(
            "utils.wreck_detection.visible_wreck_static_detected",
            return_value=True,
        ) as detect:
            hits = detect_visible_wreck_cells(frame, points, grid_size)

        expected = {
            (row, col)
            for row in range(grid_size)
            for col in range(grid_size)
            if not is_title_occluded_cell((row, col), grid_size)
        }
        self.assertEqual(hits, expected)
        self.assertEqual(detect.call_count, 95)

    def test_visible_wreck_scan_does_not_apply_title_cells_to_9x9_board(self):
        grid_size = 9
        points = [
            (600 + col * 35, 60 + row * 35)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch(
            "utils.wreck_detection.visible_wreck_static_detected",
            return_value=True,
        ):
            hits = detect_visible_wreck_cells(frame, points, grid_size)

        self.assertEqual(len(hits), 81)

    def test_title_overlay_is_not_static_hit_evidence(self):
        frame = np.full((720, 1280, 3), (160, 100, 40), dtype=np.uint8)
        point = (700, 80)

        with patch(
            "utils.wreck_detection.wreck_template_visible",
            return_value=True,
        ) as template_visible:
            self.assertFalse(visible_wreck_static_detected(frame, point))

        template_visible.assert_not_called()

    def test_completed_body_score_ignores_title_but_keeps_visible_hull(self):
        frame = np.full((720, 1280, 3), (160, 100, 40), dtype=np.uint8)
        point = (700, 80)
        polygon = np.asarray(
            [(660, 80), (700, 58), (740, 80), (700, 112)],
            dtype=np.float32,
        )
        # Bright title strokes occupy only the fixed UI overlay.
        cv2.rectangle(frame, (665, 58), (735, 93), (245, 245, 245), cv2.FILLED)
        title_only_score = completed_ship_body_score(
            frame,
            point,
            cell_polygon=polygon,
        )

        # A surfaced hull remains visible in the lower, unobscured part of the
        # same diamond and must still be usable as completion evidence.
        cv2.rectangle(frame, (680, 96), (720, 110), (175, 178, 180), cv2.FILLED)
        hull_score = completed_ship_body_score(
            frame,
            point,
            cell_polygon=polygon,
        )

        self.assertLess(title_only_score, COMPLETED_SHIP_BODY_MIN_SCORE)
        self.assertGreaterEqual(hull_score, COMPLETED_SHIP_BODY_MIN_SCORE)

    def test_clean_reference_cell_is_not_a_visible_wreck(self):
        reference_path = Path(__file__).parents[1] / "save_points" / "imgs" / "8.png"
        frame = cv2.imread(str(reference_path))
        point = (665, 318)

        self.assertIsNotNone(frame)
        self.assertFalse(red_hit_marker_template_visible(frame, point))
        self.assertFalse(red_hit_marker_visible(frame, point))
        self.assertFalse(visible_wreck_static_detected(frame, point))

    def test_red_hit_marker_does_not_promote_static_wreck_state(self):
        template = cv2.imread(str(Path("template") / "red_hit_marker.png"))
        self.assertIsNotNone(template)

        frame = np.zeros((360, 360, 3), dtype=np.uint8)
        x, y = 120, 110
        h, w = template.shape[:2]
        frame[y : y + h, x : x + w] = template
        point = (x + w // 2, y + h // 2)

        self.assertTrue(red_hit_marker_template_visible(frame, point))
        self.assertTrue(red_hit_marker_visible(frame, point))
        self.assertFalse(visible_wreck_static_detected(frame, point))

    def test_red_blob_without_marker_template_is_not_treated_as_hit(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.circle(frame, (120, 96), 16, (0, 0, 255), cv2.FILLED)

        point = (120, 120)

        self.assertFalse(red_hit_marker_template_visible(frame, point))
        self.assertFalse(red_hit_marker_visible(frame, point))
        self.assertFalse(visible_wreck_static_detected(frame, point))

    def test_red_submarine_marker_does_not_promote_gray_hull_to_hit(self):
        frame = np.full((240, 240, 3), (30, 70, 100), dtype=np.uint8)
        point = (120, 120)
        cv2.ellipse(frame, point, (35, 24), 0, 0, 360, (170, 170, 170), cv2.FILLED)
        cv2.ellipse(frame, (120, 96), (15, 9), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        self.assertFalse(_default_hit_detector(frame, point))
        self.assertFalse(visible_wreck_static_detected(frame, point))

    def test_offset_red_submarine_marker_does_not_become_hit(self):
        """A flag near the hull edge must still suppress hit classification."""
        frame = np.full((240, 240, 3), (30, 70, 100), dtype=np.uint8)
        point = (120, 120)
        cv2.ellipse(frame, point, (35, 24), 0, 0, 360, (170, 170, 170), cv2.FILLED)
        # 50px above the calibrated center, matching the perspective offset
        # visible in the reported screenshots.
        cv2.rectangle(frame, (113, 66), (128, 74), (0, 0, 255), cv2.FILLED)

        self.assertTrue(red_submarine_marker_visible(frame, point))
        self.assertFalse(_default_hit_detector(frame, point))
        self.assertFalse(visible_wreck_static_detected(frame, point))

    def test_red_marker_binds_to_adjacent_hull_cell_instead_of_nearest_point(self):
        """A flag can project into the tile above its surfaced submarine."""
        grid_size = 7
        points = [
            (80 + col * 40, 80 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((420, 420, 3), dtype=np.uint8)

        # The red component is centred on the calibrated point (2, 2), while
        # the visible hull begins one row below it at (3, 2).  A nearest-point
        # implementation reports (2, 2); the marker-aware binding must use the
        # stronger adjacent hull evidence instead.
        for cell in ((3, 2), (4, 2)):
            x, y = points[cell[0] * grid_size + cell[1]]
            cv2.ellipse(
                frame,
                (x, y),
                (22, 14),
                0,
                0,
                360,
                (175, 178, 180),
                cv2.FILLED,
            )
        x, y = points[2 * grid_size + 2]
        cv2.rectangle(
            frame,
            (x - 7, y - 5),
            (x + 7, y + 5),
            (0, 0, 255),
            cv2.FILLED,
        )

        self.assertEqual(
            detect_red_submarine_marker_cells(frame, points, grid_size),
            {(3, 2)},
        )

        # The same projection can shift a marker one column to the left of a
        # vertical hull.  The binding should move right to the supported body.
        frame = np.zeros((520, 520, 3), dtype=np.uint8)
        grid_size = 10
        points = [
            (80 + col * 40, 80 + row * 35)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        for cell in ((2, 8), (3, 8)):
            x, y = points[cell[0] * grid_size + cell[1]]
            cv2.ellipse(
                frame,
                (x, y),
                (22, 14),
                0,
                0,
                360,
                (175, 178, 180),
                cv2.FILLED,
            )
        x, y = points[2 * grid_size + 7]
        cv2.rectangle(
            frame,
            (x - 7, y - 5),
            (x + 7, y + 5),
            (0, 0, 255),
            cv2.FILLED,
        )

        self.assertEqual(
            detect_red_submarine_marker_cells(frame, points, grid_size),
            {(2, 8)},
        )

    def test_red_marker_only_suppresses_its_assigned_cell_not_neighbor_wreck(self):
        grid_size = 2
        points = [(60, 60), (100, 60), (60, 100), (100, 100)]
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        with (
            patch(
                "utils.wreck_detection._detect_completed_ship_anchor_cells",
                return_value={(0, 0)},
            ),
            patch(
                "utils.wreck_detection.visible_wreck_static_detected",
                return_value=True,
            ) as detect,
        ):
            hits = detect_visible_wreck_cells(frame, points, grid_size)

        self.assertNotIn((0, 0), hits)
        self.assertEqual(hits, {(0, 1), (1, 0), (1, 1)})
        self.assertTrue(all(call.kwargs["ignore_submarine_marker"] for call in detect.call_args_list))

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

    def test_red_marker_without_hull_is_not_a_completed_ship_candidate(self):
        grid_size = 7
        points = [
            (80 + col * 40, 80 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((420, 420, 3), dtype=np.uint8)
        marker_only = (2, 6)
        x, y = points[marker_only[0] * grid_size + marker_only[1]]
        cv2.circle(frame, (x + 4, y - 4), 6, (0, 0, 255), cv2.FILLED)

        candidates = detect_completed_submarine_candidate_cells(
            frame,
            points,
            grid_size,
        )

        self.assertNotIn(marker_only, candidates)

    def test_porthole_is_accepted_as_optional_completed_hull_evidence(self):
        """A bright blue porthole can identify a red-marked ship cell."""
        grid_size = 5
        points = [
            (60 + col * 40, 60 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((280, 280, 3), dtype=np.uint8)
        cell = (2, 2)
        adjacent_hull = (1, 2)
        x, y = points[cell[0] * grid_size + cell[1]]
        # Red submarine decoration above the calibrated point.
        cv2.rectangle(frame, (x - 8, y - 34), (x + 7, y - 26), (0, 0, 255), cv2.FILLED)
        # The supplied screenshots show a small luminous cyan/blue porthole;
        # no neutral-gray hull pixels are present in this cell.
        cv2.ellipse(frame, (x, y), (8, 6), 0, 0, 360, (235, 220, 190), cv2.FILLED)
        hull_x, hull_y = points[adjacent_hull[0] * grid_size + adjacent_hull[1]]
        cv2.ellipse(
            frame,
            (hull_x, hull_y),
            (15, 9),
            0,
            0,
            360,
            (185, 185, 185),
            cv2.FILLED,
        )

        candidates = detect_completed_submarine_candidate_cells(frame, points, grid_size)

        self.assertIn(cell, candidates)
        self.assertIn(adjacent_hull, candidates)

    def test_completed_ship_anchor_does_not_pull_two_cell_water_neighbor(self):
        grid_size = 7
        points = [
            (80 + col * 40, 80 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((420, 420, 3), dtype=np.uint8)
        # Real two-cell hull with its red component on the first cell.
        for cell in ((3, 3), (3, 4)):
            x, y = points[cell[0] * grid_size + cell[1]]
            cv2.ellipse(frame, (x, y), (22, 14), 0, 0, 360, (175, 178, 180), cv2.FILLED)
        x, y = points[3 * grid_size + 3]
        cv2.rectangle(frame, (x - 8, y - 34), (x + 7, y - 26), (0, 0, 255), cv2.FILLED)
        # Bright water two cells away must not be promoted by the anchor.
        noise_cell = (5, 5)
        x, y = points[noise_cell[0] * grid_size + noise_cell[1]]
        cv2.ellipse(frame, (x, y), (22, 14), 0, 0, 360, (190, 190, 190), cv2.FILLED)

        candidates = detect_completed_submarine_candidate_cells(frame, points, grid_size)

        self.assertNotIn(noise_cell, candidates)

    def test_completed_ship_anchor_rejects_weak_diagonal_neighbor(self):
        grid_size = 5
        points = [
            (60 + col * 40, 60 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((280, 280, 3), dtype=np.uint8)
        anchor = (2, 2)
        x, y = points[anchor[0] * grid_size + anchor[1]]
        cv2.rectangle(frame, (x - 8, y - 34), (x + 7, y - 26), (0, 0, 255), cv2.FILLED)
        # A low-contrast diagonal patch should not be promoted to hull body.
        diagonal = (3, 3)
        x, y = points[diagonal[0] * grid_size + diagonal[1]]
        cv2.ellipse(frame, (x, y), (8, 6), 0, 0, 360, (95, 100, 105), cv2.FILLED)

        candidates = detect_completed_submarine_candidate_cells(frame, points, grid_size)

        self.assertNotIn(diagonal, candidates)

    def test_larger_submarine_marker_still_anchors_completed_ship(self):
        """A high-resolution flag can exceed the old 260px component cap."""
        grid_size = 10
        points = [
            (400 + col * 40, 100 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ship_cells = {(4, 4), (4, 5), (4, 6)}
        for row, col in ship_cells:
            x, y = points[row * grid_size + col]
            cv2.ellipse(
                frame,
                (x, y),
                (22, 14),
                0,
                0,
                360,
                (175, 178, 180),
                cv2.FILLED,
            )

        # 27x15 is within the marker shape bounds, but its ~400px area is
        # larger than the old 260px cap and occurs in real game frames.
        x, y = points[4 * grid_size + 5]
        cv2.rectangle(
            frame,
            (x - 8, y - 16),
            (x + 18, y - 2),
            (0, 0, 255),
            cv2.FILLED,
        )

        candidates = detect_completed_submarine_candidate_cells(
            frame,
            points,
            grid_size,
        )

        # The regression is about retaining the anchor; the geometry test above
        # covers expansion from that anchor to every hull cell.
        self.assertIn((4, 5), candidates)

    def test_supplied_large_red_component_calibrates_anchor_size(self):
        """The supplied 33px component variant must not hit the old cap."""
        grid_size = 10
        points = [
            (400 + col * 40, 100 + row * 40)
            for row in range(grid_size)
            for col in range(grid_size)
        ]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cell = (4, 5)
        x, y = points[cell[0] * grid_size + cell[1]]
        # 33x14 / 462px matches the large red component crops supplied from
        # the game.  It exceeded both the previous 28px and 420px guards.
        cv2.rectangle(
            frame,
            (x - 16, y - 20),
            (x + 16, y - 7),
            (0, 0, 255),
            cv2.FILLED,
        )

        marker_cells = detect_red_submarine_marker_cells(frame, points, grid_size)

        self.assertEqual(marker_cells, {cell})
        self.assertFalse(visible_wreck_static_detected(frame, points[45]))

    def test_water_cell_next_to_submarine_is_not_a_static_wreck(self):
        """A neighboring hull must not be pulled into the water cell at (8,7)."""
        sample = (
            Path(__file__).parents[1]
            / "_debug"
            / "screenshots"
            / "probes"
            / "level_8_cell_7_r0_c7_20260831_010220_361557"
            / "before.png"
        )
        frame = cv2.imread(str(sample))
        points = read_saved_points(8, expected_n=10)

        self.assertIsNotNone(frame)
        self.assertFalse(visible_wreck_static_detected(frame, points[87]))


if __name__ == "__main__":
    unittest.main()
