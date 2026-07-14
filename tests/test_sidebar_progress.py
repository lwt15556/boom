import importlib
import unittest
from pathlib import Path

import cv2
import numpy as np


try:
    sidebar_progress = importlib.import_module("utils.sidebar_progress")
except ModuleNotFoundError:
    sidebar_progress = None


class SidebarProgressModuleTest(unittest.TestCase):
    def test_sidebar_progress_module_is_available(self):
        self.assertIsNotNone(sidebar_progress)


@unittest.skipIf(sidebar_progress is None, "sidebar progress module is not implemented yet")
class SidebarProgressTest(unittest.TestCase):
    fleet = (2, 2, 3, 3, 4, 5)

    @staticmethod
    def make_sidebar_image(completed_rows=()):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        active_bgr = cv2.cvtColor(
            np.uint8([[[26, 20, 250]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        complete_bgr = cv2.cvtColor(
            np.uint8([[[99, 99, 108]]]),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        for row_index in range(len(SidebarProgressTest.fleet)):
            center_y = 275 + row_index * 35
            color = complete_bgr if row_index in completed_rows else active_bgr
            image[center_y - 13:center_y + 14, 20:145] = color
        return image

    def test_detects_all_active_rows_from_real_level_reference(self):
        reference = Path(__file__).resolve().parents[1] / "save_points" / "imgs" / "7.png"
        image = cv2.imread(str(reference))

        progress = sidebar_progress.detect_sidebar_progress(image, self.fleet)

        self.assertIsNotNone(progress)
        self.assertTrue(progress.valid)
        self.assertEqual(progress.active_lengths, (5, 4, 3, 3, 2, 2))
        self.assertEqual(progress.completed_lengths, ())
        self.assertEqual(progress.completed_cells, 0)

    def test_maps_completed_rows_to_submarine_lengths(self):
        image = self.make_sidebar_image(completed_rows={1, 5})

        progress = sidebar_progress.detect_sidebar_progress(image, self.fleet)

        self.assertIsNotNone(progress)
        self.assertTrue(progress.valid)
        self.assertEqual(progress.completed_lengths, (4, 2))
        self.assertEqual(progress.completed_cells, 6)

    def test_detects_only_the_newly_completed_submarine(self):
        before = sidebar_progress.detect_sidebar_progress(
            self.make_sidebar_image(completed_rows={5}),
            self.fleet,
        )
        after = sidebar_progress.detect_sidebar_progress(
            self.make_sidebar_image(completed_rows={1, 5}),
            self.fleet,
        )

        completed = sidebar_progress.newly_completed_lengths(before, after)

        self.assertEqual(completed, (4,))

    def test_completed_cells_raise_but_never_reduce_confirmed_hit_count(self):
        progress = sidebar_progress.detect_sidebar_progress(
            self.make_sidebar_image(completed_rows={1, 5}),
            self.fleet,
        )

        self.assertEqual(sidebar_progress.merge_confirmed_hit_count(5, progress), 6)
        self.assertEqual(sidebar_progress.merge_confirmed_hit_count(8, progress), 8)

    def test_counts_distinct_small_wreck_templates_once(self):
        detector = getattr(sidebar_progress, "detect_partial_wreck_cells", None)
        self.assertIsNotNone(detector)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        template_paths = sorted(
            (Path(__file__).resolve().parents[1] / "template").glob("visible_wreck_*.png")
        )[:2]
        for path, top_left in zip(template_paths, ((180, 180), (380, 380))):
            template = cv2.imread(str(path))
            x, y = top_left
            image[y:y + template.shape[0], x:x + template.shape[1]] = template

        cells = detector(
            image,
            [(204, 200), (404, 400)],
            grid_size=2,
            template_paths=template_paths,
        )

        self.assertEqual(cells, {(0, 0), (0, 1)})

    def test_visible_hit_count_adds_completed_ship_cells_and_partial_wrecks(self):
        calculator = getattr(sidebar_progress, "calculate_visible_hit_count", None)
        self.assertIsNotNone(calculator)
        progress = sidebar_progress.SidebarProgress(completed_lengths=(4, 2))

        self.assertEqual(
            calculator(progress, partial_wreck_count=1),
            7,
        )

    def test_progressive_count_adds_only_new_strategy_hits(self):
        calculator = getattr(sidebar_progress, "progressive_hit_count", None)
        self.assertIsNotNone(calculator)
        self.assertEqual(
            calculator(
                initial_visual_hit_count=7,
                initial_strategy_hit_count=6,
                current_strategy_hit_count=7,
            ),
            8,
        )

    def test_resolves_all_completed_ship_cells_from_current_level_seven_frame(self):
        candidates = {
            (0, 3),
            (1, 3),
            (3, 3),
            (3, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
        }

        resolution = sidebar_progress.resolve_completed_ship_cells(
            candidates,
            completed_lengths=(4, 2, 2),
            grid_size=9,
        )

        self.assertEqual(resolution.cells, frozenset(candidates))
        self.assertEqual(resolution.unresolved_lengths, ())
        self.assertEqual(resolution.discarded_cells, frozenset())
        self.assertCountEqual(
            resolution.placements,
            (
                ((5, 4), (6, 4), (7, 4), (8, 4)),
                ((0, 3), (1, 3)),
                ((3, 3), (3, 4)),
            ),
        )

    def test_completed_ships_do_not_use_candidates_touching_another_ship(self):
        completed_cells = {
            (0, 3),
            (1, 3),
            (3, 3),
            (3, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
        }
        candidates = completed_cells | {(2, 4), (6, 3)}

        resolution = sidebar_progress.resolve_completed_ship_cells(
            candidates,
            completed_lengths=(4, 2, 2),
            grid_size=9,
        )

        self.assertEqual(resolution.cells, frozenset(completed_cells))
        self.assertEqual(resolution.unresolved_lengths, ())
        self.assertEqual(
            resolution.discarded_cells,
            frozenset({(2, 4), (6, 3)}),
        )

    def test_discards_neighbor_cell_not_part_of_completed_ships(self):
        completed_cells = {
            (3, 3),
            (3, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
        }
        candidates = completed_cells | {(7, 3)}

        resolution = sidebar_progress.resolve_completed_ship_cells(
            candidates,
            completed_lengths=(4, 2),
            grid_size=9,
        )

        self.assertEqual(resolution.cells, frozenset(completed_cells))
        self.assertEqual(resolution.unresolved_lengths, ())
        self.assertEqual(resolution.discarded_cells, frozenset({(7, 3)}))

    def test_does_not_invent_completed_ship_cells_when_candidates_are_incomplete(self):
        candidates = {(5, 4), (6, 4), (7, 4)}

        resolution = sidebar_progress.resolve_completed_ship_cells(
            candidates,
            completed_lengths=(4,),
            grid_size=9,
        )

        self.assertEqual(resolution.cells, frozenset())
        self.assertEqual(resolution.placements, ())
        self.assertEqual(resolution.unresolved_lengths, (4,))
        self.assertEqual(resolution.discarded_cells, frozenset(candidates))


if __name__ == "__main__":
    unittest.main()
