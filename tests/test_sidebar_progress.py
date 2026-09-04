import importlib
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.submarine_strategy import SubmarineStrategy


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

    def test_completed_ship_resolution_binds_lengths_to_red_anchors(self):
        candidates = {
            (1, 1), (1, 2), (1, 3),
            (5, 5), (6, 5),
        }
        anchors = {(1, 1), (5, 5)}

        resolution = sidebar_progress.resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            completed_lengths=(2, 3),
            grid_size=7,
        )

        self.assertEqual(
            resolution.placements,
            (
                ((1, 1), (1, 2), (1, 3)),
                ((5, 5), (6, 5)),
            ),
        )
        self.assertEqual(resolution.unresolved_lengths, ())

    def test_anchor_resolution_rejects_equal_score_length_swap(self):
        # Both anchors are close enough to both hulls.  The visual evidence
        # alone therefore cannot prove which sidebar length belongs to which
        # red marker; a deterministic tie-break would be unsafe.
        candidates = {
            (3, 3), (3, 4),
            (5, 3), (5, 4), (5, 5),
        }
        anchors = {(4, 4), (4, 5)}

        resolution = sidebar_progress.resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            completed_lengths=(2, 3),
            grid_size=8,
            fallback_to_global=False,
        )

        self.assertEqual(resolution.placements, ())
        self.assertEqual(resolution.unresolved_lengths, (3, 2))

    def test_anchor_resolution_prioritizes_marker_on_own_hull_over_noisy_coverage(self):
        # Level 4 startup replay: the body detector found a dense false line
        # beside the upper marker and omitted one lower-hull cell.  Selecting
        # the most detector-covered combination swaps the 4-cell and 2-cell
        # ships.  Both red markers are directly on the real hulls, which is
        # stronger evidence than the extra body pixels.
        broad_candidates = {
            (0, 0), (0, 2), (0, 3),
            (1, 0), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 2), (2, 3), (2, 5), (3, 4),
            (4, 1), (4, 2), (4, 3), (4, 4),
        }
        strict_candidates = {
            (0, 0),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (4, 1), (4, 3), (4, 4),
        }

        resolution = sidebar_progress.resolve_completed_ship_cells_by_anchors(
            broad_candidates,
            {(1, 0), (4, 2)},
            completed_lengths=(4, 2),
            grid_size=6,
            preferred_cells=strict_candidates,
            fallback_to_global=False,
        )

        self.assertEqual(
            resolution.placements,
            (
                ((0, 0), (1, 0)),
                ((4, 1), (4, 2), (4, 3), (4, 4)),
            ),
        )
        self.assertEqual(resolution.unresolved_lengths, ())

    def test_anchor_resolution_recovers_marker_cell_when_body_support_is_exact(self):
        candidates = {
            (5, 5), (5, 7), (5, 8), (5, 9),
            (8, 3), (8, 5),
        }
        anchors = {(5, 6), (8, 4)}

        resolution = sidebar_progress.resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            completed_lengths=(5, 3),
            grid_size=10,
            fallback_to_global=False,
        )

        self.assertEqual(
            resolution.placements,
            (
                ((5, 5), (5, 6), (5, 7), (5, 8), (5, 9)),
                ((8, 3), (8, 4), (8, 5)),
            ),
        )

    def test_level_22_midgame_log_geometry_matches_runtime_board_states(self):
        # Extracted from the level_22 startup vision evidence captured in the
        # debug log on 2026-09-03.  The red marker cells are omitted by the
        # body detector, while three neighboring cells are visual spillover.
        wreck_candidates = {
            (4, 6), (4, 7), (4, 8),
            (5, 5), (5, 7), (5, 8), (5, 9),
            (8, 3), (8, 5),
        }
        anchors = {(5, 6), (8, 4)}
        expected_ships = {
            (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
            (8, 3), (8, 4), (8, 5),
        }

        resolution = sidebar_progress.resolve_completed_ship_cells_by_anchors(
            wreck_candidates,
            anchors,
            completed_lengths=(5, 3),
            grid_size=10,
            fallback_to_global=False,
        )
        self.assertEqual(
            {cell for placement in resolution.placements for cell in placement},
            expected_ships,
        )

        strategy = SubmarineStrategy(10, (2, 2, 3, 4, 5))
        strategy.restore_confirmed_placements(resolution.placements)
        states = strategy.get_cell_states()
        self.assertEqual(
            {cell for cell in expected_ships if states[cell[0]][cell[1]] == "ship"},
            expected_ships,
        )
        self.assertEqual(
            [states[row][col] for row, col in sorted(wreck_candidates - expected_ships)],
            ["miss"] * 3,
        )

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
        for path, top_left in zip(
            template_paths,
            ((180, 180), (380, 380)),
            strict=True,
        ):
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

    def test_completed_ship_resolution_uses_global_solution_instead_of_greedy_noise(self):
        candidates = {
            (0, 0),
            (1, 0),
            (0, 1),  # False-positive wreck pixel creates a tempting greedy segment.
            (0, 2),
            (0, 3),
        }

        resolution = sidebar_progress.resolve_completed_ship_cells(
            candidates,
            completed_lengths=(2, 2),
            grid_size=4,
        )

        self.assertEqual(resolution.unresolved_lengths, ())
        self.assertEqual(
            resolution.cells,
            frozenset({(0, 0), (1, 0), (0, 2), (0, 3)}),
        )
        self.assertEqual(resolution.discarded_cells, frozenset({(0, 1)}))

    def test_completed_ship_resolution_prefers_new_exact_run_over_old_or_embedded_run(self):
        candidates = {
            (0, 4), (1, 4),
            (2, 6),
            (5, 4), (5, 5), (5, 6), (5, 7),
            (7, 2),
            (7, 6), (7, 7),
        }
        newly_visible = {(5, 7), (7, 2), (7, 6)}

        resolution = sidebar_progress.resolve_completed_ship_cells(
            candidates,
            completed_lengths=(2,),
            grid_size=8,
            preferred_cells=newly_visible,
        )

        self.assertEqual(
            resolution.cells,
            frozenset({(7, 6), (7, 7)}),
        )
        self.assertEqual(resolution.unresolved_lengths, ())

    def test_completed_ship_resolution_drops_upper_visual_overhang(self):
        resolution = sidebar_progress.resolve_completed_ship_cells(
            {(4, 4), (5, 4), (6, 4)},
            completed_lengths=(2,),
            grid_size=8,
        )

        self.assertEqual(resolution.cells, frozenset({(5, 4), (6, 4)}))
        self.assertEqual(resolution.discarded_cells, frozenset({(4, 4)}))


if __name__ == "__main__":
    unittest.main()
