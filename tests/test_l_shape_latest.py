import importlib
import sys
import tempfile
import unittest
from unittest.mock import call, patch

from tests.test_main_flow import FakeAdb


class LatestLShapeRuleTest(unittest.TestCase):
    """Regression coverage for the latest red-scout L-shape rule."""

    def setUp(self):
        FakeAdb.instances.clear()
        self.utils = importlib.import_module("utils")
        self.original_adb_controller = self.utils.AdbController
        self.utils.AdbController = FakeAdb
        sys.modules.pop("main", None)
        self.main = importlib.import_module("main")

        self.runtime_temp = tempfile.TemporaryDirectory()
        runtime_root = self.main.Path(self.runtime_temp.name)
        self.path_patchers = [
            patch.object(self.main, "PROBE_SAMPLE_DIR", runtime_root / "probes"),
            patch.object(
                self.main,
                "RED_SCOUT_SAMPLE_DIR",
                runtime_root / "red_scout_samples",
            ),
            patch.object(self.main, "RUN_DEBUG_DIR", runtime_root / "run_debug"),
            patch.object(self.main, "RUNTIME_DIR", runtime_root / "runtime"),
            patch.object(
                self.main,
                "STATUS_FILE",
                runtime_root / "runtime" / "status.json",
            ),
            patch.object(
                self.main,
                "LEVEL_STATE_FILE",
                runtime_root / "runtime" / "level_state.json",
            ),
        ]
        for patcher in self.path_patchers:
            patcher.start()

        self.pending_patchers = [
            patch.object(self.main, "write_pending_probe"),
            patch.object(self.main, "update_pending_probe", return_value=False),
            patch.object(self.main, "clear_pending_probe"),
            patch.object(self.main, "read_pending_probe", return_value=None),
        ]
        for patcher in self.pending_patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.pending_patchers):
            patcher.stop()
        for patcher in reversed(self.path_patchers):
            patcher.stop()
        self.runtime_temp.cleanup()
        sys.modules.pop("main", None)
        self.utils.AdbController = self.original_adb_controller
        FakeAdb.instances.clear()

    def _red_result(self, cells, *, affected_cells=None, center=(2, 2)):
        cells = frozenset(cells)
        affected = cells if affected_cells is None else frozenset(affected_cells)
        return self.main.RedScoutResult(
            center_cell=center,
            affected_cells=affected,
            hit_cells=cells,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(cells),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in cells},
        )

    def _run_red_case(
        self,
        *,
        initial_hits,
        result_hits,
        submarines,
        initial_completed_visual_hits=None,
        initial_authoritative_completed_visual_hits=None,
        initial_authoritative_completed_placements=None,
        initial_lock_completed_placements=False,
        affected_cells=None,
        center=(2, 2),
        level=1,
        grid_size=3,
    ):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        click_points = [
            (400 + (index % grid_size) * 100, 300 + (index // grid_size) * 100)
            for index in range(grid_size * grid_size)
        ]
        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                return_value=center,
            ),
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=self._red_result(
                    result_hits,
                    affected_cells=affected_cells,
                    center=center,
                ),
            ),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
            patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            # The first status emitted before a blue click is the observable
            # boundary for this regression: the false upper cell must already
            # be a miss at that point.
            completed = self.main._run_red_scout_and_blue_strategy(
                level,
                [[0] * grid_size for _row in range(grid_size)],
                click_points,
                list(submarines),
                set(initial_hits),
                settings,
                initial_completed_visual_hits=(
                    set(initial_completed_visual_hits)
                    if initial_completed_visual_hits is not None
                    else None
                ),
                initial_authoritative_completed_visual_hits=(
                    set(initial_authoritative_completed_visual_hits)
                    if initial_authoritative_completed_visual_hits is not None
                    else None
                ),
                initial_authoritative_completed_placements=(
                    initial_authoritative_completed_placements
                    if initial_authoritative_completed_placements is not None
                    else None
                ),
                initial_lock_completed_placements=initial_lock_completed_placements,
            )
        self.assertTrue(completed)
        return online_hit, scan, write_status

    def test_2x2_l_clears_upper_cell_when_it_was_an_initial_hit(self):
        upper = (0, 1)
        lower = {(1, 1), (1, 2)}

        online_hit, scan, write_status = self._run_red_case(
            initial_hits={upper},
            result_hits=lower,
            affected_cells=lower | {upper},
            submarines=(3,),
        )

        self.assertEqual(
            {call.kwargs["cell"] for call in online_hit.call_args_list},
            lower,
        )
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        first_blue_status = next(
            update
            for update in write_status.call_args_list
            if update.kwargs.get("phase") == "blue_online_scout_hits"
        )
        self.assertEqual(first_blue_status.kwargs["board_states"][0][1], "miss")

    def test_3x3_l_clears_upper_cell_from_completed_visual_snapshot(self):
        upper = (0, 0)
        lower = {(2, 0), (2, 1), (2, 2)}
        visual = lower | {upper}

        online_hit, scan, _write_status = self._run_red_case(
            initial_hits=visual,
            result_hits=lower,
            affected_cells=visual,
            submarines=(3,),
            initial_completed_visual_hits=visual,
            center=(5, 5),
            level=8,
            grid_size=10,
        )

        self.assertFalse(online_hit.called)
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        self.assertEqual(scan.call_args.kwargs["initial_completed_visual_hits"], lower)

    def test_3x3_l_clears_visual_upper_when_red_footprint_only_reports_lower(self):
        upper = (0, 0)
        lower = {(2, 0), (2, 1), (2, 2)}
        visual = lower | {upper}

        online_hit, scan, _write_status = self._run_red_case(
            initial_hits=visual,
            result_hits=lower,
            submarines=(3,),
            initial_completed_visual_hits=visual,
            center=(5, 5),
            level=8,
            grid_size=10,
        )

        self.assertFalse(online_hit.called)
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        self.assertEqual(scan.call_args.kwargs["initial_completed_visual_hits"], lower)

    def test_2x2_l_clears_committed_upper_when_same_snapshot_contains_it(self):
        upper = (0, 1)
        lower = frozenset({(1, 1), (1, 2)})
        first = self._red_result({upper}, center=(0, 0))
        second = self._red_result(
            lower,
            affected_cells=lower | {upper},
            center=(2, 2),
        )
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 2)
        click_points = [
            (400 + (index % 3) * 100, 300 + (index // 3) * 100)
            for index in range(9)
        ]
        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                side_effect=[(0, 0), (2, 2)],
            ),
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                side_effect=[first, second],
            ),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
            patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                click_points,
                [3],
                set(),
                settings,
            )

        self.assertTrue(completed)
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])

    def test_2x2_l_does_not_restore_upper_from_locked_placement(self):
        """L cleanup must survive the completed-ship restoration pass."""
        upper = (0, 1)
        upper_pair = {(0, 1), (0, 2)}
        lower = (1, 2)
        result = self._red_result(
            {lower},
            affected_cells=upper_pair | {lower},
            center=(2, 2),
        )
        placement = self.main.Placement(
            length=2,
            direction="H",
            cells=tuple(sorted(upper_pair)),
        )

        _online_hit, scan, _write_status = self._run_red_case(
            initial_hits=upper_pair,
            result_hits={lower},
            affected_cells=upper_pair | {lower},
            submarines=(2,),
            initial_authoritative_completed_visual_hits=upper_pair,
            initial_authoritative_completed_placements=(placement,),
        )

        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])

    def test_startup_locked_placement_survives_neighbor_noise(self):
        """A startup-confirmed hull must survive a noisy perimeter hit."""
        ship = tuple((row, 8) for row in range(5))
        noise = (1, 7)
        placement = self.main.Placement(length=5, direction="V", cells=ship)

        _online_hit, scan, _write_status = self._run_red_case(
            initial_hits=set(ship) | {noise},
            result_hits=set(),
            affected_cells=set(),
            submarines=(5,),
            initial_completed_visual_hits=set(ship),
            initial_authoritative_completed_visual_hits=set(ship),
            initial_authoritative_completed_placements=(placement,),
            initial_lock_completed_placements=True,
            level=13,
            grid_size=10,
        )

        self.assertEqual(scan.call_args.kwargs["initial_hits"], set(ship))
        self.assertIn(noise, scan.call_args.kwargs["initial_misses"])
        self.assertEqual(
            scan.call_args.kwargs["initial_authoritative_completed_placements"],
            (placement,),
        )

    def test_2x2_l_clears_visual_upper_when_lower_cell_is_already_hit(self):
        """Both upper-pair orientations must clear a visual flag cell.

        The red scout may report only the lower cell while the two upper
        cells are already present as completed-ship visual evidence.  That
        mixed state is the layout shown by the runtime board screenshot.
        """
        for upper_pair, lower, false_upper in (
            ({(0, 1), (0, 2)}, (1, 2), (0, 1)),
            ({(0, 1), (0, 2)}, (1, 1), (0, 2)),
        ):
            result = self._red_result(
                {lower},
                affected_cells={lower},
                center=(2, 2),
            )
            settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
            click_points = [
                (400 + (index % 3) * 100, 300 + (index // 3) * 100)
                for index in range(9)
            ]
            with (
                patch.object(
                    self.main.RedScoutPlanner,
                    "choose_center",
                    return_value=(2, 2),
                ),
                patch.object(
                    self.main,
                    "_execute_red_scout_transaction",
                    return_value=result,
                ),
                patch.object(
                    self.main,
                    "_execute_online_scout_hit",
                    return_value=self.main.ProbeResult.HIT,
                ) as online_hit,
                patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
                patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
            ):
                completed = self.main._run_red_scout_and_blue_strategy(
                    1,
                    [[0] * 3 for _row in range(3)],
                    click_points,
                    [2],
                    {lower},
                    settings,
                    initial_completed_visual_hits=set(upper_pair),
                )

            self.assertTrue(completed)
            self.assertNotIn(false_upper, scan.call_args.kwargs["initial_hits"])
            self.assertIn(false_upper, scan.call_args.kwargs["initial_misses"])
            self.assertNotIn(false_upper, {call.kwargs["cell"] for call in online_hit.call_args_list})

    def test_2x2_l_initial_visual_upper_is_cleared_before_red_scout_blue_attack(self):
        upper_pair = {(0, 1), (0, 2)}
        lower = (1, 2)
        result = self._red_result(set(), affected_cells=set(), center=(2, 0))
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        click_points = [
            (400 + (index % 3) * 100, 300 + (index // 3) * 100)
            for index in range(9)
        ]
        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(2, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(self.main, "_execute_online_scout_hit") as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
            patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                click_points,
                [2],
                upper_pair | {lower},
                settings,
                initial_completed_visual_hits=upper_pair,
            )

        self.assertTrue(completed)
        online_hit.assert_not_called()
        self.assertNotIn((0, 1), scan.call_args.kwargs["initial_hits"])
        self.assertIn((0, 1), scan.call_args.kwargs["initial_misses"])

    def test_2x2_l_all_visual_cells_are_cleared_before_red_scout_blue_attack(self):
        """An all-visual L must be normalized even without a separate live hit."""
        upper = (0, 1)
        lower = {(1, 1), (1, 2)}
        visual = {upper} | lower
        result = self._red_result(set(), affected_cells=set(), center=(2, 0))
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        click_points = [
            (400 + (index % 3) * 100, 300 + (index // 3) * 100)
            for index in range(9)
        ]

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=None),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(self.main, "_execute_online_scout_hit") as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
            patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                click_points,
                [2],
                visual,
                settings,
                initial_completed_visual_hits=visual,
            )

        self.assertTrue(completed)
        online_hit.assert_not_called()
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        self.assertEqual(scan.call_args.kwargs["initial_completed_visual_hits"], lower)

    def test_2x2_l_authoritative_visual_upper_is_cleared_for_both_orientations(self):
        """An authoritative visual upper cell is still a flag artifact."""
        for lower, false_upper in (
            ((1, 2), (0, 1)),
            ((1, 1), (0, 2)),
        ):
            upper_pair = {(0, 1), (0, 2)}
            result = self._red_result(
                {lower},
                affected_cells={lower},
                center=(2, 0),
            )
            settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
            click_points = [
                (400 + (index % 3) * 100, 300 + (index // 3) * 100)
                for index in range(9)
            ]
            with (
                patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(2, 0)),
                patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
                patch.object(self.main, "_execute_online_scout_hit") as online_hit,
                patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
                patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
            ):
                completed = self.main._run_red_scout_and_blue_strategy(
                    1,
                    [[0] * 3 for _row in range(3)],
                    click_points,
                    [2],
                    upper_pair,
                    settings,
                    initial_completed_visual_hits=upper_pair,
                    initial_authoritative_completed_visual_hits=upper_pair,
                )

            self.assertTrue(completed)
            online_hit.assert_not_called()
            self.assertNotIn(false_upper, scan.call_args.kwargs["initial_hits"])
            self.assertIn(false_upper, scan.call_args.kwargs["initial_misses"])
            self.assertEqual(
                scan.call_args.kwargs["initial_completed_visual_hits"],
                upper_pair - {false_upper} | {lower},
            )

    def test_l_normalization_does_not_join_current_hit_with_unrelated_visual_ship(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        current_hit = (0, 5)
        unrelated_visual = {(2, 5), (2, 6), (2, 7)}
        result = self._red_result(
            {current_hit},
            affected_cells={current_hit},
            center=(5, 5),
        )
        click_points = [
            (400 + (index % 10) * 40, 300 + (index // 10) * 40)
            for index in range(100)
        ]
        hit_map = [[0] * 10 for _row in range(10)]
        for row, col in unrelated_visual | {current_hit}:
            hit_map[row][col] = 1

        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                return_value=(5, 5),
            ),
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=result,
            ),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
            patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                8,
                hit_map,
                click_points,
                [3],
                unrelated_visual | {current_hit},
                settings,
                initial_completed_visual_hits=unrelated_visual,
            )

        self.assertTrue(completed)
        self.assertFalse(online_hit.called)
        self.assertIn(current_hit, scan.call_args.kwargs["initial_hits"])
        self.assertNotIn(current_hit, scan.call_args.kwargs["initial_misses"])
        self.assertEqual(hit_map[current_hit[0]][current_hit[1]], 1)
        for row, col in unrelated_visual:
            self.assertEqual(hit_map[row][col], 1)


if __name__ == "__main__":
    unittest.main()
