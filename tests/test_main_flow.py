import importlib
import inspect
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from utils.sidebar_progress import SidebarProgress


class FakeAdb:
    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        FakeAdb.instances.append(self)

    def delay(self, seconds):
        self.calls.append(("delay", seconds))
        return self

    def close_app(self, package_name):
        self.calls.append(("close_app", package_name))

    def wait_until_app_stopped(self, package_name, timeout=3.0, poll_interval=0.1):
        self.calls.append(
            ("wait_until_app_stopped", package_name, timeout, poll_interval)
        )
        return True

    def open_app(self, package_name):
        self.calls.append(("open_app", package_name))
        return self

    def click(self, x, y):
        self.calls.append(("click", x, y))

    def back(self):
        self.calls.append(("back",))

    def read_screenshot(self, output_path=None):
        self.calls.append(("read_screenshot", output_path))
        return object()

    def swipe(self, start_x, start_y, end_x, end_y):
        self.calls.append(("swipe", start_x, start_y, end_x, end_y))
        return self

    def enable_weak_network(self, package_name):
        self.calls.append(("enable_weak_network", package_name))

    def disable_weak_network(self, package_name):
        self.calls.append(("disable_weak_network", package_name))

    def enable_reject_network(self, package_name):
        self.calls.append(("enable_reject_network", package_name))

    def disable_reject_network(self, package_name):
        self.calls.append(("disable_reject_network", package_name))

    def verify_app_network_isolated(self, package_name):
        self.calls.append(("verify_app_network_isolated", package_name))
        return SimpleNamespace(safe=True, detail="isolated")


class DummyMatch:
    def __init__(self, center):
        self.center = center


def dummy_hit_result(state):
    return SimpleNamespace(
        state=state,
        confidence=0.9 if state == "hit" else 0.1,
        score=0.9 if state == "hit" else 0.1,
        changed_ratio=0.2,
        center_gray_ratio=0.2 if state == "hit" else 0.0,
        gray_excess=0.1 if state == "hit" else 0.0,
        component_ratio=0.1 if state == "hit" else 0.0,
        s_drop=20.0 if state == "hit" else 0.0,
        edge_density=0.1,
        rough_center=(400, 300),
        refined_center=(400, 300),
    )


class MainFlowTest(unittest.TestCase):
    def setUp(self):
        FakeAdb.instances.clear()
        self.utils = importlib.import_module("utils")
        self.original_adb_controller = self.utils.AdbController
        self.utils.AdbController = FakeAdb
        sys.modules.pop("main", None)
        self.main = importlib.import_module("main")
        self.adb = self.main.adb
        self.runtime_temp = tempfile.TemporaryDirectory()
        runtime_root = self.main.Path(self.runtime_temp.name)
        self.runtime_path_patchers = [
            patch.object(self.main, "PROBE_SAMPLE_DIR", runtime_root / "probes"),
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
        for patcher in self.runtime_path_patchers:
            patcher.start()
        self.pending_probe_patchers = [
            patch.object(self.main, "write_pending_probe"),
            patch.object(self.main, "update_pending_probe", return_value=False),
            patch.object(self.main, "clear_pending_probe"),
            patch.object(self.main, "read_pending_probe", return_value=None),
        ]
        for patcher in self.pending_probe_patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.pending_probe_patchers):
            patcher.stop()
        for patcher in reversed(self.runtime_path_patchers):
            patcher.stop()
        self.runtime_temp.cleanup()
        sys.modules.pop("main", None)
        self.utils.AdbController = self.original_adb_controller
        FakeAdb.instances.clear()

    def _valid_red_result(self, center=(1, 1)):
        return self.main.RedScoutResult(
            center_cell=center,
            affected_cells=frozenset({(1, 1), (1, 2)}),
            hit_cells=frozenset({(1, 2)}),
            miss_cells=frozenset({(1, 1)}),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset({(0, 0), (0, 1)})),
            valid=True,
            confidence_by_cell={(1, 1): 0.9, (1, 2): 0.9},
        )

    def test_rejected_second_instance_does_not_register_or_run_network_cleanup(self):
        with (
            patch.object(
                self.main,
                "acquire_main_lock",
                side_effect=self.main.AlreadyRunningError("already running"),
            ),
            patch.object(self.main, "register_exit_cleanup") as register_cleanup,
            patch.object(self.main, "cleanup_weak_network") as cleanup_weak,
            patch.object(self.main, "cleanup_reject_network") as cleanup_reject,
            patch.object(self.main, "release_main_lock") as release_lock,
            patch.object(self.main.adb, "ensure_root_shell", create=True) as ensure_root,
        ):
            exit_code = self.main.run_main_entrypoint()

        self.assertEqual(exit_code, 2)
        register_cleanup.assert_not_called()
        cleanup_weak.assert_not_called()
        cleanup_reject.assert_not_called()
        release_lock.assert_not_called()
        ensure_root.assert_not_called()

    def test_red_mode_runs_configured_attempts_then_seeds_strategy(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        results = [self._valid_red_result() for _ in range(3)]
        with (
            patch.object(self.main, "_execute_red_scout_transaction", side_effect=results) as execute,
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                level=1,
                hit_map=[[0] * 3 for _ in range(3)],
                click_points=[(row, col) for row in range(3) for col in range(3)],
                submarines=[3],
                initial_hits=set(),
                settings=settings,
            )

        self.assertTrue(completed)
        self.assertEqual(execute.call_count, 3)
        online_hit.assert_called_once()
        self.assertEqual(execute.call_args.args[4], 3)
        self.assertEqual(execute.call_args.args[5], [(row, col) for row in range(3) for col in range(3)])
        self.assertEqual(scan.call_args.kwargs["initial_hits"], {(1, 2)})
        self.assertEqual(scan.call_args.kwargs["initial_misses"], set())
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], set())
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {(1, 1)})
        self.assertTrue(scan.call_args.kwargs["commit_scout_hits_online"])
        self.assertEqual(write_status.call_args.kwargs["phase"], "blue_attack")
        self.assertEqual(write_status.call_args.kwargs["red_scout_current"], 3)
        self.assertEqual(write_status.call_args.kwargs["red_scout_total"], 3)
        phases = [call.kwargs["phase"] for call in write_status.call_args_list if "phase" in call.kwargs]
        self.assertEqual(phases[-1], "blue_attack")
        self.assertEqual(phases.count("red_scout_capture"), 3)

    def test_red_scout_commits_new_hits_online_before_next_red_attempt(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 2)
        first = self._valid_red_result()
        second = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=frozenset({(0, 0)}),
            hit_cells=frozenset(),
            miss_cells=frozenset({(0, 0)}),
            unknown_cells=frozenset(),
            footprint=first.footprint,
            valid=True,
            confidence_by_cell={(0, 0): 0.9},
        )
        events = []

        def red_attempt(*_args):
            result = first if not events else second
            events.append(("red", result.center_cell))
            return result

        def online_hit(**kwargs):
            events.append(("blue", kwargs["cell"]))
            return self.main.ProbeResult.HIT

        def finish_scan(*_args, **kwargs):
            events.append(("scan", None))
            return True

        with (
            patch.object(self.main, "_execute_red_scout_transaction", side_effect=red_attempt),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                side_effect=online_hit,
            ) as online_commit,
            patch.object(self.main, "_scan_level_by_strategy", side_effect=finish_scan) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                [(400, 300)] * 9,
                [3],
                set(),
                settings,
                initial_visual_hit_count=0,
            )

        self.assertTrue(completed)
        self.assertEqual([event[0] for event in events], ["red", "blue", "red", "scan"])
        self.assertTrue(online_commit.call_args.kwargs["activity_ready"])
        self.assertEqual(scan.call_args.kwargs["initial_hits"], {(1, 2)})
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], set())
        self.assertEqual(scan.call_args.kwargs["initial_visual_hit_count"], 1)

    def test_repeated_red_scout_hit_is_committed_online_only_once(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 2)
        repeated = self._valid_red_result()

        with (
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                side_effect=[repeated, repeated],
            ),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                [(400, 300)] * 9,
                [3],
                set(),
                settings,
            )

        online_hit.assert_called_once()
        self.assertEqual(online_hit.call_args.kwargs["cell"], (1, 2))

    def test_red_scout_false_positive_becomes_real_miss_before_final_scan(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)

        with (
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=self._valid_red_result(),
            ),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.MISS,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                [(400, 300)] * 9,
                [3],
                set(),
                settings,
            )

        self.assertTrue(completed)
        self.assertEqual(scan.call_args.kwargs["initial_hits"], set())
        self.assertEqual(scan.call_args.kwargs["initial_misses"], {(1, 2)})
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], set())
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {(1, 1)})

    def test_red_scout_does_not_commit_initial_visible_hit_again(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)

        with (
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=self._valid_red_result(),
            ),
            patch.object(self.main, "_execute_online_scout_hit") as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                [(400, 300)] * 9,
                [3],
                {(1, 2)},
                settings,
            )

        self.assertTrue(completed)
        online_hit.assert_not_called()
        self.assertEqual(scan.call_args.kwargs["initial_hits"], {(1, 2)})
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], set())

    def test_online_blue_victory_stops_remaining_red_scout_attempts(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        result = self._valid_red_result()

        with (
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=result,
            ) as red_attempt,
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy") as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                [(400, 300)] * 9,
                [3],
                set(),
                settings,
            )

        self.assertTrue(completed)
        red_attempt.assert_called_once()
        online_hit.assert_called_once()
        scan.assert_not_called()

    def test_red_mode_publishes_cumulative_board_after_each_attempt(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 2)
        first = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset({(0, 0), (0, 1)}),
            hit_cells=frozenset({(0, 0)}),
            miss_cells=frozenset({(0, 1)}),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset({(-1, -1), (-1, 0)})),
            valid=True,
            confidence_by_cell={(0, 0): 0.9, (0, 1): 0.9},
        )
        second = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset({(2, 1), (2, 2)}),
            hit_cells=frozenset({(2, 2)}),
            miss_cells=frozenset({(2, 1)}),
            unknown_cells=frozenset(),
            footprint=first.footprint,
            valid=True,
            confidence_by_cell={(2, 1): 0.9, (2, 2): 0.9},
        )

        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                side_effect=[(1, 1), (2, 2)],
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
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [3],
                {(1, 0)},
                settings,
            )

        board_updates = [
            call.kwargs
            for call in write_status.call_args_list
            if call.kwargs.get("phase") == "red_scout_capture"
            and "board_states" in call.kwargs
        ]
        self.assertEqual(len(board_updates), 2)
        first_board = board_updates[0]["board_states"]
        self.assertEqual(first_board[0][0], "scout_hit")
        self.assertEqual(first_board[0][1], "scout_miss")
        self.assertEqual(first_board[1][0], "hit")
        second_board = board_updates[1]["board_states"]
        self.assertEqual(second_board[0][0], "hit")
        self.assertEqual(second_board[0][1], "scout_miss")
        self.assertEqual(second_board[1][0], "hit")
        self.assertEqual(second_board[2][1], "scout_miss")
        self.assertEqual(second_board[2][2], "scout_hit")

    def test_red_result_wiring_passes_full_grid_to_analyzer(self):
        click_points = [(row, col) for row in range(3) for col in range(3)]
        expected = self._valid_red_result()
        with patch.object(
            self.main.RedScoutAnalyzer,
            "analyze",
            return_value=expected,
        ) as analyze:
            result = self.main._analyze_red_result(
                "before", ["after"], click_points, 3, (1, 1)
            )

        self.assertIs(result, expected)
        analyze.assert_called_once_with(
            before_image="before",
            after_images=["after"],
            click_points=click_points,
            grid_size=3,
            center_cell=(1, 1),
            excluded_cells=set(),
            learned_footprint=None,
        )

    def test_red_transaction_capture_does_not_precede_preflight(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        result = self._valid_red_result()
        phases = []

        def execute(*args):
            phases.extend([
                "red_scout_preflight", "red_scout_capture",
                "red_scout_discard", "red_scout_verify_ammo",
            ])
            return result

        with (
            patch.object(self.main, "_execute_red_scout_transaction", side_effect=execute),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )

        self.assertEqual(phases[:2], ["red_scout_preflight", "red_scout_capture"])
        status = write_status.call_args.kwargs
        self.assertEqual(status["phase"], "blue_attack")
        self.assertEqual(status["red_scout_current"], 1)
        self.assertEqual(status["red_scout_total"], 1)

    def test_red_progress_is_zero_when_planner_has_no_center_immediately(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=None),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )
        self.assertEqual(write_status.call_args.kwargs["red_scout_current"], 0)
        self.assertEqual(write_status.call_args.kwargs["red_scout_total"], 3)

    def test_red_progress_counts_transaction_before_later_planner_stop(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        valid = self._valid_red_result()
        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", side_effect=[(1, 1), None]),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=valid),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )
        self.assertEqual(write_status.call_args.kwargs["red_scout_current"], 1)

    def test_red_progress_counts_invalid_transaction(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        invalid = self._valid_red_result()
        invalid = self.main.RedScoutResult(**{**invalid.__dict__, "valid": False})
        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(1, 1)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=invalid),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )
        self.assertEqual(write_status.call_args.kwargs["red_scout_current"], 1)

    def test_red_scout_never_reuses_center_after_invalid_results(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        invalid = self._valid_red_result()
        invalid = self.main.RedScoutResult(
            center_cell=invalid.center_cell,
            affected_cells=frozenset(),
            hit_cells=frozenset(),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=None,
            valid=False,
            confidence_by_cell={},
        )
        centers = []

        def execute(_level, center, *_args):
            centers.append(center)
            return invalid

        with (
            patch.object(self.main, "_execute_red_scout_transaction", side_effect=execute),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                [(0, 0)] * 9,
                [3],
                set(),
                settings,
            )

        self.assertEqual(len(centers), 3)
        self.assertEqual(len(set(centers)), 3)

    def test_red_phase_stops_before_blue_when_victory_appears_during_scout(self):
        completed_result = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset(),
            hit_cells=frozenset(),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=None,
            valid=False,
            confidence_by_cell={},
            level_completed=True,
        )
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 2)

        with (
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=completed_result,
            ),
            patch.object(self.main, "_scan_level_by_strategy") as scan,
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [3],
                set(),
                settings,
            )

        self.assertTrue(completed)
        scan.assert_not_called()
        self.assertEqual(write_status.call_args.kwargs["phase"], "level_complete")

    def test_blue_only_mode_never_enters_red_transaction(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.BLUE_ONLY, 3)
        with (
            patch.object(self.main, "_execute_red_scout_transaction") as execute,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )

        self.assertTrue(completed)
        execute.assert_not_called()
        scan.assert_called_once()
        self.assertNotIn("commit_scout_hits_online", scan.call_args.kwargs)

    def test_online_scout_hit_keeps_network_connected_and_clicks_target_once(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(
                self.main,
                "locate_red_bomb_button",
                return_value=DummyMatch((1100, 660)),
            ),
            patch.object(self.main, "red_bomb_selected", return_value=False),
            patch.object(
                self.main,
                "classify_diamond_hit",
                side_effect=lambda *_args, **_kwargs: dummy_hit_result("hit"),
            ),
            patch.object(self.main, "find_victory_banner", return_value=None),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=True),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                return_value=(False, None, ()),
            ),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(self.main, "write_runtime_status"),
        ):
            result = self.main._execute_online_scout_hit(
                level=1,
                hit_map=hit_map,
                cell=(1, 1),
                point=(640, 360),
                index=4,
                submarines=[3],
            )

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertEqual(result, self.main.ProbeResult.HIT)
        self.assertEqual(hit_map[1][1], 1)
        self.assertEqual(self.adb.calls.count(("click", 640, 360)), 1)
        self.assertEqual(self.adb.calls.count(("click", *self.main.BLUE_BOMB_POINT)), 1)
        self.assertIn(("disable_reject_network", package_name), self.adb.calls)
        self.assertIn(("disable_weak_network", package_name), self.adb.calls)
        self.assertNotIn(("enable_reject_network", package_name), self.adb.calls)
        self.assertNotIn(("enable_weak_network", package_name), self.adb.calls)

    def test_online_scout_hit_ready_fast_path_skips_redundant_waits(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)
        red_match = DummyMatch((1100, 660))

        with (
            patch.object(self.main, "wait_until_occur") as wait_until,
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(self.main, "detect_sidebar_progress", return_value=None),
            patch.object(self.main, "find_template", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "locate_red_bomb_button", return_value=red_match),
            patch.object(self.main, "red_bomb_selected", return_value=False),
            patch.object(
                self.main,
                "classify_diamond_hit",
                side_effect=lambda *_args, **_kwargs: dummy_hit_result("hit"),
            ),
            patch.object(self.main, "find_victory_banner", return_value=None),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=True),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                return_value=(False, None, ()),
            ),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(self.main, "write_runtime_status"),
        ):
            result = self.main._execute_online_scout_hit(
                level=1,
                hit_map=hit_map,
                cell=(1, 1),
                point=(640, 360),
                index=4,
                submarines=[3],
                activity_ready=True,
            )

        self.assertEqual(result, self.main.ProbeResult.HIT)
        wait_until.assert_not_called()
        self.assertNotIn(
            ("delay", self.main.ONLINE_SCOUT_NETWORK_SETTLE_SECONDS),
            self.adb.calls,
        )
        self.assertIn(("delay", 0.1), self.adb.calls)
        self.assertNotIn(
            ("delay", self.main.ONLINE_SCOUT_BLUE_SELECT_SETTLE_SECONDS),
            self.adb.calls,
        )
        self.assertEqual(self.adb.read_screenshot.call_count, 6)

    def test_fast_blue_selection_waits_remaining_window_when_switch_is_slow(self):
        selection_screen = object()
        early_screen = object()
        confirmed_screen = object()
        self.adb.read_screenshot = Mock(side_effect=[early_screen, confirmed_screen])

        with (
            patch.object(
                self.main,
                "locate_red_bomb_button",
                return_value=DummyMatch((1100, 660)),
            ),
            patch.object(
                self.main,
                "red_bomb_selected",
                side_effect=[True, False],
            ) as selected,
        ):
            result = self.main._select_blue_bomb_for_online_scout(
                self.main.Path("unused"),
                selection_screen,
                fast=True,
            )

        self.assertIs(result, confirmed_screen)
        self.assertEqual(selected.call_count, 2)
        self.assertIn(("delay", 0.1), self.adb.calls)
        self.assertIn(("delay", 0.15), self.adb.calls)
        self.assertEqual(self.adb.calls.count(("click", *self.main.BLUE_BOMB_POINT)), 1)

    def test_blue_selection_refuses_to_fire_when_red_button_cannot_be_verified(self):
        with patch.object(self.main, "locate_red_bomb_button", return_value=None):
            with self.assertRaisesRegex(self.main.ProbeNotReadyError, "red bomb button"):
                self.main._select_blue_bomb_for_online_scout(
                    self.main.Path("unused"),
                    object(),
                    fast=True,
                )

        self.assertNotIn(("click", *self.main.BLUE_BOMB_POINT), self.adb.calls)

    def test_standard_blue_selection_stops_when_red_bomb_remains_selected(self):
        selection_screen = object()
        before_screen = object()
        red_match = DummyMatch((1100, 660))
        self.adb.read_screenshot = Mock(return_value=before_screen)

        with (
            patch.object(
                self.main,
                "locate_red_bomb_button",
                return_value=red_match,
            ),
            patch.object(
                self.main,
                "red_bomb_selected",
                return_value=True,
            ) as selected,
        ):
            with self.assertRaises(self.main.ProbeNotReadyError):
                self.main._select_blue_bomb_for_online_scout(
                    self.main.Path("unused"),
                    selection_screen,
                    fast=False,
                )

        selected.assert_called_once_with(before_screen, red_match)
        self.assertIn(
            ("delay", self.main.ONLINE_SCOUT_BLUE_SELECT_SETTLE_SECONDS),
            self.adb.calls,
        )
        self.assertNotIn(("delay", 0.15), self.adb.calls)

    def test_online_scout_hit_victory_frame_completes_level(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((40, 38))),
            patch.object(
                self.main,
                "handle_victory_prompt",
                side_effect=[False, True],
            ) as handle_victory,
            patch.object(
                self.main,
                "locate_red_bomb_button",
                return_value=DummyMatch((1100, 660)),
            ),
            patch.object(self.main, "red_bomb_selected", return_value=False),
            patch.object(
                self.main,
                "classify_diamond_hit",
                side_effect=lambda *_args, **_kwargs: dummy_hit_result("miss"),
            ),
            patch.object(
                self.main,
                "find_victory_banner",
                return_value=DummyMatch((640, 360)),
            ),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                return_value=(False, None, ()),
            ),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(self.main, "write_runtime_status"),
        ):
            result = self.main._execute_online_scout_hit(
                level=1,
                hit_map=hit_map,
                cell=(1, 1),
                point=(640, 360),
                index=4,
                submarines=[3],
            )

        self.assertEqual(result, self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE)
        self.assertEqual(hit_map[1][1], 1)
        self.assertEqual(self.adb.calls.count(("click", 640, 360)), 1)
        self.assertEqual(handle_victory.call_count, 2)

    def test_online_scout_hit_unknown_stops_without_clicking_twice(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)
        weak_hit = dummy_hit_result("hit")
        weak_hit.score = 0.70
        weak_hit.confidence = 0.70
        frame_results = [weak_hit] + [dummy_hit_result("miss") for _ in range(6)]

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(
                self.main,
                "locate_red_bomb_button",
                return_value=DummyMatch((1100, 660)),
            ),
            patch.object(self.main, "red_bomb_selected", return_value=False),
            patch.object(
                self.main,
                "classify_diamond_hit",
                side_effect=frame_results,
            ),
            patch.object(self.main, "find_victory_banner", return_value=None),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(
                self.main,
                "apply_wreck_template_confirmation",
                side_effect=[True, False, False, False, False, False, False],
            ),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                return_value=(False, None, ()),
            ),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(self.main, "write_runtime_status"),
        ):
            with self.assertRaises(self.main.ProbeProtocolError):
                self.main._execute_online_scout_hit(
                    level=1,
                    hit_map=hit_map,
                    cell=(1, 1),
                    point=(640, 360),
                    index=4,
                    submarines=[3],
                )

        self.assertEqual(hit_map[1][1], 0)
        self.assertEqual(self.adb.calls.count(("click", 640, 360)), 1)

    def test_online_scout_hit_uses_extra_frames_without_firing_twice(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)
        weak_hit = dummy_hit_result("hit")
        weak_hit.score = 0.70
        weak_hit.confidence = 0.70
        frame_results = (
            [weak_hit]
            + [dummy_hit_result("miss") for _ in range(3)]
            + [dummy_hit_result("hit"), dummy_hit_result("hit"), dummy_hit_result("miss")]
        )

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(
                self.main,
                "locate_red_bomb_button",
                return_value=DummyMatch((1100, 660)),
            ),
            patch.object(self.main, "red_bomb_selected", return_value=False),
            patch.object(self.main, "classify_diamond_hit", side_effect=frame_results),
            patch.object(self.main, "find_victory_banner", return_value=None),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(
                self.main,
                "apply_wreck_template_confirmation",
                side_effect=[True, False, False, False, True, True, False],
            ),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                return_value=(False, None, ()),
            ),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(self.main, "write_runtime_status"),
        ):
            result = self.main._execute_online_scout_hit(
                level=1,
                hit_map=hit_map,
                cell=(1, 1),
                point=(640, 360),
                index=4,
                submarines=[3],
            )

        self.assertEqual(result, self.main.ProbeResult.HIT)
        self.assertEqual(hit_map[1][1], 1)
        self.assertEqual(self.adb.calls.count(("click", 640, 360)), 1)
        for delay in self.main.SUSPECT_HIT_EXTRA_FRAME_DELAYS:
            self.assertIn(("delay", delay), self.adb.calls)

    def test_online_scout_hit_skips_cell_that_is_already_visible(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(self.main, "locate_red_bomb_button", return_value=None),
            patch.object(self.main, "red_hit_marker_visible", return_value=True),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "classify_diamond_hit") as classify,
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(self.main, "write_runtime_status"),
        ):
            result = self.main._execute_online_scout_hit(
                level=1,
                hit_map=hit_map,
                cell=(1, 1),
                point=(640, 360),
                index=4,
                submarines=[3],
            )

        self.assertEqual(result, self.main.ProbeResult.HIT)
        self.assertEqual(hit_map[1][1], 1)
        self.assertEqual(self.adb.calls.count(("click", 640, 360)), 0)
        classify.assert_not_called()

    def test_online_scout_hit_stops_before_click_when_sidebar_is_complete(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)
        completed_progress = SidebarProgress(completed_lengths=(3,))

        with (
            patch.object(
                self.main,
                "handle_victory_prompt",
                side_effect=[False, True],
            ) as handle_victory,
            patch.object(
                self.main,
                "detect_sidebar_progress",
                return_value=completed_progress,
            ),
            patch.object(self.main, "classify_diamond_hit") as classify,
            patch.object(self.main, "write_runtime_status"),
        ):
            result = self.main._execute_online_scout_hit(
                level=1,
                hit_map=hit_map,
                cell=(1, 1),
                point=(640, 360),
                index=4,
                submarines=[3],
            )

        self.assertEqual(result, self.main.ProbeResult.LEVEL_COMPLETE)
        self.assertNotIn(("click", *self.main.BLUE_BOMB_POINT), self.adb.calls)
        self.assertNotIn(("click", 640, 360), self.adb.calls)
        self.assertEqual(handle_victory.call_count, 2)
        classify.assert_not_called()

    def test_invalid_red_result_still_preserves_classified_observations(self):
        invalid = self._valid_red_result()
        invalid = self.main.RedScoutResult(
            center_cell=invalid.center_cell,
            affected_cells=invalid.affected_cells,
            hit_cells=invalid.hit_cells,
            miss_cells=invalid.miss_cells,
            unknown_cells=invalid.unknown_cells,
            footprint=invalid.footprint,
            valid=False,
            confidence_by_cell=invalid.confidence_by_cell,
        )
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        with (
            patch.object(self.main, "_execute_red_scout_transaction", return_value=invalid),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )
        self.assertEqual(scan.call_args.kwargs["initial_hits"], {(1, 2)})
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], set())
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {(1, 1)})

    def test_red_phase_ends_when_planner_has_no_center(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=None),
            patch.object(self.main, "_execute_red_scout_transaction") as execute,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )
        execute.assert_not_called()
        scan.assert_called_once()

    def test_red_planner_progresses_covered_cells_and_resets_each_level(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 2)
        centers = []

        def execute(level, center, point, index, grid_size, all_click_points):
            centers.append((level, center))
            self.assertEqual(grid_size, 3)
            self.assertEqual(len(all_click_points), 9)
            return self._valid_red_result(center)

        with (
            patch.object(self.main, "_execute_red_scout_transaction", side_effect=execute),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )

        self.assertEqual(len(centers), 4)
        self.assertEqual(centers[0][1], (1, 1))
        self.assertNotEqual(centers[1][1], centers[0][1])
        self.assertEqual(centers[2], centers[0])

    def test_red_planner_keeps_first_valid_footprint_for_later_attempts(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        first = self._valid_red_result()
        second = self.main.RedScoutResult(
            center_cell=(1, 1), affected_cells=frozenset({(1, 1), (2, 1)}),
            hit_cells=frozenset({(2, 1)}), miss_cells=frozenset({(1, 1)}),
            unknown_cells=frozenset(), footprint=self.main.RedFootprint(frozenset({(0, 0), (1, 0)})),
            valid=True, confidence_by_cell={(1, 1): 0.9, (2, 1): 0.9},
        )
        planner_calls = []
        planned_centers = iter(((1, 1), (0, 0), (0, 2)))

        def choose_center(footprint, **_kwargs):
            planner_calls.append(footprint)
            return next(planned_centers)

        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                side_effect=choose_center,
            ),
            patch.object(self.main, "_execute_red_scout_transaction", side_effect=[first, second, second]),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            self.main._run_red_scout_and_blue_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3], set(), settings
            )
        self.assertIsNone(planner_calls[0])
        self.assertIs(planner_calls[1], first.footprint)
        self.assertIs(planner_calls[2], first.footprint)

    def test_handle_level_passes_same_settings_object_to_strategy(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.BLUE_ONLY, 7)
        expected_points = [(row, col) for row in range(3) for col in range(3)]
        with (
            patch.object(self.main.adb, "delay"),
            patch.object(self.main.adb, "read_screenshot", return_value=object()),
            patch.object(self.main, "get_click_points", return_value=(expected_points, None)),
            patch.object(self.main, "get_configured_submarines", return_value=[3]),
            patch.object(self.main, "detect_sidebar_progress", return_value=None),
            patch.object(self.main, "detect_visible_wreck_cells", return_value=set()),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=set()),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", return_value=True) as run,
        ):
            self.main.handle_game_level(1, [[0] * 3 for _ in range(3)], settings=settings)
        self.assertIs(run.call_args.kwargs["settings"], settings)

    def test_scout_observations_are_not_saved_as_real_shots(self):
        strategy = SimpleNamespace(
            shots={(0, 0): True}, blocked_cells=set(), done=True,
            remaining=SimpleNamespace(elements=lambda: iter(())),
            report_result=Mock(), report_scout_results=Mock(),
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
        )
        fake_bar = SimpleNamespace(total=3, n=0, set_postfix_str=lambda *_args, **_kwargs: None)
        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "save_level_shots") as save,
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
        ):
            self.main._scan_level_by_strategy(
                1, [[0] * 3 for _ in range(3)], [(0, 0)] * 9, [3],
                initial_scout_hits={(1, 1)}, initial_scout_misses={(1, 2)},
            )
        for call in save.call_args_list:
            self.assertNotIn((1, 1), call.args[-1])
            self.assertNotIn((1, 2), call.args[-1])

    def test_main_uses_shared_wreck_detection_helpers(self):
        from utils import wreck_detection

        self.assertIs(self.main.red_hit_marker_visible, wreck_detection.red_hit_marker_visible)
        self.assertIs(
            self.main.visible_wreck_static_detected,
            wreck_detection.visible_wreck_static_detected,
        )

    def test_get_click_points_rejects_unsafe_saved_calibration_and_uses_auto(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        quad = np.array(
            [[50, 10], [90, 50], [50, 90], [10, 50]],
            dtype=np.float32,
        )
        auto_points = [
            (50, 24), (63, 37), (76, 50),
            (37, 37), (50, 50), (63, 63),
            (24, 50), (37, 63), (50, 76),
        ]
        unsafe_saved = auto_points.copy()
        unsafe_saved[-1] = (500, 500)

        with (
            patch.object(self.main, "read_saved_points", return_value=unsafe_saved),
            patch.object(self.main, "read_saved_quad", return_value=quad),
            patch.object(
                self.main,
                "detect_diamond_centers",
                return_value=SimpleNamespace(points=auto_points, global_quad=quad),
            ) as detect,
        ):
            points, detected_quad = self.main.get_click_points(1, image)

        self.assertEqual(points, auto_points)
        np.testing.assert_array_equal(detected_quad, quad)
        detect.assert_called_once_with(image, 3)

    def test_get_click_points_stops_before_probe_when_auto_geometry_is_unsafe(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        degenerate_quad = np.array(
            [[10, 10], [20, 20], [30, 30], [40, 40]],
            dtype=np.float32,
        )
        duplicate_points = [(50, 50)] * 9

        with (
            patch.object(self.main, "USE_SAVED_POINTS", False),
            patch.object(
                self.main,
                "detect_diamond_centers",
                return_value=SimpleNamespace(
                    points=duplicate_points,
                    global_quad=degenerate_quad,
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "unsafe grid calibration"),
        ):
            self.main.get_click_points(1, image)

    def test_red_scout_never_clicks_grid_when_isolation_is_unsafe(self):
        self.adb.verify_app_network_isolated = Mock(
            return_value=SimpleNamespace(safe=False, detail="ipv6 unblocked")
        )
        with self.assertRaises(self.main.RedScoutSafetyError):
            self.main._execute_red_scout_transaction(
                level=1, center_cell=(1, 1), point=(100, 200), index=0,
                grid_size=3, all_click_points=[(0, 0)] * 9,
            )
        self.assertNotIn(("click", 100, 200), self.adb.calls)
        self.assertEqual(self.main._network_fail_closed_reason, "ipv6 unblocked")
        self.main.cleanup_weak_network("unsafe preflight")
        self.assertNotIn(("disable_weak_network", self.main.GAME_PACKAGE_NAME), self.adb.calls)

    def test_red_scout_fails_closed_when_network_verification_errors(self):
        self.adb.verify_app_network_isolated = Mock(
            side_effect=RuntimeError("adb unavailable")
        )

        with self.assertRaisesRegex(self.main.RedScoutSafetyError, "adb unavailable"):
            self.main._execute_red_scout_transaction(
                level=1,
                center_cell=(1, 1),
                point=(100, 200),
                index=0,
                grid_size=3,
                all_click_points=[(0, 0)] * 9,
            )

        self.assertNotIn(("click", 100, 200), self.adb.calls)
        self.assertIn("verification failed", self.main._network_fail_closed_reason)

    def test_red_selection_failure_clears_active_probe_without_fail_closed_stop(self):
        with (
            patch.object(self.main, "_capture_red_ammo_state", return_value=("before", "fp", DummyMatch((10, 20)))),
            patch.object(self.main, "_select_red_bomb", side_effect=RuntimeError("selection read failed")),
            patch.object(self.main, "_stop_and_latch_red_safety_failure") as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "selection read failed"):
                self.main._execute_red_scout_transaction(
                    1, (1, 1), (100, 200), 0, 3, [(0, 0)] * 9
                )
        self.assertIsNone(self.main._active_probe)
        stop.assert_not_called()

    def test_red_system_back_failure_after_grid_click_stops_fail_closed(self):
        with (
            patch.object(
                self.main,
                "_capture_red_ammo_state",
                return_value=("before", "fp", DummyMatch((10, 20))),
            ),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(
                self.main,
                "_wait_until_activity_detail_closed",
                return_value=False,
            ),
            patch.object(
                self.main,
                "_stop_and_latch_red_safety_failure",
                side_effect=self.main.RedScoutSafetyError("fail closed"),
            ) as stop,
        ):
            with self.assertRaisesRegex(self.main.RedScoutSafetyError, "fail closed"):
                self.main._execute_red_scout_transaction(
                    1, (1, 1), (100, 200), 0, 3, [(0, 0)] * 9
                )

        stop.assert_called_once()
        self.assertIn("system back did not exit", stop.call_args.args[0])
        self.assertIsNotNone(self.main._active_probe)
        self.assertTrue(self.main._active_probe.request_may_be_pending)

    def test_red_safety_stop_latches_before_failing_safety_operations(self):
        self.adb.enable_reject_network = Mock(side_effect=RuntimeError("reject failed"))
        self.adb.close_app = Mock(side_effect=RuntimeError("close failed"))
        self.adb.wait_until_app_stopped = Mock(side_effect=RuntimeError("wait failed"))
        self.adb.delay = Mock(side_effect=RuntimeError("delay failed"))
        with self.assertRaisesRegex(self.main.RedScoutSafetyError, "process did not exit"):
            self.main._stop_and_latch_red_safety_failure("first reason")
        self.assertEqual(self.main._network_fail_closed_reason, "first reason")

    def test_red_failure_keeps_isolated_when_process_does_not_exit(self):
        def record_wait_until_app_stopped(package_name, timeout=3.0, poll_interval=0.1):
            self.adb.calls.append(
                ("wait_until_app_stopped", package_name, timeout, poll_interval)
            )
            return False

        self.adb.wait_until_app_stopped = Mock(side_effect=record_wait_until_app_stopped)
        with self.assertRaises(self.main.RedScoutSafetyError) as raised:
            self.main._stop_and_latch_red_safety_failure("result capture failed")
        self.assertIn("process did not exit", str(raised.exception))
        names = [call[0] for call in self.adb.calls]
        self.assertEqual(
            names[:4],
            [
                "enable_reject_network",
                "close_app",
                "wait_until_app_stopped",
                "delay",
            ],
        )
        self.adb.wait_until_app_stopped.assert_called_once()
        self.assertNotIn("enable_weak_network", names)
        self.assertNotIn("disable_weak_network", names)
        self.assertIsNotNone(self.main._network_fail_closed_reason)

    def test_red_scout_discards_request_before_ammo_verification(self):
        analysis = self.main.RedScoutResult(
            center_cell=(1, 1), affected_cells=frozenset({(1, 1), (1, 2)}),
            hit_cells=frozenset({(1, 2)}), miss_cells=frozenset({(1, 1)}),
            unknown_cells=frozenset(), footprint=self.main.RedFootprint(frozenset({(0, 0), (0, 1)})),
            valid=True, confidence_by_cell={(1, 1): 0.9, (1, 2): 0.9},
        )
        events = []

        def exit_red_activity(*_args, **_kwargs):
            events.append("system_back_exit")
            return True

        def reenter_activity(*_args, **_kwargs):
            events.append("reenter_activity")
            return False

        def capture_result_frames():
            events.append("capture_result")
            return ["after"]

        def discard(tx):
            tx.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            tx.red_request_discarded = True
            tx.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            tx.advance(self.main.ProbePhase.COMPLETE)
            return False
        with (
            patch.object(self.main, "_capture_red_ammo_state", side_effect=[("before", "fp", DummyMatch((10, 20))), ("after", "fp", DummyMatch((10, 20)))]),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(
                self.main,
                "_exit_activity_after_probe_click",
                side_effect=exit_red_activity,
            ) as exit_activity,
            patch.object(self.main, "enter_activity", side_effect=reenter_activity),
            patch.object(self.main, "_capture_red_result_frames", side_effect=capture_result_frames),
            patch.object(self.main, "_analyze_red_result", return_value=analysis),
            patch.object(self.main, "_discard_pending_request_and_prepare_next_probe", side_effect=discard) as discard_mock,
            patch.object(self.main, "_commit_hit_request_and_prepare_next_probe") as commit_mock,
            patch.object(self.main, "ammo_fingerprint_matches", return_value=True),
            patch.object(self.main, "write_pending_probe") as write_pending,
            patch.object(self.main, "update_pending_probe") as update_pending,
            patch.object(self.main, "clear_pending_probe") as clear_pending,
            patch.object(self.main, "restart_process", return_value=False),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            result = self.main._execute_red_scout_transaction(
                level=1, center_cell=(1, 1), point=(100, 200), index=0,
                grid_size=3, all_click_points=[(0, 0)] * 9,
            )
        self.assertIs(result, analysis)
        write_pending.assert_called_once()
        update_pending.assert_called()
        clear_pending.assert_called_once()
        discard_mock.assert_called_once()
        commit_mock.assert_not_called()
        exit_activity.assert_called_once_with(
            self.main.RUN_DEBUG_DIR / "red_debug_back.png",
            use_system_back=True,
        )
        self.assertEqual(
            events,
            ["system_back_exit", "reenter_activity", "capture_result"],
        )
        phases = [call.kwargs["phase"] for call in write_status.call_args_list if "phase" in call.kwargs]
        self.assertEqual(
            phases,
            ["red_scout_preflight", "red_scout_capture", "red_scout_discard", "red_scout_verify_ammo"],
        )

    def test_red_pending_marker_is_written_before_target_click(self):
        events = []
        analysis = self._valid_red_result()

        def write_pending(**_kwargs):
            events.append("pending_written")

        def click(x, y):
            if (x, y) == (100, 200):
                events.append("target_clicked")

        def discard(transaction, **_kwargs):
            events.append("request_discarded")
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.red_request_discarded = True
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        self.adb.click = Mock(side_effect=click)
        with (
            patch.object(self.main, "_capture_red_ammo_state", side_effect=[("before", "fp", DummyMatch((10, 20))), ("after", "fp", DummyMatch((10, 20)))]),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=False),
            patch.object(self.main, "_capture_red_result_frames", return_value=["after"]),
            patch.object(self.main, "_analyze_red_result", return_value=analysis),
            patch.object(self.main, "_discard_pending_request_and_prepare_next_probe", side_effect=discard),
            patch.object(self.main, "ammo_fingerprint_matches", return_value=True),
            patch.object(self.main, "write_pending_probe", side_effect=write_pending),
            patch.object(self.main, "update_pending_probe"),
            patch.object(
                self.main,
                "clear_pending_probe",
                side_effect=lambda: events.append("pending_cleared"),
            ),
        ):
            self.main._execute_red_scout_transaction(
                1, (1, 1), (100, 200), 0, 3, [(0, 0)] * 9
            )

        self.assertLess(events.index("pending_written"), events.index("target_clicked"))
        self.assertLess(events.index("request_discarded"), events.index("pending_cleared"))

    def test_red_local_victory_is_discarded_and_not_reported_as_level_complete(self):
        def discard(transaction, **_kwargs):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.red_request_discarded = True
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(self.main, "_capture_red_ammo_state", side_effect=[("before", "fp", DummyMatch((10, 20))), ("after", "fp", DummyMatch((10, 20)))]),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=True),
            patch.object(self.main, "_discard_pending_request_and_prepare_next_probe", side_effect=discard),
            patch.object(self.main, "ammo_fingerprint_matches", return_value=True),
            patch.object(self.main, "write_pending_probe"),
            patch.object(self.main, "update_pending_probe"),
            patch.object(self.main, "clear_pending_probe") as clear_pending,
            patch.object(self.main, "_analyze_red_result") as analyze,
        ):
            result = self.main._execute_red_scout_transaction(
                1, (1, 1), (100, 200), 0, 3, [(0, 0)] * 9
            )

        self.assertFalse(result.level_completed)
        self.assertFalse(result.valid)
        clear_pending.assert_called_once()
        analyze.assert_not_called()

    def test_startup_recovery_force_stops_stale_pending_request_before_cleanup(self):
        events = []
        package_name = self.main.GAME_PACKAGE_NAME

        self.adb.enable_weak_network = Mock(
            side_effect=lambda package: events.append(("enable_drop", package))
        )
        self.adb.enable_reject_network = Mock(
            side_effect=lambda package: events.append(("enable_reject", package))
        )
        self.adb.delay = Mock(
            side_effect=lambda seconds: events.append(("delay", seconds)) or self.adb
        )
        self.adb.close_app = Mock(
            side_effect=lambda package: events.append(("close_app", package))
        )
        self.adb.wait_until_app_stopped = Mock(
            side_effect=lambda package, timeout, poll_interval: events.append(
                ("wait_stopped", package, timeout, poll_interval)
            ) or True
        )

        with (
            patch.object(
                self.main,
                "read_pending_probe",
                return_value={"mode": "red_scout", "phase": "REQUEST_PENDING"},
            ),
            patch.object(
                self.main,
                "clear_pending_probe",
                side_effect=lambda: events.append(("clear_pending",)),
            ),
            patch.object(self.main, "write_runtime_status"),
        ):
            recovered = self.main.recover_interrupted_probe_at_startup()

        self.assertTrue(recovered)
        self.assertEqual(
            events,
            [
                ("enable_drop", package_name),
                ("enable_reject", package_name),
                ("delay", self.main.PROBE_DROP_SETTLE_SECONDS),
                ("close_app", package_name),
                (
                    "wait_stopped",
                    package_name,
                    self.main.APP_STOP_TIMEOUT_SECONDS,
                    self.main.APP_STOP_POLL_SECONDS,
                ),
                ("delay", self.main.POST_FORCE_STOP_GUARD_SECONDS),
                ("clear_pending",),
            ],
        )

    def test_red_result_capture_uses_blue_frame_schedule(self):
        frames = [object() for _ in self.main.HIT_RESULT_FRAME_DELAYS]
        captured_paths = []

        def read_screenshot(path):
            captured_paths.append(path)
            return frames[len(captured_paths) - 1]

        self.adb.read_screenshot = Mock(side_effect=read_screenshot)

        result = self.main._capture_red_result_frames()

        self.assertEqual(result, frames)
        self.assertEqual(
            [call for call in self.adb.calls if call[0] == "delay"],
            [("delay", delay) for delay in self.main.HIT_RESULT_FRAME_DELAYS],
        )
        self.assertEqual(
            captured_paths,
            [
                self.main.RUN_DEBUG_DIR / f"red_result_{index}.png"
                for index in range(len(self.main.HIT_RESULT_FRAME_DELAYS))
            ],
        )

    def test_exit_activity_waits_until_detail_is_gone(self):
        frames = [
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.zeros((20, 20, 3), dtype=np.uint8),
        ]
        self.adb.read_screenshot = Mock(side_effect=frames)

        with (
            patch.object(self.main, "click_template", return_value=True),
            patch.object(
                self.main,
                "find_template",
                side_effect=[DummyMatch((40, 38)), None, None],
            ),
            patch.object(self.main, "sleep"),
        ):
            self.main._exit_activity_after_probe_click(
                self.main.RUN_DEBUG_DIR / "red_debug_quit.png"
            )

        self.assertEqual(self.adb.read_screenshot.call_count, 3)

    def test_exit_activity_retries_quit_when_first_click_is_ignored(self):
        with (
            patch.object(self.main, "click_template", return_value=True) as click_quit,
            patch.object(
                self.main,
                "_wait_until_activity_detail_closed",
                side_effect=[False, True],
            ),
        ):
            self.main._exit_activity_after_probe_click(
                self.main.RUN_DEBUG_DIR / "red_debug_quit.png"
            )

        self.assertEqual(click_quit.call_count, 2)
        self.assertNotIn(("back",), self.adb.calls)

    def test_red_exit_uses_system_back_instead_of_quit_template(self):
        with (
            patch.object(self.main, "click_template") as click_quit,
            patch.object(
                self.main,
                "_wait_until_activity_detail_closed",
                side_effect=[False, True],
            ),
        ):
            self.main._exit_activity_after_probe_click(
                self.main.RUN_DEBUG_DIR / "red_debug_back.png",
                use_system_back=True,
            )

        self.assertEqual(self.adb.calls.count(("back",)), 2)
        click_quit.assert_not_called()

    def test_re_enter_does_not_accept_stale_activity_detail_fast_path(self):
        screenshot = np.zeros((20, 20, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)
        waits = iter(
            [
                DummyMatch((1249, 269)),
                DummyMatch((40, 38)),
            ]
        )

        with (
            patch.object(
                self.main,
                "find_template",
                return_value=DummyMatch((40, 38)),
            ),
            patch.object(
                self.main,
                "wait_until_occur",
                side_effect=lambda *args, **kwargs: next(waits),
            ),
        ):
            self.main.enter_activity(re_enter=True, max_retries=1)

        self.assertIn(("click", 1249, 269), self.adb.calls)
        self.assertIn(("click", 1205, 644), self.adb.calls)

    def test_re_enter_returns_level_complete_when_victory_replaces_detail(self):
        screenshot = np.zeros((20, 20, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)
        waits = iter([DummyMatch((1249, 269)), None])
        completed = False

        with (
            patch.object(
                self.main,
                "wait_until_occur",
                side_effect=lambda *args, **kwargs: next(waits),
            ),
            patch.object(
                self.main,
                "handle_victory_prompt",
                return_value=True,
            ) as handle_victory,
        ):
            try:
                completed = self.main.enter_activity(re_enter=True, max_retries=1)
            except self.main.ProbeProtocolError:
                pass

        self.assertTrue(completed)
        handle_victory.assert_called_once_with(
            timeout=0.0,
            screenshot=screenshot,
            restore_network=False,
        )

    def test_pending_blue_probe_victory_commits_final_hit_and_completes_level(self):
        hit_map = [[0, 0], [0, 0]]

        def commit(transaction):
            transaction.advance(self.main.ProbePhase.REQUEST_COMMITTED)
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(
                self.main,
                "wait_until_occur",
                return_value=DummyMatch((40, 38)),
            ),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(
                self.main,
                "_reenter_activity_for_probe_result",
                return_value=True,
            ),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(
                self.main,
                "classify_diamond_hit",
                return_value=dummy_hit_result("miss"),
            ) as classify,
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
            patch.object(self.main, "get_configured_submarines", return_value=[]),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
            ) as discard_request,
            patch.object(
                self.main,
                "_commit_hit_request_and_prepare_next_probe",
                side_effect=commit,
            ) as commit_request,
        ):
            result = self.main._execute_probe_transaction(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertEqual(result, self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE)
        self.assertEqual(hit_map, [[0, 1], [0, 0]])
        classify.assert_not_called()
        discard_request.assert_not_called()
        commit_request.assert_called_once()

    def test_victory_banner_during_blue_result_frames_confirms_final_hit(self):
        hit_map = [[0, 0], [0, 0]]
        before = np.zeros((720, 1280, 3), dtype=np.uint8)
        victory_frame = np.ones((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(
            side_effect=[before, victory_frame, victory_frame, victory_frame, victory_frame]
        )

        def commit(transaction):
            transaction.advance(self.main.ProbePhase.REQUEST_COMMITTED)
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        def discard(transaction):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(
                self.main,
                "wait_until_occur",
                return_value=DummyMatch((40, 38)),
            ),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(
                self.main,
                "_reenter_activity_for_probe_result",
                return_value=False,
            ),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(
                self.main,
                "classify_diamond_hit",
                return_value=dummy_hit_result("miss"),
            ),
            patch.object(self.main, "find_victory_banner", return_value=DummyMatch((640, 360))),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
            patch.object(self.main, "get_configured_submarines", return_value=[]),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard,
            ) as discard_request,
            patch.object(
                self.main,
                "_commit_hit_request_and_prepare_next_probe",
                side_effect=commit,
            ) as commit_request,
        ):
            result = self.main._execute_probe_transaction(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertEqual(result, self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE)
        self.assertEqual(hit_map, [[0, 1], [0, 0]])
        discard_request.assert_not_called()
        commit_request.assert_called_once()

    def test_pending_victory_click_keeps_network_isolated(self):
        screenshot = np.zeros((20, 20, 3), dtype=np.uint8)

        with patch.object(
            self.main,
            "find_victory_banner",
            return_value=DummyMatch((10, 10)),
        ):
            handled = self.main.handle_victory_prompt(
                timeout=0.0,
                screenshot=screenshot,
                restore_network=False,
            )

        self.assertTrue(handled)
        self.assertIn(("click", *self.main.SCREEN_CONTINUE_POINT), self.adb.calls)
        self.assertNotIn(
            ("disable_reject_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )
        self.assertNotIn(
            ("disable_weak_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )

    def test_victory_handler_refuses_network_restore_while_probe_is_pending(self):
        transaction = self.main.ProbeTransaction(level=1, cell=(0, 0), index=0)
        transaction.advance(self.main.ProbePhase.REQUEST_PENDING)
        self.main._active_probe = transaction
        screenshot = np.zeros((20, 20, 3), dtype=np.uint8)

        with (
            patch.object(
                self.main,
                "find_victory_banner",
                return_value=DummyMatch((10, 10)),
            ),
            self.assertRaisesRegex(self.main.ProbeProtocolError, "待提交"),
        ):
            self.main.handle_victory_prompt(
                timeout=0.0,
                screenshot=screenshot,
                restore_network=True,
            )

        self.assertNotIn(
            ("disable_reject_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )
        self.assertNotIn(
            ("disable_weak_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )
        self.assertNotIn(("click", *self.main.SCREEN_CONTINUE_POINT), self.adb.calls)

    def test_next_level_retry_never_blind_clicks_grid_without_victory_banner(self):
        with (
            patch.object(self.main, "LEVEL_ADVANCE_RETRIES", 1),
            patch.object(
                self.main,
                "resolve_current_level_from_device",
                return_value=7,
            ),
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(self.main, "enter_activity"),
        ):
            next_level = self.main.resolve_next_level_with_retries(
                current_level=7,
                fallback_level=8,
            )

        self.assertIsNone(next_level)
        self.assertNotIn(("click", *self.main.SCREEN_CONTINUE_POINT), self.adb.calls)

    def test_enter_activity_recovers_after_activity_button_missing(self):
        waits = iter(
            [
                None,
                DummyMatch((10, 20)),
                DummyMatch((30, 40)),
                DummyMatch((50, 60)),
            ]
        )
        activity_wait_timeouts = []

        def wait_for_template(template, *args, **kwargs):
            if template == self.main.ACTIVITY_BUTTON_TEMPLATE:
                activity_wait_timeouts.append(kwargs.get("timeout"))
            return next(waits)

        with patch.object(
            self.main,
            "wait_until_occur",
            side_effect=wait_for_template,
        ):
            self.main.enter_activity(max_retries=2)

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertEqual(self.adb.calls.count(("close_app", package_name)), 1)
        self.assertEqual(self.adb.calls.count(("open_app", package_name)), 1)
        self.assertIn(("click", 10, 20), self.adb.calls)
        self.assertIn(("click", 30, 40), self.adb.calls)
        self.assertIn(("click", 1205, 644), self.adb.calls)
        self.assertEqual(self.adb.calls.count(("enable_weak_network", package_name)), 1)
        self.assertEqual(
            [
                call
                for call in self.adb.calls
                if call == ("swipe", 1000, 660, 1000, 180)
            ],
            [
                ("swipe", 1000, 660, 1000, 180),
                ("swipe", 1000, 660, 1000, 180),
            ],
        )
        self.assertEqual(
            activity_wait_timeouts,
            [
                self.main.ACTIVITY_BUTTON_WAIT_SECONDS,
                self.main.POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
            ],
        )

    def test_miss_restart_waits_longer_for_activity_after_login(self):
        login = DummyMatch((638, 592))

        with (
            patch.object(self.main, "wait_until_occur", return_value=login),
            patch.object(self.main, "enter_activity") as enter_activity,
        ):
            completed = self.main.restart_process(
                reopen_game=True,
                app_already_closed=True,
            )

        self.assertFalse(completed)
        self.assertIn(("click", 638, 592), self.adb.calls)
        enter_activity.assert_called_once_with(
            activity_button_timeout=self.main.POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        )

    def test_enter_activity_stops_after_max_retries(self):
        with patch.object(self.main, "wait_until_occur", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "最大重试次数 2"):
                self.main.enter_activity(max_retries=2)

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertEqual(self.adb.calls.count(("close_app", package_name)), 2)
        self.assertEqual(self.adb.calls.count(("open_app", package_name)), 2)

    def test_re_enter_skips_first_enter_only_actions(self):
        waits = iter(
            [
                DummyMatch((30, 40)),
                DummyMatch((50, 60)),
            ]
        )

        with patch.object(
            self.main,
            "wait_until_occur",
            side_effect=lambda *args, **kwargs: next(waits),
        ):
            self.main.enter_activity(re_enter=True, max_retries=1)

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertNotIn(("enable_weak_network", package_name), self.adb.calls)
        self.assertNotIn(("swipe", 1000, 660, 1000, 180), self.adb.calls)
        self.assertIn(("click", 30, 40), self.adb.calls)
        self.assertIn(("click", 1205, 644), self.adb.calls)

    def test_re_enter_failure_does_not_use_normal_restart_recovery(self):
        with patch.object(self.main, "wait_until_occur", return_value=None):
            with self.assertRaisesRegex(
                self.main.ProbeProtocolError,
                "第二次进入活动",
            ):
                self.main.enter_activity(re_enter=True, max_retries=1)

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertNotIn(("close_app", package_name), self.adb.calls)
        self.assertNotIn(("open_app", package_name), self.adb.calls)
        self.assertNotIn(("disable_weak_network", package_name), self.adb.calls)

    def test_enter_activity_reports_victory_detected_during_recovery(self):
        with (
            patch.object(
                self.adb,
                "read_screenshot",
                return_value=np.zeros((20, 20, 3), dtype=np.uint8),
            ),
            patch.object(
                self.main,
                "find_template",
                side_effect=[None, DummyMatch((40, 38))],
            ),
            patch.object(self.main, "handle_victory_prompt", return_value=True),
        ):
            completed = self.main.enter_activity(max_retries=2)

        self.assertTrue(completed)

    def test_preflight_victory_stops_before_retrying_old_level_cell(self):
        hit_map = [[0, 0], [0, 0]]

        with (
            patch.object(
                self.main,
                "_execute_probe_transaction",
                side_effect=self.main.ProbeNotReadyError("胜利界面正在切换"),
            ) as execute,
            patch.object(self.main, "enter_activity", return_value=True) as recover,
        ):
            result = self.main._probe_cell(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertEqual(result, self.main.ProbeResult.LEVEL_COMPLETE)
        execute.assert_called_once()
        recover.assert_called_once_with()

    def test_level_status_reset_replaces_previous_level_board(self):
        self.main._runtime_status.update(
            level=7,
            hits=19,
            board_size=9,
            board_states=[["hit"] * 9 for _row in range(9)],
            sidebar_completed_lengths=[5, 4, 3],
        )

        self.main.reset_runtime_level_status(8)

        status = self.main._runtime_status
        self.assertEqual(status["phase"], "level_loading")
        self.assertEqual(status["level"], 8)
        self.assertEqual(status["board_size"], 10)
        self.assertEqual(len(status["board_states"]), 10)
        self.assertTrue(
            all(cell == "unknown" for row in status["board_states"] for cell in row)
        )
        self.assertEqual(status["hits"], 0)
        self.assertEqual(status["sidebar_completed_lengths"], [])

    def test_cleanup_keeps_drop_when_probe_request_may_be_pending(self):
        transaction = self.main.ProbeTransaction(level=1, cell=(0, 0), index=0)
        transaction.advance(self.main.ProbePhase.REQUEST_PENDING)
        self.main._active_probe = transaction

        self.main.cleanup_weak_network("测试清理")

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertNotIn(("disable_weak_network", package_name), self.adb.calls)
        self.assertFalse(self.main._weak_network_cleanup_done)

    def test_cleanup_keeps_reject_when_probe_request_may_be_pending(self):
        transaction = self.main.ProbeTransaction(level=1, cell=(0, 0), index=0)
        transaction.advance(self.main.ProbePhase.REQUEST_PENDING)
        self.main._active_probe = transaction

        self.main.cleanup_reject_network("测试清理")

        self.assertNotIn(
            ("disable_reject_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )

    def test_cleanup_keeps_reject_when_network_is_fail_closed(self):
        self.main.latch_network_fail_closed("safety state unknown")

        self.main.cleanup_reject_network("测试清理")

        self.assertNotIn(
            ("disable_reject_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )

    def test_connection_dialog_restores_drop_and_reject_before_retry(self):
        retry = DummyMatch((320, 240))
        with (
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                return_value=DummyMatch((100, 100)),
            ),
            patch.object(self.main, "wait_until_retry_button", return_value=retry),
        ):
            handled = self.main.handle_connection_interrupted_prompt(timeout=8.0)

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertTrue(handled)
        self.assertLess(
            self.adb.calls.index(("disable_weak_network", package_name)),
            self.adb.calls.index(("disable_reject_network", package_name)),
        )
        self.assertLess(
            self.adb.calls.index(("disable_reject_network", package_name)),
            self.adb.calls.index(("click", *retry.center)),
        )

    def test_connection_dialog_never_restores_network_for_pending_probe(self):
        transaction = self.main.ProbeTransaction(level=1, cell=(0, 0), index=0)
        transaction.advance(self.main.ProbePhase.REQUEST_PENDING)
        self.main._active_probe = transaction

        with (
            patch.object(self.main, "wait_until_connection_interrupted_dialog") as dialog,
            self.assertRaisesRegex(self.main.ProbeProtocolError, "待提交"),
        ):
            self.main.handle_connection_interrupted_prompt(timeout=8.0)

        dialog.assert_not_called()
        self.assertNotIn(
            ("disable_weak_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )
        self.assertNotIn(
            ("disable_reject_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )

    def test_runtime_status_retries_when_windows_reader_temporarily_locks_file(self):
        replace = patch.object(
            self.main.Path,
            "replace",
            side_effect=[PermissionError(5, "locked"), None],
        )
        with replace as replace_mock, patch.object(self.main, "sleep"):
            self.main.write_runtime_status(test_lock_retry=True)

        self.assertEqual(replace_mock.call_count, 2)

    def test_probe_sample_retention_removes_only_old_managed_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self.main.Path(temp_dir)
            managed = []
            for index in range(3):
                directory = root / f"level_1_cell_{index}_sample"
                directory.mkdir()
                (directory / "status.json").write_text("{}", encoding="utf-8")
                os.utime(directory, (index + 1, index + 1))
                managed.append(directory)
            unrelated = root / "manual_reference"
            unrelated.mkdir()
            (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

            with patch.object(self.main, "PROBE_SAMPLE_DIR", root):
                self.main._prune_probe_sample_dirs(max_directories=2)

            self.assertFalse(managed[0].exists())
            self.assertTrue(managed[1].exists())
            self.assertTrue(managed[2].exists())
            self.assertTrue(unrelated.exists())

    def test_loose_wreck_template_alone_does_not_promote_miss_to_hit(self):
        result = dummy_hit_result("miss")

        with (
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
        ):
            confirmed = self.main.apply_wreck_template_confirmation(
                object(),
                (400, 300),
                result,
            )

        self.assertFalse(confirmed)
        self.assertEqual(result.state, "miss")

    def test_new_sidebar_completion_promotes_miss_to_hit(self):
        confirmation = getattr(self.main, "apply_sidebar_completion_confirmation", None)
        self.assertIsNotNone(confirmation)
        before = SidebarProgress(
            active_lengths=(5, 4, 3, 3, 2),
            completed_lengths=(2,),
        )
        after = SidebarProgress(
            active_lengths=(5, 3, 3, 2),
            completed_lengths=(4, 2),
        )
        result = dummy_hit_result("miss")

        with patch.object(
            self.main,
            "detect_sidebar_progress",
            side_effect=[before, after],
        ):
            confirmed, progress, newly_completed = confirmation(
                object(),
                object(),
                (2, 2, 3, 3, 4, 5),
                result,
            )

        self.assertTrue(confirmed)
        self.assertEqual(progress, after)
        self.assertEqual(newly_completed, (4,))
        self.assertEqual(result.state, "hit")
        self.assertGreaterEqual(result.confidence, 0.99)

    def test_unchanged_sidebar_does_not_promote_miss(self):
        confirmation = getattr(self.main, "apply_sidebar_completion_confirmation", None)
        self.assertIsNotNone(confirmation)
        progress = SidebarProgress(
            active_lengths=(5, 4, 3, 3, 2),
            completed_lengths=(2,),
        )
        result = dummy_hit_result("miss")

        with patch.object(
            self.main,
            "detect_sidebar_progress",
            return_value=progress,
        ):
            confirmed, after, newly_completed = confirmation(
                object(),
                object(),
                (2, 2, 3, 3, 4, 5),
                result,
            )

        self.assertFalse(confirmed)
        self.assertEqual(after, progress)
        self.assertEqual(newly_completed, ())
        self.assertEqual(result.state, "miss")

    def test_dynamic_hit_without_positive_visual_evidence_is_vetoed(self):
        evidence_gate = getattr(self.main, "enforce_positive_hit_evidence", None)
        self.assertIsNotNone(evidence_gate)
        result = dummy_hit_result("hit")
        result.score = 1.0
        result.confidence = 1.0

        vetoed = evidence_gate(
            result,
            wreck_hit=False,
            sidebar_hit=False,
        )

        self.assertTrue(vetoed)
        self.assertEqual(result.state, "miss")
        self.assertLess(result.score, self.main.SUSPECT_HIT_SCORE_THRESHOLD)
        self.assertFalse(self.main._is_near_hit_frame(result))

    def test_new_wreck_evidence_keeps_dynamic_hit(self):
        evidence_gate = getattr(self.main, "enforce_positive_hit_evidence", None)
        self.assertIsNotNone(evidence_gate)
        result = dummy_hit_result("hit")

        vetoed = evidence_gate(
            result,
            wreck_hit=True,
            sidebar_hit=False,
        )

        self.assertFalse(vetoed)
        self.assertEqual(result.state, "hit")

    def test_probe_transaction_uses_sidebar_completion_as_hit_evidence(self):
        hit_map = [[0, 0, 0] for _ in range(3)]
        progress = SidebarProgress(completed_lengths=(3,))
        probe_metadata = {}

        def confirm_sidebar(_before, _after, _fleet, result):
            result.state = "hit"
            result.score = 0.99
            result.confidence = 0.99
            return True, progress, (3,)

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "click_template", return_value=True),
            patch.object(self.main, "_wait_until_activity_detail_closed", return_value=True),
            patch.object(self.main, "enter_activity"),
            patch.object(
                self.main,
                "classify_diamond_hit",
                side_effect=lambda *_args, **_kwargs: dummy_hit_result("miss"),
            ),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                side_effect=confirm_sidebar,
            ) as sidebar_confirmation,
            patch.object(self.main, "restart_process", return_value=False),
        ):
            result = self.main._probe_cell(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
                probe_metadata=probe_metadata,
            )

        self.assertEqual(result, self.main.ProbeResult.HIT)
        self.assertEqual(hit_map[0][1], 1)
        self.assertEqual(sidebar_confirmation.call_count, len(self.main.HIT_RESULT_FRAME_DELAYS))
        self.assertEqual(self.main._runtime_status.get("sidebar_completed_cells"), 3)
        self.assertEqual(self.main._runtime_status.get("sidebar_completed_lengths"), [3])
        self.assertEqual(probe_metadata["sidebar_newly_completed_lengths"], (3,))
        self.assertEqual(probe_metadata["sidebar_completed_lengths"], (3,))

    def test_strategy_status_uses_exact_initial_visual_hit_count(self):
        signature = inspect.signature(self.main._scan_level_by_strategy)
        self.assertIn("initial_sidebar_progress", signature.parameters)
        self.assertIn("initial_visual_hit_count", signature.parameters)
        self.assertIn("initial_completed_visual_hits", signature.parameters)

        progress = SidebarProgress(completed_lengths=(4, 2))
        finished_strategy = SimpleNamespace(
            shots={
                (0, 0): True,
                (0, 1): True,
                (1, 0): True,
                (1, 1): True,
                (2, 0): True,
                (2, 1): True,
            },
            done=True,
            remaining={},
            get_confirmed_ships=lambda: [],
        )
        fake_bar = SimpleNamespace(total=19, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=finished_strategy),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress") as update_progress,
        ):
            completed = self.main._scan_level_by_strategy(
                level=7,
                hit_map=[[0] * 9 for _ in range(9)],
                click_points=[(400, 300)] * 81,
                submarines=[2, 2, 3, 3, 4, 5],
                initial_sidebar_progress=progress,
                initial_visual_hit_count=7,
            )

        self.assertTrue(completed)
        self.assertEqual(self.main._runtime_status.get("hits"), 7)
        self.assertEqual(self.main._runtime_status.get("sidebar_completed_cells"), 6)
        self.assertEqual(update_progress.call_args.args[1], 7)
        self.assertEqual(len(self.main._runtime_status.get("board_states", [])), 9)

    def test_strategy_records_initial_misses_as_real_results(self):
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=True,
            remaining={},
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
        )

        def report_result(cell, hit):
            strategy.shots[cell] = hit

        strategy.report_result = Mock(side_effect=report_result)
        fake_bar = SimpleNamespace(total=3, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots") as save_shots,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_misses={(0, 1)},
            )

        self.assertTrue(completed)
        strategy.report_result.assert_called_once_with((0, 1), False)
        self.assertEqual(strategy.shots, {(0, 1): False})
        save_shots.assert_called_with(1, 3, {(0, 1): False})
        self.assertEqual(self.main._runtime_status["board_states"][0][1], "miss")

    def test_strategy_does_not_record_old_cell_after_delayed_victory(self):
        report_result = Mock()
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((3,))),
            choose_next_cell=lambda: (0, 0),
            report_result=report_result,
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
        )
        fake_bar = SimpleNamespace(total=3, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.LEVEL_COMPLETE,
            ),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
            )

        self.assertTrue(completed)
        report_result.assert_not_called()
        self.assertEqual(self.main._runtime_status.get("phase"), "level_complete")

    def test_strategy_commits_scout_hit_online_before_offline_probe(self):
        hit_map = [[0] * 3 for _row in range(3)]
        fake_bar = SimpleNamespace(total=1, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_probe_cell") as offline_probe,
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=hit_map,
                click_points=[(400, 300)] * 9,
                submarines=[1],
                initial_scout_hits={(1, 1)},
                commit_scout_hits_online=True,
                initial_visual_hit_count=0,
            )

        self.assertTrue(completed)
        online_hit.assert_called_once()
        self.assertEqual(online_hit.call_args.kwargs["cell"], (1, 1))
        self.assertEqual(online_hit.call_args.kwargs["point"], (400, 300))
        offline_probe.assert_not_called()
        self.assertEqual(self.main._runtime_status.get("last_result"), "hit")

    def test_strategy_commits_all_scout_hits_online_before_unknown_cell(self):
        events = []
        fake_bar = SimpleNamespace(total=3, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        def online_hit(**kwargs):
            events.append(("online", kwargs["cell"]))
            return self.main.ProbeResult.HIT

        def offline_probe(_level, _hit_map, cell, _point, _index, probe_metadata=None):
            events.append(("offline", cell))
            return self.main.ProbeResult.LEVEL_COMPLETE

        with (
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "_execute_online_scout_hit", side_effect=online_hit),
            patch.object(self.main, "_probe_cell", side_effect=offline_probe),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_scout_hits={(1, 1), (1, 2)},
                commit_scout_hits_online=True,
                initial_visual_hit_count=0,
            )

        self.assertTrue(completed)
        self.assertEqual([event[0] for event in events[:2]], ["online", "online"])
        self.assertEqual(events[2][0], "offline")
        self.assertEqual({event[1] for event in events[:2]}, {(1, 1), (1, 2)})

    def test_online_scout_false_positive_does_not_increase_hit_progress(self):
        fake_bar = SimpleNamespace(total=1, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.MISS,
            ),
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.LEVEL_COMPLETE,
            ),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[1],
                initial_scout_hits={(1, 1)},
                commit_scout_hits_online=True,
                initial_visual_hit_count=0,
            )

        self.assertTrue(completed)
        miss_updates = [
            call.kwargs
            for call in write_status.call_args_list
            if call.kwargs.get("last_result") == "miss"
        ]
        self.assertEqual(len(miss_updates), 1)
        self.assertEqual(miss_updates[0]["hits"], 0)
        self.assertEqual(miss_updates[0]["board_states"][1][1], "miss")

    def test_strategy_records_final_hit_before_finishing_level(self):
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((3,))),
            choose_next_cell=lambda: (0, 0),
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
        )

        def record_result(cell, hit):
            strategy.shots[cell] = hit

        strategy.report_result = Mock(side_effect=record_result)
        fake_bar = SimpleNamespace(total=3, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE,
            ),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots") as save_shots,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_visual_hit_count=2,
            )

        self.assertTrue(completed)
        strategy.report_result.assert_called_once_with((0, 0), True)
        self.assertEqual(strategy.shots, {(0, 0): True})
        save_shots.assert_called_with(1, 3, {(0, 0): True})
        self.assertEqual(self.main._runtime_status.get("phase"), "level_complete")
        self.assertEqual(self.main._runtime_status.get("hits"), 3)
        self.assertEqual(self.main._runtime_status["board_states"][0][0], "hit")

    def test_fallback_records_final_hit_before_finishing_level(self):
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((3,))),
            choose_next_cell=lambda: None,
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
        )

        def record_result(cell, hit):
            strategy.shots[cell] = hit

        def finish_in_fallback(*_args, result_callback, **_kwargs):
            result_callback((0, 0), self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE)
            return 1

        strategy.report_result = Mock(side_effect=record_result)
        fake_bar = SimpleNamespace(total=3, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_scan_level_by_grid_order",
                side_effect=finish_in_fallback,
            ),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots") as save_shots,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_visual_hit_count=2,
            )

        self.assertTrue(completed)
        strategy.report_result.assert_called_once_with((0, 0), True)
        self.assertEqual(strategy.shots, {(0, 0): True})
        save_shots.assert_called_with(1, 3, {(0, 0): True})
        self.assertEqual(self.main._runtime_status.get("phase"), "level_complete")
        self.assertEqual(self.main._runtime_status.get("hits"), 3)
        self.assertEqual(self.main._runtime_status["board_states"][0][0], "hit")

    def test_runtime_board_snapshot_falls_back_to_shots_and_blocked_cells(self):
        strategy = SimpleNamespace(
            shots={(0, 0): True, (0, 1): False},
            blocked_cells={(1, 0)},
        )

        states = self.main.build_runtime_board_states(strategy, 3)

        self.assertEqual(states[0][0], "hit")
        self.assertEqual(states[0][1], "miss")
        self.assertEqual(states[1][0], "blocked")
        self.assertEqual(states[2][2], "unknown")

    def test_grid_scan_honors_safety_cells_added_during_fallback(self):
        skip_cells = {(0, 0)}
        points = [(400, 300)] * 9
        fake_bar = SimpleNamespace(total=8, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        def update_dynamic_skip(cell, _result, _metadata):
            if cell == (0, 1):
                skip_cells.add((0, 2))

        with (
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.MISS,
            ) as probe,
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
        ):
            scanned = self.main._scan_level_by_grid_order(
                level=1,
                hit_map=[[0] * 3 for _ in range(3)],
                click_points=points,
                skip_cells=skip_cells,
                probe_metadata_callback=update_dynamic_skip,
            )

        probed_cells = [call.args[2] for call in probe.call_args_list]
        self.assertEqual(scanned, 7)
        self.assertNotIn((0, 0), probed_cells)
        self.assertNotIn((0, 2), probed_cells)

    def test_hit_transaction_restores_network_without_reject(self):
        waits = iter(
            [
                DummyMatch((1, 1)),  # 点击前已在详情页
                DummyMatch((10, 20)),  # 第二次进入：活动按钮
                DummyMatch((30, 40)),  # 第二次进入：详情页
                DummyMatch((50, 60)),  # REJECT 后的重试按钮
                DummyMatch((70, 80)),  # 登录后下一轮：活动按钮
                DummyMatch((90, 100)),  # 登录后下一轮：详情页
            ]
        )
        hit_map = [[0, 0], [0, 0]]

        with (
            patch.object(
                self.main,
                "wait_until_occur",
                side_effect=lambda *args, **kwargs: next(waits),
            ),
            patch.object(self.main, "handle_connection_interrupted_prompt", return_value=False),
            patch.object(self.main, "click_template", return_value=True),
            patch.object(self.main, "_wait_until_activity_detail_closed", return_value=True),
            patch.object(self.main, "classify_diamond_hit", return_value=dummy_hit_result("hit")),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=True),
        ):
            result = self.main._probe_cell(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        package_name = self.main.GAME_PACKAGE_NAME
        network_calls = [
            call
            for call in self.adb.calls
            if call[0]
            in {
                "enable_reject_network",
                "disable_reject_network",
                "disable_weak_network",
                "enable_weak_network",
            }
        ]
        self.assertEqual(result, self.main.ProbeResult.HIT)
        self.assertEqual(hit_map[0][1], 1)
        self.assertIsNone(self.main._active_probe)
        self.assertEqual(
            network_calls,
            [
                ("enable_weak_network", package_name),
                ("disable_weak_network", package_name),
                ("enable_weak_network", package_name),
            ],
        )

    def test_probe_enables_drop_before_clicking_target_cell(self):
        hit_map = [[0, 0], [0, 0]]

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "click_template", return_value=True),
            patch.object(self.main, "_wait_until_activity_detail_closed", return_value=True),
            patch.object(self.main, "enter_activity"),
            patch.object(
                self.main,
                "classify_diamond_hit",
                return_value=dummy_hit_result("miss"),
            ),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
            patch.object(self.main, "restart_process"),
        ):
            result = self.main._probe_cell(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        package_name = self.main.GAME_PACKAGE_NAME
        drop_call = ("enable_weak_network", package_name)
        target_click = ("click", 400, 300)
        self.assertEqual(result, self.main.ProbeResult.MISS)
        self.assertIn(drop_call, self.adb.calls)
        self.assertLess(self.adb.calls.index(drop_call), self.adb.calls.index(target_click))

    def test_blue_probe_refuses_to_click_when_network_isolation_is_unsafe(self):
        hit_map = [[0, 0], [0, 0]]
        self.adb.verify_app_network_isolated = Mock(
            return_value=SimpleNamespace(safe=False, detail="ipv6 unblocked")
        )

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "sleep"),
            patch.object(self.main, "write_pending_probe") as write_pending,
            self.assertRaisesRegex(self.main.ProbeProtocolError, "ipv6 unblocked"),
        ):
            self.main._execute_probe_transaction(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertNotIn(("click", 400, 300), self.adb.calls)
        write_pending.assert_not_called()
        self.assertEqual(self.main._network_fail_closed_reason, "ipv6 unblocked")

    def test_blue_probe_fails_closed_when_network_verification_errors(self):
        hit_map = [[0, 0], [0, 0]]
        self.adb.verify_app_network_isolated = Mock(
            side_effect=RuntimeError("adb unavailable")
        )

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "sleep"),
            patch.object(self.main, "write_pending_probe") as write_pending,
            self.assertRaisesRegex(self.main.ProbeProtocolError, "adb unavailable"),
        ):
            self.main._execute_probe_transaction(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertNotIn(("click", 400, 300), self.adb.calls)
        write_pending.assert_not_called()
        self.assertIn("verification failed", self.main._network_fail_closed_reason)

    def test_blue_probe_persists_pending_marker_before_click_and_clears_after_discard(self):
        events = []
        hit_map = [[0, 0], [0, 0]]

        def click(x, y):
            events.append(("click", x, y))

        def discard(transaction):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        self.adb.click = Mock(side_effect=click)
        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=False),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "classify_diamond_hit", return_value=dummy_hit_result("miss")),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
            patch.object(self.main, "get_configured_submarines", return_value=[]),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_save_probe_result_json"),
            patch.object(self.main, "append_recent_probe_result"),
            patch.object(
                self.main,
                "write_pending_probe",
                side_effect=lambda **kwargs: events.append(("marker", kwargs)),
            ),
            patch.object(self.main, "update_pending_probe", return_value=True),
            patch.object(
                self.main,
                "clear_pending_probe",
                side_effect=lambda: events.append(("clear",)),
            ),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard,
            ),
        ):
            result = self.main._execute_probe_transaction(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertEqual(result, self.main.ProbeResult.MISS)
        marker_index = next(i for i, event in enumerate(events) if event[0] == "marker")
        click_index = next(i for i, event in enumerate(events) if event[0] == "click")
        clear_index = next(i for i, event in enumerate(events) if event[0] == "clear")
        self.assertLess(marker_index, click_index)
        self.assertLess(click_index, clear_index)
        self.assertEqual(events[marker_index][1]["mode"], "blue_probe")

    def test_miss_discard_force_stops_before_network_restore(self):
        transaction = self.main.ProbeTransaction(level=1, cell=(0, 1), index=1)
        transaction.advance(self.main.ProbePhase.REQUEST_PENDING)
        transaction.advance(self.main.ProbePhase.RESULT_VISIBLE)
        transaction.advance(self.main.ProbePhase.RESULT_RECORDED)

        with (
            patch.object(self.main, "handle_connection_interrupted_prompt") as dialog,
            patch.object(self.main, "wait_until_retry_prompt") as retry,
            patch.object(self.main, "restart_process") as restart,
        ):
            self.main._discard_pending_request_and_prepare_next_probe(transaction)

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertEqual(transaction.phase, self.main.ProbePhase.COMPLETE)
        self.assertEqual(
            self.adb.calls[:5],
            [
                ("enable_reject_network", package_name),
                ("delay", 0.2),
                ("close_app", package_name),
                (
                    "wait_until_app_stopped",
                    package_name,
                    self.main.APP_STOP_TIMEOUT_SECONDS,
                    self.main.APP_STOP_POLL_SECONDS,
                ),
                ("delay", self.main.POST_FORCE_STOP_GUARD_SECONDS),
            ],
        )
        self.assertNotIn(("click", 123, 456), self.adb.calls)
        dialog.assert_not_called()
        retry.assert_not_called()
        restart.assert_called_once_with(reopen_game=True, app_already_closed=True)
        self.assertLess(
            self.adb.calls.index(("wait_until_app_stopped", package_name, self.main.APP_STOP_TIMEOUT_SECONDS, self.main.APP_STOP_POLL_SECONDS)),
            self.adb.calls.index(("delay", self.main.POST_FORCE_STOP_GUARD_SECONDS)),
        )

    def test_miss_transaction_closes_app_instead_of_clicking_retry(self):
        waits = iter(
            [
                DummyMatch((1, 1)),  # 点击前已在详情页
                DummyMatch((10, 20)),  # 第二次进入：活动按钮
                DummyMatch((30, 40)),  # 第二次进入：详情页
            ]
        )
        hit_map = [[0, 0], [0, 0]]

        with (
            patch.object(
                self.main,
                "wait_until_occur",
                side_effect=lambda *args, **kwargs: next(waits),
            ),
            patch.object(self.main, "handle_connection_interrupted_prompt") as dialog,
            patch.object(self.main, "wait_until_retry_prompt") as retry,
            patch.object(self.main, "click_template", return_value=True),
            patch.object(self.main, "_wait_until_activity_detail_closed", return_value=True),
            patch.object(self.main, "classify_diamond_hit", return_value=dummy_hit_result("miss")),
            patch.object(self.main, "restart_process") as restart,
        ):
            result = self.main._probe_cell(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertEqual(result, self.main.ProbeResult.MISS)
        self.assertIsNone(self.main._active_probe)
        self.assertIn(("enable_reject_network", package_name), self.adb.calls)
        self.assertIn(("close_app", package_name), self.adb.calls)
        dialog.assert_not_called()
        retry.assert_not_called()
        restart.assert_called_once_with(reopen_game=True, app_already_closed=True)

    def test_preflight_failure_retries_the_same_cell(self):
        hit_map = [[0, 0], [0, 0]]

        with (
            patch.object(
                self.main,
                "_execute_probe_transaction",
                side_effect=[
                    self.main.ProbeNotReadyError("页面未准备好"),
                    self.main.ProbeResult.MISS,
                ],
            ) as execute,
            patch.object(self.main, "enter_activity") as recover,
        ):
            result = self.main._probe_cell(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertEqual(result, self.main.ProbeResult.MISS)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(execute.call_args_list[0], execute.call_args_list[1])
        recover.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
