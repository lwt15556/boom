import importlib
import inspect
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from utils.sidebar_progress import SidebarProgress


class FakeScreenshotCapture:
    def __init__(self, image):
        self.image = image
        self.png_bytes = b"fake-png"

    @staticmethod
    def save(path):
        return path


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

    def capture_screenshot(self):
        self.calls.append(("capture_screenshot",))
        return FakeScreenshotCapture(self.read_screenshot())

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

    def test_red_scout_sample_directories_are_unique_per_attempt(self):
        sample_root = self.main.Path(self.runtime_temp.name) / "red_scout_samples"
        with patch.object(
            self.main,
            "RED_SCOUT_SAMPLE_DIR",
            sample_root,
            create=True,
        ):
            first = self.main._create_red_scout_sample_dir(
                level=15,
                center=(4, 5),
                index=45,
                attempt=1,
            )
            second = self.main._create_red_scout_sample_dir(
                level=15,
                center=(4, 5),
                index=45,
                attempt=2,
            )

        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())
        self.assertIn("attempt_01", first.name)
        self.assertIn("attempt_02", second.name)

    def test_red_analysis_uses_median_baseline_once_for_deterministic_result(self):
        deterministic = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset(),
            hit_cells=frozenset(),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=None,
            valid=False,
            confidence_by_cell={},
            invalid_reason="too_many_strong_cells",
        )
        baselines = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (0, 30, 10)
        ]

        with patch.object(
            self.main,
            "_analyze_red_result",
            return_value=deterministic,
        ) as analyze:
            result = self.main._analyze_red_result_with_baseline_consensus(
                before_images=baselines,
                after_images=["after"],
                click_points=[(0, 0)] * 9,
                grid_size=3,
                center_cell=(1, 1),
                submarine_lengths=[3],
            )

        self.assertIs(result, deterministic)
        self.assertEqual(analyze.call_count, 1)
        np.testing.assert_array_equal(
            analyze.call_args.args[0],
            np.full((2, 2, 3), 10, dtype=np.uint8),
        )

    def test_red_analysis_uses_only_one_original_fallback_when_uncertain(self):
        primary = self.main.RedScoutResult(
            center_cell=(1, 1), affected_cells=frozenset(),
            hit_cells=frozenset(), miss_cells=frozenset(),
            unknown_cells=frozenset(), footprint=None, valid=False,
            confidence_by_cell={}, invalid_reason="insufficient_changed_cells",
        )

        def full_result(last_cell):
            cells = frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), last_cell})
            return self.main.RedScoutResult(
                center_cell=(1, 1), affected_cells=cells,
                hit_cells=frozenset({(1, 1)}), miss_cells=cells - {(1, 1)},
                unknown_cells=frozenset(),
                footprint=self.main.RedFootprint(frozenset({(0, 0)})),
                valid=True, confidence_by_cell={cell: 0.9 for cell in cells},
            )

        recovered = full_result((1, 2))
        baselines = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (0, 20, 100)
        ]
        with patch.object(
            self.main,
            "_analyze_red_result",
            side_effect=[primary, recovered],
        ) as analyze:
            result = self.main._analyze_red_result_with_baseline_consensus(
                before_images=baselines,
                after_images=["after"], click_points=[(0, 0)] * 9,
                grid_size=3, center_cell=(1, 1), submarine_lengths=[3],
            )

        self.assertIs(result, recovered)
        self.assertEqual(analyze.call_count, 2)
        np.testing.assert_array_equal(
            analyze.call_args_list[1].args[0],
            baselines[0],
        )

    def test_red_analysis_keeps_median_result_when_single_fallback_errors(self):
        primary = self.main.RedScoutResult(
            center_cell=(1, 1), affected_cells=frozenset(),
            hit_cells=frozenset(), miss_cells=frozenset(),
            unknown_cells=frozenset(), footprint=None, valid=False,
            confidence_by_cell={}, invalid_reason="insufficient_changed_cells",
        )
        alternative = self._valid_red_result()
        baselines = [
            np.full((2, 2, 3), value, dtype=np.uint8)
            for value in (0, 20, 100)
        ]

        with patch.object(
            self.main,
            "_analyze_red_result",
            side_effect=[primary, RuntimeError("secondary failed"), alternative],
        ) as analyze:
            result = self.main._analyze_red_result_with_baseline_consensus(
                before_images=baselines,
                after_images=["after"], click_points=[(0, 0)] * 9,
                grid_size=3, center_cell=(1, 1), submarine_lengths=[3],
            )

        self.assertIs(result, primary)
        self.assertEqual(analyze.call_count, 2)

    def test_red_scout_sample_retention_removes_only_oldest_managed_directory(self):
        sample_root = self.main.Path(self.runtime_temp.name) / "red_scout_samples"
        sample_root.mkdir()
        directories = []
        for index in range(3):
            path = sample_root / f"level_15_attempt_0{index + 1}_sample"
            path.mkdir()
            (path / "analysis.json").write_text("{}", encoding="utf-8")
            os.utime(path, (index + 1, index + 1))
            directories.append(path)
        unmanaged = sample_root / "keep_me"
        unmanaged.mkdir()

        with patch.object(
            self.main,
            "RED_SCOUT_SAMPLE_DIR",
            sample_root,
            create=True,
        ):
            self.main._prune_red_scout_sample_dirs(max_directories=2)

        self.assertFalse(directories[0].exists())
        self.assertTrue(directories[1].is_dir())
        self.assertTrue(directories[2].is_dir())
        self.assertTrue(unmanaged.is_dir())

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
        self.assertTrue(
            all(call.kwargs["excluded_cells"] == () for call in execute.call_args_list)
        )
        self.assertEqual(
            [call.kwargs["attempt"] for call in execute.call_args_list],
            [1, 2, 3],
        )
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
        self.assertEqual(write_status.call_args.kwargs["red_scout_valid"], 3)
        self.assertEqual(write_status.call_args.kwargs["red_scout_complete_six"], 0)
        phases = [call.kwargs["phase"] for call in write_status.call_args_list if "phase" in call.kwargs]
        self.assertEqual(phases[-1], "blue_attack")
        self.assertEqual(phases.count("red_scout_capture"), 3)

    def test_red_scout_keeps_surrounding_misses_for_final_blue_priority_scan(self):
        neighbors = frozenset({(0, 1), (2, 1), (1, 0), (1, 2)})
        result = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset({(1, 1)}) | neighbors,
            hit_cells=frozenset({(1, 1)}),
            miss_cells=neighbors,
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(
                frozenset({(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)})
            ),
            valid=True,
            confidence_by_cell={cell: 0.9 for cell in {(1, 1)} | set(neighbors)},
        )
        events = []

        def online_hit(**_kwargs):
            events.append("online_hit")
            return self.main.ProbeResult.HIT

        def final_scan(*_args, **_kwargs):
            events.append("final_scan")
            return True

        with (
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(self.main, "_execute_online_scout_hit", side_effect=online_hit),
            patch.object(self.main, "_scan_level_by_strategy", side_effect=final_scan) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_hits=set(),
                settings=self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1),
            )

        self.assertTrue(completed)
        self.assertEqual(events, ["online_hit", "final_scan"])
        self.assertEqual(scan.call_args.kwargs["initial_hits"], {(1, 1)})
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], set())
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], set(neighbors))
        self.assertTrue(scan.call_args.kwargs["commit_scout_hits_online"])

    def test_red_scout_counts_only_fully_classified_six_cell_results_as_complete(self):
        cells = frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)})
        result = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=cells,
            hit_cells=frozenset(),
            miss_cells=cells,
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset({(0, 0)})),
            valid=True,
            confidence_by_cell={cell: 0.9 for cell in cells},
        )

        with (
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                level=1,
                hit_map=[[0] * 3 for _ in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_hits=set(),
                settings=self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1),
            )

        self.assertTrue(completed)
        self.assertEqual(write_status.call_args.kwargs["red_scout_valid"], 1)
        self.assertEqual(write_status.call_args.kwargs["red_scout_complete_six"], 1)

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

        def red_attempt(*_args, **_kwargs):
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
            patch.object(self.main, "write_runtime_status") as write_status,
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
        miss_updates = [
            call.kwargs
            for call in write_status.call_args_list
            if call.kwargs.get("last_result") == "miss"
        ]
        self.assertEqual(len(miss_updates), 1)
        self.assertEqual(miss_updates[0]["phase"], "blue_online_scout_hits")
        self.assertEqual(miss_updates[0]["board_states"][1][2], "miss")
        self.assertEqual(miss_updates[0]["board_states"][1][1], "scout_miss")

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
                "before",
                ["after"],
                click_points,
                3,
                (1, 1),
                submarine_lengths=[3],
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
            submarine_lengths=[3],
        )

    def test_red_transaction_capture_does_not_precede_preflight(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        result = self._valid_red_result()
        phases = []

        def execute(*args, **_kwargs):
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

        def execute(_level, center, *_args, **_kwargs):
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
        incomplete_progress = SidebarProgress(active_lengths=(3,))
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
            ) as classify,
            patch.object(self.main, "find_victory_banner", return_value=None),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=True),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                return_value=(False, incomplete_progress, ()),
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
        self.assertEqual(classify.call_count, 3)
        self.assertEqual(hit_map[1][1], 1)
        self.assertEqual(self.adb.calls.count(("click", 640, 360)), 1)
        self.assertEqual(self.adb.calls.count(("click", *self.main.BLUE_BOMB_POINT)), 1)
        self.assertIn(("disable_reject_network", package_name), self.adb.calls)
        self.assertIn(("disable_weak_network", package_name), self.adb.calls)
        self.assertNotIn(("enable_reject_network", package_name), self.adb.calls)
        self.assertNotIn(("enable_weak_network", package_name), self.adb.calls)

    def test_online_scout_false_positive_clears_stale_hit_map_cell(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        hit_map[1][1] = 1
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
                side_effect=lambda *_args, **_kwargs: dummy_hit_result("miss"),
            ),
            patch.object(self.main, "find_victory_banner", return_value=None),
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

        self.assertEqual(result, self.main.ProbeResult.MISS)
        self.assertEqual(hit_map[1][1], 0)

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
        received_submarine_lengths = []

        def execute(level, center, point, index, grid_size, all_click_points, **_kwargs):
            centers.append((level, center))
            received_submarine_lengths.append(_kwargs.get("submarine_lengths"))
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
        self.assertEqual(received_submarine_lengths, [[3], [3], [3], [3]])

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

    def test_handle_game_level_keeps_visible_wreck_cells_from_midgame_frames(self):
        submarines = [2, 2, 3, 3, 4, 5]
        visible_hits = {
            (0, 6),
            (0, 7),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (7, 1),
            (7, 2),
            (7, 3),
        }
        partial_wreck_cells = {(2, 0), (3, 0), (4, 0)}
        sidebar_progress = SidebarProgress(
            active_lengths=(5, 4, 3, 2, 2),
            completed_lengths=(3,),
            unknown_lengths=(),
        )
        click_points = [(400 + index, 300 + index) for index in range(81)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        def run_strategy(*_args, **kwargs):
            self.assertEqual(kwargs["initial_hits"], visible_hits)
            self.assertEqual(kwargs["initial_completed_visual_hits"], {(7, 1), (7, 2), (7, 3)})
            self.assertEqual(kwargs["initial_visual_hit_count"], 9)
            return True

        with (
            patch.object(self.main.adb, "delay"),
            patch.object(self.main.adb, "read_screenshot", return_value=grid_img),
            patch.object(
                self.main,
                "get_click_points",
                return_value=(click_points, np.zeros((4, 2), dtype=np.float32)),
            ),
            patch.object(self.main, "get_configured_submarines", return_value=submarines),
            patch.object(self.main, "detect_sidebar_progress", return_value=sidebar_progress),
            patch.object(self.main, "detect_visible_wreck_cells", return_value=visible_hits),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=partial_wreck_cells),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", side_effect=run_strategy) as run,
        ):
            grid_img_result, _, completed = self.main.handle_game_level(
                7,
                [[0] * 9 for _ in range(9)],
            )

        self.assertTrue(completed)
        self.assertIs(grid_img_result, grid_img)
        run.assert_called_once()

    def test_handle_game_level_discards_suspicious_all_grid_wreck_candidates(self):
        submarines = [2, 2, 3, 3, 4, 5]
        visible_hits = {
            (row, col)
            for row in range(9)
            for col in range(9)
        }
        sidebar_progress = SidebarProgress(
            active_lengths=(2, 2),
            completed_lengths=(5, 4, 3, 3),
            unknown_lengths=(),
        )
        click_points = [(400 + index, 300 + index) for index in range(81)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        def run_strategy(*_args, **kwargs):
            self.assertEqual(kwargs["initial_hits"], set())
            self.assertEqual(kwargs["initial_completed_visual_hits"], set())
            self.assertEqual(kwargs["initial_visual_hit_count"], 15)
            return True

        with (
            patch.object(self.main.adb, "delay"),
            patch.object(self.main.adb, "read_screenshot", return_value=grid_img),
            patch.object(
                self.main,
                "get_click_points",
                return_value=(click_points, np.zeros((4, 2), dtype=np.float32)),
            ),
            patch.object(self.main, "get_configured_submarines", return_value=submarines),
            patch.object(self.main, "detect_sidebar_progress", return_value=sidebar_progress),
            patch.object(self.main, "detect_visible_wreck_cells", return_value=visible_hits),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=set()),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", side_effect=run_strategy) as run,
        ):
            _grid_img_result, _quad, completed = self.main.handle_game_level(
                7,
                [[0] * 9 for _ in range(9)],
            )

        self.assertTrue(completed)
        run.assert_called_once()

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

    def test_red_discard_recovery_timeout_keeps_game_process_running(self):
        analysis = self._valid_red_result()

        def discard_timeout(transaction):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.red_request_discarded = True
            self.main.latch_network_fail_closed("retry dialog timeout")
            raise self.main.DiscardRecoveryError("retry dialog timeout")

        with (
            patch.object(
                self.main,
                "_capture_red_ammo_state",
                return_value=("before", "fp", DummyMatch((10, 20))),
            ),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=False),
            patch.object(self.main, "_capture_red_result_frames", return_value=["after"]),
            patch.object(
                self.main,
                "_analyze_red_result_with_baseline_consensus",
                return_value=analysis,
            ),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard_timeout,
            ),
            patch.object(self.main, "write_pending_probe"),
            patch.object(self.main, "update_pending_probe"),
        ):
            with self.assertRaisesRegex(
                self.main.RedScoutSafetyError,
                "discard recovery stalled",
            ):
                self.main._execute_red_scout_transaction(
                    1,
                    (1, 1),
                    (100, 200),
                    0,
                    3,
                    [(0, 0)] * 9,
                )

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertEqual(self.main._network_fail_closed_reason, "retry dialog timeout")
        self.assertNotIn(("close_app", package_name), self.adb.calls)
        self.assertNotIn(("open_app", package_name), self.adb.calls)
        self.assertNotIn(
            "wait_until_app_stopped",
            [call[0] for call in self.adb.calls],
        )
        self.assertIsNotNone(self.main._active_probe)
        self.assertEqual(
            self.main._active_probe.phase,
            self.main.ProbePhase.REQUEST_DISCARDED,
        )

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
        analysis_started = Event()
        recovery_started = Event()

        def exit_red_activity(*_args, **_kwargs):
            events.append("system_back_exit")
            return True

        def reenter_activity(*_args, **_kwargs):
            events.append("reenter_activity")
            return False

        def capture_result_frames(*, sample_dir=None):
            self.assertIsNone(sample_dir)
            events.append("capture_result")
            return ["after"]

        def analyze_result(**_kwargs):
            events.append("analysis_started")
            analysis_started.set()
            if not recovery_started.wait(timeout=0.5):
                events.append("analysis_waited_for_recovery")
            events.append("analysis_finished")
            return analysis

        def discard(tx):
            self.assertTrue(analysis_started.wait(timeout=0.5))
            tx.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            tx.red_request_discarded = True
            events.append("discard_started")
            recovery_started.set()
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
            patch.object(
                self.main,
                "_analyze_red_result_with_baseline_consensus",
                side_effect=analyze_result,
            ) as analyze,
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
                submarine_lengths=[3],
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
            events[:5],
            [
                "system_back_exit",
                "reenter_activity",
                "capture_result",
                "analysis_started",
                "discard_started",
            ],
        )
        self.assertNotIn("analysis_waited_for_recovery", events)
        phases = [call.kwargs["phase"] for call in write_status.call_args_list if "phase" in call.kwargs]
        self.assertEqual(
            phases,
            ["red_scout_preflight", "red_scout_capture", "red_scout_discard", "red_scout_verify_ammo"],
        )
        self.assertEqual(analyze.call_args.kwargs["submarine_lengths"], [3])

    def test_red_analysis_failure_after_discard_does_not_force_stop_game(self):
        events = []

        def discard(transaction):
            events.append("discard")
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.red_request_discarded = True
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(
                self.main,
                "_capture_red_ammo_state",
                return_value=("before", "fingerprint", DummyMatch((10, 20))),
            ),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=False),
            patch.object(self.main, "_capture_red_result_frames", return_value=["after"]),
            patch.object(
                self.main,
                "_analyze_red_result_with_baseline_consensus",
                side_effect=RuntimeError("analysis failed"),
            ),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard,
            ),
            patch.object(
                self.main,
                "_verify_red_ammo_unchanged",
                side_effect=lambda *_args, **_kwargs: events.append("verify_ammo"),
            ),
            patch.object(self.main, "_stop_and_latch_red_safety_failure") as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "analysis failed"):
                self.main._execute_red_scout_transaction(
                    level=1,
                    center_cell=(1, 1),
                    point=(100, 200),
                    index=0,
                    grid_size=3,
                    all_click_points=[(0, 0)] * 9,
                    submarine_lengths=[3],
                )

        self.assertEqual(events, ["discard", "verify_ammo"])
        stop.assert_not_called()
        self.assertIsNone(self.main._active_probe)

    def test_red_scout_transaction_wires_all_artifacts_to_attempt_directory(self):
        analysis = self._valid_red_result()
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()
        match = DummyMatch((10, 20))

        def discard(transaction):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.red_request_discarded = True
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(
                self.main,
                "_create_red_scout_sample_dir",
                return_value=sample_dir,
            ) as create_sample,
            patch.object(
                self.main,
                "_capture_red_ammo_state",
                return_value=(["before-0", "before-1", "before-2"], "fingerprint", match),
            ) as capture_ammo,
            patch.object(self.main, "_select_red_bomb", return_value=True) as select_red,
            patch.object(self.main, "_exit_activity_after_probe_click") as exit_activity,
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=False),
            patch.object(
                self.main,
                "_capture_red_result_frames",
                return_value=["after"],
            ) as capture_results,
            patch.object(
                self.main,
                "_analyze_red_result_with_baseline_consensus",
                return_value=analysis,
            ) as analyze,
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard,
            ),
            patch.object(self.main, "_verify_red_ammo_unchanged") as verify_ammo,
            patch.object(self.main, "_write_red_scout_analysis") as write_analysis,
        ):
            result = self.main._execute_red_scout_transaction(
                level=15,
                center_cell=(1, 1),
                point=(100, 200),
                index=11,
                grid_size=3,
                all_click_points=[(0, 0)] * 9,
                submarine_lengths=[3],
                attempt=2,
            )

        self.assertIs(result, analysis)
        create_sample.assert_called_once_with(15, (1, 1), 11, 2)
        capture_ammo.assert_called_once_with(
            sample_dir=sample_dir,
            prefix="before",
            include_frames=True,
        )
        select_red.assert_called_once_with(match, output_path=sample_dir / "selected.png")
        exit_activity.assert_called_once_with(
            sample_dir / "exit_attempt.png",
            use_system_back=True,
        )
        capture_results.assert_called_once_with(sample_dir=sample_dir)
        self.assertEqual(
            analyze.call_args.kwargs["before_images"],
            ["before-0", "before-1", "before-2"],
        )
        write_analysis.assert_called_once_with(
            sample_dir,
            analysis,
            level=15,
            index=11,
            attempt=2,
        )
        verify_ammo.assert_called_once_with("fingerprint", sample_dir=sample_dir)

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
            patch.object(
                self.main,
                "_analyze_red_result_with_baseline_consensus",
                return_value=analysis,
            ),
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
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()

        def discard(transaction, **_kwargs):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.red_request_discarded = True
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(
                self.main,
                "_create_red_scout_sample_dir",
                return_value=sample_dir,
            ),
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
            patch.object(self.main, "_write_red_scout_analysis") as write_analysis,
        ):
            result = self.main._execute_red_scout_transaction(
                1, (1, 1), (100, 200), 0, 3, [(0, 0)] * 9, attempt=1
            )

        self.assertFalse(result.level_completed)
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_reason, "local_victory_screen")
        clear_pending.assert_called_once()
        analyze.assert_not_called()
        write_analysis.assert_called_once_with(
            sample_dir,
            result,
            level=1,
            index=0,
            attempt=1,
        )

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

    def test_red_result_capture_writes_each_attempt_to_its_sample_directory(self):
        frames = [object() for _ in self.main.HIT_RESULT_FRAME_DELAYS]
        captured_paths = []
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()

        def read_screenshot(path):
            captured_paths.append(path)
            return frames[len(captured_paths) - 1]

        self.adb.read_screenshot = Mock(side_effect=read_screenshot)

        result = self.main._capture_red_result_frames(sample_dir=sample_dir)

        self.assertEqual(result, frames)
        self.assertEqual(
            captured_paths,
            [
                sample_dir / f"after_{index}.png"
                for index in range(len(self.main.HIT_RESULT_FRAME_DELAYS))
            ],
        )

    def test_red_ammo_capture_keeps_before_and_verify_frames_separate(self):
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()
        match = DummyMatch((10, 20))
        self.adb.read_screenshot = Mock(side_effect=["b0", "b1", "b2", "v0", "v1", "v2"])

        with (
            patch.object(self.main, "locate_red_bomb_button", return_value=match),
            patch.object(self.main, "build_ammo_fingerprint", return_value="fingerprint"),
        ):
            before = self.main._capture_red_ammo_state(
                sample_dir=sample_dir,
                prefix="before",
            )
            verify = self.main._capture_red_ammo_state(
                sample_dir=sample_dir,
                prefix="verify",
            )

        self.assertEqual(before, ("b0", "fingerprint", match))
        self.assertEqual(verify, ("v0", "fingerprint", match))
        self.assertEqual(
            [call.args[0] for call in self.adb.read_screenshot.call_args_list],
            [
                sample_dir / "before_0.png",
                sample_dir / "before_1.png",
                sample_dir / "before_2.png",
                sample_dir / "verify_0.png",
                sample_dir / "verify_1.png",
                sample_dir / "verify_2.png",
            ],
        )

    def test_red_ammo_capture_can_return_all_baseline_frames_for_consensus(self):
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()
        match = DummyMatch((10, 20))
        frames = ["b0", "b1", "b2"]
        self.adb.read_screenshot = Mock(side_effect=frames)

        with (
            patch.object(self.main, "locate_red_bomb_button", return_value=match),
            patch.object(self.main, "build_ammo_fingerprint", return_value="fingerprint"),
        ):
            captured, fingerprint, captured_match = self.main._capture_red_ammo_state(
                sample_dir=sample_dir,
                prefix="before",
                include_frames=True,
            )

        self.assertEqual(captured, frames)
        self.assertEqual(fingerprint, "fingerprint")
        self.assertIs(captured_match, match)

    def test_red_selection_screenshot_uses_attempt_sample_path(self):
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()
        match = DummyMatch((10, 20))
        selected_image = object()
        self.adb.read_screenshot = Mock(return_value=selected_image)

        with patch.object(self.main, "red_bomb_selected", return_value=True) as selected:
            confirmed = self.main._select_red_bomb(
                match,
                output_path=sample_dir / "selected.png",
            )

        self.assertTrue(confirmed)
        self.adb.read_screenshot.assert_called_once_with(sample_dir / "selected.png")
        selected.assert_called_once_with(selected_image, match)

    def test_red_analysis_json_records_result_and_intermediate_diagnostics(self):
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()
        result = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset({(0, 0), (0, 1)}),
            hit_cells=frozenset({(0, 0)}),
            miss_cells=frozenset({(0, 1)}),
            unknown_cells=frozenset(),
            footprint=None,
            valid=False,
            confidence_by_cell={(0, 0): 0.95, (0, 1): 0.80},
            invalid_reason="insufficient_changed_cells",
            diagnostics={
                "stage": "insufficient_changes",
                "raw_stable_hits": ((0, 0),),
                "completed_sidebar_votes": (
                    {"lengths": (3,), "votes": 2},
                ),
            },
        )

        self.main._write_red_scout_analysis(
            sample_dir,
            result,
            level=15,
            index=11,
            attempt=2,
        )

        payload = self.main.json.loads(
            (sample_dir / "analysis.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["level"], 15)
        self.assertEqual(payload["attempt"], 2)
        self.assertEqual(payload["center"], [1, 1])
        self.assertFalse(payload["valid"])
        self.assertFalse(payload["complete_six"])
        self.assertEqual(payload["invalid_reason"], "insufficient_changed_cells")
        self.assertEqual(payload["diagnostics"]["raw_stable_hits"], [[0, 0]])
        self.assertEqual(
            payload["diagnostics"]["completed_sidebar_votes"],
            [{"lengths": [3], "votes": 2}],
        )

    def test_red_analysis_json_does_not_mark_invalid_six_cell_result_complete(self):
        sample_dir = self.main.Path(self.runtime_temp.name) / "attempt"
        sample_dir.mkdir()
        cells = frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)})
        result = self.main.RedScoutResult(
            center_cell=(1, 1), affected_cells=cells,
            hit_cells=frozenset(), miss_cells=cells,
            unknown_cells=frozenset(), footprint=None, valid=False,
            confidence_by_cell={cell: 0.9 for cell in cells},
            invalid_reason="ambiguous_result",
        )

        self.main._write_red_scout_analysis(
            sample_dir, result, level=1, index=4, attempt=1,
        )

        payload = self.main.json.loads(
            (sample_dir / "analysis.json").read_text(encoding="utf-8")
        )
        self.assertFalse(payload["complete_six"])

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

    def test_exit_wait_ignores_quit_template_match_outside_top_left(self):
        template = cv2.imread(str(self.main.QUIT_ACTIVITY_TEMPLATE))
        self.assertIsNotNone(template)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        template_height, template_width = template.shape[:2]
        frame[300:300 + template_height, 500:500 + template_width] = template
        self.adb.read_screenshot = Mock(return_value=frame)

        with (
            patch.object(
                self.main,
                "monotonic",
                side_effect=[0.0, 0.0, 0.1, 1.1],
            ),
            patch.object(self.main, "sleep"),
        ):
            closed = self.main._wait_until_activity_detail_closed(timeout=1.0)

        self.assertTrue(closed)
        self.assertEqual(self.adb.read_screenshot.call_count, 2)

    def test_exit_wait_takes_final_confirmation_when_one_absent_frame_hits_deadline(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=frame)

        with (
            patch.object(
                self.main,
                "monotonic",
                side_effect=[0.0, 0.0, 1.1],
            ),
            patch.object(self.main, "sleep"),
        ):
            closed = self.main._wait_until_activity_detail_closed(timeout=1.0)

        self.assertTrue(closed)
        self.assertEqual(self.adb.read_screenshot.call_count, 2)

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

        def commit(transaction, *, victory_wait_timeout):
            self.assertEqual(
                victory_wait_timeout,
                self.main.VICTORY_WAIT_AFTER_HIT_SECONDS,
            )
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

    def test_victory_detection_uses_center_roi_and_restores_screen_coordinates(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        local_match = self.main.MatchResult(
            template_path=self.main.VICTORY_BANNER_TEMPLATE,
            top_left=(10, 20),
            bottom_right=(110, 80),
            center=(60, 50),
            score=0.95,
        )

        with (
            patch.object(self.main, "find_template", return_value=None),
            patch.object(
                self.main,
                "find_template_multi_scale",
                return_value=local_match,
            ) as multi_scale,
        ):
            match = self.main.find_victory_banner(screenshot)

        roi = multi_scale.call_args.args[0]
        left, top, right, bottom = self.main.VICTORY_SEARCH_REGION
        offset_x = int(round(screenshot.shape[1] * left))
        offset_y = int(round(screenshot.shape[0] * top))
        self.assertEqual(
            roi.shape[:2],
            (
                int(round(screenshot.shape[0] * bottom)) - offset_y,
                int(round(screenshot.shape[1] * right)) - offset_x,
            ),
        )
        self.assertEqual(match.top_left, (offset_x + 10, offset_y + 20))
        self.assertEqual(match.bottom_right, (offset_x + 110, offset_y + 80))
        self.assertEqual(match.center, (offset_x + 60, offset_y + 50))
        self.assertEqual(match.score, local_match.score)

    def test_victory_wait_runs_full_screen_fallback_after_roi_misses(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        fallback_match = DummyMatch((640, 280))
        searches = []
        self.adb.read_screenshot = Mock(return_value=screenshot)

        def find_banner(_screenshot, *, full_screen=False):
            searches.append(full_screen)
            return fallback_match if full_screen else None

        with (
            patch.object(self.main, "find_victory_banner", side_effect=find_banner),
            patch.object(self.main, "monotonic", side_effect=[0.0, 0.1, 1.1]),
            patch.object(self.main, "sleep"),
        ):
            match = self.main.wait_until_victory_banner(timeout=1.0)

        self.assertIs(match, fallback_match)
        self.assertEqual(searches, [False, True])
        self.adb.read_screenshot.assert_called_once_with()

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

    def test_re_enter_after_client_reload_prepares_activity_list_without_reblocking_network(self):
        self.assertIn(
            "prepare_activity_list",
            inspect.signature(self.main.enter_activity).parameters,
        )
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
            self.main.enter_activity(
                re_enter=True,
                max_retries=1,
                prepare_activity_list=True,
            )

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertNotIn(("enable_weak_network", package_name), self.adb.calls)
        self.assertEqual(
            self.adb.calls.count(("swipe", 1000, 660, 1000, 180)),
            2,
        )
        self.assertNotIn(("close_app", package_name), self.adb.calls)
        self.assertNotIn(("open_app", package_name), self.adb.calls)

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

    def test_clear_probe_debug_images_save_only_before_and_best_frame(self):
        persist = getattr(self.main, "_persist_probe_debug_images", None)
        self.assertIsNotNone(persist)

        class FakeCapture:
            def __init__(self, payload):
                self.payload = payload

            def save(self, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.payload)
                return path

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = self.main.Path(temp_dir)
            frame_captures = [
                (sample_dir / f"after_{index}.png", FakeCapture(bytes([index])))
                for index in range(1, 5)
            ]
            frame_records = [
                {"result": {"state": "miss", "score": score}}
                for score in (0.10, 0.25, 0.15, 0.20)
            ]

            persist(
                sample_dir,
                FakeCapture(b"before"),
                frame_captures,
                frame_records,
                preserve_all=False,
            )

            self.assertEqual((sample_dir / "before.png").read_bytes(), b"before")
            self.assertEqual((sample_dir / "after_2.png").read_bytes(), b"\x02")
            self.assertFalse((sample_dir / "after_1.png").exists())
            self.assertFalse((sample_dir / "after_3.png").exists())
            self.assertFalse((sample_dir / "after_4.png").exists())
            self.assertEqual(
                [record["saved"] for record in frame_records],
                [False, True, False, False],
            )

    def test_uncertain_probe_debug_images_preserve_every_frame(self):
        persist = getattr(self.main, "_persist_probe_debug_images", None)
        preserve_all = getattr(self.main, "_should_preserve_all_probe_images", None)
        self.assertIsNotNone(persist)
        self.assertIsNotNone(preserve_all)

        class FakeCapture:
            def __init__(self, payload):
                self.payload = payload

            def save(self, path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.payload)
                return path

        frame_records = [
            {
                "dynamic_hit_vetoed": index == 2,
                "sidebar_completed_lengths": [],
                "result": {
                    "state": "hit" if index == 1 else "miss",
                    "score": 0.9 if index == 1 else 0.2,
                },
            }
            for index in range(1, 5)
        ]
        self.assertTrue(
            preserve_all(
                frame_records,
                suspect_extra_checked=False,
                victory_detected=False,
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = self.main.Path(temp_dir)
            captures = [
                (sample_dir / f"after_{index}.png", FakeCapture(bytes([index])))
                for index in range(1, 5)
            ]
            persist(
                sample_dir,
                FakeCapture(b"before"),
                captures,
                frame_records,
                preserve_all=True,
            )

            self.assertEqual(
                sorted(path.name for path in sample_dir.glob("*.png")),
                ["after_1.png", "after_2.png", "after_3.png", "after_4.png", "before.png"],
            )
            self.assertTrue(all(record["saved"] for record in frame_records))

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
        completion_screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=completion_screenshot)

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
            patch.object(self.main, "restart_process", return_value=False) as restart,
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
        self.assertIs(
            probe_metadata["sidebar_completion_screenshot"],
            completion_screenshot,
        )
        restart.assert_called_once_with(
            victory_wait_timeout=self.main.VICTORY_WAIT_AFTER_HIT_SECONDS,
        )

    def test_probe_metadata_resolves_trusted_completed_submarine_cells(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        click_points = [(400 + index, 300 + index) for index in range(36)]
        metadata = {
            "sidebar_completion_screenshot": screenshot,
            "sidebar_completed_lengths": (3,),
        }

        with patch.object(
            self.main,
            "detect_completed_submarine_candidate_cells",
            return_value={(2, 1), (2, 2), (2, 3), (1, 2)},
        ):
            trusted = self.main._trusted_completed_cells_from_probe_metadata(
                metadata,
                click_points,
                grid_size=6,
                anchor=(2, 2),
            )

        self.assertEqual(trusted, {(2, 1), (2, 2), (2, 3)})

    def test_consistent_incomplete_sidebar_frames_use_short_victory_wait(self):
        select_timeout = getattr(
            self.main,
            "_victory_wait_timeout_for_sidebar_samples",
            None,
        )
        self.assertIsNotNone(select_timeout)
        progress = SidebarProgress(
            active_lengths=(4,),
            completed_lengths=(5, 3, 2, 2),
        )

        timeout = select_timeout(
            [progress] * len(self.main.HIT_RESULT_FRAME_DELAYS),
            (2, 2, 3, 4, 5),
        )

        self.assertEqual(
            timeout,
            self.main.VICTORY_WAIT_AFTER_CONFIRMED_INCOMPLETE_SECONDS,
        )

    def test_uncertain_sidebar_frames_keep_full_victory_wait(self):
        select_timeout = getattr(
            self.main,
            "_victory_wait_timeout_for_sidebar_samples",
            None,
        )
        self.assertIsNotNone(select_timeout)
        incomplete = SidebarProgress(
            active_lengths=(4,),
            completed_lengths=(5, 3, 2, 2),
        )
        inconsistent = SidebarProgress(
            active_lengths=(3, 4),
            completed_lengths=(5, 2, 2),
        )
        invalid = SidebarProgress(
            active_lengths=(4,),
            completed_lengths=(5, 3, 2),
            unknown_lengths=(2,),
        )
        required_frames = len(self.main.HIT_RESULT_FRAME_DELAYS)
        cases = {
            "too_few": [incomplete] * (required_frames - 1),
            "missing": [incomplete, None, incomplete, incomplete],
            "invalid": [incomplete, invalid, incomplete, incomplete],
            "inconsistent": [incomplete, inconsistent, incomplete, incomplete],
        }

        for name, samples in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    select_timeout(samples, (2, 2, 3, 4, 5)),
                    self.main.VICTORY_WAIT_AFTER_HIT_SECONDS,
                )

    def test_completed_sidebar_frames_keep_full_victory_wait(self):
        select_timeout = getattr(
            self.main,
            "_victory_wait_timeout_for_sidebar_samples",
            None,
        )
        self.assertIsNotNone(select_timeout)
        progress = SidebarProgress(completed_lengths=(5, 4, 3, 2, 2))

        timeout = select_timeout(
            [progress] * len(self.main.HIT_RESULT_FRAME_DELAYS),
            (2, 2, 3, 4, 5),
        )

        self.assertEqual(timeout, self.main.VICTORY_WAIT_AFTER_HIT_SECONDS)

    def test_adaptive_frames_require_consistent_incomplete_sidebar_progress(self):
        can_stop = getattr(self.main, "_can_stop_probe_frames_early", None)
        self.assertIsNotNone(can_stop)
        records = [
            {
                "dynamic_hit_vetoed": False,
                "result": {
                    "state": "hit",
                    "score": 0.99,
                    "evidence_vetoed": False,
                },
            }
            for _index in range(3)
        ]
        incomplete = SidebarProgress(active_lengths=(3,), completed_lengths=(2,))
        complete = SidebarProgress(completed_lengths=(3, 2))
        inconsistent = SidebarProgress(active_lengths=(2, 3))

        self.assertTrue(can_stop(records, [incomplete] * 3, (2, 3)))
        self.assertFalse(can_stop(records, [complete] * 3, (2, 3)))
        self.assertFalse(
            can_stop(records, [incomplete, inconsistent, incomplete], (2, 3))
        )
        self.assertFalse(can_stop(records, [incomplete, None, incomplete], (2, 3)))

    def test_probe_hit_uses_short_wait_after_consistent_incomplete_sidebar_frames(self):
        hit_map = [[0, 0, 0] for _ in range(3)]
        progress = SidebarProgress(
            active_lengths=(3,),
            completed_lengths=(2,),
        )

        def confirm_sidebar(_before, _after, _fleet, result):
            result.state = "hit"
            result.score = 0.99
            result.confidence = 0.99
            return True, progress, (2,)

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "click_template", return_value=True),
            patch.object(self.main, "_wait_until_activity_detail_closed", return_value=True),
            patch.object(self.main, "enter_activity"),
            patch.object(self.main, "get_configured_submarines", return_value=[2, 3]),
            patch.object(
                self.main,
                "classify_diamond_hit",
                side_effect=lambda *_args, **_kwargs: dummy_hit_result("miss"),
            ) as classify,
            patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
            patch.object(
                self.main,
                "apply_sidebar_completion_confirmation",
                side_effect=confirm_sidebar,
            ),
            patch.object(self.main, "restart_process", return_value=False) as restart,
        ):
            result = self.main._probe_cell(
                level=1,
                hit_map=hit_map,
                cell=(0, 1),
                point=(400, 300),
                index=1,
            )

        self.assertEqual(result, self.main.ProbeResult.HIT)
        self.assertEqual(classify.call_count, 3)
        restart.assert_called_once_with(
            victory_wait_timeout=(
                self.main.VICTORY_WAIT_AFTER_CONFIRMED_INCOMPLETE_SECONDS
            ),
        )

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

    def test_strategy_prioritizes_all_scout_miss_neighbors_before_normal_search(self):
        targets = [(0, 1), (2, 1), (1, 0), (1, 2)]
        strategy = SimpleNamespace(
            shots={(1, 1): True},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((3,))),
            get_confirmed_ships=lambda: [],
            get_accounted_completed_lengths=lambda: [],
            get_priority_scout_miss_recheck_targets=Mock(
                side_effect=[targets, []]
            ),
            choose_next_cell=Mock(return_value=(0, 0)),
        )

        def report_result(cell, hit):
            strategy.shots[cell] = hit
            if all(target in strategy.shots for target in targets):
                strategy.done = True

        strategy.report_result = Mock(side_effect=report_result)
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(
                self.main,
                "_probe_cell",
                side_effect=[
                    self.main.ProbeResult.HIT,
                    self.main.ProbeResult.MISS,
                    self.main.ProbeResult.MISS,
                    self.main.ProbeResult.MISS,
                ],
            ) as probe,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_hits={(1, 1)},
                initial_visual_hit_count=1,
            )

        self.assertTrue(completed)
        self.assertEqual(
            [call.args[2] for call in probe.call_args_list],
            targets,
        )
        self.assertEqual(probe.call_count, 4)
        strategy.choose_next_cell.assert_not_called()
        self.assertEqual(
            [call.args for call in strategy.report_result.call_args_list],
            [
                ((0, 1), True),
                ((2, 1), False),
                ((1, 0), False),
                ((1, 2), False),
            ],
        )
        self.assertEqual(self.main._runtime_status["phase"], "supplemental_recheck")
        self.assertEqual(self.main._runtime_status["supplemental_rechecks_done"], 4)

    def test_strategy_prioritizes_aligned_hit_line_ends_before_normal_search(self):
        targets = [(1, 0), (1, 3)]
        strategy = SimpleNamespace(
            shots={(1, 1): True, (1, 2): True},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((4,))),
            get_confirmed_ships=lambda: [],
            get_accounted_completed_lengths=lambda: [],
            get_priority_scout_miss_recheck_targets=Mock(
                side_effect=[targets, []]
            ),
            choose_next_cell=Mock(return_value=None),
        )

        def report_result(cell, hit):
            strategy.shots[cell] = hit
            if all(target in strategy.shots for target in targets):
                strategy.done = True

        strategy.report_result = Mock(side_effect=report_result)
        fake_bar = SimpleNamespace(
            total=4,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(self.main, "_scan_level_by_grid_order", return_value=0),
            patch.object(
                self.main,
                "_probe_cell",
                side_effect=[
                    self.main.ProbeResult.MISS,
                    self.main.ProbeResult.MISS,
                ],
            ) as probe,
        ):
            completed = self.main._scan_level_by_strategy(
                level=2,
                hit_map=[[0] * 4 for _row in range(4)],
                click_points=[(400, 300)] * 16,
                submarines=[4],
                initial_hits={(1, 1), (1, 2)},
                initial_visual_hit_count=2,
            )

        self.assertTrue(completed)
        self.assertEqual(
            [call.args[2] for call in probe.call_args_list],
            targets,
        )
        strategy.choose_next_cell.assert_not_called()

    def test_supplemental_neighbor_recheck_stops_when_victory_appears(self):
        strategy = SimpleNamespace(
            shots={(1, 1): True},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((3,))),
            get_confirmed_ships=lambda: [],
            get_accounted_completed_lengths=lambda: [],
            get_priority_scout_miss_recheck_targets=Mock(
                return_value=[(0, 1), (2, 1), (1, 0), (1, 2)]
            ),
            choose_next_cell=Mock(return_value=(0, 0)),
            report_result=Mock(),
        )
        fake_bar = SimpleNamespace(
            total=1,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.LEVEL_COMPLETE,
            ),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[1],
                initial_hits={(1, 1)},
                initial_visual_hit_count=1,
            )

        self.assertTrue(completed)
        self.assertEqual(self.main._runtime_status["phase"], "level_complete")
        self.assertEqual(self.main._runtime_status["supplemental_rechecks_done"], 1)
        self.assertEqual(self.main._runtime_status["last_result"], "level_complete")
        self.assertEqual(self.main._runtime_status["board_states"][1][1], "hit")
        strategy.report_result.assert_not_called()
        strategy.choose_next_cell.assert_not_called()

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
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                return_value=DummyMatch((640, 360)),
            ),
            patch.object(
                self.main,
                "wait_until_retry_button",
                return_value=DummyMatch((374, 442)),
            ),
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

    def test_miss_discard_uses_connection_retry_without_closing_app(self):
        transaction = self.main.ProbeTransaction(level=1, cell=(0, 1), index=1)
        transaction.advance(self.main.ProbePhase.REQUEST_PENDING)
        transaction.advance(self.main.ProbePhase.RESULT_VISIBLE)
        transaction.advance(self.main.ProbePhase.RESULT_RECORDED)
        retry = DummyMatch((123, 456))
        package_name = self.main.GAME_PACKAGE_NAME

        def connection_dialog_after_reject(*, timeout):
            self.assertEqual(timeout, self.main.MISS_CONNECTION_DIALOG_WAIT_SECONDS)
            self.assertIn(("enable_reject_network", package_name), self.adb.calls)
            self.assertNotIn(("disable_weak_network", package_name), self.adb.calls)
            self.assertNotIn(("disable_reject_network", package_name), self.adb.calls)
            return DummyMatch((100, 100))

        def retry_button_while_isolated(*, timeout):
            self.assertEqual(timeout, self.main.MISS_RETRY_BUTTON_WAIT_SECONDS)
            self.assertNotIn(("disable_weak_network", package_name), self.adb.calls)
            self.assertNotIn(("disable_reject_network", package_name), self.adb.calls)
            return retry

        with (
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                side_effect=connection_dialog_after_reject,
            ) as dialog,
            patch.object(
                self.main,
                "wait_until_retry_button",
                side_effect=retry_button_while_isolated,
            ) as retry_wait,
            patch.object(self.main, "enter_activity", return_value=False) as enter,
        ):
            completed = self.main._discard_pending_request_and_prepare_next_probe(transaction)

        self.assertFalse(completed)
        self.assertEqual(transaction.phase, self.main.ProbePhase.COMPLETE)
        self.assertNotIn(("close_app", package_name), self.adb.calls)
        self.assertNotIn(("open_app", package_name), self.adb.calls)
        self.assertIn(("click", *retry.center), self.adb.calls)
        self.assertLess(
            self.adb.calls.index(("enable_reject_network", package_name)),
            self.adb.calls.index(("disable_weak_network", package_name)),
        )
        self.assertNotIn(("delay", 2.0), self.adb.calls)
        self.assertLess(
            self.adb.calls.index(("disable_weak_network", package_name)),
            self.adb.calls.index(("disable_reject_network", package_name)),
        )
        self.assertLess(
            self.adb.calls.index(("disable_reject_network", package_name)),
            self.adb.calls.index(("click", *retry.center)),
        )
        self.assertNotIn(("delay", 0.8), self.adb.calls)
        dialog.assert_called_once_with(
            timeout=self.main.MISS_CONNECTION_DIALOG_WAIT_SECONDS,
        )
        retry_wait.assert_called_once_with(
            timeout=self.main.MISS_RETRY_BUTTON_WAIT_SECONDS,
        )
        enter.assert_called_once_with(
            re_enter=True,
            max_retries=1,
            prepare_activity_list=True,
            activity_button_timeout=self.main.POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        )

    def test_miss_discard_keeps_network_isolated_when_dialog_is_missing(self):
        transaction = self.main.ProbeTransaction(level=1, cell=(0, 1), index=1)
        transaction.advance(self.main.ProbePhase.REQUEST_PENDING)
        transaction.advance(self.main.ProbePhase.RESULT_VISIBLE)
        transaction.advance(self.main.ProbePhase.RESULT_RECORDED)

        with (
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                return_value=None,
            ),
            self.assertRaisesRegex(self.main.ProbeProtocolError, "连接中断"),
        ):
            self.main._discard_pending_request_and_prepare_next_probe(transaction)

        package_name = self.main.GAME_PACKAGE_NAME
        self.assertEqual(transaction.phase, self.main.ProbePhase.REQUEST_DISCARDED)
        self.assertIn(("enable_reject_network", package_name), self.adb.calls)
        self.assertNotIn(("disable_weak_network", package_name), self.adb.calls)
        self.assertNotIn(("disable_reject_network", package_name), self.adb.calls)
        self.assertNotIn(("close_app", package_name), self.adb.calls)
        self.assertIn("未检测到连接中断弹窗", self.main._network_fail_closed_reason)

    def test_miss_transaction_clicks_retry_instead_of_closing_app(self):
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
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                return_value=DummyMatch((100, 100)),
            ) as dialog,
            patch.object(
                self.main,
                "wait_until_retry_button",
                return_value=DummyMatch((123, 456)),
            ) as retry,
            patch.object(self.main, "click_template", return_value=True),
            patch.object(self.main, "_wait_until_activity_detail_closed", return_value=True),
            patch.object(self.main, "classify_diamond_hit", return_value=dummy_hit_result("miss")),
            patch.object(self.main, "enter_activity", return_value=False),
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
        self.assertNotIn(("close_app", package_name), self.adb.calls)
        self.assertNotIn(("open_app", package_name), self.adb.calls)
        self.assertIn(("click", 123, 456), self.adb.calls)
        dialog.assert_called_once()
        retry.assert_called_once()

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
