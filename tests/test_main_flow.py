import importlib
import inspect
import json
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from utils.sidebar_progress import CompletedShipResolution, SidebarProgress


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
    def test_probe_result_json_preserves_unknown_decision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = self.main.Path(temp_dir)
            self.main._save_probe_result_json(
                sample_dir,
                level=1,
                cell=(0, 1),
                index=1,
                point=(640, 360),
                hit=False,
                hit_votes=0,
                frames=[],
                suspect_extra_checked=True,
                result_unknown=True,
            )
            payload = json.loads((sample_dir / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["decision"], "unknown")

    def test_optional_stable_analysis_failure_does_not_abort_probe(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        captures = [
            (self.main.Path(f"after_{index}.png"), FakeScreenshotCapture(image))
            for index in range(3)
        ]

        with patch.object(
            self.main,
            "analyze_stable_hit",
            side_effect=RuntimeError("analysis failed"),
        ):
            analysis = self.main._analyze_stable_probe_frames(
                image,
                captures,
                (10, 10),
            )

        self.assertIsNone(analysis)

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

    def test_runtime_evidence_retention_defaults_are_bounded(self):
        self.assertEqual(self.main.MAX_PROBE_SAMPLE_DIRS, 20)
        self.assertEqual(self.main.MAX_RED_SCOUT_SAMPLE_DIRS, 10)
        self.assertEqual(self.main.MAX_SCREENSHOT_STORAGE_BYTES, 500 * 1024 * 1024)

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

    def test_red_result_transition_filter_discards_dialog_frame_with_two_clean_frames(self):
        before = np.zeros((120, 160, 3), dtype=np.uint8)
        frames = [
            np.full_like(before, 10),
            np.full_like(before, 20),
            np.full_like(before, 30),
        ]
        with (
            patch.object(self.main, "find_connection_interrupted_dialog", side_effect=[None, DummyMatch((80, 60)), None]),
            patch.object(self.main, "find_victory_banner", return_value=None),
        ):
            filtered, diagnostics = self.main._filter_red_result_transition_frames(
                before,
                frames,
            )

        self.assertEqual(filtered, (frames[0], frames[2]))
        self.assertTrue(diagnostics["filter_applied"])
        self.assertEqual(diagnostics["kept_indices"], (0, 2))
        self.assertEqual(
            diagnostics["discarded_frames"],
            ({"index": 1, "reasons": ("connection_interrupted_dialog",)},),
        )

    def test_red_result_transition_filter_keeps_raw_frames_when_fewer_than_two_clean(self):
        before = np.zeros((120, 160, 3), dtype=np.uint8)
        frames = [np.full_like(before, value) for value in (10, 20, 30)]
        with (
            patch.object(
                self.main,
                "find_connection_interrupted_dialog",
                return_value=DummyMatch((80, 60)),
            ),
            patch.object(self.main, "find_victory_banner", return_value=None),
        ):
            filtered, diagnostics = self.main._filter_red_result_transition_frames(
                before,
                frames,
            )

        self.assertIs(filtered, frames)
        self.assertFalse(diagnostics["filter_applied"])
        self.assertEqual(diagnostics["reason"], "insufficient_stable_frames")

    def test_red_analysis_attaches_transition_filter_diagnostics(self):
        before = np.zeros((120, 160, 3), dtype=np.uint8)
        frames = [np.full_like(before, value) for value in (10, 20, 30)]
        result = self._valid_red_result()
        with (
            patch.object(
                self.main,
                "find_connection_interrupted_dialog",
                side_effect=[None, DummyMatch((80, 60)), None],
            ),
            patch.object(self.main, "find_victory_banner", return_value=None),
            patch.object(self.main, "_analyze_red_result", return_value=result),
        ):
            analyzed = self.main._analyze_red_result_with_baseline_consensus(
                before_images=[before],
                after_images=frames,
                click_points=[(0, 0)] * 9,
                grid_size=3,
                center_cell=(1, 1),
                submarine_lengths=[3],
            )

        self.assertIsNot(analyzed, result)
        self.assertEqual(
            analyzed.diagnostics["capture_frame_filter"]["kept_indices"],
            (0, 2),
        )
        self.assertEqual(
            analyzed.diagnostics["capture_frame_filter"]["discarded_frames"][0]["index"],
            1,
        )

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

    def test_total_screenshot_limit_removes_oldest_samples_across_modes(self):
        root = self.main.Path(self.runtime_temp.name)
        probe_root = root / "probes"
        red_root = root / "red"
        run_debug = root / "run_debug"
        probe_root.mkdir()
        red_root.mkdir()
        run_debug.mkdir()
        old_probe = probe_root / "level_1_cell_0_sample"
        current_probe = probe_root / "level_1_cell_1_sample"
        red_sample = red_root / "level_1_attempt_01_sample"
        for index, directory in enumerate((old_probe, red_sample, current_probe), start=1):
            directory.mkdir()
            (directory / "frame.png").write_bytes(b"x" * 40)
            os.utime(directory, (index, index))
        (run_debug / "latest.png").write_bytes(b"x" * 10)

        with (
            patch.object(self.main, "PROBE_SAMPLE_DIR", probe_root),
            patch.object(self.main, "RED_SCOUT_SAMPLE_DIR", red_root),
            patch.object(self.main, "RUN_DEBUG_DIR", run_debug),
        ):
            self.main._prune_screenshot_storage(
                max_bytes=90,
                protected_paths=(current_probe,),
            )

        self.assertFalse(old_probe.exists())
        self.assertTrue(red_sample.exists())
        self.assertTrue(current_probe.exists())
        self.assertTrue((run_debug / "latest.png").exists())

    def test_successful_red_scout_keeps_only_representative_images(self):
        compact = getattr(self.main, "_compact_successful_red_scout_images", None)
        self.assertIsNotNone(compact)
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = self.main.Path(temp_dir)
            for prefix, count in (("before", 3), ("after", 4), ("verify", 3)):
                for index in range(count):
                    (sample_dir / f"{prefix}_{index}.png").write_bytes(bytes([index]))
            for name in ("selected.png", "exit_attempt.png"):
                (sample_dir / name).write_bytes(b"image")
            (sample_dir / "analysis.json").write_text("{}", encoding="utf-8")

            compact(sample_dir)

            self.assertEqual(
                sorted(path.name for path in sample_dir.glob("*.png")),
                ["after_1.png", "before_1.png", "selected.png", "verify_1.png"],
            )
            self.assertTrue((sample_dir / "analysis.json").exists())

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
        excluded_by_attempt = [
            set(call.kwargs["excluded_cells"])
            for call in execute.call_args_list
        ]
        self.assertEqual(excluded_by_attempt[0], set())
        self.assertTrue(
            all(
                {(1, 1), (1, 2)} <= excluded
                for excluded in excluded_by_attempt[1:]
            )
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
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
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

    def test_red_local_victory_stops_remaining_scouts_and_enters_blue_attack(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        visual_candidates = {(0, 0)}
        stale_baseline = self.main.SurfaceWaterBaseline(
            median_gray=np.zeros((3, 3), dtype=np.uint8),
            temporal_mad=np.zeros((3, 3), dtype=np.float32),
            frame_count=2,
        )
        local_victory = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset(),
            hit_cells=frozenset(),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=None,
            valid=False,
            confidence_by_cell={},
            invalid_reason="local_victory_screen",
        )
        with (
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=local_victory,
            ) as red_attempt,
            patch.object(self.main, "_clear_red_victory_before_blue_attack"),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                [(400, 300)] * 9,
                [2, 3],
                set(),
                settings,
                initial_visual_candidates=visual_candidates,
                surface_baseline=stale_baseline,
            )

        self.assertTrue(completed)
        red_attempt.assert_called_once()
        scan.assert_called_once()
        self.assertEqual(scan.call_args.kwargs["initial_visual_candidates"], visual_candidates)
        self.assertIs(scan.call_args.kwargs["surface_baseline"], stale_baseline)
        self.assertEqual(scan.call_args.kwargs["initial_hits"], set())

    def test_red_local_victory_preserves_previous_level_evidence_and_hit_map(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 3)
        local_victory = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=frozenset(),
            hit_cells=frozenset(),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=None,
            valid=False,
            confidence_by_cell={},
            invalid_reason="local_victory_screen",
        )
        hit_map = [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
        placement = self.main.Placement(
            length=2,
            direction="H",
            cells=((0, 0), (0, 1)),
        )
        with (
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=local_victory,
            ),
            patch.object(self.main, "_clear_red_victory_before_blue_attack"),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                hit_map,
                [(400, 300)] * 9,
                [2, 3],
                {(0, 0)},
                settings,
                initial_misses={(0, 1)},
                initial_sidebar_progress=self.main.SidebarProgress(
                    active_lengths=(3,), completed_lengths=(2,)
                ),
                initial_visual_hit_count=2,
                initial_completed_visual_hits={(0, 0), (0, 1)},
                initial_red_marker_completed_cells={(0, 0), (0, 1)},
                initial_authoritative_completed_visual_hits={(0, 0), (0, 1)},
                initial_authoritative_completed_placements=(placement,),
                initial_completed_lengths=(2,),
                initial_scout_hits={(2, 2)},
                initial_scout_misses={(2, 1)},
            )

        self.assertTrue(completed)
        self.assertEqual(hit_map, [[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        kwargs = scan.call_args.kwargs
        self.assertTrue({(0, 0)} <= kwargs["initial_hits"])
        self.assertTrue({(0, 1)} <= kwargs["initial_misses"])
        self.assertTrue({(2, 2)} <= kwargs["initial_scout_hits"])
        self.assertTrue({(2, 1)} <= kwargs["initial_scout_misses"])
        self.assertEqual(kwargs["initial_completed_lengths"], (2,))
        self.assertEqual(kwargs["initial_completed_visual_hits"], {(0, 0), (0, 1)})
        self.assertEqual(kwargs["initial_authoritative_completed_visual_hits"], {(0, 0), (0, 1)})
        self.assertEqual(kwargs["initial_authoritative_completed_placements"], (placement,))
        self.assertEqual(kwargs["initial_visual_hit_count"], 2)
        self.assertEqual(
            kwargs["initial_sidebar_progress"],
            self.main.SidebarProgress(active_lengths=(3,), completed_lengths=(2,)),
        )
        self.assertEqual(kwargs["initial_red_marker_completed_cells"], {(0, 0), (0, 1)})

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
                [(400, 300)] * 9,
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
        self.assertEqual(self.adb.calls.count(("click", *self.main.BLUE_BOMB_POINT)), 0)
        self.assertIn(("disable_reject_network", package_name), self.adb.calls)
        self.assertIn(("disable_weak_network", package_name), self.adb.calls)
        self.assertNotIn(("enable_reject_network", package_name), self.adb.calls)
        self.assertNotIn(("enable_weak_network", package_name), self.adb.calls)

    def test_online_scout_visual_change_keeps_hit_map_cell(self):
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

        self.assertEqual(result, self.main.ProbeResult.HIT)
        self.assertEqual(hit_map[1][1], 1)

    def test_online_scout_batch_clicks_before_shared_result_analysis(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )
        events = []

        def classify(*args, **_kwargs):
            events.append(("classify", args[2]))
            return dummy_hit_result("hit")

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "find_victory_banner", return_value=None),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    return_value=None,
                ),
                patch.object(self.main, "classify_diamond_hit", side_effect=classify),
                patch.object(
                    self.main,
                    "apply_wreck_template_confirmation",
                    return_value=True,
                ),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, None, ()),
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    activity_ready=True,
                )

        self.assertEqual(
            self.adb.calls.count(("click", *self.main.BLUE_BOMB_POINT)),
            0,
        )
        target_events = [event for event in self.adb.calls if event[0] == "click"]
        self.assertEqual(target_events[-2:], [("click", 400, 300), ("click", 500, 300)])
        self.assertEqual(result.clicked_cells, ((1, 1), (1, 2)))
        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.HIT)
        self.assertEqual(result.results[(1, 2)], self.main.ProbeResult.HIT)
        self.assertTrue(events)

    def test_online_scout_batch_stops_before_next_tap_when_victory_appears(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        victory_frame = np.full_like(image, 80)
        self.adb.read_screenshot = Mock(side_effect=[image, victory_frame])
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(victory_frame)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(
                    self.main,
                    "find_victory_banner",
                    side_effect=lambda frame: DummyMatch((1, 1))
                    if frame is victory_frame
                    else None,
                ),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    return_value=None,
                ),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "classify_diamond_hit", return_value=dummy_hit_result("hit")),
                patch.object(self.main, "apply_wreck_template_confirmation", return_value=True),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, None, ()),
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    activity_ready=True,
                    blue_bomb_ready=True,
                    network_ready=True,
                )

        target_events = [event for event in self.adb.calls if event[0] == "click"]
        self.assertEqual(target_events, [("click", 400, 300)])
        self.assertEqual(result.clicked_cells, ((1, 1),))
        self.assertIn(
            result.stopped_reason,
            {"victory_banner_between_taps", "victory_banner_after_batch"},
        )
        self.assertTrue(result.level_completed)

    def test_online_scout_batch_red_marker_is_not_visible_hit(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        with (
            patch.object(self.main, "red_hit_marker_visible", return_value=True),
            patch.object(self.main, "red_submarine_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
        ):
            visible = self.main._visible_wreck_for_hit_state(image, (400, 300))

        self.assertFalse(visible)

    def test_online_scout_precheck_rejects_neighbor_wreck_without_strong_cell_body(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        polygon = np.asarray(
            [(380, 300), (400, 280), (420, 300), (400, 320)],
            dtype=np.float32,
        )
        with (
            patch.object(self.main, "visible_wreck_static_detected", return_value=True),
            patch.object(self.main, "completed_ship_body_score", return_value=0.183),
        ):
            visible = self.main._visible_wreck_for_hit_state(
                image,
                (400, 300),
                red_marker_cells=set(),
                cell=(0, 7),
                cell_polygon=polygon,
                require_strong_body=True,
            )

        self.assertFalse(visible)

    def test_online_scout_precheck_accepts_wreck_with_strong_cell_body(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        polygon = np.asarray(
            [(380, 300), (400, 280), (420, 300), (400, 320)],
            dtype=np.float32,
        )
        with (
            patch.object(self.main, "visible_wreck_static_detected", return_value=True),
            patch.object(
                self.main,
                "completed_ship_body_score",
                return_value=self.main.COMPLETED_SHIP_BODY_MIN_SCORE,
            ),
        ):
            visible = self.main._visible_wreck_for_hit_state(
                image,
                (400, 300),
                red_marker_cells=set(),
                cell=(1, 7),
                cell_polygon=polygon,
                require_strong_body=True,
            )

        self.assertTrue(visible)

    def test_online_scout_batch_fires_targets_when_red_marker_is_present(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(
                    self.main,
                    "red_hit_marker_visible",
                    side_effect=[True, True],
                ),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "find_victory_banner", return_value=None),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    return_value=None,
                ),
                patch.object(
                    self.main,
                    "classify_diamond_hit",
                    return_value=dummy_hit_result("hit"),
                ),
                patch.object(
                    self.main,
                    "apply_wreck_template_confirmation",
                    return_value=True,
                ),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, None, ()),
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    activity_ready=True,
                )

        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.HIT)
        self.assertEqual(result.results[(1, 2)], self.main.ProbeResult.HIT)
        self.assertEqual(result.clicked_cells, ((1, 1), (1, 2)))
        self.assertIn(("click", 400, 300), self.adb.calls)
        self.assertIn(("click", 500, 300), self.adb.calls)

    def test_online_scout_batch_without_submarine_lengths_does_not_reject_targets(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "find_victory_banner", return_value=None),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    return_value=None,
                ),
                patch.object(
                    self.main,
                    "classify_diamond_hit",
                    return_value=dummy_hit_result("hit"),
                ),
                patch.object(
                    self.main,
                    "apply_wreck_template_confirmation",
                    return_value=True,
                ),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, None, ()),
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    activity_ready=True,
                )

        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.HIT)
        self.assertEqual(result.results[(1, 2)], self.main.ProbeResult.HIT)

    def test_online_scout_batch_accepts_visual_change_for_all_targets(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )

        def classify(_before, _after, point):
            return dummy_hit_result("hit" if point == (400, 300) else "miss")

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "find_victory_banner", return_value=None),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    return_value=None,
                ),
                patch.object(self.main, "classify_diamond_hit", side_effect=classify),
                patch.object(
                    self.main,
                    "apply_wreck_template_confirmation",
                    side_effect=lambda _image, point, _result, **_kwargs: point == (400, 300),
                ),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, None, ()),
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    activity_ready=True,
                )

        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.HIT)
        self.assertEqual(result.results[(1, 2)], self.main.ProbeResult.HIT)
        self.assertEqual(result.clicked_cells, ((1, 1), (1, 2)))

    def test_online_scout_batch_promotes_changed_frames_without_sidebar_completion(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )
        completed_progress = SidebarProgress(completed_lengths=(3,))

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "find_victory_banner", return_value=None),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    return_value=None,
                ),
                patch.object(
                    self.main,
                    "classify_diamond_hit",
                    side_effect=lambda *_args, **_kwargs: dummy_hit_result("miss"),
                ),
                patch.object(
                    self.main,
                    "apply_wreck_template_confirmation",
                    return_value=False,
                ),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, completed_progress, ()),
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    activity_ready=True,
                )

        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.HIT)
        self.assertEqual(
            result.results[(1, 2)],
            self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE,
        )

    def test_online_scout_batch_accepts_changed_targets_independent_of_sidebar_mutation(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )
        completed_progress = SidebarProgress(completed_lengths=(3,))

        def mutate_sidebar_result(_before, _after, _fleet, target_result):
            target_result.state = "hit"
            target_result.score = 1.0
            target_result.confidence = 1.0
            return True, completed_progress, (3,)

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "find_victory_banner", return_value=None),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    return_value=None,
                ),
                patch.object(
                    self.main,
                    "classify_diamond_hit",
                    side_effect=lambda *_args, **_kwargs: dummy_hit_result("miss"),
                ),
                patch.object(
                    self.main,
                    "apply_wreck_template_confirmation",
                    return_value=False,
                ),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    side_effect=mutate_sidebar_result,
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    activity_ready=True,
                )

        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.HIT)
        self.assertEqual(
            result.results[(1, 2)],
            self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE,
        )

    def test_online_scout_batch_wraps_click_interval_failure_without_retrying(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        original_delay = self.adb.delay

        def delay(seconds):
            if seconds == self.main.ONLINE_SCOUT_BATCH_CLICK_INTERVAL_SECONDS:
                raise RuntimeError("interval failure")
            return original_delay(seconds)

        self.adb.delay = Mock(side_effect=delay)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image)
                for _ in self.main.ONLINE_SCOUT_BATCH_FRAME_DELAYS
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "_create_probe_sample_dir", return_value=sample_root),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                with self.assertRaisesRegex(
                    self.main.ProbeProtocolError,
                    "click interval failed after 1 taps",
                ):
                    self.main._execute_online_scout_hit_batch(
                        level=1,
                        hit_map=[[0] * 3 for _row in range(3)],
                        targets=[
                            ((1, 1), (400, 300), 4),
                            ((1, 2), (500, 300), 5),
                            ((1, 0), (300, 300), 3),
                        ],
                        submarines=[4],
                        activity_ready=True,
                    )

        self.assertIn(("click", 400, 300), self.adb.calls)
        self.assertNotIn(("click", 500, 300), self.adb.calls)
        self.assertNotIn(("click", 300, 300), self.adb.calls)

    def test_online_scout_batch_stops_when_connection_dialog_detection_fails(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            return_value=FakeScreenshotCapture(image)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    side_effect=RuntimeError("dialog matcher failed"),
                ),
                patch.object(self.main, "classify_diamond_hit") as classify,
                patch.object(self.main, "_create_probe_sample_dir", return_value=sample_root),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                with self.assertRaisesRegex(
                    self.main.ProbeProtocolError,
                    "connection dialog detection failed",
                ):
                    self.main._execute_online_scout_hit_batch(
                        level=1,
                        hit_map=[[0] * 3 for _row in range(3)],
                        targets=[((1, 1), (400, 300), 4)],
                        submarines=[3],
                        activity_ready=True,
                    )

        classify.assert_not_called()

    def test_online_scout_batch_victory_resolves_overlay_unknowns_and_closes_prompt(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        victory_frame = np.ones((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image),
                FakeScreenshotCapture(victory_frame),
            ]
        )

        def classify(_before, _after, point):
            if point == (400, 300):
                weak = dummy_hit_result("hit")
                weak.score = 0.70
                weak.confidence = 0.70
                return weak
            return dummy_hit_result("miss")

        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", side_effect=[False, True]) as handle_victory,
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(self.main, "find_victory_banner", side_effect=[None, DummyMatch((640, 360))]),
                patch.object(self.main, "find_connection_interrupted_dialog", return_value=None),
                patch.object(self.main, "classify_diamond_hit", side_effect=classify),
                patch.object(
                    self.main,
                    "apply_wreck_template_confirmation",
                    side_effect=[True, False, False, False],
                ),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, None, ()),
                ),
                patch.object(
                    self.main,
                    "_create_probe_sample_dir",
                    side_effect=lambda _level, _cell, index, **_kwargs: (
                        sample_root / f"cell_{index}"
                    ),
                ),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    activity_ready=True,
                )

        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.HIT)
        self.assertEqual(
            result.results[(1, 2)],
            self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE,
        )
        self.assertTrue(result.level_completed)
        self.assertIn(
            result.stopped_reason,
            {"victory_banner_between_taps", "victory_banner_after_batch", "victory_banner"},
        )
        self.assertEqual(handle_victory.call_count, 2)
        self.assertEqual(handle_victory.call_args.kwargs["timeout"], 0.0)

    def test_online_scout_batch_reserves_unmapped_visual_capacity(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        with (
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(self.main, "detect_sidebar_progress", return_value=None),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "find_connection_interrupted_dialog", return_value=None),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_raise_if_blue_ammo_depleted"),
        ):
            with self.assertRaisesRegex(
                self.main.ProbeProtocolError,
                "remaining submarine-cell capacity",
            ):
                self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    targets=[
                        ((1, 1), (400, 300), 4),
                        ((1, 2), (500, 300), 5),
                    ],
                    submarines=[3],
                    unmapped_visual_hits=1,
                    activity_ready=True,
                )

        self.assertNotIn(("click", *self.main.BLUE_BOMB_POINT), self.adb.calls)

    def test_online_scout_batch_discards_connection_overlay_frame(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=image)
        self.adb.capture_screenshot = Mock(
            side_effect=[
                FakeScreenshotCapture(image),
                FakeScreenshotCapture(image),
            ]
        )
        dialog_match = DummyMatch((640, 360))
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_root = self.main.Path(temp_dir)
            with (
                patch.object(self.main, "handle_victory_prompt", return_value=False),
                patch.object(self.main, "detect_sidebar_progress", return_value=None),
                patch.object(self.main, "red_hit_marker_visible", return_value=False),
                patch.object(self.main, "visible_wreck_static_detected", return_value=False),
                patch.object(
                    self.main,
                    "find_connection_interrupted_dialog",
                    side_effect=[None, None, None, dialog_match],
                ),
                patch.object(self.main, "find_victory_banner", return_value=None),
                patch.object(
                    self.main,
                    "classify_diamond_hit",
                    return_value=dummy_hit_result("miss"),
                ) as classify,
                patch.object(self.main, "apply_wreck_template_confirmation", return_value=False),
                patch.object(
                    self.main,
                    "apply_sidebar_completion_confirmation",
                    return_value=(False, None, ()),
                ),
                patch.object(self.main, "_create_probe_sample_dir", return_value=sample_root),
                patch.object(self.main, "_write_probe_status"),
                patch.object(self.main, "_save_probe_result_json"),
                patch.object(self.main, "_persist_probe_debug_images"),
                patch.object(self.main, "_analyze_stable_probe_frames", return_value=None),
                patch.object(self.main, "append_recent_probe_result"),
                patch.object(self.main, "_raise_if_blue_ammo_depleted"),
            ):
                result = self.main._execute_online_scout_hit_batch(
                    level=1,
                    hit_map=[[0] * 3 for _row in range(3)],
                    targets=[((1, 1), (400, 300), 4)],
                    submarines=[3],
                    activity_ready=True,
                )

        self.assertEqual(result.results[(1, 1)], self.main.ProbeResult.UNKNOWN)
        self.assertEqual(result.stopped_reason, "unknown_result")
        # Only the clean frame was classified; the dialog frame was discarded.
        self.assertEqual(classify.call_count, 1)
        self.assertEqual(self.adb.capture_screenshot.call_count, 2)

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
        self.assertNotIn(("delay", 0.1), self.adb.calls)
        self.assertNotIn(
            ("delay", self.main.ONLINE_SCOUT_BLUE_SELECT_SETTLE_SECONDS),
            self.adb.calls,
        )
        self.assertEqual(self.adb.read_screenshot.call_count, 5)

    def test_fast_blue_selection_waits_remaining_window_when_switch_is_slow(self):
        selection_screen = object()

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

        self.assertIs(result, selection_screen)
        selected.assert_not_called()
        self.assertNotIn(("delay", 0.1), self.adb.calls)
        self.assertNotIn(("delay", 0.15), self.adb.calls)
        self.assertEqual(self.adb.calls.count(("click", *self.main.BLUE_BOMB_POINT)), 0)

    def test_blue_selection_does_not_require_red_button_verification(self):
        with patch.object(self.main, "locate_red_bomb_button", return_value=None):
            self.main._select_blue_bomb_for_online_scout(
                self.main.Path("unused"),
                object(),
                fast=True,
            )

        self.assertNotIn(("click", *self.main.BLUE_BOMB_POINT), self.adb.calls)

    def test_standard_blue_selection_does_not_recheck_red_bomb_state(self):
        selection_screen = object()

        with patch.object(self.main, "red_bomb_selected") as selected:
            result = self.main._select_blue_bomb_for_online_scout(
                self.main.Path("unused"), selection_screen, fast=False
            )

        self.assertIs(result, selection_screen)
        selected.assert_not_called()
        self.assertNotIn(
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

    def test_online_scout_hit_visual_change_continues_without_clicking_twice(self):
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

    def test_online_scout_hit_does_not_need_extra_frames_for_visual_change(self):
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
        self.assertEqual(self.adb.calls.count(("capture_screenshot",)), 4)

    def test_online_scout_hit_does_not_treat_red_submarine_decoration_as_visible_hit(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(self.main, "locate_red_bomb_button", return_value=None),
            # Red above a submarine is decoration, never an already-visible hit.
            patch.object(self.main, "red_hit_marker_visible", return_value=True),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(
                self.main,
                "classify_diamond_hit",
                side_effect=lambda *_args, **_kwargs: dummy_hit_result("miss"),
            ) as classify,
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
        classify.assert_called()

    def test_blue_ammo_zero_stops_online_scout_before_target_click(self):
        hit_map = [[0, 0, 0] for _row in range(3)]
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)

        with (
            patch.object(self.main, "wait_until_occur", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "handle_victory_prompt", return_value=False),
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(self.main, "visible_wreck_static_detected", return_value=False),
            patch.object(self.main, "_create_probe_sample_dir", return_value=self.main.Path("unused")),
            patch.object(self.main, "_write_probe_status"),
            patch.object(self.main, "_raise_if_blue_ammo_depleted", side_effect=self.main.BlueAmmoDepletedError("zero")),
        ):
            with self.assertRaises(self.main.BlueAmmoDepletedError):
                self.main._execute_online_scout_hit(
                    level=1,
                    hit_map=hit_map,
                    cell=(1, 1),
                    point=(640, 360),
                    index=4,
                    submarines=[3],
                )

        self.assertNotIn(("click", 640, 360), self.adb.calls)

    def test_blue_ammo_zero_returns_to_base_through_reconnect_dialog(self):
        retry = DummyMatch((377, 435))
        package_name = self.main.GAME_PACKAGE_NAME

        with (
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                return_value=DummyMatch((640, 360)),
            ),
            patch.object(self.main, "wait_until_retry_button", return_value=retry),
            patch.object(
                self.main,
                "wait_until_occur",
                return_value=DummyMatch((1249, 269)),
            ),
        ):
            self.main._return_to_base_after_blue_ammo_depleted()

        enable_drop = ("enable_weak_network", package_name)
        enable_reject = ("enable_reject_network", package_name)
        disable_drop = ("disable_weak_network", package_name)
        disable_reject = ("disable_reject_network", package_name)
        retry_click = ("click", *retry.center)
        self.assertLess(self.adb.calls.index(enable_drop), self.adb.calls.index(enable_reject))
        self.assertLess(self.adb.calls.index(enable_reject), self.adb.calls.index(disable_drop))
        self.assertLess(self.adb.calls.index(disable_drop), self.adb.calls.index(disable_reject))
        self.assertLess(self.adb.calls.index(disable_reject), self.adb.calls.index(retry_click))

    def test_blue_ammo_zero_template_matches_counter_roi(self):
        template = cv2.imread(str(self.main.BLUE_BOMB_ZERO_TEMPLATE))
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        screenshot[678:694, 1146:1160] = template

        self.assertTrue(self.main._blue_bomb_zero_visible(screenshot))

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

    def test_red_batch_unknown_is_not_hidden_by_later_level_complete(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        scout_cells = frozenset({(1, 1), (1, 2)})
        red_result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=scout_cells,
            hit_cells=scout_cells,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(scout_cells),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in scout_cells},
        )
        batch_result = self.main.OnlineScoutBatchResult(
            results={
                (1, 1): self.main.ProbeResult.UNKNOWN,
                (1, 2): self.main.ProbeResult.HIT_AND_LEVEL_COMPLETE,
            },
            metadata={
                (1, 1): {"batch": True, "stable_state": "unknown"},
                (1, 2): {"batch": True, "stable_state": "hit"},
            },
        )

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(0, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=red_result),
            patch.object(
                self.main,
                "_execute_online_scout_hit_batch",
                return_value=batch_result,
            ) as batch,
            patch.object(self.main, "_execute_online_scout_hit") as single,
            patch.object(self.main, "_scan_level_by_strategy") as scan,
        ):
            with self.assertRaisesRegex(
                self.main.ProbeProtocolError,
                "unknown",
            ):
                self.main._run_red_scout_and_blue_strategy(
                    1,
                    [[0] * 3 for _row in range(3)],
                    [(0, 0), (100, 0), (200, 0), (0, 100), (100, 100), (200, 100),
                     (0, 200), (100, 200), (200, 200)],
                    [3],
                    set(),
                    settings,
                )

        batch.assert_called_once()
        single.assert_not_called()
        scan.assert_not_called()

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

    def test_red_phase_rejects_explored_center_before_transaction(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        explored = (1, 1)
        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                return_value=explored,
            ),
            patch.object(self.main, "_execute_red_scout_transaction") as execute,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            with self.assertRaisesRegex(
                self.main.RedScoutSafetyError,
                "already explored",
            ):
                self.main._run_red_scout_and_blue_strategy(
                    1,
                    [[0] * 3 for _ in range(3)],
                    [(0, 0)] * 9,
                    [3],
                    {explored},
                    settings,
                )

        execute.assert_not_called()

    def test_red_phase_rejects_completed_ship_safety_area(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        completed_ship = {(1, 1), (1, 2)}
        safety_cell = (0, 1)
        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                return_value=safety_cell,
            ),
            patch.object(self.main, "_execute_red_scout_transaction") as execute,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            with self.assertRaisesRegex(
                self.main.RedScoutSafetyError,
                "non-unknown",
            ):
                self.main._run_red_scout_and_blue_strategy(
                    1,
                    [[0] * 3 for _ in range(3)],
                    [(0, 0)] * 9,
                    [2],
                    completed_ship,
                    settings,
                    initial_completed_lengths=(2,),
                    initial_completed_visual_hits=completed_ship,
                )

        execute.assert_not_called()

    def test_red_online_hits_remove_upper_cell_from_rotated_l_before_blue_shots(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        upper_visual = (1, 1)
        lower_ship = frozenset({(1, 2), (2, 2)})
        l_hits = lower_ship | {upper_visual}
        result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=l_hits,
            hit_cells=l_hits,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset(l_hits)),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in l_hits},
        )

        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                return_value=(0, 0),
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
            patch.object(self.main, "_scan_level_by_strategy") as scan,
        ):
            self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [3],
                set(),
                settings,
            )

        self.assertEqual(
            [call.kwargs["cell"] for call in online_hit.call_args_list],
            sorted(lower_ship),
        )
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {upper_visual})
        scan.assert_called_once()

    def test_red_online_hits_remove_corner_from_three_by_three_l_before_blue_shots(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        upper_visual = (0, 0)
        lower_ship = frozenset({(2, 0), (2, 1), (2, 2)})
        l_hits = lower_ship | {upper_visual}
        result = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=l_hits,
            hit_cells=l_hits,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset(l_hits)),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in l_hits},
        )

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(1, 1)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
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
        self.assertEqual(
            [call.kwargs["cell"] for call in online_hit.call_args_list],
            sorted(lower_ship),
        )
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {upper_visual})

    def test_red_online_hits_remove_corner_from_vertical_three_by_three_l(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        upper_visual = (0, 0)
        lower_ship = frozenset({(0, 2), (1, 2), (2, 2)})
        l_hits = lower_ship | {upper_visual}
        result = self.main.RedScoutResult(
            center_cell=(1, 1),
            affected_cells=l_hits,
            hit_cells=l_hits,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset(l_hits)),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in l_hits},
        )

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(1, 1)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
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
        self.assertEqual(
            [call.kwargs["cell"] for call in online_hit.call_args_list],
            sorted(lower_ship),
        )
        self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {upper_visual})

    def test_red_scout_filters_both_flag_l_shapes_before_blue_targets(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)

        for l_hits, upper_flag in (
            (frozenset({(0, 1), (1, 1), (1, 2)}), (0, 1)),
            (frozenset({(0, 2), (1, 1), (1, 2)}), (0, 2)),
            (frozenset({(0, 1), (0, 2), (1, 2)}), (0, 1)),
            (frozenset({(0, 1), (0, 2), (1, 1)}), (0, 2)),
        ):
            result = self.main.RedScoutResult(
                center_cell=(2, 2),
                affected_cells=l_hits,
                hit_cells=l_hits,
                miss_cells=frozenset(),
                unknown_cells=frozenset(),
                footprint=self.main.RedFootprint(frozenset(l_hits)),
                valid=True,
                confidence_by_cell={cell: 0.95 for cell in l_hits},
            )
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
                patch.object(
                    self.main,
                    "_scan_level_by_strategy",
                    return_value=True,
                ) as scan,
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
            self.assertEqual(
                [call.kwargs["cell"] for call in online_hit.call_args_list],
                sorted(l_hits - {upper_flag}),
            )
            self.assertEqual(scan.call_args.kwargs["initial_scout_misses"], {upper_flag})

    def test_red_batch_filters_2x2_l_upper_cell_before_blue_taps(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        upper = (0, 1)
        lower = frozenset({(1, 1), (1, 2)})
        l_hits = lower | {upper}
        red_result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=l_hits,
            hit_cells=l_hits,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(l_hits),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in l_hits},
        )
        batch_result = self.main.OnlineScoutBatchResult(
            results={cell: self.main.ProbeResult.HIT for cell in lower},
            metadata={
                cell: {
                    "batch": True,
                    "stable_state": "hit",
                    "blue_bomb_ready": True,
                    "network_ready": True,
                }
                for cell in lower
            },
        )
        click_points = [
            (400 + (index % 3) * 100, 300 + (index // 3) * 100)
            for index in range(9)
        ]

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(0, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=red_result),
            patch.object(
                self.main,
                "_execute_online_scout_hit_batch",
                return_value=batch_result,
            ) as batch,
            patch.object(self.main, "_execute_online_scout_hit") as single,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
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
        batch.assert_called_once()
        self.assertEqual(
            {target[0] for target in batch.call_args.kwargs["targets"]},
            set(lower),
        )
        single.assert_not_called()
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        self.assertNotIn(upper, scan.call_args.kwargs["initial_scout_hits"])

    def test_red_scout_does_not_join_hits_with_unrelated_completed_visual_ship(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        result_hits = frozenset({(0, 5), (1, 5), (2, 5)})
        result = self.main.RedScoutResult(
            center_cell=(2, 5),
            affected_cells=result_hits,
            hit_cells=result_hits,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(result_hits),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in result_hits},
        )

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(2, 5)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                10,
                [[0] * 10 for _ in range(10)],
                [(0, 0)] * 100,
                [3],
                set(),
                settings,
                initial_completed_visual_hits={(2, 5), (2, 6), (2, 7)},
            )

        self.assertTrue(completed)
        self.assertEqual(
            [call.kwargs["cell"] for call in online_hit.call_args_list],
            [(0, 5)],
        )
        self.assertNotIn((0, 5), scan.call_args.kwargs["initial_misses"])

    def test_l_shape_overrides_committed_hit_protection_from_later_scout(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 2)
        first = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=frozenset({(0, 0)}),
            hit_cells=frozenset({(0, 0)}),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset({(0, 0)})),
            valid=False,
            confidence_by_cell={(0, 0): 0.95},
        )
        second_hits = frozenset({(1, 0), (1, 1)})
        second = self.main.RedScoutResult(
            center_cell=(2, 2),
            affected_cells=second_hits,
            hit_cells=second_hits,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(second_hits),
            valid=False,
            confidence_by_cell={cell: 0.95 for cell in second_hits},
        )

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
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
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
        # The ordinary committed-hit lock remains in force outside an L, but
        # the explicit 2x2 L rule has priority and removes its upper flag cell.
        self.assertNotIn((0, 0), scan.call_args.kwargs["initial_hits"])
        self.assertIn((0, 0), scan.call_args.kwargs["initial_misses"])
        self.assertIn((0, 0), {call.kwargs["cell"] for call in online_hit.call_args_list})

    def test_l_shape_clears_upper_cell_when_initial_hit_is_confirmed(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        upper = (0, 1)
        lower_pair = frozenset({(1, 1), (1, 2)})
        result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=lower_pair,
            hit_cells=lower_pair,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset(lower_pair)),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in lower_pair},
        )

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(0, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [2],
                {upper},
                settings,
                initial_completed_visual_hits={upper, *lower_pair},
            )

        self.assertTrue(completed)
        online_hit.assert_not_called()
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        self.assertEqual(
            scan.call_args.kwargs["initial_completed_visual_hits"],
            set(lower_pair),
        )

    def test_l_shape_moves_completed_upper_cell_to_current_lower_pair(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        upper = (0, 1)
        lower_pair = frozenset({(1, 1), (1, 2)})
        result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=lower_pair,
            hit_cells=lower_pair,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(lower_pair),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in lower_pair},
        )

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(0, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_execute_online_scout_hit_batch") as online_batch,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [2],
                {upper},
                settings,
                initial_completed_visual_hits={upper},
                initial_completed_lengths=(2,),
            )

        self.assertTrue(completed)
        online_hit.assert_not_called()
        online_batch.assert_not_called()
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        self.assertEqual(
            scan.call_args.kwargs["initial_completed_visual_hits"],
            set(lower_pair),
        )

    def test_l_shape_overrides_red_marker_completion_lock(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 0)
        upper = (0, 1)
        lower_pair = frozenset({(1, 1), (1, 2)})

        with patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan:
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [2],
                {upper, *lower_pair},
                settings,
                initial_completed_visual_hits={upper, *lower_pair},
                initial_red_marker_completed_cells={upper, *lower_pair},
                initial_authoritative_completed_visual_hits={upper, *lower_pair},
                initial_authoritative_completed_placements=(
                    self.main.Placement(length=2, direction="V", cells=((0, 1), (1, 1))),
                ),
                initial_completed_lengths=(2,),
            )

        self.assertTrue(completed)
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])
        self.assertNotIn(upper, scan.call_args.kwargs["initial_red_marker_completed_cells"])
        self.assertTrue(
            all(upper not in placement.cells
                for placement in scan.call_args.kwargs["initial_authoritative_completed_placements"])
        )

    def test_red_marker_projection_does_not_delete_unconfirmed_upper_cell(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 0)
        upper = (0, 1)
        lower_pair = frozenset({(1, 1), (1, 2)})

        with patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan:
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [2],
                set(lower_pair),
                settings,
                initial_completed_visual_hits={upper, *lower_pair},
                initial_red_marker_completed_cells={upper, *lower_pair},
                initial_authoritative_completed_visual_hits={upper, *lower_pair},
                initial_completed_lengths=(2,),
            )

        self.assertTrue(completed)
        self.assertIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertNotIn(upper, scan.call_args.kwargs["initial_misses"])

    def test_red_scout_completed_geometry_is_locked_without_duplicate_blue_shot(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        ship = ((0, 1), (0, 2), (0, 3), (0, 4))
        result = self.main.RedScoutResult(
            center_cell=(2, 2),
            affected_cells=frozenset(ship),
            hit_cells=frozenset(ship),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset(ship)),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in ship},
            diagnostics={
                "completed_ship_failure": None,
                "completed_lengths": (4,),
                "resolved_ship_placements": (ship,),
            },
        )

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(2, 2)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(self.main, "_execute_online_scout_hit") as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                9,
                [[0] * 10 for _ in range(10)],
                [(0, 0)] * 100,
                [4],
                set(ship),
                settings,
            )

        self.assertTrue(completed)
        online_hit.assert_not_called()
        placements = scan.call_args.kwargs["initial_authoritative_completed_placements"]
        self.assertEqual({placement.cells for placement in placements}, {ship})

    def test_completed_ship_representative_prefers_cell_without_existing_wreck(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        existing_wreck = (1, 1)
        missing_wreck = (1, 2)
        ship = (existing_wreck, missing_wreck)
        misses = frozenset({(0, 0), (0, 1), (0, 2), (1, 0)})
        affected = frozenset({*ship, *misses})
        result = self.main.RedScoutResult(
            center_cell=(2, 2),
            affected_cells=affected,
            hit_cells=frozenset(ship),
            miss_cells=misses,
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(affected),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in affected},
            diagnostics={
                "completed_ship_failure": None,
                "completed_lengths": (2,),
                "resolved_ship_placements": (ship,),
            },
        )
        click_points = [
            (400 + (index % 3) * 100, 300 + (index // 3) * 100)
            for index in range(9)
        ]

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(2, 2)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                click_points,
                [2],
                {existing_wreck},
                settings,
            )

        self.assertTrue(completed)
        online_hit.assert_called_once()
        self.assertEqual(online_hit.call_args.kwargs["cell"], missing_wreck)

    def test_red_marker_completed_ship_does_not_retry_unopened_cell_after_blue_miss(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        unopened = (9, 1)
        existing_wrecks = {(9, 2), (9, 3), (9, 4)}
        ship = (unopened, *sorted(existing_wrecks))
        misses = frozenset({(5, 6), (7, 7)})
        result = self.main.RedScoutResult(
            center_cell=(5, 6),
            affected_cells=frozenset({*ship, *misses}),
            hit_cells=frozenset(ship),
            miss_cells=misses,
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset({*ship, *misses})),
            valid=True,
            confidence_by_cell={
                cell: 0.95 for cell in {*ship, *misses}
            },
            diagnostics={
                "completed_ship_failure": None,
                "completed_lengths": (4,),
                "completed_body_candidates": ((9, 2), (9, 3)),
                "resolved_ship_placements": (ship,),
            },
        )
        click_points = [
            (300 + (index % 10) * 40, 100 + (index // 10) * 30)
            for index in range(100)
        ]

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(5, 6)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.MISS,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                9,
                [[0] * 10 for _row in range(10)],
                click_points,
                [4],
                set(existing_wrecks),
                settings,
            )

        self.assertTrue(completed)
        self.assertEqual(online_hit.call_count, 1)
        self.assertEqual(online_hit.call_args.kwargs["cell"], unopened)
        # A pending red-completed cell must bypass the shared/fast path so
        # neighboring target animations cannot be attributed to it.
        self.assertFalse(online_hit.call_args.kwargs["fast_batch"])
        self.assertEqual(scan.call_args.kwargs["initial_hits"], set(existing_wrecks))
        self.assertEqual(scan.call_args.kwargs["initial_scout_hits"], set())

    def test_completed_ship_safety_area_clears_false_hit_and_is_never_targeted(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        existing_wreck = (2, 0)
        unopened_ship_cell = (2, 1)
        false_perimeter_hit = (1, 2)
        ship = (existing_wreck, unopened_ship_cell)
        misses = frozenset({(0, 0), (0, 1), (0, 2)})
        affected = frozenset({*ship, false_perimeter_hit, *misses})
        result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=affected,
            hit_cells=frozenset({*ship, false_perimeter_hit}),
            miss_cells=misses,
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(affected),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in affected},
            diagnostics={
                "completed_ship_failure": None,
                "completed_lengths": (2,),
                "resolved_ship_placements": (ship,),
            },
        )
        click_points = [
            (400 + (index % 3) * 100, 250 + (index // 3) * 80)
            for index in range(9)
        ]

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(0, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _row in range(3)],
                click_points,
                [2],
                {existing_wreck, false_perimeter_hit},
                settings,
            )

        self.assertTrue(completed)
        self.assertEqual(
            [call.kwargs["cell"] for call in online_hit.call_args_list],
            [unopened_ship_cell],
        )
        expected_safety = {
            (1, 0), (1, 1), (1, 2), (2, 2),
        }
        self.assertEqual(scan.call_args.kwargs["initial_hits"], set(ship))
        self.assertTrue(
            expected_safety <= scan.call_args.kwargs["initial_misses"]
        )
        self.assertNotIn(
            false_perimeter_hit,
            scan.call_args.kwargs["initial_scout_hits"],
        )

    def test_red_scout_clicks_every_pending_scout_hit_in_completed_ship_result(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        ordinary_hit = (0, 0)
        pending_ship_hit = (2, 1)
        committed_ship_hit = (2, 2)
        ship = (pending_ship_hit, committed_ship_hit)
        misses = frozenset({(0, 1), (0, 2), (1, 2)})
        affected = frozenset({ordinary_hit, *ship, *misses})
        result = self.main.RedScoutResult(
            center_cell=(1, 0),
            affected_cells=affected,
            hit_cells=frozenset({ordinary_hit, *ship}),
            miss_cells=misses,
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(affected),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in affected},
            diagnostics={
                "completed_ship_failure": None,
                "completed_lengths": (2,),
                "resolved_ship_placements": (ship,),
            },
        )
        click_points = [
            (400 + (index % 3) * 100, 300 + (index // 3) * 100)
            for index in range(9)
        ]

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(1, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True),
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                click_points,
                [1, 2],
                {committed_ship_hit},
                settings,
            )

        self.assertTrue(completed)
        self.assertEqual(
            [call.kwargs["cell"] for call in online_hit.call_args_list],
            [ordinary_hit, pending_ship_hit],
        )

    def test_l_shape_clears_initial_upper_hit_before_lower_pair_batch(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        upper = (0, 1)
        lower_pair = frozenset({(1, 1), (1, 2)})
        result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=lower_pair,
            hit_cells=lower_pair,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(lower_pair),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in lower_pair},
        )
        batch_result = self.main.OnlineScoutBatchResult(
            results={cell: self.main.ProbeResult.HIT for cell in lower_pair},
            metadata={cell: {"batch": True, "stable_state": "hit"} for cell in lower_pair},
        )
        click_points = [
            (400 + (index % 3) * 100, 300 + (index // 3) * 100)
            for index in range(9)
        ]

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(0, 0)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=result),
            patch.object(
                self.main,
                "_execute_online_scout_hit_batch",
                return_value=batch_result,
            ) as online_batch,
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                return_value=self.main.ProbeResult.HIT,
            ) as online_hit,
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                click_points,
                [2],
                {upper},
                settings,
            )

        self.assertTrue(completed)
        online_batch.assert_called_once()
        self.assertEqual(
            {target[0] for target in online_batch.call_args.kwargs["targets"]},
            set(lower_pair),
        )
        online_hit.assert_not_called()
        self.assertNotIn(upper, scan.call_args.kwargs["initial_hits"])
        self.assertIn(upper, scan.call_args.kwargs["initial_misses"])

    def test_completed_ship_geometry_corrects_false_hit_in_l_shape(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        l_hits = frozenset({(1, 1), (1, 2), (2, 2)})
        visually_misplaced_ship = {(1, 1), (1, 2)}
        actual_ship = {(1, 2), (2, 2)}
        false_hit = (1, 1)
        result = self.main.RedScoutResult(
            center_cell=(0, 0),
            affected_cells=l_hits,
            hit_cells=l_hits,
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset(l_hits)),
            valid=True,
            confidence_by_cell={cell: 0.95 for cell in l_hits},
        )

        def online_hit(**kwargs):
            evidence = {
                (1, 1): (1, 7, "miss"),
                (1, 2): (3, 3, "hit"),
                (2, 2): (3, 4, "hit"),
            }
            hit_votes, frame_count, stable_state = evidence[kwargs["cell"]]
            kwargs["probe_metadata"].update(
                hit_votes=hit_votes,
                frame_count=frame_count,
                stable_state=stable_state,
            )
            if kwargs["cell"] == (1, 2):
                kwargs["probe_metadata"].update(
                    sidebar_completed_lengths=(2,),
                    sidebar_completion_screenshot=np.zeros(
                        (720, 1280, 3),
                        dtype=np.uint8,
                    ),
                )
            return self.main.ProbeResult.HIT

        with (
            patch.object(
                self.main.RedScoutPlanner,
                "choose_center",
                return_value=(0, 0),
            ),
            patch.object(
                self.main,
                "_execute_red_scout_transaction",
                return_value=result,
            ),
            patch.object(
                self.main,
                "_execute_online_scout_hit",
                side_effect=online_hit,
            ) as online,
            patch.object(
                self.main,
                "_trusted_completed_cells_from_probe_metadata",
                return_value=visually_misplaced_ship,
            ),
            patch.object(
                self.main,
                "_scan_level_by_strategy",
                return_value=True,
            ) as scan,
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                1,
                [[0] * 3 for _ in range(3)],
                [(0, 0)] * 9,
                [2, 3],
                set(),
                settings,
            )

        self.assertTrue(completed)
        blue_cells = {call.kwargs["cell"] for call in online.call_args_list}
        self.assertNotIn(false_hit, blue_cells)
        self.assertTrue(blue_cells <= actual_ship)
        scan.assert_called_once()
        self.assertEqual(scan.call_args.kwargs["initial_hits"], actual_ship)
        self.assertEqual(scan.call_args.kwargs["initial_misses"], {false_hit})
        self.assertEqual(
            scan.call_args.kwargs["initial_completed_visual_hits"],
            actual_ship,
        )

    def test_down_right_l_shape_always_discards_the_upper_flag_cell(self):
        l_hits = ((0, 5), (1, 5), (1, 6))
        misleading_evidence = {
            (0, 5): {
                "stable_state": "hit",
                "hit_votes": 7,
                "frame_count": 7,
            },
            (1, 5): {
                "stable_state": "hit",
                "hit_votes": 3,
                "frame_count": 4,
            },
            (1, 6): {
                "stable_state": "unknown",
                "hit_votes": 2,
                "frame_count": 4,
            },
        }

        false_cell = self.main._resolve_false_hit_in_l_shape(
            l_hits,
            misleading_evidence,
        )

        self.assertEqual(false_cell, (0, 5))

    def test_rotated_l_shape_discards_upper_unaligned_cell(self):
        for l_hits, false_cell in (
            (((5, 1), (5, 2), (6, 2)), (5, 1)),
            (((5, 1), (5, 2), (6, 1)), (5, 2)),
        ):
            self.assertEqual(
                self.main._resolve_false_hit_in_l_shape(l_hits, {}),
                false_cell,
            )

    def test_three_by_three_l_detection_ignores_unrelated_hits(self):
        match = self.main._find_flag_overlap_l_shape(
            {
                (0, 0),
                (2, 0),
                (2, 1),
                (2, 2),
                (6, 6),
            }
        )

        self.assertEqual(match, ((0, 0), frozenset({(2, 0), (2, 1), (2, 2)})))

    def test_complete_ship_cells_are_not_cleared_by_nearby_l_shape(self):
        cells = {
            (6, 7),
            (6, 8),
            (6, 9),
            (8, 7),
            (8, 8),
            (8, 9),
        }
        resolution = self.main.resolve_completed_ship_cells(
            cells,
            (3, 3),
            grid_size=10,
        )
        self.assertEqual(resolution.cells, frozenset(cells))

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

    def test_handle_level_skips_surface_baseline_without_submarine_config(self):
        expected_points = [(row, col) for row in range(3) for col in range(3)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        with (
            patch.object(self.main.adb, "delay"),
            patch.object(self.main.adb, "read_screenshot", return_value=grid_img),
            patch.object(self.main, "get_click_points", return_value=(expected_points, None)),
            patch.object(self.main, "get_configured_submarines", return_value=None),
            patch.object(self.main, "_capture_surface_water_baseline") as capture_baseline,
            patch.object(self.main, "_scan_level_by_grid_order", return_value=0) as scan,
        ):
            _grid_img_result, _quad, completed = self.main.handle_game_level(1, [[0] * 3 for _ in range(3)])

        self.assertFalse(completed)
        capture_baseline.assert_not_called()
        scan.assert_called_once()

    def test_handle_level_passes_surface_baseline_to_strategy(self):
        expected_points = [(row, col) for row in range(3) for col in range(3)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        baseline = self.main.SurfaceWaterBaseline(
            median_gray=np.zeros((720, 1280), dtype=np.float32),
            temporal_mad=np.zeros((720, 1280), dtype=np.float32),
            frame_count=3,
        )
        with (
            patch.object(self.main.adb, "delay"),
            patch.object(self.main.adb, "read_screenshot", return_value=grid_img),
            patch.object(self.main, "get_click_points", return_value=(expected_points, None)),
            patch.object(self.main, "get_configured_submarines", return_value=[3]),
            patch.object(self.main, "detect_sidebar_progress", return_value=None),
            patch.object(self.main, "detect_visible_wreck_cells", return_value=set()),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=set()),
            patch.object(self.main, "_capture_surface_water_baseline", return_value=baseline),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", return_value=True) as run,
        ):
            self.main.handle_game_level(1, [[0] * 3 for _ in range(3)])

        self.assertIs(run.call_args.kwargs["surface_baseline"], baseline)

    def test_startup_vision_diagnostics_save_coordinates_and_evidence(self):
        image = np.zeros((180, 180, 3), dtype=np.uint8)
        points = [(30 + col * 40, 30 + row * 40) for row in range(3) for col in range(3)]
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(self.main, "STARTUP_VISION_DIR", self.main.Path(temp_dir)):
                evidence = self.main._save_startup_vision_diagnostics(
                    1,
                    image,
                    points,
                    3,
                    wreck_candidates={(0, 1)},
                    submarine_cells={(1, 1)},
                    red_anchors={(1, 1)},
                    partial_cells={(0, 1)},
                    visible_cells=set(),
                    surface_baseline=None,
                )
            sample_dirs = list(self.main.Path(temp_dir).glob("level_1_*"))
            self.assertEqual(len(sample_dirs), 1)
            self.assertTrue((sample_dirs[0] / "board_overlay.png").exists())
            self.assertTrue((sample_dirs[0] / "cell_r1_c1.png").exists())
            payload = json.loads((sample_dirs[0] / "evidence.json").read_text(encoding="utf-8"))

        self.assertEqual(evidence[(1, 1)]["state"], "submarine")
        self.assertEqual(evidence[(0, 1)]["state"], "wreck_candidate")
        self.assertEqual(payload["cells"]["1,1"]["source"], ["completed_submarine", "red_submarine_anchor"])

    def test_visible_wreck_precheck_forwards_surface_baseline_context(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        baseline = self.main.SurfaceWaterBaseline(
            median_gray=np.zeros((720, 1280), dtype=np.float32),
            temporal_mad=np.zeros((720, 1280), dtype=np.float32),
            frame_count=3,
        )
        with (
            patch.object(self.main, "visible_wreck_static_detected", return_value=False) as detector,
            patch.object(self.main, "red_submarine_marker_visible", return_value=False),
        ):
            visible = self.main._visible_wreck_for_hit_state(
                image,
                (400, 300),
                cell=(1, 2),
                surface_baseline=baseline,
                relative_position=(0.5, 1.0),
            )

        self.assertFalse(visible)
        self.assertIs(detector.call_args.kwargs["surface_baseline"], baseline)
        self.assertEqual(detector.call_args.kwargs["relative_position"], (0.5, 1.0))

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
        hit_map = [[0] * 9 for _ in range(9)]

        def run_strategy(*_args, **kwargs):
            self.assertEqual(kwargs["initial_hits"], visible_hits)
            self.assertEqual(kwargs["initial_visual_candidates"], set())
            self.assertEqual(kwargs["initial_completed_visual_hits"], {(7, 1), (7, 2), (7, 3)})
            # A valid sidebar upgrades compact static-wreck detections to
            # ordinary hits; partial/template-only detections remain pending.
            self.assertEqual(kwargs["initial_visual_hit_count"], len(visible_hits))
            self.assertEqual(
                {
                    (row, col)
                    for row, values in enumerate(hit_map)
                    for col, value in enumerate(values)
                    if value
                },
                visible_hits,
            )
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
                hit_map,
            )

        self.assertTrue(completed)
        self.assertIs(grid_img_result, grid_img)
        run.assert_called_once()

    def test_blue_only_startup_matches_red_scout_state_mapping(self):
        submarines = [2, 2, 3, 4, 4, 5]
        visible_wrecks = {(2, 2), (7, 7)}
        completed_ship = {(4, 1), (4, 2), (4, 3), (4, 4)}
        sidebar_progress = SidebarProgress(
            active_lengths=(2, 2, 3, 4, 5),
            completed_lengths=(4,),
            unknown_lengths=(),
        )
        click_points = [(400 + index, 300 + index) for index in range(100)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        hit_map = [[0] * 10 for _ in range(10)]

        def run_strategy(*_args, **kwargs):
            self.assertEqual(kwargs["initial_hits"], visible_wrecks | completed_ship)
            self.assertEqual(
                kwargs["initial_visual_candidates"],
                set(),
            )
            self.assertEqual(kwargs["initial_completed_visual_hits"], completed_ship)
            self.assertEqual(kwargs["initial_authoritative_completed_visual_hits"], completed_ship)
            self.assertEqual(
                {placement.cells for placement in kwargs["initial_authoritative_completed_placements"]},
                {tuple(sorted(completed_ship))},
            )
            self.assertEqual(kwargs["initial_completed_blocking_placements"], ())
            self.assertEqual(kwargs["initial_completed_lengths"], (4,))
            self.assertEqual(kwargs["initial_visual_hit_count"], len(visible_wrecks | completed_ship))
            self.assertEqual(
                {
                    (row, col)
                    for row, values in enumerate(hit_map)
                    for col, value in enumerate(values)
                    if value
                },
                visible_wrecks | completed_ship,
            )
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
            patch.object(self.main, "detect_visible_wreck_cells", return_value=visible_wrecks),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=set()),
            patch.object(
                self.main,
                "detect_completed_submarine_candidate_cells",
                return_value=completed_ship,
            ),
            patch.object(self.main, "detect_red_submarine_marker_cells", return_value=set()),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", side_effect=run_strategy) as run,
        ):
            _grid_img_result, _quad, completed = self.main.handle_game_level(
                9,
                hit_map,
                settings=self.main.RedScoutSettings(self.main.ProbeMode.BLUE_ONLY, 2),
            )

        self.assertTrue(completed)
        run.assert_called_once()

    def test_blue_only_without_sidebar_matches_red_scout_candidates(self):
        submarines = [2, 2, 3, 4, 4, 5]
        visible_wrecks = {(2, 2), (7, 7)}
        click_points = [(400 + index, 300 + index) for index in range(100)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        hit_map = [[0] * 10 for _ in range(10)]

        def run_strategy(*_args, **kwargs):
            self.assertEqual(kwargs["initial_hits"], set())
            self.assertEqual(kwargs["initial_visual_candidates"], visible_wrecks)
            self.assertEqual(kwargs["initial_visual_hit_count"], 0)
            self.assertEqual(
                {
                    (row, col)
                    for row, values in enumerate(hit_map)
                    for col, value in enumerate(values)
                    if value
                },
                set(),
            )
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
            patch.object(self.main, "detect_sidebar_progress", return_value=None),
            patch.object(self.main, "detect_visible_wreck_cells", return_value=visible_wrecks),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=set()),
            patch.object(self.main, "detect_completed_submarine_candidate_cells", return_value=set()),
            patch.object(self.main, "detect_red_submarine_marker_cells", return_value=set()),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", side_effect=run_strategy) as run,
        ):
            _grid_img_result, _quad, completed = self.main.handle_game_level(
                9,
                hit_map,
                settings=self.main.RedScoutSettings(self.main.ProbeMode.BLUE_ONLY, 2),
            )

        self.assertTrue(completed)
        run.assert_called_once()

    def test_handle_game_level_uses_red_marker_as_completion_when_sidebar_is_unavailable(self):
        submarines = [2, 2, 3, 4, 4, 5]
        completed_ship = {(4, 1), (4, 2), (4, 3), (4, 4)}
        sidebar_unavailable = None
        click_points = [(400 + index, 300 + index) for index in range(100)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        hit_map = [[0] * 10 for _ in range(10)]

        def run_strategy(*_args, **kwargs):
            self.assertEqual(kwargs["initial_hits"], completed_ship)
            # (0, 0) is title-obscured on 10x10 boards and must remain
            # unknown until a blue probe confirms it.
            self.assertEqual(kwargs["initial_visual_candidates"], set())
            self.assertEqual(kwargs["initial_completed_visual_hits"], completed_ship)
            self.assertEqual(
                kwargs["initial_authoritative_completed_visual_hits"],
                completed_ship,
            )
            self.assertEqual(kwargs["initial_completed_lengths"], (4,))
            self.assertEqual(
                {placement.cells for placement in kwargs["initial_authoritative_completed_placements"]},
                {tuple(sorted(completed_ship))},
            )
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
            patch.object(self.main, "detect_sidebar_progress", return_value=sidebar_unavailable),
            patch.object(self.main, "detect_visible_wreck_cells", return_value={(0, 0)}),
            patch.object(self.main, "detect_partial_wreck_cells", return_value={(0, 1)}),
            patch.object(
                self.main,
                "detect_completed_submarine_candidate_cells",
                return_value=completed_ship | {(1, 1)},
            ),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", side_effect=run_strategy) as run,
        ):
            _grid_img_result, _quad, completed = self.main.handle_game_level(
                9,
                hit_map,
            )

        self.assertTrue(completed)
        run.assert_called_once()

    def test_handle_game_level_does_not_promote_contiguous_wrecks_without_completion_evidence(self):
        submarines = [2, 2, 3, 4, 4, 5]
        contiguous_hits = {(6, 2), (6, 3), (6, 4), (6, 5)}
        click_points = [(400 + index, 300 + index) for index in range(100)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        hit_map = [[0] * 10 for _ in range(10)]

        def run_strategy(*_args, **kwargs):
            self.assertEqual(kwargs["initial_hits"], set())
            self.assertEqual(kwargs["initial_visual_candidates"], contiguous_hits)
            self.assertEqual(kwargs["initial_completed_visual_hits"], set())
            self.assertEqual(kwargs["initial_authoritative_completed_visual_hits"], set())
            self.assertEqual(kwargs["initial_completed_lengths"], ())
            self.assertEqual(kwargs["initial_visual_hit_count"], 0)
            self.assertTrue(all(value == 0 for row in hit_map for value in row))
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
            patch.object(self.main, "detect_sidebar_progress", return_value=None),
            patch.object(self.main, "detect_visible_wreck_cells", return_value=contiguous_hits),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=set()),
            patch.object(self.main, "detect_completed_submarine_candidate_cells", return_value=set()),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", side_effect=run_strategy) as run,
        ):
            _grid_img_result, _quad, completed = self.main.handle_game_level(
                10,
                hit_map,
            )

        self.assertTrue(completed)
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
            self.assertEqual(kwargs["initial_visual_candidates"], set())
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

    def test_handle_game_level_keeps_agreeing_wrecks_when_visual_count_is_suspicious(self):
        submarines = [2, 3, 3, 4, 4, 5]
        target_wrecks = {(7, 6), (7, 8)}
        visible_hits = {
            (row, col)
            for row in range(10)
            for col in range(10)
            if (row, col) not in {(0, 0), (0, 1), (0, 2), (0, 3)}
        }
        visible_hits = set(sorted(visible_hits)[:22]) | target_wrecks
        partial_wrecks = set(target_wrecks)
        sidebar_progress = SidebarProgress(
            active_lengths=tuple(submarines),
            completed_lengths=(),
            unknown_lengths=(),
        )
        click_points = [(400 + index, 300 + index) for index in range(100)]
        grid_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        hit_map = [[0] * 10 for _ in range(10)]

        def run_strategy(*_args, **kwargs):
            self.assertIn((7, 6), kwargs["initial_hits"])
            self.assertIn((7, 8), kwargs["initial_hits"])
            self.assertNotIn((7, 6), kwargs["initial_visual_candidates"])
            self.assertNotIn((7, 8), kwargs["initial_visual_candidates"])
            self.assertEqual(hit_map[7][6], 1)
            self.assertEqual(hit_map[7][8], 1)
            return True

        with (
            patch.object(self.main.adb, "delay"),
            patch.object(self.main.adb, "read_screenshot", return_value=grid_img),
            patch.object(self.main, "get_click_points", return_value=(click_points, np.zeros((4, 2), dtype=np.float32))),
            patch.object(self.main, "get_configured_submarines", return_value=submarines),
            patch.object(self.main, "detect_sidebar_progress", return_value=sidebar_progress),
            patch.object(self.main, "detect_visible_wreck_cells", return_value=visible_hits),
            patch.object(self.main, "detect_partial_wreck_cells", return_value=partial_wrecks),
            patch.object(self.main, "detect_completed_submarine_candidate_cells", return_value=set()),
            patch.object(
                self.main,
                "resolve_completed_ship_cells",
                return_value=CompletedShipResolution(
                    cells=frozenset(),
                    placements=(),
                    unresolved_lengths=(),
                    discarded_cells=frozenset(),
                ),
            ),
            patch.object(self.main, "_run_red_scout_and_blue_strategy", side_effect=run_strategy),
        ):
            _grid_img_result, _quad, completed = self.main.handle_game_level(9, hit_map)

        self.assertTrue(completed)

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
                initial_sidebar_progress=SidebarProgress(completed_lengths=(3,)),
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
            patch.object(self.main, "_exit_activity_after_probe_click") as exit_activity,
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

    def test_red_result_frame_victory_tolerates_ammo_fingerprint_change(self):
        victory_frame = np.zeros((40, 40, 3), dtype=np.uint8)

        def discard(transaction, **_kwargs):
            transaction.advance(self.main.ProbePhase.REQUEST_DISCARDED)
            transaction.red_request_discarded = True
            transaction.advance(self.main.ProbePhase.LOGIN_RECOVERING)
            transaction.advance(self.main.ProbePhase.COMPLETE)
            return False

        with (
            patch.object(
                self.main,
                "_capture_red_ammo_state",
                side_effect=[
                    ("before", "before-fingerprint", DummyMatch((10, 20))),
                    ("after", "after-fingerprint", DummyMatch((10, 20))),
                ],
            ),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=False),
            patch.object(
                self.main,
                "_capture_red_result_frames",
                return_value=[victory_frame],
            ),
            patch.object(
                self.main,
                "find_victory_banner",
                side_effect=lambda frame: DummyMatch((1, 1))
                if frame is victory_frame
                else None,
            ),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard,
            ),
            patch.object(self.main, "ammo_fingerprint_matches", return_value=False),
            patch.object(self.main, "write_pending_probe"),
            patch.object(self.main, "update_pending_probe"),
            patch.object(self.main, "clear_pending_probe"),
            patch.object(self.main, "_stop_and_latch_red_safety_failure") as stop,
        ):
            result = self.main._execute_red_scout_transaction(
                level=1,
                center_cell=(1, 1),
                point=(100, 200),
                index=0,
                grid_size=3,
                all_click_points=[(0, 0)] * 9,
                submarine_lengths=[3],
            )

        self.assertEqual(result.invalid_reason, "local_victory_screen")
        stop.assert_not_called()

    def test_red_result_frame_non_victory_keeps_strict_ammo_validation(self):
        analysis = self._valid_red_result()

        def discard(transaction, **_kwargs):
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
            patch.object(self.main, "_analyze_red_result_with_baseline_consensus", return_value=analysis),
            patch.object(
                self.main,
                "_discard_pending_request_and_prepare_next_probe",
                side_effect=discard,
            ),
            patch.object(self.main, "_verify_red_ammo_unchanged") as verify,
            patch.object(self.main, "write_pending_probe"),
            patch.object(self.main, "update_pending_probe"),
            patch.object(self.main, "clear_pending_probe"),
        ):
            self.main._execute_red_scout_transaction(
                level=1,
                center_cell=(1, 1),
                point=(100, 200),
                index=0,
                grid_size=3,
                all_click_points=[(0, 0)] * 9,
                submarine_lengths=[3],
            )

        verify.assert_called_once_with("fingerprint", sample_dir=None)

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
            patch.object(self.main, "_capture_red_ammo_state", return_value=("before", "fp", DummyMatch((10, 20)))),
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
            patch.object(self.main, "_capture_red_ammo_state", return_value=("before", "fp", DummyMatch((10, 20)))),
            patch.object(self.main, "_select_red_bomb", return_value=True),
            patch.object(self.main, "_exit_activity_after_probe_click"),
            patch.object(self.main, "_reenter_activity_for_probe_result", return_value=True),
            patch.object(self.main, "_discard_pending_request_and_prepare_next_probe", side_effect=discard),
            patch.object(self.main, "ammo_fingerprint_matches", return_value=True),
            patch.object(self.main, "write_pending_probe"),
            patch.object(self.main, "update_pending_probe"),
            patch.object(self.main, "clear_pending_probe") as clear_pending,
            patch.object(self.main, "_verify_red_ammo_unchanged") as verify_ammo,
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
        verify_ammo.assert_not_called()
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

    def test_red_result_capture_uses_red_frame_schedule(self):
        frames = [object() for _ in self.main.RED_SCOUT_RESULT_FRAME_DELAYS]
        captured_paths = []

        def read_screenshot(path):
            captured_paths.append(path)
            return frames[len(captured_paths) - 1]

        self.adb.read_screenshot = Mock(side_effect=read_screenshot)

        result = self.main._capture_red_result_frames()

        self.assertEqual(result, frames)
        self.assertEqual(
            [call for call in self.adb.calls if call[0] == "delay"],
            [("delay", delay) for delay in self.main.RED_SCOUT_RESULT_FRAME_DELAYS],
        )
        self.assertEqual(
            captured_paths,
            [
                self.main.RUN_DEBUG_DIR / f"red_result_{index}.png"
                for index in range(len(self.main.RED_SCOUT_RESULT_FRAME_DELAYS))
            ],
        )

    def test_red_result_capture_writes_each_attempt_to_its_sample_directory(self):
        frames = [object() for _ in self.main.RED_SCOUT_RESULT_FRAME_DELAYS]
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
                for index in range(len(self.main.RED_SCOUT_RESULT_FRAME_DELAYS))
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
                "_wait_until_activity_detail_closed",
                return_value=True,
            ),
            patch.object(
                self.main,
                "wait_until_occur",
                side_effect=lambda *args, **kwargs: next(waits),
            ),
        ):
            self.main.enter_activity(re_enter=True, max_retries=1)

        self.assertIn(("click", 1249, 269), self.adb.calls)
        self.assertIn(("back",), self.adb.calls)
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

    def test_re_enter_skips_detail_timeout_when_victory_is_immediate(self):
        initial_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
        victory_screen = np.ones((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(side_effect=[initial_screen, victory_screen])
        activity_button = DummyMatch((1249, 269))
        victory_match = DummyMatch((640, 280))

        with (
            patch.object(self.main, "wait_until_occur", side_effect=[activity_button, None]) as wait,
            patch.object(self.main, "find_template", return_value=None),
            patch.object(
                self.main,
                "find_victory_banner",
                return_value=victory_match,
            ),
        ):
            completed = self.main.enter_activity(re_enter=True, max_retries=1)

        self.assertTrue(completed)
        self.assertEqual(wait.call_count, 1)
        self.assertIn(("click", *self.main.ACTIVITY_DETAIL_POINT), self.adb.calls)

    def test_completed_ship_orientation_prefers_real_hit_line_over_visual_candidates(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        click_points = [(0, 0)] * 100
        candidates = {
            *((1, col) for col in range(1, 6)),
            *((row, 1) for row in range(1, 6)),
        }
        metadata = {
            "sidebar_completion_screenshot": screenshot,
            "sidebar_completed_lengths": (5,),
        }

        with patch.object(
            self.main,
            "detect_completed_submarine_candidate_cells",
            return_value=candidates,
        ):
            trusted = self.main._trusted_completed_cells_from_probe_metadata(
                metadata,
                click_points,
                grid_size=10,
                anchor=(4, 1),
                preferred_cells={(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)},
            )

        self.assertEqual(
            trusted,
            {(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)},
        )

    def test_wait_until_occur_prioritizes_victory_alternate_over_detail(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        victory = SimpleNamespace(template_path=self.main.WIN_TEMPLATE)
        self.adb.read_screenshot = Mock(return_value=screenshot)

        with (
            patch.object(self.main, "find_template", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "monotonic", side_effect=[0.0, 0.1]),
        ):
            result = self.main.wait_until_occur(
                self.main.QUIT_ACTIVITY_TEMPLATE,
                timeout=1.0,
                alternate_matchers=(("victory", lambda _screen: victory),),
            )

        self.assertIs(result, victory)

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

        self.assertEqual(result, self.main.ProbeResult.LEVEL_COMPLETE)
        self.assertEqual(hit_map, [[0, 0], [0, 0]])
        discard_request.assert_not_called()
        commit_request.assert_not_called()

    def test_pending_victory_click_keeps_network_isolated(self):
        screenshot = np.zeros((20, 20, 3), dtype=np.uint8)

        with patch.object(
            self.main,
            "find_victory_banner",
            return_value=DummyMatch((10, 10)),
        ), patch.object(self.main, "_confirm_victory_banner_cleared", return_value=True):
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

    def test_victory_prompt_deduplicates_repeated_old_frame(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        match = DummyMatch((10, 10))

        with (
            patch.object(self.main, "find_victory_banner", return_value=match),
            patch.object(self.main, "_confirm_victory_banner_cleared", return_value=True),
        ):
            self.assertTrue(
                self.main.handle_victory_prompt(
                    timeout=0.0,
                    screenshot=screenshot,
                    restore_network=False,
                )
            )
            self.assertFalse(
                self.main.handle_victory_prompt(
                    timeout=0.0,
                    screenshot=screenshot,
                    restore_network=False,
                )
            )

        self.assertEqual(
            self.adb.calls.count(("click", *self.main.SCREEN_CONTINUE_POINT)),
            1,
        )

    def test_victory_prompt_returns_false_when_banner_clear_is_unconfirmed(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        match = DummyMatch((10, 10))

        with (
            patch.object(self.main, "find_victory_banner", return_value=match),
            patch.object(self.main, "_confirm_victory_banner_cleared", return_value=False),
        ):
            handled = self.main.handle_victory_prompt(
                timeout=0.0,
                screenshot=screenshot,
                restore_network=False,
            )

        self.assertFalse(handled)
        self.assertEqual(
            self.adb.calls.count(("click", *self.main.SCREEN_CONTINUE_POINT)),
            1,
        )

    def test_red_victory_gate_clears_banner_before_blue_attack(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        fresh_screen = np.ones((720, 1280, 3), dtype=np.uint8)
        victory = DummyMatch((640, 360))

        self.adb.read_screenshot = Mock(side_effect=[screenshot, fresh_screen])
        with (
            patch.object(self.main, "find_victory_banner", side_effect=[victory, None]),
            patch.object(self.main, "_victory_prompt_guard_matches", return_value=False),
            patch.object(self.main, "find_connection_interrupted_dialog", return_value=None),
            patch.object(self.main, "find_template", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "wait_until_connection_interrupted_dialog", return_value=DummyMatch((1, 1))),
            patch.object(self.main, "wait_until_retry_button", return_value=DummyMatch((2, 2))),
            patch.object(self.main, "enter_activity", return_value=False) as enter,
        ):
            self.main._clear_red_victory_before_blue_attack()

        enter.assert_called_once_with(
            re_enter=True,
            max_retries=1,
            prepare_activity_list=True,
            activity_button_timeout=self.main.POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        )
        self.assertIn(("click", 2, 2), self.adb.calls)
        self.assertIn(
            ("enable_weak_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )
        self.assertIn(
            ("disable_weak_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )

    def test_red_victory_gate_fails_closed_when_banner_will_not_clear(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)

        with (
            patch.object(self.main, "find_victory_banner", return_value=DummyMatch((640, 360))),
            patch.object(self.main, "_victory_prompt_guard_matches", return_value=False),
            patch.object(self.main, "wait_until_connection_interrupted_dialog", return_value=None),
            patch.object(self.main, "latch_network_fail_closed"),
        ):
            with self.assertRaises(self.main.ProbeProtocolError):
                self.main._clear_red_victory_before_blue_attack()

        self.assertIn(
            ("enable_weak_network", self.main.GAME_PACKAGE_NAME),
            self.adb.calls,
        )

    def test_red_victory_gate_rejects_next_level_before_blue_attack(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.read_screenshot = Mock(return_value=screenshot)

        with (
            patch.object(self.main, "find_victory_banner", return_value=None),
            patch.object(self.main, "find_connection_interrupted_dialog", return_value=None),
            patch.object(self.main, "find_template", return_value=DummyMatch((40, 38))),
            patch.object(self.main, "resolve_current_level", return_value=3),
            patch.object(self.main, "latch_network_fail_closed"),
        ):
            with self.assertRaises(self.main.ProbeProtocolError):
                self.main._clear_red_victory_before_blue_attack(expected_level=2)

    def test_reset_runtime_level_status_clears_victory_guard(self):
        self.main._victory_last_fingerprint = "stale"
        self.main._victory_last_screenshot_id = 123
        self.main._victory_last_click_at = 456.0

        with patch.object(self.main, "get_configured_submarines", return_value=[]):
            self.main.reset_runtime_level_status(1)

        self.assertIsNone(self.main._victory_last_fingerprint)
        self.assertIsNone(self.main._victory_last_screenshot_id)
        self.assertIsNone(self.main._victory_last_click_at)

    def test_blue_victory_latch_blocks_stale_same_level_board_tap(self):
        self.main._latch_blue_victory(3, "test")

        with self.assertRaisesRegex(self.main.ProbeProtocolError, "blue board tap blocked"):
            self.main._assert_blue_board_tap_allowed(3, "probe_cell")

        self.main._reset_blue_victory_latch()
        self.main._assert_blue_board_tap_allowed(4, "probe_cell")

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

    def test_victory_detection_accepts_compact_win_template(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        win_match = self.main.MatchResult(
            template_path=self.main.WIN_TEMPLATE,
            top_left=(100, 20),
            bottom_right=(335, 106),
            center=(217, 63),
            score=0.91,
        )

        with patch.object(
            self.main,
            "find_template_multi_scale",
            side_effect=[None, win_match],
        ) as multi_scale:
            match = self.main.find_victory_banner(screenshot)

        self.assertIs(match, win_match)
        self.assertTrue(np.array_equal(multi_scale.call_args_list[1].args[0], screenshot))
        self.assertEqual(multi_scale.call_args_list[1].args[1], self.main.WIN_TEMPLATE)

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
                "_reconnect_to_base_and_reenter_activity_after_victory",
            ) as reconnect,
            patch.object(
                self.main,
                "resolve_current_level_from_device",
                return_value=7,
            ),
        ):
            next_level = self.main.resolve_next_level_with_retries(
                current_level=7,
                fallback_level=8,
            )

        self.assertIsNone(next_level)
        reconnect.assert_not_called()
        self.assertNotIn(("click", *self.main.SCREEN_CONTINUE_POINT), self.adb.calls)
        package_name = self.main.GAME_PACKAGE_NAME
        self.assertNotIn(("enable_weak_network", package_name), self.adb.calls)
        self.assertNotIn(("enable_reject_network", package_name), self.adb.calls)
        self.assertNotIn(("disable_weak_network", package_name), self.adb.calls)
        self.assertNotIn(("disable_reject_network", package_name), self.adb.calls)

    def test_next_level_detection_directly_uses_activity_after_victory(self):
        with (
            patch.object(self.main, "LEVEL_ADVANCE_RETRIES", 1),
            patch.object(
                self.main,
                "_reconnect_to_base_and_reenter_activity_after_victory",
            ) as reconnect,
            patch.object(
                self.main,
                "resolve_current_level_from_device",
                return_value=8,
            ) as resolve_level,
        ):
            next_level = self.main.resolve_next_level_with_retries(
                current_level=7,
                fallback_level=8,
            )

        self.assertEqual(8, next_level)
        resolve_level.assert_called_once_with(
            fallback_level=8,
            fallback_is_manual=False,
        )
        reconnect.assert_not_called()

    def test_next_level_board_ready_waits_for_victory_overlay_to_clear(self):
        banner_frame = np.zeros((20, 20, 3), dtype=np.uint8)
        clean_frame = np.ones((20, 20, 3), dtype=np.uint8)

        with (
            patch.object(self.main, "NEXT_LEVEL_BOARD_READY_POLL_SECONDS", 0),
            patch.object(
                self.adb,
                "read_screenshot",
                side_effect=[banner_frame, clean_frame],
            ),
            patch.object(
                self.main,
                "find_victory_banner",
                side_effect=[DummyMatch((10, 10)), None],
            ),
            patch.object(self.main, "find_connection_interrupted_dialog", return_value=None),
            patch.object(
                self.main,
                "find_template",
                return_value=DummyMatch((5, 5)),
            ),
        ):
            ready = self.main._wait_for_next_level_board_ready(8, timeout=1.0)

        self.assertTrue(ready)
        self.assertNotIn(("click", *self.main.SCREEN_CONTINUE_POINT), self.adb.calls)

    def test_next_level_board_ready_fails_closed_when_victory_overlay_stays(self):
        banner_frame = np.zeros((20, 20, 3), dtype=np.uint8)

        with (
            patch.object(self.main, "NEXT_LEVEL_BOARD_READY_POLL_SECONDS", 0),
            patch.object(self.adb, "read_screenshot", return_value=banner_frame),
            patch.object(
                self.main,
                "find_victory_banner",
                return_value=DummyMatch((10, 10)),
            ),
            patch.object(self.main, "find_connection_interrupted_dialog", return_value=None),
            patch.object(self.main, "find_template") as find_template,
        ):
            ready = self.main._wait_for_next_level_board_ready(8, timeout=0.01)

        self.assertFalse(ready)
        find_template.assert_not_called()
        self.assertNotIn(("click", *self.main.SCREEN_CONTINUE_POINT), self.adb.calls)

    def test_victory_transition_reconnects_to_base_then_reopens_activity_list(self):
        package_name = self.main.GAME_PACKAGE_NAME
        retry = DummyMatch((320, 240))
        with (
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                return_value=DummyMatch((500, 300)),
            ) as dialog,
            patch.object(self.main, "wait_until_retry_button", return_value=retry) as retry_wait,
            patch.object(
                self.main,
                "wait_until_occur",
                return_value=DummyMatch((100, 100)),
            ) as base_wait,
            patch.object(self.main, "enter_activity", return_value=True) as enter,
        ):
            completed = self.main._reconnect_to_base_and_reenter_activity_after_victory()

        self.assertTrue(completed)
        self.assertIn(("enable_weak_network", package_name), self.adb.calls)
        self.assertIn(("enable_reject_network", package_name), self.adb.calls)
        self.assertIn(("disable_weak_network", package_name), self.adb.calls)
        self.assertIn(("disable_reject_network", package_name), self.adb.calls)
        self.assertIn(("click", *retry.center), self.adb.calls)
        dialog.assert_called_once_with(timeout=self.main.MISS_CONNECTION_DIALOG_WAIT_SECONDS)
        retry_wait.assert_called_once_with(timeout=self.main.MISS_RETRY_BUTTON_WAIT_SECONDS)
        base_wait.assert_called_once_with(
            self.main.ACTIVITY_BUTTON_TEMPLATE,
            timeout=self.main.POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
            poll_interval=self.main.ACTIVITY_REENTRY_POLL_INTERVAL_SECONDS,
        )
        enter.assert_called_once_with(
            prepare_activity_list=True,
            activity_button_timeout=self.main.POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        )

    def test_victory_transition_treats_normal_activity_entry_as_success(self):
        with (
            patch.object(
                self.main,
                "wait_until_connection_interrupted_dialog",
                return_value=DummyMatch((500, 300)),
            ),
            patch.object(
                self.main,
                "wait_until_retry_button",
                return_value=DummyMatch((320, 240)),
            ),
            patch.object(
                self.main,
                "wait_until_occur",
                return_value=DummyMatch((100, 100)),
            ),
            patch.object(self.main, "enter_activity", return_value=False) as enter,
        ):
            completed = self.main._reconnect_to_base_and_reenter_activity_after_victory()

        self.assertTrue(completed)
        enter.assert_called_once_with(
            prepare_activity_list=True,
            activity_button_timeout=self.main.POST_LOGIN_ACTIVITY_BUTTON_WAIT_SECONDS,
        )

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

    def test_committed_victory_reconnects_through_base_before_next_level(self):
        with (
            patch.object(self.main, "handle_victory_prompt", return_value=True) as handle_victory,
            patch.object(
                self.main,
                "_reconnect_to_base_and_reenter_activity_after_victory",
                return_value=True,
            ) as reconnect,
            patch.object(self.main, "enter_activity") as enter_activity,
        ):
            completed = self.main.restart_process()

        self.assertTrue(completed)
        handle_victory.assert_called_once_with(
            timeout=self.main.VICTORY_WAIT_AFTER_HIT_SECONDS,
        )
        reconnect.assert_called_once_with()
        enter_activity.assert_not_called()

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
                result_unknown=True,
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

    def test_successful_probe_with_frame_disagreement_keeps_key_frame_only(self):
        preserve_all = self.main._should_preserve_all_probe_images
        frame_records = [
            {
                "dynamic_hit_vetoed": index == 2,
                "sidebar_completed_lengths": [],
                "result": {"state": state, "score": score},
            }
            for index, (state, score) in enumerate(
                (("hit", 0.95), ("hit", 0.91), ("miss", 0.4)),
                start=1,
            )
        ]

        self.assertFalse(
            preserve_all(
                frame_records,
                suspect_extra_checked=True,
                victory_detected=False,
                result_unknown=False,
            )
        )

    def test_level_memory_log_includes_working_set_and_private_memory(self):
        with (
            patch.object(
                self.main,
                "_process_memory_usage_mb",
                return_value=(123.4, 234.5),
            ),
            patch.object(self.main.logger, "info") as info,
            patch.object(self.main, "write_runtime_status") as write_status,
        ):
            self.main._log_level_memory(7)

        info.assert_called_once_with(
            "level %s memory: working_set=%.1f MB private=%.1f MB",
            7,
            123.4,
            234.5,
        )
        write_status.assert_called_once_with(
            memory_working_set_mb=123.4,
            memory_private_mb=234.5,
        )

    def test_loose_wreck_template_alone_does_not_promote_miss_to_hit(self):
        result = dummy_hit_result("miss")
        frame = object()

        with (
            patch.object(self.main, "red_hit_marker_visible", return_value=False),
            patch.object(
                self.main,
                "visible_wreck_static_detected",
                return_value=False,
            ) as detect,
        ):
            confirmed = self.main.apply_wreck_template_confirmation(
                frame,
                (400, 300),
                result,
            )

        self.assertFalse(confirmed)
        self.assertEqual(result.state, "miss")
        detect.assert_called_once_with(
            frame,
            (400, 300),
            cell_polygon=None,
            filter_surface_reflection=False,
            filter_activity_title_overlay=False,
        )

    def test_stable_miss_rejects_transient_static_wreck_votes(self):
        transient_hit = dummy_hit_result("hit")
        transient_hit.evidence_kind = "static_wreck_hit"
        later_miss = dummy_hit_result("miss")
        later_miss.evidence_kind = "unknown"
        stable_analysis = SimpleNamespace(result=dummy_hit_result("miss"))

        self.assertTrue(
            self.main._stable_miss_rejects_transient_static_wreck(
                [transient_hit, transient_hit, later_miss, later_miss],
                stable_analysis,
                sidebar_completed=False,
                victory_detected=False,
            )
        )
        self.assertFalse(
            self.main._stable_miss_rejects_transient_static_wreck(
                [transient_hit, transient_hit, later_miss, later_miss],
                stable_analysis,
                sidebar_completed=True,
                victory_detected=False,
            )
        )

    def test_static_wreck_persistence_confirmation_uses_delayed_frame(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adb.delay = Mock(return_value=SimpleNamespace(
            capture_screenshot=Mock(return_value=FakeScreenshotCapture(image))
        ))
        with patch.object(
            self.main,
            "visible_wreck_static_detected",
            return_value=False,
        ) as detect:
            self.assertFalse(self.main._static_wreck_persists_after_delay((791, 361)))

        self.adb.delay.assert_called_once_with(
            self.main.STATIC_WRECK_PERSISTENCE_DELAY_SECONDS
        )
        detect.assert_called_once_with(
            image,
            (791, 361),
            cell_polygon=None,
            filter_surface_reflection=False,
            filter_activity_title_overlay=False,
        )

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

    def test_dynamic_hit_without_static_evidence_is_accepted_on_visual_change(self):
        evidence_gate = getattr(self.main, "enforce_positive_hit_evidence", None)
        self.assertIsNotNone(evidence_gate)
        result = dummy_hit_result("hit")
        result.score = 1.0
        result.confidence = 1.0
        result.changed_ratio = 0.2

        vetoed = evidence_gate(
            result,
            wreck_hit=False,
            sidebar_hit=False,
            accept_visual_change=True,
        )

        self.assertFalse(vetoed)
        self.assertEqual(result.state, "hit")
        self.assertFalse(result.evidence_vetoed)

    def test_blue_only_dynamic_hit_still_requires_positive_evidence(self):
        evidence_gate = getattr(self.main, "enforce_positive_hit_evidence", None)
        self.assertIsNotNone(evidence_gate)
        result = dummy_hit_result("hit")
        result.score = 1.0
        result.confidence = 1.0
        result.changed_ratio = 0.2

        vetoed = evidence_gate(
            result,
            wreck_hit=False,
            sidebar_hit=False,
        )

        self.assertTrue(vetoed)
        self.assertEqual(result.state, "miss")
        self.assertTrue(result.evidence_vetoed)

    def test_dynamic_miss_is_promoted_to_hit_on_visual_change(self):
        evidence_gate = getattr(self.main, "enforce_positive_hit_evidence", None)
        self.assertIsNotNone(evidence_gate)
        result = dummy_hit_result("miss")
        result.changed_ratio = 0.01

        vetoed = evidence_gate(
            result,
            wreck_hit=False,
            sidebar_hit=False,
            accept_visual_change=True,
        )

        self.assertFalse(vetoed)
        self.assertEqual(result.state, "hit")
        self.assertFalse(result.evidence_vetoed)

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

    def test_sustained_static_wreck_evidence_requires_all_initial_frames(self):
        records = [
            {"template_hit": True},
            {"template_hit": True},
            {"template_hit": True},
        ]

        self.assertTrue(self.main._has_sustained_static_wreck_evidence(records))
        self.assertFalse(
            self.main._has_sustained_static_wreck_evidence(
                [{"template_hit": True}, {"template_hit": False}, {"template_hit": True}]
            )
        )

    def test_completed_submarine_confirmation_requires_marker_and_hull(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = dummy_hit_result("miss")

        with (
            patch.object(self.main, "red_submarine_marker_visible", return_value=True),
            patch.object(
                self.main,
                "completed_ship_body_score",
                return_value=self.main.COMPLETED_SHIP_BODY_MIN_SCORE,
            ),
        ):
            self.assertTrue(
                self.main.apply_completed_submarine_confirmation(
                    image,
                    (640, 360),
                    result,
                )
            )

        self.assertEqual(result.state, "hit")
        self.assertEqual(result.evidence_kind, "completed_submarine")

    def test_completed_submarine_confirmation_rejects_marker_without_hull(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = dummy_hit_result("miss")

        with (
            patch.object(self.main, "red_submarine_marker_visible", return_value=True),
            patch.object(self.main, "completed_ship_body_score", return_value=0.0),
        ):
            self.assertFalse(
                self.main.apply_completed_submarine_confirmation(
                    image,
                    (640, 360),
                    result,
                )
            )

        self.assertEqual(result.state, "miss")

    def test_probe_response_gate_rejects_static_hit_without_response(self):
        record = {
            "new_wreck_hit": False,
            "sidebar_hit": False,
            "victory_banner": False,
            "result": {
                "state": "hit",
                "score": 1.0,
                "changed_ratio": 0.0,
            },
        }

        self.assertFalse(self.main._probe_record_has_visual_response(record))
        self.assertFalse(self.main._probe_has_visual_response([record]))
        self.assertFalse(self.main._probe_has_positive_hit_evidence([record]))

    def test_probe_response_gate_accepts_miss_board_change(self):
        record = {
            "new_wreck_hit": False,
            "sidebar_hit": False,
            "victory_banner": False,
            "result": {
                "state": "miss",
                "score": 0.1,
                "changed_ratio": self.main.NEAR_HIT_MIN_CHANGED_RATIO,
            },
        }

        self.assertTrue(self.main._probe_record_has_visual_response(record))
        self.assertFalse(self.main._probe_has_positive_hit_evidence([record]))

    def test_probe_response_gate_accepts_explicit_hit_evidence_without_board_delta(self):
        record = {
            "new_wreck_hit": False,
            "sidebar_hit": True,
            "victory_banner": False,
            "result": {
                "state": "hit",
                "score": 0.99,
                "changed_ratio": 0.0,
            },
        }

        self.assertTrue(self.main._probe_has_visual_response([record]))
        self.assertTrue(self.main._probe_has_positive_hit_evidence([record]))

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

    def test_probe_metadata_preserves_previously_confirmed_two_cell_ship(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        click_points = [(400 + index, 300 + index) for index in range(100)]
        metadata = {
            "sidebar_completion_screenshot": screenshot,
            "sidebar_completed_lengths": (3, 2),
        }
        previous_ship = {(3, 4), (3, 5)}
        candidates = previous_ship | {(2, 4), (4, 7), (4, 8), (4, 9)}

        with patch.object(
            self.main,
            "detect_completed_submarine_candidate_cells",
            return_value=candidates,
        ):
            trusted = self.main._trusted_completed_cells_from_probe_metadata(
                metadata,
                click_points,
                grid_size=10,
                anchor=(4, 8),
                preferred_cells=previous_ship,
            )

        self.assertEqual(
            trusted,
            previous_ship | {(4, 7), (4, 8), (4, 9)},
        )
        self.assertNotIn((2, 4), trusted)

    def test_probe_metadata_binds_two_completed_cells_to_red_anchor(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        click_points = [(400 + index, 300 + index) for index in range(16)]
        metadata = {
            "sidebar_completion_screenshot": screenshot,
            "sidebar_completed_lengths": (2, 2),
        }
        candidates = {(0, 2), (1, 2), (3, 0), (3, 1)}
        anchors = {(0, 2), (3, 0)}

        with (
            patch.object(
                self.main,
                "detect_completed_submarine_candidate_cells",
                return_value=candidates,
            ),
            patch.object(
                self.main,
                "detect_red_submarine_marker_cells",
                return_value=anchors,
            ),
        ):
            trusted = self.main._trusted_completed_cells_from_probe_metadata(
                metadata,
                click_points,
                grid_size=4,
            )

        self.assertEqual(trusted, candidates)

    def test_probe_metadata_fills_occluded_middle_cell_of_confirmed_three_cell_ship(self):
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        click_points = [(400 + index, 300 + index) for index in range(36)]
        metadata = {
            "sidebar_completion_screenshot": screenshot,
            "sidebar_completed_lengths": (3,),
        }

        with patch.object(
            self.main,
            "detect_completed_submarine_candidate_cells",
            return_value={(2, 1), (2, 3)},
        ):
            trusted = self.main._trusted_completed_cells_from_probe_metadata(
                metadata,
                click_points,
                grid_size=6,
                anchor=(2, 2),
            )

        self.assertEqual(trusted, {(2, 1), (2, 2), (2, 3)})

    def test_complete_visual_snapshot_replaces_stale_two_cell_ship_assignment(self):
        merge_snapshot = getattr(
            self.main,
            "_merge_completed_visual_snapshot",
            None,
        )
        self.assertIsNotNone(merge_snapshot)
        previous = {(3, 4), (3, 5)}
        latest = {(2, 4), (3, 4)}

        merged = merge_snapshot(previous, latest, completed_lengths=(2,))

        self.assertEqual(merged, latest)
        self.assertNotIn((3, 5), merged)

    def test_incomplete_visual_snapshot_does_not_corrupt_previous_assignment(self):
        merge_snapshot = getattr(
            self.main,
            "_merge_completed_visual_snapshot",
            None,
        )
        self.assertIsNotNone(merge_snapshot)
        previous = {(3, 4), (3, 5)}

        merged = merge_snapshot(
            previous,
            {(2, 4)},
            completed_lengths=(2,),
        )

        self.assertEqual(merged, previous)

    def test_authoritative_completed_ship_survives_a_conflicting_later_snapshot(self):
        corrected_ship = {(9, 8), (9, 9)}
        previous = corrected_ship | {(7, 3), (7, 4), (7, 5)}
        latest = {(8, 8), (9, 8)} | {(7, 3), (7, 4), (7, 5)}

        merged = self.main._merge_completed_visual_snapshot(
            previous,
            latest,
            completed_lengths=(3, 2),
            authoritative_cells=corrected_ship,
        )

        self.assertEqual(merged, previous)
        self.assertNotIn((8, 8), merged)

    def test_online_completed_placement_lock_survives_shorter_followup_geometry(self):
        settings = self.main.RedScoutSettings(self.main.ProbeMode.RED_SCOUT, 1)
        full_ship = {(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)}
        second_ship = {(3, 8), (3, 9)}
        shorter_snapshot = {(7, 3), (7, 4), (7, 5), (7, 6)} | second_ship
        red_result = self.main.RedScoutResult(
            center_cell=(5, 5),
            affected_cells=frozenset({(0, 0), (0, 1)}),
            hit_cells=frozenset({(0, 0), (0, 1)}),
            miss_cells=frozenset(),
            unknown_cells=frozenset(),
            footprint=self.main.RedFootprint(frozenset({(0, 0), (0, 1)})),
            valid=True,
            confidence_by_cell={(0, 0): 0.95, (0, 1): 0.95},
        )
        screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
        snapshots = iter((full_ship | second_ship, shorter_snapshot))

        def online_hit(*_args, **kwargs):
            metadata = kwargs["probe_metadata"]
            metadata.update(
                {
                    "sidebar_completed_lengths": (5, 2),
                    "sidebar_completion_screenshot": screenshot,
                }
            )
            return self.main.ProbeResult.HIT

        with (
            patch.object(self.main.RedScoutPlanner, "choose_center", return_value=(5, 5)),
            patch.object(self.main, "_execute_red_scout_transaction", return_value=red_result),
            patch.object(self.main, "_execute_online_scout_hit", side_effect=online_hit),
            patch.object(
                self.main,
                "_trusted_completed_cells_from_probe_metadata",
                side_effect=lambda *_args, **_kwargs: next(snapshots),
            ),
            patch.object(self.main, "_scan_level_by_strategy", return_value=True) as scan,
            patch.object(self.main, "ONLINE_SCOUT_BATCH_ENABLED", False),
        ):
            completed = self.main._run_red_scout_and_blue_strategy(
                9,
                [[0] * 10 for _row in range(10)],
                [(index % 10, index // 10) for index in range(100)],
                [2, 2, 3, 4, 4, 5],
                set(),
                settings,
            )

        self.assertTrue(completed)
        placements = scan.call_args.kwargs["initial_authoritative_completed_placements"]
        self.assertEqual(
            {placement.cells for placement in placements},
            {
                tuple(sorted(full_ship)),
                tuple(sorted(second_ship)),
            },
        )

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

    def test_adaptive_frames_stop_clear_misses_after_two_frames(self):
        can_stop = getattr(self.main, "_can_stop_probe_frames_early", None)
        self.assertIsNotNone(can_stop)
        records = [
            {
                "dynamic_hit_vetoed": False,
                "template_hit": False,
                "new_wreck_hit": False,
                "sidebar_hit": False,
                "victory_banner": False,
                "result": {"state": "miss", "score": 0.10},
            },
            {
                "dynamic_hit_vetoed": False,
                "template_hit": False,
                "new_wreck_hit": False,
                "sidebar_hit": False,
                "victory_banner": False,
                "result": {"state": "miss", "score": 0.12},
            },
        ]

        self.assertTrue(can_stop(records, [None, None], (2, 3)))

        suspect = [dict(records[0]), dict(records[1])]
        suspect[1]["result"] = {"state": "miss", "score": self.main.SUSPECT_HIT_SCORE_THRESHOLD}
        self.assertFalse(can_stop(suspect, [None, None], (2, 3)))

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
        self.assertIn("initial_authoritative_completed_placements", signature.parameters)

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
            patch.object(self.main, "_scan_level_by_grid_order", return_value=0),
        ):
            completed = self.main._scan_level_by_strategy(
                level=7,
                hit_map=[[0] * 9 for _ in range(9)],
                click_points=[(400, 300)] * 81,
                submarines=[2, 2, 3, 3, 4, 5],
                initial_sidebar_progress=progress,
                initial_visual_hit_count=7,
            )

        self.assertFalse(completed)
        self.assertEqual(self.main._runtime_status.get("hits"), 7)
        self.assertEqual(self.main._runtime_status.get("sidebar_completed_cells"), 6)
        self.assertEqual(update_progress.call_args.args[1], 7)
        self.assertEqual(len(self.main._runtime_status.get("board_states", [])), 9)

    def test_strategy_restores_authoritative_completed_placements_before_replay(self):
        restorer = Mock(return_value=())
        strategy = SimpleNamespace(
            shots={},
            done=True,
            remaining={},
            get_confirmed_ships=lambda: [],
            restore_confirmed_placements=restorer,
            report_result=Mock(),
        )
        placement = ((7, 3), (7, 4), (7, 5), (7, 6), (7, 7))
        fake_bar = SimpleNamespace(total=1, n=0, set_postfix_str=lambda *_args, **_kwargs: None)

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
        ):
            completed = self.main._scan_level_by_strategy(
                level=9,
                hit_map=[[0] * 10 for _ in range(10)],
                click_points=[(400, 300)] * 100,
                submarines=[2, 2, 3, 4, 4, 5],
                initial_authoritative_completed_placements=(placement,),
                initial_sidebar_progress=SidebarProgress(
                    completed_lengths=(2, 2, 3, 4, 4, 5),
                ),
            )

        self.assertTrue(completed)
        restorer.assert_called_once_with((placement,))

    def test_blue_only_completed_placements_stay_green_and_are_never_probed(self):
        # A sidebar/red-marker-confirmed hull is enough to block blue shots,
        # but it must not be fabricated as a real blue hit.  This is the
        # level-4 regression where (0,0)/(1,0) were green then clicked and
        # overwritten as misses.
        placement = ((0, 0), (1, 0))
        statuses = []
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.LEVEL_COMPLETE,
            ) as probe,
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(
                self.main,
                "write_runtime_status",
                side_effect=lambda **kwargs: statuses.append(kwargs),
            ),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _ in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[2, 1],
                initial_visual_candidates=set(placement),
                initial_completed_blocking_placements=(placement,),
                initial_visual_complete_cells=set(placement),
            )

        self.assertTrue(completed)
        probed_cell = probe.call_args.args[2]
        self.assertNotIn(probed_cell, placement)
        initial_status = next(
            status
            for status in statuses
            if status.get("phase") == "strategy_scan" and "board_states" in status
        )
        self.assertEqual(initial_status["board_states"][0][0], "ship")
        self.assertEqual(initial_status["board_states"][1][0], "ship")

    def test_blue_only_completed_placements_remain_skipped_in_reconciliation_scan(self):
        placement = ((0, 0), (1, 0))
        fake_bar = SimpleNamespace(
            total=2,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(self.main, "_scan_level_by_grid_order", return_value=0) as fallback,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _ in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[2],
                initial_completed_blocking_placements=(placement,),
                initial_visual_complete_cells=set(placement),
            )

        self.assertFalse(completed)
        self.assertEqual(fallback.call_args.kwargs["skip_cells"], set(placement))

    def test_strategy_done_does_not_complete_level_while_sidebar_is_active(self):
        strategy = self.main.SubmarineStrategy(3, [2])
        strategy.restore_confirmed_placements((((0, 0), (0, 1)),))
        self.assertTrue(strategy.done)
        fake_bar = SimpleNamespace(
            total=2,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(self.main, "_scan_level_by_grid_order", return_value=7) as fallback,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[2],
                initial_sidebar_progress=SidebarProgress(active_lengths=(2,)),
            )

        self.assertFalse(completed)
        self.assertEqual(fallback.call_args.kwargs["skip_cells"], set())
        self.assertFalse(fallback.call_args.kwargs["stop_when"](self.main.ProbeResult.HIT))
        self.assertEqual(self.main._runtime_status["phase"], "completion_reconcile")

    def test_completion_reconcile_rechecks_visual_cells_until_authoritative_completion(self):
        strategy = self.main.SubmarineStrategy(3, [2])
        fake_bar = SimpleNamespace(
            total=2,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )
        visual_placement = ((0, 0), (0, 1))

        def finish_in_fallback(*_args, result_callback, **kwargs):
            self.assertEqual(kwargs["skip_cells"], set())
            result_callback((0, 0), self.main.ProbeResult.LEVEL_COMPLETE)
            return 1

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(
                self.main,
                "_scan_level_by_grid_order",
                side_effect=finish_in_fallback,
            ),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[2],
                initial_sidebar_progress=SidebarProgress(active_lengths=(2,)),
                initial_authoritative_completed_placements=(visual_placement,),
            )

        self.assertTrue(completed)

    def test_strategy_done_alone_does_not_report_level_complete(self):
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=True,
            remaining=SimpleNamespace(elements=lambda: iter(())),
            get_accounted_completed_lengths=lambda: [3],
            get_confirmed_ships=lambda: [],
        )
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(
                self.main,
                "_scan_level_by_grid_order",
                return_value=0,
            ) as fallback_scan,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
            )

        self.assertFalse(completed)
        fallback_scan.assert_called_once()

    def test_active_sidebar_rechecks_visual_only_completion_in_fallback(self):
        visual_ship = {(1, 0), (1, 1), (1, 2)}
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((3,))),
            choose_next_cell=Mock(return_value=None),
            report_result=Mock(),
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
        )
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(
                self.main,
                "_scan_level_by_grid_order",
                return_value=0,
            ) as fallback_scan,
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_hits=visual_ship,
                initial_sidebar_progress=SidebarProgress(
                    active_lengths=(3,),
                    completed_lengths=(),
                ),
                initial_visual_hit_count=3,
                initial_completed_visual_hits=visual_ship,
                initial_authoritative_completed_visual_hits=set(),
            )

        self.assertFalse(completed)
        fallback_scan.assert_called_once()
        fallback_skip_cells = fallback_scan.call_args.kwargs["skip_cells"]
        self.assertTrue(visual_ship.isdisjoint(fallback_skip_cells))
        stop_when = fallback_scan.call_args.kwargs["stop_when"]
        self.assertFalse(stop_when(self.main.ProbeResult.HIT))

    def test_strategy_prioritizes_unknown_cells_before_scout_miss_rechecks(self):
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
            choose_next_cell=Mock(side_effect=[(0, 0), None]),
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
                    self.main.ProbeResult.MISS,
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
                initial_sidebar_progress=SidebarProgress(completed_lengths=(3,)),
            )

        self.assertTrue(completed)
        self.assertEqual(
            [call.args[2] for call in probe.call_args_list],
            [(0, 0), *targets],
        )
        self.assertEqual(probe.call_count, 5)
        self.assertEqual(strategy.choose_next_cell.call_count, 2)
        self.assertEqual(
            [call.args for call in strategy.report_result.call_args_list],
            [
                ((0, 0), False),
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
                initial_sidebar_progress=SidebarProgress(completed_lengths=(4,)),
            )

        self.assertTrue(completed)
        self.assertEqual(
            [call.args[2] for call in probe.call_args_list],
            targets,
        )
        strategy.choose_next_cell.assert_called_once_with()

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
            choose_next_cell=Mock(return_value=None),
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
        strategy.choose_next_cell.assert_called_once_with()

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
                initial_sidebar_progress=SidebarProgress(completed_lengths=(3,)),
            )

        self.assertTrue(completed)
        strategy.report_result.assert_called_once_with((0, 1), False)
        self.assertEqual(strategy.shots, {(0, 1): False})
        save_shots.assert_called_with(1, 3, {(0, 1): False})
        self.assertEqual(self.main._runtime_status["board_states"][0][1], "miss")

    def test_visual_candidate_is_probed_before_becoming_a_real_hit(self):
        hit_map = [[0] * 3 for _row in range(3)]
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.HIT,
            ) as probe,
            patch.object(
                self.main.SubmarineStrategy,
                "choose_next_cell",
                return_value=None,
            ),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots") as save_shots,
            patch.object(self.main, "_scan_level_by_grid_order", return_value=0),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=hit_map,
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_visual_candidates={(0, 2)},
                initial_visual_hit_count=0,
            )

        self.assertFalse(completed)
        probe.assert_called_once()
        self.assertEqual(probe.call_args.args[2], (0, 2))
        self.assertEqual(hit_map[0][2], 1)
        self.assertTrue(
            any(call.args[-1].get((0, 2)) is True for call in save_shots.call_args_list)
        )

    def test_visual_candidate_miss_is_not_promoted_to_a_hit(self):
        hit_map = [[0] * 3 for _row in range(3)]
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_probe_cell",
                return_value=self.main.ProbeResult.MISS,
            ) as probe,
            patch.object(
                self.main.SubmarineStrategy,
                "choose_next_cell",
                return_value=None,
            ),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots") as save_shots,
            patch.object(self.main, "_scan_level_by_grid_order", return_value=0),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=hit_map,
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_visual_candidates={(0, 2)},
                initial_visual_hit_count=0,
            )

        self.assertFalse(completed)
        probe.assert_called_once()
        self.assertEqual(probe.call_args.args[2], (0, 2))
        self.assertEqual(hit_map[0][2], 0)
        self.assertTrue(
            any(call.args[-1].get((0, 2)) is False for call in save_shots.call_args_list)
        )

    def test_visual_candidate_count_does_not_reserve_blue_batch_capacity(self):
        strategy = self.main.SubmarineStrategy(3, [3])
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_probe_cell",
                side_effect=[
                    self.main.ProbeResult.MISS,
                    self.main.ProbeResult.LEVEL_COMPLETE,
                ],
            ),
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[(400, 300)] * 9,
                submarines=[3],
                initial_visual_candidates={(0, 2)},
                initial_visual_hit_count=0,
            )

        self.assertTrue(completed)
        self.assertEqual(self.main._runtime_status.get("unmapped_visual_hits"), 0)

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
                initial_sidebar_progress=SidebarProgress(completed_lengths=(1,)),
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
                initial_sidebar_progress=SidebarProgress(completed_lengths=(3,)),
            )

        self.assertTrue(completed)
        self.assertEqual([event[0] for event in events[:2]], ["online", "online"])
        self.assertEqual(events[2][0], "offline")
        self.assertEqual({event[1] for event in events[:2]}, {(1, 1), (1, 2)})

    def test_strategy_uses_one_batch_for_multiple_scout_hits(self):
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((1,))),
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
            get_scout_hit_cells=lambda: {(1, 1), (1, 2)},
            report_scout_results=Mock(),
            choose_next_cell=Mock(return_value=(1, 1)),
        )

        def report_result(cell, hit):
            strategy.shots[cell] = hit
            strategy.done = len(strategy.shots) == 2

        strategy.report_result = Mock(side_effect=report_result)
        batch_outcome = self.main.OnlineScoutBatchResult(
            results={
                (1, 1): self.main.ProbeResult.HIT,
                (1, 2): self.main.ProbeResult.HIT,
            },
            metadata={
                (1, 1): {
                    "batch": True,
                    "stable_state": "hit",
                    "blue_bomb_ready": True,
                    "network_ready": True,
                },
                (1, 2): {
                    "batch": True,
                    "stable_state": "hit",
                    "blue_bomb_ready": True,
                    "network_ready": True,
                },
            },
        )

        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(
                self.main,
                "_execute_online_scout_hit_batch",
                return_value=batch_outcome,
            ) as batch,
            patch.object(self.main, "_execute_online_scout_hit") as single,
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(self.main, "write_runtime_status"),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[
                    (400, 300), (500, 300), (600, 300),
                    (400, 400), (500, 400), (600, 400),
                    (400, 500), (500, 500), (600, 500),
                ],
                submarines=[3],
                initial_scout_hits={(1, 1), (1, 2)},
                commit_scout_hits_online=True,
                initial_visual_hit_count=0,
                initial_sidebar_progress=SidebarProgress(completed_lengths=(3,)),
            )

        self.assertTrue(completed)
        batch.assert_called_once()
        single.assert_not_called()
        self.assertEqual(
            strategy.report_result.call_args_list,
            [
                call((1, 1), True),
                call((1, 2), True),
            ],
        )
        strategy.choose_next_cell.assert_called_once_with()

    def test_strategy_does_not_batch_past_unmapped_visual_capacity(self):
        fake_bar = SimpleNamespace(
            total=3,
            n=0,
            set_postfix_str=lambda *_args, **_kwargs: None,
        )
        strategy = SimpleNamespace(
            shots={},
            blocked_cells=set(),
            done=False,
            remaining=SimpleNamespace(elements=lambda: iter((1,))),
            get_accounted_completed_lengths=lambda: [],
            get_confirmed_ships=lambda: [],
            get_scout_hit_cells=lambda: {(1, 1), (1, 2)},
            report_scout_results=Mock(),
            choose_next_cell=Mock(side_effect=[(1, 1), (1, 2)]),
        )

        def report_result(cell, hit):
            strategy.shots[cell] = hit
            strategy.done = len(strategy.shots) >= 2

        strategy.report_result = Mock(side_effect=report_result)
        with (
            patch.object(self.main, "SubmarineStrategy", return_value=strategy),
            patch.object(self.main, "load_saved_level_shots", return_value={}),
            patch.object(self.main, "_execute_online_scout_hit", return_value=self.main.ProbeResult.HIT) as single,
            patch.object(self.main, "_execute_online_scout_hit_batch") as batch,
            patch.object(self.main, "fixed_progress_bar", return_value=nullcontext(fake_bar)),
            patch.object(self.main, "update_fixed_progress"),
            patch.object(self.main, "save_level_shots"),
            patch.object(self.main, "write_runtime_status"),
        ):
            completed = self.main._scan_level_by_strategy(
                level=1,
                hit_map=[[0] * 3 for _row in range(3)],
                click_points=[
                    (400, 300), (500, 300), (600, 300),
                    (400, 400), (500, 400), (600, 400),
                    (400, 500), (500, 500), (600, 500),
                ],
                submarines=[3],
                initial_scout_hits={(1, 1), (1, 2)},
                commit_scout_hits_online=True,
                initial_visual_hit_count=2,
                initial_sidebar_progress=SidebarProgress(completed_lengths=(3,)),
            )

        self.assertTrue(completed)
        batch.assert_not_called()
        self.assertEqual(single.call_count, 2)

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

    def test_runtime_board_snapshot_renders_blocked_safety_cells_as_misses(self):
        strategy = SimpleNamespace(
            shots={(0, 0): True, (0, 1): False},
            blocked_cells={(1, 0)},
        )

        states = self.main.build_runtime_board_states(strategy, 3)

        self.assertEqual(states[0][0], "hit")
        self.assertEqual(states[0][1], "miss")
        self.assertEqual(states[1][0], "miss")
        self.assertEqual(states[2][2], "unknown")

    def test_runtime_board_snapshot_renders_visual_complete_ship_green_over_real_hits(self):
        strategy = SimpleNamespace(
            shots={(0, 0): True},
            visual_complete_cells={(0, 0), (0, 1), (0, 2)},
            get_cell_states=lambda: [
                ["hit", "unknown", "unknown"],
                ["unknown", "unknown", "unknown"],
                ["unknown", "unknown", "unknown"],
            ],
        )

        states = self.main.build_runtime_board_states(strategy, 3)

        self.assertEqual(states[0][0], "ship")
        self.assertEqual(states[0][1], "ship")
        self.assertEqual(states[0][2], "ship")

        strategy.shots[(0, 1)] = False
        states = self.main.build_runtime_board_states(strategy, 3)
        self.assertEqual(states[0][1], "ship")

    def test_runtime_board_snapshot_keeps_completed_ship_safety_ring_as_misses(self):
        strategy = self.main.SubmarineStrategy(5, [2, 1])
        false_perimeter_hit = (1, 1)
        strategy.report_result(false_perimeter_hit, True)
        strategy.restore_confirmed_placements((((2, 1), (2, 2)),))

        states = self.main.build_runtime_board_states(strategy, 5)

        self.assertEqual(states[2][1], "ship")
        self.assertEqual(states[2][2], "ship")
        self.assertIn(false_perimeter_hit, strategy.blocked_cells)
        self.assertEqual(states[1][1], "miss")

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
            patch.object(self.main, "_exit_activity_after_probe_click") as exit_activity,
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
        exit_activity.assert_called_once_with(
            self.main.RUN_DEBUG_DIR / "debug_quit1.png",
            use_system_back=True,
        )

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
