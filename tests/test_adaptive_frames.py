import importlib
import json
import tempfile
import unittest
from pathlib import Path


try:
    adaptive_frames = importlib.import_module("utils.adaptive_frames")
except ModuleNotFoundError:
    adaptive_frames = None


def frame(state="hit", *, vetoed=False, evidence_vetoed=False):
    return {
        "dynamic_hit_vetoed": vetoed,
        "result": {
            "state": state,
            "score": 0.95 if state == "hit" else 0.10,
            "evidence_vetoed": evidence_vetoed,
        },
    }


class AdaptiveFramesTest(unittest.TestCase):
    def test_three_explicit_stable_hits_can_stop_early(self):
        self.assertIsNotNone(adaptive_frames)

        self.assertTrue(
            adaptive_frames.can_stop_after_stable_hit_frames(
                [frame(), frame(), frame()],
            )
        )

    def test_mixed_or_vetoed_frames_cannot_stop_early(self):
        self.assertIsNotNone(adaptive_frames)
        cases = (
            [frame(), frame("miss"), frame()],
            [frame(), frame(vetoed=True), frame()],
            [frame(), frame(evidence_vetoed=True), frame()],
            [frame(), frame()],
        )

        for records in cases:
            with self.subTest(records=records):
                self.assertFalse(
                    adaptive_frames.can_stop_after_stable_hit_frames(records)
                )

    def test_replay_report_disables_rule_when_any_final_decision_disagrees(self):
        self.assertIsNotNone(adaptive_frames)
        report = adaptive_frames.evaluate_adaptive_hit_replay(
            [
                {"decision": "hit", "frames": [frame()] * 4},
                {"decision": "miss", "frames": [frame()] * 4},
                {"decision": "miss", "frames": [frame("miss")] * 4},
            ]
        )

        self.assertEqual(report.total_samples, 3)
        self.assertEqual(report.eligible_samples, 2)
        self.assertEqual(report.mismatches, 1)
        self.assertFalse(report.safe_to_enable)

    def test_replay_report_enables_rule_after_nonempty_exact_replay(self):
        self.assertIsNotNone(adaptive_frames)
        report = adaptive_frames.evaluate_adaptive_hit_replay(
            [
                {"decision": "hit", "frames": [frame()] * 4},
                {"decision": "miss", "frames": [frame("miss")] * 4},
            ]
        )

        self.assertEqual(report.eligible_samples, 1)
        self.assertEqual(report.mismatches, 0)
        self.assertTrue(report.safe_to_enable)

    def test_directory_replay_ignores_invalid_json_and_counts_valid_results(self):
        self.assertIsNotNone(adaptive_frames)
        replay_directory = getattr(
            adaptive_frames,
            "evaluate_probe_sample_directory",
            None,
        )
        self.assertIsNotNone(replay_directory)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid_dir = root / "level_1_cell_0_sample"
            valid_dir.mkdir()
            (valid_dir / "result.json").write_text(
                json.dumps({"decision": "hit", "frames": [frame()] * 4}),
                encoding="utf-8",
            )
            broken_dir = root / "level_1_cell_1_sample"
            broken_dir.mkdir()
            (broken_dir / "result.json").write_text("{broken", encoding="utf-8")

            report = replay_directory(root)

        self.assertEqual(report.total_samples, 1)
        self.assertEqual(report.eligible_samples, 1)
        self.assertTrue(report.safe_to_enable)


if __name__ == "__main__":
    unittest.main()
