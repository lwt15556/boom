import json
import tempfile
import unittest
from pathlib import Path

from tools.evaluate_vision_samples import build_report, collect_samples


class VisionSampleReportTest(unittest.TestCase):
    def test_report_keeps_unlabelled_samples_out_of_accuracy(self):
        samples = [
            {"decision": "hit", "expected": "hit", "evidence_kind": "dynamic_attack_hit", "level": 1},
            {"decision": "miss", "expected": "hit", "evidence_kind": "static_wreck_hit", "level": 1},
            {"decision": "unknown", "expected": None, "evidence_kind": "unknown", "level": 2},
        ]
        report = build_report(samples)
        self.assertEqual(report["sample_count"], 3)
        self.assertEqual(report["labelled_count"], 2)
        self.assertEqual(report["review_accuracy"], 0.5)
        self.assertEqual(report["unknown_rate"], 1 / 3)

    def test_collects_review_and_evidence_from_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "probe"
            sample.mkdir()
            (sample / "result.json").write_text(
                json.dumps({"level": 3, "cell": [1, 2], "decision": "hit", "evidence_kind": "completed_submarine"}),
                encoding="utf-8",
            )
            (sample / "review.json").write_text(json.dumps({"expected": "hit"}), encoding="utf-8")
            records = collect_samples(root)
        self.assertEqual(records[0]["evidence_kind"], "completed_submarine")
        self.assertEqual(records[0]["expected"], "hit")


if __name__ == "__main__":
    unittest.main()
