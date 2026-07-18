import json
import tempfile
import unittest
from pathlib import Path

from utils.probe_evaluation import evaluate_probe_samples


class ProbeEvaluationTest(unittest.TestCase):
    def test_counts_confusion_matrix_and_unlabeled_samples(self):
        samples = [
            {"decision": "hit", "ground_truth": "hit"},
            {"decision": "hit", "ground_truth": "miss"},
            {"decision": "miss", "ground_truth": "hit"},
            {"decision": "unknown", "ground_truth": "miss"},
            {"decision": "miss"},
        ]

        report = evaluate_probe_samples(samples)

        self.assertEqual(report.total_samples, 5)
        self.assertEqual(report.labeled_samples, 4)
        self.assertEqual(report.unlabeled_samples, 1)
        self.assertEqual(report.true_hits, 1)
        self.assertEqual(report.false_hits, 1)
        self.assertEqual(report.missed_hits, 1)
        self.assertEqual(report.true_misses, 0)
        self.assertEqual(report.unknown, 1)
        self.assertAlmostEqual(report.hit_precision, 0.5)
        self.assertAlmostEqual(report.hit_recall, 0.5)

    def test_directory_uses_review_file_without_mutating_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = Path(temp_dir) / "sample"
            sample_dir.mkdir()
            (sample_dir / "result.json").write_text(
                json.dumps({"decision": "miss"}),
                encoding="utf-8",
            )
            (sample_dir / "review.json").write_text(
                json.dumps({"ground_truth": "hit", "note": "visible wreck"}),
                encoding="utf-8",
            )

            report = evaluate_probe_samples(temp_dir)

        self.assertEqual(report.labeled_samples, 1)
        self.assertEqual(report.missed_hits, 1)

    def test_directory_uses_final_unknown_status_over_boolean_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_dir = Path(temp_dir) / "sample"
            sample_dir.mkdir()
            (sample_dir / "result.json").write_text(
                json.dumps({"decision": "miss", "ground_truth": "hit"}),
                encoding="utf-8",
            )
            (sample_dir / "status.json").write_text(
                json.dumps({"stage": "complete", "decision": "unknown"}),
                encoding="utf-8",
            )

            report = evaluate_probe_samples(temp_dir)

        self.assertEqual(report.unknown, 1)
        self.assertEqual(report.missed_hits, 0)

    def test_rejects_invalid_labels_without_counting_them_as_truth(self):
        report = evaluate_probe_samples(
            [
                {"decision": "hit", "ground_truth": "maybe"},
                {"decision": "broken", "ground_truth": "hit"},
            ]
        )

        self.assertEqual(report.total_samples, 2)
        self.assertEqual(report.labeled_samples, 0)
        self.assertEqual(report.invalid_samples, 2)


if __name__ == "__main__":
    unittest.main()
