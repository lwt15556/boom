import tempfile
import unittest
from pathlib import Path

from utils.pending_probe import (
    clear_pending_probe,
    has_pending_probe,
    read_pending_probe,
    update_pending_probe,
    write_pending_probe,
)


class PendingProbeTest(unittest.TestCase):
    def test_pending_probe_survives_process_memory_and_clears_explicitly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pending_probe.json"
            write_pending_probe(
                mode="red_scout",
                level=3,
                cell=(2, 2),
                index=12,
                phase="REQUEST_PENDING",
                path=path,
            )

            self.assertTrue(has_pending_probe(path))
            payload = read_pending_probe(path)
            self.assertEqual(payload["mode"], "red_scout")
            self.assertEqual(payload["cell"], [2, 2])

            self.assertTrue(
                update_pending_probe(
                    path=path,
                    phase="REQUEST_DISCARDED",
                    request_discarded=True,
                )
            )
            payload = read_pending_probe(path)
            self.assertEqual(payload["phase"], "REQUEST_DISCARDED")
            self.assertTrue(payload["request_discarded"])

            clear_pending_probe(path)
            self.assertFalse(has_pending_probe(path))

    def test_update_missing_pending_probe_is_a_noop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.json"
            self.assertFalse(update_pending_probe(path=path, phase="INTERRUPTED"))
            self.assertFalse(path.exists())

    def test_corrupt_pending_probe_is_treated_as_interrupted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pending_probe.json"
            path.write_text("{not-json", encoding="utf-8")

            payload = read_pending_probe(path)

            self.assertTrue(has_pending_probe(path))
            self.assertEqual(payload["phase"], "INTERRUPTED")
            self.assertTrue(payload["state_unknown"])

    def test_non_object_pending_probe_is_treated_as_interrupted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "pending_probe.json"
            path.write_text("[]", encoding="utf-8")

            payload = read_pending_probe(path)

            self.assertTrue(has_pending_probe(path))
            self.assertEqual(payload["phase"], "INTERRUPTED")
            self.assertTrue(payload["state_unknown"])


if __name__ == "__main__":
    unittest.main()
