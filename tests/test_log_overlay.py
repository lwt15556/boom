import tempfile
import unittest
from pathlib import Path

from tools.log_overlay import _read_log_chunk


class LogOverlayTest(unittest.TestCase):
    def test_incremental_log_reader_advances_without_repeating_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.log"
            path.write_text("第一行\n第二行\n", encoding="utf-8")

            first_lines, position = _read_log_chunk(path, 0)
            no_lines, unchanged_position = _read_log_chunk(path, position)
            with path.open("a", encoding="utf-8") as file:
                file.write("第三行\n")
            appended_lines, final_position = _read_log_chunk(path, position)

        self.assertEqual(first_lines, ["第一行", "第二行"])
        self.assertEqual(no_lines, [])
        self.assertEqual(unchanged_position, position)
        self.assertEqual(appended_lines, ["第三行"])
        self.assertGreater(final_position, position)


if __name__ == "__main__":
    unittest.main()
