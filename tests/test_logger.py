import tempfile
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils import logger


class LoggerRetentionTest(unittest.TestCase):
    def test_file_handler_rotates_at_five_mb_and_keeps_four_files_total(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = logger.build_file_handler(Path(temp_dir) / "bbma.log")
            try:
                self.assertIsInstance(handler, RotatingFileHandler)
                self.assertEqual(handler.maxBytes, 5 * 1024 * 1024)
                self.assertEqual(handler.backupCount, 3)
            finally:
                handler.close()


if __name__ == "__main__":
    unittest.main()
