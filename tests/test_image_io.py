import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.image_io import read_image_compat, write_image_compat


class ImageIoCompatibilityTest(unittest.TestCase):
    def test_round_trip_supports_non_ascii_windows_path(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[3:8, 5:11] = (12, 80, 220)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "中文目录" / "截图.png"
            self.assertTrue(write_image_compat(path, image))
            decoded = read_image_compat(path)

        self.assertIsNotNone(decoded)
        np.testing.assert_array_equal(decoded, image)


if __name__ == "__main__":
    unittest.main()
