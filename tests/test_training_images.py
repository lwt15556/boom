import unittest
from itertools import combinations
from pathlib import Path

import cv2

from config import LEVEL_GRID_SIZES, SUBMARINES
from save_points.points import read_saved_points
from utils.sidebar_progress import (
    detect_sidebar_progress,
    resolve_completed_ship_cells,
    resolve_completed_ship_cells_by_anchors,
    resolution_has_unique_anchor_support,
)
from utils.level_title_recognition import recognize_level_title
from utils.wreck_detection import (
    detect_completed_submarine_candidate_cells,
    detect_red_submarine_marker_cells,
    detect_visible_wreck_cells,
)


def _training_root() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    return next(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and any(ord(character) > 127 for character in path.name)
            and (path / "before.png").exists()
        ),
        None,
    )


TRAINING_ROOT = _training_root()


@unittest.skipUnless(
    TRAINING_ROOT is not None,
    "local training images are not present",
)
class TrainingImageRecognitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert TRAINING_ROOT is not None
        cls.training_root = TRAINING_ROOT

    def _points(self, level: int):
        expected_n = LEVEL_GRID_SIZES.get(level, 10)
        return read_saved_points(level, expected_n=expected_n)

    def _scaled_points(self, level: int, image):
        points = self._points(level)
        self.assertIsNotNone(points)
        scale_x = image.shape[1] / 1280.0
        scale_y = image.shape[0] / 720.0
        return [
            (int(round(x * scale_x)), int(round(y * scale_y)))
            for x, y in points
        ]

    def test_level_22_wreck_image_matches_visible_wreck_cells(self):
        image = cv2.imread(str(self.training_root / "before.png"))
        self.assertIsNotNone(image)
        cells = detect_visible_wreck_cells(image, self._points(22), 10)
        self.assertEqual(cells, {(5, 5), (5, 6), (5, 7), (5, 8)})

    def test_level_22_complete_submarine_image_restores_full_length(self):
        image = cv2.imread(str(self.training_root / "after_1.png"))
        self.assertIsNotNone(image)
        points = self._points(22)
        candidates = detect_completed_submarine_candidate_cells(image, points, 10)
        anchors = detect_red_submarine_marker_cells(image, points, 10)
        progress = detect_sidebar_progress(image, (2, 2, 3, 4, 5))
        self.assertEqual(progress.completed_lengths, (5,))

        resolution = resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            progress.completed_lengths,
            grid_size=10,
            preferred_cells=candidates,
            fallback_to_global=False,
        )
        self.assertEqual(
            resolution.placements,
            (((5, 5), (5, 6), (5, 7), (5, 8), (5, 9)),),
        )

    def test_level_10_complete_submarines_match_sidebar_lengths(self):
        image = cv2.imread(str(self.training_root / "debug_quit1_retry_1.png"))
        self.assertIsNotNone(image)
        points = self._points(10)
        candidates = detect_completed_submarine_candidate_cells(image, points, 10)
        anchors = detect_red_submarine_marker_cells(image, points, 10)
        progress = detect_sidebar_progress(image, (2, 2, 3, 4, 4, 5))
        self.assertEqual(progress.completed_lengths, (5, 4, 4, 2, 2))

        resolution = resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            progress.completed_lengths,
            grid_size=10,
            preferred_cells=candidates,
            fallback_to_global=False,
        )
        self.assertEqual(
            tuple(sorted(map(len, resolution.placements), reverse=True)),
            (5, 4, 4, 2, 2),
        )
        self.assertEqual(resolution.unresolved_lengths, ())

    def test_level_7_uncertain_sidebar_does_not_write_ship_cells(self):
        image = cv2.imread(
            str(self.training_root / "4dbf663a-0660-45f6-8352-dca4ef84a4b6.png")
        )
        self.assertIsNotNone(image)
        points = self._scaled_points(7, image)
        progress = detect_sidebar_progress(image, (2, 2, 3, 3, 4, 5))
        self.assertIsNotNone(progress)
        self.assertFalse(progress.valid)

        candidates = detect_completed_submarine_candidate_cells(image, points, 9)
        anchors = detect_red_submarine_marker_cells(image, points, 9)
        resolution = resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            progress.completed_lengths,
            grid_size=9,
            preferred_cells=candidates,
            fallback_to_global=False,
        )
        self.assertEqual(resolution.placements, ())
        self.assertEqual(resolution.unresolved_lengths, (5, 4, 3, 2))

    def test_level_7_scaled_image_restores_confirmed_submarines(self):
        image_path = self.training_root / "屏幕截图 2026-07-13 232351.png"
        if not image_path.exists():
            self.skipTest("scaled level-7 training screenshot is not present")
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image)
        points = self._scaled_points(7, image)
        progress = detect_sidebar_progress(image, (2, 2, 3, 3, 4, 5))
        self.assertIsNotNone(progress)
        self.assertTrue(progress.valid)
        self.assertEqual(progress.completed_lengths, (4, 2, 2))

        candidates = detect_completed_submarine_candidate_cells(image, points, 9)
        anchors = detect_red_submarine_marker_cells(image, points, 9)
        resolution = resolve_completed_ship_cells_by_anchors(
            candidates,
            anchors,
            progress.completed_lengths,
            grid_size=9,
            preferred_cells=candidates,
            fallback_to_global=False,
        )
        self.assertEqual(
            tuple(sorted(map(len, resolution.placements), reverse=True)),
            (4, 2, 2),
        )
        self.assertEqual(resolution.unresolved_lengths, ())

    def test_all_current_training_images_resolve_reported_complete_fleet(self):
        checked = 0
        expected_checked = 0
        for image_path in sorted(self.training_root.glob("*.png")):
            image = cv2.imread(str(image_path))
            self.assertIsNotNone(image, image_path.name)
            title = recognize_level_title(
                image,
                reference_dir=Path(__file__).resolve().parents[1] / "save_points" / "imgs",
            )
            self.assertIsNotNone(title, image_path.name)
            self.assertTrue(title.confident, image_path.name)
            level = title.level
            grid_size = LEVEL_GRID_SIZES.get(level, 10)
            points = self._scaled_points(level, image)
            fleet = SUBMARINES[level]
            progress = detect_sidebar_progress(image, fleet)
            self.assertIsNotNone(progress, image_path.name)
            if not progress.completed_lengths:
                continue
            expected_checked += 1

            candidates = detect_completed_submarine_candidate_cells(
                image,
                points,
                grid_size,
            )
            anchors = detect_red_submarine_marker_cells(image, points, grid_size)
            resolution = resolve_completed_ship_cells_by_anchors(
                candidates,
                anchors,
                progress.completed_lengths,
                grid_size=grid_size,
                preferred_cells=candidates,
                fallback_to_global=False,
            )
            if resolution.unresolved_lengths:
                broad = detect_completed_submarine_candidate_cells(
                    image,
                    points,
                    grid_size,
                    preserve_alternatives=True,
                )
                broad_resolution = resolve_completed_ship_cells_by_anchors(
                    broad or candidates,
                    anchors,
                    progress.completed_lengths,
                    grid_size=grid_size,
                    preferred_cells=candidates,
                    fallback_to_global=False,
                    allow_ambiguous=True,
                )
                if (
                    not broad_resolution.unresolved_lengths
                    and resolution_has_unique_anchor_support(
                        broad_resolution.placements,
                        anchors,
                    )
                ):
                    resolution = broad_resolution
                else:
                    resolution = resolve_completed_ship_cells(
                        broad or candidates,
                        progress.completed_lengths,
                        grid_size=grid_size,
                        preferred_cells=candidates,
                    )
            self.assertEqual(
                resolution.unresolved_lengths,
                (),
                image_path.name,
            )
            self.assertEqual(
                tuple(sorted(map(len, resolution.placements), reverse=True)),
                tuple(sorted(progress.completed_lengths, reverse=True)),
                image_path.name,
            )
            # A completed submarine is one contiguous horizontal or vertical
            # run. Different submarines keep the one-cell exclusion ring, so
            # bright wreck clusters cannot become a second complete ship.
            for placement in resolution.placements:
                cells = tuple(placement)
                rows = {row for row, _ in cells}
                cols = {col for _, col in cells}
                self.assertTrue(len(rows) == 1 or len(cols) == 1, image_path.name)
                if len(rows) == 1:
                    columns = sorted(col for _, col in cells)
                    self.assertEqual(
                        columns,
                        list(range(columns[0], columns[-1] + 1)),
                        image_path.name,
                    )
                else:
                    rows_sorted = sorted(row for row, _ in cells)
                    self.assertEqual(
                        rows_sorted,
                        list(range(rows_sorted[0], rows_sorted[-1] + 1)),
                        image_path.name,
                    )
            for first, second in combinations(resolution.placements, 2):
                self.assertFalse(
                    any(
                        max(abs(row_a - row_b), abs(col_a - col_b)) <= 1
                        for row_a, col_a in first
                        for row_b, col_b in second
                    ),
                    image_path.name,
                )
            checked += 1

        # Derive the expected count from the current corpus so new captures
        # are included automatically instead of causing a stale fixed-count
        # failure.
        self.assertEqual(checked, expected_checked)
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
