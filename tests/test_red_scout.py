import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import utils.red_scout as red_scout_module
from utils.image_match import MatchResult
from utils.red_scout import (
    AmmoFingerprint,
    ProbeMode,
    RED_BOMB_TEMPLATE,
    ammo_fingerprint_matches,
    build_ammo_fingerprint,
    load_red_scout_settings,
    locate_red_bomb_button,
    red_bomb_selected,
)


class RedBombVisualTest(unittest.TestCase):
    MATCH = MatchResult(
        template_path=Path("red_bomb_button.png"),
        top_left=(1173, 619),
        bottom_right=(1256, 699),
        center=(1214, 659),
        score=0.99,
    )

    @classmethod
    def _frame_with_ammo(cls, digit: str, *, selected: bool = False):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        x1, y1 = cls.MATCH.top_left
        x2, y2 = cls.MATCH.bottom_right
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (18, 24, 160), cv2.FILLED)
        cv2.putText(
            frame,
            digit,
            (x2 - 17, y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_8,
        )
        if selected:
            cv2.rectangle(
                frame,
                cls.MATCH.top_left,
                cls.MATCH.bottom_right,
                (255, 255, 255),
                3,
            )
        return frame

    def test_ammo_fingerprint_ignores_selection_highlight(self):
        plain_frames = [self._frame_with_ammo("2") for _ in range(3)]
        selected_frames = [
            self._frame_with_ammo("2", selected=True)
            for _ in range(3)
        ]

        plain = build_ammo_fingerprint(plain_frames, self.MATCH)
        selected = build_ammo_fingerprint(selected_frames, self.MATCH)

        self.assertIsNotNone(plain)
        self.assertIsNotNone(selected)
        self.assertTrue(ammo_fingerprint_matches(plain, selected))

    def test_ammo_fingerprint_ignores_wide_selection_highlight(self):
        plain_frames = [self._frame_with_ammo("2") for _ in range(3)]
        selected_frames = []
        for _ in range(3):
            frame = self._frame_with_ammo("2")
            cv2.rectangle(
                frame,
                self.MATCH.top_left,
                (self.MATCH.bottom_right[0] - 1, self.MATCH.bottom_right[1] - 1),
                (255, 255, 255),
                5,
            )
            selected_frames.append(frame)

        plain = build_ammo_fingerprint(plain_frames, self.MATCH)
        selected = build_ammo_fingerprint(selected_frames, self.MATCH)

        self.assertIsNotNone(plain)
        self.assertIsNotNone(selected)
        self.assertTrue(ammo_fingerprint_matches(plain, selected))

    def test_ammo_fingerprint_distinguishes_digits(self):
        digit_two = build_ammo_fingerprint(
            [self._frame_with_ammo("2") for _ in range(3)],
            self.MATCH,
        )
        digit_one = build_ammo_fingerprint(
            [self._frame_with_ammo("1") for _ in range(3)],
            self.MATCH,
        )

        self.assertIsNotNone(digit_two)
        self.assertIsNotNone(digit_one)
        self.assertFalse(ammo_fingerprint_matches(digit_two, digit_one))

    def test_red_bomb_selected_detects_white_border(self):
        plain = self._frame_with_ammo("2")
        selected = self._frame_with_ammo("2")
        cv2.rectangle(
            selected,
            self.MATCH.top_left,
            self.MATCH.bottom_right,
            (255, 255, 255),
            3,
        )

        self.assertFalse(red_bomb_selected(plain, self.MATCH))
        self.assertTrue(red_bomb_selected(selected, self.MATCH))

    def test_red_bomb_selected_detects_rounded_partial_ui_border(self):
        selected = self._frame_with_ammo("2")
        x1, y1 = self.MATCH.top_left
        x2, y2 = self.MATCH.bottom_right
        gap = 20
        for start, end in (
            ((x1 + gap, y1), (x2 - gap - 1, y1)),
            ((x1 + gap, y2 - 1), (x2 - gap - 1, y2 - 1)),
            ((x1, y1 + gap), (x1, y2 - gap - 1)),
            ((x2 - 1, y1 + gap), (x2 - 1, y2 - gap - 1)),
        ):
            cv2.line(selected, start, end, (255, 255, 255), 2, cv2.LINE_AA)

        self.assertTrue(red_bomb_selected(selected, self.MATCH))

    def test_locate_red_bomb_button_normalizes_template_hit_to_full_button(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi_match = MatchResult(
            template_path=Path("red_bomb_button.png"),
            top_left=(7, 17),
            bottom_right=(48, 61),
            center=(27, 39),
            score=0.99,
        )

        with patch(
            "utils.red_scout.find_template_multi_scale",
            return_value=roi_match,
        ) as find_template:
            result = locate_red_bomb_button(frame)

        self.assertEqual(result, self.MATCH)
        roi, _template_path = find_template.call_args.args
        self.assertEqual(roi.shape, (80, 83, 3))
        self.assertEqual(
            find_template.call_args.kwargs,
            {
                "scales": (0.85, 0.95, 1.0, 1.05, 1.15),
                "threshold": 0.72,
                "shape_weight": 0.0,
            },
        )

    def test_locate_red_bomb_button_matches_real_masked_template(self):
        rng = np.random.default_rng(20260714)
        frame = rng.integers(0, 96, size=(720, 1280, 3), dtype=np.uint8)
        template = cv2.imread(str(RED_BOMB_TEMPLATE))
        self.assertIsNotNone(template)

        x, y = 1180, 630
        height, width = template.shape[:2]
        frame[y : y + height, x : x + width] = template

        match = locate_red_bomb_button(frame)

        self.assertIsNotNone(match)
        self.assertTrue(np.isfinite(match.score))
        self.assertEqual(match.top_left, self.MATCH.top_left)
        self.assertEqual(match.bottom_right, self.MATCH.bottom_right)
        self.assertEqual(match.center, self.MATCH.center)

    def test_locate_red_bomb_button_rejects_missing_template(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.assertIsNone(locate_red_bomb_button(frame))

    def test_locate_red_bomb_button_accepts_only_finite_match_scores(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for score in (0.91, np.nan, np.inf, -np.inf):
            with self.subTest(score=score):
                match = MatchResult(
                    template_path=Path("red_bomb_button.png"),
                    top_left=(1, 2), bottom_right=(3, 4), center=(2, 3),
                    score=score,
                )
                with patch(
                    "utils.red_scout.find_template_multi_scale",
                    return_value=match,
                ):
                    result = locate_red_bomb_button(frame)
                if np.isfinite(score):
                    self.assertIsNotNone(result)
                else:
                    self.assertIsNone(result)

    def test_public_image_helpers_reject_malformed_arrays(self):
        invalid_images = (
            None,
            np.empty((0, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 1), dtype=np.uint8),
            np.zeros((2, 2, 5), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.float32),
            np.full((2, 2, 3), np.nan, dtype=np.float32),
            np.zeros((2, 2, 3), dtype=object),
        )
        for image in invalid_images:
            with self.subTest(dtype=getattr(image, "dtype", None)):
                self.assertIsNone(locate_red_bomb_button(image))
                self.assertFalse(red_bomb_selected(image, self.MATCH))
                self.assertIsNone(red_scout_module._ammo_mask(image, self.MATCH))

    def test_malformed_fingerprints_and_iou_fail_closed(self):
        valid = AmmoFingerprint((2, 2), bytes([0xF0]), 4)
        malformed = (
            AmmoFingerprint((True, 2), bytes([0xF0]), 4),
            AmmoFingerprint((2, 0), bytes([0xF0]), 4),
            AmmoFingerprint((2, 2), b"", 4),
            AmmoFingerprint((2, 2), bytes([0xF0]), 3),
            AmmoFingerprint((2, 2), bytes([0xF0]), 0),
            AmmoFingerprint((2, 2), bytes([0xF0]), 5),
        )
        for fingerprint in malformed:
            with self.subTest(fingerprint=fingerprint):
                self.assertFalse(ammo_fingerprint_matches(fingerprint, valid))
        for minimum_iou in (-1, 2, np.nan, np.inf, -np.inf):
            self.assertFalse(ammo_fingerprint_matches(valid, valid, minimum_iou))
        self.assertTrue(ammo_fingerprint_matches(valid, valid, 0))
        self.assertTrue(ammo_fingerprint_matches(valid, valid, 1))


class RedScoutSettingsTest(unittest.TestCase):
    def test_defaults_to_blue_only(self):
        settings = load_red_scout_settings({})

        self.assertEqual(settings.mode, ProbeMode.BLUE_ONLY)
        self.assertEqual(settings.count, 2)

    def test_reads_red_mode_and_configurable_count(self):
        settings = load_red_scout_settings(
            {
                "BBMA_PROBE_MODE": "red_scout",
                "BBMA_RED_SCOUT_COUNT": "3",
            }
        )

        self.assertEqual(settings.mode, ProbeMode.RED_SCOUT)
        self.assertEqual(settings.count, 3)

    def test_validates_red_scout_count(self):
        cases = (
            ("1", 1),
            ("10", 10),
            ("50", 50),
            ("0", 2),
            ("51", 2),
            ("", 2),
            ("invalid", 2),
        )

        for raw_count, expected_count in cases:
            with self.subTest(raw_count=raw_count):
                settings = load_red_scout_settings(
                    {
                        "BBMA_PROBE_MODE": "red_scout",
                        "BBMA_RED_SCOUT_COUNT": raw_count,
                    }
                )

                self.assertEqual(settings.mode, ProbeMode.RED_SCOUT)
                self.assertEqual(settings.count, expected_count)

    def test_reads_settings_from_process_environment(self):
        with patch.dict(
            os.environ,
            {
                "BBMA_PROBE_MODE": "red_scout",
                "BBMA_RED_SCOUT_COUNT": "4",
            },
            clear=True,
        ):
            settings = load_red_scout_settings()

        self.assertEqual(settings.mode, ProbeMode.RED_SCOUT)
        self.assertEqual(settings.count, 4)

    def test_invalid_values_fall_back_without_enabling_red_mode(self):
        settings = load_red_scout_settings(
            {
                "BBMA_PROBE_MODE": "invalid",
                "BBMA_RED_SCOUT_COUNT": "99",
            }
        )

        self.assertEqual(settings.mode, ProbeMode.BLUE_ONLY)
        self.assertEqual(settings.count, 2)


class RedScoutAnalyzerTest(unittest.TestCase):
    GRID_SIZE = 5
    CLICK_POINTS = tuple(
        (row, col)
        for row in range(5)
        for col in range(5)
    )
    BEFORE_IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)
    AFTER_IMAGES = tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3))

    def _analyzer(self, classifier, hit_detector):
        analyzer_type = getattr(red_scout_module, "RedScoutAnalyzer", None)
        self.assertIsNotNone(analyzer_type, "RedScoutAnalyzer must be public")
        return analyzer_type(classifier=classifier, hit_detector=hit_detector)

    def test_analyzer_accepts_submarine_lengths_for_completion_evidence(self):
        parameters = inspect.signature(
            red_scout_module.RedScoutAnalyzer.analyze
        ).parameters

        self.assertIn("submarine_lengths", parameters)

    def test_invalid_result_reports_preflight_failure_without_cell_evidence(self):
        analyzer = self._analyzer(
            classifier=lambda *_args: None,
            hit_detector=lambda *_args: False,
        )

        result = analyzer.analyze(
            before_image=None,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_reason, "preflight_failed")
        self.assertEqual(result.diagnostics["stage"], "preflight")
        self.assertEqual(result.affected_cells, frozenset())

    @staticmethod
    def _classifier_for(evidence):
        def classifier(_before_image, _after_image, screen_point):
            ratio, state = evidence.get(screen_point, (0.10, "miss"))
            return SimpleNamespace(
                changed_ratio=ratio,
                state=state,
                score=ratio,
            )

        return classifier

    def test_default_hit_detector_rejects_clean_reference_cell(self):
        reference_path = Path(__file__).parents[1] / "save_points" / "imgs" / "8.png"
        reference = cv2.imread(str(reference_path))

        self.assertIsNotNone(reference)
        self.assertFalse(red_scout_module._default_hit_detector(reference, (665, 318)))

    def test_change_prefilter_keeps_shifted_candidate_and_rejects_clean_cell(self):
        before = np.zeros((200, 240, 3), dtype=np.uint8)
        after = before.copy()
        cv2.rectangle(after, (78, 72), (142, 128), (255, 255, 255), cv2.FILLED)
        candidates = {(0, 0), (0, 1)}

        filtered = red_scout_module._prefilter_candidates_by_change_upper_bound(
            before_image=before,
            after_images=(after, after.copy(), after.copy()),
            points_by_cell={(0, 0): (100, 100), (0, 1): (200, 160)},
            candidates=candidates,
            minimum_change_threshold=0.45,
        )

        self.assertEqual(filtered, {(0, 0)})
        self.assertEqual(candidates, {(0, 0), (0, 1)})

    def test_first_scout_keeps_disconnected_changed_cells_from_multi_point_bomb(self):
        evidence = {
            (2, 2): (0.96, "miss"),
            (2, 3): (0.94, "hit"),
            (3, 2): (0.93, "miss"),
            (0, 0): (0.89, "hit"),
        }
        after_images = tuple(np.ones((8, 8, 3), dtype=np.uint8) for _ in range(3))
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: image is not self.BEFORE_IMAGE and point == (2, 3),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=after_images,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
            excluded_cells=frozenset(),
            learned_footprint=None,
        )

        self.assertTrue(result.valid)
        self.assertEqual(
            result.affected_cells,
            frozenset({(2, 2), (2, 3), (3, 2)}),
        )
        self.assertEqual(result.hit_cells, frozenset({(2, 3)}))
        self.assertEqual(result.miss_cells, frozenset({(2, 2), (3, 2)}))
        self.assertEqual(result.unknown_cells, frozenset())
        self.assertEqual(
            result.footprint.offsets,
            frozenset({(0, 0), (0, 1), (1, 0)}),
        )
        self.assertEqual(
            result.confidence_by_cell,
            {
                (2, 2): 0.96,
                (2, 3): 0.94,
                (3, 2): 0.93,
            },
        )

    def test_first_scout_ignores_cells_visible_before_red_bomb(self):
        evidence = {
            (2, 2): (0.96, "miss"),
            (2, 3): (0.94, "hit"),
            (0, 0): (0.93, "hit"),
        }
        before_image = np.zeros((8, 8, 3), dtype=np.uint8)
        after_images = tuple(np.ones((8, 8, 3), dtype=np.uint8) for _ in range(3))

        def hit_detector(image, point):
            if point == (0, 0):
                return True
            return image is not before_image and point == (2, 3)

        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=hit_detector,
        )

        result = analyzer.analyze(
            before_image=before_image,
            after_images=after_images,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
            excluded_cells=frozenset(),
            learned_footprint=None,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset({(2, 2), (2, 3)}))
        self.assertEqual(result.hit_cells, frozenset({(2, 3)}))
        self.assertNotIn((-2, -2), result.footprint.offsets)
        self.assertNotIn((0, 0), result.confidence_by_cell)

    def test_first_scout_is_valid_when_random_scatter_excludes_clicked_center(self):
        evidence = {
            (0, 0): (0.96, "miss"),
            (0, 4): (0.95, "miss"),
            (1, 3): (0.94, "miss"),
            (3, 1): (0.93, "miss"),
            (4, 0): (0.92, "miss"),
            (4, 4): (0.91, "miss"),
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda _image, _point: False,
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
            excluded_cells=set(),
            learned_footprint=None,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset(evidence))
        self.assertEqual(result.miss_cells, frozenset(evidence))
        self.assertIsNotNone(result.footprint)

    def test_first_footprint_is_invalid_with_only_center_changed(self):
        analyzer = self._analyzer(
            self._classifier_for({(2, 2): (0.90, "miss")}),
            hit_detector=lambda _image, _point: False,
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
            excluded_cells=set(),
            learned_footprint=None,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.affected_cells, frozenset({(2, 2)}))
        self.assertEqual(result.miss_cells, frozenset({(2, 2)}))
        self.assertIsNone(result.footprint)

    def test_learned_footprint_does_not_hide_changed_cells(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(
            offsets=frozenset({(-2, -2), (0, -1), (0, 0), (0, 1), (1, 0)})
        )
        evidence = {
            (0, 0): (0.95, "miss"),
            (2, 1): (0.44, "miss"),
            (2, 2): (0.96, "miss"),
            (1, 1): (0.99, "hit"),
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda _image, _point: False,
        )
        excluded_cells = {(1, 2)}
        excluded_snapshot = set(excluded_cells)

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=3,
            click_points=tuple(
                (row, col)
                for row in range(3)
                for col in range(3)
            ),
            center_cell=(2, 2),
            excluded_cells=excluded_cells,
            learned_footprint=footprint,
        )

        self.assertTrue(result.valid)
        self.assertEqual(
            result.affected_cells,
            frozenset({(0, 0), (2, 2)}),
        )
        self.assertEqual(result.miss_cells, frozenset({(0, 0), (2, 2)}))
        self.assertEqual(result.unknown_cells, frozenset())
        self.assertNotIn((1, 2), result.affected_cells)
        self.assertEqual(result.footprint, footprint)
        self.assertEqual(excluded_cells, excluded_snapshot)

    def test_learned_footprint_does_not_limit_result_scan(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        evidence = {
            (0, 0): (0.90, "hit"),
            (2, 2): (0.90, "miss"),
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: (
                image is not self.BEFORE_IMAGE and point == (0, 0)
            ),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=3,
            click_points=tuple(
                (row, col)
                for row in range(3)
                for col in range(3)
            ),
            center_cell=(2, 2),
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset({(0, 0), (2, 2)}))
        self.assertEqual(result.hit_cells, frozenset({(0, 0)}))
        self.assertEqual(result.miss_cells, frozenset({(2, 2)}))
        self.assertEqual(result.footprint, footprint)

    def test_hit_requires_detector_votes_in_at_least_two_frames(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        analyzer = self._analyzer(
            self._classifier_for({(2, 2): (0.90, "hit")}),
            hit_detector=lambda image, point: point == (2, 2)
            and id(image) in {id(self.AFTER_IMAGES[0]), id(self.AFTER_IMAGES[1])},
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        self.assertEqual(result.hit_cells, frozenset({(2, 2)}))
        self.assertEqual(result.miss_cells, frozenset())
        self.assertEqual(result.unknown_cells, frozenset())

    def test_new_stable_wreck_is_hit_even_when_whole_cell_change_is_small(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        target = (2, 2)
        analyzer = self._analyzer(
            self._classifier_for({target: (0.30, "hit")}),
            hit_detector=lambda image, point: (
                point == target
                and id(image) in {
                    id(self.AFTER_IMAGES[0]),
                    id(self.AFTER_IMAGES[1]),
                    id(self.AFTER_IMAGES[2]),
                }
            ),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=target,
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset({target}))
        self.assertEqual(result.hit_cells, frozenset({target}))
        self.assertEqual(result.miss_cells, frozenset())
        self.assertEqual(result.unknown_cells, frozenset())
        self.assertEqual(result.confidence_by_cell, {target: 0.30})

    def test_scout_rejects_ambiguous_more_than_six_strong_result_cells(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        center = (1, 1)
        stable_hit = (0, 0)
        evidence = {
            stable_hit: (0.20, "hit"),
            (0, 1): (0.99, "miss"),
            (0, 2): (0.98, "miss"),
            (1, 0): (0.97, "miss"),
            center: (0.50, "miss"),
            (1, 2): (0.96, "miss"),
            (2, 0): (0.95, "miss"),
            (2, 1): (0.94, "miss"),
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: (
                point == stable_hit and image is not self.BEFORE_IMAGE
            ),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=3,
            click_points=tuple(
                (row, col)
                for row in range(3)
                for col in range(3)
            ),
            center_cell=center,
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.affected_cells, frozenset())
        self.assertEqual(result.hit_cells, frozenset())
        self.assertEqual(result.miss_cells, frozenset())
        self.assertEqual(result.unknown_cells, frozenset())
        self.assertEqual(result.confidence_by_cell, {})
        self.assertEqual(result.invalid_reason, "too_many_strong_cells")
        self.assertEqual(result.diagnostics["stage"], "limit_strong_cells")
        self.assertEqual(
            set(result.diagnostics["raw_stable_hits"]),
            {stable_hit},
        )
        self.assertEqual(len(result.diagnostics["affected_before_limit"]), 7)

    def test_scout_keeps_only_six_strong_results_and_discards_moderate_noise(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        center = (1, 1)
        stable_hit = (0, 0)
        expected = {stable_hit, center, (0, 1), (0, 2), (1, 0), (1, 2)}
        evidence = {
            stable_hit: (0.20, "hit"),
            (0, 1): (0.99, "miss"),
            (0, 2): (0.98, "miss"),
            (1, 0): (0.97, "miss"),
            center: (0.96, "miss"),
            (1, 2): (0.95, "miss"),
            (2, 0): (0.70, "miss"),
            (2, 1): (0.60, "miss"),
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: (
                point == stable_hit and image is not self.BEFORE_IMAGE
            ),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=3,
            click_points=tuple(
                (row, col)
                for row in range(3)
                for col in range(3)
            ),
            center_cell=center,
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset(expected))
        self.assertEqual(result.hit_cells, frozenset({stable_hit}))
        self.assertEqual(result.miss_cells, frozenset(expected - {stable_hit}))
        self.assertEqual(result.unknown_cells, frozenset())

    def test_completed_submarine_body_collapses_to_one_attackable_hit_cell(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        submarine = {(2, 1), (2, 2), (2, 3)}
        center = (2, 2)
        misses = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}
        evidence = {
            **{cell: (0.95, "hit") for cell in submarine},
            **{cell: (0.96, "miss") for cell in misses},
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: (
                point in submarine and image is not self.BEFORE_IMAGE
            ),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=center,
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset({center} | misses))
        self.assertEqual(result.hit_cells, frozenset({center}))
        self.assertEqual(result.miss_cells, frozenset(misses))
        self.assertEqual(result.unknown_cells, frozenset())

    def test_completed_submarine_hit_cells_are_not_readded_as_misses(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        submarine = {(2, 1), (2, 2), (2, 3)}
        misses = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}
        evidence = {
            **{cell: (0.95, "miss") for cell in submarine},
            **{cell: (0.96, "miss") for cell in misses},
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: (
                point in submarine and image is not self.BEFORE_IMAGE
            ),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(4, 4),
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        expected_hit = min(submarine)
        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset({expected_hit} | misses))
        self.assertEqual(result.hit_cells, frozenset({expected_hit}))
        self.assertEqual(result.miss_cells, frozenset(misses))
        self.assertEqual(result.unknown_cells, frozenset())

    def test_sidebar_completion_removes_surfaced_ship_visual_spill(self):
        fleet = (5, 4, 3, 2, 2)

        def make_sidebar_frame(*, completed_row: int | None = None):
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            active_bgr = cv2.cvtColor(
                np.uint8([[[26, 20, 250]]]),
                cv2.COLOR_HSV2BGR,
            )[0, 0]
            complete_bgr = cv2.cvtColor(
                np.uint8([[[99, 99, 108]]]),
                cv2.COLOR_HSV2BGR,
            )[0, 0]
            for row_index in range(len(fleet)):
                center_y = 275 + row_index * 35
                color = complete_bgr if row_index == completed_row else active_bgr
                image[center_y - 13:center_y + 14, 20:145] = color
            return image

        before_image = make_sidebar_frame()
        after_images = tuple(
            make_sidebar_frame(completed_row=1)
            for _ in range(4)
        )
        partial_ship = {(0, 3), (1, 3), (2, 3)}
        new_ship_hit = (3, 3)
        visual_spill_hit = (1, 2)
        ship_perimeter = {
            (0, 2), (0, 4),
            (1, 2), (1, 4),
            (2, 2), (2, 4),
            (3, 2), (3, 4),
            (4, 2), (4, 3), (4, 4),
        }
        outside_misses = {(0, 1), (4, 8), (7, 6), (9, 0)}
        evidence = {
            new_ship_hit: (0.95, "hit"),
            visual_spill_hit: (0.95, "miss"),
            **{cell: (0.96, "miss") for cell in ship_perimeter},
            **{cell: (0.97, "miss") for cell in outside_misses},
        }
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: (
                point in partial_ship
                or (
                    image is not before_image
                    and point in {new_ship_hit, visual_spill_hit}
                )
            ),
        )

        result = analyzer.analyze(
            before_image=before_image,
            after_images=after_images,
            grid_size=10,
            click_points=tuple(
                (row, col)
                for row in range(10)
                for col in range(10)
            ),
            center_cell=(5, 5),
            excluded_cells=set(),
            learned_footprint=None,
            submarine_lengths=fleet,
        )

        expected_fallback_miss = (0, 2)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.affected_cells), 6)
        self.assertEqual(result.hit_cells, frozenset({new_ship_hit}))
        self.assertEqual(
            result.miss_cells,
            frozenset(outside_misses | {expected_fallback_miss}),
        )
        self.assertNotIn(visual_spill_hit, result.affected_cells)
        self.assertEqual(result.unknown_cells, frozenset())

    def test_consistent_moderate_miss_fills_the_sixth_result_cell(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        stable_hit = (3, 1)
        strong_misses = {(0, 2), (8, 1), (8, 8), (9, 9)}
        moderate_miss = (0, 1)
        noise = (0, 0)
        evidence = {
            stable_hit: (0.86, "hit"),
            **{cell: (0.97, "miss") for cell in strong_misses},
            moderate_miss: (0.62, "miss"),
            noise: (0.55, "miss"),
        }
        after_images = tuple(
            np.ones((8, 8, 3), dtype=np.uint8)
            for _ in range(4)
        )
        analyzer = self._analyzer(
            self._classifier_for(evidence),
            hit_detector=lambda image, point: (
                point == stable_hit and image is not self.BEFORE_IMAGE
            ),
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=after_images,
            grid_size=10,
            click_points=tuple(
                (row, col)
                for row in range(10)
                for col in range(10)
            ),
            center_cell=(5, 5),
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        expected = {stable_hit, moderate_miss} | strong_misses
        self.assertTrue(result.valid)
        self.assertEqual(result.affected_cells, frozenset(expected))
        self.assertEqual(result.hit_cells, frozenset({stable_hit}))
        self.assertEqual(result.miss_cells, frozenset(expected - {stable_hit}))
        self.assertNotIn(noise, result.affected_cells)

    def test_single_positive_hit_frame_is_unknown_not_miss(self):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        footprint = footprint_type(offsets=frozenset({(0, 0)}))
        analyzer = self._analyzer(
            self._classifier_for({(2, 2): (0.90, "miss")}),
            hit_detector=lambda image, point: point == (2, 2)
            and image is self.AFTER_IMAGES[0],
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
            excluded_cells=set(),
            learned_footprint=footprint,
        )

        self.assertEqual(result.hit_cells, frozenset())
        self.assertEqual(result.miss_cells, frozenset())
        self.assertEqual(result.unknown_cells, frozenset({(2, 2)}))

    def test_excluded_cells_are_not_classified(self):
        classified_points = []

        def classifier(_before_image, _after_image, screen_point):
            classified_points.append(screen_point)
            ratio = 0.90 if screen_point == (2, 2) else 0.10
            return SimpleNamespace(
                changed_ratio=ratio,
                state="miss",
                score=ratio,
            )

        analyzer = self._analyzer(
            classifier,
            hit_detector=lambda _image, _point: False,
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
            excluded_cells={(2, 3)},
            learned_footprint=None,
        )

        self.assertFalse(result.valid)
        self.assertNotIn((2, 3), result.affected_cells)
        self.assertNotIn((2, 3), classified_points)
        self.assertEqual(len(classified_points), (self.GRID_SIZE**2 - 1) * 3)

    def test_malformed_grid_mapping_fails_closed(self):
        classifier_calls = []

        def classifier(*args):
            classifier_calls.append(args)
            return SimpleNamespace(changed_ratio=0.90, state="miss", score=0.90)

        analyzer = self._analyzer(
            classifier,
            hit_detector=lambda _image, _point: False,
        )

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=self.AFTER_IMAGES,
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS[:-1],
            center_cell=(2, 2),
            excluded_cells=set(),
            learned_footprint=None,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.affected_cells, frozenset())
        self.assertEqual(classifier_calls, [])

    def test_invalid_images_fail_before_callbacks(self):
        classifier_calls = []
        detector_calls = []

        def classifier(*args):
            classifier_calls.append(args)
            return SimpleNamespace(changed_ratio=0.9, state="miss")

        def detector(*args):
            detector_calls.append(args)
            return False

        analyzer = self._analyzer(classifier, detector)
        invalid_images = (
            None,
            np.empty((0, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 1), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.float32),
            "not-an-image",
        )
        for image in invalid_images:
            with self.subTest(image=image):
                result = analyzer.analyze(
                    before_image=image,
                    after_images=self.AFTER_IMAGES,
                    grid_size=self.GRID_SIZE,
                    click_points=self.CLICK_POINTS,
                    center_cell=(2, 2),
                )
                self.assertFalse(result.valid)
        self.assertEqual(classifier_calls, [])
        self.assertEqual(detector_calls, [])

        result = analyzer.analyze(
            before_image=self.BEFORE_IMAGE,
            after_images=(self.AFTER_IMAGES[0], None, self.AFTER_IMAGES[2]),
            grid_size=self.GRID_SIZE,
            click_points=self.CLICK_POINTS,
            center_cell=(2, 2),
        )
        self.assertFalse(result.valid)
        self.assertEqual(classifier_calls, [])
        self.assertEqual(detector_calls, [])

    def test_confidence_mapping_is_immutable_and_snapshotted(self):
        source = {(2, 2): 0.9, (2, 3): 0.8}
        result = red_scout_module.RedScoutResult(
            center_cell=(2, 2), affected_cells=frozenset(source),
            hit_cells=frozenset(), miss_cells=frozenset(source),
            unknown_cells=frozenset(), footprint=None, valid=True,
            confidence_by_cell=source,
        )
        source[(0, 0)] = 1.0
        self.assertNotIn((0, 0), result.confidence_by_cell)
        with self.assertRaises(TypeError):
            result.confidence_by_cell[(2, 2)] = 0.1


class RedScoutPlannerTest(unittest.TestCase):
    def _planner(self, grid_size=5):
        planner_type = getattr(red_scout_module, "RedScoutPlanner", None)
        self.assertIsNotNone(planner_type, "RedScoutPlanner must be public")
        return planner_type(grid_size)

    def _footprint(self, offsets):
        footprint_type = getattr(red_scout_module, "RedFootprint", None)
        self.assertIsNotNone(footprint_type, "RedFootprint must be public")
        return footprint_type(offsets=frozenset(offsets))

    def test_second_scout_moves_beyond_first_coverage(self):
        planner = self._planner(5)
        footprint = self._footprint({(0, 0), (0, 1), (1, 0)})
        covered_cells = {(2, 2), (2, 3), (3, 2)}
        cell_scores = {(0, 0): 50.0}
        covered_snapshot = set(covered_cells)
        scores_snapshot = dict(cell_scores)

        first = planner.choose_center(
            footprint=None,
            covered_cells=set(),
            known_cells=set(),
            cell_scores={},
        )
        second = planner.choose_center(
            footprint=footprint,
            covered_cells=covered_cells,
            known_cells=set(),
            cell_scores=cell_scores,
        )

        self.assertEqual(first, (2, 2))
        self.assertNotEqual(second, first)
        self.assertEqual(second, (0, 0))
        self.assertEqual(covered_cells, covered_snapshot)
        self.assertEqual(cell_scores, scores_snapshot)

    def test_invalid_first_result_still_moves_to_a_different_center(self):
        planner = self._planner(5)

        first = planner.choose_center(
            footprint=None,
            excluded_centers=set(),
        )
        second = planner.choose_center(
            footprint=None,
            excluded_centers={first},
        )
        third = planner.choose_center(
            footprint=None,
            excluded_centers={first, second},
        )

        self.assertEqual(first, (2, 2))
        self.assertEqual(len({first, second, third}), 3)
        self.assertNotEqual(second, first)
        self.assertNotEqual(third, first)

    def test_planner_breaks_equal_scores_in_row_major_order(self):
        planner = self._planner(5)
        footprint = self._footprint({(0, 0)})

        result = planner.choose_center(
            footprint=footprint,
            covered_cells=set(),
            known_cells=set(),
            cell_scores={},
        )

        self.assertEqual(result, (0, 0))

    def test_planner_returns_none_when_board_is_exhausted(self):
        planner = self._planner(2)
        footprint = self._footprint({(0, 0)})
        board = {(0, 0), (0, 1), (1, 0), (1, 1)}

        result = planner.choose_center(
            footprint=footprint,
            covered_cells=board,
            known_cells=set(),
            cell_scores={(0, 0): 1000.0},
        )

        self.assertIsNone(result)

    def test_planner_rejects_invalid_grid_size(self):
        planner_type = getattr(red_scout_module, "RedScoutPlanner", None)
        self.assertIsNotNone(planner_type, "RedScoutPlanner must be public")

        for grid_size in (0, -1, True):
            with self.subTest(grid_size=grid_size):
                with self.assertRaises(ValueError):
                    planner_type(grid_size)

    def test_planner_rejects_malformed_score_containers(self):
        planner = self._planner()
        footprint = self._footprint({(0, 0)})
        for scores in ("bad", 42, [(1, 2, 3)]):
            with self.subTest(scores=scores):
                self.assertIsNone(planner.choose_center(footprint, cell_scores=scores))

    def test_planner_rejects_score_iterable_raising_runtime_error(self):
        planner = self._planner()
        footprint = self._footprint({(0, 0)})

        class RuntimeErrorScores:
            def __iter__(self):
                raise RuntimeError("malformed scores")

        self.assertIsNone(
            planner.choose_center(
                footprint,
                cell_scores=RuntimeErrorScores(),
            )
        )
