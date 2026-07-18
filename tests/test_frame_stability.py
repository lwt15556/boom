import unittest

import cv2
import numpy as np

from utils.frame_stability import (
    analyze_stable_hit,
    build_stable_after_image,
    local_motion_contrast,
    register_translation,
    stable_hit_is_suspect,
)


class FrameStabilityTest(unittest.TestCase):
    @staticmethod
    def textured_image() -> np.ndarray:
        rng = np.random.default_rng(20260718)
        image = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
        return cv2.GaussianBlur(image, (5, 5), 0)

    def test_translation_registration_reduces_alignment_error(self):
        reference = self.textured_image()
        shifted = cv2.warpAffine(
            reference,
            np.float32([[1, 0, 4], [0, 1, -3]]),
            (reference.shape[1], reference.shape[0]),
            borderMode=cv2.BORDER_REFLECT,
        )

        registered, registration = register_translation(reference, shifted)

        before_error = np.mean(np.abs(reference.astype(float) - shifted.astype(float)))
        after_error = np.mean(np.abs(reference.astype(float) - registered.astype(float)))
        self.assertTrue(registration.accepted)
        self.assertAlmostEqual(registration.dx, 4.0, delta=0.6)
        self.assertAlmostEqual(registration.dy, -3.0, delta=0.6)
        self.assertLess(after_error, before_error * 0.35)

    def test_registration_rejects_implausibly_large_motion(self):
        reference = self.textured_image()
        shifted = np.roll(reference, 25, axis=1)

        registered, registration = register_translation(
            reference,
            shifted,
            max_translation=8.0,
        )

        self.assertFalse(registration.accepted)
        np.testing.assert_array_equal(registered, shifted)

    def test_temporal_median_removes_single_frame_flash(self):
        baseline = np.full((80, 100, 3), 40, dtype=np.uint8)
        frames = [baseline.copy() for _ in range(3)]
        frames[0][25:45, 30:50] = 255

        stable, registrations = build_stable_after_image(baseline, frames)

        np.testing.assert_array_equal(stable, baseline)
        self.assertEqual(len(registrations), 3)

    def test_local_motion_contrast_rewards_centered_change(self):
        before = np.zeros((140, 180, 3), dtype=np.uint8)
        centered = before.copy()
        centered[60:80, 80:100] = 220
        global_change = np.full_like(before, 80)

        centered_metrics = local_motion_contrast(before, centered, (90, 70), 70, 50)
        global_metrics = local_motion_contrast(before, global_change, (90, 70), 70, 50)

        self.assertGreater(centered_metrics.contrast, 0.1)
        self.assertLess(global_metrics.contrast, 0.05)
        self.assertGreater(centered_metrics.contrast, global_metrics.contrast)

    def test_stable_analysis_classifies_median_frame_once(self):
        baseline = np.zeros((80, 100, 3), dtype=np.uint8)
        persistent = baseline.copy()
        persistent[35:45, 45:55] = 180
        flash = persistent.copy()
        flash[5:20, 5:20] = 255
        classified_images = []

        def classifier(before, after, center, config=None):
            classified_images.append((before, after, center, config))
            return type(
                "Result",
                (),
                {"state": "hit", "score": 0.9, "refined_center": center},
            )()

        analysis = analyze_stable_hit(
            baseline,
            [flash, persistent, persistent],
            (50, 40),
            classifier=classifier,
        )

        self.assertEqual(len(classified_images), 1)
        np.testing.assert_array_equal(classified_images[0][1], persistent)
        self.assertEqual(analysis.result.state, "hit")
        self.assertGreater(analysis.motion.contrast, 0.0)

    def test_stable_suspect_requires_local_motion_and_reliable_registration(self):
        result = type("Result", (), {"state": "hit", "score": 0.9})()
        accepted = type("Registration", (), {"accepted": True})()
        rejected = type("Registration", (), {"accepted": False})()
        local_motion = type(
            "Motion", (), {"inner_ratio": 0.2, "contrast": 0.12}
        )()
        global_motion = type(
            "Motion", (), {"inner_ratio": 0.8, "contrast": 0.0}
        )()

        self.assertTrue(
            stable_hit_is_suspect(
                type(
                    "Analysis",
                    (),
                    {
                        "result": result,
                        "motion": local_motion,
                        "registrations": (accepted, accepted, rejected),
                    },
                )()
            )
        )
        self.assertFalse(
            stable_hit_is_suspect(
                type(
                    "Analysis",
                    (),
                    {
                        "result": result,
                        "motion": global_motion,
                        "registrations": (accepted, accepted, accepted),
                    },
                )()
            )
        )
        self.assertFalse(
            stable_hit_is_suspect(
                type(
                    "Analysis",
                    (),
                    {
                        "result": result,
                        "motion": local_motion,
                        "registrations": (accepted, rejected, rejected),
                    },
                )()
            )
        )


if __name__ == "__main__":
    unittest.main()
