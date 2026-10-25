from __future__ import annotations

import unittest

import numpy as np

from research.box_residual_feedback import (
    FeedbackConfig,
    correct_path_boxes,
    rasterize_path_boxes,
    summarize_rasterized_updates,
)


class FeedbackBoxTests(unittest.TestCase):
    def test_signed_residual_is_bounded_and_start_box_stays_fixed(self) -> None:
        boxes = [
            [0.10, 0.40, 0.30, 0.60],
            [0.40, 0.40, 0.60, 0.60],
            [0.80, 0.40, 0.95, 0.60],
        ]
        observations = [[0.20, 0.50], [0.45, 0.50], [0.75, 0.50]]
        report = correct_path_boxes(
            boxes,
            observations,
            [1.0, 1.0, 0.8],
            config=FeedbackConfig(gain=0.5, minimum_stability=0.75, max_shift_xy=(0.05, 0.05)),
        )
        steered = np.asarray(report["steered_boxes_xyxy"])
        np.testing.assert_allclose(steered[0], boxes[0])
        self.assertEqual(report["trusted_update"], [False, True, True])
        self.assertAlmostEqual(report["applied_shift_xy"][1][0], 0.025)
        self.assertAlmostEqual(report["applied_shift_xy"][2][0], 0.05)
        self.assertTrue(np.all(steered >= 0) and np.all(steered <= 1))
        np.testing.assert_allclose(steered[:, 2:] - steered[:, :2], np.asarray(boxes)[:, 2:] - np.asarray(boxes)[:, :2])

    def test_low_stability_and_missing_observations_leave_boxes_unchanged(self) -> None:
        boxes = [[0.2, 0.3, 0.4, 0.5], [0.5, 0.3, 0.7, 0.5]]
        report = correct_path_boxes(
            boxes,
            [None, [0.55, 0.4]],
            [0.0, 0.2],
            config=FeedbackConfig(locked_box_indices=()),
        )
        np.testing.assert_allclose(report["steered_boxes_xyxy"], boxes)
        self.assertEqual(report["trusted_update"], [False, False])

    def test_edge_clamp_preserves_box_size(self) -> None:
        box = [0.80, 0.20, 1.00, 0.40]
        report = correct_path_boxes(
            [box],
            [[0.20, 0.30]],
            [1.0],
            config=FeedbackConfig(locked_box_indices=(), gain=1.0, max_shift_xy=(0.5, 0.5)),
        )
        shifted = report["steered_boxes_xyxy"][0]
        self.assertAlmostEqual(shifted[2], 1.0)
        self.assertAlmostEqual(shifted[2] - shifted[0], 0.20)
        self.assertAlmostEqual(shifted[3] - shifted[1], 0.20)

    def test_rasterize_uses_nonempty_half_open_pixel_slices(self) -> None:
        mask = rasterize_path_boxes(
            [[0.25, 0.20, 0.50, 0.50], [0.90, 0.90, 1.00, 1.00]],
            height=10,
            width=20,
        )
        self.assertEqual(mask.shape, (2, 10, 20))
        self.assertEqual(int(mask[0].sum()), 15)
        self.assertEqual(int(mask[1].sum()), 2)
        self.assertTrue(mask[0, 2:5, 5:10].all())

    def test_rasterized_update_log_exposes_subtoken_noop_and_effective_shift(self) -> None:
        target = [[0.20, 0.30, 0.40, 0.50]]
        subtoken = [[0.201, 0.30, 0.401, 0.50]]
        one_token = [[0.20 + 1 / 52, 0.30, 0.40 + 1 / 52, 0.50]]
        no_op, moved = summarize_rasterized_updates(
            target + target,
            subtoken + one_token,
            height=30,
            width=52,
        )
        self.assertEqual(no_op["changed_token_count"], 0)
        self.assertEqual(moved["changed_token_count"], 12)
        self.assertGreater(moved["center_shift_tokens_x"], 0)

    def test_bad_boxes_and_observation_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            correct_path_boxes([[0.6, 0.2, 0.4, 0.5]], [[0.5, 0.35]], [1.0])
        with self.assertRaises(ValueError):
            correct_path_boxes([[0.2, 0.2, 0.4, 0.4]], [[0.5]], [1.0])
        with self.assertRaises(ValueError):
            rasterize_path_boxes([[0.2, 0.2, 0.4, 0.4]], height=0, width=10)


if __name__ == "__main__":
    unittest.main()
