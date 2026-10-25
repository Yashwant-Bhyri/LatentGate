from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from research.ditraj_feedback_adapter import (
    RedPathFeedback,
    UniPCX0Capture,
    boxes_from_mask,
    decode_wan_x0_hat,
    observe_red_frame,
    rasterize_like_ditraj,
)


class UniPCMultistepScheduler:
    """Small scheduler double with the pinned UniPC conversion contract."""

    def __init__(self):
        self.config = {"prediction_type": "flow_prediction", "predict_x0": True}
        self.step_index = 4
        self.timesteps = torch.arange(8)

    def convert_model_output(self, model_output, *, sample=None):
        return sample - 0.25 * model_output


class FeedbackAdapterTests(unittest.TestCase):
    def _mask(self) -> torch.Tensor:
        mask = torch.zeros((5, 1, 30, 52), dtype=torch.float32)
        for index, x1 in enumerate((22, 24, 27, 30, 32)):
            mask[index, 0, 12:18, x1 : x1 + 8] = 1.0
        return mask

    def test_mask_to_box_roundtrip_matches_baseline_floor_slices(self) -> None:
        mask = self._mask()
        boxes = boxes_from_mask(mask)
        rebuilt = torch.from_numpy(rasterize_like_ditraj(boxes, height=30, width=52))
        self.assertTrue(torch.equal(mask, rebuilt))

    def test_red_ensemble_finds_unclipped_full_frame_center(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.float32)
        yy, xx = np.ogrid[:120, :160]
        disk = (xx - 61) ** 2 + (yy - 47) ** 2 <= 23**2
        image[disk] = [1.0, 0.0, 0.0]
        result = observe_red_frame(image)
        self.assertTrue(result["scoreable"])
        self.assertEqual(result["valid_threshold_count"], 9)
        self.assertAlmostEqual(result["center_xy_normalized"][0], 61 / 160, delta=0.01)
        self.assertAlmostEqual(result["center_xy_normalized"][1], 47 / 120, delta=0.01)
        self.assertGreaterEqual(result["stability_heuristic"], 0.99)

    def test_unipc_capture_is_exact_step_scoped_and_restorable(self) -> None:
        scheduler = UniPCMultistepScheduler()
        original = scheduler.convert_model_output
        capture = UniPCX0Capture(scheduler, capture_step=4)
        sample = torch.ones((1, 2, 3))
        model_output = torch.full_like(sample, 2.0)
        capture.begin(4, scheduler.timesteps[4])
        converted = scheduler.convert_model_output(model_output, sample=sample)
        payload = capture.consume(4)
        torch.testing.assert_close(payload["x0_hat"], converted)
        self.assertEqual(payload["scheduler_step_index"], 4)
        self.assertEqual(payload["step_zero_based"], 4)
        capture.close()
        self.assertIs(scheduler.convert_model_output.__func__, original.__func__)

    def test_unipc_capture_fails_closed_on_wrong_sigma_index(self) -> None:
        scheduler = UniPCMultistepScheduler()
        capture = UniPCX0Capture(scheduler, capture_step=4)
        capture.begin(4, scheduler.timesteps[4])
        scheduler.step_index = 3
        with self.assertRaisesRegex(RuntimeError, "sigma index mismatch"):
            scheduler.convert_model_output(torch.ones(1), sample=torch.ones(1))
        capture.close()

    def test_wan_x0_inverse_normalization_and_video_layout(self) -> None:
        class FakeVAE:
            class Config:
                z_dim = 2
                latents_mean = [1.0, -2.0]
                latents_std = [2.0, 0.5]

            config = Config()

            def __init__(self):
                self.decoded_latents = None

            def parameters(self):
                return iter([torch.empty((), dtype=torch.float32)])

            def decode(self, latents, return_dict=False):
                self.decoded_latents = latents.detach().clone()
                return (torch.zeros((1, 3, 17, 2, 2), dtype=torch.float32),)

        class FakePipe:
            _execution_device = torch.device("cpu")
            vae = FakeVAE()

        normalized = torch.zeros((1, 2, 5, 1, 1), dtype=torch.float32)
        video = decode_wan_x0_hat(FakePipe(), normalized)
        torch.testing.assert_close(
            FakePipe.vae.decoded_latents[:, :, :, 0, 0],
            torch.tensor([[[1.0] * 5, [-2.0] * 5]]),
        )
        self.assertEqual(video.shape, (17, 2, 2, 3))
        np.testing.assert_allclose(video, 0.5)

    def test_step_four_feedback_moves_later_masks_toward_observed_residual(self) -> None:
        initial = self._mask()
        policy = RedPathFeedback(initial_bbox_mask=initial, gain=0.5)
        latent = torch.zeros((1, 16, 5, 30, 52))
        policy.capture = type(
            "CaptureStub",
            (),
            {"consume": lambda self, step: {"step_zero_based": step, "x0_hat": latent}},
        )()
        h, w = 120, 160
        centers = [(25.5 + index * 2.5) / 52 for index in range(5)]
        observed_centers = [centers[0], *(x - 0.10 for x in centers[1:])]

        def fake_decode(_pipe, _latents):
            video = np.zeros((17, h, w, 3), dtype=np.float32)
            yy, xx = np.ogrid[:h, :w]
            for frame in range(17):
                alpha = frame / 4.0
                lo = min(3, int(alpha))
                hi = min(4, lo + 1)
                t = min(1.0, alpha - lo)
                cx = (observed_centers[lo] * (1 - t) + observed_centers[hi] * t) * w
                cy = 0.5 * h
                disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= 23**2
                video[frame, disk] = [1.0, 0.0, 0.0]
            return video

        with patch("research.ditraj_feedback_adapter.decode_wan_x0_hat", fake_decode):
            updated = policy.after_scheduler_step(object(), 4, torch.tensor(1.0), initial)
        self.assertIsNotNone(updated)
        self.assertTrue(torch.equal(updated[0], initial[0]))
        self.assertGreater(float(policy.report["controller"]["applied_shift_xy"][1][0]), 0.0)
        self.assertGreater(policy.report["rasterization"]["changed_token_total"], 0)
        calibration = policy.report["start_anchor_calibration"]
        self.assertTrue(calibration["start_observation_trusted"])
        np.testing.assert_allclose(
            calibration["calibrated_observed_centers_xy_normalized"][0],
            [(policy.target_boxes[0][0] + policy.target_boxes[0][2]) / 2,
             (policy.target_boxes[0][1] + policy.target_boxes[0][3]) / 2],
        )
        self.assertEqual(len(policy.report["observer"]["all_17_frames"]), 17)
        self.assertEqual(policy.report["observer"]["controller_rgb_frames"], [0, 4, 8, 12, 16])

    def test_untrusted_start_observer_fails_closed_for_the_whole_update(self) -> None:
        initial = self._mask()
        policy = RedPathFeedback(initial_bbox_mask=initial, gain=0.5)
        latent = torch.zeros((1, 16, 5, 30, 52))
        policy.capture = type(
            "CaptureStub",
            (),
            {"consume": lambda self, step: {"step_zero_based": step, "x0_hat": latent}},
        )()

        with patch(
            "research.ditraj_feedback_adapter.decode_wan_x0_hat",
            lambda _pipe, _latents: np.zeros((17, 32, 48, 3), dtype=np.float32),
        ):
            updated = policy.after_scheduler_step(object(), 4, torch.tensor(1.0), initial)
        self.assertTrue(torch.equal(updated, initial))
        self.assertFalse(policy.report["start_anchor_calibration"]["start_observation_trusted"])
        self.assertEqual(policy.report["rasterization"]["changed_token_total"], 0)


if __name__ == "__main__":
    unittest.main()
