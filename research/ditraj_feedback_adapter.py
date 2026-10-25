"""Opt-in x0-hat feedback adapter for the pinned DiTraj Wan pipeline.

This module is deliberately separate from the published/open-loop runner. It
captures UniPC's exact converted clean prediction at one selected step, decodes
it with Wan's VAE, observes a controlled red object, and returns a bounded
foreground-box update for the remaining spatial-guidance steps.

The red observer and stability score are suitable only for the red-ball pilot.
The score is a threshold-agreement heuristic, not a calibrated probability.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # supports both `research.*` imports and flat notebook staging
    from .box_residual_feedback import FeedbackConfig, correct_path_boxes
except ImportError:
    from box_residual_feedback import FeedbackConfig, correct_path_boxes


PINNED_DITRAJ_COMMIT = "6fa8e613c9e7c40ec2db456ab2ba34367a076886"
FEEDBACK_STEP = 4
OBSERVATION_FRAMES = (0, 4, 8, 12, 16)
THRESHOLDS = tuple((r, gb) for r in (130, 150, 170) for gb in (100, 125, 150))
MIN_RED_PIXELS = 300
REFERENCE_FRAME_AREA = 832 * 480


def minimum_red_pixels(height: int, width: int) -> int:
    """Scale the canonical 300-pixel floor across fixture/output resolutions."""
    return max(8, int(round(MIN_RED_PIXELS * height * width / REFERENCE_FRAME_AREA)))


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} source marker, found {count}")
    return source.replace(old, new, 1)


def patch_pinned_pipeline(pipe_path: Path) -> str:
    """Add fail-closed feedback hooks to the already pinned DiTraj pipe source.

    The caller must independently verify the git commit and apply the baseline
    runner's documented performance patches first. This function also works on
    the unoptimized pinned source for local patch-shape checks.
    """
    source = pipe_path.read_text()
    if "latentgate_feedback.after_scheduler_step" in source:
        required = (
            "latentgate_feedback.before_scheduler_step",
            "latentgate_feedback.after_scheduler_step",
            "segment_mask = attention_kwargs['encoder_attention_mask'].flatten()",
            "updated_bbox_mask is not None and not torch.equal(updated_bbox_mask, bbox_mask)",
        )
        if not all(marker in source for marker in required):
            raise RuntimeError("Partial feedback patch detected in module/pipe.py")
        compile(source, str(pipe_path), "exec")
        return hashlib.sha256(pipe_path.read_bytes()).hexdigest()

    source = _replace_once(
        source,
        "        mask_step = attention_kwargs['mask_step']\n",
        "        mask_step = attention_kwargs['mask_step']\n"
        "        latentgate_feedback = attention_kwargs.pop('latentgate_feedback', None)\n"
        "        segment_mask = attention_kwargs['encoder_attention_mask'].flatten()\n"
        "        if latentgate_feedback is not None and not (\n"
        "            hasattr(latentgate_feedback, 'before_scheduler_step')\n"
        "            and hasattr(latentgate_feedback, 'after_scheduler_step')\n"
        "        ):\n"
        "            raise TypeError(\"latentgate_feedback must implement the pinned feedback adapter interface\")\n",
        "feedback-controller setup",
    )
    source = _replace_once(
        source,
        "self.create_cross_attention_mask(bbox_mask, attention_kwargs['encoder_attention_mask'].flatten())",
        "self.create_cross_attention_mask(bbox_mask, segment_mask)",
        "cross-attention segment mask",
    )
    source = _replace_once(
        source,
        "                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]\n",
        "                if latentgate_feedback is not None:\n"
        "                    latentgate_feedback.before_scheduler_step(self.scheduler, i)\n"
        "                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]\n"
        "                if latentgate_feedback is not None:\n"
        "                    updated_bbox_mask = latentgate_feedback.after_scheduler_step(\n"
        "                        self, i, t, bbox_mask\n"
        "                    )\n"
        "                    if updated_bbox_mask is not None and not torch.equal(updated_bbox_mask, bbox_mask):\n"
        "                        bbox_mask = updated_bbox_mask\n"
        "                        cross_attention_mask = self.create_cross_attention_mask(bbox_mask, segment_mask)\n",
        "scheduler feedback hook",
    )
    compile(source, str(pipe_path), "exec")
    pipe_path.write_text(source)
    return hashlib.sha256(pipe_path.read_bytes()).hexdigest()


def boxes_from_mask(mask: Any) -> list[list[float]]:
    """Recover normalized xyxy rectangle boxes from DiTraj's [T,1,H,W] mask."""
    array = mask.detach().to(device="cpu").numpy()
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"Expected bbox mask [T,1,H,W], got {tuple(array.shape)}")
    _, _, height, width = array.shape
    boxes: list[list[float]] = []
    for index, frame in enumerate(array[:, 0] > 0.5):
        ys, xs = np.nonzero(frame)
        if not len(xs):
            raise ValueError(f"bbox mask frame {index} is empty")
        # The pilot uses solid rectangles. Refuse malformed masks instead of
        # expanding a disconnected foreground into an unintended box.
        crop = frame[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        if not crop.all() or int(frame.sum()) != int(crop.size):
            raise ValueError(f"bbox mask frame {index} is not one filled rectangle")
        boxes.append([
            float(xs.min()) / width,
            float(ys.min()) / height,
            float(xs.max() + 1) / width,
            float(ys.max() + 1) / height,
        ])
    return boxes


def rasterize_like_ditraj(boxes_xyxy: Sequence[Sequence[float]], *, height: int, width: int) -> np.ndarray:
    """Rasterize using DiTraj's int/floor lower and upper endpoint semantics."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("boxes_xyxy must be finite [T,4]")
    if np.any(boxes < 0.0) or np.any(boxes > 1.0):
        raise ValueError("box coordinates must lie in [0,1]")
    mask = np.zeros((len(boxes), 1, height, width), dtype=np.float32)
    for frame, (x1, y1, x2, y2) in enumerate(boxes):
        # Keep the baseline's int() behavior (floor for nonnegative inputs).
        # The tiny epsilon only neutralizes float roundoff when a box came
        # back from an integer token boundary (e.g. 17/30*30 == 16.999999...).
        left = min(width - 1, max(0, int(np.floor(x1 * width + 1e-9))))
        top = min(height - 1, max(0, int(np.floor(y1 * height + 1e-9))))
        right = min(width, max(left + 1, int(np.floor(x2 * width + 1e-9))))
        bottom = min(height, max(top + 1, int(np.floor(y2 * height + 1e-9))))
        mask[frame, 0, top:bottom, left:right] = 1.0
    return mask


def mask_change_report(before: Any, after: Any) -> list[dict[str, float | int]]:
    """Measure the actual discrete intervention on DiTraj's latent token grid."""
    old = before.detach().to(device="cpu").numpy()[:, 0] > 0.5
    new = after.detach().to(device="cpu").numpy()[:, 0] > 0.5
    if old.shape != new.shape:
        raise ValueError("before/after masks must have the same shape")
    records: list[dict[str, float | int]] = []
    for frame, (a, b) in enumerate(zip(old, new, strict=True)):
        ay, ax = np.nonzero(a)
        by, bx = np.nonzero(b)
        intersection = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        records.append({
            "latent_frame": frame,
            "changed_token_count": int(np.logical_xor(a, b).sum()),
            "target_occupied_tokens": int(a.sum()),
            "updated_occupied_tokens": int(b.sum()),
            "mask_iou": intersection / union if union else 0.0,
            "center_shift_tokens_xy": [
                float(bx.mean() - ax.mean()), float(by.mean() - ay.mean())
            ],
        })
    return records


def observe_red_frame(rgb_float: np.ndarray) -> dict[str, Any]:
    """Return a full-frame nine-threshold red centroid and agreement heuristic."""
    if rgb_float.ndim != 3 or rgb_float.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {rgb_float.shape}")
    height, width = rgb_float.shape[:2]
    rgb = (np.clip(rgb_float, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    minimum_pixels = minimum_red_pixels(height, width)
    centers: list[tuple[float, float]] = []
    areas: list[int] = []
    for red_min, gb_max in THRESHOLDS:
        mask = (rgb[:, :, 0] > red_min) & (rgb[:, :, 1] < gb_max) & (rgb[:, :, 2] < gb_max)
        ys, xs = np.nonzero(mask)
        if len(xs) >= minimum_pixels:
            centers.append((float(xs.mean()), float(ys.mean())))
            areas.append(len(xs))
    if not centers:
        return {
            "scoreable": False,
            "valid_threshold_count": 0,
            "threshold_count": len(THRESHOLDS),
            "minimum_pixels_per_threshold": minimum_pixels,
            "stability_heuristic": 0.0,
            "center_xy_normalized": None,
            "center_spread_frame_diagonals": None,
        }
    cx = statistics.median(point[0] for point in centers)
    cy = statistics.median(point[1] for point in centers)
    diagonal = math.hypot(width, height)
    spread = statistics.median(math.hypot(x - cx, y - cy) for x, y in centers) / diagonal
    coverage = len(centers) / len(THRESHOLDS)
    stability = coverage * max(0.0, 1.0 - spread / 0.02)
    mean_area = statistics.mean(areas)
    return {
        "scoreable": True,
        "valid_threshold_count": len(centers),
        "threshold_count": len(THRESHOLDS),
        "minimum_pixels_per_threshold": minimum_pixels,
        "stability_heuristic": stability,
        "center_xy_normalized": [cx / width, cy / height],
        "center_spread_frame_diagonals": spread,
        "area_cv_across_thresholds": statistics.pstdev(areas) / mean_area if mean_area else None,
    }


def decode_wan_x0_hat(pipe: Any, normalized_latents: Any) -> np.ndarray:
    """Inverse-normalize and decode Wan x0_hat into [frames,H,W,RGB] in [0,1]."""
    import torch

    vae = pipe.vae
    device = pipe._execution_device
    vae_dtype = next(vae.parameters()).dtype
    latents = normalized_latents.to(device=device, dtype=torch.float32)
    channels = int(vae.config.z_dim)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(1, channels, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device).view(1, channels, 1, 1, 1)
    raw_latents = (latents * std + mean).to(dtype=vae_dtype)
    with torch.inference_mode():
        decoded = vae.decode(raw_latents, return_dict=False)[0]
    rgb = ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 0.5)
    return rgb.permute(0, 2, 3, 4, 1)[0].cpu().numpy()


class UniPCX0Capture:
    """Capture the exact scheduler conversion for one UniPC denoising step."""

    def __init__(self, scheduler: Any, capture_step: int):
        if type(scheduler).__name__ != "UniPCMultistepScheduler":
            raise RuntimeError(f"Expected UniPCMultistepScheduler, got {type(scheduler).__name__}")
        config = dict(scheduler.config)
        if config.get("prediction_type") != "flow_prediction" or config.get("predict_x0") is not True:
            raise RuntimeError(
                "Refusing feedback: scheduler must use flow_prediction with predict_x0=True; "
                f"got prediction_type={config.get('prediction_type')!r}, predict_x0={config.get('predict_x0')!r}"
            )
        if not callable(getattr(scheduler, "convert_model_output", None)):
            raise RuntimeError("Pinned UniPC scheduler has no convert_model_output method")
        self.scheduler = scheduler
        self.capture_step = int(capture_step)
        self.active_step: int | None = None
        self.captured: dict[str, Any] | None = None
        self.original_convert = scheduler.convert_model_output
        self._closed = False

        def convert_with_capture(model_output: Any, *args: Any, **kwargs: Any):
            converted = self.original_convert(model_output, *args, **kwargs)
            step = self.active_step
            if step == self.capture_step:
                sigma_index = getattr(self.scheduler, "step_index", None)
                if sigma_index is None or int(sigma_index) != step:
                    raise RuntimeError(
                        f"UniPC sigma index mismatch: requested step {step}, step_index={sigma_index!r}"
                    )
                sample = kwargs.get("sample")
                if sample is None and args:
                    sample = args[0]
                if sample is None:
                    raise RuntimeError("UniPC conversion did not expose the pre-step sample")
                self.captured = {
                    "step_zero_based": step,
                    "timestep": self.active_timestep,
                    "scheduler_step_index": int(sigma_index),
                    "x0_hat": converted.detach().clone(),
                    "shape": list(converted.shape),
                    "dtype": str(converted.dtype),
                    "scheduler_config": config,
                    "formula": "exact Diffusers UniPC convert_model_output result",
                }
            return converted

        self._wrapped_convert = convert_with_capture
        scheduler.convert_model_output = convert_with_capture
        self.active_timestep: float | None = None

    def begin(self, step: int, timestep: Any) -> None:
        if self.active_step is not None:
            raise RuntimeError("UniPC capture was armed while another scheduler step was active")
        if step == self.capture_step:
            self.captured = None
            self.active_step = step
            self.active_timestep = float(timestep.item()) if hasattr(timestep, "item") else float(timestep)

    def consume(self, step: int) -> dict[str, Any]:
        if step != self.capture_step or self.active_step != step:
            raise RuntimeError(f"UniPC capture consumed outside selected step {self.capture_step}")
        self.active_step = None
        self.active_timestep = None
        if self.captured is None:
            raise RuntimeError("Pinned UniPC did not call convert_model_output at the feedback step")
        payload = self.captured
        self.captured = None
        return payload

    def close(self) -> None:
        if self._closed:
            return
        self.scheduler.convert_model_output = self.original_convert
        self._closed = True


class RedPathFeedback:
    """Single-read, bounded foreground-box feedback for a five-frame path."""

    def __init__(
        self,
        *,
        initial_bbox_mask: Any,
        observe_step: int = FEEDBACK_STEP,
        gain: float = 0.5,
        minimum_stability: float = 0.65,
    ):
        self.observe_step = int(observe_step)
        self.target_boxes = boxes_from_mask(initial_bbox_mask)
        self.target_mask = initial_bbox_mask.detach().clone()
        _, _, self.mask_h, self.mask_w = self.target_mask.shape
        if len(self.target_boxes) != len(OBSERVATION_FRAMES):
            raise ValueError(
                f"Expected {len(OBSERVATION_FRAMES)} latent-time boxes, got {len(self.target_boxes)}"
            )
        # 0.65 is an in-sample development cutoff on two correlated Kubric
        # red-ball paths. It is neither calibrated nor validated for gating;
        # generated-preview observer checks must freeze a cutoff before holdout.
        self.config = FeedbackConfig(
            gain=gain,
            minimum_stability=minimum_stability,
            max_shift_xy=(1.0 / self.mask_w, 1.0 / self.mask_h),
            locked_box_indices=(0,),
        )
        self.capture: UniPCX0Capture | None = None
        self.report: dict[str, Any] | None = None

    def before_scheduler_step(self, scheduler: Any, step: int) -> None:
        if step == self.observe_step:
            if self.capture is None:
                self.capture = UniPCX0Capture(scheduler, self.observe_step)
            self.capture.begin(step, scheduler.timesteps[step])

    def after_scheduler_step(self, pipe: Any, step: int, timestep: Any, current_bbox_mask: Any):
        if step != self.observe_step:
            return None
        if self.capture is None:
            raise RuntimeError("Feedback observer reached without an installed UniPC capture")
        payload = self.capture.consume(step)
        try:
            preview = decode_wan_x0_hat(pipe, payload["x0_hat"])
        finally:
            # The VAE decode no longer needs the full x0 tensor; release it
            # before the subsequent transformer steps on a small GPU.
            del payload["x0_hat"]
        if preview.shape[0] != 17:
            raise RuntimeError(f"Expected 17 decoded x0_hat frames, got {preview.shape[0]}")
        all_frame_observations = [observe_red_frame(frame) for frame in preview]
        selected = [all_frame_observations[index] for index in OBSERVATION_FRAMES]
        raw_centers = [row["center_xy_normalized"] for row in selected]
        stability = [row["stability_heuristic"] for row in selected]
        target_centers = [
            [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            for box in self.target_boxes
        ]
        anchor_is_trusted = (
            raw_centers[0] is not None
            and stability[0] >= self.config.minimum_stability
        )
        anchor_offset = None
        calibrated_centers: list[list[float] | None]
        if anchor_is_trusted:
            # The first box is locked. Use its observed-vs-commanded offset to
            # cancel static color-centroid bias, then steer relative trajectory
            # error. If the start observation is unreliable, fail closed.
            anchor_offset = [
                float(raw_centers[0][0] - target_centers[0][0]),
                float(raw_centers[0][1] - target_centers[0][1]),
            ]
            calibrated_centers = []
            for center in raw_centers:
                if center is None:
                    calibrated_centers.append(None)
                    continue
                corrected = [center[0] - anchor_offset[0], center[1] - anchor_offset[1]]
                calibrated_centers.append(
                    corrected if all(0.0 <= coordinate <= 1.0 for coordinate in corrected) else None
                )
        else:
            calibrated_centers = [None] * len(raw_centers)
        update = correct_path_boxes(
            self.target_boxes,
            calibrated_centers,
            stability,
            config=self.config,
        )
        active_boxes = update["steered_boxes_xyxy"]
        new_mask_array = rasterize_like_ditraj(active_boxes, height=self.mask_h, width=self.mask_w)
        import torch

        updated_mask = torch.as_tensor(
            new_mask_array,
            dtype=current_bbox_mask.dtype,
            device=current_bbox_mask.device,
        )
        mask_report = mask_change_report(self.target_mask, updated_mask)
        self.report = {
            "observe_step_zero_based": int(step),
            "timestep": float(timestep.item()) if hasattr(timestep, "item") else float(timestep),
            "x0_hat_capture": {key: value for key, value in payload.items() if key != "x0_hat"},
            "observer": {
                "family": "full-frame red threshold ensemble",
                "red_gt": [130, 150, 170],
                "green_blue_lt": [100, 125, 150],
                "minimum_pixels_per_threshold_at_832x480": MIN_RED_PIXELS,
                "minimum_pixel_area_scales_with_frame_area": True,
                "all_17_frames": all_frame_observations,
                "controller_rgb_frames": list(OBSERVATION_FRAMES),
                "confidence_semantics": "threshold agreement heuristic, not calibrated probability",
            },
            "controller": update,
            "start_anchor_calibration": {
                "method": "subtract the observed-minus-commanded start offset from all later centers; controls relative path error",
                "start_observation_trusted": bool(anchor_is_trusted),
                "minimum_stability": self.config.minimum_stability,
                "observed_start_offset_xy_normalized": anchor_offset,
                "raw_observed_centers_xy_normalized": raw_centers,
                "calibrated_observed_centers_xy_normalized": calibrated_centers,
                "fallback": "if the first RGB observation is missing or below stability threshold, all boxes remain unchanged",
            },
            "rasterization": {
                "semantics": "DiTraj int() floor/floor endpoint slices",
                "latent_grid_wh": [self.mask_w, self.mask_h],
                "target_vs_updated_per_latent_frame": mask_report,
                "changed_token_total": sum(int(row["changed_token_count"]) for row in mask_report),
            },
        }
        del preview, all_frame_observations
        return updated_mask

    def close(self) -> None:
        if self.capture is not None:
            self.capture.close()
