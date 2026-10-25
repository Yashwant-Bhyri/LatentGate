"""Bounded object-state residual controller for DiTraj-style path boxes.

This is a pure, isolated policy component. It is not yet wired into the Wan or
DiTraj denoising loop. Coordinates are normalized ``xyxy`` image boxes and
observed object centers; a confidence score is used only as a threshold gate,
not as a calibrated probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeedbackConfig:
    gain: float = 0.5
    minimum_stability: float = 0.75
    # One-cell cap for the current 52x30 Wan latent guidance grid.
    max_shift_xy: tuple[float, float] = (1.0 / 52.0, 1.0 / 30.0)
    locked_box_indices: tuple[int, ...] = (0,)

    def validate(self) -> None:
        if not 0.0 <= self.gain <= 1.0:
            raise ValueError("gain must be in [0,1]")
        if not 0.0 <= self.minimum_stability <= 1.0:
            raise ValueError("minimum_stability must be in [0,1]")
        if len(self.max_shift_xy) != 2 or any(not 0.0 <= x <= 1.0 for x in self.max_shift_xy):
            raise ValueError("max_shift_xy must contain normalized nonnegative x/y limits")


def correct_path_boxes(
    target_boxes_xyxy: Sequence[Sequence[float]],
    observed_centers_xy: Sequence[Sequence[float] | None],
    observer_stability: Sequence[float],
    *,
    config: FeedbackConfig = FeedbackConfig(),
) -> dict[str, object]:
    """Move trusted target boxes toward the signed residual from current x0-hat.

    For each unlocked frame, ``delta = clip(gain * (target_center - observed),
    +/- max_shift_xy)`` if the observation is present and its stability score
    passes the configured threshold. The box size is preserved and the shifted
    box is clamped inside the frame. Untrusted/missing observations get zero
    correction; the initial box is locked by default.

    ``observer_stability`` should be a diagnostic score such as threshold
    ensemble agreement. It is not interpreted as a probability.
    """
    config.validate()
    boxes = np.asarray(target_boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] == 0:
        raise ValueError("target_boxes_xyxy must have shape [T,4]")
    if not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 1):
        raise ValueError("target boxes must be finite and normalized to [0,1]")
    if np.any(boxes[:, 0] >= boxes[:, 2]) or np.any(boxes[:, 1] >= boxes[:, 3]):
        raise ValueError("target boxes must have positive width and height")
    if len(observed_centers_xy) != len(boxes) or len(observer_stability) != len(boxes):
        raise ValueError("Need one observation and stability score per target box")

    stability = np.asarray(observer_stability, dtype=np.float64)
    if not np.isfinite(stability).all() or np.any(stability < 0) or np.any(stability > 1):
        raise ValueError("observer_stability values must be finite and in [0,1]")

    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
    observed = np.full_like(centers, np.nan)
    for index, center in enumerate(observed_centers_xy):
        if center is None:
            continue
        row = np.asarray(center, dtype=np.float64)
        if row.shape != (2,):
            raise ValueError(f"observed center {index} must contain normalized x and y")
        if np.isfinite(row).all() and np.all(row >= 0) and np.all(row <= 1):
            observed[index] = row

    trusted = np.isfinite(observed).all(axis=1) & (stability >= config.minimum_stability)
    locked = np.zeros(len(boxes), dtype=bool)
    for index in config.locked_box_indices:
        if not 0 <= int(index) < len(boxes):
            raise ValueError(f"locked box index {index} is out of range")
        locked[int(index)] = True
    trusted &= ~locked

    residual = centers - observed
    residual[~np.isfinite(residual)] = 0.0
    limits = np.asarray(config.max_shift_xy, dtype=np.float64)
    shift = np.clip(config.gain * residual, -limits, limits)
    shift[~trusted] = 0.0

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    shifted_centers = centers + shift
    half_w, half_h = widths / 2, heights / 2
    shifted_centers[:, 0] = np.clip(shifted_centers[:, 0], half_w, 1.0 - half_w)
    shifted_centers[:, 1] = np.clip(shifted_centers[:, 1], half_h, 1.0 - half_h)
    shifted_boxes = np.column_stack((
        shifted_centers[:, 0] - half_w,
        shifted_centers[:, 1] - half_h,
        shifted_centers[:, 0] + half_w,
        shifted_centers[:, 1] + half_h,
    ))

    return {
        "target_boxes_xyxy": boxes.tolist(),
        "steered_boxes_xyxy": shifted_boxes.tolist(),
        "observed_centers_xy": [None if not np.isfinite(row).all() else row.tolist() for row in observed],
        "target_residual_xy": residual.tolist(),
        "applied_shift_xy": shift.tolist(),
        "trusted_update": trusted.tolist(),
        "locked_update": locked.tolist(),
        "confidence_semantics": "observer stability threshold only; not a calibrated probability",
        "config": asdict(config),
    }


def rasterize_path_boxes(
    boxes_xyxy: Sequence[Sequence[float]], *, height: int, width: int
) -> np.ndarray:
    """Rasterize normalized xyxy boxes into a bool mask shaped [T,H,W]."""
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] == 0:
        raise ValueError("boxes_xyxy must have shape [T,4]")
    if not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 1):
        raise ValueError("boxes must be finite and normalized to [0,1]")
    mask = np.zeros((len(boxes), height, width), dtype=bool)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        left = min(width - 1, max(0, int(np.floor(x1 * width))))
        top = min(height - 1, max(0, int(np.floor(y1 * height))))
        right = min(width, max(left + 1, int(np.ceil(x2 * width))))
        bottom = min(height, max(top + 1, int(np.ceil(y2 * height))))
        mask[index, top:bottom, left:right] = True
    return mask


def summarize_rasterized_updates(
    target_boxes_xyxy: Sequence[Sequence[float]],
    steered_boxes_xyxy: Sequence[Sequence[float]],
    *,
    height: int,
    width: int,
) -> list[dict[str, float | int]]:
    """Report the effective token-grid change after box rasterization."""
    target = rasterize_path_boxes(target_boxes_xyxy, height=height, width=width)
    steered = rasterize_path_boxes(steered_boxes_xyxy, height=height, width=width)
    if target.shape != steered.shape:
        raise ValueError("target and steered paths must have the same number of boxes")
    summaries: list[dict[str, float | int]] = []
    for target_mask, steered_mask in zip(target, steered):
        target_y, target_x = np.nonzero(target_mask)
        steered_y, steered_x = np.nonzero(steered_mask)
        intersection = int(np.logical_and(target_mask, steered_mask).sum())
        union = int(np.logical_or(target_mask, steered_mask).sum())
        summaries.append({
            "target_occupied_tokens": int(target_mask.sum()),
            "steered_occupied_tokens": int(steered_mask.sum()),
            "changed_token_count": int(np.logical_xor(target_mask, steered_mask).sum()),
            "foreground_mask_iou": intersection / union if union else 0.0,
            "center_shift_tokens_x": float(steered_x.mean() - target_x.mean()),
            "center_shift_tokens_y": float(steered_y.mean() - target_y.mean()),
        })
    return summaries
