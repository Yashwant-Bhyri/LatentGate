#!/usr/bin/env python3
"""Offline step-4 feedback/action diagnostic on the existing Wan M0 clip.

This is not a closed-loop rerun: it observes the saved step-4 x0 preview from
one text-prompt-only generation, computes the proposed DiTraj box update, and
measures the exact 52x30 mask change that update would have made.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .box_residual_feedback import FeedbackConfig, correct_path_boxes
from .ditraj_feedback_adapter import (
    OBSERVATION_FRAMES,
    mask_change_report,
    observe_red_frame,
    rasterize_like_ditraj,
)
from .evaluate_ditraj_red_observer_kubric import center_error_metrics
from .evaluate_trajectory_frames import interpolate_boxes


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/run_001"
FINAL_FRAMES = RUN / "frames"
X0_FRAMES = RUN / "predicted_clean_previews/step_004/frames"
TRAJECTORY = ROOT / "research/trajectory_control_v1.json"
M0_AUDIT = ROOT / "m0/analysis/wan_t4_retry5_frame_motion_audit.json"
OUTPUT = ROOT / "research/m0_step4_feedback_mechanism_diagnostic_v1.json"
GRID_W, GRID_H = 52, 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_observations(directory: Path) -> tuple[list[dict[str, Any]], dict[str, str], int, int]:
    paths = [directory / f"frame_{index:03d}.png" for index in range(17)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Expected all 17 frames in {directory}")
    observations: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    width = height = 0
    for path in paths:
        rgb_u8 = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb_u8.shape[:2]
        if width == 0:
            width, height = w, h
        elif (width, height) != (w, h):
            raise ValueError(f"Frame size mismatch in {directory}: {(w, h)} != {(width, height)}")
        row = observe_red_frame(rgb_u8.astype(np.float32) / 255.0)
        observations.append(row)
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    return observations, hashes, width, height


def summarize_errors(
    observations: list[dict[str, Any]], target_boxes: list[list[float]], width: int, height: int
) -> dict[str, Any]:
    errors_px: list[float] = []
    errors_diag: list[float] = []
    scoreable = 0
    for observation, box in zip(observations, target_boxes, strict=True):
        center = observation["center_xy_normalized"]
        if center is None:
            continue
        target_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        px, diag, _ = center_error_metrics(center, target_center, width, height)
        errors_px.append(px)
        errors_diag.append(diag)
        scoreable += 1
    return {
        "scoreable_coverage": scoreable / len(observations),
        "mean_center_error_px_valid_only": float(np.mean(errors_px)) if errors_px else None,
        "p95_center_error_px_valid_only": float(np.percentile(errors_px, 95)) if errors_px else None,
        "mean_center_error_frame_diagonals_valid_only": float(np.mean(errors_diag)) if errors_diag else None,
    }


def evaluate() -> dict[str, Any]:
    trajectory = json.loads(TRAJECTORY.read_text())
    m0 = json.loads(M0_AUDIT.read_text())
    anchors = trajectory["target_boxes_yxyx"]
    target_boxes_all = interpolate_boxes(anchors, 17)
    target_key_boxes = [target_boxes_all[index] for index in OBSERVATION_FRAMES]
    final_obs, final_hashes, width, height = read_observations(FINAL_FRAMES)
    x0_obs, x0_hashes, x0_width, x0_height = read_observations(X0_FRAMES)
    if (x0_width, x0_height) != (width, height) or (width, height) != (832, 480):
        raise ValueError(f"Unexpected M0 preview size {(x0_width, x0_height)} / {(width, height)}")

    raw_centers = [x0_obs[index]["center_xy_normalized"] for index in OBSERVATION_FRAMES]
    stability = [float(x0_obs[index]["stability_heuristic"]) for index in OBSERVATION_FRAMES]
    target_centers = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in target_key_boxes]
    anchor_trusted = raw_centers[0] is not None and stability[0] >= 0.65
    anchor_offset = None
    calibrated: list[list[float] | None] = [None] * len(raw_centers)
    if anchor_trusted:
        anchor_offset = [raw_centers[0][axis] - target_centers[0][axis] for axis in range(2)]
        for i, center in enumerate(raw_centers):
            if center is None:
                continue
            corrected = [center[axis] - anchor_offset[axis] for axis in range(2)]
            calibrated[i] = corrected if all(0 <= value <= 1 for value in corrected) else None

    config = FeedbackConfig(
        gain=0.5,
        minimum_stability=0.65,
        max_shift_xy=(1.0 / GRID_W, 1.0 / GRID_H),
        locked_box_indices=(0,),
    )
    update = correct_path_boxes(target_key_boxes, calibrated, stability, config=config)
    target_mask = torch.from_numpy(rasterize_like_ditraj(target_key_boxes, height=GRID_H, width=GRID_W))
    updated_mask = torch.from_numpy(
        rasterize_like_ditraj(update["steered_boxes_xyxy"], height=GRID_H, width=GRID_W)
    )
    raster = mask_change_report(target_mask, updated_mask)

    per_keyframe: list[dict[str, Any]] = []
    for position, frame_index in enumerate(OBSERVATION_FRAMES):
        final_center = final_obs[frame_index]["center_xy_normalized"]
        preview_center = raw_centers[position]
        target_center = target_centers[position]
        raw_start_error_px = None
        relative_path_error_grid_xy = None
        final_error_px = None
        if preview_center is not None:
            final_error_px = center_error_metrics(preview_center, target_center, width, height)[0]
            if frame_index == 0:
                raw_start_error_px = final_error_px
            if raw_centers[0] is not None:
                relative_path_error_grid_xy = [
                    ((preview_center[axis] - raw_centers[0][axis])
                     - (target_center[axis] - target_centers[0][axis]))
                    * [GRID_W, GRID_H][axis]
                    for axis in range(2)
                ]
        if final_center is None:
            final_center_px = None
        else:
            final_center_px = [final_center[0] * width, final_center[1] * height]
        applied = update["applied_shift_xy"][position]
        per_keyframe.append({
            "rgb_frame_index": frame_index,
            "target_center_px": [target_center[0] * width, target_center[1] * height],
            "step4_x0_center_px": [preview_center[0] * width, preview_center[1] * height] if preview_center else None,
            "final_rgb_center_px": final_center_px,
            "step4_x0_raw_center_error_px": final_error_px,
            "step4_x0_anchor_relative_error_continuous_control_grid_xy": relative_path_error_grid_xy,
            "observer_stability": stability[position],
            "accepted_by_development_cutoff": stability[position] >= config.minimum_stability,
            "applied_box_shift_xy_normalized": applied,
            "applied_box_shift_continuous_control_grid_xy": [applied[0] * GRID_W, applied[1] * GRID_H],
            "rasterized_mask_change": raster[position],
        })

    x0_relative_errors = [
        np.linalg.norm(row["step4_x0_anchor_relative_error_continuous_control_grid_xy"])
        for row in per_keyframe
        if row["step4_x0_anchor_relative_error_continuous_control_grid_xy"] is not None
    ]
    commanded_dx_px = (target_centers[-1][0] - target_centers[0][0]) * width
    observed_final_dx_px = (
        (final_obs[-1]["center_xy_normalized"][0] - final_obs[0]["center_xy_normalized"][0]) * width
        if final_obs[-1]["center_xy_normalized"] is not None and final_obs[0]["center_xy_normalized"] is not None
        else None
    )
    action_trusted_count = sum(bool(value) for value in update["trusted_update"])
    return {
        "experiment_id": "EXP-M0-STEP4-FEEDBACK-MECHANISM-DIAGNOSTIC-001",
        "status": "offline_counterfactual_diagnostic_no_model_rerun",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sample": {
            "source_run": m0["source_run"],
            "model": m0["source_generation"]["model"],
            "seed": m0["source_generation"]["seed"],
            "prompt": m0["source_generation"]["prompt"],
            "note": "Text-prompt-only M0 generation; not a DiTraj path-conditioned run.",
        },
        "method": {
            "step4_x0_frames": "saved zero-based step-4 predicted-clean RGB preview",
            "final_rgb_frames": "final decoded frames from the same M0 sample",
            "observer": "same nine-threshold red-centroid observer used by the proposed feedback adapter",
            "counterfactual": "apply one bounded step-4 residual to the five original target boxes and rasterize, but do not re-run denoising",
            "controller_config": json.loads(json.dumps(config.__dict__)),
            "latent_grid_wh": [GRID_W, GRID_H],
        },
        "input_sha256": {
            **final_hashes,
            **x0_hashes,
            TRAJECTORY.relative_to(ROOT).as_posix(): sha256(TRAJECTORY),
            M0_AUDIT.relative_to(ROOT).as_posix(): sha256(M0_AUDIT),
        },
        "final_output_vs_commanded_path": {
            **summarize_errors(final_obs, target_boxes_all, width, height),
            "commanded_endpoint_dx_px": commanded_dx_px,
            "observed_endpoint_dx_px": observed_final_dx_px,
            "fraction_of_commanded_horizontal_displacement": observed_final_dx_px / commanded_dx_px if observed_final_dx_px is not None and commanded_dx_px else None,
            "source_audit_red_centroid_endpoint_dx_px": m0["measurements"]["centroid_endpoint_shift_xy_pixels"][0],
        },
        "step4_x0_hat_vs_commanded_path": {
            "controller_frame_indices": list(OBSERVATION_FRAMES),
            "observer_scoreable_coverage": sum(row["center_xy_normalized"] is not None for row in x0_obs) / len(x0_obs),
            "start_observation_trusted_at_0_65_in_sample_cutoff": bool(anchor_trusted),
            "raw_start_error_px": per_keyframe[0]["step4_x0_raw_center_error_px"],
            "anchor_offset_xy_normalized": anchor_offset,
            "mean_l2_anchor_relative_error_continuous_control_grid_units": float(np.mean(x0_relative_errors)) if x0_relative_errors else None,
            "trusted_unlocked_action_box_count_of_4": action_trusted_count,
            "rasterized_changed_token_total": sum(int(row["changed_token_count"]) for row in raster),
            "per_keyframe": per_keyframe,
        },
        "limitations": [
            "One existing text-prompt-only clip and one saved x0 checkpoint; not DiTraj and not a matched path-conditioned baseline.",
            "The same red proxy is used for observing the saved clip; there are no independent human labels here.",
            "The step-4 box update is counterfactual only; the denoising loop was not resumed with the modified mask.",
            "The 0.65 observer cutoff is in-sample on two synthetic paths and is only used here to mirror the implemented development controller.",
            "A rasterized mask shift demonstrates controller actuation, not predictable object motion or efficacy.",
        ],
    }


def main() -> None:
    result = evaluate()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "status": result["status"],
        "final_path": result["final_output_vs_commanded_path"],
        "step4": {key: value for key, value in result["step4_x0_hat_vs_commanded_path"].items() if key != "per_keyframe"},
    }, indent=2))


if __name__ == "__main__":
    main()
