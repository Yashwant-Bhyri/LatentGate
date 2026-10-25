#!/usr/bin/env python3
"""Calibrate the pilot's red-ball observer on Kubric's known segmentation masks.

This is a controlled observer check, not a video-generation gain result. RGB is
the only input to the red observer; binary Kubric masks are used afterward as
known localization truth. Results are written beside this script.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .ditraj_feedback_adapter import THRESHOLDS, observe_red_frame


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "kubric_gate0_fixture_v1"
OUTPUT = ROOT / "ditraj_red_observer_kubric_calibration_v1.json"
STABILITY_CUTOFF = 0.65


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def center_error_metrics(
    predicted_xy_normalized: list[float] | np.ndarray,
    truth_xy_normalized: list[float] | np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, list[float]]:
    """Return Euclidean pixel error, frame-diagonal error, and signed pixel deltas."""
    delta_px = (
        np.asarray(predicted_xy_normalized, dtype=np.float64)
        - np.asarray(truth_xy_normalized, dtype=np.float64)
    ) * np.asarray([width, height], dtype=np.float64)
    error_px = float(np.linalg.norm(delta_px))
    return error_px, error_px / math.hypot(width, height), [float(v) for v in delta_px]


def evaluate() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for direction in ("left", "right"):
        for frame_index in range(17):
            frame_path = FIXTURE / direction / "frames" / f"frame_{frame_index:03d}.png"
            mask_path = FIXTURE / direction / "masks" / f"mask_{frame_index:03d}.png"
            if not frame_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing known-truth fixture frame or mask for {direction}/{frame_index}")
            rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
            mask_image = np.asarray(Image.open(mask_path).convert("L"))
            if set(np.unique(mask_image).tolist()) - {0, 255}:
                raise ValueError(f"Expected binary segmentation mask, got values {np.unique(mask_image)}")
            mask = mask_image > 0
            ys, xs = np.nonzero(mask)
            if not len(xs):
                raise ValueError(f"Ground-truth object mask is empty: {mask_path}")
            height, width = mask.shape
            truth_xy = [float(xs.mean()) / width, float(ys.mean()) / height]
            observation = observe_red_frame(rgb)
            scoreable = bool(observation["scoreable"])
            if scoreable:
                predicted = np.asarray(observation["center_xy_normalized"], dtype=np.float64)
                error_px, error, delta_px_list = center_error_metrics(predicted, truth_xy, width, height)
            else:
                error = 1.0
                error_px = math.hypot(width, height)
                delta_px_list = None
            records.append({
                "direction": direction,
                "frame_index": frame_index,
                "scoreable": scoreable,
                "valid_threshold_count": int(observation["valid_threshold_count"]),
                "stability_heuristic": float(observation["stability_heuristic"]),
                "truth_mask_center_xy_normalized": truth_xy,
                "observed_center_xy_normalized": observation["center_xy_normalized"],
                "center_error_frame_diagonals_missing_penalized": error,
                "center_error_px_missing_penalized": error_px,
                "center_error_components_px_xy": delta_px_list,
            })
            inputs[frame_path.relative_to(ROOT).as_posix()] = sha256(frame_path)
            inputs[mask_path.relative_to(ROOT).as_posix()] = sha256(mask_path)

    # For this path-control use, the object begins in a shared commanded box.
    # Anchor every RGB trajectory at its first frame to quantify motion
    # tracking separately from a static threshold-centroid offset.
    token_grid_xy = np.asarray([52.0, 30.0])
    for direction in ("left", "right"):
        sample = [row for row in records if row["direction"] == direction]
        first = sample[0]
        if not first["scoreable"]:
            for row in sample:
                row["anchor_relative_error_tokens_xy"] = None
            continue
        first_truth = np.asarray(first["truth_mask_center_xy_normalized"], dtype=np.float64)
        first_observed = np.asarray(first["observed_center_xy_normalized"], dtype=np.float64)
        for row in sample:
            if not row["scoreable"]:
                row["anchor_relative_error_tokens_xy"] = None
                continue
            truth = np.asarray(row["truth_mask_center_xy_normalized"], dtype=np.float64)
            observed = np.asarray(row["observed_center_xy_normalized"], dtype=np.float64)
            relative_error = ((observed - first_observed) - (truth - first_truth)) * token_grid_xy
            row["anchor_relative_error_tokens_xy"] = relative_error.tolist()
            row["controller_keyframe"] = row["frame_index"] in (0, 4, 8, 12, 16)

    overall: dict[str, Any] = {}
    for direction in ("left", "right", "all"):
        sample = records if direction == "all" else [row for row in records if row["direction"] == direction]
        scoreable = [row for row in sample if row["scoreable"]]
        trusted = [row for row in scoreable if row["stability_heuristic"] >= STABILITY_CUTOFF]
        anchored = [row for row in sample if row.get("anchor_relative_error_tokens_xy") is not None]
        relative = np.asarray([row["anchor_relative_error_tokens_xy"] for row in anchored], dtype=np.float64)
        controller_anchored = [row for row in anchored if row.get("controller_keyframe", False)]
        controller_relative = np.asarray(
            [row["anchor_relative_error_tokens_xy"] for row in controller_anchored], dtype=np.float64
        )
        start_rows = [row for row in sample if row["frame_index"] == 0]
        start_errors = [float(row["center_error_px_missing_penalized"]) for row in start_rows]
        errors = [float(row["center_error_frame_diagonals_missing_penalized"]) for row in sample]
        valid_errors = [float(row["center_error_frame_diagonals_missing_penalized"]) for row in scoreable]
        trusted_errors = [float(row["center_error_frame_diagonals_missing_penalized"]) for row in trusted]
        overall[direction] = {
            "frames": len(sample),
            "scoreable_coverage": len(scoreable) / len(sample),
            "mean_valid_center_error_frame_diagonals": statistics.mean(valid_errors) if valid_errors else None,
            "p95_valid_center_error_frame_diagonals": percentile(valid_errors, 95),
            "maximum_valid_center_error_frame_diagonals": max(valid_errors) if valid_errors else None,
            "mean_missing_penalized_center_error_frame_diagonals": statistics.mean(errors),
            "mean_valid_center_error_px": statistics.mean(
                row["center_error_px_missing_penalized"] for row in scoreable
            ) if scoreable else None,
            "p95_valid_center_error_px": percentile(
                [row["center_error_px_missing_penalized"] for row in scoreable], 95
            ),
            "first_frame_absolute_center_error_mean_px_missing_penalized": statistics.mean(start_errors) if start_errors else None,
            "first_frame_absolute_center_error_p95_px_missing_penalized": percentile(start_errors, 95),
            "stability_cutoff": STABILITY_CUTOFF,
            "trusted_coverage": len(trusted) / len(sample),
            "mean_trusted_center_error_frame_diagonals": statistics.mean(trusted_errors) if trusted_errors else None,
            "mean_stability_heuristic": statistics.mean(row["stability_heuristic"] for row in sample),
            "mean_valid_threshold_votes_out_of_9": statistics.mean(row["valid_threshold_count"] for row in sample),
            "anchor_relative_trajectory_error_mean_abs_control_grid_units_xy": np.abs(relative).mean(axis=0).tolist() if len(relative) else None,
            "anchor_relative_trajectory_error_p95_abs_control_grid_units_xy": np.percentile(np.abs(relative), 95, axis=0).tolist() if len(relative) else None,
            "anchor_relative_trajectory_error_mean_l2_control_grid_units": float(np.linalg.norm(relative, axis=1).mean()) if len(relative) else None,
            "controller_keyframe_count": len(controller_anchored),
            "controller_keyframe_anchor_relative_mean_abs_control_grid_units_xy": np.abs(controller_relative).mean(axis=0).tolist() if len(controller_relative) else None,
            "controller_keyframe_anchor_relative_p95_abs_control_grid_units_xy": np.percentile(np.abs(controller_relative), 95, axis=0).tolist() if len(controller_relative) else None,
            "controller_keyframe_anchor_relative_mean_l2_control_grid_units": float(np.linalg.norm(controller_relative, axis=1).mean()) if len(controller_relative) else None,
        }

    return {
        "experiment_id": "EXP-DITRAJ-RED-OBSERVER-KUBRIC-CALIBRATION-001",
        "status": "known_truth_observer_calibration_cpu_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": {
            "id": "EXP-LATENTGATE-KUBRIC-GATE0-RENDER-001",
            "source_commit": "61f2422c84bab75006df33c6989e0b483db3ccfe",
            "directions": ["left", "right"],
            "frames": 34,
            "frame_size_wh": [160, 96],
            "truth": "Kubric binary segmentation masks; observer receives RGB only",
            "latent_control_grid_wh_for_relative_error": [52, 30],
            "relative_error_units": "continuous displacement scaled to the 52x30 control grid; not a quantized token error",
        },
        "observer": {
            "family": "full-frame nine-threshold red centroid ensemble",
            "thresholds_red_gt": [130, 150, 170],
            "thresholds_green_blue_lt": [100, 125, 150],
            "minimum_area": "300 pixels at 832x480, scaled by image area with an 8-pixel floor",
            "center": "coordinate-wise median of scoreable threshold-specific centroids",
            "stability": "threshold coverage times clipped 1 - median center spread / 0.02 frame diagonals; heuristic, not calibrated probability; 0.65 is an in-sample development cutoff, not a held-out pass gate",
        },
        "aggregate": overall,
        "per_frame": records,
        "input_sha256": inputs,
        "limitations": [
            "This tests a red Kubric sphere on its controlled background, not generated Wan previews or other object classes.",
            "Threshold agreement is not a calibrated identity probability; stability cutoff 0.65 is provisional and fixture-specific.",
            "The 0.65 cutoff and its coverage were assessed on these same two development paths; no held-out path or generated preview validates it yet.",
            "The first-frame anchor cancels a fixed observer offset for relative path measurements; absolute first-frame localization is reported separately because the start box is locked and cannot be corrected.",
            "The 34 synthetic frames are correlated across two trajectories and do not replace held-out generated samples or human labels.",
            "No controller was run against model output; this is observer localization evidence only.",
        ],
    }


def main() -> None:
    result = evaluate()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
