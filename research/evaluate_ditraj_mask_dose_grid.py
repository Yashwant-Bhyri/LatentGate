#!/usr/bin/env python3
"""Map discrete late-mask action doses on the paired DiTraj control paths.

This CPU-only analysis measures changes to the actual 5x30x52 attention-mask
grid. It does not load Wan, run DiTraj, or predict rendered object motion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ditraj_feedback_adapter import OBSERVATION_FRAMES, mask_change_report, rasterize_like_ditraj
from evaluate_trajectory_frames import interpolate_boxes


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "research/path_counterfactual_spec_v1.json"
ADAPTER_PATH = ROOT / "research/ditraj_feedback_adapter.py"
OUTPUT_PATH = ROOT / "research/ditraj_mask_dose_grid_v1.json"
GRID_W, GRID_H = 52, 30
OUTPUT_W, OUTPUT_H = 832, 480
ACTION_DOSES_CELLS_X = (-2, -1, 0, 1, 2)
LOCKED_SLICE_INDEX = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raster_sizes(mask: np.ndarray) -> list[list[int]]:
    result = []
    for frame in mask[:, 0] > 0.5:
        ys, xs = np.nonzero(frame)
        if not len(xs):
            raise ValueError("Dose map generated an empty target box")
        result.append([int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)])
    return result


def score_action(base_mask: np.ndarray, action_mask: np.ndarray) -> dict[str, Any]:
    base = base_mask[:, 0] > 0.5
    action = action_mask[:, 0] > 0.5
    intersection = int(np.logical_and(base, action).sum())
    union = int(np.logical_or(base, action).sum())
    records = mask_change_report(torch.from_numpy(base_mask), torch.from_numpy(action_mask))
    return {
        "changed_token_count_total": int(np.logical_xor(base, action).sum()),
        "whole_path_mask_iou": intersection / union if union else 0.0,
        "start_slice_unchanged": bool(np.array_equal(base[LOCKED_SLICE_INDEX], action[LOCKED_SLICE_INDEX])),
        "rasterized_box_sizes_wh_tokens": raster_sizes(action_mask),
        "per_latent_slice": records,
    }


def evaluate() -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    frame_count = int(spec["model"]["frames"])
    controls = list(OBSERVATION_FRAMES)
    if controls != spec["paired_paths"]["latent_temporal_samples_from_17_frames"]:
        raise ValueError("Adapter control slices disagree with the paired-path spec")

    paths: dict[str, Any] = {}
    for path_name in ("left", "right"):
        anchors = spec["paired_paths"][path_name]["anchors_yxyx"]
        all_boxes = interpolate_boxes(anchors, frame_count)
        target_boxes = [all_boxes[index] for index in controls]
        base_mask = rasterize_like_ditraj(target_boxes, height=GRID_H, width=GRID_W)
        actions: dict[str, Any] = {}

        for dose in ACTION_DOSES_CELLS_X:
            shifted = [box.copy() for box in target_boxes]
            delta_x = dose / GRID_W
            for index in range(1, len(shifted)):
                shifted[index][0] += delta_x
                shifted[index][2] += delta_x
            if any(min(box) < 0.0 or max(box) > 1.0 for box in shifted):
                raise ValueError(f"Dose {dose} moves a {path_name} box outside normalized bounds")

            action_mask = rasterize_like_ditraj(shifted, height=GRID_H, width=GRID_W)
            summary = score_action(base_mask, action_mask)
            summary["requested_shift_cells_xy"] = [dose, 0]
            summary["future_slices_only"] = True
            summary["normalized_box_size_preserved"] = all(
                np.allclose(
                    [after[2] - after[0], after[3] - after[1]],
                    [before[2] - before[0], before[3] - before[1]],
                    rtol=0.0,
                    atol=1e-12,
                )
                for before, after in zip(target_boxes, shifted, strict=True)
            )
            actions[str(dose)] = summary

        paths[path_name] = {
            "target_boxes_xyxy_control_slices": target_boxes,
            "base_occupied_tokens_per_slice": [
                int(frame.sum()) for frame in (base_mask[:, 0] > 0.5)
            ],
            "actions_by_horizontal_cell_shift": actions,
        }

    return {
        "experiment_id": "EXP-DITRAJ-LATE-MASK-DOSE-MAP-001",
        "status": "cpu_only_mask_input_geometry_audit",
        "scope": "Late foreground/background mask action after the shared prefix; early RoPE and the first latent-time box remain fixed.",
        "model_outcome_claim": "none; this report measures mask inputs only and contains no generated-video response",
        "grid": {
            "latent_spatial_grid_wh": [GRID_W, GRID_H],
            "output_frame_wh": [OUTPUT_W, OUTPUT_H],
            "nominal_output_pixels_per_cell_xy": [OUTPUT_W // GRID_W, OUTPUT_H // GRID_H],
            "latent_time_rgb_frames": controls,
            "locked_slice_index": LOCKED_SLICE_INDEX,
        },
        "action_doses_horizontal_grid_cells": list(ACTION_DOSES_CELLS_X),
        "inputs_sha256": {
            SPEC_PATH.relative_to(ROOT).as_posix(): sha256(SPEC_PATH),
            ADAPTER_PATH.relative_to(ROOT).as_posix(): sha256(ADAPTER_PATH),
        },
        "paths": paths,
        "interpretation": [
            "Compare no-op and both signs at one- and two-cell doses before choosing an action scale for a model run.",
            "A changed mask, cell-center shift, or IoU change establishes discrete input actuation only; object response requires matched generated clips and independent labels.",
            "These late-mask doses do not test a full-run DiTraj path shift or the early STD-RoPE intervention.",
        ],
    }


def main() -> None:
    report = evaluate()
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    concise = {
        "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
        "status": report["status"],
        "paths": {},
    }
    for path_name, path in report["paths"].items():
        concise["paths"][path_name] = {
            dose: {
                "changed_token_count_total": result["changed_token_count_total"],
                "whole_path_mask_iou": result["whole_path_mask_iou"],
                "start_slice_unchanged": result["start_slice_unchanged"],
                "center_shift_tokens_xy": [
                    row["center_shift_tokens_xy"] for row in result["per_latent_slice"]
                ],
            }
            for dose, result in path["actions_by_horizontal_cell_shift"].items()
        }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
