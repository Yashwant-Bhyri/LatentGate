#!/usr/bin/env python3
"""Measure SAM 2.1 mask sensitivity to a small first-frame point offset.

This local MPS observer experiment perturbs the oracle point prompt vertically
on two known-truth Kubric clips. It tests prompt sensitivity only; it does not
simulate an automatic detector or validate a Wan/DiTraj controller observer.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor

from sam2_observer_probe import (
    KUBRIC_ROOT,
    MODEL_ID,
    MODEL_REVISION,
    ROOT,
    load_frames,
    run_tracker,
    seed_from_mask,
    sha256,
    summarize_case,
)


BASE_REPORT = ROOT / "research/sam2_observer_probe_v1/report.json"
OUT = ROOT / "research/sam2_prompt_sensitivity_v1"
VERTICAL_OFFSETS_PX = (-4.0, 4.0)


def summarize_signed_errors(case: dict) -> dict:
    dx: list[float] = []
    dy: list[float] = []
    for record in case["records"]:
        pred = record["pred_bbox_center_xy_px"]
        box = record["gt_bbox_xyxy_px"]
        if pred is None or box is None:
            continue
        truth = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        dx.append(float(pred[0] - truth[0]))
        dy.append(float(pred[1] - truth[1]))
    case["mean_signed_bbox_center_error_xy_px"] = [
        statistics.mean(dx) if dx else None,
        statistics.mean(dy) if dy else None,
    ]
    case["p95_bbox_center_error_px"] = (
        float(np.percentile([math.hypot(x, y) for x, y in zip(dx, dy)], 95))
        if dx else None
    )
    return case


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("This probe is scoped to the local Apple MPS device; MPS is unavailable.")
    OUT.mkdir(parents=True, exist_ok=True)
    base_report = json.loads(BASE_REPORT.read_text())
    base_by_name = {case["name"]: case for case in base_report["cases"]}
    device = "mps"
    dtype = torch.float16

    load_start = time.perf_counter()
    model = Sam2VideoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(
        device=device, dtype=dtype
    ).eval()
    processor = Sam2VideoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model_load_seconds = time.perf_counter() - load_start

    kubric_gt = json.loads((KUBRIC_ROOT / "ground_truth.json").read_text())
    results: list[dict] = []
    for direction in ("left", "right"):
        frames, frame_paths = load_frames(KUBRIC_ROOT / direction / "frames")
        mask_paths = sorted((KUBRIC_ROOT / direction / "masks").glob("mask_*.png"))
        gt_masks = [np.asarray(Image.open(path).convert("L")) > 0 for path in mask_paths]
        gt_boxes = [row["annotation_from_mask"]["bbox_xyxy_px"] for row in kubric_gt[direction]]
        base_seed = seed_from_mask(gt_masks[0])

        # Reuse the previously recorded center-prompt baseline, preserving its
        # per-frame boxes so signed error can be summarized consistently.
        baseline = dict(base_by_name[f"kubric_{direction}"])
        baseline["prompt_variant"] = "oracle_center_baseline_from_EXP-SAM2-OBSERVER-TRANSFER-PROBE-001"
        results.append(summarize_signed_errors(baseline))

        for offset_y in VERTICAL_OFFSETS_PX:
            prompt_xy = [float(base_seed[0]), float(base_seed[1] + offset_y)]
            ix, iy = int(round(prompt_xy[0])), int(round(prompt_xy[1]))
            if not (0 <= iy < gt_masks[0].shape[0] and 0 <= ix < gt_masks[0].shape[1]):
                raise ValueError(f"Prompt falls outside frame: {prompt_xy}")
            if not gt_masks[0][iy, ix]:
                raise ValueError(f"Prompt is not inside the first-frame target: {direction} {prompt_xy}")

            started = time.perf_counter()
            predicted_masks = run_tracker(model, processor, frames, prompt_xy, device, dtype)
            elapsed = time.perf_counter() - started
            variant = "seed_y_minus_4px" if offset_y < 0 else "seed_y_plus_4px"
            case = summarize_case(
                f"kubric_{direction}_{variant}",
                frames,
                frame_paths,
                predicted_masks,
                prompt_xy,
                OUT,
                gt_masks,
                gt_boxes,
            )
            case["prompt_variant"] = variant
            case["prompt_offset_from_oracle_center_xy_px"] = [0.0, offset_y]
            case["prompt_remained_inside_first_frame_truth_mask"] = True
            case["runtime_seconds"] = elapsed
            results.append(summarize_signed_errors(case))

            partial = {
                "experiment_id": "EXP-SAM2-PROMPT-SENSITIVITY-PROBE-001",
                "status": "running_local_mps",
                "completed_cases": results,
            }
            (OUT / "partial_report.json").write_text(json.dumps(partial, indent=2) + "\n")

    report = {
        "experiment_id": "EXP-SAM2-PROMPT-SENSITIVITY-PROBE-001",
        "status": "completed_local_mps_prompt_sensitivity_probe",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0"},
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pillow": PIL.__version__,
            "numpy": np.__version__,
        },
        "hardware": {"device": device, "dtype": str(dtype), "mps_available": True, "cuda_available": torch.cuda.is_available()},
        "model_load_seconds": model_load_seconds,
        "source_baseline_report": str(BASE_REPORT.relative_to(ROOT)),
        "source_inputs_sha256": {
            direction: {
                "frames": {path.name: sha256(path) for path in sorted((KUBRIC_ROOT / direction / "frames").glob("frame_*.png"))},
                "masks": {path.name: sha256(path) for path in sorted((KUBRIC_ROOT / direction / "masks").glob("mask_*.png"))},
            }
            for direction in ("left", "right")
        },
        "design": {
            "sequences": ["kubric_left", "kubric_right"],
            "frames_per_sequence": 17,
            "baseline_prompt": "first-frame target-mask centroid, reused from the prior SAM2 observer report",
            "perturbations_xy_px": [[0.0, offset] for offset in VERTICAL_OFFSETS_PX],
            "prompt_validity": "Every perturbed positive point was checked to remain inside the known first-frame target mask. This is an oracle-prompt sensitivity test, not detector evaluation.",
            "metrics": ["coverage", "mask IoU", "signed tight-box-center bias", "p95 tight-box-center error", "mask area ratio"],
        },
        "cases": results,
        "limits": [
            "The original center-prompt baseline was reused; only the plus/minus 4 px vertical variants were newly run.",
            "The initial point is selected using known synthetic target masks; this does not measure detector-in-the-wild prompt quality.",
            "Only two correlated simple Kubric sphere clips and two vertical offsets were assessed.",
            "No Wan labels, DiTraj generation, control actuation, or feedback efficacy were tested.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "partial_report.json").unlink(missing_ok=True)
    print(json.dumps({
        "report": str((OUT / "report.json").relative_to(ROOT)),
        "model_load_seconds": round(model_load_seconds, 2),
        "summary": [{
            "name": case["name"],
            "prompt_variant": case["prompt_variant"],
            "coverage": case["mask_coverage"],
            "mean_iou": case["mean_mask_iou_to_kubric_gt"],
            "mean_signed_center_error_xy_px": case["mean_signed_bbox_center_error_xy_px"],
            "mean_center_error_px": case["mean_bbox_center_error_to_gt_px"],
            "mean_area_ratio": case["mean_area_ratio_pred_over_gt"],
            "runtime_seconds": round(case.get("runtime_seconds", 0.0), 2),
        } for case in results],
    }, indent=2))


if __name__ == "__main__":
    main()
