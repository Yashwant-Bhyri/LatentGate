#!/usr/bin/env python3
"""Screen decoded Wan x0-hat previews as a within-run object-position observer.

This CPU-only analysis reuses the already generated M0 run. It compares each
selected predicted-clean preview against the final RGB output using a small
red-mask threshold ensemble. It is an instrumentation/observability screen,
not independent annotation, confidence calibration, a causal control test, or
evidence of trajectory-policy improvement.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/run_001"
FINAL_DIR = RUN / "frames"
PREVIEW_ROOT = RUN / "predicted_clean_previews"
STEPS = (4, 14, 24, 29)
THRESHOLDS = tuple((r, gb) for r in (130, 150, 170) for gb in (100, 125, 150))
MIN_PIXELS = 300


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def observe(path: Path, frame_diag: float) -> dict[str, Any]:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    candidate_centers: list[tuple[float, float]] = []
    candidate_boxes: list[tuple[float, float, float, float]] = []
    candidate_areas: list[int] = []
    for r_gt, gb_lt in THRESHOLDS:
        mask = (rgb[:, :, 0] > r_gt) & (rgb[:, :, 1] < gb_lt) & (rgb[:, :, 2] < gb_lt)
        ys, xs = np.nonzero(mask)
        if len(xs) < MIN_PIXELS:
            continue
        candidate_centers.append((float(xs.mean()), float(ys.mean())))
        candidate_boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))
        candidate_areas.append(int(len(xs)))

    if not candidate_centers:
        return {
            "scoreable": False,
            "valid_threshold_count": 0,
            "threshold_count": len(THRESHOLDS),
            "confidence_proxy": 0.0,
            "center_xy_px": None,
            "bbox_xyxy_px": None,
            "center_spread_median_frame_diagonals": None,
            "area_cv_across_thresholds": None,
        }

    center_x = statistics.median(c[0] for c in candidate_centers)
    center_y = statistics.median(c[1] for c in candidate_centers)
    center_spread = statistics.median(
        math.hypot(x - center_x, y - center_y) for x, y in candidate_centers
    ) / frame_diag
    area_mean = statistics.mean(candidate_areas)
    area_cv = statistics.pstdev(candidate_areas) / area_mean if area_mean else None
    # This deliberately simple stability score is a ranking heuristic only;
    # it is not a calibrated probability of correct identity or localization.
    coverage = len(candidate_centers) / len(THRESHOLDS)
    confidence_proxy = coverage * max(0.0, 1.0 - center_spread / 0.02)
    box = tuple(statistics.median(v[i] for v in candidate_boxes) for i in range(4))
    return {
        "scoreable": True,
        "valid_threshold_count": len(candidate_centers),
        "threshold_count": len(THRESHOLDS),
        "confidence_proxy": confidence_proxy,
        "center_xy_px": [center_x, center_y],
        "bbox_xyxy_px": list(box),
        "center_spread_median_frame_diagonals": center_spread,
        "area_cv_across_thresholds": area_cv,
    }


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.0


def main() -> None:
    final_paths = [FINAL_DIR / f"frame_{i:03d}.png" for i in range(17)]
    preview_paths = {
        step: [PREVIEW_ROOT / f"step_{step:03d}/frames/frame_{i:03d}.png" for i in range(17)]
        for step in STEPS
    }
    missing = [str(p) for p in [*final_paths, *(p for files in preview_paths.values() for p in files)] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing M0 observer-screen inputs: " + ", ".join(missing[:8]))

    first_image = Image.open(final_paths[0])
    frame_diag = math.hypot(first_image.width, first_image.height)
    reference = [observe(path, frame_diag) for path in final_paths]
    step_reports: dict[str, Any] = {}
    for step, paths in preview_paths.items():
        observations = [observe(path, frame_diag) for path in paths]
        errors = []
        ious = []
        paired = []
        for i, (pred, ref) in enumerate(zip(observations, reference)):
            if pred["scoreable"] and ref["scoreable"]:
                pc, rc = pred["center_xy_px"], ref["center_xy_px"]
                error = math.hypot(pc[0] - rc[0], pc[1] - rc[1]) / frame_diag
                iou = box_iou(pred["bbox_xyxy_px"], ref["bbox_xyxy_px"])
                errors.append(error)
                ious.append(iou)
                paired.append({
                    "frame_index": i,
                    "predicted_clean_center_xy_px": pc,
                    "final_rgb_center_xy_px": rc,
                    "center_error_frame_diagonals": error,
                    "bbox_iou": iou,
                    "confidence_proxy": pred["confidence_proxy"],
                    "center_spread_median_frame_diagonals": pred["center_spread_median_frame_diagonals"],
                })
        errors_with_missing_penalty = errors + [1.0] * (len(final_paths) - len(errors))
        step_reports[str(step)] = {
            "scoreable_frame_coverage": len(paired) / len(final_paths),
            "mean_center_error_frame_diagonals_valid_only": statistics.mean(errors) if errors else None,
            "mean_center_error_frame_diagonals_missing_penalized": statistics.mean(errors_with_missing_penalty),
            "mean_bbox_iou_valid_only": statistics.mean(ious) if ious else None,
            "mean_confidence_proxy": statistics.mean(o["confidence_proxy"] for o in observations),
            "mean_center_spread_median_frame_diagonals": statistics.mean(
                o["center_spread_median_frame_diagonals"] for o in observations if o["scoreable"]
            ) if paired else None,
            "per_frame": paired,
        }

    result = {
        "experiment_id": "EXP-M0-X0-OBSERVER-SCREEN-001",
        "status": "measured_on_existing_saved_generation_no_new_model_run",
        "source_run": str(RUN.relative_to(ROOT)),
        "reference": "final RGB frames from this same generated sample, measured with the same red-mask ensemble",
        "observer": "decoded UniPC x0_hat previews from saved predicted_clean_previews",
        "threshold_ensemble": {
            "red_gt": [130, 150, 170],
            "green_blue_lt": [100, 125, 150],
            "minimum_pixels": MIN_PIXELS,
            "total_thresholds": len(THRESHOLDS),
            "center": "median of scoreable threshold-specific centroids",
            "confidence_proxy": "threshold coverage times a clipped center-stability heuristic; not calibrated",
        },
        "frame_count": len(final_paths),
        "frame_diagonal_px": frame_diag,
        "preview_steps_zero_based": list(STEPS),
        "per_step": step_reports,
        "input_sha256": {
            "final_frames": {p.name: sha256(p) for p in final_paths},
            "previews": {
                str(step): {p.name: sha256(p) for p in paths}
                for step, paths in preview_paths.items()
            },
        },
        "limitations": [
            "The final RGB frame is a same-sample proxy, not independent ground truth or a blinded human label.",
            "Both observer and reference use a red-threshold family; threshold agreement is not object-identity calibration.",
            "Only one direction-specific prompt, one seed, and four saved denoising steps are available.",
            "This estimates within-sample x0-hat localization consistency only; it does not test a causal feedback policy or improve a generated result.",
        ],
    }
    out = ROOT / "research/m0_predicted_clean_observer_screen_v1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(out.relative_to(ROOT)),
        "step_summary": {
            step: {k: v for k, v in report.items() if k != "per_frame"}
            for step, report in step_reports.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
