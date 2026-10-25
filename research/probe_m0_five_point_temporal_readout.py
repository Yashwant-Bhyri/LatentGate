#!/usr/bin/env python3
"""Measure proxy-track information lost by sampling a saved 17-frame clip at 5 anchors.

Uses an existing CPU color-mask audit; it does not read model weights or run inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FRAME_AUDIT = ROOT / "m0/analysis/wan_t4_retry5_frame_motion_audit.json"
BASELINE_TRACE = ROOT / "research/m0_trajectory_baseline_v1.json"
SAMPLING_AUDIT = ROOT / "research/m0_temporal_sampling_audit_v1.json"
THRESHOLD_AUDIT = ROOT / "research/m0_mask_threshold_sensitivity_v1.json"
FRAMES_DIR = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/run_001/frames"
OUTPUT = ROOT / "research/m0_temporal_downsampling_probe_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate(rows: list[dict], a: int, b: int, frame: int, field: str):
    va, vb = rows[a][field], rows[b][field]
    if a == b:
        return va
    t = (frame - a) / (b - a)
    if isinstance(va, list):
        return [x + t * (y - x) for x, y in zip(va, vb)]
    return va + t * (vb - va)


def summarize_proxy_trace(trace: list[dict], anchors: list[int]) -> dict:
    if any(trace[a]["centroid_xy_pixels"] is None for a in anchors):
        return {
            "full_frame_coverage": sum(
                row["centroid_xy_pixels"] is not None for row in trace
            ) / len(trace),
            "unsampled_center_mean_px": None,
            "unsampled_center_max_px": None,
            "unsampled_area_mean_relative_error_pct": None,
        }

    center_errors = []
    area_relative_errors = []
    for frame, row in enumerate(trace):
        left = max(anchor for anchor in anchors if anchor <= frame)
        right = min(anchor for anchor in anchors if anchor >= frame)
        center_hat = interpolate(trace, left, right, frame, "centroid_xy_pixels")
        area_hat = interpolate(trace, left, right, frame, "pixel_count")
        center = row["centroid_xy_pixels"]
        if center is not None and frame not in anchors:
            center_errors.append(
                math.hypot(center[0] - center_hat[0], center[1] - center_hat[1])
            )
        if frame not in anchors and row["pixel_count"]:
            area_relative_errors.append(
                100 * abs(row["pixel_count"] - area_hat) / row["pixel_count"]
            )

    return {
        "full_frame_coverage": sum(
            row["centroid_xy_pixels"] is not None for row in trace
        ) / len(trace),
        "unsampled_center_mean_px": mean(center_errors) if center_errors else None,
        "unsampled_center_max_px": max(center_errors) if center_errors else None,
        "unsampled_area_mean_relative_error_pct": (
            mean(area_relative_errors) if area_relative_errors else None
        ),
    }


def main() -> None:
    frame_audit = json.loads(FRAME_AUDIT.read_text())
    baseline_trace = json.loads(BASELINE_TRACE.read_text())
    sampling_audit = json.loads(SAMPLING_AUDIT.read_text())
    threshold_audit = json.loads(THRESHOLD_AUDIT.read_text())
    rows = sorted(
        [
            {
                "frame": row["frame"],
                "pixel_count": row["mask_pixels"],
                "centroid_xy_pixels": row["mask_center_xy_px"],
            }
            for row in baseline_trace["records"]
        ],
        key=lambda row: row["frame"],
    )
    anchors = sampling_audit["temporal_alignment"][
        "derived_nominal_rgb_frame_anchors"
    ]
    frame_ids = [int(row["frame"]) for row in rows]
    if frame_ids != list(range(len(rows))):
        raise ValueError("Expected one contiguous per-frame record starting at frame 0")
    if anchors[0] != frame_ids[0] or anchors[-1] != frame_ids[-1]:
        raise ValueError("Sampling anchors must include the first and last source frames")

    frame_paths = sorted(FRAMES_DIR.glob("frame_*.png"))
    if len(frame_paths) != len(rows):
        raise ValueError(f"Expected {len(rows)} RGB frames, found {len(frame_paths)}")
    rgb_frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
    grid = threshold_audit["inputs"]["rgb_threshold_grid"]
    min_pixels = int(threshold_audit["inputs"]["minimum_mask_pixels"])
    threshold_cases = []
    base_recomputed_trace = None
    for r_gt, g_lt, b_lt in product(
        grid["red_greater_than"],
        grid["green_less_than"],
        grid["blue_less_than"],
    ):
        trace = []
        for rgb in rgb_frames:
            mask = (
                (rgb[:, :, 0] > r_gt)
                & (rgb[:, :, 1] < g_lt)
                & (rgb[:, :, 2] < b_lt)
            )
            ys, xs = np.nonzero(mask)
            centroid = (
                [float(xs.mean()), float(ys.mean())]
                if len(xs) >= min_pixels
                else None
            )
            trace.append(
                {"pixel_count": int(len(xs)), "centroid_xy_pixels": centroid}
            )
        metrics = summarize_proxy_trace(trace, anchors)
        threshold_cases.append(
            {
                "threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
                **{
                    key: (round(value, 6) if value is not None else None)
                    for key, value in metrics.items()
                },
            }
        )
        if (r_gt, g_lt, b_lt) == (150, 125, 125):
            base_recomputed_trace = trace

    if base_recomputed_trace is None:
        raise ValueError("Threshold grid omitted the registered baseline threshold")
    baseline_center_delta = []
    baseline_area_delta = []
    rounded_audit_center_delta = []
    rounded_audit_rows = sorted(
        frame_audit["measurements"]["per_frame"], key=lambda row: row["frame"]
    )
    for source_row, recomputed in zip(rows, base_recomputed_trace):
        source_center = source_row["centroid_xy_pixels"]
        computed_center = recomputed["centroid_xy_pixels"]
        if source_center is None or computed_center is None:
            baseline_center_delta.append(None)
        else:
            baseline_center_delta.append(
                math.hypot(
                    source_center[0] - computed_center[0],
                    source_center[1] - computed_center[1],
                )
            )
        baseline_area_delta.append(
            abs(source_row["pixel_count"] - recomputed["pixel_count"])
        )
    for source_row, rounded_row in zip(rows, rounded_audit_rows):
        rounded_audit_center_delta.append(
            math.hypot(
                source_row["centroid_xy_pixels"][0]
                - rounded_row["centroid_xy_pixels"][0],
                source_row["centroid_xy_pixels"][1]
                - rounded_row["centroid_xy_pixels"][1],
            )
        )

    fully_covered_cases = [
        case for case in threshold_cases if case["full_frame_coverage"] == 1.0
    ]

    def range_summary(field: str) -> dict:
        values = [
            case[field]
            for case in fully_covered_cases
            if case[field] is not None
        ]
        return {
            "min": round(min(values), 6),
            "median": round(median(values), 6),
            "max": round(max(values), 6),
        }

    center_errors = []
    area_errors = []
    per_frame = []
    for frame, row in enumerate(rows):
        left = max(a for a in anchors if a <= frame)
        right = min(a for a in anchors if a >= frame)
        center_hat = interpolate(rows, left, right, frame, "centroid_xy_pixels")
        area_hat = interpolate(rows, left, right, frame, "pixel_count")
        cx, cy = row["centroid_xy_pixels"]
        dx, dy = cx - center_hat[0], cy - center_hat[1]
        center_error = math.hypot(dx, dy)
        area_abs = abs(row["pixel_count"] - area_hat)
        area_relative_pct = 100 * area_abs / row["pixel_count"]
        center_errors.append((frame, center_error))
        area_errors.append((frame, area_abs, area_relative_pct))
        per_frame.append(
            {
                "frame": frame,
                "is_anchor": frame in anchors,
                "center_error_px": round(center_error, 6),
                "center_error_xy_px": [round(dx, 6), round(dy, 6)],
                "mask_pixel_count_interpolation_abs_error": round(area_abs, 6),
                "mask_pixel_count_interpolation_relative_error_pct": round(
                    area_relative_pct, 6
                ),
            }
        )

    omitted_center = [x for x in center_errors if x[0] not in anchors]
    omitted_area = [x for x in area_errors if x[0] not in anchors]
    x_centers = [row["centroid_xy_pixels"][0] for row in rows]
    interval_motion = []
    for left, right in zip(anchors, anchors[1:]):
        interval_motion.append(
            {
                "frame_interval": [left, right],
                "x_center_displacement_px": round(x_centers[right] - x_centers[left], 6),
                "mean_x_displacement_px_per_rgb_interval": round(
                    (x_centers[right] - x_centers[left]) / (right - left), 6
                ),
            }
        )

    transient_window_coverage = []
    for duration in range(1, min(len(rows), 6) + 1):
        starts = range(len(rows) - duration + 1)
        detected_starts = [
            start
            for start in starts
            if any(start <= anchor < start + duration for anchor in anchors)
        ]
        possible_count = len(rows) - duration + 1
        detected_count = len(detected_starts)
        transient_window_coverage.append(
            {
                "event_duration_frames": duration,
                "possible_start_positions": possible_count,
                "start_positions_intersecting_an_anchor": detected_count,
                "start_positions_missed": possible_count - detected_count,
                "coverage_if_start_uniform_over_positions_pct": round(
                    100 * detected_count / possible_count, 6
                ),
            }
        )

    result = {
        "experiment_id": "EXP-M0-FIVE-POINT-TEMPORAL-READOUT-PROBE-001",
        "status": "completed_cpu_posthoc_proxy_sampling_probe",
        "question": (
            "For this saved M0 clip, how closely does piecewise-linear interpolation "
            "from the five nominal latent-time anchors reproduce the existing "
            "17-frame red-mask center and area proxy traces, and how often would "
            "the same schedule intersect brief contiguous event windows?"
        ),
        "scope": (
            "Post-hoc CPU color-threshold analysis of saved RGB frames; no model "
            "inference or learned image segmentation."
        ),
        "sampling": {
            "source_frame_count": len(rows),
            "anchor_frames": anchors,
            "unsampled_frame_count": len(rows) - len(anchors),
            "interpolation": "piecewise linear between adjacent anchors",
            "center_measure": (
                "RGB red-threshold mask centroid; 36 combinations from the existing "
                "4x3x3 threshold grid, with (150,125,125) as the baseline"
            ),
            "area_measure": "RGB red-threshold mask pixel count",
            "transient_window_coverage": (
                "Enumerate every contiguous event window of duration 1..6 frames; "
                "count a window as seen when it intersects at least one anchor."
            ),
        },
        "center_proxy_error": {
            "all_frames_mean_px": round(mean(x[1] for x in center_errors), 6),
            "all_frames_rmse_px": round(
                math.sqrt(mean(x[1] ** 2 for x in center_errors)), 6
            ),
            "unsampled_frames_mean_px": round(mean(x[1] for x in omitted_center), 6),
            "unsampled_frames_rmse_px": round(
                math.sqrt(mean(x[1] ** 2 for x in omitted_center)), 6
            ),
            "unsampled_frames_max_px": round(
                max(omitted_center, key=lambda x: x[1])[1], 6
            ),
            "unsampled_max_error_frame": max(omitted_center, key=lambda x: x[1])[0],
        },
        "mask_area_proxy_error": {
            "unsampled_frames_mean_relative_error_pct": round(
                mean(x[2] for x in omitted_area), 6
            ),
            "unsampled_frames_max_relative_error_pct": round(
                max(omitted_area, key=lambda x: x[2])[2], 6
            ),
            "unsampled_max_error_frame": max(omitted_area, key=lambda x: x[2])[0],
        },
        "threshold_robustness": {
            "threshold_case_count": len(threshold_cases),
            "full_coverage_case_count": len(fully_covered_cases),
            "baseline_trace_recomputed_from_rgb_frames": {
                "max_centroid_difference_from_full_precision_saved_baseline_px": round(
                    max(x for x in baseline_center_delta if x is not None), 9
                ),
                "max_mask_pixel_count_difference": max(baseline_area_delta),
                "max_centroid_difference_from_rounded_motion_audit_px": round(
                    max(rounded_audit_center_delta), 9
                ),
            },
            "unsampled_center_mean_error_px_across_thresholds": range_summary(
                "unsampled_center_mean_px"
            ),
            "unsampled_center_max_error_px_across_thresholds": range_summary(
                "unsampled_center_max_px"
            ),
            "unsampled_area_mean_relative_error_pct_across_thresholds": range_summary(
                "unsampled_area_mean_relative_error_pct"
            ),
            "per_threshold_results": threshold_cases,
        },
        "coarse_motion": {
            "full_17_frame_x_center_strictly_increasing": all(
                x_centers[i + 1] > x_centers[i] for i in range(len(x_centers) - 1)
            ),
            "anchor_interval_x_displacements": interval_motion,
        },
        "transient_event_sampling": transient_window_coverage,
        "per_frame_interpolation_errors": per_frame,
        "interpretation": (
            "Five anchors preserve this clip's smooth proxy center and mask-area "
            "trends under linear interpolation. Across all 36 red-threshold choices, "
            "all frames remain scoreable; unsampled center mean error ranges from "
            "0.7189 to 0.9468 px, and mean mask-area error ranges from 0.4363% to "
            "1.5957%. However, schedule enumeration shows that the same five-point "
            "readout can miss brief events: 70.6% of possible one-frame windows "
            "and 50.0% of two-frame windows do not intersect an anchor. This is one "
            "text-only sample plus a schedule-only calculation; it does not validate "
            "the observer, prove object motion fidelity, or establish general "
            "sampling sufficiency."
        ),
        "compute": {
            "model_inference": False,
            "gpu_used": False,
            "learned_segmentation": False,
            "rgb_threshold_masks_recomputed": True,
        },
        "follow_up_resource_needs": {
            "for_this_probe": "None beyond the saved M0 RGB frames and CPU analysis.",
            "for_generalization": (
                "Use several held-out clips/seeds with independent framewise "
                "visibility, identity, and box labels; no additional GPU is needed "
                "for the sampling calculation itself."
            ),
        },
        "inputs_sha256": {
            str(FRAME_AUDIT.relative_to(ROOT)): sha256(FRAME_AUDIT),
            str(BASELINE_TRACE.relative_to(ROOT)): sha256(BASELINE_TRACE),
            str(SAMPLING_AUDIT.relative_to(ROOT)): sha256(SAMPLING_AUDIT),
            str(THRESHOLD_AUDIT.relative_to(ROOT)): sha256(THRESHOLD_AUDIT),
        },
        "rgb_frame_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in frame_paths
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
