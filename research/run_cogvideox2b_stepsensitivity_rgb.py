#!/usr/bin/env python3
"""Run the preregistered RGB-only scoreability check on two archived clips."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "research/cogvideox2b_stepsensitivity_inputs_v1.json"
PROTOCOL = ROOT / "research/2026-10-12_cogvideox2b_stepsensitivity_rgb_protocol.md"
OUTPUT = ROOT / "research/cogvideox2b_stepsensitivity_rgb_v1.json"
MINIMUM_MASK_PIXELS = 300
RED_THRESHOLDS = [130, 150, 170, 190]
GREEN_THRESHOLDS = [100, 125, 150]
BLUE_THRESHOLDS = [100, 125, 150]
MATCHED_MANIFEST_FIELDS = [
    "model", "prompt", "seed", "width", "height", "guidance_scale"
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_case(
    frames: list[np.ndarray], hashes: dict[str, str], r_gt: int, g_lt: int, b_lt: int
) -> dict:
    rows = []
    for index, rgb in enumerate(frames):
        mask = (
            (rgb[:, :, 0] > r_gt)
            & (rgb[:, :, 1] < g_lt)
            & (rgb[:, :, 2] < b_lt)
        )
        ys, xs = np.nonzero(mask)
        count = int(xs.size)
        center = [float(xs.mean()), float(ys.mean())] if count >= MINIMUM_MASK_PIXELS else None
        rows.append({
            "frame": index,
            "raw_red_mask_pixels": count,
            "mask_area_fraction": count / float(rgb.shape[0] * rgb.shape[1]),
            "centroid_xy_px": center,
        })

    centers = [row["centroid_xy_px"] for row in rows]
    width, height = int(frames[0].shape[1]), int(frames[0].shape[0])
    valid_count = sum(center is not None for center in centers)
    full_coverage = valid_count == len(frames)
    adjacent = []
    path_length = 0.0
    for left, right in zip(centers, centers[1:]):
        if left is None or right is None:
            continue
        dx = (right[0] - left[0]) / width
        dy = (right[1] - left[1]) / height
        adjacent.append(dx)
        path_length += math.hypot(dx, dy)

    area = [row["mask_area_fraction"] for row in rows]
    area_mean = statistics.mean(area)
    endpoint_dx = endpoint_dy = endpoint_area_ratio = None
    direction = None
    if full_coverage:
        endpoint_dx = (centers[-1][0] - centers[0][0]) / width
        endpoint_dy = (centers[-1][1] - centers[0][1]) / height
        direction = "right" if endpoint_dx > 0 else "left" if endpoint_dx < 0 else "stationary"
        if area[0] > 0:
            endpoint_area_ratio = area[-1] / area[0]

    total_steps = len(adjacent)
    duplicate_transitions = [
        [index, index + 1]
        for index in range(len(rows) - 1)
        if hashes[f"frame_{index:02d}.png"] == hashes[f"frame_{index + 1:02d}.png"]
    ]
    return {
        "threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
        "valid_frame_count": valid_count,
        "coverage_fraction": valid_count / len(frames),
        "fully_covered": full_coverage,
        "endpoint_dx_normalized_by_width": endpoint_dx,
        "endpoint_dy_normalized_by_height": endpoint_dy,
        "endpoint_direction": direction,
        "valid_adjacent_step_count": total_steps,
        "right_step_fraction": sum(dx > 0 for dx in adjacent) / total_steps if total_steps else None,
        "left_step_fraction": sum(dx < 0 for dx in adjacent) / total_steps if total_steps else None,
        "still_step_fraction": sum(dx == 0 for dx in adjacent) / total_steps if total_steps else None,
        "normalized_centroid_path_length": path_length,
        "endpoint_area_ratio": endpoint_area_ratio,
        "mask_area_fraction_mean": area_mean,
        "mask_area_fraction_cv": statistics.pstdev(area) / area_mean if area_mean else None,
        "exact_duplicate_adjacent_transitions": duplicate_transitions,
        "per_frame": rows,
    }


def main() -> None:
    pin = json.loads(INPUTS.read_text(encoding="utf-8"))
    if pin.get("analysis_id") != "EXP-COGVIDEOX2B-STEP-SENSITIVITY-RGB-001":
        raise ValueError("Unexpected analysis ID in frozen input file")

    runs = []
    for item in pin["runs"]:
        manifest_path = Path(item["manifest_path"])
        if sha256(manifest_path) != item["manifest_sha256"]:
            raise ValueError(f"Manifest hash changed for {item['run_id']}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in item["manifest_fields"].items():
            if manifest.get(key) != expected:
                raise ValueError(f"Manifest field {key} changed for {item['run_id']}")
        if int(manifest.get("frames", -1)) != int(item["frame_count"]):
            raise ValueError(f"Manifest frame count changed for {item['run_id']}")

        run_dir = manifest_path.parent
        expected_names = [frame["name"] for frame in item["frames"]]
        paths = sorted((run_dir / "frames").glob("frame_*.png"))
        if [path.name for path in paths] != expected_names:
            raise ValueError(f"Frame names changed for {item['run_id']}")

        frames, frame_hashes = [], {}
        for path, expected in zip(paths, item["frames"]):
            digest = sha256(path)
            if digest != expected["sha256"]:
                raise ValueError(f"Frame hash changed: {path}")
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            if [int(rgb.shape[1]), int(rgb.shape[0])] != expected["size_px"]:
                raise ValueError(f"Frame dimensions changed: {path}")
            frames.append(rgb)
            frame_hashes[path.name] = digest

        cases = [
            analyze_case(frames, frame_hashes, r_gt, g_lt, b_lt)
            for r_gt, g_lt, b_lt in product(RED_THRESHOLDS, GREEN_THRESHOLDS, BLUE_THRESHOLDS)
        ]
        complete = [case for case in cases if case["fully_covered"]]
        direction_counts = {
            direction: sum(case["endpoint_direction"] == direction for case in complete)
            for direction in ("left", "right", "stationary")
        }
        max_raw_mask_by_frame = [
            max(case["per_frame"][frame_index]["raw_red_mask_pixels"] for case in cases)
            for frame_index in range(len(frames))
        ]
        runs.append({
            "run_id": item["run_id"],
            "manifest_fields": item["manifest_fields"],
            "manifest_sha256": item["manifest_sha256"],
            "frame_count": len(frames),
            "frame_size_px": item["frames"][0]["size_px"],
            "frame_sha256": frame_hashes,
            "measurement": {
                "threshold_case_count": len(cases),
                "fully_covered_case_count": len(complete),
                "threshold_cases_with_any_missing_frame": len(cases) - len(complete),
                "threshold_cases_with_any_valid_centroid": sum(case["valid_frame_count"] > 0 for case in cases),
                "valid_frame_observations": sum(case["valid_frame_count"] for case in cases),
                "valid_frame_observation_count_total": len(cases) * len(frames),
                "maximum_valid_frames_in_any_threshold_case": max(case["valid_frame_count"] for case in cases),
                "maximum_raw_mask_pixels_any_threshold_frame": max(max_raw_mask_by_frame),
                "maximum_raw_mask_pixels_by_frame_across_thresholds": max_raw_mask_by_frame,
                "endpoint_direction_counts": direction_counts,
                "endpoint_dx_normalized_range_when_fully_covered": (
                    [
                        min(case["endpoint_dx_normalized_by_width"] for case in complete),
                        max(case["endpoint_dx_normalized_by_width"] for case in complete),
                    ] if complete else None
                ),
                "endpoint_dx_normalized_median_when_fully_covered": (
                    statistics.median(case["endpoint_dx_normalized_by_width"] for case in complete)
                    if complete else None
                ),
                "exact_duplicate_adjacent_transitions_any_threshold": sorted({
                    tuple(pair) for case in cases for pair in case["exact_duplicate_adjacent_transitions"]
                }),
            },
            "cases": cases,
        })

    if len(runs) != 2:
        raise ValueError("Expected exactly two preregistered run entries")
    left, right = runs
    matched = all(
        left["manifest_fields"].get(field) == right["manifest_fields"].get(field)
        for field in MATCHED_MANIFEST_FIELDS
    ) and left["frame_count"] == right["frame_count"] and left["frame_size_px"] == right["frame_size_px"]
    left_map = {json.dumps(case["threshold"], sort_keys=True): case for case in left["cases"]}
    right_map = {json.dumps(case["threshold"], sort_keys=True): case for case in right["cases"]}
    paired_complete = [
        (left_map[key], right_map[key]) for key in left_map
        if left_map[key]["fully_covered"] and right_map[key]["fully_covered"]
    ]

    result = {
        "experiment_id": pin["analysis_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "inputs_manifest": str(INPUTS.relative_to(ROOT)),
        "inputs_manifest_sha256": sha256(INPUTS),
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "compute": "CPU-only thresholding on archived RGB PNGs; no model, inference, GPU, or human labels",
        "matched_manifest_comparison": {
            "fields_matched_by_manifest": MATCHED_MANIFEST_FIELDS,
            "nominally_matched_on_recorded_frame_count_and_dimensions": matched,
            "recorded_step_counts": [left["manifest_fields"]["denoising_steps"], right["manifest_fields"]["denoising_steps"]],
            "threshold_cases_fully_covered_in_both_runs": len(paired_complete),
            "paired_endpoint_dx_difference_range_30_minus_8": (
                [
                    min(b["endpoint_dx_normalized_by_width"] - a["endpoint_dx_normalized_by_width"] for a, b in paired_complete),
                    max(b["endpoint_dx_normalized_by_width"] - a["endpoint_dx_normalized_by_width"] for a, b in paired_complete),
                ] if paired_complete else None
            ),
            "causal_step_count_interpretation_allowed": False,
        },
        "measurement_spec": {
            "minimum_mask_pixels": MINIMUM_MASK_PIXELS,
            "red_greater_than": RED_THRESHOLDS,
            "green_less_than": GREEN_THRESHOLDS,
            "blue_less_than": BLUE_THRESHOLDS,
            "threshold_case_count_per_run": len(left["cases"]),
            "normalization": "endpoint x displacement divided by frame width; endpoint y displacement divided by frame height",
            "low_coverage_policy": "do not infer direction from partial detections; preserve missing frames",
        },
        "runs": runs,
        "limitations": [
            "One saved output per recorded setting; no independent seed replication.",
            "Matching manifest fields do not establish identical code, checkpoint revision, environment, or effective initial noise.",
            "Both clips use 480x320, below the model's documented 720x480 output resolution.",
            "A 300-pixel absolute mask floor is only comparable within this matched-resolution pair.",
            "RGB threshold masks are not ground-truth segmentation or an independent tracker.",
            "No latent semantics, latent-to-frame alignment, causal intervention, policy efficacy, or downstream robotics claim follows.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["matched_manifest_comparison"], indent=2))
    for run in result["runs"]:
        print(json.dumps({"run_id": run["run_id"], **run["measurement"]}, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
