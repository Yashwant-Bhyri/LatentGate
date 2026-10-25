#!/usr/bin/env python3
"""Run the preregistered CPU-only five-anchor RGB readout on saved frames."""

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
PREREG = ROOT / "research/cogvideox2b_50step_five_anchor_preregistration_v1.json"
PROTOCOL = ROOT / "research/2026-10-15_cogvideox2b_50step_five_anchor_readout.md"
OUTPUT = ROOT / "research/cogvideox2b_50step_five_anchor_readout_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def masked_frame(rgb: np.ndarray, r_gt: int, g_lt: int, b_lt: int) -> tuple[int, list[float] | None]:
    mask = (rgb[:, :, 0] > r_gt) & (rgb[:, :, 1] < g_lt) & (rgb[:, :, 2] < b_lt)
    ys, xs = np.nonzero(mask)
    count = int(xs.size)
    center = [float(xs.mean()), float(ys.mean())] if count >= 300 else None
    return count, center


def interpolate(values: list[float], anchors: list[int], frame: int) -> float:
    for left, right in zip(anchors, anchors[1:]):
        if left <= frame <= right:
            fraction = (frame - left) / (right - left)
            return values[left] + fraction * (values[right] - values[left])
    raise ValueError(f"Frame {frame} is outside the registered anchor range")


def summarize_case(
    rgb_frames: list[np.ndarray], r_gt: int, g_lt: int, b_lt: int,
    anchors: list[int], omitted: list[int],
) -> dict:
    rows = [masked_frame(rgb, r_gt, g_lt, b_lt) for rgb in rgb_frames]
    counts = [row[0] for row in rows]
    centers = [row[1] for row in rows]
    center_errors: list[float] = []
    area_abs_errors: list[float] = []
    area_rel_errors: list[float] = []
    for idx in omitted:
        observed = centers[idx]
        assert observed is not None
        pred_x = interpolate([center[0] for center in centers if center is not None], anchors, idx)
        pred_y = interpolate([center[1] for center in centers if center is not None], anchors, idx)
        center_errors.append(math.hypot(observed[0] - pred_x, observed[1] - pred_y))
        predicted_area = interpolate([float(value) for value in counts], anchors, idx)
        area_abs = abs(counts[idx] - predicted_area)
        area_abs_errors.append(area_abs)
        area_rel_errors.append(100.0 * area_abs / counts[idx])
    return {
        "threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
        "minimum_mask_pixels": min(counts),
        "center_error_px": {
            "mean": statistics.mean(center_errors),
            "rmse": math.sqrt(statistics.mean(error * error for error in center_errors)),
            "max": max(center_errors),
        },
        "mask_area_absolute_error_px": {
            "mean": statistics.mean(area_abs_errors),
            "max": max(area_abs_errors),
        },
        "mask_area_absolute_relative_error_percent": {
            "mean": statistics.mean(area_rel_errors),
            "max": max(area_rel_errors),
        },
    }


def event_coverage(anchors: list[int], frame_count: int) -> list[dict]:
    out = []
    anchor_set = set(anchors)
    for duration in range(1, 5):
        placements = frame_count - duration + 1
        covered = sum(
            any(frame in anchor_set for frame in range(start, start + duration))
            for start in range(placements)
        )
        out.append({
            "duration_frames": duration,
            "covered_placements": covered,
            "possible_placements": placements,
            "coverage_fraction": covered / placements,
        })
    return out


def main() -> None:
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    source = prereg["source"]
    run_dir = Path(source["run_directory"])
    manifest_path = Path(source["manifest_path"])
    pin_path = ROOT / source["prior_rgb_input_map"]

    if sha256(pin_path) != source["prior_rgb_input_map_sha256"]:
        raise ValueError("Pinned prior RGB input map hash differs")
    if sha256(manifest_path) != source["manifest_sha256"]:
        raise ValueError("Manifest hash differs from preregistration")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != pin["manifest_sha256"]:
        raise ValueError("Manifest hash differs from the frozen prior input map")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, expected in source["expected_manifest_fields"].items():
        if manifest.get(key) != expected:
            raise ValueError(f"Manifest field {key!r} differs: {manifest.get(key)!r} != {expected!r}")
    for key, expected in (
        ("baseline_config_path", "trajectory_control_v1.json"),
        ("threshold_grid_path", "m0_mask_threshold_sensitivity_v1.json"),
    ):
        p = ROOT / "research" / expected
        prereg_key = key.replace("_path", "_sha256")
        if sha256(p) != prereg["proxy"][prereg_key]:
            raise ValueError(f"Frozen proxy config hash differs for {p.name}")

    frame_names = source["frame_names"]
    frames_dir = run_dir / "frames"
    paths = [frames_dir / name for name in frame_names]
    discovered = sorted(path.name for path in frames_dir.glob("frame_*.png"))
    if discovered != frame_names:
        raise ValueError("Frame names/order differ from preregistration")
    frame_hashes = {path.name: sha256(path) for path in paths}
    if frame_hashes != pin["frames"]:
        raise ValueError("At least one RGB frame hash differs from the frozen input map")

    rgb_frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if [int(rgb.shape[1]), int(rgb.shape[0])] != source["frame_size_px"]:
            raise ValueError(f"Unexpected dimensions for {path.name}: {rgb.shape}")
        rgb_frames.append(rgb)

    grid = prereg["proxy"]["threshold_grid"]
    threshold_cases = list(product(
        grid["r_greater_than"], grid["g_less_than"], grid["b_less_than"]
    ))
    baseline = prereg["proxy"]["baseline_rgb_threshold"]
    baseline_case = (
        baseline["r_greater_than"], baseline["g_less_than"], baseline["b_less_than"]
    )
    coverage = []
    for r_gt, g_lt, b_lt in threshold_cases:
        frame_counts = [masked_frame(rgb, r_gt, g_lt, b_lt)[0] for rgb in rgb_frames]
        coverage.append({
            "threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
            "all_frames_scoreable": min(frame_counts) >= prereg["proxy"]["minimum_mask_pixels"],
            "minimum_mask_pixels": min(frame_counts),
            "below_floor_frames": [
                idx for idx, count in enumerate(frame_counts)
                if count < prereg["proxy"]["minimum_mask_pixels"]
            ],
        })
    if len(threshold_cases) != grid["case_count"]:
        raise ValueError("Threshold grid case count differs from preregistration")

    coverage_pass = all(case["all_frames_scoreable"] for case in coverage)
    result = {
        "experiment_id": prereg["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "completed" if coverage_pass else "stopped_proxy_coverage_failure",
        "preregistration": str(PREREG.relative_to(ROOT)),
        "preregistration_sha256": sha256(PREREG),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "compute": "CPU-only RGB threshold masks and interpolation; no model, inference, latent arrays, GPU, cloud runtime, or human labels",
        "source": {
            "run_directory": str(run_dir),
            "manifest_sha256": sha256(manifest_path),
            "input_map": source["prior_rgb_input_map"],
            "input_map_sha256": sha256(pin_path),
            "frame_count": len(paths),
            "frame_size_px": source["frame_size_px"],
            "frame_sha256": frame_hashes,
            "order_basis": "zero-padded filename order; latent-time alignment not established",
        },
        "coverage_gate": {
            "minimum_mask_pixels": prereg["proxy"]["minimum_mask_pixels"],
            "threshold_case_count": len(coverage),
            "fully_scoreable_threshold_case_count": sum(c["all_frames_scoreable"] for c in coverage),
            "passed": coverage_pass,
            "cases": coverage,
        },
    }

    if coverage_pass:
        anchors = prereg["method"]["anchors"]
        omitted = prereg["method"]["omitted_frames"]
        summaries = [
            summarize_case(rgb_frames, r_gt, g_lt, b_lt, anchors, omitted)
            for r_gt, g_lt, b_lt in threshold_cases
        ]
        by_threshold = {
            (case["threshold"]["r_gt"], case["threshold"]["g_lt"], case["threshold"]["b_lt"]): case
            for case in summaries
        }

        def metric_range(path: tuple[str, ...]) -> list[float]:
            values = []
            for case in summaries:
                value = case
                for key in path:
                    value = value[key]
                values.append(float(value))
            return [min(values), max(values)]

        result["measurement"] = {
            "anchors": anchors,
            "omitted_frames": omitted,
            "baseline_threshold": by_threshold[baseline_case],
            "threshold_case_count": len(summaries),
            "threshold_ranges": {
                "center_error_mean_px": metric_range(("center_error_px", "mean")),
                "center_error_rmse_px": metric_range(("center_error_px", "rmse")),
                "center_error_max_px": metric_range(("center_error_px", "max")),
                "mask_area_absolute_error_mean_px": metric_range(("mask_area_absolute_error_px", "mean")),
                "mask_area_absolute_error_max_px": metric_range(("mask_area_absolute_error_px", "max")),
                "mask_area_absolute_relative_error_mean_percent": metric_range(("mask_area_absolute_relative_error_percent", "mean")),
                "mask_area_absolute_relative_error_max_percent": metric_range(("mask_area_absolute_relative_error_percent", "max")),
            },
            "per_threshold": summaries,
            "event_window_schedule_coverage": event_coverage(anchors, len(paths)),
        }
    else:
        result["measurement"] = None
        result["stop_reason"] = "At least one preregistered threshold case had a frame below the 300-pixel floor; interpolation and event-window metrics were not computed."

    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "coverage_cases": result["coverage_gate"]["fully_scoreable_threshold_case_count"],
        "result": str(OUTPUT.relative_to(ROOT)),
        "measurement": result.get("measurement", {}).get("threshold_ranges") if result.get("measurement") else None,
    }, indent=2))


if __name__ == "__main__":
    main()
