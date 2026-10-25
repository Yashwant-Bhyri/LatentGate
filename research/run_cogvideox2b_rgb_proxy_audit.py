#!/usr/bin/env python3
"""Run the preregistered CPU-only RGB proxy audit on saved CogVideoX-2B frames."""

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
RUN_DIR = Path("/Users/yash/Documents/Resume/video-diffusion-lab/run/cogvideox_2b_chunk13")
INPUTS = ROOT / "research/cogvideox2b_rgb_proxy_inputs_v1.json"
PROTOCOL = ROOT / "research/2026-10-12_cogvideox2b_rgb_proxy_audit.md"
OUTPUT = ROOT / "research/cogvideox2b_rgb_proxy_audit_v1.json"
MINIMUM_MASK_PIXELS = 300
RED_THRESHOLDS = [130, 150, 170, 190]
GREEN_THRESHOLDS = [100, 125, 150]
BLUE_THRESHOLDS = [100, 125, 150]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_case(
    rgb_frames: list[np.ndarray], r_gt: int, g_lt: int, b_lt: int
) -> dict:
    frame_rows = []
    for frame_idx, rgb in enumerate(rgb_frames):
        mask = (
            (rgb[:, :, 0] > r_gt)
            & (rgb[:, :, 1] < g_lt)
            & (rgb[:, :, 2] < b_lt)
        )
        ys, xs = np.nonzero(mask)
        count = int(len(xs))
        center = None
        if count >= MINIMUM_MASK_PIXELS:
            center = [float(xs.mean()), float(ys.mean())]
        frame_rows.append(
            {
                "frame": frame_idx,
                "mask_pixels": count,
                "mask_area_fraction": count / float(rgb.shape[0] * rgb.shape[1]),
                "centroid_xy_px": center,
            }
        )

    valid_centers = [row["centroid_xy_px"] for row in frame_rows]
    valid_count = sum(center is not None for center in valid_centers)
    fully_covered = valid_count == len(frame_rows)
    width = rgb_frames[0].shape[1]
    height = rgb_frames[0].shape[0]

    adjacent_dx_norm = []
    path_length_norm = 0.0
    for a, b in zip(valid_centers, valid_centers[1:]):
        if a is None or b is None:
            continue
        dx = (b[0] - a[0]) / width
        dy = (b[1] - a[1]) / height
        adjacent_dx_norm.append(dx)
        path_length_norm += math.hypot(dx, dy)

    area = [row["mask_area_fraction"] for row in frame_rows]
    area_mean = statistics.mean(area)
    endpoint_dx = endpoint_dy = endpoint_area_ratio = None
    direction = None
    if fully_covered:
        first, last = valid_centers[0], valid_centers[-1]
        endpoint_dx = (last[0] - first[0]) / width
        endpoint_dy = (last[1] - first[1]) / height
        direction = "right" if endpoint_dx > 0 else "left" if endpoint_dx < 0 else "stationary"
        if area[0] > 0:
            endpoint_area_ratio = area[-1] / area[0]

    steps = len(adjacent_dx_norm)
    right_steps = sum(dx > 0 for dx in adjacent_dx_norm)
    left_steps = sum(dx < 0 for dx in adjacent_dx_norm)
    still_steps = sum(dx == 0 for dx in adjacent_dx_norm)
    return {
        "threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
        "valid_frame_count": valid_count,
        "coverage_fraction": valid_count / len(frame_rows),
        "fully_covered": fully_covered,
        "endpoint_dx_normalized_by_width": endpoint_dx,
        "endpoint_dy_normalized_by_height": endpoint_dy,
        "endpoint_direction": direction,
        "valid_adjacent_step_count": steps,
        "right_step_fraction": right_steps / steps if steps else None,
        "left_step_fraction": left_steps / steps if steps else None,
        "still_step_fraction": still_steps / steps if steps else None,
        "normalized_centroid_path_length": path_length_norm,
        "endpoint_area_ratio": endpoint_area_ratio,
        "mask_area_fraction_mean": area_mean,
        "mask_area_fraction_cv": (
            statistics.pstdev(area) / area_mean if area_mean else None
        ),
        "per_frame": frame_rows,
    }


def main() -> None:
    pin = json.loads(INPUTS.read_text(encoding="utf-8"))
    manifest_path = Path(pin["manifest_path"])
    if sha256(manifest_path) != pin["manifest_sha256"]:
        raise ValueError("CogVideoX run manifest hash differs from the preregistered value")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["frames"]) != pin["frame_count"]:
        raise ValueError("Manifest frame count differs from the preregistration")
    if [int(manifest["width"]), int(manifest["height"])] != pin["frame_size_px"]:
        raise ValueError("Manifest frame dimensions differ from the preregistration")

    frames_dir = RUN_DIR / "frames"
    paths = sorted(frames_dir.glob("frame_*.png"))
    names = [path.name for path in paths]
    expected_names = sorted(pin["frames"])
    if names != expected_names:
        raise ValueError(f"Frame names differ from preregistration: {names}")

    hashes = {path.name: sha256(path) for path in paths}
    if hashes != pin["frames"]:
        raise ValueError("At least one PNG hash differs from the preregistered input map")

    rgb_frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        if [int(rgb.shape[1]), int(rgb.shape[0])] != pin["frame_size_px"]:
            raise ValueError(f"Unexpected frame dimensions for {path.name}: {rgb.shape}")
        rgb_frames.append(rgb)

    cases = [
        summarize_case(rgb_frames, r_gt, g_lt, b_lt)
        for r_gt, g_lt, b_lt in product(RED_THRESHOLDS, GREEN_THRESHOLDS, BLUE_THRESHOLDS)
    ]
    complete = [case for case in cases if case["fully_covered"]]

    def values(key: str) -> list[float]:
        return [float(case[key]) for case in complete if case[key] is not None]

    dx_values = values("endpoint_dx_normalized_by_width")
    dy_values = values("endpoint_dy_normalized_by_height")
    directions = {name: sum(case["endpoint_direction"] == name for case in complete) for name in ("left", "right", "stationary")}
    duplicate_transitions = [
        [i, i + 1] for i in range(len(paths) - 1) if hashes[paths[i].name] == hashes[paths[i + 1].name]
    ]

    protocol_sha = sha256(PROTOCOL)
    result = {
        "experiment_id": "EXP-LATENTGATE-COGVIDEOX2B-RGB-PROXY-001",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "compute": "CPU-only RGB threshold masks; no model, inference, GPU, or human labels",
        "source": {
            "run_directory": str(RUN_DIR),
            "model": manifest["model"],
            "prompt": manifest["prompt"],
            "seed": manifest["seed"],
            "manifest_sha256": pin["manifest_sha256"],
            "frame_count": len(paths),
            "frame_size_px": pin["frame_size_px"],
            "frame_sha256": hashes,
            "order_basis": "zero-padded frame filename order; latent-time mapping not established",
        },
        "measurement": {
            "minimum_mask_pixels": MINIMUM_MASK_PIXELS,
            "red_greater_than": RED_THRESHOLDS,
            "green_less_than": GREEN_THRESHOLDS,
            "blue_less_than": BLUE_THRESHOLDS,
            "threshold_case_count": len(cases),
            "fully_covered_case_count": len(complete),
            "endpoint_direction_counts": directions,
            "leftward_endpoint_fraction": directions["left"] / len(complete) if complete else None,
            "endpoint_dx_normalized_range": [min(dx_values), max(dx_values)] if dx_values else None,
            "endpoint_dx_normalized_median": statistics.median(dx_values) if dx_values else None,
            "endpoint_dy_normalized_range": [min(dy_values), max(dy_values)] if dy_values else None,
            "exact_duplicate_adjacent_transitions": duplicate_transitions,
        },
        "cases": cases,
        "limitations": [
            "One generated clip and one seed; no sample-level generalization.",
            "RGB color mask is a proxy, not ground-truth segmentation or an independent observer.",
            "File-name order is used for RGB measurement; it does not establish latent axis or latent-to-frame alignment.",
            "No causal intervention, physical trajectory, policy efficacy, or downstream robotics claim follows.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["measurement"], indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
