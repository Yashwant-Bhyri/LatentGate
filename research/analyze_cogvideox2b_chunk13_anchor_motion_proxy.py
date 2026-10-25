#!/usr/bin/env python3
"""Measure dense adjacent RGB change against anchors in CogVideoX-2B chunk13.

Uses the same bounded whole-frame normalized RGB MAE as the Wan retry5
anchor-motion proxy. No models, latents, inference, labels, or optical flow.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_MAP_PATH = ROOT / "research/cogvideox2b_rgb_proxy_inputs_v1.json"
OUTPUT_PATH = ROOT / "research/cogvideox2b_chunk13_anchor_motion_proxy_v1.json"
ANCHORS = [0, 4, 8, 12, 15]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_rgb_mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} != {b.shape}")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(dtype=np.float64) / 255.0)


def main() -> None:
    input_map = json.loads(INPUT_MAP_PATH.read_text())
    map_digest = sha256(INPUT_MAP_PATH)
    manifest_path = pathlib.Path(input_map["manifest_path"])
    if not manifest_path.is_file():
        raise RuntimeError(f"Pinned manifest is missing: {manifest_path}")
    actual_manifest_digest = sha256(manifest_path)
    if actual_manifest_digest != input_map["manifest_sha256"]:
        raise RuntimeError(
            f"Pinned manifest hash mismatch: {actual_manifest_digest} != {input_map['manifest_sha256']}"
        )
    manifest = json.loads(manifest_path.read_text())
    frame_count = int(input_map["frame_count"])
    expected_size = tuple(int(x) for x in input_map["frame_size_px"])
    if frame_count != 16 or manifest.get("frames") != frame_count:
        raise RuntimeError(f"Unexpected frame count: input map={frame_count}, manifest={manifest.get('frames')}")
    if sorted(ANCHORS) != ANCHORS or ANCHORS[0] != 0 or ANCHORS[-1] != frame_count - 1:
        raise RuntimeError("Anchor schedule is malformed for the pinned sequence.")
    if set(input_map["frames"]) != {f"frame_{i:02d}.png" for i in range(frame_count)}:
        raise RuntimeError("Input map does not contain exactly the 16 expected PNG frames.")

    frame_dir = manifest_path.parent / "frames"
    frames: list[np.ndarray] = []
    frame_hashes: list[dict[str, str | int]] = []
    for index in range(frame_count):
        name = f"frame_{index:02d}.png"
        path = frame_dir / name
        if not path.is_file():
            raise RuntimeError(f"Missing pinned RGB frame: {path}")
        expected_hash = input_map["frames"][name]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frame hash mismatch for {name}: {actual_hash} != {expected_hash}")
        frame_hashes.append({"frame": index, "path": str(path), "sha256": actual_hash})

    # Decode PNG pixels only after all expected frame hashes have passed.
    for index in range(frame_count):
        name = f"frame_{index:02d}.png"
        with Image.open(frame_dir / name) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            if image.size != expected_size:
                raise RuntimeError(f"Unexpected dimensions for {name}: {image.size} != {expected_size}")
            frame = np.asarray(image, dtype=np.uint8)
        frames.append(frame)

    adjacent = []
    edge_maes = []
    for index in range(frame_count - 1):
        score = normalized_rgb_mae(frames[index], frames[index + 1])
        edge_maes.append(score)
        adjacent.append(
            {
                "from_frame": index,
                "to_frame": index + 1,
                "normalized_rgb_mae": score,
                "rgb_mae_8bit_levels": score * 255.0,
            }
        )

    intervals = []
    for start, end in zip(ANCHORS, ANCHORS[1:]):
        step_scores = edge_maes[start:end]
        endpoint = normalized_rgb_mae(frames[start], frames[end])
        path_length = float(sum(step_scores))
        intervals.append(
            {
                "start_anchor": start,
                "end_anchor": end,
                "interval_length_frames": end - start,
                "anchor_endpoint_normalized_rgb_mae": endpoint,
                "adjacent_transitions": adjacent[start:end],
                "adjacent_path_length_normalized": path_length,
                "mean_adjacent_step_normalized_rgb_mae": path_length / (end - start),
                "path_change_not_visible_in_endpoint_contrast": max(0.0, path_length - endpoint),
                "fraction_of_adjacent_path_length_not_represented_by_endpoint_contrast": (
                    max(0.0, 1.0 - endpoint / path_length) if path_length > 0 else 0.0
                ),
                "peak_adjacent_to_endpoint_mae_ratio": max(step_scores) / endpoint if endpoint > 0 else None,
            }
        )

    path_total = float(sum(edge_maes))
    chord_total = float(sum(interval["anchor_endpoint_normalized_rgb_mae"] for interval in intervals))
    result = {
        "audit_id": "EXP-COGVIDEOX2B-CHUNK13-ANCHOR-MOTION-PROXY-001",
        "status": "completed_cpu_rgb_difference_proxy",
        "created_date": "2026-10-16",
        "scope": "Existing hash-pinned CogVideoX-2B chunk13 RGB PNG sequence only; no model, latent, inference, label, GPU, or cloud access.",
        "source_run": str(manifest_path.parent),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": actual_manifest_digest,
        },
        "pinned_input_map": {
            "path": str(INPUT_MAP_PATH.relative_to(ROOT)),
            "sha256": map_digest,
            "declared_manifest_sha256": input_map["manifest_sha256"],
        },
        "frame_count": frame_count,
        "frame_shape_hwc": [expected_size[1], expected_size[0], 3],
        "frame_sha256": frame_hashes,
        "anchors": ANCHORS,
        "interval_lengths": [end - start for start, end in zip(ANCHORS, ANCHORS[1:])],
        "method": {
            "name": "adjacent RGB mean absolute difference",
            "formula": "mean(abs(uint8_RGB(frame_t) - uint8_RGB(frame_t+1))) / 255",
            "range_per_pair": [0.0, 1.0],
            "aggregation": "All H×W×3 channel values; full frame; no mask, optical flow, threshold, registration, or object segmentation.",
            "interpretation": "Pixel-change proxy only; it can respond to object motion, background/camera changes, lighting, texture/flicker, and rendering effects.",
            "same_definition_as": "research/2026-10-16_m0_retry5_anchor_motion_proxy.md",
        },
        "adjacent_frame_transitions": adjacent,
        "anchor_interval_comparison": intervals,
        "summary": {
            "adjacent_transition_count": len(adjacent),
            "anchor_endpoint_contrast_count": len(intervals),
            "intermediate_frames_not_in_anchor_set": [i for i in range(frame_count) if i not in ANCHORS],
            "mean_adjacent_normalized_rgb_mae": float(np.mean(edge_maes)),
            "median_adjacent_normalized_rgb_mae": float(np.median(edge_maes)),
            "max_adjacent_normalized_rgb_mae": float(max(edge_maes)),
            "dense_adjacent_path_length_normalized": path_total,
            "sum_of_anchor_endpoint_contrasts_normalized": chord_total,
            "fraction_of_dense_path_length_not_represented_by_anchor_endpoint_contrasts": (
                max(0.0, 1.0 - chord_total / path_total) if path_total > 0 else 0.0
            ),
        },
        "limitations": [
            "One saved sequence, one prompt/seed/configuration; exploratory only.",
            "Whole-frame RGB MAE is not optical flow or an object-motion estimate; no red-mask center/area proxy is recomputed.",
            "Path excess compares the sum of adjacent whole-frame L1 changes with non-adjacent anchor endpoint contrasts. It is not an event count, recall, or the fraction of object motion missed.",
            "This does not establish general sparse-sampling sufficiency or change the registered full-sequence/all-frame policy.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)}; verified {len(frame_hashes)} pinned frames; "
        f"computed {len(adjacent)} adjacent transitions and {len(intervals)} anchor intervals."
    )


if __name__ == "__main__":
    main()
