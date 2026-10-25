#!/usr/bin/env python3
"""Measure dense adjacent-frame RGB change against the saved Wan M0 anchors.

This is a bounded pixel-change proxy on existing PNGs, not optical flow,
object tracking, or ground-truth motion. It uses no model or latent tensors.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "m0/runs/wan_t4_2026-10-02_retry5"
FRAME_DIR = RUN_DIR / "run_001/frames"
MANIFEST_PATH = RUN_DIR / "manifest.json"
OUTPUT_PATH = ROOT / "research/m0_retry5_anchor_motion_proxy_v1.json"
FRAME_COUNT = 17
ANCHORS = [0, 4, 8, 12, 16]


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_name(index: int) -> str:
    return f"frame_{index:03d}.png"


def normalized_rgb_mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} != {b.shape}")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(dtype=np.float64) / 255.0)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    frames: list[np.ndarray] = []
    frame_hashes: list[dict[str, str | int]] = []
    expected_shapes: set[tuple[int, ...]] = set()

    # Verify the selected frame file hashes before decoding any PNG pixels.
    paths = [FRAME_DIR / frame_name(i) for i in range(FRAME_COUNT)]
    for i, path in enumerate(paths):
        relative_to_run = f"run_001/frames/{path.name}"
        expected = manifest["files_sha256"].get(relative_to_run)
        if expected is None or not path.is_file():
            raise RuntimeError(f"Missing frame or manifest hash: {relative_to_run}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frame hash mismatch for {relative_to_run}: {actual} != {expected}")
        frame_hashes.append({"frame": i, "path": str(path.relative_to(ROOT)), "sha256": actual})

    for path in paths:
        with Image.open(path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            frame = np.asarray(image, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected H×W×3 RGB image for {path}, got {frame.shape}")
        expected_shapes.add(tuple(frame.shape))
        frames.append(frame)

    if len(expected_shapes) != 1:
        raise RuntimeError(f"Inconsistent frame dimensions: {sorted(expected_shapes)}")
    height, width, channels = next(iter(expected_shapes))

    adjacent = []
    edge_maes = []
    for i in range(FRAME_COUNT - 1):
        score = normalized_rgb_mae(frames[i], frames[i + 1])
        edge_maes.append(score)
        adjacent.append(
            {
                "from_frame": i,
                "to_frame": i + 1,
                "normalized_rgb_mae": score,
                "rgb_mae_8bit_levels": score * 255.0,
            }
        )

    anchor_intervals = []
    for start, end in zip(ANCHORS, ANCHORS[1:]):
        step_scores = edge_maes[start:end]
        endpoint = normalized_rgb_mae(frames[start], frames[end])
        path_length = float(sum(step_scores))
        anchor_intervals.append(
            {
                "start_anchor": start,
                "end_anchor": end,
                "anchor_endpoint_normalized_rgb_mae": endpoint,
                "adjacent_transitions": adjacent[start:end],
                "adjacent_path_length_normalized": path_length,
                "mean_adjacent_step_normalized_rgb_mae": path_length / (end - start),
                "path_change_not_visible_in_endpoint_contrast": max(0.0, path_length - endpoint),
                "fraction_of_adjacent_path_length_not_represented_by_endpoint_contrast": (
                    max(0.0, 1.0 - endpoint / path_length) if path_length > 0 else 0.0
                ),
                "peak_adjacent_to_endpoint_mae_ratio": (
                    max(step_scores) / endpoint if endpoint > 0 else None
                ),
            }
        )

    dense_path = float(sum(edge_maes))
    anchor_chords = float(sum(x["anchor_endpoint_normalized_rgb_mae"] for x in anchor_intervals))
    result = {
        "audit_id": "EXP-M0-RETRY5-ANCHOR-MOTION-PROXY-001",
        "status": "completed_cpu_rgb_difference_proxy",
        "created_date": "2026-10-16",
        "scope": "Existing hash-verified retry5 RGB PNGs only; no tensor or model access, inference, decode, human labels, GPU, or cloud runtime.",
        "source_run": "m0/runs/wan_t4_2026-10-02_retry5/run_001",
        "source_manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha256(MANIFEST_PATH),
        },
        "frame_count": FRAME_COUNT,
        "frame_shape_hwc": [height, width, channels],
        "frame_sha256": frame_hashes,
        "anchors": ANCHORS,
        "method": {
            "name": "adjacent RGB mean absolute difference",
            "formula": "mean(abs(uint8_RGB(frame_t) - uint8_RGB(frame_t+1))) / 255",
            "range_per_pair": [0.0, 1.0],
            "aggregation": "All H×W×3 channel values; full frame; no mask, optical flow, threshold, registration, or object segmentation.",
            "interpretation": "Apparent pixel-change proxy. It mixes object motion, background/camera changes, lighting, texture/flicker, and image rendering effects; it is not optical flow or a motion ground truth."
        },
        "adjacent_frame_transitions": adjacent,
        "anchor_interval_comparison": anchor_intervals,
        "summary": {
            "adjacent_transition_count": len(adjacent),
            "anchor_endpoint_contrast_count": len(anchor_intervals),
            "intermediate_frames_not_in_anchor_set": [i for i in range(FRAME_COUNT) if i not in ANCHORS],
            "anchor_only_omits_direct_measurement_of_all_adjacent_one_frame_transitions": True,
            "mean_adjacent_normalized_rgb_mae": float(np.mean(edge_maes)),
            "median_adjacent_normalized_rgb_mae": float(np.median(edge_maes)),
            "max_adjacent_normalized_rgb_mae": float(max(edge_maes)),
            "dense_adjacent_path_length_normalized": dense_path,
            "sum_of_four_anchor_endpoint_contrasts_normalized": anchor_chords,
            "fraction_of_dense_path_length_not_represented_by_anchor_endpoint_contrasts": (
                max(0.0, 1.0 - anchor_chords / dense_path) if dense_path > 0 else 0.0
            ),
        },
        "limitations": [
            "One generated clip, one prompt, and one seed; exploratory only.",
            "Pixel differences are not a semantic/object motion measure and may respond to background texture, lighting, flicker, and encoding/rendering changes.",
            "The four anchor endpoint contrasts do not reveal when or how changes occurred inside each four-frame interval; 12 intermediate frames and all 16 adjacent transition magnitudes are omitted by anchor-only sampling.",
            "This does not alter the registered recommendation to generate full sequences and score all decoded RGB frames."
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)}; verified {len(frame_hashes)} frame hashes; "
        f"computed {len(adjacent)} adjacent transitions and {len(anchor_intervals)} anchor intervals."
    )


if __name__ == "__main__":
    main()
