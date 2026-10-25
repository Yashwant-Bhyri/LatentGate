#!/usr/bin/env python3
"""Apply the registered CPU Farneback diagnostic to saved Wan retry5 RGBs.

The script verifies the Wan manifest and existing RGB-MAE result hashes,
reuses the registered CogVideoX Farneback settings, and reads only RGB PNGs.
No learned model, latent arrays, inference, or decoding of latent tensors.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import cv2
import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
WAN_MANIFEST_PATH = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/manifest.json"
WAN_RGB_MAE_PATH = ROOT / "research/m0_retry5_anchor_motion_proxy_v1.json"
COG_FLOW_REFERENCE_PATH = ROOT / "research/cogvideox2b_chunk13_farneback_flow_v1.json"
OUTPUT_PATH = ROOT / "research/m0_wan_retry5_farneback_flow_v1.json"
WAN_ANCHORS = [0, 4, 8, 12, 16]
EXPECTED_COG_FLOW_SHA256 = "216656c47c99bea09f99c419b983225330db719c1df9f546894b2b6e1ca91f3d"
EXPECTED_WAN_RGB_MAE_SHA256 = "09ef38b0fe988d607004acbd1b90683f69d7a8ad61148102b5d4746a01b02ae6"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_ranks(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values)
    order = np.argsort(values_array)
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def flow_summary(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    diagonal_px: float,
    parameters: dict[str, float | int],
) -> dict[str, float]:
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        parameters["pyr_scale"],
        parameters["levels"],
        parameters["winsize"],
        parameters["iterations"],
        parameters["poly_n"],
        parameters["poly_sigma"],
        parameters["flags"],
    )
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    mean_px = float(np.mean(magnitude, dtype=np.float64))
    return {
        "mean_magnitude_px": mean_px,
        "mean_magnitude_frame_diagonal_fraction": mean_px / diagonal_px,
        "median_magnitude_px": float(np.median(magnitude)),
        "p95_magnitude_px": float(np.percentile(magnitude, 95)),
        "p99_magnitude_px": float(np.percentile(magnitude, 99)),
        "max_magnitude_px": float(np.max(magnitude)),
    }


def main() -> None:
    # Verify the exact registered method/result and reference RGB evidence first.
    cog_flow_digest = sha256(COG_FLOW_REFERENCE_PATH)
    if cog_flow_digest != EXPECTED_COG_FLOW_SHA256:
        raise RuntimeError(f"Registered CogVideoX Farneback result changed: {cog_flow_digest}")
    cog_flow_reference = json.loads(COG_FLOW_REFERENCE_PATH.read_text())
    implementation = cog_flow_reference["implementation"]
    parameters = implementation["parameters"]
    if implementation.get("function") != "cv2.calcOpticalFlowFarneback":
        raise RuntimeError("Registered reference does not identify calcOpticalFlowFarneback.")
    if cv2.__version__ != implementation["version"]:
        raise RuntimeError(
            f"OpenCV version mismatch: installed {cv2.__version__}, reference {implementation['version']}"
        )
    cv2.ocl.setUseOpenCL(False)
    if cv2.ocl.useOpenCL():
        raise RuntimeError("OpenCL remained enabled; refusing non-CPU execution.")
    cv2.setNumThreads(1)

    rgb_mae_digest = sha256(WAN_RGB_MAE_PATH)
    if rgb_mae_digest != EXPECTED_WAN_RGB_MAE_SHA256:
        raise RuntimeError(f"Registered Wan RGB-MAE result changed: {rgb_mae_digest}")
    rgb_mae_reference = json.loads(WAN_RGB_MAE_PATH.read_text())
    if rgb_mae_reference.get("anchors") != WAN_ANCHORS or rgb_mae_reference.get("frame_count") != 17:
        raise RuntimeError("Wan RGB-MAE result has a different frame count or anchor schedule.")

    manifest_digest = sha256(WAN_MANIFEST_PATH)
    manifest = json.loads(WAN_MANIFEST_PATH.read_text())
    manifest_frames = {
        int(path.rsplit("frame_", 1)[1].split(".", 1)[0]): digest
        for path, digest in manifest["files_sha256"].items()
        if path.startswith("run_001/frames/frame_") and path.endswith(".png")
    }
    if sorted(manifest_frames) != list(range(17)):
        raise RuntimeError(f"Wan retry5 manifest contains an unexpected frame set: {sorted(manifest_frames)}")
    rgb_hashes = {row["frame"]: row["sha256"] for row in rgb_mae_reference["frame_sha256"]}
    if rgb_hashes != manifest_frames:
        raise RuntimeError("Wan RGB-MAE reference hashes differ from the current run manifest.")

    run_dir = WAN_MANIFEST_PATH.parent
    frame_dir = run_dir / "run_001/frames"
    frame_paths = [frame_dir / f"frame_{index:03d}.png" for index in range(17)]
    frame_hash_rows = []
    for index, path in enumerate(frame_paths):
        if not path.is_file():
            raise RuntimeError(f"Missing hash-pinned Wan RGB frame: {path}")
        actual = sha256(path)
        if actual != manifest_frames[index]:
            raise RuntimeError(f"Frame hash mismatch for {path.name}: {actual} != {manifest_frames[index]}")
        frame_hash_rows.append({"frame": index, "path": str(path.relative_to(ROOT)), "sha256": actual})

    # Only decode the verified PNGs; no checkpoint or tensor file is opened.
    gray_frames = []
    frame_size = None
    for path in frame_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if frame_size is None:
                frame_size = image.size
            elif image.size != frame_size:
                raise RuntimeError(f"Inconsistent Wan frame size at {path.name}: {image.size} != {frame_size}")
            rgb = np.asarray(image, dtype=np.uint8)
        gray_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    width, height = frame_size
    diagonal_px = float(np.hypot(width, height))

    rgb_mae_by_pair = {
        (row["from_frame"], row["to_frame"]): row["normalized_rgb_mae"]
        for row in rgb_mae_reference["adjacent_frame_transitions"]
    }
    flow_rows = []
    adjacent_means = []
    adjacent_rgb_mae = []
    for index in range(16):
        stats = flow_summary(gray_frames[index], gray_frames[index + 1], diagonal_px, parameters)
        adjacent_means.append(stats["mean_magnitude_px"])
        adjacent_rgb_mae.append(rgb_mae_by_pair[(index, index + 1)])
        flow_rows.append(
            {
                "from_frame": index,
                "to_frame": index + 1,
                **stats,
                "rgb_mae_normalized": rgb_mae_by_pair[(index, index + 1)],
            }
        )
    flow_ranks = average_ranks(adjacent_means)
    mae_ranks = average_ranks(adjacent_rgb_mae)
    for row, flow_rank, mae_rank in zip(flow_rows, flow_ranks, mae_ranks):
        row["flow_mean_rank_ascending"] = float(flow_rank)
        row["rgb_mae_rank_ascending"] = float(mae_rank)
    spearman = float(np.corrcoef(flow_ranks, mae_ranks)[0, 1])

    intervals = []
    rgb_intervals = rgb_mae_reference["anchor_interval_comparison"]
    for interval_index, (start, end) in enumerate(zip(WAN_ANCHORS, WAN_ANCHORS[1:])):
        endpoint = flow_summary(gray_frames[start], gray_frames[end], diagonal_px, parameters)
        path_length = float(sum(adjacent_means[start:end]))
        endpoint_mean = endpoint["mean_magnitude_px"]
        rgb_row = rgb_intervals[interval_index]
        if (rgb_row["start_anchor"], rgb_row["end_anchor"]) != (start, end):
            raise RuntimeError("Wan RGB-MAE interval schedule differs from the requested anchors.")
        intervals.append(
            {
                "start_anchor": start,
                "end_anchor": end,
                "interval_frames": end - start,
                "adjacent_flow_mean_magnitude_path_px": path_length,
                "adjacent_flow_path_diagonal_fraction": path_length / diagonal_px,
                "endpoint_flow": endpoint,
                "path_minus_endpoint_mean_magnitude_px": path_length - endpoint_mean,
                "signed_flow_path_excess_fraction": (
                    (path_length - endpoint_mean) / path_length if path_length > 0 else None
                ),
                "flow_path_to_endpoint_ratio": path_length / endpoint_mean if endpoint_mean > 0 else None,
                "rgb_mae_path_normalized": rgb_row["adjacent_path_length_normalized"],
                "rgb_mae_endpoint_normalized": rgb_row["anchor_endpoint_normalized_rgb_mae"],
                "signed_rgb_mae_path_excess_fraction": (
                    (rgb_row["adjacent_path_length_normalized"] - rgb_row["anchor_endpoint_normalized_rgb_mae"])
                    / rgb_row["adjacent_path_length_normalized"]
                    if rgb_row["adjacent_path_length_normalized"] > 0
                    else None
                ),
            }
        )

    flow_path_total = float(sum(adjacent_means))
    flow_endpoint_total = float(sum(row["endpoint_flow"]["mean_magnitude_px"] for row in intervals))
    cog_adjacent = cog_flow_reference["adjacent_frame_flow"]
    cog_mean_flow = float(cog_flow_reference["summary"]["mean_adjacent_mean_flow_magnitude_px"])
    cog_mean_diagonal_fraction = float(
        np.mean([row["mean_magnitude_frame_diagonal_fraction"] for row in cog_adjacent])
    )

    result = {
        "audit_id": "EXP-M0-WAN-RETRY5-FARNEBACK-FLOW-001",
        "status": "completed_cpu_cross_clip_classical_flow_proxy",
        "created_date": "2026-10-17",
        "scope": "Hash-verified decoded Wan retry5 RGB PNGs only. No model/latent access, inference, generation, GPU, cloud, or dependency installation.",
        "source_manifest": {
            "path": str(WAN_MANIFEST_PATH.relative_to(ROOT)),
            "sha256": manifest_digest,
        },
        "rgb_mae_reference": {
            "path": str(WAN_RGB_MAE_PATH.relative_to(ROOT)),
            "sha256": rgb_mae_digest,
        },
        "cogvideox_farneback_reference": {
            "path": str(COG_FLOW_REFERENCE_PATH.relative_to(ROOT)),
            "sha256": cog_flow_digest,
            "version": implementation["version"],
            "parameters": parameters,
        },
        "frame_count": 17,
        "frame_size_px": [width, height],
        "frame_diagonal_px": diagonal_px,
        "frame_sha256": frame_hash_rows,
        "anchors": WAN_ANCHORS,
        "implementation": {
            "library": "OpenCV Python",
            "version": cv2.__version__,
            "function": "cv2.calcOpticalFlowFarneback",
            "algorithm": implementation["algorithm"],
            "input": "8-bit grayscale converted from decoded RGB with cv2.COLOR_RGB2GRAY",
            "parameters": parameters,
            "backend": "CPU NumPy Mat; OpenCL explicitly disabled; cv2.setNumThreads(1); no learned weights",
        },
        "normalization_and_metrics": cog_flow_reference["normalization_and_metrics"],
        "adjacent_frame_flow": flow_rows,
        "anchor_interval_comparison": intervals,
        "within_clip_ranking": {
            "adjacent_pair_count": 16,
            "spearman_flow_mean_vs_rgb_mae": spearman,
            "highest_flow_mean_pair": {
                key: max(flow_rows, key=lambda row: row["mean_magnitude_px"])[key]
                for key in ("from_frame", "to_frame", "mean_magnitude_px", "rgb_mae_normalized")
            },
            "highest_rgb_mae_pair": {
                key: max(flow_rows, key=lambda row: row["rgb_mae_normalized"])[key]
                for key in ("from_frame", "to_frame", "mean_magnitude_px", "rgb_mae_normalized")
            },
        },
        "summary": {
            "mean_adjacent_mean_flow_magnitude_px": float(np.mean(adjacent_means)),
            "mean_adjacent_mean_flow_frame_diagonal_fraction": float(
                np.mean([row["mean_magnitude_frame_diagonal_fraction"] for row in flow_rows])
            ),
            "median_adjacent_mean_flow_magnitude_px": float(np.median(adjacent_means)),
            "max_adjacent_mean_flow_magnitude_px": float(max(adjacent_means)),
            "adjacent_flow_path_px": flow_path_total,
            "anchor_endpoint_flow_magnitude_sum_px": flow_endpoint_total,
            "signed_flow_path_excess_fraction": (
                (flow_path_total - flow_endpoint_total) / flow_path_total if flow_path_total > 0 else None
            ),
        },
        "cross_clip_reference_comparison": {
            "reference_clip": "CogVideoX-2B chunk13",
            "reference_frame_count": 16,
            "reference_resolution_px": cog_flow_reference["frame_size_px"],
            "flow_pair_ranking_vs_rgb_mae_spearman": cog_flow_reference["summary"][
                "descriptive_spearman_flow_vs_rgb_mae_across_15_pairs"
            ],
            "flow_peak_pair": cog_flow_reference["summary"]["highest_adjacent_flow_mean_pair"],
            "rgb_mae_peak_pair": cog_flow_reference["summary"]["highest_adjacent_rgb_mae_pair"],
            "mean_adjacent_flow_magnitude_px": cog_mean_flow,
            "mean_adjacent_flow_frame_diagonal_fraction": cog_mean_diagonal_fraction,
            "signed_flow_path_excess_fraction": cog_flow_reference["summary"]["signed_path_excess_fraction"],
            "warning": "Reference results are from a different model, prompt, seed, frame count, and resolution; metrics are descriptive and not a controlled model comparison.",
        },
        "limitations": [
            "One saved Wan sequence. Whole-frame flow magnitude is an algorithmic proxy, not an object track or ground truth.",
            "Cross-clip comparison is uncontrolled: Wan retry5 and CogVideoX chunk13 differ in model, prompt, seed, frame count, resolution, and generation configuration.",
            "Pairwise rank correlations are descriptive within each clip only. Endpoint and adjacent flow fields are independently estimated; their means are not pointwise-composed trajectories.",
            "No claim about object/path influence, event recall, sampling sufficiency, or model superiority follows. Full-sequence/all-frame policy unchanged.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)}; verified 17 Wan RGB hashes; computed "
        f"16 adjacent and 4 endpoint flow fields with OpenCV {cv2.__version__}."
    )


if __name__ == "__main__":
    main()
