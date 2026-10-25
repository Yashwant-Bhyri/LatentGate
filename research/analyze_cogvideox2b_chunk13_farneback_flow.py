#!/usr/bin/env python3
"""CPU Farneback optical-flow diagnostic for saved CogVideoX-2B chunk13 RGBs.

Uses only hash-pinned PNG frames and the existing RGB-MAE result as a
comparison reference. No learned model, latent array, GPU, or cloud access.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import cv2
import numpy as np
from PIL import Image


ROOT = pathlib.Path(__file__).resolve().parents[1]
INPUT_MAP_PATH = ROOT / "research/cogvideox2b_rgb_proxy_inputs_v1.json"
RGB_REFERENCE_PATH = ROOT / "research/cogvideox2b_chunk13_anchor_motion_proxy_v1.json"
OUTPUT_PATH = ROOT / "research/cogvideox2b_chunk13_farneback_flow_v1.json"
ANCHORS = [0, 4, 8, 12, 15]
FARNEBACK_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flow_summary(prev_gray: np.ndarray, next_gray: np.ndarray, diagonal: float) -> dict[str, float]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        FARNEBACK_PARAMETERS["pyr_scale"],
        FARNEBACK_PARAMETERS["levels"],
        FARNEBACK_PARAMETERS["winsize"],
        FARNEBACK_PARAMETERS["iterations"],
        FARNEBACK_PARAMETERS["poly_n"],
        FARNEBACK_PARAMETERS["poly_sigma"],
        FARNEBACK_PARAMETERS["flags"],
    )
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    mean_px = float(np.mean(magnitude, dtype=np.float64))
    return {
        "mean_magnitude_px": mean_px,
        "mean_magnitude_frame_diagonal_fraction": mean_px / diagonal,
        "median_magnitude_px": float(np.median(magnitude)),
        "p95_magnitude_px": float(np.percentile(magnitude, 95)),
        "p99_magnitude_px": float(np.percentile(magnitude, 99)),
        "max_magnitude_px": float(np.max(magnitude)),
    }


def average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values))
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def main() -> None:
    # This classical OpenCV implementation operates on NumPy Mats and does not
    # load weights. Explicitly disable OpenCL and keep its CPU worker count fixed.
    cv2.ocl.setUseOpenCL(False)
    if cv2.ocl.useOpenCL():
        raise RuntimeError("OpenCL remained enabled; refusing non-CPU execution.")
    cv2.setNumThreads(1)

    input_map = json.loads(INPUT_MAP_PATH.read_text())
    manifest_path = pathlib.Path(input_map["manifest_path"])
    if not manifest_path.is_file():
        raise RuntimeError(f"Pinned manifest is missing: {manifest_path}")
    manifest_hash = sha256(manifest_path)
    if manifest_hash != input_map["manifest_sha256"]:
        raise RuntimeError(f"Manifest hash mismatch: {manifest_hash} != {input_map['manifest_sha256']}")
    manifest = json.loads(manifest_path.read_text())
    frame_count = int(input_map["frame_count"])
    expected_size = tuple(int(x) for x in input_map["frame_size_px"])
    if frame_count != 16 or manifest.get("frames") != 16:
        raise RuntimeError(f"Unexpected frame count: input map={frame_count}, manifest={manifest.get('frames')}")

    frame_dir = manifest_path.parent / "frames"
    frame_paths = [frame_dir / f"frame_{i:02d}.png" for i in range(frame_count)]
    frame_hash_rows = []
    for index, path in enumerate(frame_paths):
        expected = input_map["frames"].get(path.name)
        if expected is None or not path.is_file():
            raise RuntimeError(f"Missing pinned frame or hash: {path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frame hash mismatch for {path.name}: {actual} != {expected}")
        frame_hash_rows.append({"frame": index, "path": str(path), "sha256": actual})

    # Verify the existing RGB-MAE comparison uses this exact input set.
    rgb_reference = json.loads(RGB_REFERENCE_PATH.read_text())
    if rgb_reference["frame_count"] != frame_count:
        raise RuntimeError("RGB-MAE reference frame count differs from the pinned flow input.")
    if rgb_reference.get("anchors") != ANCHORS:
        raise RuntimeError("RGB-MAE reference uses a different anchor schedule.")
    reference_hashes = {row["frame"]: row["sha256"] for row in rgb_reference["frame_sha256"]}
    current_hashes = {row["frame"]: row["sha256"] for row in frame_hash_rows}
    if reference_hashes != current_hashes:
        raise RuntimeError("RGB-MAE reference does not describe the same frame hashes.")
    rgb_mae = {
        (row["from_frame"], row["to_frame"]): row["normalized_rgb_mae"]
        for row in rgb_reference["adjacent_frame_transitions"]
    }

    # Decode source PNGs only after verifying the complete pinned frame set.
    gray_frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != expected_size:
                raise RuntimeError(f"Unexpected dimensions for {path.name}: {image.size} != {expected_size}")
            rgb = np.asarray(image, dtype=np.uint8)
        gray_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

    width, height = expected_size
    diagonal = float(np.hypot(width, height))
    adjacent_rows = []
    adjacent_mean = []
    adjacent_rgb_mae = []
    for index in range(frame_count - 1):
        summary = flow_summary(gray_frames[index], gray_frames[index + 1], diagonal)
        adjacent_mean.append(summary["mean_magnitude_px"])
        adjacent_rgb_mae.append(rgb_mae[(index, index + 1)])
        adjacent_rows.append(
            {
                "from_frame": index,
                "to_frame": index + 1,
                **summary,
                "rgb_mae_normalized": rgb_mae[(index, index + 1)],
            }
        )

    anchor_rows = []
    for start, end in zip(ANCHORS, ANCHORS[1:]):
        endpoint = flow_summary(gray_frames[start], gray_frames[end], diagonal)
        path_length = float(sum(adjacent_mean[start:end]))
        endpoint_mean = endpoint["mean_magnitude_px"]
        anchor_rows.append(
            {
                "start_anchor": start,
                "end_anchor": end,
                "interval_frames": end - start,
                "endpoint_flow": endpoint,
                "adjacent_flow_mean_magnitude_path_px": path_length,
                "adjacent_flow_path_diagonal_fraction": path_length / diagonal,
                "path_minus_endpoint_mean_magnitude_px": path_length - endpoint_mean,
                "path_to_endpoint_mean_magnitude_ratio": path_length / endpoint_mean if endpoint_mean > 0 else None,
                "signed_path_excess_fraction": (
                    (path_length - endpoint_mean) / path_length if path_length > 0 else None
                ),
                "rgb_mae_endpoint_normalized": rgb_reference["anchor_interval_comparison"][
                    ANCHORS.index(start)
                ]["anchor_endpoint_normalized_rgb_mae"],
                "rgb_mae_adjacent_path_normalized": sum(
                    rgb_mae[(index, index + 1)] for index in range(start, end)
                ),
            }
        )

    flow_path_total = float(sum(adjacent_mean))
    endpoint_total = float(sum(row["endpoint_flow"]["mean_magnitude_px"] for row in anchor_rows))
    diagonal = float(diagonal)
    flow_ranks = average_ranks(adjacent_mean)
    rgb_ranks = average_ranks(adjacent_rgb_mae)
    spearman = float(np.corrcoef(flow_ranks, rgb_ranks)[0, 1])

    result = {
        "audit_id": "EXP-COGVIDEOX2B-CHUNK13-FARNEBACK-FLOW-001",
        "status": "completed_cpu_classical_dense_optical_flow_proxy",
        "created_date": "2026-10-17",
        "scope": "Hash-pinned CogVideoX-2B chunk13 RGB PNGs only; no model weights, latent arrays, inference, generation, GPU, or cloud runtime.",
        "source_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "pinned_input_map": {
            "path": str(INPUT_MAP_PATH.relative_to(ROOT)),
            "sha256": sha256(INPUT_MAP_PATH),
            "declared_manifest_sha256": input_map["manifest_sha256"],
        },
        "rgb_mae_comparison_reference": {
            "path": str(RGB_REFERENCE_PATH.relative_to(ROOT)),
            "sha256": sha256(RGB_REFERENCE_PATH),
            "method": "normalized whole-frame RGB MAE, existing result; not recomputed here",
        },
        "frame_count": frame_count,
        "frame_size_px": [width, height],
        "frame_diagonal_px": diagonal,
        "frame_sha256": frame_hash_rows,
        "anchors": ANCHORS,
        "implementation": {
            "library": "OpenCV Python",
            "version": cv2.__version__,
            "function": "cv2.calcOpticalFlowFarneback",
            "algorithm": "Farneback polynomial-expansion dense optical flow",
            "input": "8-bit grayscale PNG frames created from RGB with cv2.COLOR_RGB2GRAY",
            "parameters": FARNEBACK_PARAMETERS,
            "backend": "CPU NumPy Mat path; OpenCL explicitly disabled; cv2.setNumThreads(1); no weights/model",
        },
        "normalization_and_metrics": {
            "flow_vector_units": "pixels between the supplied source and target frames; adjacent pairs are per output-frame interval, anchor endpoints span 3 or 4 intervals",
            "reported_normalization": "mean magnitude divided by frame diagonal sqrt(720^2 + 480^2); raw pixel magnitudes are also retained",
            "field_reduction": "mean, median, p95, and maximum magnitude over every pixel; no ROI or validity mask",
            "path_definition": "sum of mean magnitudes from independently estimated adjacent-pair flow fields; not pointwise flow composition or a tracked trajectory",
            "endpoint_definition": "mean magnitude from a separate direct Farneback estimate between the anchor images",
            "rgb_mae_definition": "Existing comparison only: mean(abs(uint8_RGB(a)-uint8_RGB(b)))/255; included to contextualize, not recomputed",
        },
        "adjacent_frame_flow": adjacent_rows,
        "anchor_interval_comparison": anchor_rows,
        "summary": {
            "adjacent_pair_count": len(adjacent_rows),
            "anchor_endpoint_pair_count": len(anchor_rows),
            "mean_adjacent_mean_flow_magnitude_px": float(np.mean(adjacent_mean)),
            "median_adjacent_mean_flow_magnitude_px": float(np.median(adjacent_mean)),
            "max_adjacent_mean_flow_magnitude_px": float(max(adjacent_mean)),
            "dense_adjacent_flow_path_px": flow_path_total,
            "dense_adjacent_flow_path_diagonal_fraction": flow_path_total / diagonal,
            "sum_anchor_endpoint_flow_magnitudes_px": endpoint_total,
            "sum_endpoint_to_adjacent_path_ratio": endpoint_total / flow_path_total if flow_path_total > 0 else None,
            "signed_path_excess_fraction": (
                (flow_path_total - endpoint_total) / flow_path_total if flow_path_total > 0 else None
            ),
            "highest_adjacent_flow_mean_pair": {
                key: max(adjacent_rows, key=lambda row: row["mean_magnitude_px"])[key]
                for key in ("from_frame", "to_frame", "mean_magnitude_px", "rgb_mae_normalized")
            },
            "highest_adjacent_rgb_mae_pair": {
                key: max(adjacent_rows, key=lambda row: row["rgb_mae_normalized"])[key]
                for key in ("from_frame", "to_frame", "mean_magnitude_px", "rgb_mae_normalized")
            },
            "descriptive_spearman_flow_vs_rgb_mae_across_15_pairs": spearman,
        },
        "limitations": [
            "One saved 16-frame sequence; flow is an exploratory algorithmic proxy, not human labels or general evidence.",
            "Whole-frame Farneback output can be noisy or ambiguous in textureless/repetitive regions and may respond to generated texture, brightness, or rendering changes. No object mask, identity, or camera model is used.",
            "Mean flow magnitude drops direction and spatial correspondence. Summed mean adjacent magnitudes are not the path of tracked pixels; direct multi-frame endpoint flow is a separate estimate and can fail differently.",
            "Comparison with RGB MAE is descriptive only. Flow estimates apparent displacement from image structure; RGB MAE measures intensity change. Neither establishes object/path influence, event recall, or sampling sufficiency.",
            "This analysis does not alter the registered full-sequence/all-frame policy.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Wrote {OUTPUT_PATH.relative_to(ROOT)}; verified {len(frame_hash_rows)} frames; "
        f"estimated {len(adjacent_rows)} adjacent and {len(anchor_rows)} anchor endpoint flow fields "
        f"using OpenCV {cv2.__version__} CPU Farneback."
    )


if __name__ == "__main__":
    main()
