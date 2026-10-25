#!/usr/bin/env python3
"""Measure adjacent RGB MAE and Farneback flow for the saved 8-step 2B pilot.

This is an image-only diagnostic. It does not rerun the known failed red-mask
gate and does not read model weights, latent arrays, or generation code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT_MAP = ROOT / "research/cogvideox2b_stepsensitivity_inputs_v1.json"
SPARSE_PREREG = ROOT / "research/cogvideox_2b_sparse_frame_preregistration_v1.json"
MASK_AUDIT = ROOT / "research/cogvideox2b_stepsensitivity_rgb_v1.json"
SPARSE_READOUT = ROOT / "research/cogvideox_2b_five_anchor_readout_v1.json"
FLOW_METHOD_REFERENCE = ROOT / "research/cogvideox2b_chunk13_farneback_flow_v1.json"
RGB_MAE_METHOD_REFERENCE = ROOT / "research/cogvideox2b_chunk13_anchor_motion_proxy_v1.json"
OUTPUT = ROOT / "research/cogvideox2b_pilot_adjacent_motion_v1.json"

EXPECTED_INPUT_MAP_SHA256 = "b412f4f31d051d31f40060d17b34e47443ea5c8c23e857e996409cc7453dbcdd"
EXPECTED_SPARSE_PREREG_SHA256 = "a684caf7b9cb0a3170cc8803c0119a720e3e45f7df0e015880f298239022f4fd"
EXPECTED_MASK_AUDIT_SHA256 = "4855a996bf9e3c908d4c7fa89a722e45449bf6f38ece8766bc70b11455607acc"
EXPECTED_SPARSE_READOUT_SHA256 = "4b73e15a6e0b1eb88f8543b121336f5796206217ebd80f93ca6ee155c994aee4"
EXPECTED_FLOW_METHOD_REFERENCE_SHA256 = "216656c47c99bea09f99c419b983225330db719c1df9f546894b2b6e1ca91f3d"
EXPECTED_RGB_MAE_METHOD_REFERENCE_SHA256 = "7f603bf5db6500ca862d1f003f9a870dfc29a1dc35e4efb32feee91739a6fbe3"
ANCHORS = [0, 2, 4, 6, 8]
FARNEBACK_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalized_rgb_mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} != {b.shape}")
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(dtype=np.float64) / 255.0)


def flow_summary(previous_gray: np.ndarray, next_gray: np.ndarray) -> dict[str, float]:
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
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
    return {
        "mean_magnitude_px": float(np.mean(magnitude, dtype=np.float64)),
        "median_magnitude_px": float(np.median(magnitude)),
        "p95_magnitude_px": float(np.percentile(magnitude, 95)),
        "p99_magnitude_px": float(np.percentile(magnitude, 99)),
        "max_magnitude_px": float(np.max(magnitude)),
    }


def main() -> None:
    # Validate all registration files before loading any frame pixels.
    expected_artifacts = {
        INPUT_MAP: EXPECTED_INPUT_MAP_SHA256,
        SPARSE_PREREG: EXPECTED_SPARSE_PREREG_SHA256,
        MASK_AUDIT: EXPECTED_MASK_AUDIT_SHA256,
        SPARSE_READOUT: EXPECTED_SPARSE_READOUT_SHA256,
        FLOW_METHOD_REFERENCE: EXPECTED_FLOW_METHOD_REFERENCE_SHA256,
        RGB_MAE_METHOD_REFERENCE: EXPECTED_RGB_MAE_METHOD_REFERENCE_SHA256,
    }
    for path, expected in expected_artifacts.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Registered input changed: {path}: {actual} != {expected}")

    input_map = load_json(INPUT_MAP)
    sparse_prereg = load_json(SPARSE_PREREG)
    mask_audit = load_json(MASK_AUDIT)
    sparse_readout = load_json(SPARSE_READOUT)
    flow_method_ref = load_json(FLOW_METHOD_REFERENCE)
    mae_method_ref = load_json(RGB_MAE_METHOD_REFERENCE)

    run = next(row for row in input_map["runs"] if row["run_id"] == "cogvideox_2b_pilot")
    prereg_run = next(row for row in sparse_prereg["runs"] if row["run_name"] == "cogvideox_2b_pilot")
    manifest_path = Path(run["manifest_path"])
    manifest_hash = sha256(manifest_path)
    if manifest_hash != run["manifest_sha256"] or manifest_hash != prereg_run["manifest_sha256"]:
        raise RuntimeError(f"Pilot manifest hash mismatch: {manifest_hash}")
    manifest = load_json(manifest_path)
    if run["frame_count"] != 9 or manifest.get("frames") != 9:
        raise RuntimeError("Expected the registered nine-frame pilot.")
    expected_size = (
        int(run["manifest_fields"]["width"]),
        int(run["manifest_fields"]["height"]),
    )
    if expected_size != (480, 320):
        raise RuntimeError(f"Unexpected registered image size: {expected_size}")
    for field, expected in run["manifest_fields"].items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"Manifest field mismatch for {field}: {manifest.get(field)!r} != {expected!r}")
    prereg_hashes = prereg_run["rgb_frame_sha256"]
    input_hashes = {row["name"]: row["sha256"] for row in run["frames"]}
    if prereg_hashes != input_hashes:
        raise RuntimeError("Sparse-readout preregistration and step-sensitivity input map disagree on frame hashes.")
    if any(tuple(row["size_px"]) != expected_size for row in run["frames"]):
        raise RuntimeError("Registered frame dimensions are not uniform or differ from the manifest.")
    if sparse_prereg["method"]["anchors"] != ANCHORS:
        raise RuntimeError("The registered anchor schedule changed.")

    # Preserve the previous mask failure from its registered result; do not recompute masks.
    mask_run = next(row for row in mask_audit["runs"] if row["run_id"] == "cogvideox_2b_pilot")
    mask_measurement = mask_run["measurement"]
    if mask_measurement["fully_covered_case_count"] != 0 or mask_measurement["maximum_raw_mask_pixels_any_threshold_frame"] != 0:
        raise RuntimeError("The registered fixed-mask gate result differs from the known 0/9 failure.")
    if sparse_readout.get("status") != "stopped_proxy_coverage_failure_no_interpolation_metrics":
        raise RuntimeError("The registered sparse-readout gate status changed.")

    flow_impl = flow_method_ref["implementation"]
    if flow_impl.get("version") != cv2.__version__ or flow_impl.get("parameters") != FARNEBACK_PARAMETERS:
        raise RuntimeError("Installed OpenCV or Farneback settings differ from the registered method reference.")
    cv2.ocl.setUseOpenCL(False)
    if cv2.ocl.useOpenCL():
        raise RuntimeError("OpenCL remained enabled; refusing non-CPU execution.")
    cv2.setNumThreads(1)

    frames_dir = manifest_path.parent / "frames"
    frame_paths = [frames_dir / f"frame_{index:02d}.png" for index in range(9)]
    frame_hashes = []
    for index, path in enumerate(frame_paths):
        if not path.is_file():
            raise RuntimeError(f"Missing registered source frame: {path}")
        expected_hash = input_hashes[path.name]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frame hash mismatch for {path.name}: {actual_hash} != {expected_hash}")
        frame_hashes.append({"frame": index, "path": str(path), "sha256": actual_hash})
    print("Verified manifest, input-map, and all 9 frame hashes.", flush=True)

    # Decode RGB PNGs only after every map/manifest/frame hash passed.
    rgb_frames = []
    gray_frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != (expected_size[1], expected_size[0]):
            raise RuntimeError(f"Unexpected decoded size for {path.name}: {rgb.shape[1::-1]}")
        rgb_frames.append(rgb)
        gray_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    print("Decoded the 9 pinned RGB frames; computing adjacent and anchor-endpoint metrics.", flush=True)

    adjacent_rows = []
    adjacent_mae = []
    adjacent_flow = []
    for index in range(8):
        mae = normalized_rgb_mae(rgb_frames[index], rgb_frames[index + 1])
        flow = flow_summary(gray_frames[index], gray_frames[index + 1])
        adjacent_mae.append(mae)
        adjacent_flow.append(flow["mean_magnitude_px"])
        adjacent_rows.append(
            {
                "from_frame": index,
                "to_frame": index + 1,
                "normalized_rgb_mae": mae,
                "rgb_mae_8bit_levels": mae * 255.0,
                "farneback": flow,
            }
        )

    intervals = []
    for start, end in zip(ANCHORS, ANCHORS[1:]):
        mae_endpoint = normalized_rgb_mae(rgb_frames[start], rgb_frames[end])
        mae_path = float(sum(adjacent_mae[start:end]))
        flow_endpoint = flow_summary(gray_frames[start], gray_frames[end])
        flow_path = float(sum(adjacent_flow[start:end]))
        flow_endpoint_mean = flow_endpoint["mean_magnitude_px"]
        intervals.append(
            {
                "start_anchor": start,
                "end_anchor": end,
                "interval_frames": end - start,
                "rgb_mae_adjacent_path_normalized": mae_path,
                "rgb_mae_direct_anchor_endpoint_normalized": mae_endpoint,
                "rgb_mae_path_minus_endpoint": mae_path - mae_endpoint,
                "rgb_mae_path_to_endpoint_ratio": mae_path / mae_endpoint if mae_endpoint > 0 else None,
                "farneback_adjacent_path_mean_magnitude_px": flow_path,
                "farneback_direct_anchor_endpoint": flow_endpoint,
                "farneback_path_minus_endpoint_px": flow_path - flow_endpoint_mean,
                "farneback_path_to_endpoint_ratio": flow_path / flow_endpoint_mean if flow_endpoint_mean > 0 else None,
            }
        )

    mae_endpoint_total = float(sum(row["rgb_mae_direct_anchor_endpoint_normalized"] for row in intervals))
    flow_endpoint_total = float(sum(row["farneback_direct_anchor_endpoint"]["mean_magnitude_px"] for row in intervals))
    dense_mae_total = float(sum(adjacent_mae))
    dense_flow_total = float(sum(adjacent_flow))
    max_mae_index = int(np.argmax(adjacent_mae))
    max_flow_index = int(np.argmax(adjacent_flow))
    result = {
        "audit_id": "EXP-COGVIDEOX2B-PILOT-ADJACENT-MOTION-001",
        "status": "completed_cpu_saved_rgb_adjacent_motion_proxy",
        "created_date": "2026-10-17",
        "scope": "Image-only adjacent-pair RGB-MAE and Farneback diagnostic on the existing 8-step CogVideoX-2B pilot. No mask gate rerun; no model/latent access, inference, generation, GPU, or cloud.",
        "source_run": run["run_id"],
        "run_conditions": run["manifest_fields"],
        "source_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "registered_input_map": {"path": str(INPUT_MAP.relative_to(ROOT)), "sha256": EXPECTED_INPUT_MAP_SHA256},
        "sparse_readout_preregistration": {
            "path": str(SPARSE_PREREG.relative_to(ROOT)),
            "sha256": EXPECTED_SPARSE_PREREG_SHA256,
            "anchors": ANCHORS,
        },
        "frame_count": 9,
        "frame_size_px": list(expected_size),
        "anchors": ANCHORS,
        "frame_sha256": frame_hashes,
        "prior_fixed_red_mask_gate": {
            "path": str(MASK_AUDIT.relative_to(ROOT)),
            "sha256": EXPECTED_MASK_AUDIT_SHA256,
            "baseline_result": "0/9 scoreable; maximum red-mask pixels 0 across the fixed threshold result",
            "recomputed_here": False,
        },
        "prior_five_anchor_readout": {
            "path": str(SPARSE_READOUT.relative_to(ROOT)),
            "sha256": EXPECTED_SPARSE_READOUT_SHA256,
            "status": sparse_readout["status"],
            "note": "The prior red-mask coverage failure stopped its mask interpolation metrics; it did not register pixelwise RGB-MAE or Farneback values.",
        },
        "rgb_mae_method": {
            "definition": "mean(abs(uint8_RGB(frame_t)-uint8_RGB(frame_t+1)))/255 across all pixels and channels",
            "reference_method": str(RGB_MAE_METHOD_REFERENCE.relative_to(ROOT)),
            "reference_sha256": EXPECTED_RGB_MAE_METHOD_REFERENCE_SHA256,
        },
        "farneback_method": {
            "library": "OpenCV Python",
            "version": cv2.__version__,
            "function": "cv2.calcOpticalFlowFarneback",
            "parameters": FARNEBACK_PARAMETERS,
            "reference_method": str(FLOW_METHOD_REFERENCE.relative_to(ROOT)),
            "reference_sha256": EXPECTED_FLOW_METHOD_REFERENCE_SHA256,
            "backend": "CPU; OpenCL disabled; one OpenCV thread; 8-bit grayscale converted from RGB",
            "field_reduction": "mean, median, p95, p99, and max magnitude across the full frame",
        },
        "normalization_and_path_definitions": {
            "rgb_mae": "normalized by 255; bounded whole-frame mean absolute RGB intensity difference",
            "flow": "source-image pixel units; path is sum of independently estimated adjacent-frame mean magnitudes, not a tracked trajectory",
            "endpoint": "direct metric from separately supplied anchor images; four intervals from registered anchors [0,2,4,6,8]",
        },
        "adjacent_transitions": adjacent_rows,
        "anchor_interval_comparison": intervals,
        "summary": {
            "adjacent_transition_count": len(adjacent_rows),
            "anchor_endpoint_interval_count": len(intervals),
            "rgb_mae_dense_adjacent_path_total_normalized": dense_mae_total,
            "rgb_mae_sum_of_anchor_endpoint_contrasts_normalized": mae_endpoint_total,
            "rgb_mae_dense_path_minus_endpoint_total": dense_mae_total - mae_endpoint_total,
            "rgb_mae_dense_path_to_endpoint_total_ratio": dense_mae_total / mae_endpoint_total if mae_endpoint_total > 0 else None,
            "rgb_mae_signed_path_excess_fraction": (dense_mae_total - mae_endpoint_total) / dense_mae_total if dense_mae_total > 0 else None,
            "farneback_dense_adjacent_path_total_px": dense_flow_total,
            "farneback_sum_of_anchor_endpoint_magnitudes_px": flow_endpoint_total,
            "farneback_dense_path_minus_endpoint_total_px": dense_flow_total - flow_endpoint_total,
            "farneback_dense_path_to_endpoint_total_ratio": dense_flow_total / flow_endpoint_total if flow_endpoint_total > 0 else None,
            "farneback_signed_path_excess_fraction": (dense_flow_total - flow_endpoint_total) / dense_flow_total if dense_flow_total > 0 else None,
            "largest_rgb_mae_transition": {
                "from_frame": max_mae_index,
                "to_frame": max_mae_index + 1,
                "normalized_rgb_mae": adjacent_mae[max_mae_index],
            },
            "largest_farneback_transition": {
                "from_frame": max_flow_index,
                "to_frame": max_flow_index + 1,
                "mean_magnitude_px": adjacent_flow[max_flow_index],
            },
        },
        "limitations": [
            "Clip-local descriptive metrics only; do not infer a cross-model or denoising-step effect.",
            "Whole-frame RGB MAE and Farneback are image-space proxies, not human labels or object motion ground truth.",
            "Anchor path/endpoint differences do not establish sampling sufficiency or event recall.",
            "The previously registered 0/9 red-mask gate failure was preserved and not recomputed or retuned.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Wrote {OUTPUT.relative_to(ROOT)}; verified manifest and {len(frame_hashes)} RGB frames; "
        f"computed {len(adjacent_rows)} adjacent transitions and {len(intervals)} direct anchor endpoints."
    )


if __name__ == "__main__":
    main()
