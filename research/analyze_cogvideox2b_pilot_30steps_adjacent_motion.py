#!/usr/bin/env python3
"""Measure adjacent RGB-MAE and Farneback for saved CogVideoX-2B pilot_30steps.

Reuses the exact image metrics/helpers from analyze_cogvideox2b_pilot_adjacent_motion.py.
The existing failed red-mask gate is recorded by reference and is not recomputed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import analyze_cogvideox2b_pilot_adjacent_motion as method_reference


ROOT = Path(__file__).resolve().parents[1]
INPUT_MAP = ROOT / "research/cogvideox2b_stepsensitivity_inputs_v1.json"
SPARSE_PREREG = ROOT / "research/cogvideox_2b_sparse_frame_preregistration_v1.json"
MASK_AUDIT = ROOT / "research/cogvideox2b_stepsensitivity_rgb_v1.json"
SPARSE_READOUT = ROOT / "research/cogvideox_2b_five_anchor_readout_v1.json"
FLOW_METHOD_REFERENCE = ROOT / "research/cogvideox2b_chunk13_farneback_flow_v1.json"
RGB_MAE_METHOD_REFERENCE = ROOT / "research/cogvideox2b_chunk13_anchor_motion_proxy_v1.json"
METHOD_RUNNER = ROOT / "research/analyze_cogvideox2b_pilot_adjacent_motion.py"
OUTPUT = ROOT / "research/cogvideox2b_pilot_30steps_adjacent_motion_v1.json"

EXPECTED_INPUT_MAP_SHA256 = "b412f4f31d051d31f40060d17b34e47443ea5c8c23e857e996409cc7453dbcdd"
EXPECTED_SPARSE_PREREG_SHA256 = "a684caf7b9cb0a3170cc8803c0119a720e3e45f7df0e015880f298239022f4fd"
EXPECTED_MASK_AUDIT_SHA256 = "4855a996bf9e3c908d4c7fa89a722e45449bf6f38ece8766bc70b11455607acc"
EXPECTED_SPARSE_READOUT_SHA256 = "4b73e15a6e0b1eb88f8543b121336f5796206217ebd80f93ca6ee155c994aee4"
EXPECTED_FLOW_METHOD_REFERENCE_SHA256 = "216656c47c99bea09f99c419b983225330db719c1df9f546894b2b6e1ca91f3d"
EXPECTED_RGB_MAE_METHOD_REFERENCE_SHA256 = "7f603bf5db6500ca862d1f003f9a870dfc29a1dc35e4efb32feee91739a6fbe3"
EXPECTED_METHOD_RUNNER_SHA256 = "e04e6b9272cc8b8b70a2667b6b1d45b62add21c355a26df4620afdadb7fa8606"
ANCHORS = [0, 2, 4, 6, 8]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    # Check all method/input registrations before opening any source RGB pixels.
    expected_artifacts = {
        INPUT_MAP: EXPECTED_INPUT_MAP_SHA256,
        SPARSE_PREREG: EXPECTED_SPARSE_PREREG_SHA256,
        MASK_AUDIT: EXPECTED_MASK_AUDIT_SHA256,
        SPARSE_READOUT: EXPECTED_SPARSE_READOUT_SHA256,
        FLOW_METHOD_REFERENCE: EXPECTED_FLOW_METHOD_REFERENCE_SHA256,
        RGB_MAE_METHOD_REFERENCE: EXPECTED_RGB_MAE_METHOD_REFERENCE_SHA256,
        METHOD_RUNNER: EXPECTED_METHOD_RUNNER_SHA256,
    }
    for path, expected in expected_artifacts.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Registered artifact changed: {path}: {actual} != {expected}")

    input_map = load_json(INPUT_MAP)
    sparse_prereg = load_json(SPARSE_PREREG)
    mask_audit = load_json(MASK_AUDIT)
    sparse_readout = load_json(SPARSE_READOUT)
    flow_method_ref = load_json(FLOW_METHOD_REFERENCE)

    run = next(row for row in input_map["runs"] if row["run_id"] == "cogvideox_2b_pilot_30steps")
    prereg_run = next(row for row in sparse_prereg["runs"] if row["run_name"] == "cogvideox_2b_pilot_30steps")
    manifest_path = Path(run["manifest_path"])
    manifest_hash = sha256(manifest_path)
    if manifest_hash != run["manifest_sha256"] or manifest_hash != prereg_run["manifest_sha256"]:
        raise RuntimeError(f"Pilot_30steps manifest hash mismatch: {manifest_hash}")
    manifest = load_json(manifest_path)
    if run["frame_count"] != 9 or manifest.get("frames") != 9:
        raise RuntimeError("Expected the registered nine-frame 30-step pilot.")
    expected_size = (int(run["manifest_fields"]["width"]), int(run["manifest_fields"]["height"]))
    if expected_size != (480, 320) or run["manifest_fields"]["denoising_steps"] != 30:
        raise RuntimeError(f"Unexpected registered run conditions: {run['manifest_fields']}")
    for field, expected in run["manifest_fields"].items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"Manifest field mismatch for {field}: {manifest.get(field)!r} != {expected!r}")
    input_hashes = {row["name"]: row["sha256"] for row in run["frames"]}
    if prereg_run["rgb_frame_sha256"] != input_hashes:
        raise RuntimeError("Sparse-readout preregistration and input map disagree on frame hashes.")
    if any(tuple(row["size_px"]) != expected_size for row in run["frames"]):
        raise RuntimeError("Registered frame dimensions differ from the manifest.")
    if sparse_prereg["method"]["anchors"] != ANCHORS:
        raise RuntimeError("The registered anchor schedule changed.")

    # Preserve the prior fixed-red-mask failure; never rerun the gate here.
    mask_run = next(row for row in mask_audit["runs"] if row["run_id"] == "cogvideox_2b_pilot_30steps")
    mask_measurement = mask_run["measurement"]
    if mask_measurement["fully_covered_case_count"] != 0 or mask_measurement["maximum_valid_frames_in_any_threshold_case"] != 0:
        raise RuntimeError("The registered mask gate result differs from the known 0/9 failure.")
    if mask_measurement["maximum_raw_mask_pixels_any_threshold_frame"] != 203:
        raise RuntimeError("The registered fixed-threshold scoreability evidence changed.")
    if sparse_readout.get("status") != "stopped_proxy_coverage_failure_no_interpolation_metrics":
        raise RuntimeError("The registered sparse-readout gate status changed.")

    # Reuse the exact previously registered method settings.
    cv2.ocl.setUseOpenCL(False)
    if cv2.ocl.useOpenCL():
        raise RuntimeError("OpenCL remained enabled; refusing non-CPU execution.")
    cv2.setNumThreads(1)
    flow_impl = flow_method_ref["implementation"]
    if flow_impl["version"] != cv2.__version__ or flow_impl["parameters"] != method_reference.FARNEBACK_PARAMETERS:
        raise RuntimeError("OpenCV or Farneback settings differ from the registered method.")

    frames_dir = manifest_path.parent / "frames"
    frame_paths = [frames_dir / f"frame_{index:02d}.png" for index in range(9)]
    frame_hashes = []
    for index, path in enumerate(frame_paths):
        if not path.is_file():
            raise RuntimeError(f"Missing registered frame: {path}")
        expected_hash = input_hashes[path.name]
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Frame hash mismatch for {path.name}: {actual_hash} != {expected_hash}")
        frame_hashes.append({"frame": index, "path": str(path), "sha256": actual_hash})
    print("Verified manifest, input-map, and all 9 pilot_30steps frame hashes.", flush=True)

    # Decode only after all hashes match.
    rgb_frames = []
    gray_frames = []
    for path in frame_paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != (expected_size[1], expected_size[0]):
            raise RuntimeError(f"Unexpected decoded size for {path.name}: {rgb.shape[1::-1]}")
        rgb_frames.append(rgb)
        gray_frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

    adjacent_rows = []
    adjacent_mae = []
    adjacent_flow = []
    for index in range(8):
        mae = method_reference.normalized_rgb_mae(rgb_frames[index], rgb_frames[index + 1])
        flow = method_reference.flow_summary(gray_frames[index], gray_frames[index + 1])
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

    interval_rows = []
    for start, end in zip(ANCHORS, ANCHORS[1:]):
        mae_path = float(sum(adjacent_mae[start:end]))
        mae_endpoint = method_reference.normalized_rgb_mae(rgb_frames[start], rgb_frames[end])
        flow_path = float(sum(adjacent_flow[start:end]))
        flow_endpoint = method_reference.flow_summary(gray_frames[start], gray_frames[end])
        flow_endpoint_mean = flow_endpoint["mean_magnitude_px"]
        interval_rows.append(
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

    dense_mae = float(sum(adjacent_mae))
    mae_endpoint_total = float(sum(row["rgb_mae_direct_anchor_endpoint_normalized"] for row in interval_rows))
    dense_flow = float(sum(adjacent_flow))
    flow_endpoint_total = float(sum(row["farneback_direct_anchor_endpoint"]["mean_magnitude_px"] for row in interval_rows))
    max_mae_index = int(np.argmax(adjacent_mae))
    max_flow_index = int(np.argmax(adjacent_flow))
    result = {
        "audit_id": "EXP-COGVIDEOX2B-PILOT-30STEPS-ADJACENT-MOTION-001",
        "status": "completed_cpu_saved_rgb_adjacent_motion_proxy",
        "created_date": "2026-10-17",
        "scope": "Image-only adjacent RGB-MAE and Farneback diagnostic on saved CogVideoX-2B pilot_30steps; clip-local only, no cross-step inference comparison.",
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
            "result": "0/9 scoreable; maximum mask size 203 pixels in one frame, below the 300-pixel floor",
            "threshold_grid_full_tracks": 0,
            "recomputed_here": False,
        },
        "prior_five_anchor_readout": {
            "path": str(SPARSE_READOUT.relative_to(ROOT)),
            "sha256": EXPECTED_SPARSE_READOUT_SHA256,
            "status": sparse_readout["status"],
            "note": "Previous mask coverage failure stopped its mask interpolation metrics; no adjacent RGB-MAE or Farneback values were registered there.",
        },
        "method_reuse": {
            "runner": str(METHOD_RUNNER.relative_to(ROOT)),
            "runner_sha256": EXPECTED_METHOD_RUNNER_SHA256,
            "rgb_mae_definition": "mean(abs(uint8_RGB(frame_t)-uint8_RGB(frame_t+1)))/255 across all pixels/channels",
            "farneback_parameters": method_reference.FARNEBACK_PARAMETERS,
            "farneback_method_reference": str(FLOW_METHOD_REFERENCE.relative_to(ROOT)),
            "farneback_method_reference_sha256": EXPECTED_FLOW_METHOD_REFERENCE_SHA256,
            "rgb_mae_method_reference": str(RGB_MAE_METHOD_REFERENCE.relative_to(ROOT)),
            "rgb_mae_method_reference_sha256": EXPECTED_RGB_MAE_METHOD_REFERENCE_SHA256,
            "opencv_version": cv2.__version__,
            "backend": "CPU; OpenCL disabled; one OpenCV thread; grayscale converted from RGB",
        },
        "normalization_and_path_definitions": {
            "rgb_mae": "normalized by 255; whole-frame mean absolute RGB difference",
            "flow": "source-image pixels; path sums independently estimated adjacent-pair mean magnitudes; not tracked pixel motion",
            "endpoint": "direct estimate between each pair of registered anchor frames",
        },
        "adjacent_transitions": adjacent_rows,
        "anchor_interval_comparison": interval_rows,
        "summary": {
            "adjacent_transition_count": 8,
            "anchor_endpoint_interval_count": 4,
            "rgb_mae_dense_adjacent_path_total_normalized": dense_mae,
            "rgb_mae_sum_of_anchor_endpoint_contrasts_normalized": mae_endpoint_total,
            "rgb_mae_dense_path_fraction_above_endpoints": (dense_mae - mae_endpoint_total) / dense_mae if dense_mae > 0 else None,
            "farneback_dense_adjacent_path_total_px": dense_flow,
            "farneback_sum_of_anchor_endpoint_magnitudes_px": flow_endpoint_total,
            "farneback_dense_path_fraction_above_endpoints": (dense_flow - flow_endpoint_total) / dense_flow if dense_flow > 0 else None,
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
            "Clip-local descriptive image-space metrics only; do not infer a denoising-step effect or cross-model comparison.",
            "Whole-frame RGB-MAE and Farneback are not human labels or ground-truth object motion.",
            "Path/endpoint differences do not establish event recall or sampling sufficiency.",
            "The existing failed red-mask gate was referenced without recomputation or retuning.",
        ],
    }
    result["runner_sha256"] = sha256(Path(__file__))
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}; verified 9 frames; computed 8 adjacent pairs and 4 anchor endpoints.", flush=True)


if __name__ == "__main__":
    main()
