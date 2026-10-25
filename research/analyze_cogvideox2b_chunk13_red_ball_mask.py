#!/usr/bin/env python3
"""Reuse the fixed Wan red-mask gate on saved CogVideoX-2B chunk13 RGBs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
INPUT_MAP = ROOT / "research/cogvideox2b_rgb_proxy_inputs_v1.json"
FLOW_JSON = ROOT / "research/cogvideox2b_chunk13_farneback_flow_v1.json"
RGB_MAE_JSON = ROOT / "research/cogvideox2b_chunk13_anchor_motion_proxy_v1.json"
WAN_MASK_PROTOCOL = ROOT / "research/2026-10-17_m0_wan_retry5_red_ball_mask_protocol.md"
OUTPUT = ROOT / "research/cogvideox2b_chunk13_red_ball_mask_v1.json"

EXPECTED_INPUT_MAP_SHA256 = "280f8dfd299cdb135974c824998b2ff36b884abfefdfb1a09391e4539ff36de1"
EXPECTED_MANIFEST_SHA256 = "2727ccdc46e659effec4cbb97639a3b50a05bae8ef950e8a8a349b5ed3b0706e"
EXPECTED_FLOW_SHA256 = "216656c47c99bea09f99c419b983225330db719c1df9f546894b2b6e1ca91f3d"
EXPECTED_RGB_MAE_SHA256 = "7f603bf5db6500ca862d1f003f9a870dfc29a1dc35e4efb32feee91739a6fbe3"
EXPECTED_WAN_MASK_PROTOCOL_SHA256 = "b2651c0aa1b941c84fedaefe6b0c28d8e8e1ceedd2b0f11af12cccdf5c920a7c"
RED_GT, GREEN_LT, BLUE_LT = 150, 125, 125
MIN_COMPONENT_PIXELS = 300
ANCHORS = [0, 4, 8, 12, 15]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_reference_frame_hashes(reference: dict, name: str, pinned: dict[str, str]) -> None:
    if "frame_sha256" not in reference:
        raise RuntimeError(f"{name} reference has no per-frame hashes.")
    ref_hashes = {Path(row["path"]).name: row["sha256"] for row in reference["frame_sha256"]}
    if ref_hashes != pinned:
        raise RuntimeError(f"{name} frame hashes differ from the pinned input map.")


def main() -> None:
    # Verify all pins and source file hashes before decoding any RGB pixels.
    input_map_hash = sha256(INPUT_MAP)
    if input_map_hash != EXPECTED_INPUT_MAP_SHA256:
        raise RuntimeError(f"Pinned input map changed: {input_map_hash}")
    if sha256(WAN_MASK_PROTOCOL) != EXPECTED_WAN_MASK_PROTOCOL_SHA256:
        raise RuntimeError("The already fixed Wan red-mask protocol changed; refusing to proceed.")
    flow_hash = sha256(FLOW_JSON)
    mae_hash = sha256(RGB_MAE_JSON)
    if flow_hash != EXPECTED_FLOW_SHA256 or mae_hash != EXPECTED_RGB_MAE_SHA256:
        raise RuntimeError("A pinned Farneback or RGB-MAE reference result changed.")

    input_map = load_json(INPUT_MAP)
    manifest_path = Path(input_map["manifest_path"])
    manifest_hash = sha256(manifest_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256 or input_map["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Source manifest changed: {manifest_hash}")
    manifest = load_json(manifest_path)
    frame_count = int(input_map["frame_count"])
    if frame_count != 16 or manifest.get("frames") != frame_count:
        raise RuntimeError("Expected a 16-frame source sequence in both input map and manifest.")
    if input_map.get("frame_size_px") != [manifest.get("width"), manifest.get("height")]:
        raise RuntimeError("Input-map and source-manifest frame dimensions differ.")
    pinned_hashes: dict[str, str] = input_map["frames"]
    expected_names = [f"frame_{index:02d}.png" for index in range(frame_count)]
    if sorted(pinned_hashes) != expected_names:
        raise RuntimeError("Pinned input map has unexpected frame names or count.")

    flow_ref = load_json(FLOW_JSON)
    mae_ref = load_json(RGB_MAE_JSON)
    verify_reference_frame_hashes(flow_ref, "Farneback", pinned_hashes)
    verify_reference_frame_hashes(mae_ref, "RGB-MAE", pinned_hashes)
    if flow_ref.get("anchors") != ANCHORS or mae_ref.get("anchors") != ANCHORS:
        raise RuntimeError("Pinned flow/RGB-MAE anchor schedule differs from the source clip.")

    frames_dir = manifest_path.parent / "frames"
    frame_paths = [frames_dir / name for name in expected_names]
    source_hashes = []
    for index, path in enumerate(frame_paths):
        actual = sha256(path) if path.is_file() else None
        expected = pinned_hashes[path.name]
        if actual != expected:
            raise RuntimeError(f"Frame {index} hash mismatch: expected {expected}; got {actual}")
        source_hashes.append(
            {"frame": index, "path": str(path), "sha256": actual}
        )

    cv2.ocl.setUseOpenCL(False)
    if cv2.ocl.useOpenCL():
        raise RuntimeError("OpenCL remained enabled; refusing non-CPU execution.")
    cv2.setNumThreads(1)

    isolation_rows = []
    selected_components = []
    for index, path in enumerate(frame_paths):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        mask = (
            (rgb[:, :, 0] > RED_GT)
            & (rgb[:, :, 1] < GREEN_LT)
            & (rgb[:, :, 2] < BLUE_LT)
        ).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        components = [
            {
                "label": label,
                "area_px": int(stats[label, cv2.CC_STAT_AREA]),
                "bbox_xyxy_half_open_px": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "centroid_xy_px": [float(centroids[label, 0]), float(centroids[label, 1])],
            }
            for label in range(1, count)
        ]
        components.sort(key=lambda row: (-row["area_px"], row["label"]))
        mask_pixels = int(np.count_nonzero(mask))
        largest_area = components[0]["area_px"] if components else 0
        outside_area = mask_pixels - largest_area
        isolated = largest_area >= MIN_COMPONENT_PIXELS and outside_area < MIN_COMPONENT_PIXELS
        isolation_rows.append(
            {
                "frame": index,
                "threshold_mask_pixels": mask_pixels,
                "connected_component_count": len(components),
                "largest_component_area_px": largest_area,
                "largest_component_bbox_xyxy_half_open_px": (
                    components[0]["bbox_xyxy_half_open_px"] if components else None
                ),
                "second_largest_component_area_px": components[1]["area_px"] if len(components) > 1 else 0,
                "mask_pixels_outside_largest_px": outside_area,
                "isolation_check_passed": isolated,
            }
        )
        selected_components.append(components[0] if components else None)

    failures = [row["frame"] for row in isolation_rows if not row["isolation_check_passed"]]
    output = {
        "audit_id": "EXP-COGVIDEOX2B-CHUNK13-RED-BALL-MASK-001",
        "created_date": "2026-10-17",
        "status": "stopped_mask_ambiguous" if failures else "completed_cpu_red_mask_component_proxy",
        "scope": "Second saved-sequence diagnostic on existing CogVideoX-2B chunk13 RGB PNGs; not a controlled Wan-vs-CogVideoX model comparison.",
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
            "note": "Manifest pins sequence metadata/frame count; individual frame SHA-256 values are pinned in the input map.",
        },
        "pinned_input_map": {"path": str(INPUT_MAP.relative_to(ROOT)), "sha256": input_map_hash},
        "frame_count": frame_count,
        "frame_size_px": input_map["frame_size_px"],
        "frame_sha256": source_hashes,
        "mask_protocol": {
            "path": str(WAN_MASK_PROTOCOL.relative_to(ROOT)),
            "sha256": EXPECTED_WAN_MASK_PROTOCOL_SHA256,
            "rgb_threshold": {"r_gt": RED_GT, "g_lt": GREEN_LT, "b_lt": BLUE_LT},
            "connectivity": 8,
            "minimum_component_pixels": MIN_COMPONENT_PIXELS,
            "isolation_rule": "largest component >=300 px and fewer than 300 threshold-mask pixels in all other components combined in every frame",
            "threshold_tuning_or_temporal_association": False,
        },
        "rgb_mae_reference": {"path": str(RGB_MAE_JSON.relative_to(ROOT)), "sha256": mae_hash},
        "farneback_reference": {"path": str(FLOW_JSON.relative_to(ROOT)), "sha256": flow_hash},
        "isolation_check": {
            "passed": not failures,
            "failed_frames": failures,
            "per_frame_component_evidence": isolation_rows,
        },
        "limitations": [
            "A fixed color-connected component is an image-space proxy, not semantic object identity or ground-truth tracking.",
            "This is a second saved-sequence diagnostic, not a controlled Wan-vs-CogVideoX model comparison.",
            "One clip per saved sequence; no general sampling evidence or policy change follows.",
        ],
    }

    if not failures:
        frame_rows = []
        for index, component in enumerate(selected_components):
            x0, y0, x1, y1 = component["bbox_xyxy_half_open_px"]
            frame_rows.append(
                {
                    "frame": index,
                    "component_area_px": component["area_px"],
                    "bbox_xyxy_half_open_px": component["bbox_xyxy_half_open_px"],
                    "bbox_width_px": x1 - x0,
                    "bbox_height_px": y1 - y0,
                    "centroid_xy_px": component["centroid_xy_px"],
                }
            )
        flow_rows = {row["from_frame"]: row for row in flow_ref["adjacent_frame_flow"]}
        mae_rows = {row["from_frame"]: row for row in mae_ref["adjacent_frame_transitions"]}
        adjacent_rows = []
        displacements = []
        for left, right in zip(frame_rows, frame_rows[1:]):
            dx = right["centroid_xy_px"][0] - left["centroid_xy_px"][0]
            dy = right["centroid_xy_px"][1] - left["centroid_xy_px"][1]
            distance = math.hypot(dx, dy)
            displacements.append(distance)
            flow = flow_rows[left["frame"]]
            mae = mae_rows[left["frame"]]
            adjacent_rows.append(
                {
                    "from_frame": left["frame"],
                    "to_frame": right["frame"],
                    "centroid_dx_px": dx,
                    "centroid_dy_px": dy,
                    "centroid_displacement_px": distance,
                    "farneback_full_frame_mean_magnitude_px": flow["mean_magnitude_px"],
                    "rgb_mae_normalized": mae["normalized_rgb_mae"],
                }
            )
        max_displacement = max(adjacent_rows, key=lambda row: row["centroid_displacement_px"])
        farneback_peak = flow_ref["summary"]["highest_adjacent_flow_mean_pair"]
        rgb_peak = flow_ref["summary"]["highest_adjacent_rgb_mae_pair"]
        output["measurements"] = {
            "per_frame": frame_rows,
            "adjacent_pairs": adjacent_rows,
            "summary": {
                "mean_adjacent_centroid_displacement_px": float(np.mean(displacements)),
                "sum_adjacent_centroid_displacement_px": float(np.sum(displacements)),
                "direct_frame_0_to_15_centroid_displacement_px": math.hypot(
                    frame_rows[-1]["centroid_xy_px"][0] - frame_rows[0]["centroid_xy_px"][0],
                    frame_rows[-1]["centroid_xy_px"][1] - frame_rows[0]["centroid_xy_px"][1],
                ),
                "largest_mask_centroid_displacement_pair": {
                    "from_frame": max_displacement["from_frame"],
                    "to_frame": max_displacement["to_frame"],
                    "displacement_px": max_displacement["centroid_displacement_px"],
                },
                "existing_farneback_peak_pair": farneback_peak,
                "existing_rgb_mae_peak_pair": rgb_peak,
                "existing_flow_vs_rgb_mae_spearman": flow_ref["summary"][
                    "descriptive_spearman_flow_vs_rgb_mae_across_15_pairs"
                ],
            },
        }
    else:
        output["stop_reason"] = (
            "The fixed isolation gate failed. No per-frame centroid track or adjacent displacement values were calculated. "
            "Use independently specified or human-reviewed object masks; do not tune this color threshold."
        )

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}; status={output['status']}; failed_frames={failures}")


if __name__ == "__main__":
    main()
