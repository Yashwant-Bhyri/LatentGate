#!/usr/bin/env python3
"""Apply the preregistered red-mask isolation check to Wan retry5 RGB frames.

No model/latent arrays or inference are used. The mask and ambiguity rules are
fixed in 2026-10-17_m0_wan_retry5_red_ball_mask_protocol.md before execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/manifest.json"
FRAMES = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/run_001/frames"
PROTOCOL = ROOT / "research/2026-10-17_m0_wan_retry5_red_ball_mask_protocol.md"
FLOW_JSON = ROOT / "research/m0_wan_retry5_farneback_flow_v1.json"
RGB_MAE_JSON = ROOT / "research/m0_retry5_anchor_motion_proxy_v1.json"
OUTPUT = ROOT / "research/m0_wan_retry5_red_ball_mask_v1.json"

EXPECTED_MANIFEST_SHA256 = "a190ba398f7db0bca7ae0e7b4471050bfd3af581dd38cdfb67c59ee6dcffbfd8"
EXPECTED_FLOW_SHA256 = "cf57ee689c05769c76c0642fd796e9d0b62787887d4ec75e30736ddc66922ac1"
EXPECTED_RGB_MAE_SHA256 = "09ef38b0fe988d607004acbd1b90683f69d7a8ad61148102b5d4746a01b02ae6"
EXPECTED_PROTOCOL_SHA256 = "b2651c0aa1b941c84fedaefe6b0c28d8e8e1ceedd2b0f11af12cccdf5c920a7c"
RED_GT, GREEN_LT, BLUE_LT = 150, 125, 125
MIN_COMPONENT_PIXELS = 300
ANCHORS = [0, 4, 8, 12, 16]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    manifest_hash = sha256(MANIFEST)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Wan retry5 manifest changed: {manifest_hash}")
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("The preregistered mask protocol changed; refusing to score.")
    flow_hash = sha256(FLOW_JSON)
    mae_hash = sha256(RGB_MAE_JSON)
    if flow_hash != EXPECTED_FLOW_SHA256 or mae_hash != EXPECTED_RGB_MAE_SHA256:
        raise RuntimeError("A pinned Farneback or RGB-MAE comparison result changed.")

    manifest = load_json(MANIFEST)
    manifest_frames = {
        int(path.rsplit("frame_", 1)[1].split(".", 1)[0]): digest
        for path, digest in manifest["files_sha256"].items()
        if path.startswith("run_001/frames/frame_") and path.endswith(".png")
    }
    if sorted(manifest_frames) != list(range(17)):
        raise RuntimeError(f"Expected exactly frames 0..16 in manifest; found {sorted(manifest_frames)}")

    flow_ref = load_json(FLOW_JSON)
    mae_ref = load_json(RGB_MAE_JSON)
    for reference, name in ((flow_ref, "Farneback"), (mae_ref, "RGB-MAE")):
        hashes = {row["frame"]: row["sha256"] for row in reference["frame_sha256"]}
        if hashes != manifest_frames:
            raise RuntimeError(f"{name} frame hashes do not match the Wan manifest.")
        if reference.get("anchors") != ANCHORS or reference.get("frame_count") != 17:
            raise RuntimeError(f"{name} reference has a different frame or anchor schedule.")

    # Recheck source bytes before loading any RGB pixels.
    frame_paths = [FRAMES / f"frame_{index:03d}.png" for index in range(17)]
    source_hashes = []
    for index, path in enumerate(frame_paths):
        actual = sha256(path) if path.is_file() else None
        expected = manifest_frames[index]
        if actual != expected:
            raise RuntimeError(f"Frame {index} hash mismatch: expected {expected}; got {actual}")
        source_hashes.append({"frame": index, "path": str(path.relative_to(ROOT)), "sha256": actual})

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
                "second_largest_component_area_px": (
                    components[1]["area_px"] if len(components) > 1 else 0
                ),
                "mask_pixels_outside_largest_px": outside_area,
                "isolation_check_passed": isolated,
            }
        )
        selected_components.append(components[0] if components else None)

    failures = [row["frame"] for row in isolation_rows if not row["isolation_check_passed"]]
    output = {
        "audit_id": "EXP-M0-WAN-RETRY5-RED-BALL-MASK-001",
        "created_date": "2026-10-17",
        "status": "stopped_mask_ambiguous" if failures else "completed_cpu_red_mask_component_proxy",
        "scope": "Preregistered deterministic RGB mask on existing, hash-verified Wan retry5 PNGs only.",
        "protocol": {
            "path": str(PROTOCOL.relative_to(ROOT)),
            "sha256": EXPECTED_PROTOCOL_SHA256,
            "rgb_threshold": {"r_gt": RED_GT, "g_lt": GREEN_LT, "b_lt": BLUE_LT},
            "connectivity": 8,
            "minimum_component_pixels": MIN_COMPONENT_PIXELS,
            "isolation_rule": "largest component >= 300 pixels and fewer than 300 threshold-mask pixels in all other components combined, in every frame",
            "threshold_tuning_or_temporal_association": False,
        },
        "source_manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": manifest_hash},
        "frame_count": len(source_hashes),
        "frame_size_px": [832, 480],
        "frame_sha256": source_hashes,
        "rgb_mae_reference": {"path": str(RGB_MAE_JSON.relative_to(ROOT)), "sha256": mae_hash},
        "farneback_reference": {"path": str(FLOW_JSON.relative_to(ROOT)), "sha256": flow_hash},
        "isolation_check": {
            "passed": not failures,
            "failed_frames": failures,
            "per_frame_component_evidence": isolation_rows,
        },
        "limitations": [
            "A color-connected component is an image-space proxy, not semantic object identity or ground-truth tracking.",
            "The deterministic isolation gate is intentionally conservative; failure requires independently specified or human-reviewed masks, not threshold tuning.",
            "One saved clip only; no sampling-policy change follows from this diagnostic.",
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
        output["measurements"] = {
            "per_frame": frame_rows,
            "adjacent_pairs": adjacent_rows,
            "summary": {
                "mean_adjacent_centroid_displacement_px": float(np.mean(displacements)),
                "sum_adjacent_centroid_displacement_px": float(np.sum(displacements)),
                "direct_frame_0_to_16_centroid_displacement_px": math.hypot(
                    frame_rows[-1]["centroid_xy_px"][0] - frame_rows[0]["centroid_xy_px"][0],
                    frame_rows[-1]["centroid_xy_px"][1] - frame_rows[0]["centroid_xy_px"][1],
                ),
                "largest_mask_centroid_displacement_pair": {
                    "from_frame": max_displacement["from_frame"],
                    "to_frame": max_displacement["to_frame"],
                    "displacement_px": max_displacement["centroid_displacement_px"],
                },
                "existing_farneback_peak_pair": flow_ref["within_clip_ranking"]["highest_flow_mean_pair"],
                "existing_rgb_mae_peak_pair": flow_ref["within_clip_ranking"]["highest_rgb_mae_pair"],
                "farneback_vs_rgb_mae_spearman": flow_ref["within_clip_ranking"]["spearman_flow_mean_vs_rgb_mae"],
            },
        }
    else:
        output["stop_reason"] = (
            "The preregistered component isolation rule failed. No per-frame centroid track or "
            "adjacent object-displacement values were calculated. Obtain independent human-reviewed "
            "object masks/ROIs or an independently specified segmentation method before continuing."
        )

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}; status={output['status']}; failed_frames={failures}")


if __name__ == "__main__":
    main()
