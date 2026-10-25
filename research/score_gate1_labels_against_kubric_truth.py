#!/usr/bin/env python3
"""After both raters pass the blinded agreement screen, score against Kubric truth.

This is a known-truth diagnostic for the synthetic packets. It is deliberately
separate from inter-rater agreement and from generated-video policy evaluation.
The script recomputes the registered blind agreement gate and refuses to open
the unblind key unless the pooled and per-clip screen passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
import platform
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from score_annotation_agreement import (  # noqa: E402
    DEFAULT_CENTER_COVERAGE_MIN,
    DEFAULT_CENTER_P95_CELLS_MAX,
    DEFAULT_STATE_AGREEMENT_MIN,
    load_packet_image_keys,
    load_rows,
    map_source_delta_to_grid,
    percentile,
    summarize as summarize_blind_agreement,
    validate_rows_against_packet,
)

PACKET_ROOT = ROOT / "research" / "gate1_manual_annotation_packets_v5"
TRUTH_PATH = ROOT / "research" / "kubric_gate0_fixture_v1" / "ground_truth.json"
DEFAULT_KEY_PATH = PACKET_ROOT / "PRIVATE_UNBLIND_KEY.json"
DEFAULT_REFERENCE_PACKET = PACKET_ROOT / "synthetic_reliability" / "rater_01.zip"
SOURCE_WIDTH, SOURCE_HEIGHT = 160, 96
TARGET_WIDTH, TARGET_HEIGHT = 832, 480
GRID_WIDTH, GRID_HEIGHT = 52, 30
FRAME_COUNT = 17


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def check_complete(rows: dict, label: str) -> None:
    clip_ids = sorted({cid for cid, _ in rows})
    if len(clip_ids) != 2:
        raise ValueError(f"{label} must contain both blinded clips; found {len(clip_ids)}")
    expected = set(range(FRAME_COUNT))
    for clip_id in clip_ids:
        frames = {frame for cid, frame in rows if cid == clip_id}
        if frames != expected:
            raise ValueError(f"{label} clip {clip_id} must contain frames 0..16")


def compare_rater(
    rows: dict,
    *,
    key: dict,
    truth: dict,
) -> dict[str, Any]:
    if set(key) != {cid for cid, _ in rows}:
        raise ValueError("Unblind key clip IDs do not match the completed rater CSV")

    per_clip: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for blind_id, direction in sorted(key.items()):
        truth_rows = truth.get(direction)
        if not isinstance(truth_rows, list) or len(truth_rows) != FRAME_COUNT:
            raise ValueError(f"Ground truth for {direction} must contain 17 frames")
        if [int(row["frame_index"]) for row in truth_rows] != list(range(FRAME_COUNT)):
            raise ValueError(f"Ground-truth frame sequence for {direction} is incomplete or unordered")

        records = []
        for frame_idx, truth_row in enumerate(truth_rows):
            annotation = rows[(blind_id, frame_idx)]
            mask_anno = truth_row["annotation_from_mask"]
            expected_visible = bool(mask_anno["object_visible"])
            expected_count = int(mask_anno["instance_count"])
            expected_identity = bool(mask_anno["identity_consistent"])
            category = {
                "visibility_correct": annotation["object_visible"] == expected_visible,
                "instance_count_correct": annotation["instance_count"] == expected_count,
                "identity_correct": annotation["identity_consistent"] == expected_identity,
            }
            expected_box_raw = mask_anno["bbox_xyxy_px"]
            human_box = annotation["bbox"]
            annotation_is_unique_target = (
                annotation["object_visible"]
                and annotation["instance_count"] == 1
                and annotation["identity_consistent"]
                and human_box is not None
            )
            expected_is_unique_target = (
                expected_visible
                and expected_count == 1
                and expected_identity
                and expected_box_raw is not None
            )
            geometry_scoreable = (
                expected_is_unique_target
                and annotation_is_unique_target
            )
            record: dict[str, Any] = {
                "frame_index": frame_idx,
                **category,
                "all_categories_correct": all(category.values()),
                "has_human_box": human_box is not None,
                "expected_visible": expected_visible,
                "expected_unique_target": expected_is_unique_target,
                "annotation_claims_unique_target": annotation_is_unique_target,
                "geometry_scoreable": geometry_scoreable,
            }

            if geometry_scoreable:
                expected_box = [float(value) for value in expected_box_raw]
                human_center = (
                    (human_box[0] + human_box[2]) / 2.0,
                    (human_box[1] + human_box[3]) / 2.0,
                )
                mask_center = (
                    (expected_box[0] + expected_box[2]) / 2.0,
                    (expected_box[1] + expected_box[3]) / 2.0,
                )
                target_center_norm = truth_row["projected_target_center_xy_normalized"]
                projected_center = (
                    float(target_center_norm[0]) * SOURCE_WIDTH,
                    float(target_center_norm[1]) * SOURCE_HEIGHT,
                )
                dx_mask = human_center[0] - mask_center[0]
                dy_mask = human_center[1] - mask_center[1]
                dx_projected = human_center[0] - projected_center[0]
                dy_projected = human_center[1] - projected_center[1]
                mask_cell_delta = map_source_delta_to_grid(
                    dx_mask, dy_mask,
                    source_width=SOURCE_WIDTH,
                    source_height=SOURCE_HEIGHT,
                    target_width=TARGET_WIDTH,
                    target_height=TARGET_HEIGHT,
                    grid_width=GRID_WIDTH,
                    grid_height=GRID_HEIGHT,
                )
                projected_cell_delta = map_source_delta_to_grid(
                    dx_projected, dy_projected,
                    source_width=SOURCE_WIDTH,
                    source_height=SOURCE_HEIGHT,
                    target_width=TARGET_WIDTH,
                    target_height=TARGET_HEIGHT,
                    grid_width=GRID_WIDTH,
                    grid_height=GRID_HEIGHT,
                )
                edge_errors = [abs(h - e) for h, e in zip(human_box, expected_box)]
                record.update({
                    "center_dx_vs_mask_source_px": dx_mask,
                    "center_dy_vs_mask_source_px": dy_mask,
                    "center_distance_vs_mask_source_px": math.hypot(dx_mask, dy_mask),
                    "center_abs_dx_vs_mask_cells": abs(mask_cell_delta["dx_cells"]),
                    "center_abs_dy_vs_mask_cells": abs(mask_cell_delta["dy_cells"]),
                    "center_dx_vs_projected_source_px": dx_projected,
                    "center_dy_vs_projected_source_px": dy_projected,
                    "center_distance_vs_projected_source_px": math.hypot(dx_projected, dy_projected),
                    "center_abs_dx_vs_projected_cells": abs(projected_cell_delta["dx_cells"]),
                    "center_abs_dy_vs_projected_cells": abs(projected_cell_delta["dy_cells"]),
                    "bbox_iou_vs_mask": box_iou(human_box, expected_box),
                    "bbox_edge_abs_errors_source_px": edge_errors,
                })
            else:
                record.update({
                    "center_dx_vs_mask_source_px": None,
                    "center_dy_vs_mask_source_px": None,
                    "center_distance_vs_mask_source_px": None,
                    "center_abs_dx_vs_mask_cells": None,
                    "center_abs_dy_vs_mask_cells": None,
                    "center_dx_vs_projected_source_px": None,
                    "center_dy_vs_projected_source_px": None,
                    "center_distance_vs_projected_source_px": None,
                    "center_abs_dx_vs_projected_cells": None,
                    "center_abs_dy_vs_projected_cells": None,
                    "bbox_iou_vs_mask": None,
                    "bbox_edge_abs_errors_source_px": None,
                })
            records.append(record)
            all_records.append({"blind_clip_id": blind_id, **record})

        def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
            scoreable = [item for item in items if item["center_distance_vs_mask_source_px"] is not None]
            edge_by_index = [[item["bbox_edge_abs_errors_source_px"][i] for item in scoreable] for i in range(4)]
            return {
                "frames": len(items),
                "visibility_accuracy": statistics.mean(item["visibility_correct"] for item in items),
                "instance_count_accuracy": statistics.mean(item["instance_count_correct"] for item in items),
                "identity_accuracy": statistics.mean(item["identity_correct"] for item in items),
                "joint_categorical_accuracy": statistics.mean(item["all_categories_correct"] for item in items),
                "expected_visible_box_coverage": sum(item["has_human_box"] for item in items if item["expected_visible"]) / max(1, sum(item["expected_visible"] for item in items)),
                "box_scoreable_frames": len(scoreable),
                "unique_identity_consistent_box_coverage": len(scoreable) / max(1, sum(item["expected_visible"] for item in items)),
                "center_distance_vs_mask_source_px": summarize([item["center_distance_vs_mask_source_px"] for item in scoreable]),
                "center_abs_dx_vs_mask_cells": summarize([item["center_abs_dx_vs_mask_cells"] for item in scoreable]),
                "center_abs_dy_vs_mask_cells": summarize([item["center_abs_dy_vs_mask_cells"] for item in scoreable]),
                "signed_center_bias_vs_mask_source_px": {
                    "dx_mean": statistics.mean(item["center_dx_vs_mask_source_px"] for item in scoreable) if scoreable else None,
                    "dy_mean": statistics.mean(item["center_dy_vs_mask_source_px"] for item in scoreable) if scoreable else None,
                },
                "center_distance_vs_projected_source_px": summarize([item["center_distance_vs_projected_source_px"] for item in scoreable]),
                "center_abs_dx_vs_projected_cells": summarize([item["center_abs_dx_vs_projected_cells"] for item in scoreable]),
                "center_abs_dy_vs_projected_cells": summarize([item["center_abs_dy_vs_projected_cells"] for item in scoreable]),
                "bbox_iou_vs_mask": summarize([item["bbox_iou_vs_mask"] for item in scoreable]),
                "bbox_edge_abs_error_source_px": {
                    name: summarize(edge_by_index[index])
                    for index, name in enumerate(("x1", "y1", "x2", "y2"))
                },
            }

        per_clip[blind_id] = {"unblinded_direction": direction, "metrics": aggregate(records)}

    # Quantify the unavoidable difference between the discrete mask box center
    # and the continuous projected 3D object center, independent of human labels.
    truth_representation_gap = []
    for direction in key.values():
        for row in truth[direction]:
            box = row["annotation_from_mask"]["bbox_xyxy_px"]
            target = row["projected_target_center_xy_normalized"]
            mask_cx, mask_cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            projected_x, projected_y = target[0] * SOURCE_WIDTH, target[1] * SOURCE_HEIGHT
            truth_representation_gap.append({
                "direction": direction,
                "frame_index": int(row["frame_index"]),
                "dx_source_px": mask_cx - projected_x,
                "dy_source_px": mask_cy - projected_y,
                "distance_source_px": math.hypot(mask_cx - projected_x, mask_cy - projected_y),
            })

    return {
        "per_clip": per_clip,
        "pooled": {
            "frames": len(all_records),
            "visibility_accuracy": statistics.mean(item["visibility_correct"] for item in all_records),
            "instance_count_accuracy": statistics.mean(item["instance_count_correct"] for item in all_records),
            "identity_accuracy": statistics.mean(item["identity_correct"] for item in all_records),
            "joint_categorical_accuracy": statistics.mean(item["all_categories_correct"] for item in all_records),
            "expected_visible_box_coverage": sum(item["has_human_box"] for item in all_records if item["expected_visible"]) / max(1, sum(item["expected_visible"] for item in all_records)),
            "scoreable_box_frames": sum(item["center_distance_vs_mask_source_px"] is not None for item in all_records),
            "unique_identity_consistent_box_coverage": sum(item["center_distance_vs_mask_source_px"] is not None for item in all_records) / max(1, sum(item["expected_visible"] for item in all_records)),
            "center_distance_vs_mask_source_px": summarize([item["center_distance_vs_mask_source_px"] for item in all_records if item["center_distance_vs_mask_source_px"] is not None]),
            "center_abs_dx_vs_mask_cells": summarize([item["center_abs_dx_vs_mask_cells"] for item in all_records if item["center_abs_dx_vs_mask_cells"] is not None]),
            "center_abs_dy_vs_mask_cells": summarize([item["center_abs_dy_vs_mask_cells"] for item in all_records if item["center_abs_dy_vs_mask_cells"] is not None]),
            "signed_center_bias_vs_mask_source_px": {
                "dx_mean": statistics.mean(item["center_dx_vs_mask_source_px"] for item in all_records if item["center_dx_vs_mask_source_px"] is not None) if any(item["center_dx_vs_mask_source_px"] is not None for item in all_records) else None,
                "dy_mean": statistics.mean(item["center_dy_vs_mask_source_px"] for item in all_records if item["center_dy_vs_mask_source_px"] is not None) if any(item["center_dy_vs_mask_source_px"] is not None for item in all_records) else None,
            },
            "center_distance_vs_projected_source_px": summarize([item["center_distance_vs_projected_source_px"] for item in all_records if item["center_distance_vs_projected_source_px"] is not None]),
            "bbox_iou_vs_mask": summarize([item["bbox_iou_vs_mask"] for item in all_records if item["bbox_iou_vs_mask"] is not None]),
        },
        "mask_bbox_center_vs_projected_center_truth_gap": summarize([r["distance_source_px"] for r in truth_representation_gap]),
        "frame_records": all_records,
        "threshold_policy": "Descriptive known-truth diagnostic only; no new pass threshold. Keep separate from inter-rater process-screen thresholds.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rater_a", type=Path)
    parser.add_argument("rater_b", type=Path)
    parser.add_argument("--unblind-key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--ground-truth", type=Path, default=TRUTH_PATH)
    parser.add_argument("--reference-packet", type=Path, default=DEFAULT_REFERENCE_PACKET,
                        help="Blinded v5 packet ZIP used to verify CSV keys before the unblind gate; its private key is never read.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rater_a.resolve() == args.rater_b.resolve():
        parser.error("Provide two separate rater CSV files")

    started = time.perf_counter()
    rows_a = load_rows(args.rater_a, frame_width=SOURCE_WIDTH, frame_height=SOURCE_HEIGHT)
    rows_b = load_rows(args.rater_b, frame_width=SOURCE_WIDTH, frame_height=SOURCE_HEIGHT)
    expected_image_keys = load_packet_image_keys(args.reference_packet)
    packet_key_validation = {
        "reference_packet_sha256": sha256(args.reference_packet),
        "rater_a": validate_rows_against_packet(rows_a, expected_image_keys, label="Rater A"),
        "rater_b": validate_rows_against_packet(rows_b, expected_image_keys, label="Rater B"),
    }
    check_complete(rows_a, "Rater A")
    check_complete(rows_b, "Rater B")
    if set(rows_a) != set(rows_b):
        raise ValueError("Rater CSVs must have identical blinded clip/frame keys")

    # Recompute the preregistered pairwise screen from these exact inputs. Do
    # not trust a separate/stale report: a failed or mismatched pair must stay
    # blind while the instructions or labels are repaired.
    blind_agreement = summarize_blind_agreement(
        rows_a,
        rows_b,
        source_width=SOURCE_WIDTH,
        source_height=SOURCE_HEIGHT,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        state_min=DEFAULT_STATE_AGREEMENT_MIN,
        center_p95_cells_max=DEFAULT_CENTER_P95_CELLS_MAX,
        center_coverage_min=DEFAULT_CENTER_COVERAGE_MIN,
        expected_clips=2,
        expected_frames=FRAME_COUNT,
    )
    if not blind_agreement["overall"]["pass_screen"]:
        raise ValueError(
            "Blinded inter-rater screen failed; refusing to unblind. "
            "Resolve the disagreement or repair the annotation process while clips remain blind."
        )

    # Only after the exact CSV pair passes pooled and per-clip blind criteria is the key read.
    key_record = json.loads(args.unblind_key.read_text(encoding="utf-8"))
    key = key_record["synthetic"]
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    if set(key) != {clip_id for clip_id, _ in rows_a}:
        raise ValueError("Unblind key clip IDs do not match the two complete submissions")
    if set(key.values()) != {"left", "right"}:
        raise ValueError("Expected one left and one right truth sequence")

    result = {
        "experiment_id": "EXP-LATENTGATE-GATE1-HUMAN-TO-KUBRIC-TRUTH-001",
        "status": "diagnostic_only_not_human_gate_or_policy_efficacy",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rater_inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (args.rater_a, args.rater_b)
        ],
        "packet_key_validation": packet_key_validation,
        "unblind_key_sha256": sha256(args.unblind_key),
        "ground_truth_sha256": sha256(args.ground_truth),
        "coordinate_system": {
            "source_wh": [SOURCE_WIDTH, SOURCE_HEIGHT],
            "target_wh": [TARGET_WIDTH, TARGET_HEIGHT],
            "grid_wh": [GRID_WIDTH, GRID_HEIGHT],
            "pixel_edge_boxes_right_bottom_exclusive": True,
        },
        "blind_agreement_gate": {
            "status": "passed_before_unblinding",
            "thresholds": {
                "joint_visibility_count_identity_exact_rate_min": DEFAULT_STATE_AGREEMENT_MIN,
                "p95_absolute_center_disagreement_cells_per_axis_max": DEFAULT_CENTER_P95_CELLS_MAX,
                "unique_visible_center_pair_coverage_min": DEFAULT_CENTER_COVERAGE_MIN,
                "expected_clips": 2,
                "expected_frames_per_clip": FRAME_COUNT,
            },
            "overall": blind_agreement["overall"],
            "per_clip": blind_agreement["per_clip"],
            "aggregation_rule": blind_agreement["aggregation_rule"],
        },
        "rater_a": compare_rater(rows_a, key=key, truth=truth),
        "rater_b": compare_rater(rows_b, key=key, truth=truth),
        "compute": {
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
            "gpu_used": False,
            "peak_vram_mb": None,
            "model_inference": False,
        },
        "limitation": "Kubric known-truth diagnostic only. It measures shared rater bias on this rendered fixture; it does not replace inter-rater reliability, establish observer transfer to Wan, or measure policy efficacy.",
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
