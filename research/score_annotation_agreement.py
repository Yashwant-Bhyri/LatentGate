#!/usr/bin/env python3
"""Compare two blinded box-label CSVs without unblinding experimental arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from pathlib import Path
from typing import Any


REQUIRED = {
    "blind_clip_id", "frame_index", "image_file", "object_visible", "instance_count",
    "identity_consistent", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
}
DEFAULT_STATE_AGREEMENT_MIN = 0.95
DEFAULT_CENTER_P95_CELLS_MAX = 0.5
DEFAULT_CENTER_COVERAGE_MIN = 0.90


def as_bool(value: str, name: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{name} must be explicitly true/false, got {value!r}")


def load_rows(path: Path, *, frame_width: int, frame_height: int) -> dict[tuple[str, int], dict[str, Any]]:
    if frame_width < 1 or frame_height < 1:
        raise ValueError("Source frame dimensions must be positive")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for raw in reader:
            clip_id_value = raw.get("blind_clip_id")
            clip_id = clip_id_value.strip() if isinstance(clip_id_value, str) else ""
            if not clip_id:
                raise ValueError(f"{path} has an empty blind_clip_id")
            frame = int(raw["frame_index"])
            image_file_value = raw.get("image_file")
            image_file = image_file_value.strip() if isinstance(image_file_value, str) else ""
            if not image_file:
                raise ValueError(f"{path} has an empty image_file for blind frame {(clip_id, frame)}")
            key = clip_id, frame
            if key in rows:
                raise ValueError(f"{path} has a duplicate row for {key}")
            visible = as_bool(raw["object_visible"], "object_visible")
            identity = as_bool(raw["identity_consistent"], "identity_consistent")
            count = int(raw["instance_count"])
            coords = [raw[f"bbox_{axis}_px"].strip() for axis in ("x1", "y1", "x2", "y2")]
            box = [float(v) for v in coords] if all(coords) else None
            if any(coords) and not all(coords):
                raise ValueError(f"{path} has a partially filled box for {key}")
            if box and not (box[2] > box[0] and box[3] > box[1]):
                raise ValueError(f"{path} has a degenerate box for {key}: {box}")
            if box and not (0 <= box[0] < box[2] <= frame_width and 0 <= box[1] < box[3] <= frame_height):
                raise ValueError(
                    f"{path} has a box outside the declared {frame_width}x{frame_height} source frame "
                    f"for {key}: {box}"
                )
            if visible and (count < 1 or box is None):
                raise ValueError(f"Visible target at {key} needs count >= 1 and a box")
            if not visible and (count != 0 or box is not None):
                raise ValueError(f"Absent target at {key} needs count 0 and no box")
            if not visible and identity:
                raise ValueError(f"Absent target at {key} must be marked identity No / uncertain")
            rows[key] = {
                "object_visible": visible,
                "instance_count": count,
                "identity_consistent": identity,
                "image_file": image_file,
                "bbox": box,
            }
    return rows


def load_packet_image_keys(packet_path: Path) -> dict[tuple[str, int], str]:
    """Read the packet's blinded CSV frame-to-image mapping, never its private key."""
    with zipfile.ZipFile(packet_path) as archive:
        members = archive.namelist()
        if any("UNBLIND" in Path(name).name.upper() or "PRIVATE_KEY" in Path(name).name.upper()
               for name in members):
            raise ValueError(f"Refusing packet key audit because {packet_path} contains a private-key-looking member")
        csv_members = [name for name in members if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"{packet_path} must contain exactly one annotation template CSV")
        with archive.open(csv_members[0]) as binary_stream:
            stream = io.TextIOWrapper(binary_stream, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(stream)
            required = {"blind_clip_id", "frame_index", "image_file"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{packet_path} template is missing frame-key columns: {sorted(missing)}")
            image_keys: dict[tuple[str, int], str] = {}
            for raw in reader:
                clip_id = (raw.get("blind_clip_id") or "").strip()
                image_file = (raw.get("image_file") or "").strip()
                if not clip_id or not image_file:
                    raise ValueError(f"{packet_path} has an empty blind clip ID or image filename")
                try:
                    frame_idx = int(raw["frame_index"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{packet_path} has an invalid frame index") from error
                key = (clip_id, frame_idx)
                if key in image_keys:
                    raise ValueError(f"{packet_path} has a duplicate frame key")
                image_keys[key] = image_file
    if not image_keys:
        raise ValueError(f"{packet_path} annotation template has no frame keys")
    return image_keys


def validate_rows_against_packet(
    rows: dict[tuple[str, int], dict[str, Any]],
    expected_image_keys: dict[tuple[str, int], str],
    *,
    label: str,
) -> dict[str, Any]:
    """Fail closed unless a submission preserves the packet's exact blind frame/image map."""
    if set(rows) != set(expected_image_keys):
        missing_count = len(set(expected_image_keys) - set(rows))
        extra_count = len(set(rows) - set(expected_image_keys))
        raise ValueError(
            f"{label} does not match the archived packet frame keys "
            f"(missing={missing_count}, extra={extra_count})"
        )
    mismatches = [key for key, image_file in expected_image_keys.items()
                  if rows[key].get("image_file") != image_file]
    if mismatches:
        raise ValueError(
            f"{label} does not match the archived packet key-to-image map "
            f"({len(mismatches)} filename mismatches)"
        )
    return {
        "exact_frame_key_set": True,
        "exact_key_to_image_mapping": True,
        "frame_keys_validated": len(expected_image_keys),
        "packet_alias_direction_read": False,
        "private_unblind_key_read": False,
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def map_source_delta_to_grid(
    dx_source_px: float,
    dy_source_px: float,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    grid_width: int,
    grid_height: int,
) -> dict[str, float]:
    """Map source-image axis deltas into target pixels and target-grid cells."""
    if min(source_width, source_height, target_width, target_height, grid_width, grid_height) < 1:
        raise ValueError("Source, target, and grid dimensions must be positive")
    dx_target_px = dx_source_px * target_width / source_width
    dy_target_px = dy_source_px * target_height / source_height
    return {
        "dx_target_px": dx_target_px,
        "dy_target_px": dy_target_px,
        "dx_cells": dx_target_px / (target_width / grid_width),
        "dy_cells": dy_target_px / (target_height / grid_height),
    }


def summarize(
    rows_a: dict,
    rows_b: dict,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    grid_width: int,
    grid_height: int,
    state_min: float,
    center_p95_cells_max: float,
    center_coverage_min: float,
    expected_clips: int,
    expected_frames: int,
) -> dict[str, Any]:
    if min(source_width, source_height, target_width, target_height, grid_width, grid_height) < 1:
        raise ValueError("Source, target, and grid dimensions must be positive")
    if set(rows_a) != set(rows_b):
        missing_b = sorted(set(rows_a) - set(rows_b))
        missing_a = sorted(set(rows_b) - set(rows_a))
        raise ValueError(f"Rater frame keys differ: only A={missing_b}; only B={missing_a}")
    if not rows_a:
        raise ValueError("No annotation rows found")
    clip_ids = sorted({clip_id for clip_id, _ in rows_a})
    if len(clip_ids) != expected_clips:
        raise ValueError(f"Expected {expected_clips} distinct clips, found {len(clip_ids)}: {clip_ids}")
    for clip_id in clip_ids:
        present = {frame for cid, frame in rows_a if cid == clip_id}
        expected = set(range(expected_frames))
        if present != expected:
            raise ValueError(f"{clip_id} must have frames 0..{expected_frames - 1}; got {sorted(present)}")

    per_frame: list[dict[str, Any]] = []
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for clip_id, frame_idx in sorted(rows_a):
        a, b = rows_a[(clip_id, frame_idx)], rows_b[(clip_id, frame_idx)]
        if a["image_file"] != b["image_file"]:
            raise ValueError(f"Rater image_file values differ for blind frame {(clip_id, frame_idx)}")
        fields_equal = {
            field: a[field] == b[field]
            for field in ("object_visible", "instance_count", "identity_consistent")
        }
        boxes = a["bbox"] is not None and b["bbox"] is not None
        center_comparable = (
            boxes
            and a["object_visible"] and b["object_visible"]
            and a["instance_count"] == b["instance_count"] == 1
            and a["identity_consistent"] and b["identity_consistent"]
        )
        dx_source = dy_source = dx_target = dy_target = dx_cells = dy_cells = None
        source_distance = target_distance = iou = None
        if boxes:
            ca = ((a["bbox"][0] + a["bbox"][2]) / 2, (a["bbox"][1] + a["bbox"][3]) / 2)
            cb = ((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
            dx_source, dy_source = abs(ca[0] - cb[0]), abs(ca[1] - cb[1])
            mapped = map_source_delta_to_grid(
                dx_source, dy_source,
                source_width=source_width, source_height=source_height,
                target_width=target_width, target_height=target_height,
                grid_width=grid_width, grid_height=grid_height,
            )
            dx_target, dy_target = mapped["dx_target_px"], mapped["dy_target_px"]
            dx_cells, dy_cells = mapped["dx_cells"], mapped["dy_cells"]
            source_distance = math.hypot(dx_source, dy_source)
            target_distance = math.hypot(dx_target, dy_target)
            iou = box_iou(a["bbox"], b["bbox"])
        record = {
            "blind_clip_id": clip_id,
            "frame_index": frame_idx,
            "state_exact": all(fields_equal.values()),
            "visibility_exact": fields_equal["object_visible"],
            "instance_count_exact": fields_equal["instance_count"],
            "identity_exact": fields_equal["identity_consistent"],
            "image_file_match": True,
            "both_have_box": boxes,
            "center_comparable": center_comparable,
            "center_abs_dx_source_px": dx_source,
            "center_abs_dy_source_px": dy_source,
            "center_abs_dx_target_px": dx_target,
            "center_abs_dy_target_px": dy_target,
            "center_abs_dx_cells": dx_cells,
            "center_abs_dy_cells": dy_cells,
            "center_distance_source_px": source_distance,
            "center_distance_target_px": target_distance,
            "box_iou": iou,
        }
        per_frame.append(record)
        by_clip.setdefault(clip_id, []).append(record)

    def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
        both_boxes = [r for r in records if r["both_have_box"]]
        centers = [r for r in records if r["center_comparable"]]
        dx_cells = [r["center_abs_dx_cells"] for r in centers]
        dy_cells = [r["center_abs_dy_cells"] for r in centers]
        state_rate = statistics.mean(r["state_exact"] for r in records)
        p95x, p95y = percentile(dx_cells, 0.95), percentile(dy_cells, 0.95)
        worst_center = max(centers, key=lambda r: r["center_distance_target_px"], default=None)
        worst_x = max(centers, key=lambda r: r["center_abs_dx_target_px"], default=None)
        worst_y = max(centers, key=lambda r: r["center_abs_dy_target_px"], default=None)
        return {
            "frame_pairs": len(records),
            "joint_visibility_count_identity_exact_rate": state_rate,
            "visibility_exact_rate": statistics.mean(r["visibility_exact"] for r in records),
            "instance_count_exact_rate": statistics.mean(r["instance_count_exact"] for r in records),
            "identity_exact_rate": statistics.mean(r["identity_exact"] for r in records),
            "both_box_coverage": len(both_boxes) / len(records),
            "unique_visible_center_pair_coverage": len(centers) / len(records),
            "center_pair_count": len(centers),
            "center_abs_dx_cells_p50": percentile(dx_cells, 0.50),
            "center_abs_dx_cells_p95": p95x,
            "center_abs_dy_cells_p50": percentile(dy_cells, 0.50),
            "center_abs_dy_cells_p95": p95y,
            "center_abs_dx_target_px_p95": percentile([r["center_abs_dx_target_px"] for r in centers], 0.95),
            "center_abs_dy_target_px_p95": percentile([r["center_abs_dy_target_px"] for r in centers], 0.95),
            "center_distance_target_px_p95": percentile([r["center_distance_target_px"] for r in centers], 0.95),
            "center_abs_dx_cells_max": max(dx_cells, default=None),
            "center_abs_dy_cells_max": max(dy_cells, default=None),
            "center_distance_target_px_max": worst_center["center_distance_target_px"] if worst_center else None,
            "center_distance_target_px_max_frame": worst_center["frame_index"] if worst_center else None,
            "center_abs_dx_target_px_max_frame": worst_x["frame_index"] if worst_x else None,
            "center_abs_dy_target_px_max_frame": worst_y["frame_index"] if worst_y else None,
            "center_distance_source_px_p95": percentile([r["center_distance_source_px"] for r in centers], 0.95),
            "bbox_iou_p50": percentile([r["box_iou"] for r in both_boxes], 0.50),
            "bbox_iou_p05": percentile([r["box_iou"] for r in both_boxes], 0.05),
            "pass_screen": bool(
                state_rate >= state_min
                and p95x is not None and p95y is not None
                and len(centers) / len(records) >= center_coverage_min
                and p95x <= center_p95_cells_max and p95y <= center_p95_cells_max
            ),
        }

    clips = {clip_id: aggregate(records) for clip_id, records in sorted(by_clip.items())}
    overall = aggregate(per_frame)
    pooled_pass = overall["pass_screen"]
    all_clips_pass = all(result["pass_screen"] for result in clips.values())
    overall["pooled_thresholds_pass"] = pooled_pass
    overall["all_clip_thresholds_pass"] = all_clips_pass
    overall["pass_screen"] = pooled_pass and all_clips_pass
    return {
        "metric": "blinded_human_annotation_agreement_v2",
        "coordinate_unit": "continuous pixel-edge box coordinates; tight-box midpoint; disagreements mapped per axis from source frame to target frame, then to target-grid cells",
        "source_frame_wh": [source_width, source_height],
        "target_frame_wh": [target_width, target_height],
        "control_grid_wh": [grid_width, grid_height],
        "coordinate_transform": {
            "x_target_px": "x_source_px * target_width / source_width",
            "y_target_px": "y_source_px * target_height / source_height",
            "x_cells": "x_source_px * grid_width / source_width",
            "y_cells": "y_source_px * grid_height / source_height",
        },
        "screen_thresholds": {
            "joint_visibility_count_identity_exact_rate_min": state_min,
            "p95_absolute_center_disagreement_cells_per_axis_max": center_p95_cells_max,
            "unique_visible_center_pair_coverage_min": center_coverage_min,
        },
        "aggregation_rule": "Pass only if the pooled screen and each clip-specific screen pass; pooled agreement cannot hide a weak mirrored direction. Maximum target-resolution center disagreement is diagnostic and does not add a new threshold.",
        "expected_clip_count": expected_clips,
        "expected_frames_per_clip": expected_frames,
        "overall": overall,
        "per_clip": clips,
        "frame_records": per_frame,
        "limitation": "This screening statistic is descriptive for the labeled clips; frames within clips are not independent samples. Source-to-target mapping assumes the declared frames correspond by direct anisotropic resize; it does not correct camera or content differences between separately generated images.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rater_a", type=Path)
    parser.add_argument("rater_b", type=Path)
    parser.add_argument("--source-width", type=int, required=True,
                        help="Width of the image resolution in which the boxes were annotated (160 for Kubric packets; 832 for Wan M0).")
    parser.add_argument("--source-height", type=int, required=True,
                        help="Height of the image resolution in which the boxes were annotated (96 for Kubric packets; 480 for Wan M0).")
    parser.add_argument("--target-width", type=int, default=832)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--grid-width", type=int, default=52)
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--expected-clips", type=int, required=True,
                        help="Expected distinct blind clip IDs; set this from the packet (2 for mirrored Kubric, 1 for M0).")
    parser.add_argument("--reference-packet", type=Path, required=True,
                        help="The corresponding blinded annotation packet ZIP; only its CSV frame/image keys are read.")
    parser.add_argument("--expected-frames", type=int, default=17)
    parser.add_argument("--state-agreement-min", type=float, default=DEFAULT_STATE_AGREEMENT_MIN)
    parser.add_argument("--center-p95-cells-max", type=float, default=DEFAULT_CENTER_P95_CELLS_MAX)
    parser.add_argument("--center-coverage-min", type=float, default=DEFAULT_CENTER_COVERAGE_MIN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (min(args.source_width, args.source_height, args.target_width, args.target_height, args.grid_width, args.grid_height) < 1
            or args.expected_clips < 1 or args.expected_frames < 1 or not 0 <= args.state_agreement_min <= 1
            or args.center_p95_cells_max < 0 or not 0 <= args.center_coverage_min <= 1):
        parser.error("Dimensions and error thresholds must be positive/nonnegative; agreement and coverage must lie in [0,1]")
    rows_a = load_rows(args.rater_a, frame_width=args.source_width, frame_height=args.source_height)
    rows_b = load_rows(args.rater_b, frame_width=args.source_width, frame_height=args.source_height)
    expected_image_keys = load_packet_image_keys(args.reference_packet)
    packet_key_validation = {
        "reference_packet": args.reference_packet.name,
        "reference_packet_sha256": hashlib.sha256(args.reference_packet.read_bytes()).hexdigest(),
        "rater_a": validate_rows_against_packet(rows_a, expected_image_keys, label="Rater A"),
        "rater_b": validate_rows_against_packet(rows_b, expected_image_keys, label="Rater B"),
    }
    report = summarize(
        rows_a,
        rows_b,
        source_width=args.source_width, source_height=args.source_height,
        target_width=args.target_width, target_height=args.target_height,
        grid_width=args.grid_width, grid_height=args.grid_height,
        state_min=args.state_agreement_min,
        center_p95_cells_max=args.center_p95_cells_max,
        center_coverage_min=args.center_coverage_min,
        expected_clips=args.expected_clips,
        expected_frames=args.expected_frames,
    )
    report["packet_key_validation"] = packet_key_validation
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
