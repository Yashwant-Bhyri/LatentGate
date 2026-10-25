#!/usr/bin/env python3
"""Score blinded per-frame object boxes against an image-space target path.

Unlike ``evaluate_trajectory_frames.py``, this module consumes structured
human annotations, keeps missing frames in the denominator, and measures
monotonicity only over adjacent frame indices that are both scoreable. Frames
with duplicate instances are hard failures and their ambiguous boxes do not
contribute coordinates to path metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import secrets
import statistics
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


ANNOTATION_COLUMNS = [
    "blind_clip_id",
    "frame_index",
    "image_file",
    "object_visible",
    "instance_count",
    "identity_consistent",
    "bbox_x1_px",
    "bbox_y1_px",
    "bbox_x2_px",
    "bbox_y2_px",
    "notes",
]


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.png$)", path.name)
    if not match:
        raise ValueError(f"Frame filename has no trailing number: {path.name}")
    return int(match.group(1))


def interpolate_target_boxes(
    anchors_yxyx: Sequence[Sequence[float]], frame_count: int
) -> list[list[float]]:
    """Expand normalized [frame,y1,y2,x1,x2] anchors to per-frame xyxy boxes."""
    points = sorted((list(map(float, row)) for row in anchors_yxyx), key=lambda r: r[0])
    if frame_count < 1 or not points:
        raise ValueError("A positive frame_count and at least one target anchor are required")
    if any(len(row) != 5 for row in points):
        raise ValueError("Each target anchor must be [frame,y1,y2,x1,x2]")
    if any(row[0] < 0 or row[0] >= frame_count for row in points):
        raise ValueError("Target anchor lies outside the frame range")
    if len({row[0] for row in points}) != len(points):
        raise ValueError("Target anchor frame indices must be unique")

    result: list[list[float]] = []
    for frame_idx in range(frame_count):
        if frame_idx <= points[0][0]:
            values = points[0][1:]
        elif frame_idx >= points[-1][0]:
            values = points[-1][1:]
        else:
            left = max(i for i, row in enumerate(points) if row[0] <= frame_idx)
            a, b = points[left], points[left + 1]
            alpha = (frame_idx - a[0]) / (b[0] - a[0])
            values = [a[j] + alpha * (b[j] - a[j]) for j in range(1, 5)]
        y1, y2, x1, x2 = values
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"Invalid normalized target box at frame {frame_idx}: {values}")
        result.append([x1, y1, x2, y2])
    return result


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{field} must be an explicit true/false value, got {value!r}")


def read_annotation_csv(path: Path, blind_clip_id: str) -> list[dict[str, Any]]:
    """Read one clip from the structured blinded annotation template."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = set(ANNOTATION_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["blind_clip_id"] != blind_clip_id:
                continue
            visible = _parse_bool(row["object_visible"], "object_visible")
            consistent = _parse_bool(row["identity_consistent"], "identity_consistent")
            try:
                instance_count = int(row["instance_count"])
                frame_idx = int(row["frame_index"])
            except (TypeError, ValueError) as error:
                raise ValueError("frame_index and instance_count must be filled integers") from error
            coords = [row[k].strip() for k in ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")]
            bbox = [float(v) for v in coords] if all(coords) else None
            if any(coords) and not all(coords):
                raise ValueError(f"Frame {frame_idx} has a partially filled bounding box")
            rows.append({
                "frame_index": frame_idx,
                "object_visible": visible,
                "instance_count": instance_count,
                "identity_consistent": consistent,
                "bbox_xyxy_px": bbox,
            })
    if not rows:
        raise ValueError(f"No annotation rows found for blind clip {blind_clip_id!r}")
    return rows


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def evaluate_annotations(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_count: int,
    width: int,
    height: int,
    target_boxes_xyxy_normalized: Sequence[Sequence[float]],
    missing_center_error_penalty: float = 1.0,
) -> dict[str, Any]:
    """Score boxes; missing/identity-ambiguous observations receive max error."""
    if frame_count < 1 or width < 1 or height < 1:
        raise ValueError("frame_count, width, and height must be positive")
    if len(rows) != frame_count or len(target_boxes_xyxy_normalized) != frame_count:
        raise ValueError("Need exactly one annotation row and target box per frame")
    if not 0 <= missing_center_error_penalty <= 1:
        raise ValueError("The missing-center penalty must be between 0 and 1")

    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        frame_idx = int(row["frame_index"])
        if frame_idx in indexed:
            raise ValueError(f"Duplicate annotation row for frame {frame_idx}")
        if not 0 <= frame_idx < frame_count:
            raise ValueError(f"Annotation frame {frame_idx} is out of range")
        indexed[frame_idx] = row
    if set(indexed) != set(range(frame_count)):
        absent = sorted(set(range(frame_count)) - set(indexed))
        raise ValueError(f"Annotation sheet must include explicit rows for every frame; absent={absent}")

    diagonal = math.hypot(width, height)
    centers: list[tuple[float, float] | None] = []
    per_frame_error: list[float] = []
    per_frame_iou: list[float] = []
    records: list[dict[str, Any]] = []
    visible_frames: list[int] = []
    scoreable_frames: list[int] = []
    missing_frames: list[int] = []
    identity_failure_frames: list[int] = []
    duplicate_frames: list[int] = []

    for frame_idx in range(frame_count):
        row = indexed[frame_idx]
        visible = _parse_bool(row["object_visible"], "object_visible")
        identity_ok = _parse_bool(row["identity_consistent"], "identity_consistent")
        instance_count = int(row["instance_count"])
        if instance_count < 0:
            raise ValueError(f"Frame {frame_idx} has a negative instance_count")
        bbox_raw = row.get("bbox_xyxy_px")
        bbox = [float(value) for value in bbox_raw] if bbox_raw is not None else None
        if bbox is not None:
            if len(bbox) != 4:
                raise ValueError(f"Frame {frame_idx} bbox must have four coordinates")
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Frame {frame_idx} bbox is outside the image or degenerate: {bbox}")
        if visible and (instance_count < 1 or bbox is None):
            raise ValueError(f"Visible target at frame {frame_idx} needs a box and instance_count >= 1")

        target = [float(v) for v in target_boxes_xyxy_normalized[frame_idx]]
        if len(target) != 4 or not (0 <= target[0] < target[2] <= 1 and 0 <= target[1] < target[3] <= 1):
            raise ValueError(f"Invalid target box at frame {frame_idx}: {target}")
        target_px = [target[0] * width, target[1] * height, target[2] * width, target[3] * height]

        duplicate = instance_count > 1
        scoreable = visible and identity_ok and bbox is not None and not duplicate
        if visible:
            visible_frames.append(frame_idx)
        if duplicate:
            duplicate_frames.append(frame_idx)
        if visible and not identity_ok:
            identity_failure_frames.append(frame_idx)
        if not scoreable:
            missing_frames.append(frame_idx)
            centers.append(None)
            error = missing_center_error_penalty
            box_iou = 0.0
        else:
            scoreable_frames.append(frame_idx)
            assert bbox is not None
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            target_cx = (target_px[0] + target_px[2]) / 2
            target_cy = (target_px[1] + target_px[3]) / 2
            error = math.hypot(cx - target_cx, cy - target_cy) / diagonal
            box_iou = _iou(bbox, target_px)
            centers.append((cx, cy))
        per_frame_error.append(error)
        per_frame_iou.append(box_iou)
        records.append({
            "frame_index": frame_idx,
            "object_visible": visible,
            "instance_count": instance_count,
            "identity_consistent": identity_ok,
            "hard_fail": duplicate,
            "hard_fail_reasons": ["duplicate_instances"] if duplicate else [],
            "scoreable": scoreable,
            "center_error_frame_diagonals": error,
            "bbox_iou": box_iou,
        })

    valid_errors = [per_frame_error[i] for i in scoreable_frames]
    adjacent_dx: list[float] = []
    for idx in range(frame_count - 1):
        if centers[idx] is not None and centers[idx + 1] is not None:
            adjacent_dx.append(centers[idx + 1][0] - centers[idx][0])
    target_start_cx = (target_boxes_xyxy_normalized[0][0] + target_boxes_xyxy_normalized[0][2]) / 2
    target_end_cx = (target_boxes_xyxy_normalized[-1][0] + target_boxes_xyxy_normalized[-1][2]) / 2
    target_direction_sign = 1 if target_end_cx > target_start_cx else -1 if target_end_cx < target_start_cx else 0
    monotonic_fraction = (
        sum(dx * target_direction_sign > 0 for dx in adjacent_dx) / len(adjacent_dx)
        if adjacent_dx and target_direction_sign else None
    )
    endpoint_dx_px = (
        centers[-1][0] - centers[0][0]
        if centers[0] is not None and centers[-1] is not None
        else None
    )
    endpoint_delta_diag = endpoint_dx_px / diagonal if endpoint_dx_px is not None else None
    target_aligned_progress = (
        endpoint_delta_diag * target_direction_sign
        if endpoint_delta_diag is not None and target_direction_sign
        else None
    )

    return {
        "scorer": "blinded_manual_boxes_v2",
        "frame_count": frame_count,
        "frame_size_wh": [width, height],
        "frame_diagonal_px": diagonal,
        "missing_center_error_penalty_frame_diagonals": missing_center_error_penalty,
        "hard_fail": bool(duplicate_frames),
        "hard_fail_reasons": ["duplicate_instances"] if duplicate_frames else [],
        "visible_frame_coverage": len(visible_frames) / frame_count,
        "scoreable_identity_coverage": len(scoreable_frames) / frame_count,
        "mean_center_error_frame_diagonals_valid_only": statistics.mean(valid_errors) if valid_errors else None,
        "mean_center_error_frame_diagonals_missing_penalized": statistics.mean(per_frame_error),
        "fde_frame_diagonals_missing_penalized": per_frame_error[-1],
        "mean_bbox_iou_all_frames_missing_as_zero": statistics.mean(per_frame_iou),
        "target_direction_sign_x": target_direction_sign,
        "endpoint_delta_x_frame_diagonals": endpoint_delta_diag,
        "target_aligned_endpoint_progress_frame_diagonals": target_aligned_progress,
        "center_x_monotonic_step_fraction_target_direction": monotonic_fraction,
        "center_x_monotonic_valid_adjacent_pair_count": len(adjacent_dx),
        "missing_or_identity_ambiguous_frames": missing_frames,
        "identity_failure_frames": identity_failure_frames,
        "duplicate_instance_frames": duplicate_frames,
        "duplicate_frame_rate": len(duplicate_frames) / frame_count,
        "records": records,
    }


def make_blinded_annotation_package(
    arm_frame_dirs: Mapping[str, Path], output_dir: Path, *, frame_count: int
) -> tuple[Path, Path]:
    """Make an annotation-only ZIP and keep its arm key outside that ZIP."""
    if not arm_frame_dirs or frame_count < 1:
        raise ValueError("At least one arm and a positive frame_count are required")
    if len(set(arm_frame_dirs)) != len(arm_frame_dirs):
        raise ValueError("Arm names must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_root = output_dir / "annotation_package"
    archive_path = output_dir / "blinded_annotation_package.zip"
    key_path = output_dir / "PRIVATE_UNBLIND_KEY.json"
    if package_root.exists() or archive_path.exists() or key_path.exists():
        raise FileExistsError(f"Refusing to overwrite an annotation output under {output_dir}")
    package_root.mkdir()

    blind_ids: dict[str, str] = {}
    for arm_name in arm_frame_dirs:
        while True:
            blind_id = f"clip_{secrets.token_hex(5)}"
            if blind_id not in blind_ids.values():
                blind_ids[arm_name] = blind_id
                break

    with (package_root / "annotation_template.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(ANNOTATION_COLUMNS)
        for arm_name, frame_dir in arm_frame_dirs.items():
            paths = sorted(Path(frame_dir).glob("*.png"), key=frame_number)
            numbers = [frame_number(path) for path in paths]
            if numbers != list(range(frame_count)):
                raise ValueError(f"{arm_name}: expected contiguous frames 0..{frame_count - 1}, got {numbers}")
            blind_id = blind_ids[arm_name]
            destination = package_root / blind_id
            destination.mkdir()
            for idx, source in enumerate(paths):
                # Re-encode to strip incidental PNG metadata that could reveal a source path.
                with Image.open(source) as image:
                    image.convert("RGB").save(destination / f"frame_{idx:03d}.png")
                writer.writerow([blind_id, idx, f"{blind_id}/frame_{idx:03d}.png", "", "", "", "", "", "", "", ""])

    (package_root / "README.txt").write_text(
        "Blinded per-frame annotation instructions\n"
        "For each frame, record whether the target object is visible, the number of visible instances, "
        "whether identity matches the same object as frame 0, and a tight pixel-edge bounding box. "
        "Coordinates are continuous image-edge coordinates in [0,width] x [0,height]; x2/y2 are the "
        "right/bottom edges. The scored object center is the midpoint of this box. If invisible, use "
        "instance_count=0 and leave all four box coordinates blank; if visible but identity is ambiguous, "
        "set identity_consistent=false. Annotate without consulting any target path or method labels. "
        "Record occlusion, blur, or shape concerns in notes.\n",
        encoding="utf-8",
    )
    key_path.write_text(json.dumps({blind_ids[arm]: arm for arm in arm_frame_dirs}, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_root))
    return archive_path, key_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations_csv", type=Path)
    parser.add_argument("trajectory_json", type=Path)
    parser.add_argument("--blind-clip-id", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.trajectory_json.read_text(encoding="utf-8"))
    frame_count = int(config.get("num_frames", config.get("num_video_frames", 0)))
    anchors = config.get("target_boxes_yxyx", [])
    target_boxes = interpolate_target_boxes(anchors, frame_count)
    rows = read_annotation_csv(args.annotations_csv, args.blind_clip_id)
    report = evaluate_annotations(
        rows,
        frame_count=frame_count,
        width=args.width,
        height=args.height,
        target_boxes_xyxy_normalized=target_boxes,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
