#!/usr/bin/env python3
"""Measure a red object's image-space trajectory from a directory of PNG frames.

This is a deliberately narrow screening evaluator for the LatentGate red-ball
pilot. It reuses the M0 audit's RGB threshold so existing and future clips can
be compared under one measurement rule. It is not a general object tracker or
a ground-truth 3D geometry evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.png$)", path.name)
    if not match:
        raise ValueError(f"Frame filename has no trailing number: {path.name}")
    return int(match.group(1))


def interpolate_boxes(anchors: list[list[float]], n_frames: int) -> list[list[float]]:
    """Interpolate [frame, y1, y2, x1, x2] anchors to normalized xyxy boxes."""
    if not anchors:
        return []
    points = sorted(anchors, key=lambda row: row[0])
    if points[0][0] < 0 or points[-1][0] >= n_frames:
        raise ValueError("Trajectory anchor frame lies outside the supplied clip")

    out: list[list[float]] = []
    for idx in range(n_frames):
        if idx <= points[0][0]:
            row = points[0]
            vals = row[1:]
        elif idx >= points[-1][0]:
            row = points[-1]
            vals = row[1:]
        else:
            left = max(i for i, row in enumerate(points) if row[0] <= idx)
            a, b = points[left], points[left + 1]
            alpha = (idx - a[0]) / (b[0] - a[0])
            vals = [a[j] + alpha * (b[j] - a[j]) for j in range(1, 5)]
        y1, y2, x1, x2 = vals
        out.append([x1, y1, x2, y2])
    return out


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "population_std": None, "population_cv": None}
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return {
        "mean": mean,
        "population_std": std,
        "population_cv": std / mean if mean else None,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.frames)
    paths = sorted(root.glob("*.png"), key=frame_number)
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {root}")

    cfg = json.loads(Path(args.trajectory).read_text()) if args.trajectory else {}
    mask_cfg = cfg.get("mask_proxy", {})
    r_gt = int(mask_cfg.get("r_gt", args.r_gt))
    g_lt = int(mask_cfg.get("g_lt", args.g_lt))
    b_lt = int(mask_cfg.get("b_lt", args.b_lt))
    anchors = cfg.get("target_boxes_yxyx", [])
    target_boxes = interpolate_boxes(anchors, len(paths)) if anchors else []

    records: list[dict[str, Any]] = []
    width = height = None
    for idx, path in enumerate(paths):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        if width is None:
            width, height = w, h
        elif (w, h) != (width, height):
            raise ValueError(f"Inconsistent frame size at {path}: {(w, h)} != {(width, height)}")

        mask = (rgb[:, :, 0] > r_gt) & (rgb[:, :, 1] < g_lt) & (rgb[:, :, 2] < b_lt)
        ys, xs = np.nonzero(mask)
        detected = None
        if len(xs) >= args.min_pixels:
            detected = [
                float(xs.min()), float(ys.min()),
                float(xs.max() + 1), float(ys.max() + 1),
            ]
            cx, cy = float(xs.mean()), float(ys.mean())
            area = int(len(xs))
        else:
            cx = cy = None
            area = int(len(xs))

        target = None
        if target_boxes:
            x1, y1, x2, y2 = target_boxes[idx]
            target = [x1 * w, y1 * h, x2 * w, y2 * h]
        center_error = None
        iou = None
        if detected is not None and target is not None:
            d_cx = (detected[0] + detected[2]) / 2
            d_cy = (detected[1] + detected[3]) / 2
            t_cx = (target[0] + target[2]) / 2
            t_cy = (target[1] + target[3]) / 2
            center_error = math.hypot(d_cx - t_cx, d_cy - t_cy)
            iou = bbox_iou(detected, target)

        records.append({
            "frame": frame_number(path),
            "file": path.name,
            "mask_pixels": area,
            "mask_center_xy_px": [cx, cy] if cx is not None else None,
            "detected_bbox_xyxy_px": detected,
            "target_bbox_xyxy_px": target,
            "center_error_px": center_error,
            "bbox_iou": iou,
        })

    valid = [r for r in records if r["mask_center_xy_px"] is not None]
    centers = [r["mask_center_xy_px"] for r in valid]
    areas = [float(r["mask_pixels"]) for r in valid]
    widths = [r["detected_bbox_xyxy_px"][2] - r["detected_bbox_xyxy_px"][0] for r in valid]
    heights = [r["detected_bbox_xyxy_px"][3] - r["detected_bbox_xyxy_px"][1] for r in valid]
    errors = [r["center_error_px"] for r in valid if r["center_error_px"] is not None]
    ious = [r["bbox_iou"] for r in valid if r["bbox_iou"] is not None]
    diag = math.hypot(width or 0, height or 0)
    dx = [centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)]

    return {
        "evaluator": "red_rgb_threshold_image_space_screening_proxy",
        "frames_dir": str(root),
        "trajectory_config": str(Path(args.trajectory)) if args.trajectory else None,
        "frame_count": len(paths),
        "frame_size_wh": [width, height],
        "mask_threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
        "minimum_mask_pixels": args.min_pixels,
        "valid_frame_coverage": len(valid) / len(paths),
        "mean_center_error_px_on_valid_frames": statistics.mean(errors) if errors else None,
        "mean_center_error_frame_diagonals_on_valid_frames": statistics.mean(errors) / diag if errors and diag else None,
        "mean_bbox_iou_on_valid_frames": statistics.mean(ious) if ious else None,
        "red_mask_area_pixels": summarize(areas),
        "detected_bbox_width_px": summarize(widths),
        "detected_bbox_height_px": summarize(heights),
        "center_x_monotonic_step_fraction": sum(v > 0 for v in dx) / len(dx) if dx else None,
        "records": records,
        "limitations": [
            "The RGB threshold is a color-specific proxy and can include reflections or other red pixels.",
            "All reported motion and size measurements are image-space only.",
            "This evaluator does not verify rigid 3D motion, physical contact, or metric scale.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", help="Directory containing frame_000.png, frame_001.png, ...")
    parser.add_argument("--trajectory", help="JSON config with target_boxes_yxyx anchors")
    parser.add_argument("--output", help="Write JSON report to this path; stdout if omitted")
    parser.add_argument("--min-pixels", type=int, default=300)
    parser.add_argument("--r-gt", type=int, default=150)
    parser.add_argument("--g-lt", type=int, default=125)
    parser.add_argument("--b-lt", type=int, default=125)
    args = parser.parse_args()
    report = evaluate(args)
    payload = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()
