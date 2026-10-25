#!/usr/bin/env python3
"""Threshold-sensitivity audit for the saved M0 red-ball clip.

This reuses the existing RGB-threshold image-space evaluator over the same
final PNGs and their VAE round-trip controls. It is a CPU-only measurement of
how sensitive the apparent red-object drift is to mask thresholds and image
round-tripping; it does not estimate metric 3D scale or validate physical
object geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from evaluate_trajectory_frames import evaluate  # noqa: E402


FINAL_FRAMES = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/run_001/frames"
ROUNDTRIP_FRAMES = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/run_001/vae_roundtrip_frames"
R_THRESHOLDS = (130, 150, 170, 190)
GB_CEILINGS = (100, 125, 150)
MIN_PIXELS = 300
SAMPLE_FRAMES = (0, 4, 8, 12, 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_metrics(frames_dir: Path, r_gt: int, gb_lt: int) -> dict[str, Any]:
    report = evaluate(SimpleNamespace(
        frames=str(frames_dir),
        trajectory=None,
        min_pixels=MIN_PIXELS,
        r_gt=r_gt,
        g_lt=gb_lt,
        b_lt=gb_lt,
    ))
    records = report["records"]
    valid = [r for r in records if r["mask_center_xy_px"] is not None]
    areas = [r["mask_pixels"] for r in valid]
    boxes = [r["detected_bbox_xyxy_px"] for r in valid]
    widths = [b[2] - b[0] for b in boxes]
    heights = [b[3] - b[1] for b in boxes]
    centers = [r["mask_center_xy_px"] for r in valid]
    frame_diag = math.hypot(*report["frame_size_wh"])

    def cv(values: list[float]) -> float | None:
        mean = statistics.mean(values) if values else 0.0
        return statistics.pstdev(values) / mean if mean else None

    def endpoint_ratio(values: list[float]) -> float | None:
        return values[-1] / values[0] if len(values) >= 2 and values[0] else None

    dx = [centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)]
    summary = {
        "frame_count": report["frame_count"],
        "coverage": report["valid_frame_coverage"],
        "area_start_px": areas[0] if areas else None,
        "area_end_px": areas[-1] if areas else None,
        "area_endpoint_ratio": endpoint_ratio(areas),
        "area_cv": cv(areas),
        "bbox_width_start_px": widths[0] if widths else None,
        "bbox_width_end_px": widths[-1] if widths else None,
        "bbox_width_endpoint_ratio": endpoint_ratio(widths),
        "bbox_width_cv": cv(widths),
        "bbox_height_start_px": heights[0] if heights else None,
        "bbox_height_end_px": heights[-1] if heights else None,
        "bbox_height_endpoint_ratio": endpoint_ratio(heights),
        "bbox_height_cv": cv(heights),
        "center_x_monotonic_step_fraction": sum(v > 0 for v in dx) / len(dx) if dx else None,
        "center_x_total_shift_px": centers[-1][0] - centers[0][0] if len(centers) >= 2 else None,
        "center_y_total_shift_px": centers[-1][1] - centers[0][1] if len(centers) >= 2 else None,
        "center_path_start_end_distance_frame_diagonals": (
            math.hypot(centers[-1][0] - centers[0][0], centers[-1][1] - centers[0][1]) / frame_diag
            if len(centers) >= 2 and frame_diag else None
        ),
    }
    return {"summary": summary, "records": records}


def pair_metrics(final: dict[str, Any], roundtrip: dict[str, Any]) -> dict[str, Any]:
    a = final["records"]
    b = roundtrip["records"]
    if len(a) != len(b):
        raise ValueError("Final and VAE round-trip frame counts differ")
    paired = []
    for ra, rb in zip(a, b):
        if ra["frame"] != rb["frame"]:
            raise ValueError("Final and VAE round-trip frame numbers differ")
        ma, mb = ra["mask_pixels"], rb["mask_pixels"]
        ca, cb = ra["mask_center_xy_px"], rb["mask_center_xy_px"]
        ba, bb = ra["detected_bbox_xyxy_px"], rb["detected_bbox_xyxy_px"]
        if ca is not None and cb is not None and ba is not None and bb is not None:
            paired.append({
                "frame": ra["frame"],
                "area_relative_difference": abs(ma - mb) / max(ma, 1),
                "center_difference_px": math.hypot(ca[0] - cb[0], ca[1] - cb[1]),
                "bbox_iou": __import__("evaluate_trajectory_frames").bbox_iou(ba, bb),
            })
    return {
        "paired_valid_frame_coverage": len(paired) / len(a) if a else None,
        "mean_area_relative_difference": statistics.mean(r["area_relative_difference"] for r in paired) if paired else None,
        "max_area_relative_difference": max((r["area_relative_difference"] for r in paired), default=None),
        "mean_center_difference_px": statistics.mean(r["center_difference_px"] for r in paired) if paired else None,
        "max_center_difference_px": max((r["center_difference_px"] for r in paired), default=None),
        "mean_bbox_iou": statistics.mean(r["bbox_iou"] for r in paired) if paired else None,
    }


def make_overlay(final_dir: Path, out_path: Path, threshold_sets: list[tuple[int, int, int]]) -> None:
    """Create a compact visual audit at five time points for three thresholds."""
    cols, rows = len(SAMPLE_FRAMES), len(threshold_sets)
    thumb_w, thumb_h, label_h = 416, 240, 26
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (r_gt, g_lt, b_lt) in enumerate(threshold_sets):
        for col, frame_idx in enumerate(SAMPLE_FRAMES):
            path = final_dir / f"frame_{frame_idx:03d}.png"
            image = Image.open(path).convert("RGB")
            rgb = image.load()
            mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
            px = mask.load()
            for y in range(image.height):
                for x in range(image.width):
                    red, green, blue = rgb[x, y]
                    if red > r_gt and green < g_lt and blue < b_lt:
                        px[x, y] = (0, 255, 70, 105)
            overlay = Image.alpha_composite(image.convert("RGBA"), mask).convert("RGB")
            overlay.thumbnail((thumb_w, thumb_h))
            x0, y0 = col * thumb_w, row * (thumb_h + label_h)
            canvas.paste(overlay, (x0, y0 + label_h))
            if col == 0:
                draw.text((x0 + 5, y0 + 6), f"mask R>{r_gt}, G<{g_lt}, B<{b_lt}", fill="black")
            draw.text((x0 + 6, y0 + label_h + 5), f"frame {frame_idx:02d}", fill="yellow")
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "research/m0_mask_threshold_sensitivity_v1.json")
    parser.add_argument("--overlay", type=Path, default=ROOT / "m0/analysis/m0_mask_threshold_contact_sheet.png")
    args = parser.parse_args()

    if not FINAL_FRAMES.is_dir() or not ROUNDTRIP_FRAMES.is_dir():
        raise FileNotFoundError("Expected saved final and VAE round-trip frame directories")

    cases = []
    for r_gt, g_lt, b_lt in itertools.product(R_THRESHOLDS, GB_CEILINGS, GB_CEILINGS):
        final = frame_metrics(FINAL_FRAMES, r_gt, g_lt)
        roundtrip = frame_metrics(ROUNDTRIP_FRAMES, r_gt, g_lt)
        cases.append({
            "threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
            "final": final["summary"],
            "vae_roundtrip": roundtrip["summary"],
            "final_vs_vae_roundtrip_mask_sensitivity": pair_metrics(final, roundtrip),
        })

    passing = [
        row for row in cases
        if row["final"]["coverage"] == 1.0
        and row["final"]["area_endpoint_ratio"] is not None
        and row["final"]["area_endpoint_ratio"] > 1.0
        and row["final"]["center_x_monotonic_step_fraction"] == 1.0
    ]
    result = {
        "experiment_id": "EXP-M0-MASK-THRESHOLD-SENSITIVITY-001",
        "status": "measured_on_existing_artifacts_no_new_model_run",
        "evaluator": "research/evaluate_trajectory_frames.py",
        "inputs": {
            "final_frames_dir": str(FINAL_FRAMES.relative_to(ROOT)),
            "vae_roundtrip_frames_dir": str(ROUNDTRIP_FRAMES.relative_to(ROOT)),
            "frame_count": 17,
            "minimum_mask_pixels": MIN_PIXELS,
            "rgb_threshold_grid": {
                "red_greater_than": list(R_THRESHOLDS),
                "green_less_than": list(GB_CEILINGS),
                "blue_less_than": list(GB_CEILINGS),
                "total_cases_per_sequence": len(cases),
            },
            "final_frames_sha256": {
                p.name: sha256(p) for p in sorted(FINAL_FRAMES.glob("*.png"))
            },
            "vae_roundtrip_frames_sha256": {
                p.name: sha256(p) for p in sorted(ROUNDTRIP_FRAMES.glob("*.png"))
            },
        },
        "summary": {
            "total_threshold_cases": len(cases),
            "cases_with_full_coverage_positive_area_growth_and_monotonic_rightward_motion": len(passing),
            "robustness_fraction": len(passing) / len(cases),
            "baseline_threshold_case": next(
                row for row in cases
                if row["threshold"] == {"r_gt": 150, "g_lt": 125, "b_lt": 125}
            ),
            "interpretation": (
                "Describes robustness of an RGB-mask image-space proxy over a preregistered threshold grid and a VAE round-trip control. "
                "A positive result is not evidence of metric 3D scale change or physical object deformation."
            ),
        },
        "cases": cases,
        "limitations": [
            "The red RGB mask can include reflections and excludes darker parts of the ball.",
            "Endpoint apparent-size change is image-space; depth and object pose are not measured.",
            "The VAE round-trip sequence is a decoder-path control, not independent geometry ground truth.",
            "The RGB threshold-grid pass rule is descriptive and does not establish statistical significance.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    make_overlay(FINAL_FRAMES, args.overlay, [(130, 100, 100), (150, 125, 125), (190, 150, 150)])
    print(json.dumps({
        "output": str(args.output),
        "overlay": str(args.overlay),
        "threshold_cases": len(cases),
        "robustness_fraction": result["summary"]["robustness_fraction"],
        "baseline": result["summary"]["baseline_threshold_case"]["final"],
    }, indent=2))


if __name__ == "__main__":
    main()
