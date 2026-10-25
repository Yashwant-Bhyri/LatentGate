#!/usr/bin/env python3
"""Check whether current red-observer features distinguish visibility errors.

This CPU analysis reuses the saved SAM2 stress composites, clean Kubric RGB,
fixture projection truth, and per-frame red-observer outputs. It does not tune
or change the registered feedback cutoff. The clean two-path feature envelope
and small global-brightness variants are descriptive development diagnostics,
not a held-out threshold-selection set.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

try:
    from .ditraj_feedback_adapter import observe_red_frame
except ImportError:
    from ditraj_feedback_adapter import observe_red_frame


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research/kubric_gate0_fixture_v1"
STRESS = ROOT / "research/sam2_visibility_stress_v1/report.json"
ANCHORS = (0, 4, 8, 12, 16)
GRID_W, GRID_H = 52, 30
FRAME_W, FRAME_H = 160, 96
STABILITY_CUTOFF = 0.65
BRIGHTNESS_FACTORS = (0.8, 0.9, 1.1, 1.2)
OUT = ROOT / "research/sam2_visibility_stress_v1/red_feature_discriminability_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path: Path) -> dict:
    image = Image.open(path).convert("RGB")
    row = observe_red_frame(np.asarray(image, dtype=np.float32) / 255.0)
    return row


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    stress = json.loads(STRESS.read_text())
    truth = json.loads((FIXTURE / "ground_truth.json").read_text())

    clean: dict[str, dict[int, dict]] = {direction: {} for direction in ("left", "right")}
    clean_values: list[float] = []
    clean_stabilities: list[float] = []
    for direction in clean:
        paths = sorted((FIXTURE / direction / "frames").glob("frame_*.png"))
        for index in ANCHORS:
            row = measure(paths[index])
            clean[direction][index] = row
            if row["area_cv_across_thresholds"] is not None:
                clean_values.append(float(row["area_cv_across_thresholds"]))
            clean_stabilities.append(float(row["stability_heuristic"]))
    envelope = [min(clean_values), max(clean_values)]

    risky_occlusion_anchors = []
    stress_rows = []
    all_stress_area: list[float] = []
    all_stress_stability: list[float] = []
    for case in stress["cases"]:
        direction = case["disturbance"]["direction"]
        rows = case["red_proxy"]["framewise"]
        start = rows[0]
        start_center = start["center_xy_normalized"]
        start_target = truth[direction][0]["projected_target_center_xy_normalized"]
        start_trusted = (
            start_center is not None
            and float(start["stability_heuristic"]) >= STABILITY_CUTOFF
        )
        anchor_rows = []
        for index in ANCHORS:
            observation = rows[index]
            target = truth[direction][index]["projected_target_center_xy_normalized"]
            center = observation["center_xy_normalized"]
            stability = float(observation["stability_heuristic"])
            area_cv = observation["area_cv_across_thresholds"]
            if area_cv is not None:
                all_stress_area.append(float(area_cv))
            all_stress_stability.append(stability)
            scoreable = center is not None
            trusted = scoreable and stability >= STABILITY_CUTOFF
            relative_error = None
            if scoreable and start_trusted:
                relative_error = [
                    ((center[0] - target[0]) - (start_center[0] - start_target[0])) * GRID_W,
                    ((center[1] - target[1]) - (start_center[1] - start_target[1])) * GRID_H,
                ]
            risky = bool(
                trusted and index != 0 and relative_error is not None
                and max(abs(value) for value in relative_error) > 0.5
            )
            outside_envelope = bool(
                area_cv is not None and (area_cv < envelope[0] or area_cv > envelope[1])
            )
            row = {
                "frame_index": index,
                "stability": stability,
                "area_cv_across_thresholds": area_cv,
                "trusted_by_registered_stability_cutoff": bool(trusted),
                "start_calibrated_path_error_control_grid_xy": relative_error,
                "accepted_error_over_half_cell": risky,
                "area_cv_outside_clean_anchor_envelope": outside_envelope,
            }
            anchor_rows.append(row)
            if risky:
                risky_occlusion_anchors.append({"case": case["name"], **row})
        stress_rows.append({
            "case": case["name"],
            "start_anchor_trusted": bool(start_trusted),
            "controller_anchors": anchor_rows,
        })

    brightness_rows = []
    for factor in BRIGHTNESS_FACTORS:
        values: list[float] = []
        scoreable = 0
        trusted_count = 0
        out_count = 0
        total = 0
        for direction in ("left", "right"):
            paths = sorted((FIXTURE / direction / "frames").glob("frame_*.png"))
            for index in ANCHORS:
                with Image.open(paths[index]) as source:
                    shifted = ImageEnhance.Brightness(source.convert("RGB")).enhance(factor)
                    row = observe_red_frame(np.asarray(shifted, dtype=np.float32) / 255.0)
                total += 1
                scoreable += int(row["scoreable"])
                trusted_count += int(row["scoreable"] and row["stability_heuristic"] >= STABILITY_CUTOFF)
                value = row["area_cv_across_thresholds"]
                if value is not None:
                    values.append(float(value))
                    out_count += int(value < envelope[0] or value > envelope[1])
        brightness_rows.append({
            "global_brightness_factor": factor,
            "control_anchor_scoreability_fraction": scoreable / total,
            "anchors_above_registered_stability_cutoff": trusted_count,
            "anchors": total,
            "area_cv_median": statistics.median(values) if values else None,
            "area_cv_outside_original_clean_envelope": out_count,
            "area_cv_measured_count": len(values),
        })

    clean_area_range = [float(value) for value in envelope]
    clean_stability_range = [float(min(clean_stabilities)), float(max(clean_stabilities))]
    risky_stability = [float(row["stability"]) for row in risky_occlusion_anchors]
    risky_area = [float(row["area_cv_across_thresholds"]) for row in risky_occlusion_anchors]
    analysis = {
        "experiment_id": "EXP-RED-OBSERVER-FEATURE-DISCRIMINABILITY-001",
        "status": "completed_cpu_diagnostic_not_threshold_validation",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "wall_seconds": time.perf_counter() - start_time,
            "device": f"{platform.system()} {platform.machine()} CPU",
            "python": sys.version.split()[0],
            "gpu_used": False,
            "peak_vram_bytes": None,
            "peak_vram_note": "CPU-only deterministic scoring; no GPU memory allocated.",
        },
        "inputs": {
            "stress_report": str(STRESS.relative_to(ROOT)),
            "stress_report_sha256": sha256(STRESS),
            "fixture_truth": "research/kubric_gate0_fixture_v1/ground_truth.json",
            "fixture_truth_sha256": sha256(FIXTURE / "ground_truth.json"),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
        },
        "geometry": {
            "frame_wh": [FRAME_W, FRAME_H],
            "control_grid_wh": [GRID_W, GRID_H],
            "controller_anchor_frames": list(ANCHORS),
        },
        "registered_red_stability_cutoff": STABILITY_CUTOFF,
        "clean_two_path_development_anchor_ranges": {
            "area_cv_across_thresholds": clean_area_range,
            "stability_heuristic": clean_stability_range,
            "anchor_count": len(clean_values),
            "all_clean_anchors_scoreable": len(clean_values) == len(clean_stabilities),
        },
        "accepted_occlusion_anchors_with_over_half_cell_relative_path_error": risky_occlusion_anchors,
        "feature_ranges_on_these_false_accepts": {
            "stability_heuristic": [min(risky_stability), max(risky_stability)] if risky_stability else None,
            "area_cv_across_thresholds": [min(risky_area), max(risky_area)] if risky_area else None,
            "area_cv_outside_clean_envelope_count": sum(
                bool(row["area_cv_outside_clean_anchor_envelope"]) for row in risky_occlusion_anchors
            ),
            "false_accept_count": len(risky_occlusion_anchors),
        },
        "area_cv_brightness_control": {
            "factors": list(BRIGHTNESS_FACTORS),
            "original_clean_envelope_used_without_retuning": clean_area_range,
            "results": brightness_rows,
        },
        "all_stress_control_anchor_rows": stress_rows,
        "decision": (
            "The registered stability score overlaps accepted occlusion failures and clean anchors. An in-sample area-CV envelope is only a candidate diagnostic: it was derived from two simple paths, and must not be deployed or used to tune the feedback gate."
        ),
        "next_falsifiable_step": (
            "Before any observer-controlled DiTraj run, freeze a candidate visibility gate on separate development renders, then test it on independently rendered paths with occlusion and photometric variation. Require it to reject every >0.5-cell false-accept anchor while retaining at least 90% of clean eligible anchors; report coverage, path error, visibility/identity, background stability, and apparent size separately."
        ),
        "limitations": [
            "The clean feature envelope comes from the same two correlated fixture paths, not independent videos.",
            "Brightness transforms are simple global pixel changes and do not represent generated-video appearance variation.",
            "No human labels, generated-video observer accuracy, model inference, or policy efficacy is established.",
        ],
    }
    OUT.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUT.relative_to(ROOT)),
        "clean_area_cv_range": clean_area_range,
        "clean_stability_range": clean_stability_range,
        "false_accepts": len(risky_occlusion_anchors),
        "false_accept_area_cv": risky_area,
        "brightness_controls": brightness_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
