#!/usr/bin/env python3
"""Known-truth diagnostic of a fixed start-anchored SAM2/red-observer fusion.

This only reuses saved observer predictions from two simple Kubric paths and
four nuisance composites. It does not run a model, fit thresholds, or validate
human labels. Ground truth is used only to score the already-frozen rule.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM_REPORT = ROOT / "research/sam2_visibility_stress_v1/report.json"
GT_PATH = ROOT / "research/kubric_gate0_fixture_v1/ground_truth.json"
OUTPUT = ROOT / "research/sam2_visibility_stress_v1/fusion_diagnostic_v1.json"
ANCHORS = (0, 4, 8, 12, 16)
GRID_PER_PIXEL = (52.0 / 160.0, 30.0 / 96.0)
REGISTERED_RED_STABILITY_MIN = 0.65
REGISTERED_RESIDUAL_AGREEMENT_MAX_CELLS = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def vec_sub(a, b):
    return [float(a[0]) - float(b[0]), float(a[1]) - float(b[1])]


def anchored_error_at(observed_px, truth_px, frame_index):
    """Observer start-relative displacement error in latent-grid cells."""
    observed_delta = vec_sub(observed_px[frame_index], observed_px[0])
    truth_delta = vec_sub(truth_px[frame_index], truth_px[0])
    return [(observed_delta[i] - truth_delta[i]) * GRID_PER_PIXEL[i]
            for i in range(2)]


def metrics(errors):
    return {
        "mean_absolute_xy_cells": [
            sum(abs(e[i]) for e in errors) / len(errors) for i in range(2)
        ],
        "mean_radial_error_cells": sum(
            (e[0] ** 2 + e[1] ** 2) ** 0.5 for e in errors
        ) / len(errors),
        "max_absolute_axis_error_cells": [
            max(abs(e[i]) for e in errors) for i in range(2)
        ],
    }


def main():
    started = time.perf_counter()
    report = json.loads(SAM_REPORT.read_text())
    truth = json.loads(GT_PATH.read_text())
    fixture_wh = report["fixture"]["resolution_wh"]
    width, height = float(fixture_wh[0]), float(fixture_wh[1])
    result_cases = []

    for case in report["cases"]:
        name = case["name"]
        direction = "left" if name.startswith("left_") else "right"
        gt_rows = truth[direction]
        sam_rows = case["records"]
        red_rows = case["red_proxy"]["framewise"]
        assert len(gt_rows) == len(sam_rows) == len(red_rows) == 17

        gt_px = [
            [float(r["projected_target_center_xy_normalized"][0]) * width,
             float(r["projected_target_center_xy_normalized"][1]) * height]
            for r in gt_rows
        ]
        sam_px = [list(map(float, r["pred_bbox_center_xy_px"])) for r in sam_rows]
        red_px = [
            [float(r["center_xy_normalized"][0]) * width,
             float(r["center_xy_normalized"][1]) * height]
            for r in red_rows
        ]
        sam_residuals = {i: anchored_error_at(sam_px, gt_px, i)
                         for i in ANCHORS}
        red_residuals = {i: anchored_error_at(red_px, gt_px, i)
                         for i in ANCHORS}

        details = []
        sam_errors, red_errors = [], []
        fused_errors, fusion_used_errors = [], []
        eligible_count = accepted_count = fused_count = 0
        for i in ANCHORS[1:]:
            s_err = sam_residuals[i]
            r_err = red_residuals[i]
            sam_errors.append(s_err)
            red_errors.append(r_err)
            start_ok = (bool(red_rows[0]["scoreable"])
                        and float(red_rows[0]["stability_heuristic"])
                        >= REGISTERED_RED_STABILITY_MIN)
            anchor_ok = (bool(red_rows[i]["scoreable"])
                         and float(red_rows[i]["stability_heuristic"])
                         >= REGISTERED_RED_STABILITY_MIN)
            red_eligible = start_ok and anchor_ok
            eligible_count += int(red_eligible)
            disagreement = ((s_err[0] - r_err[0]) ** 2
                            + (s_err[1] - r_err[1]) ** 2) ** 0.5
            accept = (red_eligible and disagreement
                      <= REGISTERED_RESIDUAL_AGREEMENT_MAX_CELLS)
            accepted_count += int(accept)
            if accept:
                candidate = [(s_err[j] + r_err[j]) / 2.0 for j in range(2)]
                fused_count += 1
                fusion_used_errors.append(candidate)
            else:
                candidate = s_err
            fused_errors.append(candidate)
            details.append({
                "frame_index": i,
                "sam2_anchored_error_cells_xy": s_err,
                "red_anchored_error_cells_xy": r_err,
                "red_eligible_from_saved_scoreability_and_stability": red_eligible,
                "observer_disagreement_cells_l2": disagreement,
                "fusion_accepted": accept,
                "selected_error_cells_xy": candidate,
            })

        result_cases.append({
            "case": name,
            "n_future_controller_anchors": len(ANCHORS) - 1,
            "red_eligible_count": eligible_count,
            "agreement_accepted_count": accepted_count,
            "fallback_to_sam2_count": len(ANCHORS) - 1 - fused_count,
            "sam2_error": metrics(sam_errors),
            "red_error_diagnostic_only": metrics(red_errors),
            "disagreement_gated_fusion_error": metrics(fused_errors),
            "accepted_fusion_only_error": metrics(fusion_used_errors) if fusion_used_errors else None,
            "anchors": details,
        })

    result = {
        "experiment_id": "EXP-SAM2-RED-START-ANCHORED-FUSION-DIAGNOSTIC-001",
        "status": "completed_cpu_known_truth_development_diagnostic_only",
        "inputs": {
            "sam2_visibility_stress_report": str(SAM_REPORT.relative_to(ROOT)),
            "sam2_visibility_stress_report_sha256": sha256(SAM_REPORT),
            "kubric_ground_truth": str(GT_PATH.relative_to(ROOT)),
            "kubric_ground_truth_sha256": sha256(GT_PATH),
        },
        "protocol": {
            "anchor_frame_indices": list(ANCHORS),
            "score_future_anchors_only": True,
            "fixed_grid_cells_per_pixel_xy": list(GRID_PER_PIXEL),
            "start_anchored_error_definition": "((observer_i-observer_0)-(truth_i-truth_0)) * [52/160,30/96]",
            "red_eligibility": "saved scoreable flag and saved registered stability >= 0.65 at both frame 0 and the current anchor",
            "fusion_rule": "average the SAM2 and red start-relative residuals only if their L2 disagreement <= 0.5 latent-grid cell; otherwise use SAM2 residual",
            "threshold_origin": "The red stability limit 0.65 is the existing in-sample heuristic. The 0.5-cell residual-agreement limit was fixed for this diagnostic before scoring and was not fit on these outcomes; it is not a previously registered or production threshold.",
        },
        "cases": result_cases,
        "claim_boundary": [
            "Six cases are only two correlated paths crossed with two hand-designed nuisance factors.",
            "Known synthetic geometry scores this frozen rule; it is not an independent human observer validation.",
            "SAM2 is a noisy proposal, not ground truth. Its systematic vertical localization bias remains.",
            "No generated video, diffusion inference, actuator, policy efficacy, generalization, or human-label reliability was tested.",
            "Framewise anchors within each clip are correlated; counts are descriptive only.",
        ],
        "runtime": {
            "device": platform.platform(),
            "python": sys.version.split()[0],
            "compute": "CPU-only Python JSON arithmetic",
            "elapsed_seconds": None,
            "peak_vram_bytes": None,
        },
        "artifacts": {
            "analyzer": str(Path(__file__).resolve().relative_to(ROOT)),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
        },
    }
    result["runtime"]["elapsed_seconds"] = time.perf_counter() - started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": sha256(OUTPUT),
        "elapsed_seconds": result["runtime"]["elapsed_seconds"],
        "cases": [{
            "case": c["case"],
            "sam2_mean_l2": round(c["sam2_error"]["mean_radial_error_cells"], 4),
            "gated_mean_l2": round(c["disagreement_gated_fusion_error"]["mean_radial_error_cells"], 4),
            "fusions": c["agreement_accepted_count"],
            "eligible": c["red_eligible_count"],
        } for c in result_cases],
    }, indent=2))


if __name__ == "__main__":
    main()
