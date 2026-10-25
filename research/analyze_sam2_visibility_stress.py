#!/usr/bin/env python3
"""Score predeclared red-observer anchors on the synthetic SAM2 stress run.

This is a deterministic post-hoc diagnostic using only the fixture truth and
the already saved per-frame observer outputs. It does not rerun SAM 2 or
change the registered observer threshold.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research/sam2_visibility_stress_v1/report.json"
TRUTH = ROOT / "research/kubric_gate0_fixture_v1/ground_truth.json"
OUT = ROOT / "research/sam2_visibility_stress_v1/controller_anchor_analysis_v1.json"
CUTOFF = 0.65
CONTROL_FRAMES = (0, 4, 8, 12, 16)
GRID_W, GRID_H = 52, 30
FRAME_W, FRAME_H = 160, 96


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads(RUN.read_text())
    truth = json.loads(TRUTH.read_text())
    cases = []
    for case in report["cases"]:
        direction = case["disturbance"]["direction"]
        observations = case["red_proxy"]["framewise"]
        targets = truth[direction]
        start = observations[0]
        start_center = start["center_xy_normalized"]
        start_target = targets[0]["projected_target_center_xy_normalized"]
        start_trusted = (
            start_center is not None
            and start["stability_heuristic"] >= CUTOFF
        )
        rows = []
        for index in CONTROL_FRAMES:
            observation = observations[index]
            target = targets[index]["projected_target_center_xy_normalized"]
            center = observation["center_xy_normalized"]
            scoreable = center is not None
            trusted = scoreable and observation["stability_heuristic"] >= CUTOFF
            raw_error = None
            anchor_relative_error = None
            if scoreable:
                raw_error = [
                    (center[0] - target[0]) * GRID_W,
                    (center[1] - target[1]) * GRID_H,
                ]
                if start_trusted:
                    anchor_relative_error = [
                        ((center[0] - target[0]) - (start_center[0] - start_target[0])) * GRID_W,
                        ((center[1] - target[1]) - (start_center[1] - start_target[1])) * GRID_H,
                    ]
            rows.append({
                "frame_index": index,
                "scoreable": scoreable,
                "stability_heuristic": observation["stability_heuristic"],
                "trusted_by_current_in_sample_cutoff": bool(trusted),
                "raw_target_center_error_control_grid_xy": raw_error,
                "after_start_anchor_error_control_grid_xy": anchor_relative_error,
                "accepted_residual_exceeds_half_cell_on_either_axis": (
                    bool(trusted and anchor_relative_error is not None and
                         max(abs(value) for value in anchor_relative_error) > 0.5)
                ),
            })
        accepted = [row for row in rows if row["trusted_by_current_in_sample_cutoff"]]
        trusted_anchor = start_trusted
        cases.append({
            "case": case["name"],
            "observer": "full-frame nine-threshold red-centroid ensemble",
            "stability_cutoff": CUTOFF,
            "cutoff_status": "in-sample development heuristic; not a calibrated confidence or held-out threshold",
            "start_anchor_trusted": bool(trusted_anchor),
            "accepted_control_anchor_count": len(accepted),
            "accepted_future_anchor_count": sum(row["frame_index"] != 0 for row in accepted),
            "accepted_future_anchors_over_half_cell_after_start_calibration": sum(
                row["accepted_residual_exceeds_half_cell_on_either_axis"] for row in accepted if row["frame_index"] != 0
            ),
            "controller_anchor_rows": rows,
            "interpretation": (
                "The current gate rejects this case at the first anchor and should fail closed."
                if not trusted_anchor else
                "The first anchor is trusted; inspect whether later trusted residuals exceed the half-cell observer tolerance."
            ),
        })

    analysis = {
        "experiment_id": "EXP-SAM2-VISIBILITY-DISTRACTOR-STRESS-001",
        "status": "completed_deterministic_anchor_error_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_report": str(RUN.relative_to(ROOT)),
        "input_report_sha256": sha256(RUN),
        "truth_path": str(TRUTH.relative_to(ROOT)),
        "truth_sha256": sha256(TRUTH),
        "control_grid_wh": [GRID_W, GRID_H],
        "frame_wh": [FRAME_W, FRAME_H],
        "control_frames": list(CONTROL_FRAMES),
        "cases": cases,
        "decision": (
            "A high threshold-agreement score is not sufficient to certify path-state accuracy. The same-color distractor is rejected by the existing cutoff in this fixture, but dynamic pole occlusion can leave trusted anchors with >0.5-cell relative error."
        ),
        "limitations": [
            "Two correlated synthetic mirrored paths only; disturbance geometry and cutoff were not selected on independent held-out videos.",
            "The 0.65 cutoff is evaluated in-sample and this diagnostic does not calibrate it.",
            "No human annotation, generated-video measurement, or policy efficacy result is included.",
        ],
    }
    OUT.write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUT.relative_to(ROOT)),
        "cases": [{
            "case": row["case"],
            "start_anchor_trusted": row["start_anchor_trusted"],
            "accepted_control_anchor_count": row["accepted_control_anchor_count"],
            "accepted_future_anchors_over_half_cell_after_start_calibration": row[
                "accepted_future_anchors_over_half_cell_after_start_calibration"
            ],
        } for row in cases],
    }, indent=2))


if __name__ == "__main__":
    main()
