#!/usr/bin/env python3
"""Validate the Gate 1 known-truth scorer with exact and shared-bias fixtures.

Both input CSV pairs are generated from the local Kubric fixture in a temporary
directory. They are synthetic scorer checks, never human annotation returns.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "research" / "score_gate1_labels_against_kubric_truth.py"
TRUTH_PATH = ROOT / "research" / "kubric_gate0_fixture_v1" / "ground_truth.json"
OUTPUT_PATH = ROOT / "research" / "gate1_truth_scorer_oracle_selfcheck_v3.json"
FIELDS = [
    "blind_clip_id", "frame_index", "image_file", "object_visible", "instance_count",
    "identity_consistent", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
]
SOURCE_W, SOURCE_H = 160, 96


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_oracle_csv(
    path: Path,
    key: dict[str, str],
    truth: dict,
    dx_px: float = 0.0,
    identity_uncertain_at: set[tuple[str, int]] | tuple[str, int] | None = None,
) -> None:
    if identity_uncertain_at is None:
        uncertain_frames: set[tuple[str, int]] = set()
    elif isinstance(identity_uncertain_at, tuple):
        uncertain_frames = {identity_uncertain_at}
    else:
        uncertain_frames = set(identity_uncertain_at)
    rows = []
    for blind_id, direction in sorted(key.items()):
        for row in truth[direction]:
            annotation = row["annotation_from_mask"]
            box = annotation["bbox_xyxy_px"]
            if box is None:
                coords = ["", "", "", ""]
            else:
                shifted = [float(value) for value in box]
                shifted[0] += dx_px
                shifted[2] += dx_px
                coords = [format(value, ".9g") for value in shifted]
            identity_consistent = bool(annotation["identity_consistent"])
            if (blind_id, int(row["frame_index"])) in uncertain_frames:
                identity_consistent = False
            rows.append([
                blind_id,
                str(int(row["frame_index"])),
                f"{blind_id}_frame_{int(row['frame_index']):03d}.png",
                "true" if annotation["object_visible"] else "false",
                str(int(annotation["instance_count"])),
                "true" if identity_consistent else "false",
                *coords,
            ])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def write_reference_packet(path: Path, key: dict[str, str]) -> None:
    """Create a temporary frame-key-only packet for synthetic scorer calls."""
    text = io.StringIO(newline="")
    writer = csv.writer(text)
    writer.writerow(["blind_clip_id", "frame_index", "image_file"])
    for blind_id in sorted(key):
        for frame_idx in range(17):
            writer.writerow([blind_id, frame_idx, f"{blind_id}_frame_{frame_idx:03d}.png"])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("annotation_template.csv", text.getvalue())


def rewrite_one_image_key(source: Path, destination: Path, *, clip_id: str, frame_idx: int) -> None:
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["blind_clip_id"] == clip_id and int(row["frame_index"]) == frame_idx:
                row["image_file"] = "wrong_common_frame.png"
            writer.writerow(row)


def run_scorer(
    a: Path,
    b: Path,
    output: Path,
    *,
    unblind_key: Path,
    reference_packet: Path,
    ground_truth: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable, str(SCORER), str(a), str(b),
        "--reference-packet", str(reference_packet),
        "--output", str(output),
    ]
    if ground_truth is not None:
        command.extend(["--ground-truth", str(ground_truth)])
    # Never let this harness inherit the scorer's default real packet-key path.
    command.extend(["--unblind-key", str(unblind_key)])
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def compact_case(result: dict[str, Any]) -> dict[str, Any]:
    pooled = result["rater_a"]["pooled"]
    return {
        "status": result["status"],
        "frames": pooled["frames"],
        "joint_categorical_accuracy": pooled["joint_categorical_accuracy"],
        "identity_accuracy": pooled["identity_accuracy"],
        "expected_visible_box_coverage": pooled["expected_visible_box_coverage"],
        "unique_identity_consistent_box_coverage": pooled["unique_identity_consistent_box_coverage"],
        "box_scoreable_frames": pooled["scoreable_box_frames"],
        "center_distance_vs_mask_source_px": pooled["center_distance_vs_mask_source_px"],
        "center_abs_dx_vs_mask_cells": pooled["center_abs_dx_vs_mask_cells"],
        "signed_center_bias_vs_mask_source_px": pooled["signed_center_bias_vs_mask_source_px"],
        "bbox_iou_vs_mask": pooled["bbox_iou_vs_mask"],
        "mask_bbox_center_vs_projected_center_truth_gap": result["rater_a"]["mask_bbox_center_vs_projected_center_truth_gap"],
        "identical_inputs_give_identical_known_truth_metrics": result["rater_a"] == result["rater_b"],
    }


def main() -> None:
    started = time.perf_counter()
    # Use disposable IDs and a temporary key; this measurement harness never
    # needs to read the real packet's private blind-to-direction mapping.
    key = {"oracle_clip_a": "left", "oracle_clip_b": "right"}
    truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="gate1_truth_oracle_") as tmp:
        work = Path(tmp)
        synthetic_key_path = work / "synthetic_unblind_key.json"
        synthetic_key_path.write_text(json.dumps({"synthetic": key}), encoding="utf-8")
        reference_packet = work / "synthetic_reference_packet.zip"
        write_reference_packet(reference_packet, key)
        synthetic_key_hash = sha256(synthetic_key_path)
        exact_a, exact_b = work / "exact_a.csv", work / "exact_b.csv"
        biased_a, biased_b = work / "shared_bias_a.csv", work / "shared_bias_b.csv"
        ambiguous_a, ambiguous_b = work / "identity_uncertain_a.csv", work / "identity_uncertain_b.csv"
        ambiguous_identity_truth_path = work / "truth_identity_uncertain.json"
        ambiguous_count_truth_path = work / "truth_duplicate_instance.json"
        failed_gate_a, failed_gate_b = work / "failed_gate_a.csv", work / "failed_gate_b.csv"
        miskeyed_a, miskeyed_b = work / "miskeyed_a.csv", work / "miskeyed_b.csv"
        write_oracle_csv(exact_a, key, truth)
        write_oracle_csv(exact_b, key, truth)
        first_blind_id = sorted(key)[0]
        rewrite_one_image_key(exact_a, miskeyed_a, clip_id=first_blind_id, frame_idx=0)
        rewrite_one_image_key(exact_b, miskeyed_b, clip_id=first_blind_id, frame_idx=0)
        write_oracle_csv(biased_a, key, truth, dx_px=2.0)
        write_oracle_csv(biased_b, key, truth, dx_px=2.0)
        write_oracle_csv(ambiguous_a, key, truth, identity_uncertain_at=(first_blind_id, 8))
        write_oracle_csv(ambiguous_b, key, truth, identity_uncertain_at=(first_blind_id, 8))
        write_oracle_csv(failed_gate_a, key, truth)
        write_oracle_csv(
            failed_gate_b,
            key,
            truth,
            identity_uncertain_at={(blind_id, 0) for blind_id in key},
        )

        ambiguous_identity_truth = json.loads(json.dumps(truth))
        ambiguous_identity_truth["right"][0]["annotation_from_mask"]["identity_consistent"] = False
        ambiguous_identity_truth_path.write_text(json.dumps(ambiguous_identity_truth), encoding="utf-8")
        ambiguous_count_truth = json.loads(json.dumps(truth))
        ambiguous_count_truth["left"][0]["annotation_from_mask"]["instance_count"] = 2
        ambiguous_count_truth_path.write_text(json.dumps(ambiguous_count_truth), encoding="utf-8")

        exact = run_scorer(exact_a, exact_b, work / "exact_result.json", unblind_key=synthetic_key_path, reference_packet=reference_packet)
        biased = run_scorer(biased_a, biased_b, work / "biased_result.json", unblind_key=synthetic_key_path, reference_packet=reference_packet)
        ambiguous = run_scorer(ambiguous_a, ambiguous_b, work / "ambiguous_result.json", unblind_key=synthetic_key_path, reference_packet=reference_packet)
        truth_identity_uncertain = run_scorer(
            exact_a,
            exact_b,
            work / "truth_identity_uncertain_result.json",
            ground_truth=ambiguous_identity_truth_path,
            unblind_key=synthetic_key_path,
            reference_packet=reference_packet,
        )
        truth_duplicate = run_scorer(
            exact_a,
            exact_b,
            work / "truth_duplicate_result.json",
            ground_truth=ambiguous_count_truth_path,
            unblind_key=synthetic_key_path,
            reference_packet=reference_packet,
        )
        missing_key = work / "must_not_read_unblind_key.json"
        miskeyed_process = subprocess.run(
            [
                sys.executable,
                str(SCORER),
                str(miskeyed_a),
                str(miskeyed_b),
                "--reference-packet",
                str(reference_packet),
                "--unblind-key",
                str(missing_key),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        failed_gate_process = subprocess.run(
            [
                sys.executable,
                str(SCORER),
                str(failed_gate_a),
                str(failed_gate_b),
                "--reference-packet",
                str(reference_packet),
                "--unblind-key",
                str(missing_key),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        exact_summary = compact_case(exact)
        biased_summary = compact_case(biased)
        ambiguous_summary = compact_case(ambiguous)
        truth_identity_uncertain_summary = compact_case(truth_identity_uncertain)
        truth_duplicate_summary = compact_case(truth_duplicate)

    exact_mask_error = exact_summary["center_distance_vs_mask_source_px"]
    biased_mask_error = biased_summary["center_distance_vs_mask_source_px"]
    biased_dx_cells = biased_summary["center_abs_dx_vs_mask_cells"]
    biased_x = biased_summary["signed_center_bias_vs_mask_source_px"]["dx_mean"]
    biased_iou = biased_summary["bbox_iou_vs_mask"]["mean"]
    ambiguous_metrics = ambiguous["rater_a"]["pooled"]
    truth_identity_metrics = truth_identity_uncertain["rater_a"]["pooled"]
    truth_duplicate_metrics = truth_duplicate["rater_a"]["pooled"]

    assert exact_summary["frames"] == 34
    assert exact_summary["joint_categorical_accuracy"] == 1.0
    assert exact_mask_error["mean"] == 0.0 and exact_mask_error["max"] == 0.0
    assert exact_summary["bbox_iou_vs_mask"]["mean"] == 1.0
    assert biased_mask_error["mean"] == 2.0
    assert math.isclose(biased_x, 2.0, abs_tol=1e-9)
    assert math.isclose(biased_dx_cells["p95"], 0.65, abs_tol=1e-9)
    assert math.isclose(biased_iou, 18.0 / 22.0, abs_tol=1e-9)
    assert ambiguous_summary["expected_visible_box_coverage"] == 1.0
    assert ambiguous_summary["box_scoreable_frames"] == 33
    assert ambiguous_summary["unique_identity_consistent_box_coverage"] == 33 / 34
    assert ambiguous_summary["identity_accuracy"] == 33 / 34
    assert ambiguous_metrics["center_distance_vs_mask_source_px"]["max"] == 0.0
    assert exact_summary["identical_inputs_give_identical_known_truth_metrics"]
    assert biased_summary["identical_inputs_give_identical_known_truth_metrics"]
    assert truth_identity_metrics["scoreable_box_frames"] == 33
    assert truth_duplicate_metrics["scoreable_box_frames"] == 33
    assert truth_identity_metrics["unique_identity_consistent_box_coverage"] == 33 / 34
    assert truth_duplicate_metrics["unique_identity_consistent_box_coverage"] == 33 / 34
    assert failed_gate_process.returncode != 0
    assert "blinded inter-rater screen failed; refusing to unblind" in failed_gate_process.stderr.lower()
    assert "must_not_read_unblind_key.json" not in failed_gate_process.stderr
    assert miskeyed_process.returncode != 0
    assert "does not match the archived packet key-to-image map" in miskeyed_process.stderr
    assert "must_not_read_unblind_key.json" not in miskeyed_process.stderr

    result = {
        "experiment_id": "EXP-LATENTGATE-GATE1-TRUTH-SCORER-ORACLE-SELFCHECK-001",
        "status": "synthetic_measurement_validation_only_no_human_labels",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "Construct temporary oracle CSV pairs from Kubric mask truth using disposable clip IDs, a matching synthetic packet key map, and a synthetic temporary unblind key. Score exact, shared +2 source-pixel x-bias, and rater-uncertain cases; separately mutate temporary truth records to mark one frame identity-uncertain and another duplicated, and require both to be excluded from path geometry. Also submit a pair with the same wrong filename in both CSVs and a deliberately below-threshold pair; both must be rejected before the nonexistent unblind-key path is accessed. The real packet key is never read; all temporary files are deleted on exit.",
        "coordinate_contract": {"source_wh": [SOURCE_W, SOURCE_H], "target_wh": [832, 480], "grid_wh": [52, 30]},
        "cases": {
            "exact_oracle": exact_summary,
            "shared_plus_2px_x_bias": biased_summary,
            "one_identity_uncertain_frame": ambiguous_summary,
            "ground_truth_identity_uncertain_frame": truth_identity_uncertain_summary,
            "ground_truth_duplicate_instance_frame": truth_duplicate_summary,
            "common_wrong_packet_filename_blocked_before_unblind": {
                "process_exit_code": miskeyed_process.returncode,
                "expected_packet_key_error_seen": "does not match the archived packet key-to-image map" in miskeyed_process.stderr,
                "key_read_error_seen": "must_not_read_unblind_key.json" in miskeyed_process.stderr,
            },
            "below_threshold_pair_blocked_before_unblind": {
                "process_exit_code": failed_gate_process.returncode,
                "expected_gate_error_seen": "blinded inter-rater screen failed; refusing to unblind" in failed_gate_process.stderr.lower(),
                "key_read_error_seen": "must_not_read_unblind_key.json" in failed_gate_process.stderr,
            },
        },
        "validation": {
            "exact_oracle_zero_mask_center_error": True,
            "known_shared_bias_detected_despite_identical_raters": True,
            "identity_uncertain_box_excluded_from_path_error_but_retained_in_coverage": True,
            "ground_truth_identity_uncertain_frame_excluded_from_path_error": True,
            "ground_truth_duplicate_frame_excluded_from_path_error": True,
            "common_wrong_packet_filename_refused_before_unblinding": True,
            "below_threshold_pair_refused_before_unblinding": True,
            "synthetic_temporary_key_passed_for_every_accepted_case": True,
            "real_packet_key_read": False,
            "expected_bias_in_target_grid_cells": 2.0 * 52 / SOURCE_W,
            "interpretation": "The diagnostic exposes common-mode box-center bias against known mask truth; this validates arithmetic only and says nothing about actual human labels or generated-video observer transfer.",
        },
        "inputs": {
            "scorer_sha256": sha256(SCORER),
            "synthetic_test_key_sha256": synthetic_key_hash,
            "ground_truth_sha256": sha256(TRUTH_PATH),
        },
        "compute": {
            "runtime_seconds_including_seven_scorer_processes": round(time.perf_counter() - started, 6),
            "scorer_processes": 7,
            "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
            "gpu_used": False,
            "peak_vram_mb": None,
            "model_inference": False,
        },
        "limitations": [
            "Synthetic fixture-derived inputs only; no human-return CSV was read or created outside the temporary directory.",
            "This does not pass or replace inter-rater agreement, the second mirrored-seed reliability block, generated-Wan observer validation, or any policy-efficacy gate.",
            "The injected shift is a measurement-path sanity case, not a proposed annotation threshold or an empirical rater bias estimate.",
            "The identity-uncertain case validates scoring eligibility semantics only; it is not an actual occlusion or identity-switch sample.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH.relative_to(ROOT)), "cases": result["cases"], "compute": result["compute"]}, indent=2))


if __name__ == "__main__":
    main()
