#!/usr/bin/env python3
"""CPU-only diagnostics for Gate 1 resolution mapping and fail-closed inputs.

This creates synthetic scorer inputs; it does not represent human annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from score_annotation_agreement import (
    DEFAULT_CENTER_COVERAGE_MIN,
    DEFAULT_CENTER_P95_CELLS_MAX,
    DEFAULT_STATE_AGREEMENT_MIN,
    load_packet_image_keys,
    load_rows,
    map_source_delta_to_grid,
    summarize,
    validate_rows_against_packet,
)


ROOT = Path(__file__).resolve().parent
SCORER = ROOT / "score_annotation_agreement.py"
DEFAULT_OUTPUT = ROOT / "gate1_scorer_tail_diagnostic_v6.json"


def build_rows() -> tuple[dict, dict]:
    rater_a: dict = {}
    rater_b: dict = {}
    for clip_id in ("mirror_left", "mirror_right"):
        for frame in range(17):
            key = (clip_id, frame)
            box = [15.0, 15.0, 25.0, 25.0]
            base = {
                "object_visible": True,
                "instance_count": 1,
                "identity_consistent": True,
                "image_file": f"{clip_id}_frame_{frame:03d}.png",
            }
            rater_a[key] = {**base, "bbox": box}
            # Two 2.8846-source-pixel disagreements map to 15 px on the
            # 832x480 target, preserving the pooled-pass/per-clip-fail probe.
            shift = (15.0 * 160 / 832) if clip_id == "mirror_left" and frame in (15, 16) else 0.0
            rater_b[key] = {
                **base,
                "bbox": [coordinate + shift if axis in (0, 2) else coordinate
                         for axis, coordinate in enumerate(box)],
            }
    return rater_a, rater_b


def build_single_outlier_rows() -> tuple[dict, dict]:
    """One 2.5-cell disagreement among 34 labels probes p95 tail sensitivity."""
    rater_a: dict = {}
    rater_b: dict = {}
    shift_source_px = 2.5 * 160 / 52
    for clip_id in ("mirror_left", "mirror_right"):
        for frame in range(17):
            key = (clip_id, frame)
            box = [15.0, 15.0, 25.0, 25.0]
            base = {
                "object_visible": True,
                "instance_count": 1,
                "identity_consistent": True,
                "image_file": f"{clip_id}_frame_{frame:03d}.png",
            }
            rater_a[key] = {**base, "bbox": box}
            shift = shift_source_px if key == ("mirror_left", 16) else 0.0
            rater_b[key] = {
                **base,
                "bbox": [coordinate + shift if axis in (0, 2) else coordinate
                         for axis, coordinate in enumerate(box)],
            }
    return rater_a, rater_b


def write_label_csv(path: Path, rows: dict) -> None:
    columns = [
        "blind_clip_id", "frame_index", "image_file", "object_visible", "instance_count",
        "identity_consistent", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for (clip_id, frame_idx), row in sorted(rows.items()):
            bbox = row["bbox"] or ["", "", "", ""]
            writer.writerow({
                "blind_clip_id": clip_id,
                "frame_index": frame_idx,
                "image_file": row["image_file"],
                "object_visible": str(row["object_visible"]).lower(),
                "instance_count": row["instance_count"],
                "identity_consistent": str(row["identity_consistent"]).lower(),
                "bbox_x1_px": bbox[0], "bbox_y1_px": bbox[1],
                "bbox_x2_px": bbox[2], "bbox_y2_px": bbox[3],
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    a, b = build_rows()
    report = summarize(
        a, b,
        source_width=160, source_height=96,
        target_width=832, target_height=480,
        grid_width=52, grid_height=30,
        state_min=DEFAULT_STATE_AGREEMENT_MIN,
        center_p95_cells_max=DEFAULT_CENTER_P95_CELLS_MAX,
        center_coverage_min=DEFAULT_CENTER_COVERAGE_MIN,
        expected_clips=2, expected_frames=17,
    )
    left_a = {key: value for key, value in a.items() if key[0] == "mirror_left"}
    left_b = {key: value for key, value in b.items() if key[0] == "mirror_left"}
    try:
        summarize(
            left_a, left_b,
            source_width=160, source_height=96,
            target_width=832, target_height=480,
            grid_width=52, grid_height=30,
            state_min=DEFAULT_STATE_AGREEMENT_MIN,
            center_p95_cells_max=DEFAULT_CENTER_P95_CELLS_MAX,
            center_coverage_min=DEFAULT_CENTER_COVERAGE_MIN,
            expected_clips=2, expected_frames=17,
        )
        missing_clip_rejected = False
        missing_clip_error = None
    except ValueError as error:
        missing_clip_rejected = True
        missing_clip_error = str(error)
    outlier_a, outlier_b = build_single_outlier_rows()
    outlier_report = summarize(
        outlier_a, outlier_b,
        source_width=160, source_height=96,
        target_width=832, target_height=480,
        grid_width=52, grid_height=30,
        state_min=DEFAULT_STATE_AGREEMENT_MIN,
        center_p95_cells_max=DEFAULT_CENTER_P95_CELLS_MAX,
        center_coverage_min=DEFAULT_CENTER_COVERAGE_MIN,
        expected_clips=2, expected_frames=17,
    )
    outlier_left = outlier_report["per_clip"]["mirror_left"]
    single_outlier_sensitivity = {
        "outlier_clip": "mirror_left",
        "outlier_frame": 16,
        "absolute_center_disagreement_cells": outlier_left["center_abs_dx_cells_max"],
        "outlier_p95_disagreement_cells": outlier_left["center_abs_dx_cells_p95"],
        "pooled_p95_disagreement_cells": outlier_report["overall"]["center_abs_dx_cells_p95"],
        "outlier_max_frame": outlier_left["center_abs_dx_target_px_max_frame"],
        "per_clip_screen_pass": outlier_left["pass_screen"],
        "pooled_screen_pass": outlier_report["overall"]["pass_screen"],
        "interpretation": "A single 2.5-cell frame outlier among 34 label pairs can still pass the registered p95 screen. The max remains an explicit diagnostic and must receive the registered worst-frame review.",
    }
    one_px_mapped = map_source_delta_to_grid(
        1, 1, source_width=160, source_height=96,
        target_width=832, target_height=480, grid_width=52, grid_height=30,
    )
    old_8px_mapped = map_source_delta_to_grid(
        8, 8, source_width=160, source_height=96,
        target_width=832, target_height=480, grid_width=52, grid_height=30,
    )
    scale_checks = {
        "one_source_px_x_target_px": one_px_mapped["dx_target_px"],
        "one_source_px_y_target_px": one_px_mapped["dy_target_px"],
        "one_source_px_x_cells": one_px_mapped["dx_cells"],
        "one_source_px_y_cells": one_px_mapped["dy_cells"],
        "old_raw_8px_x_cell_equivalent": old_8px_mapped["dx_cells"],
        "old_raw_8px_y_cell_equivalent": old_8px_mapped["dy_cells"],
    }
    expected_image_keys = {
        key: value["image_file"] for key, value in a.items()
    }
    packet_rater_a_key_check = validate_rows_against_packet(a, expected_image_keys, label="Synthetic rater A")
    packet_rater_b_key_check = validate_rows_against_packet(b, expected_image_keys, label="Synthetic rater B")
    common_wrong_image_rows = {key: {**value} for key, value in a.items()}
    common_wrong_image_rows[("mirror_left", 0)]["image_file"] = "wrong_common_frame.png"
    common_wrong_packet_filename_rejected = False
    common_wrong_packet_filename_reason = None
    try:
        validate_rows_against_packet(
            common_wrong_image_rows, expected_image_keys, label="Synthetic common-wrong-image pair"
        )
    except ValueError as error:
        common_wrong_packet_filename_rejected = True
        common_wrong_packet_filename_reason = str(error)
    packet_root = ROOT / "gate1_manual_annotation_packets_v5" / "synthetic_reliability"
    packet_rater1_keys = load_packet_image_keys(packet_root / "rater_01.zip")
    packet_rater2_keys = load_packet_image_keys(packet_root / "rater_02.zip")
    packet_root_key_count = len(packet_rater1_keys)
    packet_maps_equal = packet_rater1_keys == packet_rater2_keys
    agreement_cli_packet_validation_passed = False
    agreement_cli_common_wrong_image_rejected = False
    agreement_cli_key_error = None
    with tempfile.TemporaryDirectory(prefix="gate1-agreement-cli-keycheck-") as temp_dir:
        temp_root = Path(temp_dir)
        reference_packet = temp_root / "synthetic_reference_packet.zip"
        with zipfile.ZipFile(reference_packet, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            template = io.StringIO(newline="")
            writer = csv.writer(template)
            writer.writerow(["blind_clip_id", "frame_index", "image_file"])
            for key, image_file in sorted(expected_image_keys.items()):
                writer.writerow([key[0], key[1], image_file])
            archive.writestr("annotation_template.csv", template.getvalue())
        rater_a_csv, rater_b_csv = temp_root / "rater_a.csv", temp_root / "rater_b.csv"
        write_label_csv(rater_a_csv, a)
        write_label_csv(rater_b_csv, b)
        result_path = temp_root / "agreement.json"
        command = [
            sys.executable, str(SCORER), str(rater_a_csv), str(rater_b_csv),
            "--source-width", "160", "--source-height", "96", "--expected-clips", "2",
            "--reference-packet", str(reference_packet), "--output", str(result_path),
        ]
        good_process = subprocess.run(command, check=False, capture_output=True, text=True)
        if good_process.returncode == 0:
            cli_report = json.loads(result_path.read_text(encoding="utf-8"))
            agreement_cli_packet_validation_passed = (
                cli_report["packet_key_validation"]["rater_a"]["exact_key_to_image_mapping"]
                and cli_report["packet_key_validation"]["rater_b"]["exact_key_to_image_mapping"]
            )
        wrong_a, wrong_b = {key: {**value} for key, value in a.items()}, {key: {**value} for key, value in b.items()}
        wrong_a[("mirror_left", 0)]["image_file"] = "wrong_common_frame.png"
        wrong_b[("mirror_left", 0)]["image_file"] = "wrong_common_frame.png"
        write_label_csv(rater_a_csv, wrong_a)
        write_label_csv(rater_b_csv, wrong_b)
        bad_process = subprocess.run(command, check=False, capture_output=True, text=True)
        agreement_cli_common_wrong_image_rejected = (
            bad_process.returncode != 0
            and "does not match the archived packet key-to-image map" in bad_process.stderr
        )
        agreement_cli_key_error = bad_process.stderr.strip().splitlines()[-1] if bad_process.stderr.strip() else None
    duration = time.perf_counter() - t0
    invalid_box_rejected = False
    invalid_box_reason = None
    blank_clip_id_rejected = False
    blank_clip_id_reason = None
    blank_image_file_rejected = False
    blank_image_file_reason = None
    with tempfile.TemporaryDirectory(prefix="gate1-invalid-box-") as temp_dir:
        invalid_path = Path(temp_dir) / "out_of_frame.csv"
        columns = [
            "blind_clip_id", "frame_index", "image_file", "object_visible", "instance_count",
            "identity_consistent", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
        ]
        with invalid_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow({
                "blind_clip_id": "bad", "frame_index": 0, "image_file": "bad_frame.png", "object_visible": "true",
                "instance_count": 1, "identity_consistent": "true",
                "bbox_x1_px": 10, "bbox_y1_px": 10, "bbox_x2_px": 161, "bbox_y2_px": 40,
            })
        try:
            load_rows(invalid_path, frame_width=160, frame_height=96)
        except ValueError as error:
            invalid_box_rejected = True
            invalid_box_reason = str(error)
        blank_id_path = Path(temp_dir) / "blank_clip_id.csv"
        with blank_id_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow({
                "blind_clip_id": "   ", "frame_index": 0, "image_file": "bad_frame.png", "object_visible": "true",
                "instance_count": 1, "identity_consistent": "true",
                "bbox_x1_px": 10, "bbox_y1_px": 10, "bbox_x2_px": 20, "bbox_y2_px": 20,
            })
        try:
            load_rows(blank_id_path, frame_width=160, frame_height=96)
        except ValueError as error:
            blank_clip_id_rejected = True
            blank_clip_id_reason = str(error)
        blank_image_path = Path(temp_dir) / "blank_image_file.csv"
        with blank_image_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerow({
                "blind_clip_id": "valid", "frame_index": 0, "image_file": "   ", "object_visible": "true",
                "instance_count": 1, "identity_consistent": "true",
                "bbox_x1_px": 10, "bbox_y1_px": 10, "bbox_x2_px": 20, "bbox_y2_px": 20,
            })
        try:
            load_rows(blank_image_path, frame_width=160, frame_height=96)
        except ValueError as error:
            blank_image_file_rejected = True
            blank_image_file_reason = str(error)
    mismatched_image_file_rejected = False
    mismatched_image_file_reason = None
    mismatched_b = {key: {**value} for key, value in b.items()}
    mismatched_b[("mirror_left", 0)]["image_file"] = "different_frame.png"
    try:
        summarize(
            a, mismatched_b,
            source_width=160, source_height=96,
            target_width=832, target_height=480,
            grid_width=52, grid_height=30,
            state_min=DEFAULT_STATE_AGREEMENT_MIN,
            center_p95_cells_max=DEFAULT_CENTER_P95_CELLS_MAX,
            center_coverage_min=DEFAULT_CENTER_COVERAGE_MIN,
            expected_clips=2, expected_frames=17,
        )
    except ValueError as error:
        mismatched_image_file_rejected = True
        mismatched_image_file_reason = str(error)
    report.update({
        "experiment_id": "EXP-LATENTGATE-GATE1-SCORER-UNIT-MAPPING-006",
        "purpose": "Verify CLI-level archived packet frame/image authentication, common-wrong-filename rejection, source-resolution mapping, cross-rater filename matching, out-of-frame and empty-key rejection, weak-clip pooling, expected-clip completeness, and p95 sensitivity to a single worst-frame outlier.",
        "synthetic_input_only": True,
        "scenario": {
            "clips": 2,
            "frames_per_clip": 17,
            "source_frame_wh": [160, 96],
            "target_frame_wh": [832, 480],
            "control_grid_wh": [52, 30],
            "state_labels": "all agree as visible, one instance, identity consistent",
            "mirror_left_box_center_disagreement_source_px": {"frames": [15, 16], "x": 15.0 * 160 / 832, "y": 0.0},
            "mirror_left_box_center_disagreement_target_px": {"frames": [15, 16], "x": 15.0, "y": 0.0},
            "mirror_right_box_center_disagreement_target_px": {"all_frames": 0.0},
        },
        "coordinate_scale_checks": scale_checks,
        "expected_interpretation": "One source px maps to 0.325 x-cells and 0.3125 y-cells. The former raw 8 px tolerance equals 2.6 x-cells and 2.5 y-cells on this source frame. After correct mapping, pooled p95 passes while mirror_left fails its per-clip p95; the overall screen fails, and submitting only one of the two expected clips is rejected. A separate one-frame 2.5-cell outlier passes the current p95 screen, so the reported per-clip maximum must be reviewed even on a numeric pass.",
        "single_outlier_sensitivity": single_outlier_sensitivity,
        "incomplete_clip_submission_rejected": missing_clip_rejected,
        "incomplete_clip_rejection_reason": missing_clip_error,
        "out_of_frame_box_rejected": invalid_box_rejected,
        "out_of_frame_box_rejection_reason": invalid_box_reason,
        "blank_clip_id_rejected": blank_clip_id_rejected,
        "blank_clip_id_rejection_reason": blank_clip_id_reason,
        "blank_image_file_rejected": blank_image_file_rejected,
        "blank_image_file_rejection_reason": blank_image_file_reason,
        "mismatched_image_file_rejected": mismatched_image_file_rejected,
        "mismatched_image_file_rejection_reason": mismatched_image_file_reason,
        "synthetic_packet_key_validation": {
            "rater_a": packet_rater_a_key_check,
            "rater_b": packet_rater_b_key_check,
        },
        "common_wrong_packet_filename_rejected": common_wrong_packet_filename_rejected,
        "common_wrong_packet_filename_rejection_reason": common_wrong_packet_filename_reason,
        "registered_v5_reference_packet_key_count": packet_root_key_count,
        "registered_v5_rater_packet_key_maps_equal": packet_maps_equal,
        "agreement_cli_packet_validation_passed": agreement_cli_packet_validation_passed,
        "agreement_cli_common_wrong_image_rejected": agreement_cli_common_wrong_image_rejected,
        "agreement_cli_common_wrong_image_error": agreement_cli_key_error,
        "started_at_utc": started.isoformat(),
        "duration_seconds": duration,
        "device": {
            "kind": "CPU-only",
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "gpu_used": False,
            "peak_vram_mb": None,
        },
        "scorer_sha256": hashlib.sha256(SCORER.read_bytes()).hexdigest(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert report["overall"]["pooled_thresholds_pass"] is True
    assert report["per_clip"]["mirror_left"]["pass_screen"] is False
    assert report["per_clip"]["mirror_right"]["pass_screen"] is True
    assert report["overall"]["pass_screen"] is False
    assert abs(report["per_clip"]["mirror_left"]["center_distance_target_px_max"] - 15.0) < 1e-9
    assert missing_clip_rejected
    assert invalid_box_rejected
    assert blank_clip_id_rejected
    assert blank_image_file_rejected
    assert mismatched_image_file_rejected
    assert common_wrong_packet_filename_rejected
    assert packet_root_key_count == 34
    assert packet_maps_equal
    assert agreement_cli_packet_validation_passed
    assert agreement_cli_common_wrong_image_rejected
    assert single_outlier_sensitivity["per_clip_screen_pass"] is True
    assert single_outlier_sensitivity["pooled_screen_pass"] is True
    assert abs(single_outlier_sensitivity["absolute_center_disagreement_cells"] - 2.5) < 1e-9
    assert abs(single_outlier_sensitivity["outlier_p95_disagreement_cells"] - 0.5) < 1e-9
    assert single_outlier_sensitivity["outlier_max_frame"] == 16
    assert abs(scale_checks["one_source_px_x_cells"] - 0.325) < 1e-12
    assert abs(scale_checks["one_source_px_y_cells"] - 0.3125) < 1e-12
    assert abs(scale_checks["old_raw_8px_x_cell_equivalent"] - 2.6) < 1e-12
    assert abs(scale_checks["old_raw_8px_y_cell_equivalent"] - 2.5) < 1e-12
    print(json.dumps({
        "output": str(args.output),
        "duration_seconds": duration,
        "pooled_center_p95_target_px": report["overall"]["center_distance_target_px_p95"],
        "left_center_p95_target_px": report["per_clip"]["mirror_left"]["center_distance_target_px_p95"],
        "left_p95_x_cells": report["per_clip"]["mirror_left"]["center_abs_dx_cells_p95"],
        "release_pass": report["overall"]["pass_screen"],
    }, indent=2))


if __name__ == "__main__":
    main()
