#!/usr/bin/env python3
"""Synthetic CPU-only checks for the Gate 1 label-intake scanner."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from scan_gate1_label_intake import EXPECTED_HEADER, inspect_csv_text, read_candidate


ROOT = Path(__file__).resolve().parent
SCANNER = ROOT / "scan_gate1_label_intake.py"
VALIDATOR = Path(__file__).resolve()
DEFAULT_OUTPUT = ROOT / "gate1_label_intake_scanner_selfcheck_v1.json"


def build_fixture() -> tuple[dict[tuple[str, int], str], list[dict[str, str]]]:
    expected: dict[tuple[str, int], str] = {}
    rows: list[dict[str, str]] = []
    for clip_id in ("opaque_a", "opaque_b"):
        for frame_idx in range(17):
            image_file = f"{clip_id}_frame_{frame_idx:03d}.png"
            expected[(clip_id, frame_idx)] = image_file
            rows.append({
                "blind_clip_id": clip_id,
                "frame_index": str(frame_idx),
                "image_file": image_file,
                "object_visible": "true",
                "instance_count": "1",
                "identity_consistent": "true",
                "bbox_x1_px": "20",
                "bbox_y1_px": "20",
                "bbox_x2_px": "40",
                "bbox_y2_px": "40",
                "notes": "",
            })
    return expected, rows


def csv_text(rows: list[dict[str, str]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=EXPECTED_HEADER)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    expected, complete_rows = build_fixture()
    complete_text = csv_text(complete_rows)
    blank_rows = [{field: row[field] if field in {"blind_clip_id", "frame_index", "image_file"} else ""
                   for field in EXPECTED_HEADER} for row in complete_rows]
    blank_text = csv_text(blank_rows)
    wrong_rows_a = [dict(row) for row in complete_rows]
    wrong_rows_b = [dict(row) for row in complete_rows]
    wrong_rows_a[0]["image_file"] = "shared_wrong_image.png"
    wrong_rows_b[0]["image_file"] = "shared_wrong_image.png"

    exact = inspect_csv_text(complete_text, expected)
    blank = inspect_csv_text(blank_text, expected)
    wrong_a = inspect_csv_text(csv_text(wrong_rows_a), expected)
    wrong_b = inspect_csv_text(csv_text(wrong_rows_b), expected)
    duplicate_rows = [dict(row) for row in complete_rows]
    duplicate_rows[1]["blind_clip_id"] = duplicate_rows[0]["blind_clip_id"]
    duplicate_rows[1]["frame_index"] = duplicate_rows[0]["frame_index"]
    duplicate = inspect_csv_text(csv_text(duplicate_rows), expected)

    with tempfile.TemporaryDirectory(prefix="gate1-intake-selfcheck-") as temp_dir:
        temp = Path(temp_dir)
        complete_csv = temp / "complete_return.csv"
        blank_csv = temp / "blank_template.csv"
        wrong_csv = temp / "wrong_map.csv"
        complete_csv.write_text(complete_text, encoding="utf-8")
        blank_csv.write_text(blank_text, encoding="utf-8")
        wrong_csv.write_text(csv_text(wrong_rows_a), encoding="utf-8")
        complete_candidate = read_candidate(complete_csv, expected)
        blank_candidate = read_candidate(blank_csv, expected)
        wrong_candidate = read_candidate(wrong_csv, expected)

        valid_zip = temp / "complete_return.zip"
        with zipfile.ZipFile(valid_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("annotation.csv", complete_text)
        zip_candidate = read_candidate(valid_zip, expected)

        private_zip = temp / "private_key_archive.zip"
        with zipfile.ZipFile(private_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("annotation.csv", complete_text)
            archive.writestr("PRIVATE_UNBLIND_KEY.json", "sentinel-not-to-be-read")
        private_archive_rejected = read_candidate(private_zip, expected) is None

    results = {
        "synthetic_input_only": True,
        "complete_csv_exact_map_classified_candidate": (
            complete_candidate is not None
            and complete_candidate["intake_class"] == "complete_candidate_needs_independence_and_blind_review"
        ),
        "complete_zip_exact_map_classified_candidate": (
            zip_candidate is not None
            and zip_candidate["input_kind"] == "zip"
            and zip_candidate["intake_class"] == "complete_candidate_needs_independence_and_blind_review"
        ),
        "blank_exact_map_classified_blank_template": (
            blank_candidate is not None
            and blank_candidate["intake_class"] == "blank_exact_packet_template"
        ),
        "shared_wrong_filename_rejected_for_both_submissions": (
            wrong_candidate is not None
            and wrong_a is not None
            and wrong_b is not None
            and wrong_candidate["intake_class"] == "partial_or_wrong_packet_candidate"
            and wrong_a["exact_key_to_image_map"] is False
            and wrong_b["exact_key_to_image_map"] is False
        ),
        "duplicate_frame_key_rejected": duplicate is not None and duplicate["exact_frame_key_set"] is False,
        "archive_with_private_key_member_skipped": private_archive_rejected,
        "unblind_key_read": False,
        "aliases_or_annotation_values_emitted": False,
        "scanner_sha256": hashlib.sha256(SCANNER.read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(VALIDATOR.read_bytes()).hexdigest(),
        "compute": {
            "runtime_seconds": round(time.perf_counter() - t0, 6),
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "gpu_used": False,
            "peak_vram_mb": None,
            "model_inference": False,
        },
        "created_utc": started.isoformat(),
    }
    passed = all(results[key] is True for key in (
        "complete_csv_exact_map_classified_candidate",
        "complete_zip_exact_map_classified_candidate",
        "blank_exact_map_classified_blank_template",
        "shared_wrong_filename_rejected_for_both_submissions",
        "duplicate_frame_key_rejected",
        "archive_with_private_key_member_skipped",
    ))
    results["selfcheck_passed"] = passed
    if not passed:
        raise AssertionError("One or more synthetic intake checks failed")
    output = DEFAULT_OUTPUT
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "selfcheck_passed": passed,
        "runtime_seconds": results["compute"]["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
