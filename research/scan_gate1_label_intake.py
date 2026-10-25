#!/usr/bin/env python3
"""Inventory candidate Gate 1 CSV/ZIP returns without exposing blind labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from score_annotation_agreement import load_packet_image_keys


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.resolve()
SCANNER_PATH = Path(__file__).resolve()
DEFAULT_REFERENCE_PACKET = (
    ROOT / "gate1_manual_annotation_packets_v5" / "synthetic_reliability" / "rater_01.zip"
)
EXPECTED_HEADER = [
    "blind_clip_id", "frame_index", "image_file", "object_visible", "instance_count",
    "identity_consistent", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "notes",
]
ANNOTATION_FIELDS = EXPECTED_HEADER[3:]
STATE_FIELDS = ["object_visible", "instance_count", "identity_consistent"]
BOX_FIELDS = EXPECTED_HEADER[6:10]


def validate_private_output_path(output: Path) -> Path:
    """Require a fresh file in an existing owner-only directory outside the repo."""
    expanded = output.expanduser()
    if expanded.name in {"", ".", ".."}:
        raise ValueError("--output must name a new file")
    try:
        parent = expanded.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("--output parent must already exist") from exc
    if not parent.is_dir():
        raise ValueError("--output parent must be an existing directory")
    parent_stat = parent.stat()
    if not hasattr(os, "getuid") or parent_stat.st_uid != os.getuid():
        raise ValueError("--output parent must be owned by the current user")
    if parent_stat.st_mode & 0o077 or parent_stat.st_mode & 0o700 != 0o700:
        raise ValueError("--output parent must be owner-only and accessible (mode 0700)")

    target = parent / expanded.name
    try:
        target.resolve(strict=False).relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("--output must be outside the project tree")
    if target.exists() or target.is_symlink():
        raise ValueError("--output must be a fresh, nonexistent file")
    return target


def write_private_report(output: Path, report_text: str) -> None:
    """Create a mode-0600 report without following or replacing an existing path."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(report_text)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv_text(text: str) -> tuple[list[dict[str, str]], int, int, int]:
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != EXPECTED_HEADER:
        raise ValueError("CSV does not have the exact registered v5 header")
    rows = list(reader)
    annotation_rows = sum(
        any((row.get(field) or "").strip() for field in ANNOTATION_FIELDS)
        for row in rows
    )
    complete_state_rows = sum(
        all((row.get(field) or "").strip() for field in STATE_FIELDS)
        for row in rows
    )
    box_rows = sum(
        any((row.get(field) or "").strip() for field in BOX_FIELDS)
        for row in rows
    )
    return rows, annotation_rows, complete_state_rows, box_rows


def inspect_csv_text(
    text: str,
    expected_image_keys: dict[tuple[str, int], str],
) -> dict[str, Any] | None:
    try:
        rows, annotation_rows, complete_state_rows, box_rows = read_csv_text(text)
    except (csv.Error, ValueError):
        return None

    observed: dict[tuple[str, int], str] = {}
    duplicate_keys = False
    invalid_key_rows = 0
    for row in rows:
        clip_id = (row.get("blind_clip_id") or "").strip()
        image_file = (row.get("image_file") or "").strip()
        try:
            frame_index = int(row.get("frame_index") or "")
        except ValueError:
            invalid_key_rows += 1
            continue
        key = (clip_id, frame_index)
        if not clip_id or not image_file:
            invalid_key_rows += 1
            continue
        if key in observed:
            duplicate_keys = True
        observed[key] = image_file

    exact_frame_keys = not duplicate_keys and set(observed) == set(expected_image_keys)
    exact_key_to_image_map = exact_frame_keys and all(
        observed.get(key) == image_file for key, image_file in expected_image_keys.items()
    )
    return {
        "exact_v5_header": True,
        "row_count": len(rows),
        "annotation_rows_with_any_nonkey_value": annotation_rows,
        "rows_with_all_visibility_count_identity_fields": complete_state_rows,
        "rows_with_any_box_coordinate": box_rows,
        "exact_frame_key_set": exact_frame_keys,
        "exact_key_to_image_map": exact_key_to_image_map,
        "duplicate_frame_keys": duplicate_keys,
        "invalid_key_rows": invalid_key_rows,
        "full_packet_rows": len(rows) == len(expected_image_keys),
    }


def read_candidate(path: Path, expected_image_keys: dict[tuple[str, int], str]) -> dict[str, Any] | None:
    try:
        if path.suffix.lower() == ".csv":
            text = path.read_text(encoding="utf-8-sig")
        else:
            with zipfile.ZipFile(path) as archive:
                members = archive.namelist()
                if any(
                    "UNBLIND" in Path(name).name.upper()
                    or "PRIVATE_KEY" in Path(name).name.upper()
                    for name in members
                ):
                    return None
                csv_members = [name for name in members if name.lower().endswith(".csv")]
                if len(csv_members) != 1:
                    return None
                info = archive.getinfo(csv_members[0])
                if info.file_size > 10_000_000:
                    return None
                text = archive.read(csv_members[0]).decode("utf-8-sig")
        result = inspect_csv_text(text, expected_image_keys)
        if result is None:
            return None
        stat = path.stat()
        result.update({
            "path": str(path.resolve()),
            "input_kind": "csv" if path.suffix.lower() == ".csv" else "zip",
            "bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
        lower_parts = [part.lower() for part in path.parts]
        lower_name = path.name.lower()
        known_nonhuman = (
            "qa" in lower_parts
            or "smoke" in lower_name
            or "synthetic" in lower_parts
        )
        if known_nonhuman:
            result["intake_class"] = "known_qa_or_synthetic_exclude"
        elif result["annotation_rows_with_any_nonkey_value"] == 0 and result["exact_key_to_image_map"]:
            result["intake_class"] = "blank_exact_packet_template"
        elif result["annotation_rows_with_any_nonkey_value"] == 0:
            result["intake_class"] = "blank_or_wrong_packet_template"
        elif (
            result["exact_key_to_image_map"]
            and result["full_packet_rows"]
            and result["rows_with_all_visibility_count_identity_fields"] == len(expected_image_keys)
        ):
            result["intake_class"] = "complete_candidate_needs_independence_and_blind_review"
        else:
            result["intake_class"] = "partial_or_wrong_packet_candidate"
        return result
    except (OSError, UnicodeError, zipfile.BadZipFile, csv.Error, RuntimeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots", nargs="*", type=Path,
        default=[ROOT, Path.home() / "Downloads", Path.home() / "Desktop"],
    )
    parser.add_argument("--reference-packet", type=Path, default=DEFAULT_REFERENCE_PACKET)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="fresh JSON path in an existing owner-only directory outside the project",
    )
    args = parser.parse_args()
    try:
        output_path = validate_private_output_path(args.output)
    except ValueError as exc:
        parser.error(str(exc))

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    expected_image_keys = load_packet_image_keys(args.reference_packet)
    candidates: list[dict[str, Any]] = []
    path_count = 0
    csv_zip_count = 0
    for root in args.roots:
        if not root.exists():
            continue
        for base, _, files in os.walk(root, followlinks=False):
            for filename in files:
                suffix = Path(filename).suffix.lower()
                if suffix not in {".csv", ".zip"}:
                    continue
                path_count += 1
                candidate = read_candidate(Path(base) / filename, expected_image_keys)
                if candidate is not None:
                    csv_zip_count += 1
                    candidates.append(candidate)

    report = {
        "schema": "latentgate_gate1_label_intake_scan_v1",
        "created_utc": started.isoformat(),
        "roots": [str(root.expanduser().resolve()) for root in args.roots],
        "reference_packet": str(args.reference_packet.expanduser().resolve()),
        "reference_packet_sha256": sha256(args.reference_packet),
        "scanner_sha256": sha256(SCANNER_PATH),
        "expected_frame_key_count": len(expected_image_keys),
        "csv_and_zip_paths_scanned": path_count,
        "exact_v5_header_candidates": csv_zip_count,
        "intake_class_counts": {
            label: sum(candidate["intake_class"] == label for candidate in candidates)
            for label in (
                "known_qa_or_synthetic_exclude",
                "blank_exact_packet_template",
                "blank_or_wrong_packet_template",
                "complete_candidate_needs_independence_and_blind_review",
                "partial_or_wrong_packet_candidate",
            )
        },
        "candidates": candidates,
        "privacy": {
            "annotation_values_emitted": False,
            "blind_clip_aliases_emitted": False,
            "alias_to_direction_mapping_read_or_emitted": False,
            "unblind_or_private_key_read": False,
            "archives_extracted": False,
        },
        "compute": {
            "runtime_seconds": round(time.perf_counter() - t0, 6),
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "gpu_used": False,
            "peak_vram_mb": None,
            "model_inference": False,
        },
    }
    try:
        write_private_report(output_path, json.dumps(report, indent=2) + "\n")
    except OSError as exc:
        parser.error(f"could not create private report: {exc}")
    print(json.dumps({
        "output": str(output_path),
        "csv_and_zip_paths_scanned": path_count,
        "exact_v5_header_candidates": csv_zip_count,
        "intake_class_counts": report["intake_class_counts"],
        "runtime_seconds": report["compute"]["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
