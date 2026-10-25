#!/usr/bin/env python3
"""Verify blinded Gate 1 v5 packet images against the canonical RGB fixture.

This verifier intentionally never opens the private unblind key. It compares
unordered image-hash multisets so it cannot emit a blind-alias-to-direction map.
"""

from __future__ import annotations

import base64
import collections
import csv
import hashlib
import io
import json
import platform
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


RESEARCH = Path(__file__).resolve().parent
FIXTURE = RESEARCH / "kubric_gate0_fixture_v1"
PACKET_DIR = RESEARCH / "gate1_manual_annotation_packets_v5" / "synthetic_reliability"
PACKETS = (PACKET_DIR / "rater_01.zip", PACKET_DIR / "rater_02.zip")
REPORT = RESEARCH / "gate1_v5_packet_fixture_pixels_audit_v1.json"
SOURCE_FIELDS = (
    "object_visible", "instance_count", "identity_consistent",
    "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_packet(path: Path) -> dict[str, Any]:
    packet_bytes = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(packet_bytes)) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP integrity failure in {path.name}")
        expected = {"index.html", "annotation_template.csv", "README.txt"}
        if set(archive.namelist()) != expected:
            raise ValueError(f"Unexpected archive members in {path.name}")
        html = archive.read("index.html").decode("utf-8")
        csv_text = archive.read("annotation_template.csv").decode("utf-8-sig")

    match = re.search(r"const frames = (.*?);\nconst columns =", html, re.DOTALL)
    if not match:
        raise ValueError(f"Embedded frame array missing from {path.name}")
    frames = json.loads(match.group(1))
    rows_by_blind_id: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    frame_keys: list[tuple[str, int, str]] = []
    frame_hashes: list[str] = []
    for row in frames:
        blind_id = str(row["blind_clip_id"])
        frame_index = int(row["frame_index"])
        if (int(row["width"]), int(row["height"])) != (160, 96):
            raise ValueError(f"Unexpected frame size in {path.name}")
        if not str(row["image_file"]).endswith(f"/frame_{frame_index:03d}.png"):
            raise ValueError(f"Unexpected blinded frame alias in {path.name}")
        uri = str(row["data_uri"])
        prefix = "data:image/png;base64,"
        if not uri.startswith(prefix):
            raise ValueError(f"Unexpected image data URI in {path.name}")
        png_bytes = base64.b64decode(uri[len(prefix):], validate=True)
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Embedded frame is not a PNG in {path.name}")
        image_hash = digest(png_bytes)
        frame_hashes.append(image_hash)
        frame_keys.append((blind_id, frame_index, str(row["image_file"])))
        rows_by_blind_id[blind_id].append({"frame_index": frame_index, "sha256": image_hash})

    if len(frames) != 34 or len(rows_by_blind_id) != 2:
        raise ValueError(f"Expected two blinded clips x 17 frames in {path.name}")
    for blind_id, rows in rows_by_blind_id.items():
        if sorted(int(row["frame_index"]) for row in rows) != list(range(17)):
            raise ValueError(f"Incomplete or duplicate frame indices for a blind clip in {path.name}")

    labels = list(csv.DictReader(io.StringIO(csv_text)))
    if len(labels) != 34:
        raise ValueError(f"Expected 34 annotation rows in {path.name}")
    if any(any((row.get(field) or "").strip() for field in SOURCE_FIELDS) for row in labels):
        raise ValueError(f"A scored label field is already populated in {path.name}")
    manifest_key_counts = collections.Counter(frame_keys)
    csv_keys = [
        (str(row["blind_clip_id"]), int(row["frame_index"]), str(row["image_file"]))
        for row in labels
    ]
    csv_key_counts = collections.Counter(csv_keys)
    if any(count != 1 for count in manifest_key_counts.values()) or any(count != 1 for count in csv_key_counts.values()):
        raise ValueError(f"Duplicate annotation/frame manifest keys in {path.name}")
    csv_key_mapping_matches = csv_key_counts == manifest_key_counts
    if not csv_key_mapping_matches:
        raise ValueError(f"CSV frame keys do not map one-to-one to embedded frames in {path.name}")

    return {
        "archive": path.name,
        "archive_sha256": digest(packet_bytes),
        "frame_count": len(frames),
        "clip_count": len(rows_by_blind_id),
        "frames_per_clip": sorted(len(rows) for rows in rows_by_blind_id.values()),
        "frame_hash_multiset": collections.Counter(frame_hashes),
        "csv_rows": len(labels),
        "scored_fields_blank": True,
        "csv_keys_match_embedded_frame_manifest": csv_key_mapping_matches,
    }


def main() -> None:
    started = time.perf_counter()
    # Flatten the canonical source paths before hashing: do not associate a
    # packet's opaque clip ID with a direction or disclose a key mapping.
    source_paths = sorted(FIXTURE.glob("*/frames/frame_*.png"))
    if len(source_paths) != 34:
        raise ValueError(f"Expected exactly 34 source RGB frames, found {len(source_paths)}")
    source_hashes = collections.Counter(digest(p.read_bytes()) for p in source_paths)
    packet_records = [extract_packet(path) for path in PACKETS]
    source_matches = [record["frame_hash_multiset"] == source_hashes for record in packet_records]
    rater_packets_match = packet_records[0]["frame_hash_multiset"] == packet_records[1]["frame_hash_multiset"]
    errors = []
    if not all(source_matches):
        errors.append("one or more blinded packet image multisets differ from canonical fixture RGB frames")
    if not rater_packets_match:
        errors.append("rater packet image multisets differ")

    result = {
        "audit_id": "EXP-LATENTGATE-GATE1-V5-PACKET-FIXTURE-PIXELS-001",
        "status": "pass" if not errors else "fail",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "exact image-content provenance for blinded Kubric v5 annotation packets",
        "privacy_boundary": "PRIVATE_UNBLIND_KEY.json was not opened; results contain no blind-ID-to-direction mapping",
        "privacy": {
            "private_unblind_key_opened": False,
            "no_blind_id_direction_map_emitted": True,
        },
        "source_fixture": {
            "ground_truth_sha256": digest((FIXTURE / "ground_truth.json").read_bytes()),
            "frame_count": len(source_paths),
            "unique_png_hashes": len(source_hashes),
            "hash_multiset": "omitted to avoid exposing frame-to-blind-alias correspondences",
        },
        "rater_packets": [
            {k: v for k, v in record.items() if k != "frame_hash_multiset"}
            for record in packet_records
        ],
        "checks": {
            "rater_01_exact_source_frame_multiset": source_matches[0],
            "rater_02_exact_source_frame_multiset": source_matches[1],
            "rater_packets_have_same_image_multiset": rater_packets_match,
            "two_blind_clips_with_17_distinct_frame_indices_each": all(
                record["clip_count"] == 2 and record["frames_per_clip"] == [17, 17]
                for record in packet_records
            ),
            "csv_keys_match_embedded_frames_for_both_raters": all(
                record["csv_keys_match_embedded_frame_manifest"] for record in packet_records
            ),
            "all_label_fields_blank": all(record["scored_fields_blank"] for record in packet_records),
        },
        "errors": errors,
        "limitations": [
            "Exact packet/source equality verifies what files raters receive, not that human labels are reliable.",
            "The comparison is deliberately unordered and does not establish direction-to-alias mapping.",
            "Kubric fixture and generated-data distribution terms remain unresolved; packets remain local.",
            "This does not validate generated-video observers or policy efficacy.",
        ],
        "compute": {
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "python": sys.version.split()[0],
            "system": platform.system(),
            "machine": platform.machine(),
            "gpu_used": False,
            "peak_vram_mb": None,
        },
    }
    REPORT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
