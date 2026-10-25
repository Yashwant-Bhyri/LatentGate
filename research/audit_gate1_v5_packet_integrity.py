#!/usr/bin/env python3
"""Audit Gate 1 v5 packets for blindness, offline behavior, and archive integrity."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import re
import sys
import time
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET_ROOT = ROOT / "research" / "gate1_manual_annotation_packets_v5"
MANIFEST_PATH = PACKET_ROOT / "packet_manifest.json"
KEY_PATH = PACKET_ROOT / "PRIVATE_UNBLIND_KEY.json"
SYNTHETIC_DIR = PACKET_ROOT / "synthetic_reliability"
RATER_ARCHIVES = [SYNTHETIC_DIR / "rater_01.zip", SYNTHETIC_DIR / "rater_02.zip"]
CSV_LABEL_FIELDS = (
    "object_visible",
    "instance_count",
    "identity_consistent",
    "bbox_x1_px",
    "bbox_y1_px",
    "bbox_x2_px",
    "bbox_y2_px",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_packet(path: Path, manifest_archives: dict[str, dict]) -> dict:
    relative = str(path.relative_to(PACKET_ROOT))
    archive_bytes = path.read_bytes()
    if sha256(archive_bytes) != manifest_archives[relative]["sha256"]:
        raise AssertionError(f"manifest hash mismatch: {relative}")

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.testzip() is not None:
            raise AssertionError(f"ZIP CRC failure: {relative}")
        expected_members = {"index.html", "annotation_template.csv", "README.txt"}
        if set(archive.namelist()) != expected_members:
            raise AssertionError(f"unexpected packet members: {relative}")
        html = archive.read("index.html").decode("utf-8")
        csv_bytes = archive.read("annotation_template.csv")
        readme = archive.read("README.txt").decode("utf-8")

    if "PRIVATE_UNBLIND_KEY.json" in relative or "PRIVATE_UNBLIND_KEY.json" in readme:
        raise AssertionError(f"private unblind key leaked into packet: {relative}")
    for csp_part in ("default-src 'none'", "img-src data:", "script-src 'unsafe-inline'"):
        if csp_part not in html:
            raise AssertionError(f"expected restrictive CSP missing from {relative}: {csp_part}")
    html_without_images = re.sub(
        r"data:image/png;base64,[A-Za-z0-9+/=]+", "data:image/png;base64,<removed>", html
    )
    forbidden = re.compile(
        r"<script\s+src=|<img\s+src=[\"']https?://|\bfetch\s*\(|"
        r"XMLHttpRequest|sendBeacon|new\s+WebSocket",
        re.IGNORECASE,
    )
    if forbidden.search(html_without_images):
        raise AssertionError(f"external code/network API found in {relative}")
    if "Math.min(960,Math.max(640,f.width))" not in html:
        raise AssertionError(f"v5 display scaling missing from {relative}")
    if "getScreenCTM().inverse()" not in html:
        raise AssertionError(f"source-space pointer transform missing from {relative}")

    frame_match = re.search(r"const frames = (.*?);\nconst columns =", html, re.DOTALL)
    if not frame_match:
        raise AssertionError(f"embedded frame manifest not found in {relative}")
    frames = json.loads(frame_match.group(1))
    if len(frames) != 34 or {f["width"] for f in frames} != {160} or {f["height"] for f in frames} != {96}:
        raise AssertionError(f"unexpected synthetic packet shape: {relative}")

    clip_frames: OrderedDict[str, list[str]] = OrderedDict()
    for frame in frames:
        blind_id = frame["blind_clip_id"]
        image_file = frame["image_file"]
        if not image_file.startswith(f"{blind_id}/frame_"):
            raise AssertionError(f"image alias mismatch in {relative}")
        if any(label in (blind_id + image_file).lower() for label in ("left", "right")):
            raise AssertionError(f"direction leaked in frame alias in {relative}")
        clip_frames.setdefault(blind_id, []).append(sha256(frame["data_uri"].encode("ascii")))
    if len(clip_frames) != 2 or any(len(hashes) != 17 for hashes in clip_frames.values()):
        raise AssertionError(f"expected 2 blinded clips x 17 frames in {relative}")

    csv_rows = list(csv.DictReader(io.TextIOWrapper(io.BytesIO(csv_bytes), encoding="utf-8-sig")))
    if len(csv_rows) != 34:
        raise AssertionError(f"expected 34 annotation rows in {relative}")
    if any(any((row.get(field) or "").strip() for field in CSV_LABEL_FIELDS) for row in csv_rows):
        raise AssertionError(f"packet template contains a scored annotation in {relative}")

    return {
        "archive": relative,
        "archive_sha256": sha256(archive_bytes),
        "clip_order": list(clip_frames),
        "clip_frame_sha256": clip_frames,
        "packet_key": re.search(r'const packetKey = "([^"]+)"', html).group(1),
        "frame_count": len(frames),
        "source_size_wh": [160, 96],
        "template_rows": len(csv_rows),
        "scored_fields_blank": True,
        "offline_csp_and_source_coordinate_mapping_present": True,
    }


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_archives = {item["path"]: item for item in manifest["archives"]}
    packets = [read_packet(path, manifest_archives) for path in RATER_ARCHIVES]
    if packets[0]["clip_order"] != list(reversed(packets[1]["clip_order"])):
        raise AssertionError("raters must receive the same clips in reversed order")
    if packets[0]["clip_order"] != packets[1]["clip_order"] and set(packets[0]["clip_order"]) != set(packets[1]["clip_order"]):
        raise AssertionError("rater packets do not share the same blind clip aliases")
    for blind_id in packets[0]["clip_order"]:
        if packets[0]["clip_frame_sha256"][blind_id] != packets[1]["clip_frame_sha256"][blind_id]:
            raise AssertionError(f"frame content mismatch for blind clip {blind_id}")

    private_key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    key_ids = set(private_key["synthetic"])
    packet_ids = set(packets[0]["clip_order"])
    if key_ids != packet_ids:
        raise AssertionError("external unblind key does not map exactly the blinded clip IDs")
    if any("PRIVATE_UNBLIND_KEY.json" in packet["archive"] for packet in packets):
        raise AssertionError("unblind key appears in a distributed archive")

    for packet in packets:
        packet.pop("clip_frame_sha256")
    result = {
        "audit_id": "EXP-LATENTGATE-GATE1-V5-PACKET-INTEGRITY-001",
        "status": "integrity_and_blinding_structure_pass_human_labels_pending",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
        "gpu_used": False,
        "peak_vram_mb": None,
        "packet_version": manifest["packet_version"],
        "rater_packets": packets,
        "checks": {
            "same_embedded_pixels_for_each_shared_blind_clip": True,
            "mirrored_clip_order_reversed_between_raters": True,
            "unblind_key_outside_archives": True,
            "restrictive_csp_and_no_external_code_or_network_api": True,
            "source_pixel_coordinate_mapping_present": True,
            "all_packet_templates_blank": True,
            "only_expected_offline_files_in_zip": True,
        },
        "limitation": "This is a static packet-integrity audit. It does not measure human annotation agreement, generated-video observer accuracy, or policy efficacy.",
    }
    output = ROOT / "research" / "gate1_v5_packet_integrity_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output.relative_to(ROOT))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
