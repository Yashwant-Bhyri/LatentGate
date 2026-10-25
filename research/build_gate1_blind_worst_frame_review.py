#!/usr/bin/env python3
"""Create an offline, still-blinded visual review of Gate 1 label outliers.

The report shows only the packet's blind IDs, source frames, and two submitted
label sets. It never loads the packet's private direction key or Kubric truth.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import platform
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from score_annotation_agreement import (
    DEFAULT_CENTER_COVERAGE_MIN,
    DEFAULT_CENTER_P95_CELLS_MAX,
    DEFAULT_STATE_AGREEMENT_MIN,
    load_packet_image_keys,
    load_rows,
    summarize,
    validate_rows_against_packet,
)


REQUIRED_IMAGE_FIELD = "image_file"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_paths(path: Path) -> dict[tuple[str, int], str]:
    image_paths: dict[tuple[str, int], str] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if REQUIRED_IMAGE_FIELD not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing the {REQUIRED_IMAGE_FIELD!r} column")
        for row in reader:
            key = (row["blind_clip_id"].strip(), int(row["frame_index"]))
            if key in image_paths:
                raise ValueError(f"{path} has a duplicate row for {key}")
            image_path = (row.get(REQUIRED_IMAGE_FIELD) or "").strip()
            member = PurePosixPath(image_path)
            if (not image_path or member.is_absolute() or ".." in member.parts
                    or "\\" in image_path):
                raise ValueError(f"Unsafe or empty packet image path for {key}: {image_path!r}")
            image_paths[key] = image_path
    return image_paths


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _box_svg(box: list[float] | None, color: str, label: str) -> str:
    if box is None:
        return ""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        f'<rect x="{x1:.6f}" y="{y1:.6f}" width="{x2-x1:.6f}" '
        f'height="{y2-y1:.6f}" fill="{color}" fill-opacity="0.05" '
        f'stroke="{color}" stroke-width="0.75" vector-effect="non-scaling-stroke"/>'
        f'<circle cx="{cx:.6f}" cy="{cy:.6f}" r="1.2" fill="{color}"/>'
        f'<text x="{x1:.6f}" y="{max(4.0, y1-1.5):.6f}" fill="{color}" '
        'font-size="3.5" stroke="white" stroke-width="0.6" '
        f'paint-order="stroke">{html.escape(label)}</text>'
    )


def build_review(
    rater_a_csv: Path,
    rater_b_csv: Path,
    packet_zip: Path,
    output_html: Path,
    *,
    source_width: int,
    source_height: int,
    expected_clips: int,
    expected_frames: int = 17,
    target_width: int = 832,
    target_height: int = 480,
    grid_width: int = 52,
    grid_height: int = 30,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows_a = load_rows(rater_a_csv, frame_width=source_width, frame_height=source_height)
    rows_b = load_rows(rater_b_csv, frame_width=source_width, frame_height=source_height)
    expected_image_keys = load_packet_image_keys(packet_zip)
    packet_key_validation = {
        "rater_a": validate_rows_against_packet(rows_a, expected_image_keys, label="Rater A"),
        "rater_b": validate_rows_against_packet(rows_b, expected_image_keys, label="Rater B"),
    }
    images_a, images_b = load_image_paths(rater_a_csv), load_image_paths(rater_b_csv)
    if set(images_a) != set(rows_a) or set(images_b) != set(rows_b):
        raise ValueError("CSV image-path rows must exactly match the scored label rows")
    if images_a != images_b:
        raise ValueError("The raters' blind frame-to-image mappings differ")

    agreement = summarize(
        rows_a,
        rows_b,
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        grid_width=grid_width,
        grid_height=grid_height,
        state_min=DEFAULT_STATE_AGREEMENT_MIN,
        center_p95_cells_max=DEFAULT_CENTER_P95_CELLS_MAX,
        center_coverage_min=DEFAULT_CENTER_COVERAGE_MIN,
        expected_clips=expected_clips,
        expected_frames=expected_frames,
    )

    selected_by_clip: dict[str, dict[int, set[str]]] = {}
    clip_review_summary: dict[str, Any] = {}
    for clip_id in sorted(agreement["per_clip"]):
        records = [r for r in agreement["frame_records"] if r["blind_clip_id"] == clip_id]
        center_records = [r for r in records if r["center_comparable"]]
        selected: dict[int, set[str]] = {}

        def select(record: dict[str, Any], reason: str) -> None:
            selected.setdefault(record["frame_index"], set()).add(reason)

        radial = max(center_records, key=lambda r: (r["center_distance_target_px"], -r["frame_index"]), default=None)
        max_x = max(center_records, key=lambda r: (r["center_abs_dx_target_px"], -r["frame_index"]), default=None)
        max_y = max(center_records, key=lambda r: (r["center_abs_dy_target_px"], -r["frame_index"]), default=None)
        if radial is not None:
            select(radial, "maximum radial center disagreement")
        if max_x is not None and max_x["center_abs_dx_target_px"] > 0:
            select(max_x, "maximum horizontal disagreement")
        if max_y is not None and max_y["center_abs_dy_target_px"] > 0:
            select(max_y, "maximum vertical disagreement")

        # Review visibility/count/identity disagreements separately from path geometry.
        state_mismatches = [r for r in records if not r["state_exact"]]
        for record in state_mismatches:
            select(record, "categorical visibility/count/identity disagreement")

        # Preserve uncertain/non-unique labels as their own review stratum; do not
        # turn their boxes into path-error observations.
        for record in records:
            row_a, row_b = rows_a[(clip_id, record["frame_index"])], rows_b[(clip_id, record["frame_index"])]
            if (not row_a["object_visible"] or not row_b["object_visible"]
                    or row_a["instance_count"] != 1 or row_b["instance_count"] != 1
                    or not row_a["identity_consistent"] or not row_b["identity_consistent"]):
                select(record, "visibility/identity/count prevents unique-target geometry")

        selected_by_clip[clip_id] = selected
        clip_metrics = agreement["per_clip"][clip_id]
        clip_review_summary[clip_id] = {
            "agreement_screen_pass": clip_metrics["pass_screen"],
            "center_comparable_frames": len(center_records),
            "maximum_radial": None if radial is None else {
                "frame_index": radial["frame_index"],
                "target_px": radial["center_distance_target_px"],
            },
            "maximum_horizontal": None if max_x is None else {
                "frame_index": max_x["frame_index"],
                "target_px": max_x["center_abs_dx_target_px"],
                "grid_cells": max_x["center_abs_dx_cells"],
            },
            "maximum_vertical": None if max_y is None else {
                "frame_index": max_y["frame_index"],
                "target_px": max_y["center_abs_dy_target_px"],
                "grid_cells": max_y["center_abs_dy_cells"],
            },
            "categorical_disagreement_frames": [r["frame_index"] for r in state_mismatches],
            "review_frame_indices": sorted(selected),
            "p95_abs_dx_cells": clip_metrics["center_abs_dx_cells_p95"],
            "p95_abs_dy_cells": clip_metrics["center_abs_dy_cells_p95"],
        }

    output_html.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_html.with_suffix(".json")
    image_data: dict[str, str] = {}
    with zipfile.ZipFile(packet_zip) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Packet ZIP integrity failure at {bad_member}")
        names = set(archive.namelist())
        if any("UNBLIND" in PurePosixPath(name).name.upper() or "PRIVATE_KEY" in PurePosixPath(name).name.upper()
               for name in names):
            raise ValueError("Refusing review: packet ZIP appears to contain a private/unblind key")
        if "index.html" not in names:
            raise ValueError("Packet ZIP must contain its offline index.html frame manifest")
        page = archive.read("index.html").decode("utf-8")
        marker = "const frames = "
        start = page.find(marker)
        end = page.find("];", start + len(marker)) if start >= 0 else -1
        if start < 0 or end < 0:
            raise ValueError("Packet index has no parseable embedded frame manifest")
        embedded_frames = json.loads(page[start + len(marker):end + 1])
        embedded: dict[str, str] = {}
        for frame in embedded_frames:
            member = frame.get("image_file")
            uri = frame.get("data_uri")
            if not isinstance(member, str) or not isinstance(uri, str) or not uri.startswith("data:image/png;base64,"):
                raise ValueError("Packet frame manifest contains an unsupported image record")
            encoded = uri.partition(",")[2]
            raw = base64.b64decode(encoded, validate=True)
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Packet frame {member!r} is not a PNG")
            if member in embedded:
                raise ValueError(f"Packet frame manifest duplicates {member!r}")
            embedded[member] = uri
        for key in sorted(images_a):
            member = images_a[key]
            if member not in embedded:
                raise ValueError(f"Packet ZIP has no frame image {member!r}")
            image_data[member] = embedded[member]

    all_cards: list[str] = []
    record_by_key = {(r["blind_clip_id"], r["frame_index"]): r for r in agreement["frame_records"]}
    for clip_id in sorted(selected_by_clip):
        for frame_index in sorted(selected_by_clip[clip_id]):
            key = (clip_id, frame_index)
            record = record_by_key[key]
            row_a, row_b = rows_a[key], rows_b[key]
            member = images_a[key]
            reasons = ", ".join(sorted(selected_by_clip[clip_id][frame_index]))
            display = lambda row: (
                f"visible={row['object_visible']}, count={row['instance_count']}, "
                f"identity_consistent={row['identity_consistent']}"
            )
            svg = (
                f'<svg viewBox="0 0 {source_width} {source_height}" role="img" '
                f'aria-label="Blinded labels on {html.escape(clip_id)} frame {frame_index}">'
                f'<image x="0" y="0" width="{source_width}" height="{source_height}" '
                f'href="{image_data[member]}" preserveAspectRatio="none"/>'
                f'{_box_svg(row_a["bbox"], "#ef7d22", "A")}'
                f'{_box_svg(row_b["bbox"], "#1677c8", "B")}</svg>'
            )
            all_cards.append(
                f'<article class="frame-card"><h3>Blind clip {html.escape(clip_id)} · frame {frame_index:02d}</h3>'
                f'<p class="reason">Review: {html.escape(reasons)}</p><div class="image">{svg}</div>'
                f'<p><b>Rater A:</b> {html.escape(display(row_a))}<br>'
                f'<b>Rater B:</b> {html.escape(display(row_b))}</p>'
                f'<p>Center comparable for path geometry: {record["center_comparable"]}<br>'
                f'|Δx|: {_fmt(record["center_abs_dx_source_px"])} source px / '
                f'{_fmt(record["center_abs_dx_cells"])} grid cells; '
                f'|Δy|: {_fmt(record["center_abs_dy_source_px"])} source px / '
                f'{_fmt(record["center_abs_dy_cells"])} grid cells; '
                f'radial: {_fmt(record["center_distance_source_px"])} source px; '
                f'box IoU: {_fmt(record["box_iou"])}</p></article>'
            )

    result: dict[str, Any] = {
        "artifact": "gate1_blinded_worst_frame_review_v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blind_agreement_screen_pass": agreement["overall"]["pass_screen"],
        "screen_thresholds": agreement["screen_thresholds"],
        "source_frame_wh": agreement["source_frame_wh"],
        "target_frame_wh": agreement["target_frame_wh"],
        "control_grid_wh": agreement["control_grid_wh"],
        "overall": agreement["overall"],
        "per_clip_review": clip_review_summary,
        "packet_key_validation": packet_key_validation,
        "inputs": {
            "rater_a_filename": rater_a_csv.name,
            "rater_a_sha256": sha256(rater_a_csv),
            "rater_b_filename": rater_b_csv.name,
            "rater_b_sha256": sha256(rater_b_csv),
            "packet_zip_filename": packet_zip.name,
            "packet_zip_sha256": sha256(packet_zip),
        },
        "privacy": {
            "private_key_loaded": False,
            "fixture_truth_loaded": False,
            "unblinded_direction_included": False,
            "offline_assets_embedded": True,
        },
        "compute": {
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
            "gpu_used": False,
            "peak_vram_mb": None,
            "model_inference": False,
        },
        "limitation": "This is a blinded visual review aid for worst disagreement and categorical flags, not an annotation scorer replacement, human agreement result, truth audit, or policy-efficacy result.",
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    cards = "\n".join(all_cards) or '<p class="empty">No frame met the review-selection rules. Inspect coverage and per-clip summary before proceeding.</p>'
    escaped_summary = html.escape(json.dumps({"overall": result["overall"], "per_clip_review": clip_review_summary}, indent=2))
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline';">
<title>Blinded Gate 1 worst-frame review</title>
<style>
body{{font:16px/1.45 system-ui,sans-serif;margin:24px;max-width:1120px;color:#17202a}}h1,h2{{line-height:1.2}}.notice{{padding:12px 16px;background:#eef6ff;border-left:4px solid #1677c8}}.legend{{display:flex;gap:22px;margin:16px 0}}.swatch{{display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:-2px}}.orange{{background:#ef7d22}}.blue{{background:#1677c8}}.frame-card{{border:1px solid #ccd4dc;border-radius:10px;padding:16px;margin:20px 0;max-width:760px}}.image{{width:min(640px,100%);image-rendering:pixelated}}svg{{width:100%;height:auto;display:block;background:#222}}.reason{{font-weight:600}}details{{margin:20px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:14px;border-radius:8px}}.empty{{padding:18px;background:#fff5e6}}
</style></head><body>
<h1>Blinded Gate 1 worst-frame review</h1>
<p class="notice"><b>Review aid only.</b> It shows blind clip IDs and the two submitted boxes. No private key or fixture truth is loaded. Keep visibility/count/identity review separate from path geometry. If the registered screen fails, correct labels while still blind and rescore; do not unblind.</p>
<div class="legend"><span><i class="swatch orange"></i>Rater A box and midpoint</span><span><i class="swatch blue"></i>Rater B box and midpoint</span></div>
<p>Agreement screen: <b>{'PASS' if result['blind_agreement_screen_pass'] else 'FAIL'}</b>. Review the maximum radial-error frame in each clip, plus nonzero axis-maximum frames when they differ, and any categorical/eligibility flags listed below.</p>
{cards}
<details><summary>Per-clip metrics and review frame selection</summary><pre>{escaped_summary}</pre></details>
<p>Machine-readable summary: {html.escape(json_path.name)}. This page is standalone and offline.</p>
</body></html>'''
    output_html.write_text(page, encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rater_a_csv", type=Path)
    parser.add_argument("rater_b_csv", type=Path)
    parser.add_argument("--packet-zip", type=Path, required=True,
                        help="One rater's blinded packet ZIP, which must contain the same frame paths and no key.")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path; a JSON sidecar is also written.")
    parser.add_argument("--source-width", type=int, default=160)
    parser.add_argument("--source-height", type=int, default=96)
    parser.add_argument("--expected-clips", type=int, default=2)
    parser.add_argument("--expected-frames", type=int, default=17)
    args = parser.parse_args()
    result = build_review(
        args.rater_a_csv, args.rater_b_csv, args.packet_zip, args.output,
        source_width=args.source_width,
        source_height=args.source_height,
        expected_clips=args.expected_clips,
        expected_frames=args.expected_frames,
    )
    print(json.dumps({
        "html": str(args.output),
        "summary": str(args.output.with_suffix('.json')),
        "blind_agreement_screen_pass": result["blind_agreement_screen_pass"],
        "per_clip_review": result["per_clip_review"],
        "privacy": result["privacy"],
        "compute": result["compute"],
    }, indent=2))


if __name__ == "__main__":
    main()
