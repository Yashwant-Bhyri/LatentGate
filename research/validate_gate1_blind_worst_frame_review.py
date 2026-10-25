#!/usr/bin/env python3
"""Synthetic mechanics check for the blinded Gate 1 visual review helper."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from build_gate1_blind_worst_frame_review import build_review


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "research/gate1_manual_annotation_packets_v5/synthetic_reliability/rater_01.zip"
OUTPUT = ROOT / "research/gate1_blind_worst_frame_review_selfcheck_v2.json"
WIDTH, HEIGHT = 160, 96
FIELDS = [
    "blind_clip_id", "frame_index", "image_file", "object_visible", "instance_count",
    "identity_consistent", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px", "notes",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    with zipfile.ZipFile(PACKET) as archive:
        names = set(archive.namelist())
        assert not any("UNBLIND" in Path(name).name.upper() or "PRIVATE_KEY" in Path(name).name.upper() for name in names)
        template_member = "annotation_template.csv"
        template_text = archive.read(template_member).decode("utf-8-sig")
    template_rows = list(csv.DictReader(io.StringIO(template_text)))
    clips: dict[str, list[dict[str, str]]] = {}
    for row in template_rows:
        clips.setdefault(row["blind_clip_id"], []).append(row)
    assert len(clips) == 2 and all(len(rows) == 17 for rows in clips.values())
    injected_clip = sorted(clips)[0]
    injected_frame = 16
    shift_px = 2.5 * WIDTH / 52
    exact_box = [45.0, 30.0, 65.0, 50.0]

    with tempfile.TemporaryDirectory(prefix="gate1_blind_worst_review_") as tmp:
        work = Path(tmp)
        csv_a, csv_b = work / "rater_a.csv", work / "rater_b.csv"
        generated_rows: list[tuple[dict[str, str], list[float]]] = []
        for blind_id in sorted(clips):
            for row in sorted(clips[blind_id], key=lambda r: int(r["frame_index"])):
                generated_rows.append((row, exact_box))
        for path, shifted in ((csv_a, False), (csv_b, True)):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for row, box in generated_rows:
                    coords = list(box)
                    if shifted and row["blind_clip_id"] == injected_clip and int(row["frame_index"]) == injected_frame:
                        coords[0] += shift_px
                        coords[2] += shift_px
                    writer.writerow({
                        **row,
                        "object_visible": "true",
                        "instance_count": "1",
                        "identity_consistent": "true",
                        "bbox_x1_px": format(coords[0], ".12g"),
                        "bbox_y1_px": format(coords[1], ".12g"),
                        "bbox_x2_px": format(coords[2], ".12g"),
                        "bbox_y2_px": format(coords[3], ".12g"),
                        "notes": "",
                    })
        html_path = work / "blind_review.html"
        result = build_review(
            csv_a, csv_b, PACKET, html_path,
            source_width=WIDTH,
            source_height=HEIGHT,
            expected_clips=2,
        )
        html_text = html_path.read_text(encoding="utf-8")
        first_clip_review = result["per_clip_review"][injected_clip]

        assert result["blind_agreement_screen_pass"] is True
        assert first_clip_review["maximum_radial"]["frame_index"] == injected_frame
        assert first_clip_review["maximum_horizontal"]["frame_index"] == injected_frame
        assert math.isclose(first_clip_review["maximum_horizontal"]["grid_cells"], 2.5, abs_tol=1e-9)
        assert first_clip_review["review_frame_indices"] == [injected_frame]
        assert result["privacy"] == {
            "private_key_loaded": False,
            "fixture_truth_loaded": False,
            "unblinded_direction_included": False,
            "offline_assets_embedded": True,
        }
        assert "<script" not in html_text.lower()
        assert "https://" not in html_text.lower() and "http://" not in html_text.lower()
        assert html_text.count('<article class="frame-card">') == 2
        assert "PRIVATE_UNBLIND_KEY" not in html_text
        assert "mirror_left" not in html_text.lower() and "mirror_right" not in html_text.lower()

        result_summary = {
            "experiment_id": "EXP-LATENTGATE-GATE1-BLINDED-WORST-FRAME-REVIEW-SELF-CHECK-001",
            "status": "synthetic_review_mechanics_only_no_human_labels",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": "Read only the public v5 blinded packet ZIP and its blank template; generate temporary complete synthetic rater CSVs with a single +2.5 control-grid-cell x disagreement at frame 16 of one opaque blind clip; build and inspect a temporary offline report. No truth fixture or private key is loaded.",
            "injected_case": {
                "blind_clip_id": injected_clip,
                "frame_index": injected_frame,
                "horizontal_shift_source_px": shift_px,
                "horizontal_shift_grid_cells": 2.5,
            },
            "validation": {
                "registered_agreement_screen_passes_with_single_outlier": result["blind_agreement_screen_pass"],
                "radial_and_x_maxima_select_injected_frame": first_clip_review["maximum_radial"]["frame_index"] == injected_frame and first_clip_review["maximum_horizontal"]["frame_index"] == injected_frame,
                "offline_html_contains_blind_review_cards": html_text.count('<article class="frame-card">') == 2,
                "no_external_or_script_resources": "<script" not in html_text.lower() and "https://" not in html_text.lower() and "http://" not in html_text.lower(),
                "private_key_or_truth_loaded": False,
                "direction_aliases_rendered": False,
            },
            "review_summary": {
                "blind_agreement_screen_pass": result["blind_agreement_screen_pass"],
                "per_clip_review": result["per_clip_review"],
                "privacy": result["privacy"],
            },
            "inputs": {
                "review_builder_sha256": sha256(ROOT / "research/build_gate1_blind_worst_frame_review.py"),
                "packet_zip_sha256": sha256(PACKET),
            },
            "compute": {
                "runtime_seconds_including_html_build": round(time.perf_counter() - started, 6),
                "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
                "gpu_used": False,
                "peak_vram_mb": None,
                "model_inference": False,
            },
            "limitations": [
                "Synthetic arbitrary boxes and one injected outlier validate report selection/rendering only; they are not human labels or annotation precision evidence.",
                "The generated report and synthetic CSVs are temporary and deleted on exit.",
                "The helper must only be used with the two still-blinded submissions and a v5 packet archive; it never validates path policy efficacy.",
            ],
        }
    OUTPUT.write_text(json.dumps(result_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), **result_summary["validation"], "compute": result_summary["compute"]}, indent=2))


if __name__ == "__main__":
    main()
