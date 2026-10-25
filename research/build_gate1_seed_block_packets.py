#!/usr/bin/env python3
"""Build blinded labels for a second Kubric seed block after the first passes.

The builder requires a pass record tied to the exact first-block CSVs,
agreement report, and completed blind-review artifact. It emits two identical
frame sets under shared blind IDs, with the sequence order reversed by rater.
The private direction key is stored separately from the rater ZIPs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import secrets
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from build_blinded_box_packets import add_packet, png_frame  # noqa: E402
from kubric_gate0_fixture_v1.scoring import FRAME_COUNT, FRAME_SIZE_WH  # noqa: E402
from score_annotation_agreement import (  # noqa: E402
    DEFAULT_CENTER_COVERAGE_MIN,
    DEFAULT_CENTER_P95_CELLS_MAX,
    DEFAULT_STATE_AGREEMENT_MIN,
    load_packet_image_keys,
    load_rows,
    summarize,
    validate_rows_against_packet,
)


GATE_RECORD_VERSION = "gate1_first_seed_block_pass_v2"
FIRST_BLOCK_PACKET_PATH = RESEARCH / "gate1_manual_annotation_packets_v5" / "synthetic_reliability" / "rater_01.zip"
FIRST_BLOCK_PACKET_SHA256 = "0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8"
FIRST_BLOCK_UNBLIND_KEY_SHA256 = "ad99e6d97500952472d553d87f02c97ff16b8775b045c3a0f66acd4440441292"
FIRST_BLOCK_GROUND_TRUTH_SHA256 = "0dd6d5518de3c82d2e73d912c44fc712978d65dd93c38f8eb24d37500f8ad56b"
SOURCE_WIDTH, SOURCE_HEIGHT = FRAME_SIZE_WH
TARGET_WIDTH, TARGET_HEIGHT = 832, 480
GRID_WIDTH, GRID_HEIGHT = 52, 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def png_size_wh(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Fixture frame is not a valid PNG with an IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def referenced_file(record_path: Path, item: dict[str, Any], label: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"{label} must contain a file path")
    expected_hash = item.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"{label} must contain a SHA-256 digest")
    path = Path(item["path"]).expanduser()
    if not path.is_absolute():
        path = record_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    actual_hash = sha256(path)
    if actual_hash != expected_hash.lower():
        raise ValueError(f"{label} SHA-256 mismatch for {path}")
    return path


def compare_screen_summary(actual: dict[str, Any], recomputed: dict[str, Any], label: str) -> None:
    fields = (
        "frame_pairs",
        "joint_visibility_count_identity_exact_rate",
        "unique_visible_center_pair_coverage",
        "center_abs_dx_cells_p95",
        "center_abs_dy_cells_p95",
        "pass_screen",
    )
    for field in fields:
        if actual.get(field) != recomputed.get(field):
            raise ValueError(f"The first-block agreement report does not match current CSVs ({label}.{field})")


def validate_first_block_pass(record_path: Path) -> dict[str, Any]:
    record_path = record_path.expanduser().resolve()
    if not record_path.is_file():
        raise ValueError(f"First-block pass record does not exist: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("record_version") != GATE_RECORD_VERSION or record.get("decision") != "pass":
        raise ValueError("The first-block record must declare the supported version and decision='pass'")

    rater_items = record.get("rater_csvs")
    if not isinstance(rater_items, list) or len(rater_items) != 2:
        raise ValueError("The pass record must bind exactly two completed rater CSVs")
    rater_paths = [
        referenced_file(record_path, item, f"rater_csvs[{index}]")
        for index, item in enumerate(rater_items)
    ]
    if rater_paths[0] == rater_paths[1]:
        raise ValueError("The pass record must point to two distinct rater CSV files")
    packet_path = referenced_file(record_path, record.get("first_block_packet_zip"), "first_block_packet_zip")
    if packet_path != FIRST_BLOCK_PACKET_PATH.resolve() or sha256(packet_path) != FIRST_BLOCK_PACKET_SHA256:
        raise ValueError("The pass record must bind the registered v5 first-block rater_01 packet ZIP")
    rows_a = load_rows(rater_paths[0], frame_width=SOURCE_WIDTH, frame_height=SOURCE_HEIGHT)
    rows_b = load_rows(rater_paths[1], frame_width=SOURCE_WIDTH, frame_height=SOURCE_HEIGHT)
    expected_image_keys = load_packet_image_keys(packet_path)
    packet_key_validation = {
        "reference_packet_sha256": sha256(packet_path),
        "rater_a": validate_rows_against_packet(rows_a, expected_image_keys, label="Rater A"),
        "rater_b": validate_rows_against_packet(rows_b, expected_image_keys, label="Rater B"),
    }
    agreement = summarize(
        rows_a,
        rows_b,
        source_width=SOURCE_WIDTH,
        source_height=SOURCE_HEIGHT,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        state_min=DEFAULT_STATE_AGREEMENT_MIN,
        center_p95_cells_max=DEFAULT_CENTER_P95_CELLS_MAX,
        center_coverage_min=DEFAULT_CENTER_COVERAGE_MIN,
        expected_clips=2,
        expected_frames=FRAME_COUNT,
    )
    if not agreement["overall"]["pass_screen"]:
        raise ValueError("The exact first-block CSV pair fails the pooled or per-clip screen")

    report_path = referenced_file(record_path, record.get("agreement_report"), "agreement_report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("expected_clip_count") != 2 or report.get("expected_frames_per_clip") != FRAME_COUNT:
        raise ValueError("The referenced agreement report is not the registered two-clip, 17-frame screen")
    if not report.get("overall", {}).get("pass_screen"):
        raise ValueError("The referenced agreement report does not record a passing pooled/per-clip screen")
    report_key_check = report.get("packet_key_validation", {})
    if report_key_check.get("reference_packet_sha256") != sha256(packet_path):
        raise ValueError("The agreement report did not validate against the registered first-block packet map")
    if any(report_key_check.get(rater, {}).get("exact_key_to_image_mapping") is not True
           for rater in ("rater_a", "rater_b")):
        raise ValueError("The agreement report does not record exact packet key-to-image validation for both raters")
    if set(report.get("per_clip", {})) != set(agreement["per_clip"]):
        raise ValueError("The referenced agreement report does not match the two CSV blind IDs")
    compare_screen_summary(report["overall"], agreement["overall"], "overall")
    for blind_id, clip_summary in agreement["per_clip"].items():
        compare_screen_summary(report["per_clip"][blind_id], clip_summary, f"per_clip[{blind_id}]")

    truth_report_path = referenced_file(
        record_path,
        record.get("known_truth_report"),
        "known_truth_report",
    )
    truth_report = json.loads(truth_report_path.read_text(encoding="utf-8"))
    if truth_report.get("status") != "diagnostic_only_not_human_gate_or_policy_efficacy":
        raise ValueError("The referenced known-truth report has an unexpected status")
    if truth_report.get("unblind_key_sha256") != FIRST_BLOCK_UNBLIND_KEY_SHA256:
        raise ValueError("The known-truth report does not use the registered first-block private key")
    if truth_report.get("ground_truth_sha256") != FIRST_BLOCK_GROUND_TRUTH_SHA256:
        raise ValueError("The known-truth report does not use the registered first-block Kubric truth")
    if truth_report.get("blind_agreement_gate", {}).get("status") != "passed_before_unblinding":
        raise ValueError("The referenced known-truth report does not record a prior blinded pass")
    truth_key_check = truth_report.get("packet_key_validation", {})
    if truth_key_check.get("reference_packet_sha256") != sha256(packet_path):
        raise ValueError("The known-truth report did not validate against the registered first-block packet map")
    if any(truth_key_check.get(rater, {}).get("exact_key_to_image_mapping") is not True
           for rater in ("rater_a", "rater_b")):
        raise ValueError("The known-truth report does not record exact packet key-to-image validation for both raters")
    truth_report_inputs = truth_report.get("rater_inputs")
    if (
        not isinstance(truth_report_inputs, list)
        or len(truth_report_inputs) != 2
        or any(not isinstance(item, dict) or not isinstance(item.get("sha256"), str)
               for item in truth_report_inputs)
    ):
        raise ValueError("The known-truth report does not bind two rater CSV hashes")
    report_input_hashes = sorted(item["sha256"] for item in truth_report_inputs)
    if report_input_hashes != sorted(sha256(path) for path in rater_paths):
        raise ValueError("The known-truth report does not reference the exact rater CSVs in the pass record")

    review = record.get("blind_review")
    if not isinstance(review, dict):
        raise ValueError("The pass record must include a completed blind worst-frame review")
    if review.get("human_review_complete") is not True or review.get("corrections_rescored") is not True:
        raise ValueError("The blind review must be complete, including rescoring after any corrections")
    review_path = referenced_file(record_path, review.get("artifact"), "blind_review.artifact")
    if review_path.suffix.lower() != ".json":
        raise ValueError("blind_review.artifact must reference the machine-readable JSON sidecar")
    review_report = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review_report, dict):
        raise ValueError("The blind-review JSON must contain an object")
    if review_report.get("artifact") != "gate1_blinded_worst_frame_review_v1":
        raise ValueError("The blind-review JSON is not a supported Gate 1 review artifact")
    if review_report.get("blind_agreement_screen_pass") is not True:
        raise ValueError("The blind-review artifact does not record a passing agreement screen")
    review_key_check = review_report.get("packet_key_validation", {})
    if any(review_key_check.get(rater, {}).get("exact_key_to_image_mapping") is not True
           for rater in ("rater_a", "rater_b")):
        raise ValueError("The blind-review artifact does not record packet key-to-image verification")
    review_inputs = review_report.get("inputs")
    review_input_hashes = [
        review_inputs.get("rater_a_sha256"), review_inputs.get("rater_b_sha256")
    ] if isinstance(review_inputs, dict) else []
    if len(review_input_hashes) != 2 or any(
        not isinstance(value, str) or len(value) != 64 for value in review_input_hashes
    ):
        raise ValueError("The blind-review artifact must contain both rater CSV SHA-256 values")
    if sorted(review_input_hashes) != sorted(sha256(path) for path in rater_paths):
        raise ValueError("The blind-review artifact does not bind the exact rater CSVs in the pass record")
    if review_inputs.get("packet_zip_sha256") != sha256(packet_path):
        raise ValueError("The blind-review artifact was not generated from the registered first-block packet ZIP")
    if review_inputs.get("packet_zip_filename") != packet_path.name:
        raise ValueError("The blind-review artifact does not name the registered first-block packet ZIP")
    review_privacy = review_report.get("privacy")
    if not isinstance(review_privacy, dict) or any(
        review_privacy.get(field) is not False
        for field in ("private_key_loaded", "fixture_truth_loaded", "unblinded_direction_included")
    ):
        raise ValueError("The blind-review artifact must confirm that no key, fixture truth, or direction was loaded")
    per_clip_review = review_report.get("per_clip_review")
    if not isinstance(per_clip_review, dict) or set(per_clip_review) != set(agreement["per_clip"]):
        raise ValueError("The blind-review artifact does not cover the exact two first-block blind clip IDs")
    if any(
        item.get("agreement_screen_pass") is not True
        for item in per_clip_review.values()
        if isinstance(item, dict)
    ) or any(not isinstance(item, dict) for item in per_clip_review.values()):
        raise ValueError("The blind-review artifact does not record a passing screen for every clip")

    attestation = review.get("reviewer_attestation")
    if not isinstance(attestation, dict) or not isinstance(attestation.get("reviewer"), str) or not attestation["reviewer"].strip():
        raise ValueError("The pass record must identify the human reviewer")
    if attestation.get("decision") != "pass":
        raise ValueError("The human reviewer must explicitly attest decision='pass'")
    reviewed_at = parse_utc_timestamp(attestation.get("reviewed_utc"), "blind_review.reviewer_attestation.reviewed_utc")
    review_generated_at = parse_utc_timestamp(review_report.get("generated_utc"), "blind-review generated_utc")
    if reviewed_at <= review_generated_at:
        raise ValueError("The reviewer attestation must postdate the machine-generated blind-review artifact")
    truth_generated_at = parse_utc_timestamp(truth_report.get("created_utc"), "known-truth created_utc")
    if truth_generated_at <= reviewed_at:
        raise ValueError("The known-truth diagnostic must postdate the blinded human review")

    return {
        "record_sha256": sha256(record_path),
        "agreement_report_path": str(report_path),
        "agreement_report_sha256": sha256(report_path),
        "first_block_packet_zip_path": str(packet_path),
        "first_block_packet_zip_sha256": sha256(packet_path),
        "packet_key_validation": packet_key_validation,
        "known_truth_report_path": str(truth_report_path),
        "known_truth_report_sha256": sha256(truth_report_path),
        "blind_review_artifact_path": str(review_path),
        "blind_review_artifact_sha256": sha256(review_path),
        "blind_reviewer": attestation["reviewer"].strip(),
        "blind_reviewed_utc": reviewed_at.isoformat(),
        "rater_csvs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in rater_paths
        ],
        "agreement_screen": {
            "pooled_pass": agreement["overall"]["pooled_thresholds_pass"],
            "each_clip_pass": agreement["overall"]["all_clip_thresholds_pass"],
            "per_clip_count": len(agreement["per_clip"]),
        },
        "human_review_complete": True,
        "corrections_rescored": True,
    }


def load_fixture(fixture_root: Path, seed: int) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    fixture_root = fixture_root.expanduser().resolve()
    report_path = fixture_root / "fixture_report.json"
    if not report_path.is_file():
        raise ValueError(f"Missing fixture report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("seed") != seed:
        raise ValueError(f"Fixture seed {report.get('seed')!r} does not match requested seed {seed}")
    if report.get("frame_size_wh") != list(FRAME_SIZE_WH) or report.get("frames_per_path") != FRAME_COUNT:
        raise ValueError("Fixture dimensions do not match the registered 160x96, 17-frame packet contract")
    if report.get("status") != "kubric_rendered_truth_scored":
        raise ValueError("Fixture report does not record a successfully rendered and truth-scored fixture")

    ground_truth_path = fixture_root / "ground_truth.json"
    if not ground_truth_path.is_file() or report.get("ground_truth_sha256") != sha256(ground_truth_path):
        raise ValueError("Fixture ground truth is missing or does not match its report hash")

    frame_hashes = report.get("rgb_frame_sha256")
    expected_frame_names = {
        f"{direction}/frames/frame_{frame_index:03d}.png"
        for direction in ("left", "right")
        for frame_index in range(FRAME_COUNT)
    }
    if not isinstance(frame_hashes, dict) or set(frame_hashes) != expected_frame_names:
        raise ValueError("Fixture report must bind all 34 expected RGB frame PNGs by relative path and SHA-256")

    paths: dict[str, list[Path]] = {}
    for direction in ("left", "right"):
        frame_dir = fixture_root / direction / "frames"
        frames = sorted(
            frame_dir.glob("frame_*.png"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        if len(frames) != FRAME_COUNT:
            raise ValueError(f"Expected {FRAME_COUNT} {direction} frames, found {len(frames)}")
        if [int(path.stem.rsplit("_", 1)[-1]) for path in frames] != list(range(FRAME_COUNT)):
            raise ValueError(f"{direction} frame indices must be exactly 0..{FRAME_COUNT - 1}")
        for path in frames:
            relative_path = path.relative_to(fixture_root).as_posix()
            expected_hash = frame_hashes[relative_path]
            if not isinstance(expected_hash, str) or sha256(path) != expected_hash.lower():
                raise ValueError(f"Fixture frame does not match the report hash: {relative_path}")
            if png_size_wh(path) != FRAME_SIZE_WH:
                raise ValueError(f"Fixture frame dimensions do not match {FRAME_SIZE_WH}: {relative_path}")
        paths[direction] = frames
    return report, paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--first-block-pass-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be a non-negative integer")
    started = time.perf_counter()

    fixture_root = args.fixture_dir or (RESEARCH / f"kubric_gate1_seed_block_{args.seed}")
    out_root = args.output_dir or (RESEARCH / f"gate1_seed_block_packets_{args.seed}")
    fixture_root = fixture_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()
    if out_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing packet directory: {out_root}")

    first_block = validate_first_block_pass(args.first_block_pass_record)
    fixture_report, fixture_paths = load_fixture(fixture_root, args.seed)
    out_root.mkdir(parents=True)
    packet_dir = out_root / "synthetic_reliability"
    packet_dir.mkdir()

    privacy_note = (
        "This packet is a local, offline annotation page. Extract the ZIP and open index.html in a modern browser. "
        "The page does not contact a server. Label every frame independently; do not interpolate boxes.\n\n"
        "For each frame independently record visibility, visible target-instance count, continuity with frame 0, and the "
        "tight visible-object box. Keep all fields independent. Box the visible silhouette, excluding shadow and detached "
        "glare. Use continuous pixel-edge coordinates; right and bottom edges are exclusive. If absent, explicitly choose "
        "No, enter count 0, mark identity No / uncertain, and leave the box empty. Do not look up or guess an experimental "
        "condition. Keep the two CSVs private until both raters finish.\n"
    )

    clip_ids = {direction: "clip_" + secrets.token_hex(5) for direction in ("left", "right")}
    frames: list[dict[str, Any]] = []
    for direction in ("left", "right"):
        blind_id = clip_ids[direction]
        for path in fixture_paths[direction]:
            frame_index = int(path.stem.rsplit("_", 1)[-1])
            frames.append(png_frame(path, blind_id, frame_index, f"{blind_id}/{path.name}"))

    for rater in (1, 2):
        ordered = list(frames)
        first_direction = "left" if rater == 1 else "right"
        ordered.sort(key=lambda frame: (
            frame["blind_clip_id"] != clip_ids[first_direction],
            frame["frame_index"],
        ))
        add_packet(
            packet_dir / f"rater_{rater:02d}.zip",
            ordered,
            f"gate1-seed-{args.seed}-rater-{rater}",
            privacy_note,
        )

    private_key = {
        "created_for": "Gate 1 second independently seeded mirrored block; keep private",
        "fixture_seed": args.seed,
        "fixture_report_sha256": sha256(fixture_root / "fixture_report.json"),
        "ground_truth_sha256": sha256(fixture_root / "ground_truth.json"),
        "first_block_pass_record_sha256": first_block["record_sha256"],
        "synthetic": {clip_ids[direction]: direction for direction in ("left", "right")},
    }
    (out_root / "PRIVATE_UNBLIND_KEY.json").write_text(
        json.dumps(private_key, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_root / "README.md").write_text(
        "# Gate 1 second-seed annotation packets\n\n"
        "This packet was built only after the first mirrored pair passed the pooled and per-clip screen and its blind "
        "worst-frame review was completed. Distribute each ZIP separately to an independent rater; never send "
        "`PRIVATE_UNBLIND_KEY.json`. The pair contains the two mirrored 17-frame clips from one separately seeded, "
        "known-truth Kubric render. Each ZIP uses the offline v5 annotation interface and blank CSV fallback.\n\n"
        "Label every frame independently. Keep visibility, instance count, identity, and box decisions independent. "
        "Do not interpolate boxes or discuss labels before both CSVs are returned. These labels measure reference "
        "reliability only; they are not video-diffusion or policy-efficacy evidence.\n",
        encoding="utf-8",
    )

    archives = sorted(out_root.rglob("*.zip"))
    manifest = {
        "packet_version": "gate1_second_seed_block_packets_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "build_wall_seconds": round(time.perf_counter() - started, 3),
        "build_host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "purpose": "second independent human annotation reliability block; no diffusion or vision inference",
        "gpu_used": False,
        "fixture_seed": args.seed,
        "fixture_report_sha256": sha256(fixture_root / "fixture_report.json"),
        "ground_truth_sha256": sha256(fixture_root / "ground_truth.json"),
        "fixture_report": str(fixture_root / "fixture_report.json"),
        "first_block_pass_record": first_block,
        "source": {
            "builder": "research/build_gate1_seed_block_packets.py",
            "builder_sha256": sha256(Path(__file__).resolve()),
            "interface_builder": "research/build_blinded_box_packets.py",
            "frame_size_wh": list(FRAME_SIZE_WH),
            "frames_per_direction": FRAME_COUNT,
            "first_block_agreement_screen_recomputed_from_bound_csvs": True,
            "first_block_blind_review_completion_recorded": True,
            "fixture_id": fixture_report.get("fixture_id"),
            "scene_parameters": fixture_report.get("scene_parameters"),
        },
        "archives": [
            {
                "path": str(path.relative_to(out_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in archives
        ],
        "label_fields_populated": 0,
        "limitations": [
            "This is a second synthetic annotation-reliability block, not policy-efficacy evidence.",
            "The two directions are a mirrored pair within one seeded scene; they are not independent scene seeds.",
            "The first block's gate record is a prerequisite and is bound by file hashes; keep its private key separate.",
        ],
    }
    (out_root / "packet_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out_root),
        "fixture_seed": args.seed,
        "first_block_pass_record_sha256": first_block["record_sha256"],
        "packet_archives": [item["path"] for item in manifest["archives"]],
        "gpu_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
