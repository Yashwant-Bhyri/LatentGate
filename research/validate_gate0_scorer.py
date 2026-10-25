#!/usr/bin/env python3
"""CPU-only known-truth checks for the Gate 0 manual trajectory scorer.

The generated raster clips are a lightweight fallback fixture, not Kubric
renders. The output explicitly retains that limitation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from path_annotation_scoring import (  # noqa: E402
    ANNOTATION_COLUMNS,
    evaluate_annotations,
    interpolate_target_boxes,
    make_blinded_annotation_package,
    read_annotation_csv,
)
from kubric_gate0_fixture_v1.scoring import (  # noqa: E402
    recompute_fixture_scores,
)


WIDTH, HEIGHT, FRAME_COUNT = 200, 100, 17
BOX_W, BOX_H = 20.0, 16.0
Y_CENTER = HEIGHT / 2


def target_boxes(start_x: float, end_x: float) -> list[list[float]]:
    boxes = []
    for index in range(FRAME_COUNT):
        alpha = index / (FRAME_COUNT - 1)
        cx = start_x + alpha * (end_x - start_x)
        boxes.append([
            (cx - BOX_W / 2) / WIDTH,
            (Y_CENTER - BOX_H / 2) / HEIGHT,
            (cx + BOX_W / 2) / WIDTH,
            (Y_CENTER + BOX_H / 2) / HEIGHT,
        ])
    return boxes


def annotation_rows(boxes: list[list[float]], width: int = WIDTH, height: int = HEIGHT) -> list[dict[str, Any]]:
    rows = []
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rows.append({
            "frame_index": index,
            "object_visible": True,
            "instance_count": 1,
            "identity_consistent": True,
            "bbox_xyxy_px": [x1 * width, y1 * height, x2 * width, y2 * height],
        })
    return rows


def render_clip(boxes: list[list[float]], directory: Path) -> None:
    directory.mkdir(parents=True)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        image = Image.new("RGB", (WIDTH, HEIGHT), (185, 185, 185))
        draw = ImageDraw.Draw(image)
        bounds = [round(x1 * WIDTH), round(y1 * HEIGHT), round(x2 * WIDTH), round(y2 * HEIGHT)]
        draw.ellipse(bounds, fill=(230, 35, 25))
        image.save(directory / f"frame_{index:03d}.png")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def compact_score(report: dict[str, Any]) -> dict[str, Any]:
    """Keep aggregate validation evidence without duplicating frame records."""
    fields = (
        "mean_center_error_frame_diagonals_missing_penalized",
        "fde_frame_diagonals_missing_penalized",
        "scoreable_identity_coverage",
        "mean_bbox_iou_all_frames_missing_as_zero",
        "target_direction_sign_x",
        "endpoint_delta_x_frame_diagonals",
        "target_aligned_endpoint_progress_frame_diagonals",
        "center_x_monotonic_step_fraction_target_direction",
        "center_x_monotonic_valid_adjacent_pair_count",
        "missing_or_identity_ambiguous_frames",
        "identity_failure_frames",
        "duplicate_instance_frames",
        "hard_fail",
        "hard_fail_reasons",
    )
    return {field: report[field] for field in fields}


def main() -> None:
    right_targets = target_boxes(100, 140)
    left_targets = target_boxes(100, 60)
    right_rows = annotation_rows(right_targets)
    left_rows = annotation_rows(left_targets)

    right = evaluate_annotations(
        right_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
        target_boxes_xyxy_normalized=right_targets,
    )
    left = evaluate_annotations(
        left_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
        target_boxes_xyxy_normalized=left_targets,
    )
    check(right["mean_center_error_frame_diagonals_missing_penalized"] == 0.0, "Exact right path should score zero ADE")
    check(left["mean_center_error_frame_diagonals_missing_penalized"] == 0.0, "Exact left path should score zero ADE")
    check(right["endpoint_delta_x_frame_diagonals"] > 0, "Rightward path must have positive signed x displacement")
    check(left["endpoint_delta_x_frame_diagonals"] < 0, "Leftward path must have negative signed x displacement")
    check(right["center_x_monotonic_valid_adjacent_pair_count"] == FRAME_COUNT - 1, "Complete path denominator should be 16 adjacent pairs")
    check(right["center_x_monotonic_step_fraction_target_direction"] == 1.0, "Exact right path should be monotonic")

    missing_rows = [dict(row) for row in right_rows]
    missing_rows[8].update(object_visible=False, instance_count=0, identity_consistent=False, bbox_xyxy_px=None)
    missing = evaluate_annotations(
        missing_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
        target_boxes_xyxy_normalized=right_targets,
    )
    check(math.isclose(missing["mean_center_error_frame_diagonals_missing_penalized"], 1 / FRAME_COUNT), "Missing center must receive a 1.0-diagonal penalty")
    check(math.isclose(missing["scoreable_identity_coverage"], 16 / FRAME_COUNT), "Missing frame must lower scoreable coverage")
    check(missing["center_x_monotonic_valid_adjacent_pair_count"] == 14, "A missing middle frame must remove both neighboring pairs")
    check(missing["missing_or_identity_ambiguous_frames"] == [8], "Missing frame must be reported explicitly")

    endpoint_missing_rows = [dict(row) for row in right_rows]
    endpoint_missing_rows[-1].update(object_visible=False, instance_count=0, identity_consistent=False, bbox_xyxy_px=None)
    endpoint_missing = evaluate_annotations(
        endpoint_missing_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
        target_boxes_xyxy_normalized=right_targets,
    )
    check(endpoint_missing["fde_frame_diagonals_missing_penalized"] == 1.0, "Missing final object must receive the maximum FDE penalty")

    duplicate_rows = [dict(row) for row in right_rows]
    duplicate_rows[5]["instance_count"] = 2
    duplicate = evaluate_annotations(
        duplicate_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
        target_boxes_xyxy_normalized=right_targets,
    )
    check(duplicate["duplicate_instance_frames"] == [5] and duplicate["duplicate_frame_rate"] == 1 / FRAME_COUNT, "Duplicate object must be flagged")
    check(duplicate["hard_fail"] is True, "A duplicate instance must fail the scorer gate")
    check(duplicate["hard_fail_reasons"] == ["duplicate_instances"], "Duplicate failure reason must be explicit")
    check(duplicate["scoreable_identity_coverage"] == 16 / FRAME_COUNT, "Duplicate frame must be unscoreable")
    check(duplicate["mean_center_error_frame_diagonals_valid_only"] == 0.0, "Duplicate center must be excluded from valid path error")
    check(math.isclose(duplicate["mean_center_error_frame_diagonals_missing_penalized"], 1 / FRAME_COUNT), "Duplicate frame must receive the missing-frame penalty")
    check(duplicate["center_x_monotonic_valid_adjacent_pair_count"] == 14, "Duplicate frame must remove both adjacent path pairs")
    check(duplicate["records"][5]["scoreable"] is False and duplicate["records"][5]["center_error_frame_diagonals"] == 1.0, "Duplicate record must not contribute its ambiguous box center")

    switched_rows = [dict(row) for row in right_rows]
    switched_rows[6]["identity_consistent"] = False
    switched = evaluate_annotations(
        switched_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
        target_boxes_xyxy_normalized=right_targets,
    )
    check(switched["identity_failure_frames"] == [6], "Identity switch must be reported")
    check(math.isclose(switched["mean_center_error_frame_diagonals_missing_penalized"], 1 / FRAME_COUNT), "Identity-ambiguous frame must receive missing penalty")

    shifted_rows = [dict(row) for row in right_rows]
    for row in shifted_rows:
        row["bbox_xyxy_px"] = [row["bbox_xyxy_px"][0] + 10, row["bbox_xyxy_px"][1], row["bbox_xyxy_px"][2] + 10, row["bbox_xyxy_px"][3]]
    shifted = evaluate_annotations(
        shifted_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
        target_boxes_xyxy_normalized=right_targets,
    )
    scaled_rows = [dict(row) for row in shifted_rows]
    for row in scaled_rows:
        row["bbox_xyxy_px"] = [value * 2 for value in row["bbox_xyxy_px"]]
    scaled = evaluate_annotations(
        scaled_rows, frame_count=FRAME_COUNT, width=WIDTH * 2, height=HEIGHT * 2,
        target_boxes_xyxy_normalized=right_targets,
    )
    check(math.isclose(
        shifted["mean_center_error_frame_diagonals_missing_penalized"],
        scaled["mean_center_error_frame_diagonals_missing_penalized"],
        rel_tol=1e-12,
    ), "Diagonal normalization must be resolution invariant")

    interpolated = interpolate_target_boxes([[0, 0.42, 0.58, 0.45, 0.55], [16, 0.42, 0.58, 0.65, 0.75]], FRAME_COUNT)
    check(len(interpolated) == FRAME_COUNT and interpolated[0][0] == 0.45 and interpolated[-1][0] == 0.65, "Path normalization/interpolation endpoints must be preserved")

    package_report: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="latentgate_gate0_") as tmp:
        temp = Path(tmp)
        right_dir, left_dir = temp / "right_secret_arm", temp / "left_secret_arm"
        render_clip(right_targets, right_dir)
        render_clip(left_targets, left_dir)
        archive_path, key_path = make_blinded_annotation_package(
            {"right_secret_arm": right_dir, "left_secret_arm": left_dir},
            temp / "blind_out",
            frame_count=FRAME_COUNT,
        )
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            csv_text = archive.read("annotation_template.csv").decode("utf-8")
            annotation_instructions = archive.read("README.txt").decode("utf-8")
            reader = csv.DictReader(csv_text.splitlines())
            csv_rows = list(reader)
        key = json.loads(key_path.read_text(encoding="utf-8"))
        clip_dirs = {Path(name).parts[0] for name in names if name.endswith(".png")}
        check("PRIVATE_UNBLIND_KEY.json" not in names, "Unblinding key must not be inside package")
        check("right_secret_arm" not in "\n".join(names + [csv_text]), "Right-arm label leaked into annotation package")
        check("left_secret_arm" not in "\n".join(names + [csv_text]), "Left-arm label leaked into annotation package")
        check(len(clip_dirs) == 2 and set(key) == clip_dirs, "Blind IDs must map to both arms in the external key")
        check(len(csv_rows) == 2 * FRAME_COUNT, "Annotation template must include every frame for each clip")
        check(ANNOTATION_COLUMNS == list(csv.DictReader([csv_text.splitlines()[0]]).fieldnames or []), "Structured annotation schema must be present")
        check("midpoint of this box" in annotation_instructions, "Annotation guide must define center as the box midpoint")
        check("x2/y2 are the right/bottom edges" in annotation_instructions, "Annotation guide must define box-edge coordinates")
        right_blind_id = next(blind_id for blind_id, arm in key.items() if arm == "right_secret_arm")
        completed_csv = temp / "completed_synthetic_annotations.csv"
        with completed_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ANNOTATION_COLUMNS)
            writer.writeheader()
            for row in csv_rows:
                if row["blind_clip_id"] == right_blind_id:
                    box = right_targets[int(row["frame_index"])]
                    x1, y1, x2, y2 = box[0] * WIDTH, box[1] * HEIGHT, box[2] * WIDTH, box[3] * HEIGHT
                    row.update({
                        "object_visible": "true",
                        "instance_count": "1",
                        "identity_consistent": "true",
                        "bbox_x1_px": str(x1),
                        "bbox_y1_px": str(y1),
                        "bbox_x2_px": str(x2),
                        "bbox_y2_px": str(y2),
                    })
                writer.writerow(row)
        csv_roundtrip_rows = read_annotation_csv(completed_csv, right_blind_id)
        csv_roundtrip = evaluate_annotations(
            csv_roundtrip_rows, frame_count=FRAME_COUNT, width=WIDTH, height=HEIGHT,
            target_boxes_xyxy_normalized=right_targets,
        )
        check(csv_roundtrip["mean_center_error_frame_diagonals_missing_penalized"] == 0.0, "Filled blinded CSV must round-trip through the scorer")
        package_report = {
            "zip_entry_count": len(names),
            "clips": len(clip_dirs),
            "annotation_rows": len(csv_rows),
            "structured_columns": ANNOTATION_COLUMNS,
            "key_outside_zip": key_path.name not in names,
            "arm_labels_absent_from_zip_and_template": True,
            "completed_csv_roundtrip_zero_error": csv_roundtrip["mean_center_error_frame_diagonals_missing_penalized"] == 0.0,
        }

    checks = {
        "exact_right_path_zero_error": True,
        "exact_left_path_zero_error": True,
        "signed_left_right_endpoint_displacement": True,
        "target_path_interpolation": True,
        "missing_frame_penalty_and_coverage": True,
        "missing_frame_adjacent_pair_denominator": True,
        "missing_final_frame_fde_penalty": True,
        "duplicate_instance_detection": True,
        "duplicate_instances_hard_fail_and_are_excluded_from_path_centers": True,
        "identity_switch_penalty": True,
        "frame_diagonal_normalization_resolution_invariant": True,
        "blinded_annotation_package_schema_and_key_separation": True,
        "annotation_box_midpoint_and_edge_coordinate_contract": True,
        "filled_annotation_csv_roundtrip": True,
    }
    counterfactual_spec = json.loads(
        (ROOT / "research/path_counterfactual_spec_v1.json").read_text(encoding="utf-8")
    )
    check(
        counterfactual_spec["annotation_schema"] == ANNOTATION_COLUMNS,
        "Counterfactual spec annotation columns must exactly match the scorer template",
    )
    checks["counterfactual_spec_matches_scorer_annotation_schema"] = True
    reliability_package_manifest_path = ROOT / "research/kubric_gate0_annotation_reliability/package_manifest.json"
    reliability_package_manifest = None
    reliability_package_summary = None
    if reliability_package_manifest_path.is_file():
        reliability_package_manifest = json.loads(
            reliability_package_manifest_path.read_text(encoding="utf-8")
        )
        for item in reliability_package_manifest["packages"]:
            archive_path = ROOT / item["package"]
            key_path = ROOT / item["private_unblind_key"]
            with zipfile.ZipFile(archive_path) as archive:
                archive_names = archive.namelist()
                template = archive.read("annotation_template.csv").decode("utf-8")
                guide = archive.read("README.txt").decode("utf-8")
                template_rows = list(csv.DictReader(template.splitlines()))
                check("PRIVATE_UNBLIND_KEY.json" not in archive_names, "Reliability package must keep key external")
                check(len(template_rows) == FRAME_COUNT, "Reliability package must include every fixture frame")
                check(ANNOTATION_COLUMNS == list(csv.DictReader([template.splitlines()[0]]).fieldnames or []), "Reliability package schema must match scorer")
                check("midpoint of this box" in guide, "Reliability package guide must define the scored center")
                check("x2/y2 are the right/bottom edges" in guide, "Reliability package guide must define coordinate edges")
                check("review_copy_" not in "\n".join(archive_names + [template]), "Reviewer slot label leaked into package")
            check(key_path.is_file(), "Reliability package must have an external key file")
            check(len(json.loads(key_path.read_text(encoding="utf-8"))) == 1, "Each reviewer key must map one opaque clip ID")
        checks["two_blind_kubric_reliability_packages_ready"] = True
        reliability_package_summary = {
            "manifest": "research/kubric_gate0_annotation_reliability/package_manifest.json",
            "status": reliability_package_manifest["status"],
            "independent_copies": len(reliability_package_manifest["packages"]),
            "frames_per_copy": reliability_package_manifest["frame_count"],
            "unblind_keys_external": True,
        }

    kubric_fixture_root = ROOT / "research/kubric_gate0_fixture_v1"
    kubric_fixture_path = kubric_fixture_root / "fixture_report.json"
    kubric_ground_truth_path = kubric_fixture_root / "ground_truth.json"
    kubric_fixture_report = None
    kubric_fixture_summary = None
    if kubric_ground_truth_path.is_file():
        if not kubric_fixture_path.is_file():
            raise FileNotFoundError(f"Kubric fixture metadata report is missing: {kubric_fixture_path}")
        kubric_fixture_report = json.loads(kubric_fixture_path.read_text(encoding="utf-8"))
        source_score_data = recompute_fixture_scores(kubric_fixture_root)
        kubric_path_scores = source_score_data["path_scores"]
        cached_path_scores = kubric_fixture_report["path_scores"]
        cache_fields = (
            "mean_center_error_frame_diagonals_valid_only",
            "mean_center_error_frame_diagonals_missing_penalized",
            "fde_frame_diagonals_missing_penalized",
            "scoreable_identity_coverage",
            "mean_bbox_iou_all_frames_missing_as_zero",
            "target_direction_sign_x",
            "endpoint_delta_x_frame_diagonals",
            "center_x_monotonic_step_fraction_target_direction",
            "center_x_monotonic_valid_adjacent_pair_count",
            "duplicate_instance_frames",
            "hard_fail",
            "hard_fail_reasons",
        )
        for arm, expected_sign in (("right", 1), ("left", -1)):
            arm_score = kubric_path_scores[arm]["scorer_report"]
            check(arm_score["target_direction_sign_x"] == expected_sign, f"Kubric {arm} target sign must match its arm")
            check(arm_score["scoreable_identity_coverage"] == 1.0, f"Kubric {arm} mask must cover all frames")
            check(arm_score["hard_fail"] is False, f"Kubric {arm} mask must contain exactly one connected object")
            check(arm_score["center_x_monotonic_step_fraction_target_direction"] == 1.0, f"Kubric {arm} mask centers must be monotonic")
            check(arm_score["center_x_monotonic_valid_adjacent_pair_count"] == FRAME_COUNT - 1, f"Kubric {arm} must have 16 valid adjacent pairs")
            check(arm_score["mean_center_error_frame_diagonals_missing_penalized"] < 0.03, f"Kubric {arm} mask centers must match projected truth")
            check(arm_score["mean_bbox_iou_all_frames_missing_as_zero"] >= 0.99, f"Kubric {arm} size-matched target boxes must align with source masks")
            cached_score = cached_path_scores[arm]["scorer_report"]
            for field in cache_fields:
                actual, cached = arm_score[field], cached_score[field]
                if isinstance(actual, (int, float)) and not isinstance(actual, bool):
                    check(math.isclose(actual, cached, rel_tol=1e-12, abs_tol=1e-12), f"Cached Kubric {arm} {field} must match the mask-recomputed score")
                else:
                    check(actual == cached, f"Cached Kubric {arm} {field} must match the mask-recomputed score")
        check(kubric_fixture_report["target_box_size_px"] == source_score_data["target_box_size_px"], "Cached target box size must match the source-mask scoring contract")
        check(kubric_fixture_report["score_source"] == source_score_data["score_source"], "Fixture report must identify source-mask score provenance")
        check(kubric_fixture_report["metric_scope"] == source_score_data["metric_scope"], "Fixture report must preserve the source-derived metric scope")
        checks["kubric_rendered_masks_match_projected_left_right_truth"] = True
        checks["kubric_scores_recomputed_from_source_masks_and_cache_checked"] = True
        kubric_fixture_summary = {
            "report": "research/kubric_gate0_fixture_v1/fixture_report.json",
            "score_source": source_score_data["score_source"],
            "target_box_size_px": source_score_data["target_box_size_px"],
            "metric_scope": source_score_data["metric_scope"],
            "kubric_source_commit_sha": kubric_fixture_report["kubric_source_commit_sha"],
            "python": kubric_fixture_report["python"],
            "blender_python": kubric_fixture_report["blender_python"],
            "runtime_packages": kubric_fixture_report["runtime_packages"],
            "total_rendered_frames": kubric_fixture_report["total_rendered_frames"],
            "path_scores_recomputed_from_masks": {
                arm: compact_score(kubric_path_scores[arm]["scorer_report"])
                for arm in ("right", "left")
            },
        }

    status = (
        "kubric_rendered_truth_scored_annotation_reliability_pending"
        if kubric_fixture_report
        else "local_scoring_contract_passed_kubric_render_pending"
    )
    remaining_gate0_requirements = [
        "Double-label a small random frame subset to estimate human annotation noise before using the template on generated clips."
    ]
    if not kubric_fixture_summary:
        remaining_gate0_requirements.insert(
            0,
            "Run the exact same scoring contract over a rendered Kubric clip with its ground-truth masks/boxes and point centers.",
        )

    report = {
        "experiment_id": "EXP-LATENTGATE-GATE0-SCORER-VALIDATION-001",
        "status": status,
        "fixture": "deterministic PIL scorer fixtures plus paired 17-frame Kubric/Blender left/right mask-grounded path renders when the Kubric fixture report is present",
        "cpu_only": True,
        "checks": checks,
        "scored_cases": {
            "right_exact": compact_score(right),
            "left_exact": compact_score(left),
            "middle_missing": compact_score(missing),
            "final_missing": {k: endpoint_missing[k] for k in ("scoreable_identity_coverage", "fde_frame_diagonals_missing_penalized", "missing_or_identity_ambiguous_frames")},
            "duplicate_at_5": {k: duplicate[k] for k in ("duplicate_instance_frames", "duplicate_frame_rate", "hard_fail", "hard_fail_reasons", "scoreable_identity_coverage", "mean_center_error_frame_diagonals_valid_only", "mean_center_error_frame_diagonals_missing_penalized")},
            "identity_switch_at_6": {k: switched[k] for k in ("identity_failure_frames", "scoreable_identity_coverage", "mean_center_error_frame_diagonals_missing_penalized")},
            "normalization_probe": {
                "200x100_error": shifted["mean_center_error_frame_diagonals_missing_penalized"],
                "400x200_error": scaled["mean_center_error_frame_diagonals_missing_penalized"],
            },
        },
        "kubric_fixture": kubric_fixture_summary,
        "annotation_reliability_packages": reliability_package_summary,
        "blinding_package": package_report,
        "environment": {
            "kubric_in_default_python": importlib.util.find_spec("kubric") is not None,
            "blender_cli": shutil.which("blender") is not None,
            "kubric_fixture_renderer": kubric_fixture_report["blender_python"] if kubric_fixture_report else None,
            "docker_cli": shutil.which("docker") is not None,
        },
        "remaining_gate0_requirements": remaining_gate0_requirements,
        "limits": [
            "The procedural PIL fixture verifies score math and blinding mechanics; the Kubric fixture recomputes its center and size-matched box scores from binary source masks plus projected ground-truth centers, but does not measure human annotation reliability.",
            "No Wan or DiTraj inference was run, and no GPU/cloud compute was used.",
        ],
    }
    output = ROOT / "research/gate0_scorer_validation_v1.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "checks": len(report["checks"])}, indent=2))


if __name__ == "__main__":
    main()
