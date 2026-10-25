#!/usr/bin/env python3
"""Run the preregistered CPU-only five-anchor RGB audit for two CogVideoX pilots.

Only manifests, the preregistration, pinned proxy sources, and PNG RGB frames are
read. This script does not inspect latent files or run model inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "research/cogvideox_2b_sparse_frame_preregistration_v1.json"
OUTPUT = ROOT / "research/cogvideox_2b_five_anchor_readout_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(registration: dict) -> list[dict]:
    proxy = registration["proxy"]
    source_entries = [
        (proxy["code_path"], proxy["code_sha256"]),
        (proxy["fixed_threshold_config_path"], proxy["fixed_threshold_config_sha256"]),
        (proxy["threshold_grid_path"], proxy["threshold_grid_sha256"]),
    ]
    checks = []
    for rel_path, expected in source_entries:
        path = ROOT / rel_path
        actual = sha256(path) if path.is_file() else None
        checks.append(
            {
                "path": rel_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    return checks


def validate_runs(registration: dict) -> tuple[list[dict], dict[str, list[Path]]]:
    checks = []
    frame_paths_by_run: dict[str, list[Path]] = {}
    for run in registration["runs"]:
        run_dir = Path(run["path"])
        manifest_path = run_dir / "manifest.json"
        manifest_actual_hash = sha256(manifest_path) if manifest_path.is_file() else None
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        field_checks = {
            key: {
                "expected": expected,
                "actual": manifest.get(key),
                "pass": manifest.get(key) == expected,
            }
            for key, expected in run["expected_manifest_fields"].items()
        }
        expected_names = [f"frame_{i:02d}.png" for i in range(9)]
        paths = sorted((run_dir / "frames").glob("frame_*.png"))
        actual_names = [path.name for path in paths]
        names_pass = actual_names == expected_names
        frame_checks = []
        for name, expected_hash in run["rgb_frame_sha256"].items():
            path = run_dir / "frames" / name
            if not path.is_file():
                frame_checks.append(
                    {
                        "name": name,
                        "expected_sha256": expected_hash,
                        "actual_sha256": None,
                        "pass": False,
                        "failure": "missing_file",
                    }
                )
                continue
            actual_hash = sha256(path)
            try:
                with Image.open(path) as image:
                    rgb_size = image.convert("RGB").size
                    image_format = image.format
            except Exception as exc:  # record decode failure, then stop before metrics
                rgb_size, image_format = None, None
                decode_error = str(exc)
            else:
                decode_error = None
            frame_checks.append(
                {
                    "name": name,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "sha256_pass": actual_hash == expected_hash,
                    "rgb_size_wh": list(rgb_size) if rgb_size else None,
                    "expected_rgb_size_wh": [480, 320],
                    "size_pass": rgb_size == (480, 320),
                    "format": image_format,
                    "decode_error": decode_error,
                    "pass": actual_hash == expected_hash
                    and rgb_size == (480, 320)
                    and decode_error is None,
                }
            )
        expected_frame_hash_count = len(run["rgb_frame_sha256"]) == 9
        passed = (
            manifest_actual_hash == run["manifest_sha256"]
            and all(item["pass"] for item in field_checks.values())
            and names_pass
            and expected_frame_hash_count
            and len(frame_checks) == 9
            and all(item["pass"] for item in frame_checks)
        )
        checks.append(
            {
                "run_name": run["run_name"],
                "manifest_sha256_expected": run["manifest_sha256"],
                "manifest_sha256_actual": manifest_actual_hash,
                "manifest_hash_pass": manifest_actual_hash == run["manifest_sha256"],
                "manifest_field_checks": field_checks,
                "rgb_filenames_expected": expected_names,
                "rgb_filenames_actual": actual_names,
                "rgb_filenames_pass": names_pass,
                "expected_frame_hash_count_pass": expected_frame_hash_count,
                "rgb_frame_checks": frame_checks,
                "pass": passed,
            }
        )
        frame_paths_by_run[run["run_name"]] = [
            run_dir / "frames" / name for name in expected_names
        ]
    return checks, frame_paths_by_run


def make_trace(frame_paths: list[Path], threshold: tuple[int, int, int]) -> list[dict]:
    r_gt, g_lt, b_lt = threshold
    trace = []
    for path in frame_paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        mask = (
            (rgb[:, :, 0] > r_gt)
            & (rgb[:, :, 1] < g_lt)
            & (rgb[:, :, 2] < b_lt)
        )
        ys, xs = np.nonzero(mask)
        trace.append(
            {
                "pixel_count": int(len(xs)),
                "center_xy_px": (
                    [float(xs.mean()), float(ys.mean())] if len(xs) else None
                ),
            }
        )
    return trace


def interpolate_scalar(trace: list[dict], field: str, frame: int, left: int, right: int):
    a, b = trace[left][field], trace[right][field]
    if a is None or b is None:
        return None
    t = (frame - left) / (right - left)
    if isinstance(a, list):
        return [a[index] + t * (b[index] - a[index]) for index in range(len(a))]
    return a + t * (b - a)


def metrics_for_trace(trace: list[dict], anchors: list[int], omitted: list[int]) -> dict:
    center_errors = []
    area_abs_errors = []
    area_relative_errors = []
    for frame in omitted:
        left = max(anchor for anchor in anchors if anchor < frame)
        right = min(anchor for anchor in anchors if anchor > frame)
        center = trace[frame]["center_xy_px"]
        center_hat = interpolate_scalar(trace, "center_xy_px", frame, left, right)
        if center is None or center_hat is None:
            raise ValueError(f"Unexpected missing center at omitted frame {frame}")
        center_errors.append(math.hypot(center[0] - center_hat[0], center[1] - center_hat[1]))
        area_hat = interpolate_scalar(trace, "pixel_count", frame, left, right)
        area_error = abs(trace[frame]["pixel_count"] - area_hat)
        area_abs_errors.append(area_error)
        area_relative_errors.append(100 * area_error / trace[frame]["pixel_count"])
    return {
        "omitted_frame_center_errors_px": [round(value, 6) for value in center_errors],
        "center_mae_px": round(mean(center_errors), 6),
        "center_rmse_px": round(math.sqrt(mean(value * value for value in center_errors)), 6),
        "center_max_error_px": round(max(center_errors), 6),
        "omitted_frame_area_absolute_errors_px": [round(value, 6) for value in area_abs_errors],
        "area_mae_px": round(mean(area_abs_errors), 6),
        "omitted_frame_area_relative_errors_pct": [round(value, 6) for value in area_relative_errors],
        "area_mape_pct": round(mean(area_relative_errors), 6),
        "area_max_relative_error_pct": round(max(area_relative_errors), 6),
    }


def event_coverage(frame_count: int, anchors: list[int]) -> list[dict]:
    result = []
    for duration in range(1, 5):
        possible_starts = frame_count - duration + 1
        covered = sum(
            any(start <= anchor < start + duration for anchor in anchors)
            for start in range(possible_starts)
        )
        result.append(
            {
                "duration_frames": duration,
                "possible_start_positions": possible_starts,
                "anchor_intersecting_positions": covered,
                "coverage_pct": round(100 * covered / possible_starts, 6),
            }
        )
    return result


def main() -> int:
    registration = json.loads(PREREG.read_text())
    result = {
        "experiment_id": registration["experiment_id"],
        "preregistration_path": str(PREREG.relative_to(ROOT)),
        "preregistration_sha256": sha256(PREREG),
        "status": "validating_inputs_before_rgb_metric_computation",
        "source_integrity_checks": [],
        "run_integrity_checks": [],
        "proxy_coverage_checks": [],
        "run_results": [],
        "event_coverage_schedule_only": None,
        "compute": {
            "device": "local CPU",
            "model_inference": False,
            "gpu_used": False,
            "latent_files_read": False,
            "human_labels_used": False,
            "new_generation": False,
        },
    }

    source_checks = validate_sources(registration)
    result["source_integrity_checks"] = source_checks
    if not all(check["pass"] for check in source_checks):
        result["status"] = "stopped_source_integrity_failure_no_rgb_metrics"
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2

    run_checks, frame_paths_by_run = validate_runs(registration)
    result["run_integrity_checks"] = run_checks
    if not all(check["pass"] for check in run_checks):
        result["status"] = "stopped_input_integrity_failure_no_rgb_metrics"
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2

    proxy = registration["proxy"]
    baseline_threshold = (
        proxy["baseline_rgb_threshold"]["r_greater_than"],
        proxy["baseline_rgb_threshold"]["g_less_than"],
        proxy["baseline_rgb_threshold"]["b_less_than"],
    )
    threshold_grid = proxy["threshold_grid"]
    thresholds = [
        (r, g, b)
        for r in threshold_grid["r_greater_than"]
        for g in threshold_grid["g_less_than"]
        for b in threshold_grid["b_less_than"]
    ]
    if len(thresholds) != threshold_grid["case_count"]:
        raise ValueError("Preregistered threshold grid case count is inconsistent")
    min_pixels = proxy["minimum_mask_pixels"]

    traces_by_run: dict[str, dict[tuple[int, int, int], list[dict]]] = {}
    coverage_failures = []
    coverage_summaries = []
    for run in registration["runs"]:
        name = run["run_name"]
        traces_by_run[name] = {}
        for threshold in thresholds:
            trace = make_trace(frame_paths_by_run[name], threshold)
            traces_by_run[name][threshold] = trace
            failed_frames = [
                index for index, frame in enumerate(trace)
                if frame["pixel_count"] < min_pixels
            ]
            coverage_summaries.append(
                {
                    "run_name": name,
                    "threshold": {
                        "r_gt": threshold[0],
                        "g_lt": threshold[1],
                        "b_lt": threshold[2],
                    },
                    "scoreable_frames": 9 - len(failed_frames),
                    "total_frames": 9,
                    "minimum_mask_pixels_observed": min(
                        frame["pixel_count"] for frame in trace
                    ),
                    "failed_frames": failed_frames,
                    "pass": not failed_frames,
                }
            )
            if failed_frames:
                coverage_failures.append(coverage_summaries[-1])

    result["proxy_coverage_checks"] = coverage_summaries
    if coverage_failures:
        result["status"] = "stopped_proxy_coverage_failure_no_interpolation_metrics"
        result["proxy_coverage_failures"] = coverage_failures
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 2

    anchors = registration["method"]["anchors"]
    omitted = registration["method"]["omitted_frames"]
    baseline_results = []
    sweep_results = []
    for run in registration["runs"]:
        name = run["run_name"]
        threshold_results = []
        for threshold in thresholds:
            metrics = metrics_for_trace(traces_by_run[name][threshold], anchors, omitted)
            case = {
                "threshold": {"r_gt": threshold[0], "g_lt": threshold[1], "b_lt": threshold[2]},
                **metrics,
            }
            threshold_results.append(case)
            if threshold == baseline_threshold:
                baseline_results.append({"run_name": name, **case})

        def range_metric(key: str) -> dict:
            values = [case[key] for case in threshold_results]
            return {
                "min": round(min(values), 6),
                "median": round(median(values), 6),
                "max": round(max(values), 6),
            }

        sweep_results.append(
            {
                "run_name": name,
                "threshold_case_count": len(threshold_results),
                "center_mae_px_across_thresholds": range_metric("center_mae_px"),
                "center_max_error_px_across_thresholds": range_metric("center_max_error_px"),
                "area_mape_pct_across_thresholds": range_metric("area_mape_pct"),
                "area_max_relative_error_pct_across_thresholds": range_metric("area_max_relative_error_pct"),
                "per_threshold_results": threshold_results,
            }
        )

    result.update(
        {
            "status": "completed_descriptive_cpu_rgb_audit",
            "baseline_threshold_results": baseline_results,
            "threshold_sensitivity_results": sweep_results,
            "event_coverage_schedule_only": event_coverage(9, anchors),
            "interpretation_limits": registration["comparison_limits"],
        }
    )
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "baseline_threshold_results": baseline_results,
                "threshold_sensitivity_ranges": [
                    {key: value for key, value in run.items() if key != "per_threshold_results"}
                    for run in sweep_results
                ],
                "event_coverage_schedule_only": result["event_coverage_schedule_only"],
                "report_path": str(OUTPUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
