#!/usr/bin/env python3
"""Read-only integrity and measurement audit for the Gate 1 Kubric fixture.

Checks that saved RGB/mask files, per-frame JSON truth, the cached RGB hashes,
mask-derived boxes, projected centers, and basic path/size records still agree.
This does not inspect annotation packets or human labels and cannot pass Gate 1.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent / "kubric_gate0_fixture_v1"
REPORT_OUT = Path(__file__).resolve().parent / "gate1_fixture_artifact_alignment_audit_v1.json"
DIRECTIONS = ("right", "left")
FRAME_COUNT = 17
FRAME_SIZE = (160, 96)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def component_count_8(mask: np.ndarray) -> int:
    """Count 8-connected foreground components without adding dependencies."""
    ys, xs = np.nonzero(mask)
    remaining = set(zip(xs.tolist(), ys.tolist()))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return count


def cv(values: list[float]) -> float | None:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else None


def main() -> None:
    started = time.perf_counter()
    truth_path = ROOT / "ground_truth.json"
    cached_path = ROOT / "fixture_report.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    cached = json.loads(cached_path.read_text(encoding="utf-8"))
    # ``refresh_fixture_report`` intentionally rebuilds scores from masks and
    # does not preserve the renderer's original per-RGB-file hash map. Keep the
    # report hash as provenance, but compare frame hashes only if that map is
    # present in this cached report.
    cached_rgb_hashes = cached.get("rgb_frame_sha256")
    errors: list[str] = []
    inventory: list[dict[str, Any]] = []
    arms: dict[str, Any] = {}
    expected_rgb_paths: set[str] = set()
    expected_mask_paths: set[str] = set()
    projected_gaps_px: list[float] = []

    for direction in DIRECTIONS:
        rows = sorted(truth[direction], key=lambda row: int(row["frame_index"]))
        if len(rows) != FRAME_COUNT or [int(r["frame_index"]) for r in rows] != list(range(FRAME_COUNT)):
            errors.append(f"{direction}: expected one truth row for each frame 0..{FRAME_COUNT - 1}")

        frame_sizes: list[list[int]] = []
        box_areas: list[int] = []
        mask_areas: list[int] = []
        component_counts: list[int] = []
        mask_sizes: list[list[int]] = []
        center_errors: list[float] = []
        reference_rgb: np.ndarray | None = None
        reference_mask: np.ndarray | None = None
        background_change_fractions: list[float] = []
        background_change_mae: list[float] = []

        for row in rows:
            frame_index = int(row["frame_index"])
            rgb_rel = str(row["rgb_file"])
            mask_rel = str(row["mask_file"])
            expected_rgb_paths.add(rgb_rel)
            expected_mask_paths.add(mask_rel)
            rgb_path = (ROOT / rgb_rel).resolve()
            mask_path = (ROOT / mask_rel).resolve()
            if not rgb_path.is_relative_to(ROOT.resolve()) or not mask_path.is_relative_to(ROOT.resolve()):
                errors.append(f"{direction}/{frame_index}: a truth path escapes the fixture root")
                continue
            if not rgb_path.is_file() or not mask_path.is_file():
                errors.append(f"{direction}/{frame_index}: missing RGB or mask artifact")
                continue

            with Image.open(rgb_path) as im:
                rgb_mode = im.mode
                rgb_size = list(im.size)
                rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
            with Image.open(mask_path) as im:
                mask_mode = im.mode
                mask_size = list(im.size)
                mask_values = np.asarray(im.convert("L"), dtype=np.uint8)
            if tuple(rgb_size) != FRAME_SIZE or tuple(mask_size) != FRAME_SIZE:
                errors.append(f"{direction}/{frame_index}: RGB/mask resolution is not {FRAME_SIZE}")
            if rgb_mode != "RGBA" or mask_mode != "L":
                errors.append(f"{direction}/{frame_index}: unexpected PNG modes {rgb_mode}/{mask_mode}")
            unique_mask_values = np.unique(mask_values).tolist()
            if not set(unique_mask_values).issubset({0, 255}):
                errors.append(f"{direction}/{frame_index}: mask is not binary 0/255")
            mask = mask_values > 0
            pixel_count = int(mask.sum())
            box = bbox_xyxy(mask)
            components = component_count_8(mask)
            saved_truth_box = row.get("mask_bbox_xyxy_px")
            ann = row.get("annotation_from_mask", {})
            if pixel_count != int(row.get("mask_pixel_count", -1)):
                errors.append(f"{direction}/{frame_index}: saved mask pixel count differs from truth JSON")
            if box != saved_truth_box or box != ann.get("bbox_xyxy_px"):
                errors.append(f"{direction}/{frame_index}: mask-derived box differs from truth/annotation JSON")
            if components != int(ann.get("instance_count", -1)):
                errors.append(f"{direction}/{frame_index}: connected components differ from truth instance count")
            if bool(box) != bool(ann.get("object_visible")):
                errors.append(f"{direction}/{frame_index}: mask visibility differs from truth annotation")
            if components != 1 or not bool(ann.get("identity_consistent")):
                errors.append(f"{direction}/{frame_index}: fixture target is not one connected identity")
            if box is None:
                errors.append(f"{direction}/{frame_index}: target mask is empty")
                continue

            x0, y0, x1, y1 = box
            width, height = x1 - x0, y1 - y0
            frame_sizes.append([width, height])
            box_areas.append(width * height)
            mask_areas.append(pixel_count)
            component_counts.append(components)

            center_norm = row["projected_target_center_xy_normalized"]
            projected_xy = [float(center_norm[0]) * FRAME_SIZE[0], float(center_norm[1]) * FRAME_SIZE[1]]
            box_center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
            gap = float(np.linalg.norm(np.asarray(projected_xy) - np.asarray(box_center)))
            projected_gaps_px.append(gap)
            center_errors.append(gap)

            rel_rgb_hash = sha256(rgb_path)
            cached_hash = cached_rgb_hashes.get(rgb_rel) if cached_rgb_hashes else None
            if cached_rgb_hashes is not None and cached_hash != rel_rgb_hash:
                errors.append(f"{direction}/{frame_index}: current RGB hash differs from cached fixture report")
            inventory.append({
                "direction": direction,
                "frame_index": frame_index,
                "rgb_file": rgb_rel,
                "rgb_sha256": rel_rgb_hash,
                "mask_file": mask_rel,
                "mask_sha256": sha256(mask_path),
                "rgb_mode": rgb_mode,
                "mask_mode": mask_mode,
                "resolution_wh": rgb_size,
                "mask_unique_values": unique_mask_values,
                "mask_pixel_count": pixel_count,
                "mask_bbox_xyxy_px": box,
                "connected_components_8": components,
                "projected_center_xy_px": projected_xy,
                "mask_bbox_center_xy_px": box_center,
                "projected_to_mask_box_center_px": gap,
            })

            if reference_rgb is None:
                reference_rgb, reference_mask = rgb, mask
            else:
                assert reference_mask is not None
                # Exclude the union of target masks; shadow or illumination changes remain visible here.
                shared_background = ~(reference_mask | mask)
                channel_delta = np.abs(rgb.astype(np.int16) - reference_rgb.astype(np.int16))
                changed = np.any(channel_delta != 0, axis=2) & shared_background
                denom = int(shared_background.sum())
                background_change_fractions.append(float(changed.sum() / denom) if denom else 0.0)
                background_change_mae.append(float(channel_delta[shared_background].mean()) if denom else 0.0)

        arms[direction] = {
            "frame_count": len(rows),
            "mask_connected_components": {
                "min": min(component_counts) if component_counts else None,
                "max": max(component_counts) if component_counts else None,
            },
            "mask_bbox_width_px": {"min": min((v[0] for v in frame_sizes), default=None), "max": max((v[0] for v in frame_sizes), default=None)},
            "mask_bbox_height_px": {"min": min((v[1] for v in frame_sizes), default=None), "max": max((v[1] for v in frame_sizes), default=None)},
            "mask_bbox_area_cv": cv([float(v) for v in box_areas]) if box_areas else None,
            "mask_pixel_area_cv": cv([float(v) for v in mask_areas]) if mask_areas else None,
            "projected_to_mask_bbox_center_error_px": {
                "mean": statistics.fmean(center_errors) if center_errors else None,
                "p95": float(np.quantile(center_errors, 0.95, method="linear")) if center_errors else None,
                "max": max(center_errors) if center_errors else None,
            },
            "outside_target_union_rgb_change_vs_frame0": {
                "scope": "exact RGB changes outside the union of reference/current object masks; includes moving shadow effects",
                "fraction_changed_mean": statistics.fmean(background_change_fractions) if background_change_fractions else None,
                "fraction_changed_max": max(background_change_fractions) if background_change_fractions else None,
                "mean_absolute_channel_delta_mean": statistics.fmean(background_change_mae) if background_change_mae else None,
            },
        }

    actual_rgb = {
        str(p.relative_to(ROOT)) for p in ROOT.glob("*/frames/frame_*.png")
    }
    actual_masks = {
        str(p.relative_to(ROOT)) for p in ROOT.glob("*/masks/mask_*.png")
    }
    if actual_rgb != expected_rgb_paths:
        errors.append("actual RGB inventory differs from paths indexed by ground-truth JSON")
    if actual_masks != expected_mask_paths:
        errors.append("actual mask inventory differs from paths indexed by ground-truth JSON")
    if cached_rgb_hashes is not None and set(cached_rgb_hashes) != expected_rgb_paths:
        errors.append("cached RGB hash-map keys differ from ground-truth RGB paths")

    gap_values = projected_gaps_px
    result = {
        "audit_id": "EXP-LATENTGATE-GATE1-FIXTURE-ARTIFACT-ALIGNMENT-001",
        "status": "pass" if not errors else "fail",
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "saved Kubric fixture file/JSON alignment and source-mask measurements only",
        "inputs": {
            "fixture_root": str(ROOT.relative_to(ROOT.parent)),
            "ground_truth_sha256": sha256(truth_path),
            "fixture_report_sha256": sha256(cached_path),
        },
        "counts": {
            "directions": len(DIRECTIONS),
            "truth_rows": sum(len(truth[d]) for d in DIRECTIONS),
            "rgb_files": len(actual_rgb),
            "mask_files": len(actual_masks),
            "aligned_frame_records": len(inventory),
            "cached_rgb_hashes_present": cached_rgb_hashes is not None,
            "cached_rgb_hashes_matched": (
                sum(item["rgb_sha256"] == cached_rgb_hashes.get(item["rgb_file"]) for item in inventory)
                if cached_rgb_hashes is not None else None
            ),
        },
        "mask_bbox_center_vs_projected_center_px": {
            "mean": statistics.fmean(gap_values) if gap_values else None,
            "p95": float(np.quantile(gap_values, 0.95, method="linear")) if gap_values else None,
            "max": max(gap_values) if gap_values else None,
        },
        "per_direction": arms,
        "frame_inventory": inventory,
        "errors": errors,
        "limitations": [
            "This is not a human-label agreement result and cannot pass Gate 1.",
            "The fixture was generated by a single pinned synthetic renderer; this does not establish annotation accuracy on generated video.",
            "Outside-target RGB change is descriptive and includes moving shadows; it is not a camera-motion estimator.",
            "No new rendering, model inference, GPU, cloud runtime, or external data was used.",
        ],
        "compute": {
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "gpu_used": False,
            "peak_vram_mb": None,
        },
    }
    REPORT_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "frame_inventory"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
