#!/usr/bin/env python3
"""Inventory saved M0/Wan and CogVideoX RGB sequences; no model inference."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research/m0_saved_rgb_sequence_inventory_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indexed_pngs(frames_dir: Path) -> list[tuple[int, Path]]:
    indexed = []
    for path in frames_dir.glob("frame_*.png"):
        match = re.fullmatch(r"frame_(\d+)\.png", path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    return sorted(indexed)


def summarize_event_windows(anchors: list[int], frame_count: int, max_duration: int = 4) -> list[dict]:
    anchor_set = set(anchors)
    out = []
    for duration in range(1, max_duration + 1):
        possible = frame_count - duration + 1
        seen_starts = [
            start for start in range(possible)
            if any(frame in anchor_set for frame in range(start, start + duration))
        ]
        missed_starts = sorted(set(range(possible)) - set(seen_starts))
        out.append({
            "event_duration_frames": duration,
            "possible_start_positions": possible,
            "intersecting_anchor_count": len(seen_starts),
            "missed_start_count": len(missed_starts),
            "missed_start_positions_zero_based": missed_starts,
        })
    return out


def scoreability(rgb_frames: list[np.ndarray], thresholds: dict, minimum_pixels: int) -> dict:
    cases = []
    all_counts: dict[tuple[int, int, int], list[int]] = {}
    for r_gt, g_lt, b_lt in product(
        thresholds["red_greater_than"],
        thresholds["green_less_than"],
        thresholds["blue_less_than"],
    ):
        counts = []
        for rgb in rgb_frames:
            mask = (
                (rgb[:, :, 0] > r_gt)
                & (rgb[:, :, 1] < g_lt)
                & (rgb[:, :, 2] < b_lt)
            )
            counts.append(int(np.count_nonzero(mask)))
        all_counts[(r_gt, g_lt, b_lt)] = counts
        cases.append({
            "threshold": {"r_gt": r_gt, "g_lt": g_lt, "b_lt": b_lt},
            "scoreable_frame_count": sum(value >= minimum_pixels for value in counts),
            "minimum_mask_pixels": min(counts),
            "maximum_mask_pixels": max(counts),
            "all_frames_scoreable": all(value >= minimum_pixels for value in counts),
        })

    baseline = (150, 125, 125)
    baseline_counts = all_counts[baseline]
    maximum_by_frame = [
        max(counts[frame_index] for counts in all_counts.values())
        for frame_index in range(len(rgb_frames))
    ]
    return {
        "rule": "R > r_gt AND G < g_lt AND B < b_lt",
        "minimum_mask_pixels": minimum_pixels,
        "baseline_threshold": {"r_gt": 150, "g_lt": 125, "b_lt": 125},
        "baseline_scoreable_frame_count": sum(value >= minimum_pixels for value in baseline_counts),
        "baseline_minimum_mask_pixels": min(baseline_counts),
        "baseline_maximum_mask_pixels": max(baseline_counts),
        "baseline_per_frame_mask_pixels": baseline_counts,
        "threshold_grid_maximum_mask_pixels_any_frame": max(maximum_by_frame),
        "threshold_grid_maximum_mask_pixels_by_frame": maximum_by_frame,
        "threshold_case_count": len(cases),
        "full_sequence_scoreable_threshold_case_count": sum(case["all_frames_scoreable"] for case in cases),
        "per_threshold_summary": cases,
    }


def main() -> None:
    trajectory = json.loads((ROOT / "research/trajectory_control_v1.json").read_text())
    sensitivity = json.loads((ROOT / "research/m0_mask_threshold_sensitivity_v1.json").read_text())
    threshold_cfg = sensitivity["inputs"]["rgb_threshold_grid"]
    min_pixels = sensitivity["inputs"]["minimum_mask_pixels"]
    wan_thresholds = {
        "red_greater_than": threshold_cfg["red_greater_than"],
        "green_less_than": threshold_cfg["green_less_than"],
        "blue_less_than": threshold_cfg["blue_less_than"],
    }

    wan_specs = []
    for retry in ("retry4", "retry5"):
        manifest_path = ROOT / f"m0/runs/wan_t4_2026-10-02_{retry}/manifest.json"
        packet_path = manifest_path.parent / "generation_packet.json"
        manifest = json.loads(manifest_path.read_text())
        packet = json.loads(packet_path.read_text())
        generation = packet["generation"]
        files = manifest["files_sha256"]
        frame_hashes = {
            Path(name).name: digest
            for name, digest in files.items()
            if re.fullmatch(r"run_001/frames/frame_\d{3}\.png", name)
        }
        wan_specs.append({
            "run_id": f"wan_t4_2026-10-02_{retry}",
            "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "manifest_path": str(manifest_path),
            "generation_packet_path": str(packet_path),
            "frames_dir": str(manifest_path.parent / "run_001/frames"),
            "manifest_sha256": sha256(manifest_path),
            "generation_packet_sha256": sha256(packet_path),
            "prompt": generation["prompt"],
            "seed": generation["seed"],
            "expected_frame_count": 17,
            "expected_size_wh": [832, 480],
            "manifest_frame_hashes": frame_hashes,
            "run_status": manifest.get("status"),
            "m0_assessment": manifest.get("m0_completion_assessment", {}).get("status"),
            "step_count": 30,
            "anchor_schedule": [0, 4, 8, 12, 16],
            "anchor_schedule_source": "research/m0_temporal_sampling_audit_v1.json and research/m0_temporal_downsampling_probe_v1.json",
            "integrity_pin_source": "manifest files_sha256",
        })

    cog_root = Path("/Users/yash/Documents/Resume/video-diffusion-lab/run")
    proxy_pin = json.loads((ROOT / "research/cogvideox2b_rgb_proxy_inputs_v1.json").read_text())
    step_pin = json.loads((ROOT / "research/cogvideox2b_stepsensitivity_inputs_v1.json").read_text())
    step_runs = {row["run_id"]: row for row in step_pin["runs"]}
    cog_specs = []
    for run_id, anchor_schedule, pin in (
        ("cogvideox_2b_chunk13", [0, 4, 8, 12, 15], proxy_pin),
        ("cogvideox_2b_pilot", [0, 2, 4, 6, 8], step_runs["cogvideox_2b_pilot"]),
        ("cogvideox_2b_pilot_30steps", [0, 2, 4, 6, 8], step_runs["cogvideox_2b_pilot_30steps"]),
    ):
        manifest_path = cog_root / run_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if run_id == "cogvideox_2b_chunk13":
            frame_hashes = pin["frames"]
            pin_manifest_hash = pin["manifest_sha256"]
            expected_size = pin["frame_size_px"]
            expected_count = pin["frame_count"]
        else:
            frame_hashes = {frame["name"]: frame["sha256"] for frame in pin["frames"]}
            pin_manifest_hash = pin["manifest_sha256"]
            expected_size = pin["frames"][0]["size_px"]
            expected_count = pin["frame_count"]
        cog_specs.append({
            "run_id": run_id,
            "model": manifest["model"],
            "manifest_path": str(manifest_path),
            "frames_dir": str(manifest_path.parent / "frames"),
            "manifest_sha256": sha256(manifest_path),
            "expected_manifest_sha256": pin_manifest_hash,
            "expected_frame_count": expected_count,
            "expected_size_wh": expected_size,
            "expected_frame_hashes": frame_hashes,
            "run_status": "saved_output",
            "step_count": manifest["denoising_steps"],
            "prompt": manifest["prompt"],
            "seed": manifest["seed"],
            "anchor_schedule": anchor_schedule,
            "anchor_schedule_source": (
                "research/cogvideox2b_50step_five_anchor_preregistration_v1.json"
                if run_id == "cogvideox_2b_chunk13"
                else "research/cogvideox_2b_sparse_frame_preregistration_v1.json"
            ),
            "integrity_pin_source": "existing preregistered RGB input map",
        })

    specs = wan_specs + cog_specs
    runs = []
    for spec in specs:
        manifest_path = Path(spec["manifest_path"])
        frames_dir = Path(spec["frames_dir"])
        manifest = json.loads(manifest_path.read_text())
        indexed = indexed_pngs(frames_dir)
        names = [path.name for _, path in indexed]
        expected_names = [f"frame_{idx:0{len(names[0].split('_')[1].split('.')[0])}d}.png" for idx in range(spec["expected_frame_count"])] if names else []
        sequential = [index for index, _ in indexed] == list(range(spec["expected_frame_count"]))

        rgb_frames = []
        image_records = []
        dimensions_ok = True
        for index, path in indexed:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            dims = [int(rgb.shape[1]), int(rgb.shape[0])]
            dimensions_ok = dimensions_ok and dims == spec["expected_size_wh"]
            rgb_frames.append(rgb)
            image_records.append({
                "frame_index": index,
                "file": path.name,
                "sha256": sha256(path),
                "size_wh": dims,
                "readable_rgb": True,
            })

        actual_hashes = {row["file"]: row["sha256"] for row in image_records}
        expected_hashes = spec.get("manifest_frame_hashes", spec.get("expected_frame_hashes", {}))
        manifest_hash_matches_prior_pin = (
            spec["manifest_sha256"] == spec["expected_manifest_sha256"]
            if "expected_manifest_sha256" in spec else None
        )
        frame_hashes_match_source = actual_hashes == expected_hashes
        complete = (
            len(indexed) == spec["expected_frame_count"]
            and sequential
            and dimensions_ok
            and all(row["readable_rgb"] for row in image_records)
            and frame_hashes_match_source
            and manifest_hash_matches_prior_pin is not False
        )
        proxy = scoreability(rgb_frames, wan_thresholds, min_pixels)
        run = {
            "run_id": spec["run_id"],
            "model": spec["model"],
            "manifest_path": spec["manifest_path"],
            "manifest_sha256": spec["manifest_sha256"],
            "manifest_sha256_matches_prior_input_pin": manifest_hash_matches_prior_pin,
            "run_status": spec["run_status"],
            "m0_assessment": spec.get("m0_assessment"),
            "step_count": spec["step_count"],
            "prompt": spec.get("prompt"),
            "seed": spec.get("seed"),
            "declared_frame_count": spec["expected_frame_count"],
            "available_frame_count": len(indexed),
            "frame_dimensions_wh": spec["expected_size_wh"],
            "sequential_filenames": sequential and names == expected_names,
            "all_frames_readable_and_dimension_correct": dimensions_ok and len(image_records) == len(indexed),
            "all_frame_hashes_match_manifest_or_input_map": frame_hashes_match_source,
            "complete_hash_matched_rgb_sequence": complete,
            "rgb_sequence_content_sha256": hashlib.sha256(
                json.dumps([row["sha256"] for row in image_records], separators=(",", ":")).encode()
            ).hexdigest(),
            "frame_records": image_records,
            "fixed_red_proxy_scoreability": proxy,
            "sparse_anchor_schedule": {
                "anchor_indices_zero_based": spec["anchor_schedule"],
                "source": spec["anchor_schedule_source"],
                "short_event_window_coverage": summarize_event_windows(
                    spec["anchor_schedule"], spec["expected_frame_count"]
                ),
            },
            "integrity_pin_source": spec["integrity_pin_source"],
        }
        runs.append(run)

    content_groups: dict[str, list[str]] = {}
    for run in runs:
        content_groups.setdefault(run["rgb_sequence_content_sha256"], []).append(run["run_id"])
    unique_content_id = {content_hash: idx + 1 for idx, content_hash in enumerate(sorted(content_groups))}
    for run in runs:
        run["unique_sequence_id"] = f"RGBSEQ-{unique_content_id[run['rgb_sequence_content_sha256']]:02d}"
        run["duplicate_run_ids_same_exact_rgb_sequence"] = content_groups[run["rgb_sequence_content_sha256"]]

    result = {
        "experiment_id": "EXP-M0-SAVED-RGB-SEQUENCE-INVENTORY-001",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "completed_cpu_only_inventory",
        "scope": "Read-only inventory and fixed red-mask scoreability of saved RGB PNGs; no interpolation reanalysis, latent inspection, inference, GPU, cloud runtime, or new generation.",
        "proxy_configuration": {
            "threshold_grid_source": "research/m0_mask_threshold_sensitivity_v1.json",
            "baseline_source": "research/trajectory_control_v1.json",
            "threshold_case_count": 36,
            "minimum_mask_pixels": min_pixels,
            "baseline_threshold": trajectory["mask_proxy"],
        },
        "run_count": len(runs),
        "unique_rgb_sequence_count": len(content_groups),
        "runs": runs,
        "matched_comparison_assessment": {
            "informative_existing_matched_multiclip_comparison_available": False,
            "nominal_pair": "CogVideoX-2B 8-step and 30-step outputs share recorded prompt, seed, resolution, and frame count, but both fail the fixed proxy full-track gate; checkpoint/revision, run code, and effective initial-noise provenance are absent.",
            "other_confounders": "CogVideoX 50-step output differs in frame count and resolution. Wan retry4/retry5 are exact RGB replay duplicates, not independent clips; retry4 has incomplete instrumentation. Wan and CogVideoX use different backbones/settings.",
            "motion_claim_allowed": False,
        },
        "smallest_informative_followup": {
            "design": "After the current saved-Wan two-rater annotation screen and fresh free-T4 admission, run two additional unique seeds under the exact pinned M0 Wan2.1 T2V 1.3B configuration (same prompt, 832x480, 17 frames, 30 steps, guidance 5.0, scheduler and revision fixed). Together with the existing seed-2026 clip this gives three unique full sequences for a small exploratory robustness screen.",
            "generation_unit": "Jointly denoise one complete 17-frame sequence per seed; save/hash all decoded frames and run manifests.",
            "primary_readout": "Score every decoded frame with blinded human visibility/count/identity and tight-box-midpoint labels; apply the existing agreement/review gates unchanged. Report metrics by clip/seed, not by pooling frames as independent samples.",
            "secondary_readout": "On those same complete per-frame labels, calculate the existing five-anchor schedule as a secondary sparse-summary diagnostic; do not omit frames from primary scoring.",
            "interpretation": "Exploratory only; it cannot satisfy or replace the registered observer-release gate (three development blocks followed by five fresh held-out blocks, with its existing annotation and observer thresholds). Do not include a chunked arm absent measured resource/control-horizon pressure.",
            "compute_estimate": "Two additional M0-like generations are about 320 seconds (5.3 minutes) of generation-only T4 time at the measured 160.02 seconds per baseline run, excluding setup, model download, labeling, and artifact I/O. Requires fresh free-runtime admission; no paid compute.",
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "run_count": result["run_count"],
        "unique_rgb_sequence_count": result["unique_rgb_sequence_count"],
        "runs": [{
            "run_id": row["run_id"],
            "complete": row["complete_hash_matched_rgb_sequence"],
            "frames": row["available_frame_count"],
            "baseline_scoreable": f"{row['fixed_red_proxy_scoreability']['baseline_scoreable_frame_count']}/{row['declared_frame_count']}",
            "grid_full": f"{row['fixed_red_proxy_scoreability']['full_sequence_scoreable_threshold_case_count']}/{row['fixed_red_proxy_scoreability']['threshold_case_count']}",
            "unique": row["unique_sequence_id"],
        } for row in runs],
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
