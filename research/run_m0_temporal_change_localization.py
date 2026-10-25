#!/usr/bin/env python3
"""Run the preregistered CPU-only M0 temporal-change localization readout."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "m0" / "runs" / "wan_t4_2026-10-02_retry5"
RUN_DIR = RUN_ROOT / "run_001"
RUN_MANIFEST = RUN_ROOT / "manifest.json"
FRAME_AUDIT = ROOT / "m0" / "analysis" / "wan_t4_retry5_frame_motion_audit.json"
TEMPORAL_AUDIT = ROOT / "research" / "m0_temporal_sampling_audit_v1.json"
OUTPUT = ROOT / "m0" / "analysis" / "m0_temporal_change_localization_v1.json"
FRAME_INDICES = [0, 4, 8, 12, 16]
POST_STEPS = [0, 4, 9, 14, 19, 24, 29]
HEIGHT, WIDTH = 480, 832
CONTROL_H, CONTROL_W = 30, 52
PIXELS_PER_TOKEN = 16
RIDGE_ALPHA = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive, dtype=bool).reshape(-1)
    total_positive = int(positive.sum())
    if total_positive == 0 or not np.isfinite(scores).all():
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_positive = positive[order].astype(np.float64)
    precision = np.cumsum(sorted_positive) / np.arange(1, len(scores) + 1)
    return float((precision * sorted_positive).sum() / total_positive)


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if left_ranks.std() <= 1e-12 or right_ranks.std() <= 1e-12:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def patch_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f"Unexpected image array dimensions: {values.shape}")
    if values.ndim == 2:
        return values.reshape(CONTROL_H, PIXELS_PER_TOKEN, CONTROL_W, PIXELS_PER_TOKEN).mean(axis=(1, 3))
    if values.ndim == 3 and values.shape[2] == 3:
        return values.reshape(
            CONTROL_H, PIXELS_PER_TOKEN, CONTROL_W, PIXELS_PER_TOKEN, 3
        ).mean(axis=(1, 3))
    raise ValueError(f"Unexpected image array rank/channels: {values.shape}")


def validate_mapping() -> dict[str, Any]:
    mapping = json.loads(TEMPORAL_AUDIT.read_text(encoding="utf-8"))
    generation = mapping["generation"]
    alignment = mapping["temporal_alignment"]
    expected_anchors = [int(value) for value in alignment["derived_nominal_rgb_frame_anchors"]]
    if generation["rgb_frames"] != 17 or generation["resolution_wh"] != [WIDTH, HEIGHT]:
        raise ValueError("Temporal audit dimensions disagree with the M0 source run")
    if generation["latent_shape_bcthw"] != [1, 16, 5, 60, 104]:
        raise ValueError("Temporal audit latent shape disagrees with the M0 source run")
    if not alignment["anchors_match"] or expected_anchors != FRAME_INDICES:
        raise ValueError("Nominal RGB anchors do not match the preregistered positions")
    if mapping["model_outcome_claim"] != "none; no model inference or chunked-generation comparison":
        raise ValueError("Unexpected temporal audit scope")
    return mapping


def load_proxy_frames(
    audit: dict[str, Any], manifest_hashes: dict[str, str]
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    audit_by_frame = {int(item["frame"]): item for item in audit["measurements"]["per_frame"]}
    rgb_frames: list[np.ndarray] = []
    red_masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for frame_idx in FRAME_INDICES:
        path = RUN_DIR / "frames" / f"frame_{frame_idx:03d}.png"
        relative = path.relative_to(RUN_ROOT).as_posix()
        actual_hash = sha256(path)
        expected_hash = manifest_hashes.get(relative)
        if expected_hash is None or actual_hash != expected_hash:
            raise ValueError(f"Manifest SHA-256 mismatch for {relative}: {actual_hash} vs {expected_hash}")
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Unexpected frame shape for {path}: {rgb.shape}")
        red = (rgb[..., 0] > 150) & (rgb[..., 1] < 125) & (rgb[..., 2] < 125)
        ys, xs = np.nonzero(red)
        if not len(xs):
            raise ValueError(f"Red-mask proxy is empty in frame {frame_idx}")
        expected = audit_by_frame[frame_idx]
        center = [float(xs.mean()), float(ys.mean())]
        count = int(red.sum())
        if count != int(expected["pixel_count"]):
            raise ValueError(
                f"Frame {frame_idx} red-mask count mismatch: {count} != {expected['pixel_count']}"
            )
        expected_center = expected["centroid_xy_pixels"]
        if max(abs(center[0] - expected_center[0]), abs(center[1] - expected_center[1])) > 0.011:
            raise ValueError(
                f"Frame {frame_idx} red-mask centroid mismatch: {center} != {expected_center}"
            )
        rgb_frames.append(rgb)
        red_masks.append(red)
        records.append({
            "decoded_frame_index": frame_idx,
            "red_mask_pixel_count": count,
            "red_mask_centroid_xy_pixels": center,
            "sha256": actual_hash,
        })

    transitions: list[dict[str, Any]] = []
    for index in range(len(FRAME_INDICES) - 1):
        red_xor = np.logical_xor(red_masks[index], red_masks[index + 1])
        red_union = np.logical_or(red_masks[index], red_masks[index + 1])
        rgb_delta = np.abs(rgb_frames[index + 1].astype(np.float32) - rgb_frames[index].astype(np.float32))
        rgb_delta_per_pixel = rgb_delta.mean(axis=2)
        transitions.append({
            "latent_time_pair": [index, index + 1],
            "decoded_frame_pair": [FRAME_INDICES[index], FRAME_INDICES[index + 1]],
            "red_xor_occupancy": patch_mean(red_xor.astype(np.float32)).astype(np.float64),
            "red_swept_occupancy": patch_mean(red_union.astype(np.float32)).astype(np.float64),
            "rgb_mean_absolute_difference": patch_mean(rgb_delta_per_pixel).astype(np.float64),
            "red_xor_changed_pixel_count": int(red_xor.sum()),
            "red_swept_pixel_count": int(red_union.sum()),
            "rgb_mean_absolute_difference_per_pixel": float(rgb_delta_per_pixel.mean()),
        })
    return rgb_frames, red_masks, records, transitions


def temporal_patch_features(tensor: torch.Tensor) -> np.ndarray:
    if list(tensor.shape) != [1, 16, 5, 60, 104]:
        raise ValueError(f"Unexpected latent shape: {list(tensor.shape)}")
    array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    if not np.isfinite(array).all():
        raise ValueError("Latent tensor contains non-finite values")
    transition_features: list[np.ndarray] = []
    for time_index in range(4):
        signed_delta = array[:, time_index + 1] - array[:, time_index]
        patches = signed_delta.reshape(16, CONTROL_H, 2, CONTROL_W, 2).transpose(1, 3, 0, 2, 4)
        transition_features.append(patches.reshape(CONTROL_H, CONTROL_W, 64).astype(np.float64))
    return np.stack(transition_features, axis=0)


def ridge_predictions(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    y_mean = float(train_y.mean())
    y_centered = train_y - y_mean
    gram = x_train.T @ x_train
    coefficients = np.linalg.solve(
        gram + RIDGE_ALPHA * np.eye(gram.shape[1]), x_train.T @ y_centered
    )
    return x_test @ coefficients + y_mean


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    denominator = float(np.asarray(weights, dtype=np.float64).sum())
    if denominator <= 1e-12:
        return None
    return float((np.asarray(values, dtype=np.float64) * weights).sum() / denominator)


def enrichment_record(energy: np.ndarray, weights: np.ndarray) -> dict[str, float | None]:
    inside = weighted_mean(energy, weights)
    outside = weighted_mean(energy, 1.0 - weights)
    ratio = None if inside is None or outside is None or outside <= 1e-12 else inside / outside
    return {"mean_energy_inside": inside, "mean_energy_outside": outside, "inside_to_outside_ratio": ratio}


def analyze_snapshot(
    step: int,
    path: Path,
    loaded: dict[str, Any],
    transitions: list[dict[str, Any]],
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    relative = path.relative_to(RUN_ROOT).as_posix()
    actual_hash = sha256(path)
    expected_hash = manifest_hashes.get(relative)
    if expected_hash is None or actual_hash != expected_hash:
        raise ValueError(f"Manifest SHA-256 mismatch for {relative}: {actual_hash} vs {expected_hash}")
    if int(loaded["step_zero_based"]) != step:
        raise ValueError(f"Step metadata mismatch in {path}")
    features = temporal_patch_features(loaded["latent_after_scheduler_step"])
    folds: list[dict[str, Any]] = []
    for held_out in range(4):
        train_pairs = [index for index in range(4) if index != held_out]
        train_x = features[train_pairs].reshape(-1, 64)
        train_y = np.stack([transitions[index]["red_xor_occupancy"] for index in train_pairs]).reshape(-1)
        test_x = features[held_out]
        target = transitions[held_out]["red_xor_occupancy"]
        predicted = ridge_predictions(train_x, train_y, test_x.reshape(-1, 64)).reshape(CONTROL_H, CONTROL_W)
        site_prior = np.mean([transitions[index]["red_xor_occupancy"] for index in train_pairs], axis=0)
        positive = target.reshape(-1) > 0.0
        target_flat = target.reshape(-1)
        prediction_flat = predicted.reshape(-1)
        prior_flat = site_prior.reshape(-1)
        energy = np.abs(features[held_out]).mean(axis=2)
        energy_flat = energy.reshape(-1)
        union = transitions[held_out]["red_swept_occupancy"]
        rgb_change = transitions[held_out]["rgb_mean_absolute_difference"]
        folds.append({
            "latent_time_pair": transitions[held_out]["latent_time_pair"],
            "decoded_frame_pair": transitions[held_out]["decoded_frame_pair"],
            "changed_red_patch_count": int(positive.sum()),
            "changed_red_patch_fraction": float(positive.mean()),
            "latent_readout": {
                "red_xor_average_precision": average_precision(prediction_flat, positive),
                "red_xor_fraction_rmse": float(np.sqrt(np.mean((prediction_flat - target_flat) ** 2))),
            },
            "spatial_site_prior": {
                "red_xor_average_precision": average_precision(prior_flat, positive),
                "red_xor_fraction_rmse": float(np.sqrt(np.mean((prior_flat - target_flat) ** 2))),
            },
            "absolute_latent_delta_energy": {
                "red_xor_average_precision": average_precision(energy_flat, positive),
                "spearman_with_patch_rgb_change": spearman_correlation(energy_flat, rgb_change.reshape(-1)),
                "red_xor_energy_enrichment": enrichment_record(energy, target),
                "red_swept_energy_enrichment": enrichment_record(energy, union),
            },
            "rgb_change_summary": {
                "mean_absolute_difference_per_pixel": transitions[held_out]["rgb_mean_absolute_difference_per_pixel"],
                "red_xor_changed_pixel_count": transitions[held_out]["red_xor_changed_pixel_count"],
                "red_swept_pixel_count": transitions[held_out]["red_swept_pixel_count"],
            },
        })

    readout_ap = [fold["latent_readout"]["red_xor_average_precision"] for fold in folds]
    prior_ap = [fold["spatial_site_prior"]["red_xor_average_precision"] for fold in folds]
    readout_rmse = [fold["latent_readout"]["red_xor_fraction_rmse"] for fold in folds]
    prior_rmse = [fold["spatial_site_prior"]["red_xor_fraction_rmse"] for fold in folds]
    return {
        "snapshot": f"post_step_{step:03d}",
        "source_file": relative,
        "source_sha256": actual_hash,
        "tensor_shape": list(loaded["latent_after_scheduler_step"].shape),
        "tensor_dtype": str(loaded["latent_after_scheduler_step"].dtype),
        "folds": folds,
        "within_clip_summary": {
            "transition_pair_count": 4,
            "mean_latent_readout_red_xor_ap": float(np.mean(readout_ap)),
            "mean_spatial_site_prior_red_xor_ap": float(np.mean(prior_ap)),
            "mean_ap_difference_latent_minus_prior": float(np.mean(readout_ap) - np.mean(prior_ap)),
            "latent_readout_wins_vs_prior_transition_count": int(sum(a > b for a, b in zip(readout_ap, prior_ap))),
            "mean_latent_readout_red_xor_rmse": float(np.mean(readout_rmse)),
            "mean_spatial_site_prior_red_xor_rmse": float(np.mean(prior_rmse)),
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUTPUT}")
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    manifest_hashes = manifest["files_sha256"]
    config_path = RUN_ROOT / "config_snapshot.yaml"
    config_hash = sha256(config_path)
    if config_hash != manifest_hashes.get("config_snapshot.yaml"):
        raise ValueError("Manifest SHA-256 mismatch for config_snapshot.yaml")
    frame_audit = json.loads(FRAME_AUDIT.read_text(encoding="utf-8"))
    temporal_mapping = validate_mapping()
    _, _, proxy_records, transitions = load_proxy_frames(frame_audit, manifest_hashes)

    results: list[dict[str, Any]] = []
    latent_paths: list[Path] = []
    for step in POST_STEPS:
        path = RUN_DIR / "latents" / f"post_step_{step:03d}.pt"
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        results.append(analyze_snapshot(step, path, loaded, transitions, manifest_hashes))
        latent_paths.append(path)

    frame_paths = [RUN_DIR / "frames" / f"frame_{index:03d}.png" for index in FRAME_INDICES]
    used_inputs = [
        RUN_MANIFEST,
        config_path,
        FRAME_AUDIT,
        TEMPORAL_AUDIT,
        *frame_paths,
        *latent_paths,
    ]
    payload = {
        "experiment_id": "EXP-LATENTGATE-M0-TEMPORAL-CHANGE-LOCALIZATION-001",
        "protocol": "research/2026-10-12_m0_temporal_change_localization.md",
        "analysis_code": "research/run_m0_temporal_change_localization.py",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execution": {
            "device": "cpu",
            "model_weights_loaded": False,
            "new_inference_or_generation": False,
            "human_labels_used": False,
            "ridge_alpha": RIDGE_ALPHA,
        },
        "source_run": {
            "path": "m0/runs/wan_t4_2026-10-02_retry5",
            "manifest_sha256": sha256(RUN_MANIFEST),
            "frame_motion_audit_sha256": sha256(FRAME_AUDIT),
            "temporal_sampling_audit_sha256": sha256(TEMPORAL_AUDIT),
            "latent_shape_bcthw": [1, 16, 5, 60, 104],
            "control_grid_wh": [CONTROL_W, CONTROL_H],
            "decoded_frame_shape_hwc": [HEIGHT, WIDTH, 3],
            "frame_to_latent_time_indices": FRAME_INDICES,
            "nominal_rgb_anchor_mapping_verified": temporal_mapping["temporal_alignment"]["anchors_match"],
            "frame_pair_stride": 4,
        },
        "input_file_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in used_inputs
        },
        "proxy_frame_records": proxy_records,
        "transition_records": [
            {
                "latent_time_pair": item["latent_time_pair"],
                "decoded_frame_pair": item["decoded_frame_pair"],
                "red_xor_changed_pixel_count": item["red_xor_changed_pixel_count"],
                "red_swept_pixel_count": item["red_swept_pixel_count"],
                "rgb_mean_absolute_difference_per_pixel": item["rgb_mean_absolute_difference_per_pixel"],
            }
            for item in transitions
        ],
        "analysis_spec": {
            "feature": "signed adjacent-time latent difference; flattened 2x2 spatial block across 16 channels into 64 values",
            "primary_target": "fractional red-proxy XOR occupancy in corresponding 16x16 RGB patches",
            "secondary_target": "mean absolute RGB difference per pixel in corresponding 16x16 patches",
            "cross_validation": "leave one of four adjacent latent-time pairs out; train on all spatial tokens from the other three pairs",
            "readout": "train-fold standardized ridge regression, intercept, alpha=1.0",
            "spatial_baseline": "training-pair mean red-XOR occupancy at each spatial token",
            "positive_patch_rule": "red-XOR occupancy greater than zero",
            "direct_energy": "mean absolute value of the 64 signed-difference features",
            "sample_count_boundary": "one generated clip; four correlated transition pairs; denoising snapshots are repeated views of the same generation",
        },
        "limitations": [
            "Single generated clip, prompt, and seed; four transition folds are correlated and are not independent samples.",
            "The red RGB threshold and RGB differences are same-output proxies, not human labels or ground truth.",
            "Red-mask XOR can include threshold shimmer; RGB change also includes lighting and appearance changes.",
            "The nominal latent/RGB anchor mapping does not define exact VAE temporal receptive fields.",
            "This readout does not establish object understanding, forecasting, geometry/contact, causal control, policy efficacy, generalization, or M1/M2/M3 completion.",
        ],
        "snapshots": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "snapshots": [
            {"snapshot": item["snapshot"], **item["within_clip_summary"]} for item in results
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
