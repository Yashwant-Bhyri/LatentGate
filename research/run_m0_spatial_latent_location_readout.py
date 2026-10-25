#!/usr/bin/env python3
"""Run the preregistered CPU-only M0 spatial-latent location readout."""

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
OUTPUT = ROOT / "m0" / "analysis" / "m0_spatial_latent_location_readout_v1.json"
FRAME_INDICES = [0, 4, 8, 12, 16]
POST_STEPS = [0, 4, 9, 14, 19, 24, 29]
CLEAN_STEPS = [4, 14, 24, 29]
HEIGHT, WIDTH = 480, 832
CONTROL_H, CONTROL_W = 30, 52
PATCH_PIXELS = 16
RIDGE_ALPHA = 1.0
OCCUPANCY_THRESHOLD = 0.25


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


def weighted_center(weights: np.ndarray) -> list[float] | None:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return None
    yy, xx = np.indices(weights.shape, dtype=np.float64)
    cx = float(((xx + 0.5) * weights).sum() / total)
    cy = float(((yy + 0.5) * weights).sum() / total)
    return [cx, cy]


def load_proxy_labels(
    audit: dict[str, Any], manifest_hashes: dict[str, str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    occupancies: list[np.ndarray] = []
    audit_by_frame = {int(item["frame"]): item for item in audit["measurements"]["per_frame"]}
    observed: list[dict[str, Any]] = []
    for frame_idx in FRAME_INDICES:
        frame_path = RUN_DIR / "frames" / f"frame_{frame_idx:03d}.png"
        rel_path = frame_path.relative_to(RUN_ROOT).as_posix()
        actual_hash = sha256(frame_path)
        expected_hash = manifest_hashes.get(rel_path)
        if expected_hash is None or actual_hash != expected_hash:
            raise ValueError(f"Manifest SHA-256 mismatch for {rel_path}: {actual_hash} vs {expected_hash}")
        with Image.open(frame_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Unexpected frame shape for {frame_path}: {rgb.shape}")
        red = (rgb[..., 0] > 150) & (rgb[..., 1] < 125) & (rgb[..., 2] < 125)
        ys, xs = np.nonzero(red)
        if not len(xs):
            raise ValueError(f"Red-mask proxy is empty in frame {frame_idx}")
        expected = audit_by_frame[frame_idx]
        center = [float(xs.mean()), float(ys.mean())]
        count = int(red.sum())
        if count != int(expected["pixel_count"]):
            raise ValueError(
                f"Frame {frame_idx} proxy count mismatch: {count} != {expected['pixel_count']}"
            )
        expected_center = expected["centroid_xy_pixels"]
        if max(abs(center[0] - expected_center[0]), abs(center[1] - expected_center[1])) > 0.011:
            raise ValueError(
                f"Frame {frame_idx} proxy centroid mismatch: {center} != {expected_center}"
            )
        occupancy = red.reshape(CONTROL_H, PATCH_PIXELS, CONTROL_W, PATCH_PIXELS).mean(axis=(1, 3))
        occupancies.append(occupancy.astype(np.float64))
        observed.append({
            "decoded_frame_index": frame_idx,
            "red_mask_pixel_count": count,
            "red_mask_centroid_xy_pixels": center,
            "target_center_xy_control_cells": [
                float((xs + 0.5).mean() / PATCH_PIXELS),
                float((ys + 0.5).mean() / PATCH_PIXELS),
            ],
            "proxy_sha256": actual_hash,
        })
    return np.stack(occupancies, axis=0), observed


def patch_features(tensor: torch.Tensor) -> np.ndarray:
    if list(tensor.shape) != [1, 16, 5, 60, 104]:
        raise ValueError(f"Unexpected latent shape: {list(tensor.shape)}")
    array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()[0]
    if not np.isfinite(array).all():
        raise ValueError("Latent contains non-finite values")
    # [C,T,H,W] -> [T,30,52,C,2,2] -> 64 values per model token.
    patches = array.reshape(16, 5, 30, 2, 52, 2).transpose(1, 2, 4, 0, 3, 5)
    return patches.reshape(5, CONTROL_H, CONTROL_W, 64).astype(np.float64)


def ridge_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, float]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    y_mean = float(train_y.mean())
    y_centered = train_y - y_mean
    gram = x_train.T @ x_train
    coef = np.linalg.solve(gram + RIDGE_ALPHA * np.eye(gram.shape[0]), x_train.T @ y_centered)
    return x_test @ coef + y_mean, y_mean


def evaluate_snapshot(
    name: str,
    path: Path,
    tensor: torch.Tensor,
    occupancy: np.ndarray,
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    rel_path = path.relative_to(RUN_ROOT).as_posix()
    actual_hash = sha256(path)
    expected_hash = manifest_hashes.get(rel_path)
    if expected_hash is None or actual_hash != expected_hash:
        raise ValueError(f"Manifest SHA-256 mismatch for {rel_path}: {actual_hash} vs {expected_hash}")
    x = patch_features(tensor)
    folds: list[dict[str, Any]] = []
    for held_out in range(5):
        train_indices = [idx for idx in range(5) if idx != held_out]
        train_x = x[train_indices].reshape(-1, x.shape[-1])
        train_y = occupancy[train_indices].reshape(-1)
        test_x = x[held_out].reshape(-1, x.shape[-1])
        target = occupancy[held_out]
        scores, train_mean = ridge_scores(train_x, train_y, test_x)
        score_map = scores.reshape(CONTROL_H, CONTROL_W)
        residual_map = np.clip(score_map - train_mean, 0.0, None)
        predicted_center = weighted_center(residual_map)
        target_center = weighted_center(target)
        prior_map = occupancy[train_indices].mean(axis=0)
        prior_center = weighted_center(prior_map)
        if target_center is None or prior_center is None:
            raise ValueError(f"Target/prior center is unscorable at latent time {held_out}")
        if predicted_center is None:
            feature_error = None
        else:
            feature_error = float(np.linalg.norm(np.asarray(predicted_center) - target_center))
        prior_error = float(np.linalg.norm(np.asarray(prior_center) - target_center))
        positive = target.reshape(-1) >= OCCUPANCY_THRESHOLD
        folds.append({
            "latent_time_index": held_out,
            "decoded_frame_index": FRAME_INDICES[held_out],
            "target_center_xy_control_cells": target_center,
            "feature_center_xy_control_cells": predicted_center,
            "coordinate_prior_center_xy_control_cells": prior_center,
            "feature_center_error_control_cells": feature_error,
            "coordinate_prior_center_error_control_cells": prior_error,
            "feature_center_error_pixels": None if feature_error is None else feature_error * PATCH_PIXELS,
            "coordinate_prior_center_error_pixels": prior_error * PATCH_PIXELS,
            "feature_average_precision": average_precision(scores, positive),
            "coordinate_prior_average_precision": average_precision(prior_map, positive),
            "positive_token_count_at_occupancy_threshold": int(positive.sum()),
            "feature_center_status": "ok" if predicted_center is not None else "unscorable_zero_positive_residual_mass",
        })
    feature_errors = [f["feature_center_error_control_cells"] for f in folds if f["feature_center_error_control_cells"] is not None]
    prior_errors = [f["coordinate_prior_center_error_control_cells"] for f in folds]
    feature_aps = [f["feature_average_precision"] for f in folds if f["feature_average_precision"] is not None]
    prior_aps = [f["coordinate_prior_average_precision"] for f in folds if f["coordinate_prior_average_precision"] is not None]
    return {
        "snapshot": name,
        "source_file": rel_path,
        "source_sha256": actual_hash,
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "folds": folds,
        "within_clip_summary": {
            "scored_timepoint_folds": len(feature_errors),
            "total_timepoint_folds": len(folds),
            "mean_feature_center_error_control_cells": float(np.mean(feature_errors)) if feature_errors else None,
            "mean_coordinate_prior_center_error_control_cells": float(np.mean(prior_errors)),
            "mean_feature_minus_prior_error_control_cells": (
                float(np.mean(feature_errors) - np.mean(prior_errors)) if feature_errors else None
            ),
            "mean_feature_average_precision": float(np.mean(feature_aps)) if feature_aps else None,
            "mean_coordinate_prior_average_precision": float(np.mean(prior_aps)) if prior_aps else None,
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUTPUT}")
    manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    manifest_hashes = manifest["files_sha256"]
    audit = json.loads(FRAME_AUDIT.read_text(encoding="utf-8"))
    occupancy, proxy_records = load_proxy_labels(audit, manifest_hashes)
    results: list[dict[str, Any]] = []

    for step in POST_STEPS:
        path = RUN_DIR / "latents" / f"post_step_{step:03d}.pt"
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if int(loaded["step_zero_based"]) != step:
            raise ValueError(f"Step metadata mismatch in {path}")
        results.append(evaluate_snapshot(
            f"post_step_{step:03d}", path, loaded["latent_after_scheduler_step"], occupancy, manifest_hashes
        ))

    for step in CLEAN_STEPS:
        path = RUN_DIR / "predicted_clean" / f"pre_step_{step:03d}.pt"
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if int(loaded["step_zero_based"]) != step or int(loaded["scheduler_step_index"]) != step:
            raise ValueError(f"Step metadata mismatch in {path}")
        results.append(evaluate_snapshot(
            f"predicted_clean_pre_step_{step:03d}", path, loaded["x0_hat"], occupancy, manifest_hashes
        ))

    used_inputs = [
        RUN_MANIFEST,
        FRAME_AUDIT,
        *[RUN_DIR / "frames" / f"frame_{idx:03d}.png" for idx in FRAME_INDICES],
        *[RUN_DIR / "latents" / f"post_step_{step:03d}.pt" for step in POST_STEPS],
        *[RUN_DIR / "predicted_clean" / f"pre_step_{step:03d}.pt" for step in CLEAN_STEPS],
    ]
    payload = {
        "experiment_id": "EXP-LATENTGATE-M0-SPATIAL-LOCATION-READOUT-001",
        "protocol": "research/2026-10-12_m0_spatial_latent_location_readout.md",
        "analysis_code": "research/run_m0_spatial_latent_location_readout.py",
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
            "latent_shape_bcthw": [1, 16, 5, 60, 104],
            "control_grid_wh": [CONTROL_W, CONTROL_H],
            "decoded_frame_shape_hwc": [HEIGHT, WIDTH, 3],
            "frame_to_latent_time_indices": FRAME_INDICES,
            "source_frame_motion_audit_method": audit["method"]["mask"],
        },
        "input_file_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in used_inputs
        },
        "proxy_label_records": proxy_records,
        "analysis_spec": {
            "feature": "flatten each 2x2 VAE-latent spatial patch across 16 channels into 64 values",
            "target": "fractional red-mask occupancy in corresponding 16x16 decoded RGB patch",
            "cross_validation": "leave one of five latent timepoints out; train on all spatial tokens at the other four times",
            "readout": "train-fold standardized ridge regression, intercept, alpha=1; center from positive prediction residual above train target mean",
            "coordinate_baseline": "training-time mean occupancy map",
            "average_precision_positive_threshold": OCCUPANCY_THRESHOLD,
            "independent_sample_count": 1,
            "temporal_fold_count": 5,
        },
        "limitations": [
            "Single generated clip, prompt, and seed; five correlated within-clip temporal folds are not independent samples.",
            "The red RGB threshold is a proxy from the same generated output, not independent labels or segmentation truth.",
            "A readout result does not establish generalization, object understanding, geometry, contact, causal control, policy efficacy, or milestone completion.",
            "Post-step diffusion latents and predicted-clean x0-hat tensors are different state types and are reported separately.",
        ],
        "snapshots": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "snapshots": [{"snapshot": r["snapshot"], **r["within_clip_summary"]} for r in results],
    }, indent=2))


if __name__ == "__main__":
    main()
