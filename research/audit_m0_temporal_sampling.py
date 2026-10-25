#!/usr/bin/env python3
"""Audit RGB-frame versus latent-time sampling in the saved Wan M0 packet.

This is a CPU metadata audit. It does not load model weights or run inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/generation_packet.json"
PATH_SPEC_PATH = ROOT / "research/path_counterfactual_spec_v1.json"
OUTPUT_PATH = ROOT / "research/m0_temporal_sampling_audit_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    packet = json.loads(PACKET_PATH.read_text())
    path_spec = json.loads(PATH_SPEC_PATH.read_text())

    generation = packet["generation"]
    latent_shape = packet["tensor_map"]["initial_latents"]["shape"]
    vae = packet["tensor_map"]["vae"]
    frame_count = int(generation["num_frames"])
    temporal_ratio = int(vae["temporal_compression_ratio"])
    spatial_ratio = int(vae["spatial_compression_ratio"])
    latent_time_count = int(latent_shape[2])
    latent_height, latent_width = int(latent_shape[3]), int(latent_shape[4])

    if len(latent_shape) != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got latent shape {latent_shape}")
    if (frame_count - 1) % temporal_ratio:
        raise ValueError("Frame count does not align with the VAE temporal stride")

    derived_frame_anchors = list(range(0, frame_count, temporal_ratio))
    if len(derived_frame_anchors) != latent_time_count:
        raise ValueError(
            "Derived nominal RGB-frame anchor count does not match the latent "
            f"temporal axis ({len(derived_frame_anchors)} != {latent_time_count})"
        )
    configured_frame_anchors = path_spec["paired_paths"][
        "latent_temporal_samples_from_17_frames"
    ]
    post_steps = packet["capture_status"]["post_step_latent_steps_captured"]
    x0_steps = packet["capture_status"]["predicted_clean_steps_captured"]

    temporal_positions = [
        {
            "latent_time_index": latent_index,
            "nominal_rgb_frame_anchor": frame_index,
            "frame_stride_from_previous_position": (
                None if latent_index == 0 else temporal_ratio
            ),
        }
        for latent_index, frame_index in enumerate(derived_frame_anchors)
    ]

    result = {
        "audit_id": "EXP-M0-TEMPORAL-SAMPLING-AUDIT-001",
        "status": "completed_cpu_metadata_audit",
        "scope": (
            "Compare saved Wan M0 decoded-frame count, latent temporal axis, VAE "
            "compression metadata, and the planned DiTraj RGB-frame anchors."
        ),
        "model_outcome_claim": "none; no model inference or chunked-generation comparison",
        "generation": {
            "rgb_frames": frame_count,
            "resolution_wh": [generation["width"], generation["height"]],
            "latent_shape_bcthw": latent_shape,
            "latent_temporal_positions": latent_time_count,
            "vae_temporal_compression_ratio": temporal_ratio,
            "vae_spatial_compression_ratio": spatial_ratio,
            "spatial_shape_matches_vae_ratio": [
                generation["height"] // spatial_ratio,
                generation["width"] // spatial_ratio,
            ]
            == [latent_height, latent_width],
        },
        "temporal_alignment": {
            "formula": "latent_t = (rgb_frames - 1) / temporal_compression_ratio + 1",
            "derived_nominal_rgb_frame_anchors": derived_frame_anchors,
            "configured_ditraj_frame_anchors": configured_frame_anchors,
            "anchors_match": derived_frame_anchors == configured_frame_anchors,
            "positions": temporal_positions,
            "interpretation_limit": (
                "These are nominal stride anchors, not disjoint RGB-frame groups or "
                "a measurement of each latent's exact temporal receptive field."
            ),
        },
        "saved_temporal_diagnostics": {
            "post_step_latent_checkpoints": post_steps,
            "post_step_checkpoint_count": len(post_steps),
            "latent_time_slices_across_saved_checkpoints": len(post_steps)
            * latent_time_count,
            "predicted_clean_steps": x0_steps,
            "predicted_clean_time_slices": len(x0_steps) * latent_time_count,
            "transformer_summary_call_count": packet["capture_status"][
                "hidden_state_summary_count"
            ],
            "transformer_summary_semantics": packet["capture_semantics"][
                "transformer_hidden"
            ],
        },
        "recommendation": {
            "framewise_view": (
                "Keep all decoded RGB frames as the evaluation/readout unit for "
                "visibility, identity, box, and trajectory measurements."
            ),
            "latent_view": (
                "Index internal-state measurements by latent time position and "
                "denoising step; this run has five latent positions, not seventeen."
            ),
            "chunking": (
                "Do not make chunked generation a fixed architecture requirement. "
                "Consider it only after a measured memory, latency, or control-horizon "
                "constraint; compare against the full-sequence baseline because "
                "chunking changes temporal context."
            ),
        },
        "limitations": [
            "Metadata only; no model was loaded and no GPU was used.",
            "The audit does not determine exact VAE receptive-field support per latent position.",
            "No equivalence or quality comparison between full-sequence and chunked inference was run.",
        ],
        "inputs_sha256": {
            str(PACKET_PATH.relative_to(ROOT)): sha256(PACKET_PATH),
            str(PATH_SPEC_PATH.relative_to(ROOT)): sha256(PATH_SPEC_PATH),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
