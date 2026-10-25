# M0 saved Wan latent-slot dynamics

**Date:** 2026-10-16  
**Scope:** CPU-only diagnostics on existing retry4/retry5 tensor snapshots. Checkpoint files were hash-verified and loaded with `torch.load(map_location="cpu", weights_only=True)`. No model weights, inference, decode, RGB inspection/scoring, GPU, or cloud runtime.

## Integrity and tensor inventory

All 20 selected `.pt` files match the SHA-256 entries in their run manifests: seven post-step checkpoints in each retry, four predicted-clean snapshots in retry5, and each run's `initial_noise.pt`. Safe loading succeeded with PyTorch 2.9.1; there was no pickle fallback.

Each post-step file contains `step_zero_based` and `latent_after_scheduler_step`, with the tensor float32 `[1,16,5,60,104]`. Each predicted-clean file contains `step_zero_based`, `timestep`, `sigma`, `x_i` (float32), `guided_velocity` (float16), `x0_hat` (float32), `formula`, `scheduler_step_index`, and `scheduler_thresholding`; the three tensors have shape `[1,16,5,60,104]`. Initial noise contains `latents` (float32 `[1,16,5,60,104]`) and `seed`.

Retry4 and retry5 are exact matches for initial noise and all seven post-step tensors: **8/8 `torch.equal` comparisons passed**, and the corresponding bytes also have identical SHA-256 hashes. Retry4 has no saved predicted-clean snapshots, so those changes are measured within retry5 only.

## Per-slot changes

For each pair of consecutive saved snapshots, metrics are computed independently for each temporal slot over its `C×H×W = 16×60×104` values. The table reports mean absolute change; the machine-readable result also records RMS change, maximum absolute change, and RMS change relative to the earlier slot tensor.

Post-step tensors (`latent_after_scheduler_step`), per-slot mean absolute change; slots are ordered 0–4:

| Saved-step interval | Mean absolute change by slot |
|---|---|
| 000→004 | `[0.045299, 0.052978, 0.057720, 0.059481, 0.059568]` |
| 004→009 | `[0.077035, 0.097848, 0.111754, 0.117874, 0.120240]` |
| 009→014 | `[0.102645, 0.130738, 0.148303, 0.155787, 0.157962]` |
| 014→019 | `[0.143853, 0.183077, 0.207026, 0.217392, 0.219243]` |
| 019→024 | `[0.215646, 0.272014, 0.307657, 0.321877, 0.321052]` |
| 024→029 | `[0.347493, 0.447485, 0.504087, 0.525946, 0.523585]` |

Predicted-clean tensors (`x0_hat`), per-slot mean absolute change; slots are ordered 0–4:

| Saved-snapshot interval | Mean absolute change by slot |
|---|---|
| 004→014 | `[0.121879, 0.186656, 0.202065, 0.214117, 0.211371]` |
| 014→024 | `[0.049289, 0.076975, 0.082542, 0.091253, 0.124451]` |
| 024→029 | `[0.042644, 0.049310, 0.052009, 0.055613, 0.087564]` |

These are within-run tensor-space differences, not decoded visual changes. They do not establish object/path influence, semantic content, controllability, or policy efficacy.

## Temporal mapping and interpretation

The five latent indices map to nominal RGB anchors `[0,4,8,12,16]`. The exact VAE source audit establishes causal-prefix structural support on this 17-frame clip: slot 0 can depend on frame 0; slot 1 on frames 0–4; slot 2 on 0–8; slot 3 on 0–12; and slot 4 on 0–16. Thus slots overlap in temporal support and are not independent single-frame measurements. The support audit's qualification applies: structural reachability does not prove nonzero numerical influence from every frame for these weights and inputs.

## Reproduction and evidence

- Runner: [`analyze_m0_latent_slot_dynamics.py`](analyze_m0_latent_slot_dynamics.py)
- Machine result with every selected artifact path/hash, key/shape/dtype inventory, retry equality check, and full per-slot statistics: [`m0_latent_slot_dynamics_v1.json`](m0_latent_slot_dynamics_v1.json)
- Temporal-support source audit: [`m0_wan_vae_temporal_support_audit_v1.json`](m0_wan_vae_temporal_support_audit_v1.json)
- Input run manifests: `m0/runs/wan_t4_2026-10-02_retry4/manifest.json` and `m0/runs/wan_t4_2026-10-02_retry5/manifest.json`; their hashes are recorded in the machine result.

The measurements describe one saved run. No RGB proxy inventory was repeated, and no tensor was decoded or passed through the model.
