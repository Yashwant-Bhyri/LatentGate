# CogVideoX-2B cross-backbone alignment gate audit

**Date:** 2026-10-12 UTC  
**Scope:** Read-only source and artifact audit. No model, inference, GPU, notebook cell, or human-label work was run.

## Decision

Do not score the saved CogVideoX-2B latent checkpoints against decoded-frame changes yet. The artifact metadata does not establish which saved tensor axis is time or how its indices align with the exported PNG frames. A matching dimension of 16 in the manifest and frame count is not sufficient evidence.

## Evidence inspected

- The candidate is `/Users/yash/Documents/Resume/video-diffusion-lab/run/cogvideox_2b_chunk13/`. Its manifest identifies CogVideoX-2B, 16 final frames at 720×480, and 11 saved checkpoints. Each checkpoint is recorded with shape `[1, 4, 16, 60, 90]`, without axis names, dtype, source revision, or a frame-index mapping.
- The accompanying ZIP contains only the manifest, 16 PNGs, 11 NPY tensors, and contact sheet; it adds no runner or provenance record.
- The only local `run_pilot.py` is an Apple MLX Wan2.1 implementation. It does not define the CogVideoX serialization or decoding path. No local CogVideoX-2B notebook or runner was found. The local CogVideoX-5B I2V notebook is a separate workflow.
- The linked private CogVideoX-2B Colab notebook was opened while signed into `yashwantkumarbhyri@gmail.com`; Google Drive returned **You need access**. No access request was sent and no account switch was made.

## Required evidence to reopen this check

Provide or share the original CogVideoX-2B notebook or exact runner source and run log. The read-only audit must establish:

1. The exact object passed to `np.save`, its axis order, and any `permute`/reshape operations.
2. The pipeline revision, dependency versions, and VAE configuration, including temporal compression.
3. The frame decode/export path and the temporal index map from the saved tensor to the 16 ordered PNGs.
4. Hashes for all input tensors and frames before any CPU-only analysis.

If those records are unavailable, keep the time mapping unknown and exclude this run from temporal-localization results. Its visible output and latent tensors remain feasibility artifacts only; they are not an independent sample-level replication of M0.
