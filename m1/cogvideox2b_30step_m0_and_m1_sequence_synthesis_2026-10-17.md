# M0 30-step saved-clip diagnostic and M1 sequence architecture synthesis

**Date:** 2026-10-17  
**Scope:** Local synthesis of one existing CogVideoX-2B saved RGB clip and the pinned CogVideoX-5B-I2V source path. This is not a cross-model experiment. No inference, generation, upload, GPU/runtime, or tests were run for this synthesis.

## M0 evidence stays clip-local

The new diagnostic analyzes all nine hash-verified RGB frames from `cogvideox_2b_pilot_30steps` (480×320, recorded seed 2026, 30 denoising steps). Its existing fixed red-mask gate scored 0/9 frames, so neither proxy produces an object track or event label.

Across the eight adjacent transitions, normalized whole-frame RGB-MAE sums to `0.00095454`; four direct endpoint contrasts at `[0,2,4,6,8]` sum to `0.00092827`. The adjacent sum is 2.75% above the endpoint total. Farneback gives the opposite relation: its adjacent mean-magnitude sum is `0.18714 px`, versus `0.19119 px` for the four independent endpoint estimates, so the endpoints exceed the adjacent sum by 2.16% of the adjacent total. Both proxies peak at frame 0→1.

For the peak-mean transition 0→1, the pixelwise Farneback magnitude median is zero, its 99th percentile is about `3.9×10⁻⁵ px`, and its maximum is `36.68 px`. The mean is therefore tail-sensitive and must not be read as broad, typical displacement.

This records how two image-wide metrics summarize this saved clip. Their different path/endpoint relations show that they are not interchangeable. It does not establish object displacement, event recall, whether the anchor schedule is adequate, or an effect of denoising steps. The separate 8-step clip, Wan retry5, and CogVideoX-2B `chunk13` clip are not pooled with it.

The M0 inventory follow-up found no further eligible saved sequence: five run records reduce to four unique complete RGB sequences because Wan retry4 and retry5 are exact duplicates, and all four unique sequences already have adjacent RGB-MAE and Farneback results. Additional CPU work on this inventory would repeat existing measurements rather than add a new clip-level observation.

## What the M1 source supports

The registered M1 runner makes one 49-frame CogVideoX-5B-I2V pipeline call for the baseline and one for its same-runtime replay. The pinned VAE maps 49 RGB frames to 13 temporal latent positions and decodes those positions back to all 49 frames. The pipeline exposes no RGB frame-index selector. Selecting `[0,4,…,48]` from the returned PIL sequence happens after denoising and full VAE decode; it can reduce later review or file storage only if other frames are discarded. `output_type="latent"` returns all 13 final latent positions, not a sparse RGB subset.

The VAE’s internal temporal batches carry convolution cache and are concatenated into the full output. They are decode memory plumbing inside the same jointly denoised sequence, not independent generation chunks. VAE slicing is batch-axis only and does not help M1’s one-video batch. Enabled VAE tiling is spatial; source code predicts smaller per-tile activations with overlap work, but M1 VRAM and latency effects are unmeasured. A custom streaming decoder that discards non-anchor RGB chunks is unimplemented and could not remove the per-frame VAE convolution work.

These source facts support one full-sequence generation call and an all-49-frame output contract. They do not establish temporal coherence or output quality. No matched full-versus-chunked generation has been run, and no memory, latency, decode-time, or feedback-horizon measurement shows that generation chunking is needed.

## Smallest valid next M1 runtime experiment

Keep the existing registered smoke: after the canonical Drive notebook and bundle are refreshed and read back against the local hashes, obtain a fresh explicit no-paid T4 path in the designated account, and pass the guarded T4/compute-capability/package preflight. If any admission, identity, hash, or package gate fails, make no model call.

On that one admitted runtime, run only the seed-2026 full 49-frame baseline followed by its exact same-runtime replay. Retain and score every decoded frame, indices 0–48. Record the runtime, elapsed generation time, peak GPU memory, output hashes, and configured capture coverage. Do not substitute the 30-step nine-frame clip, anchor-only output, or independent chunks for this smoke.

**Pass** only if the registered ticket criteria all hold: Tesla T4 at compute capability 7.5; each configured post-step latent, pooled summary, spatial map, and predicted-clean capture appears exactly once; 49 frames are returned; the ordered score table covers all 49 indices with explicit uncertainty/missingness; captured tensors and RGB frames match bitwise on replay; and blinded review finds the scene and action scoreable. Any failed condition is a stop for route/configuration review. A pass permits review of this single-smoke route only; it does not establish sparse/chunked equivalence or authorize a seed batch. Consider a separate matched chunking ablation only if this completed smoke documents a concrete memory, latency, or control-horizon constraint.

## Source records

- M0 30-step report: [`2026-10-17_cogvideox2b_pilot_30steps_adjacent_motion.md`](../research/2026-10-17_cogvideox2b_pilot_30steps_adjacent_motion.md); machine result: [`cogvideox2b_pilot_30steps_adjacent_motion_v1.json`](../research/cogvideox2b_pilot_30steps_adjacent_motion_v1.json).
- M1 full-sequence and sparse RGB decode source audit: [`cogvideox_sparse_rgb_decode_audit_2026-10-17.md`](cogvideox_sparse_rgb_decode_audit_2026-10-17.md); registered smoke decision: [`geometry_scale_drift_ticket.json`](geometry_scale_drift_ticket.json).
