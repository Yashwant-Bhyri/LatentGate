# CogVideoX-2B 50-step five-anchor RGB readout

**Status:** Preregistered before interpolation metrics.  
**Registered:** 2026-10-15 06:46 UTC.

## Question

For the one saved 50-step CogVideoX-2B sequence that passes the existing fixed red-mask coverage gate, how closely do five RGB frames `[0,4,8,12,15]` reproduce the centroid and mask area of the other 11 frames?

## Fixed inputs and gates

- Use only the existing `manifest.json` and the 16 `frames/frame_00.png` through `frame_15.png` listed in `cogvideox2b_rgb_proxy_inputs_v1.json`. Do not read latent arrays.
- Require the manifest SHA-256, pinned input-map SHA-256, expected manifest fields, all RGB frame hashes, and 720×480 dimensions to match. Stop on any mismatch.
- Use the existing 36 red-mask thresholds, minimum 300 pixels, and unmodified mask rule. Require all 16 frames to pass the floor in every threshold case before calculating interpolation or event-window metrics. Stop at coverage failure; do not tune thresholds.
- Interpolate centroid x/y and mask pixel count linearly between the five anchors. Report center mean/RMSE/max and area absolute pixel / relative percent mean/max across omitted frames, for all 36 cases and the fixed baseline case.
- Enumerate schedule-only coverage for all possible contiguous event windows of 1–4 frames in a 16-frame sequence.

## Limits

This is a CPU-only descriptive check on one saved clip. It is not an independent sample, a general sampling claim, a chunked generation comparison, evidence of latent-to-frame alignment, or evidence that any enumerated event occurred. No generation, inference, latent inspection, GPU, cloud runtime, or human labeling is part of this measurement.

The exact frozen inputs and machine-readable rules are in `cogvideox2b_50step_five_anchor_preregistration_v1.json`.
