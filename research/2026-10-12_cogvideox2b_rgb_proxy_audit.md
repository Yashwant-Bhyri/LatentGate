# CogVideoX-2B RGB-only red-object proxy audit

**Status:** Preregistered before calculating RGB proxy metrics.  
**Question:** Does the visible red-object path in this one saved CogVideoX-2B output move in the direction requested by its prompt, and is that observation stable across the existing M0 color-threshold grid?

## Fixed inputs

- Run directory: `/Users/yash/Documents/Resume/video-diffusion-lab/run/cogvideox_2b_chunk13/`.
- Use only the run's `manifest.json` and `frames/frame_00.png` through `frame_15.png`. Do not read or score the NPY latent files.
- The pre-analysis SHA-256 values are frozen in `research/cogvideox2b_rgb_proxy_inputs_v1.json`. Stop if a file is missing or its hash differs.
- Require 16 RGB frames, each 720×480, matching the manifest. File-name order is treated as output order; this does not establish any latent-time mapping.

## Fixed measurement

For every frame, create a binary red proxy mask using all 36 registered M0 thresholds: `R > {130,150,170,190}`, `G < {100,125,150}`, and `B < {100,125,150}`. Keep the M0 minimum mask size of 300 pixels. Do not tune thresholds or add tracking, morphology, learned detectors, or human labels.

For each threshold, report frame coverage; start-to-end centroid displacement normalized by width and height; the fractions of valid adjacent steps moving right, left, or staying fixed; normalized path length; mask-area endpoint ratio; and mask-area coefficient of variation. Keep duplicate frames as zero-motion steps. Primary descriptive summary: among threshold cases with detections on all 16 frames, report the fraction with negative normalized horizontal endpoint displacement and the full range of that displacement. Also report exact duplicate frame transitions by byte hash.

## Interpretation boundary

This is an RGB-only, one-clip prompt-adherence description. A red-color mask is not ground truth, a semantic tracker, or a physical trajectory. It does not establish cross-model generalization, causal controllability, latent semantics, or policy efficacy. The CogVideoX latent-to-frame map remains unverified and is out of scope for this measurement.
