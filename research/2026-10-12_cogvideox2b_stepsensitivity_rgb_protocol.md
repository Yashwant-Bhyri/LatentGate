# Archived CogVideoX-2B RGB proxy scoreability check

**Registered:** 2026-10-12, before pixel-mask measurement.  
**Experiment:** `EXP-COGVIDEOX2B-STEP-SENSITIVITY-RGB-001`  
**Status:** protocol and input hashes frozen; analysis not yet run.

## Question

Do the two archived 480×320, nine-frame CogVideoX-2B outputs produce a sufficiently covered red-threshold proxy track to describe the prompted ball's image-plane direction? The manifests record the same model name, prompt, seed 2026, guidance scale 6.0, image size, and frame count, with 8 versus 30 denoising steps. The archived 50-step output is not pooled here because it has 720×480 pixels and 16 frames; its separate RGB audit is already recorded.

This is a scoreability and saved-image description. Manifest agreement does not prove the code, checkpoint, environment, or effective initial noise was identical across runs. Do not attribute any difference to denoising-step count alone.

## Frozen inputs

- Read only `manifest.json` and `frames/frame_00.png` through `frame_08.png` from `cogvideox_2b_pilot/` and `cogvideox_2b_pilot_30steps/`.
- Input paths, expected manifest fields, dimensions, frame counts, and SHA-256 hashes are frozen in `research/cogvideox2b_stepsensitivity_inputs_v1.json`.
- Stop if any manifest or frame hash, filename, frame count, or dimension differs. Do not inspect or score latent NPYs, MP4s, or contact sheets in the measurement runner.

## Fixed measurement

- For each frame, threshold RGB pixels with all 36 combinations: `R > {130,150,170,190}`, `G < {100,125,150}`, and `B < {100,125,150}`.
- Preserve the existing 300-pixel minimum mask size. Record raw pixel area for each threshold and frame; a centroid is valid only when at least 300 pixels survive. This absolute floor has a different relative size than it would at 720×480, so the result applies only within this matched 480×320 pair.
- For every threshold case and run, report valid-frame count, full-clip coverage, and normalized endpoint displacement (`dx / width`, `dy / height`) only if all nine frames are valid. Also report right/left/still adjacent-step fractions, normalized path length, mask-area fractions, and exact duplicate-frame transitions.
- Report empty/under-minimum detections as such. Do not tune the thresholds, lower the area floor, add morphology/tracking, interpolate missing frames, or hand-select a detector.
- Compare coverage and endpoint-direction availability across the two saved runs descriptively. If either run has no fully covered threshold case, state that direction is not scoreable under this registered proxy; do not infer motion direction from partial detections.

## Interpretation limits

There is one saved output per recorded setting, no independent seed replication, no pinned CogVideoX checkpoint revision, no original 2B runner/notebook or dependency lock, and no generation logs. The two archived runs are below the model's documented 720×480 resolution. Therefore this is not a controlled step-count ablation or a representative model-quality estimate. The color mask is not ground-truth segmentation or an independent object tracker. No latent semantics, latent-to-frame alignment, object identity, causal controllability, policy efficacy, or downstream robotics result follows.
