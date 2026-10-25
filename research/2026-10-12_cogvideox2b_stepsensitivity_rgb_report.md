# CogVideoX-2B archived-output scoreability report

**Date:** 2026-10-12  
**Experiment:** `EXP-COGVIDEOX2B-STEP-SENSITIVITY-RGB-001`  
**Status:** complete, CPU-only, RGB proxy only.

## Result

The preregistered fixed 36-threshold red-mask proxy could not track either archived 480×320 clip across its nine frames.

| Saved output | Manifest conditions | Full tracks | Strongest raw red mask | Direction |
|---|---|---:|---:|---|
| `cogvideox_2b_pilot` | 8 steps, 480×320, 9 frames | 0/36 threshold cases | 0 pixels in every frame | Unscoreable |
| `cogvideox_2b_pilot_30steps` | 30 steps, 480×320, 9 frames | 0/36 threshold cases | 203 pixels in frame 0; below the 300-pixel centroid floor | Unscoreable |

The archived 50-step result is a separate configuration: 720×480 and 16 frames. Its prior preregistered RGB sweep detected the proxy in all 36 cases and measured a leftward endpoint shift of 4.87–5.02% of frame width. That is the opposite of the prompt's requested rightward direction under this color proxy, but the three runs cannot establish a denoising-step trend: resolution, frame count, and unrecorded pipeline details differ.

## Provenance and limits

The 8- and 30-step manifests agree on model name, prompt, seed 2026, guidance scale, frame count, and resolution, but this is nominal metadata matching only. The original CogVideoX-2B notebook/runner and run logs are absent locally; no pinned checkpoint revision, dependency versions, effective noise tensors, or full invocation records are available. The 480×320 outputs are below the model's documented 720×480 resolution. Both independent file reviews found that ZIP contents match the extracted files; the ZIPs contain no additional runner or environment provenance.

The fixed red-color threshold is not ground-truth segmentation or an independent tracker. Empty masks are preserved as missing observations; no interpolation or threshold relaxation was applied. No latent NPY was read. This result does not establish object identity, trajectory direction in the unscoreable clips, latent semantics, latent-to-frame alignment, causality, generalization, or policy efficacy.

## Reproduction record

- Protocol: [frozen procedure](2026-10-12_cogvideox2b_stepsensitivity_rgb_protocol.md)
- Input frame/manifest hashes: [input map](cogvideox2b_stepsensitivity_inputs_v1.json)
- CPU analysis: [runner](run_cogvideox2b_stepsensitivity_rgb.py)
- Full per-threshold data: [result JSON](cogvideox2b_stepsensitivity_rgb_v1.json)
- Existing 50-step RGB audit: [protocol](2026-10-12_cogvideox2b_rgb_proxy_audit.md), [result JSON](cogvideox2b_rgb_proxy_audit_v1.json)

This is a side diagnostic. It does not advance the project's open human-label gate or authorize a 2B latent comparison. The next useful 2B step is to recover the original notebook/runner and generation log; if unavailable, future comparisons need documented, supported-resolution generations with repeated matched seeds and an independent framewise observer.
