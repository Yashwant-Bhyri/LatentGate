# CogVideoX-2B chunk13 dense RGB-change proxy versus anchors

**Date:** 2026-10-16  
**Scope:** CPU-only pixel-change analysis of the existing 16-frame, 720×480 CogVideoX-2B `chunk13` sequence. All frame hashes were checked against the pinned input map before reading pixels. No model, latent arrays, inference, labels, GPU, cloud runtime, or new generation.

## Measure

Used the same measure as the Wan retry5 proxy: for each frame pair, `mean(abs(uint8_RGB(a) - uint8_RGB(b))) / 255` over all H×W×3 values. Each pair score is bounded to `[0,1]`. It is whole-frame pixel change, not optical flow or an object-motion measure; background, lighting, rendering, and texture can contribute.

The anchor schedule is `[0,4,8,12,15]`, yielding intervals of 4, 4, 4, and 3 transitions.

## All adjacent-frame scores

Scores are normalized RGB MAE; parentheses show average 8-bit intensity levels per channel.

| Pair | Score | Pair | Score | Pair | Score |
|---|---:|---|---:|---|---:|
| 0→1 | 0.000000 (0.000) | 5→6 | 0.000674 (0.172) | 10→11 | 0.000778 (0.198) |
| 1→2 | 0.000000 (0.000) | 6→7 | 0.000454 (0.116) | 11→12 | 0.001262 (0.322) |
| 2→3 | 0.000000 (0.000) | 7→8 | 0.000916 (0.234) | 12→13 | 0.001795 (0.458) |
| 3→4 | 0.000707 (0.180) | 8→9 | 0.001284 (0.327) | 13→14 | 0.001263 (0.322) |
| 4→5 | 0.000666 (0.170) | 9→10 | 0.001664 (0.424) | 14→15 | 0.000790 (0.201) |

Mean adjacent-pair MAE is `0.000817`; the maximum is frame `12→13` at `0.001795` (`0.458/255`). Frames 0–3 have identical hashes, so their first three adjacent changes are exactly zero.

## Anchor endpoint contrast versus dense path

The “path” is the sum of adjacent whole-frame MAEs in each interval; the “endpoint contrast” is the direct MAE between its two anchor images.

| Anchor interval | Transitions | Endpoint contrast | Sum of adjacent path | Path excess beyond endpoint |
|---|---:|---:|---:|---:|
| 0→4 | 4 | 0.000707 | 0.000707 | 0.0% |
| 4→8 | 4 | 0.002482 | 0.002711 | 8.4% |
| 8→12 | 4 | 0.004122 | 0.004988 | 17.4% |
| 12→15 | 3 | 0.003338 | 0.003848 | 13.3% |
| **All intervals** | **15** | **0.010648** | **0.012254** | **13.1%** |

Anchor-only sampling leaves 11 intermediate images out of the 16-frame sequence and does not observe the one-frame transition magnitudes directly. The aggregate path excess is `0.001605` normalized MAE (13.1% of the dense path), showing pixel variation beyond endpoint contrasts for this clip. It is not a missed-motion percentage, event count, or recall estimate.

## Limits and artifacts

This is one saved sequence under one prompt/seed/configuration. It does not establish object identity, trajectory accuracy, event-detection performance, or general sparse-sampling sufficiency. The existing full-sequence/all-frame policy is unchanged.

- Reproducer: [`analyze_cogvideox2b_chunk13_anchor_motion_proxy.py`](analyze_cogvideox2b_chunk13_anchor_motion_proxy.py)
- Machine result with pinned manifest/input hashes, all 16 frame hashes, every adjacent score, and interval metrics: [`cogvideox2b_chunk13_anchor_motion_proxy_v1.json`](cogvideox2b_chunk13_anchor_motion_proxy_v1.json)
- Existing pinned input map: [`cogvideox2b_rgb_proxy_inputs_v1.json`](cogvideox2b_rgb_proxy_inputs_v1.json)
