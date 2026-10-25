# M0 retry5 dense RGB-change proxy versus five anchors

**Date:** 2026-10-16  
**Scope:** CPU-only comparison of the existing, manifest-hash-verified retry5 RGB PNGs. No weights, latent tensors, inference, VAE decode, human labels, GPU, or cloud runtime.

## Measure

For each consecutive frame pair, compute whole-frame normalized RGB mean absolute difference:

`MAE_t = mean(abs(RGB[t] - RGB[t+1])) / 255`

This is bounded from 0 to 1 and averages over every pixel and color channel. It uses no mask, optical flow, threshold, image registration, or segmentation. It measures pixel change, not object motion; background changes, lighting, rendering, or texture flicker can also affect it.

All 17 frame hashes matched the retry5 manifest. Frames are 832×480 RGB. The five anchors are `[0,4,8,12,16]`.

## Dense one-frame changes

Values below are normalized RGB MAE, with the equivalent 8-bit intensity-level MAE in parentheses.

| Frame pair | RGB MAE | Frame pair | RGB MAE |
|---|---:|---|---:|
| 0→1 | 0.004580 (1.168) | 8→9 | 0.002903 (0.740) |
| 1→2 | 0.002652 (0.676) | 9→10 | 0.001257 (0.321) |
| 2→3 | 0.001941 (0.495) | 10→11 | 0.001946 (0.496) |
| 3→4 | 0.002884 (0.735) | 11→12 | 0.001932 (0.493) |
| 4→5 | 0.003714 (0.947) | 12→13 | 0.003415 (0.871) |
| 5→6 | 0.003265 (0.832) | 13→14 | 0.002040 (0.520) |
| 6→7 | 0.002977 (0.759) | 14→15 | 0.001237 (0.315) |
| 7→8 | 0.001838 (0.469) | 15→16 | 0.001048 (0.267) |

The mean over 16 adjacent pairs is `0.002477`; the largest is frame `0→1` at `0.004580` (`1.168/255`). Thus the largest whole-frame average is 1.168 intensity levels per channel. Whole-frame averaging can still conceal localized changes.

## What the five anchors omit

Each anchor interval contains four adjacent transitions. The endpoint images provide one four-frame contrast, but not the timing or sequence of the four local changes. Comparing the sum of those adjacent MAEs (pixelwise path length) with the anchor endpoint MAE (pixelwise chord) gives:

| Anchor interval | Endpoint MAE | Sum of 1-frame MAEs | Path excess beyond endpoint contrast |
|---|---:|---:|---:|
| 0→4 | 0.009766 | 0.012057 | 19.0% |
| 4→8 | 0.010091 | 0.011794 | 14.4% |
| 8→12 | 0.005778 | 0.008038 | 28.1% |
| 12→16 | 0.006382 | 0.007740 | 17.5% |
| **All intervals** | **0.032017** | **0.039628** | **19.2%** |

The five-anchor set omits 12 of 17 frame images and does not directly observe any of the 16 adjacent one-frame transition magnitudes. The 19.2% aggregate path excess is a pixelwise total-variation comparison against four endpoint contrasts; it is **not** an estimate of missed object motion or an event-detection recall. Intermediate frame content can vary and return toward its starting appearance, while all-frame RGB MAE can also reflect non-object changes.

## Scope limit and artifacts

This is one generated clip, one prompt, and one seed. It provides an exploratory dense RGB-change trace only; no human labels or independent clips were used. It does not establish object identity, path direction, motion accuracy, controllability, or general sampling sufficiency. The registered full-sequence/all-frame evaluation policy is unchanged.

- Reproducer: [`analyze_m0_retry5_anchor_motion_proxy.py`](analyze_m0_retry5_anchor_motion_proxy.py)
- Full result with all frame hashes, 16 transition scores, interval summaries, and input-manifest hash: [`m0_retry5_anchor_motion_proxy_v1.json`](m0_retry5_anchor_motion_proxy_v1.json)
- Source run: `m0/runs/wan_t4_2026-10-02_retry5/run_001/frames/`
