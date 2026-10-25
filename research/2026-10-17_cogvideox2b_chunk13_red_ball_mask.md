# CogVideoX-2B chunk13 red-ball mask and adjacent-frame diagnostic

**Date:** 2026-10-17  
**Scope:** Second saved-sequence diagnostic using the already fixed Wan red-mask rule on 16 CogVideoX-2B `chunk13` RGB frames. CPU-only; no model/latent arrays, inference, generation, GPU, cloud, installation, or tests.

## Integrity and isolation result

All 16 RGB file hashes match the pinned input map (`280f8d…`); that map binds the source manifest hash (`2727cc…`) and its 16-frame metadata. The source manifest contains no per-frame digest table, so its hash/count were cross-checked while the per-frame digests were checked against the pinned input map and both prior RGB-analysis results.

Reused the fixed mask rule exactly: `R > 150, G < 125, B < 125`, 8-connected components, with the largest component required to contain at least 300 pixels and all other foreground components combined fewer than 300 pixels. No threshold changes, morphology, or temporal association were used. The isolation gate passed **16/16 frames**. Largest component area is **1,792–2,015 px**; outside foreground is at most **2 px**. Visual crop review shows the box on the red ball and excludes the floor shadow/reflection.

## Per-frame box, area, and centroid

Boxes are `[x,y,width,height]` source pixels, converted from half-open bounds. Centroids are `(x,y)` source pixels.

| Frame | Box `[x,y,w,h]` | Area (px) | Centroid `(x,y)` |
|---:|---:|---:|---:|
| 0 | `[425,221,51,52]` | 2,015 | (449.85, 246.53) |
| 1 | `[425,221,51,52]` | 2,015 | (449.85, 246.53) |
| 2 | `[425,221,51,52]` | 2,015 | (449.85, 246.53) |
| 3 | `[425,221,51,52]` | 2,015 | (449.85, 246.53) |
| 4 | `[423,220,50,52]` | 1,998 | (447.48, 245.80) |
| 5 | `[421,220,50,51]` | 1,979 | (445.49, 244.67) |
| 6 | `[419,219,50,52]` | 1,982 | (443.36, 244.08) |
| 7 | `[418,218,50,51]` | 1,992 | (442.51, 242.93) |
| 8 | `[416,217,49,51]` | 1,959 | (439.58, 241.99) |
| 9 | `[412,215,49,52]` | 1,938 | (435.57, 240.16) |
| 10 | `[408,212,48,50]` | 1,884 | (431.00, 236.42) |
| 11 | `[406,210,49,50]` | 1,880 | (429.39, 234.35) |
| 12 | `[401,210,49,49]` | 1,808 | (424.71, 233.57) |
| 13 | `[396,206,48,49]` | 1,812 | (419.44, 230.01) |
| 14 | `[394,202,47,51]` | 1,792 | (416.77, 226.27) |
| 15 | `[391,201,48,50]` | 1,801 | (414.28, 224.74) |

## Adjacent centroid displacement beside prior full-frame metrics

| Pair | Mask centroid `Δ(x,y)`; distance (px) | Farneback mean flow (px) | RGB MAE (normalized) |
|---|---:|---:|---:|
| 0→1 | (+0.00,+0.00); 0.00 | 0.00000 | 0.000000 |
| 1→2 | (+0.00,+0.00); 0.00 | 0.00000 | 0.000000 |
| 2→3 | (+0.00,+0.00); 0.00 | 0.00000 | 0.000000 |
| 3→4 | (-2.37,-0.73); 2.48 | 0.03289 | 0.000707 |
| 4→5 | (-1.99,-1.13); 2.29 | 0.03043 | 0.000666 |
| 5→6 | (-2.13,-0.60); 2.21 | 0.03074 | 0.000674 |
| 6→7 | (-0.85,-1.15); 1.43 | 0.02004 | 0.000454 |
| 7→8 | (-2.93,-0.94); 3.08 | 0.04172 | 0.000916 |
| 8→9 | (-4.01,-1.83); 4.41 | 0.05831 | 0.001284 |
| 9→10 | (-4.57,-3.74); 5.90 | 0.07741 | 0.001664 |
| 10→11 | (-1.61,-2.07); 2.62 | 0.03429 | 0.000778 |
| 11→12 | (-4.69,-0.79); 4.75 | 0.06157 | 0.001262 |
| 12→13 | (-5.27,-3.55); 6.35 | 0.08304 | 0.001795 |
| 13→14 | (-2.67,-3.74); 4.60 | 0.05893 | 0.001263 |
| 14→15 | (-2.49,-1.53); 2.92 | 0.03529 | 0.000790 |

The mask-centroid step peaks at **12→13 (6.35 px)**. The already registered full-frame Farneback and RGB-MAE peaks both occur at **12→13** (flow **0.08304 px**, RGB MAE **0.001795**); the two whole-frame pair rankings had Spearman ρ=0.986. Frames 0–3 are byte-identical, so their three adjacent centroid steps and full-frame changes are zero.

Mean adjacent centroid displacement is 2.87 px; the sum of adjacent steps is 43.04 px, versus 41.72 px direct from frame 0 to 15.

This is a second saved-sequence diagnostic, not a controlled Wan-versus-CogVideoX model comparison: the saved clips differ in model, resolution, length, and generation settings. The fixed color mask and flow/MAE values are image-space proxies from one sequence each, not general sampling evidence or a policy test.

## Artifacts

- Fixed mask rule: [Wan mask protocol](2026-10-17_m0_wan_retry5_red_ball_mask_protocol.md)
- Reproducer: [analyze_cogvideox2b_chunk13_red_ball_mask.py](analyze_cogvideox2b_chunk13_red_ball_mask.py)
- Machine-readable masks, frame hashes, boxes, and joined pair metrics: [cogvideox2b_chunk13_red_ball_mask_v1.json](cogvideox2b_chunk13_red_ball_mask_v1.json)
- Existing comparison references: [Farneback flow](cogvideox2b_chunk13_farneback_flow_v1.json), [RGB MAE](cogvideox2b_chunk13_anchor_motion_proxy_v1.json), and [pinned frame map](cogvideox2b_rgb_proxy_inputs_v1.json)
