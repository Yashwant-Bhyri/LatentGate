# Wan retry5 red-ball mask and adjacent-frame diagnostic

**Date:** 2026-10-17  
**Scope:** Deterministic CPU image-space mask on the existing 17-frame Wan retry5 clip. No model/latent arrays, inference, generation, GPU, cloud, installation, or tests.

## Fixed method and isolation result

All 17 source PNG hashes match the run manifest (`a190ba…`). Before scoring, the [mask protocol](2026-10-17_m0_wan_retry5_red_ball_mask_protocol.md) fixed the existing M0 rule `R > 150, G < 125, B < 125`, 8-connected components, and an isolation gate: the largest component must contain at least 300 pixels and all other components combined fewer than 300 pixels. No morphology, threshold sweep, or temporal association was used.

The rule passed all 17 frames. The largest component spans **10,306–12,981 px**; foreground outside it is **0–3 px** per frame (the second component, when present, is tiny). A visual review of the 17-frame contact sheet found the component on the red ball, while the floor reflection remains outside the box.

## Per-frame mask box and centroid

Boxes are `[x,y,width,height]` pixels, converted from half-open bounds; area is thresholded component pixels. Centroids use source-image pixel coordinates. The final column is the displacement from the preceding frame; frame 0 has no preceding pair.

| Frame | Box `[x,y,w,h]` | Area (px) | Centroid `(x,y)` | Previous-pair `Δ(x,y)`; distance (px) |
|---:|---:|---:|---:|---:|
| 0 | `[153,240,120,127]` | 10,306 | (209.01, 301.14) | — |
| 1 | `[154,241,122,126]` | 10,509 | (211.12, 300.95) | (+2.12,-0.19); 2.12 |
| 2 | `[157,241,121,126]` | 10,709 | (214.03, 300.55) | (+2.91,-0.40); 2.93 |
| 3 | `[158,240,122,128]` | 10,829 | (215.77, 300.39) | (+1.75,-0.15); 1.75 |
| 4 | `[162,238,124,131]` | 10,964 | (220.28, 299.98) | (+4.51,-0.42); 4.53 |
| 5 | `[164,238,124,131]` | 11,130 | (222.62, 299.78) | (+2.34,-0.20); 2.35 |
| 6 | `[168,236,125,134]` | 11,413 | (227.10, 298.91) | (+4.48,-0.87); 4.56 |
| 7 | `[171,235,127,136]` | 11,710 | (231.11, 298.74) | (+4.01,-0.17); 4.01 |
| 8 | `[173,235,127,137]` | 11,867 | (233.06, 298.95) | (+1.95,+0.21); 1.96 |
| 9 | `[175,235,128,138]` | 11,951 | (235.44, 299.67) | (+2.38,+0.72); 2.49 |
| 10 | `[176,235,128,139]` | 12,053 | (236.73, 300.15) | (+1.29,+0.47); 1.38 |
| 11 | `[177,236,129,140]` | 12,270 | (238.42, 301.48) | (+1.69,+1.33); 2.15 |
| 12 | `[177,237,131,141]` | 12,529 | (239.76, 303.34) | (+1.34,+1.86); 2.29 |
| 13 | `[178,238,131,142]` | 12,544 | (240.04, 304.61) | (+0.28,+1.27); 1.30 |
| 14 | `[178,239,132,143]` | 12,804 | (240.90, 306.21) | (+0.86,+1.60); 1.82 |
| 15 | `[178,239,132,144]` | 12,848 | (241.15, 306.94) | (+0.25,+0.74); 0.78 |
| 16 | `[178,240,133,144]` | 12,981 | (241.57, 307.40) | (+0.42,+0.46); 0.62 |

## Comparison with existing full-frame peaks

The largest red-mask centroid step is **5→6** at **4.56 px**. Existing full-frame Farneback peaks at **3→4** (0.14876 px mean flow); existing normalized RGB MAE peaks at **0→1** (0.004580). Thus the ball-local centroid peak is not either full-frame peak. The JSON joins centroid displacement, full-frame flow, and RGB MAE for every adjacent pair.

Mean adjacent centroid displacement is 2.32 px; the sum of adjacent steps is 37.05 px, versus 33.16 px from frame 0 to 16. These are image-space summaries for one clip, not a ground-truth object track or evidence for changing the all-frame policy.

## Artifacts

- Protocol fixed before scoring: [`2026-10-17_m0_wan_retry5_red_ball_mask_protocol.md`](2026-10-17_m0_wan_retry5_red_ball_mask_protocol.md)
- Reproducer: [`analyze_wan_retry5_red_ball_mask.py`](analyze_wan_retry5_red_ball_mask.py)
- Machine-readable frame/pair data and verified hashes: [`m0_wan_retry5_red_ball_mask_v1.json`](m0_wan_retry5_red_ball_mask_v1.json)
- Comparison references: [`Wan Farneback result`](m0_wan_retry5_farneback_flow_v1.json) and [`Wan RGB-MAE result`](m0_retry5_anchor_motion_proxy_v1.json)
