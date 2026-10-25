# CogVideoX-2B chunk13 Farneback flow proxy

**Date:** 2026-10-17  
**Scope:** Dense CPU optical-flow estimates on the existing hash-pinned 16-frame, 720×480 `chunk13` RGB sequence. No model weights, latent arrays, inference, generation, GPU, cloud runtime, dependency installation, or tests.

## Implementation

Used OpenCV Python **4.11.0**, `cv2.calcOpticalFlowFarneback` (classical Farneback polynomial-expansion flow; no learned weights). RGB PNGs were converted to 8-bit grayscale with `cv2.COLOR_RGB2GRAY`. OpenCL was explicitly disabled, and OpenCV was set to one CPU thread.

Parameters: `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, `flags=0`.

Flow vectors are in source-image pixels. I report raw mean magnitude in pixels plus mean divided by the 865.33 px frame diagonal. Field summaries are over every image pixel with no mask. An adjacent-pair field estimates displacement over one output-frame interval; an anchor endpoint field is a separate estimate over its 3- or 4-frame gap.

## Adjacent-pair results

The table shows mean flow magnitude in pixels and the field's 99th percentile. The existing same-frame-set normalized RGB MAE is shown for comparison.

| Pair | Mean flow (px) | P99 flow (px) | RGB MAE |
|---|---:|---:|---:|
| 0→1 | 0.00000 | 0.00 | 0.000000 |
| 1→2 | 0.00000 | 0.00 | 0.000000 |
| 2→3 | 0.00000 | 0.00 | 0.000000 |
| 3→4 | 0.03289 | 2.27 | 0.000707 |
| 4→5 | 0.03043 | 2.10 | 0.000666 |
| 5→6 | 0.03074 | 2.13 | 0.000674 |
| 6→7 | 0.02004 | 1.32 | 0.000454 |
| 7→8 | 0.04172 | 2.73 | 0.000916 |
| 8→9 | 0.05831 | 4.05 | 0.001284 |
| 9→10 | 0.07741 | 5.38 | 0.001664 |
| 10→11 | 0.03429 | 2.27 | 0.000778 |
| 11→12 | 0.06157 | 4.33 | 0.001262 |
| 12→13 | 0.08304 | 6.08 | 0.001795 |
| 13→14 | 0.05893 | 4.38 | 0.001263 |
| 14→15 | 0.03529 | 2.67 | 0.000790 |

Mean full-frame flow magnitude is `0.03764 px` per adjacent pair; maximum is `0.08304 px` at 12→13. The median and 95th percentile are zero for each adjacent field, while non-identical pairs have nonzero P99 estimates. This reflects a sparse full-frame field summary and does not identify the moving object.

## Adjacent-flow path versus anchor endpoints

Anchors are `[0,4,8,12,15]`. “Path” is the sum of mean magnitudes from the adjacent-pair fields. “Endpoint” is a separate direct Farneback estimate between the two anchor images.

| Interval | Adjacent-flow path (px) | Endpoint flow (px) | Signed path excess | RGB-MAE path excess |
|---|---:|---:|---:|---:|
| 0→4 | 0.03289 | 0.03289 | 0.0% | 0.0% |
| 4→8 | 0.12294 | 0.11981 | +2.5% | +8.4% |
| 8→12 | 0.23158 | 0.23606 | −1.9% | +17.4% |
| 12→15 | 0.17725 | 0.17996 | −1.5% | +13.3% |
| **All intervals** | **0.56466** | **0.56872** | **−0.7%** | **+13.1%** |

The direct endpoint estimate is slightly larger than the summed adjacent mean-flow path overall. These means are not tracked pointwise and need not satisfy a path-length inequality: the endpoint and adjacent flow fields are estimated independently.

## What this adds beyond RGB MAE

RGB MAE measures intensity change; Farneback estimates apparent pixel displacement from local image structure. It therefore adds a displacement-style magnitude field and a separate endpoint-flow comparison. On these 15 adjacent pairs, flow mean magnitude and RGB MAE have descriptive Spearman rank correlation `0.986`, and both peak at 12→13. The interval path/endpoint comparisons differ: flow is −1.9% for 8→12 while RGB MAE path excess is +17.4%. This is a method comparison on one clip, not independent evidence or an object-motion result.

Both measures cover the full image and can react to background, texture, lighting, and rendering artifacts. Farneback is also ambiguous on smooth/repetitive surfaces; mean magnitude drops direction and pixel correspondence. No object mask or identity tracking was used. Nothing here estimates event recall or supports general sparse-sampling claims; the full-sequence/all-frame policy remains unchanged.

## Artifacts

- Reproducer: [`analyze_cogvideox2b_chunk13_farneback_flow.py`](analyze_cogvideox2b_chunk13_farneback_flow.py)
- Machine result with OpenCV version/parameters, 15 dense fields, four endpoint fields, and exact frame hashes: [`cogvideox2b_chunk13_farneback_flow_v1.json`](cogvideox2b_chunk13_farneback_flow_v1.json)
- Frame inputs: [`cogvideox2b_rgb_proxy_inputs_v1.json`](cogvideox2b_rgb_proxy_inputs_v1.json)
- RGB-MAE comparator: [`cogvideox2b_chunk13_anchor_motion_proxy_v1.json`](cogvideox2b_chunk13_anchor_motion_proxy_v1.json)
