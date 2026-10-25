# Wan retry5 Farneback flow cross-clip diagnostic

**Date:** 2026-10-17  
**Scope:** CPU-only optical-flow comparison on the existing, manifest-hash-verified Wan retry5 RGB sequence. The RGB-MAE result and registered CogVideoX flow result were used only as metadata/comparison references. No model or latent arrays were loaded; no inference, generation, GPU, cloud, package installation, or tests.

## Integrity and implementation

All **17 RGB frame hashes** matched the retry5 manifest (`a190ba…`), and the existing Wan RGB-MAE result contains the same 17 hashes and anchor set `[0,4,8,12,16]`. The registered CogVideoX Farneback result hash also matched its pinned value; the current OpenCV version was **4.11.0**, matching the reference.

Reused OpenCV Python `cv2.calcOpticalFlowFarneback` with parameters `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, `flags=0`. RGB PNGs were converted to grayscale. OpenCL was disabled and one CPU thread was used; no trained weights were involved.

The vector magnitude is in source-image pixels. Mean magnitude is also divided by the frame diagonal (normalization and full per-field statistics are in the JSON). The path measure sums independently estimated adjacent-pair mean magnitudes. Each endpoint flow is a separate direct estimate between anchor images, not a pointwise-composed path.

## Pair ranking against RGB MAE

The table gives ascending ranks (1 = smallest, 16 = largest) for adjacent-pair full-frame mean flow magnitude and the existing normalized RGB MAE.

| Pair | Flow rank | RGB-MAE rank | Pair | Flow rank | RGB-MAE rank |
|---|---:|---:|---|---:|---:|
| 0→1 | 10 | 16 | 8→9 | 12 | 11 |
| 1→2 | 13 | 9 | 9→10 | 4 | 3 |
| 2→3 | 6 | 6 | 10→11 | 8 | 7 |
| 3→4 | 16 | 10 | 11→12 | 9 | 5 |
| 4→5 | 11 | 15 | 12→13 | 3 | 14 |
| 5→6 | 15 | 13 | 13→14 | 5 | 8 |
| 6→7 | 14 | 12 | 14→15 | 1 | 2 |
| 7→8 | 7 | 4 | 15→16 | 2 | 1 |

Flow and RGB-MAE rankings correlate moderately in this clip (**Spearman ρ=0.60**, 16 pairs). Their maxima differ: flow peaks at **3→4** (`0.14876 px` mean magnitude), while RGB MAE peaks at **0→1** (`0.004580`). This shows the two whole-frame proxies are not interchangeable; it does not identify which pair contains object motion.

## Anchor interval path versus endpoint

| Interval | Adjacent-flow path (px) | Direct endpoint flow (px) | Flow path excess | RGB-MAE path excess |
|---|---:|---:|---:|---:|
| 0→4 | 0.39360 | 0.36644 | +6.9% | +19.0% |
| 4→8 | 0.41321 | 0.39604 | +4.2% | +14.4% |
| 8→12 | 0.28981 | 0.25612 | +11.6% | +28.1% |
| 12→16 | 0.14089 | 0.12352 | +12.3% | +17.5% |
| **All intervals** | **1.23750** | **1.14213** | **+7.7%** | **+19.2%** |

Flow path excess is positive in every Wan interval, with smaller percentages than the RGB-MAE excess. These are summaries of independent full-frame estimates, not tracked motion or a measured object trajectory.

## Cross-clip context

| Saved sequence | Size / pairs | Mean adjacent flow | Flow-path excess vs endpoints | Flow–RGB-MAE rank correlation |
|---|---|---:|---:|---:|
| Wan retry5 | 832×480 / 16 | 0.07734 px (`0.0000805` frame diagonals) | +7.7% | 0.60 |
| CogVideoX-2B chunk13 | 720×480 / 15 | 0.03764 px (`0.0000435` frame diagonals) | −0.7% | 0.986 |

This is not a controlled model comparison: the two sequences differ in model, prompt, seed, resolution, frame count, and generation settings. Correlations are computed within each clip only. The flow statistic adds a displacement-style estimate beyond intensity change, but both are image-wide algorithmic proxies; neither is ground truth for the object or path. No policy changes follow; full-sequence generation and all-frame evaluation remain in place.

## Artifacts

- Reproducer: [`analyze_wan_retry5_farneback_flow.py`](analyze_wan_retry5_farneback_flow.py)
- Machine result with frame hashes, all 16 flow fields, four endpoint estimates, pair ranks, and cross-clip reference values: [`m0_wan_retry5_farneback_flow_v1.json`](m0_wan_retry5_farneback_flow_v1.json)
- Inputs: retry5 manifest and the existing [Wan RGB-MAE result](m0_retry5_anchor_motion_proxy_v1.json); parameter reference: [CogVideoX Farneback result](cogvideox2b_chunk13_farneback_flow_v1.json)
