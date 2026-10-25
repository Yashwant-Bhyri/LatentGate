# CogVideoX-2B pilot adjacent RGB-MAE and Farneback diagnostic

**Date:** 2026-10-17  
**Scope:** CPU-only image-space measurement of the existing 9-frame CogVideoX-2B 8-step `pilot` clip. No inference, latent access, generation, GPU, cloud, installation, or tests.

## Integrity and prior coverage

All nine RGB hashes match the registered input map and sparse-readout preregistration; the source manifest hash and its run metadata also match. The earlier fixed red-mask audit remains at 0/9 scoreable frames; this task did not rerun or retune that gate. Existing pilot artifacts record mask scoreability and a stopped five-anchor readout, but contain no adjacent-pair RGB-MAE or Farneback transition metrics. This report adds those pixelwise measurements.

## Method

RGB MAE is `mean(abs(RGB_t - RGB_t+1))/255` over all channels and pixels. Farneback uses OpenCV 4.11.0 on grayscale derived from RGB, with the registered chunk13 parameters (`pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, `flags=0`); OpenCL was disabled and one CPU thread used. The registered anchors are `[0,2,4,6,8]`. Endpoint values below are separate direct estimates between anchor frames; dense paths sum adjacent-pair metrics.

## Eight adjacent transitions

| Pair | Normalized RGB MAE | Farneback mean magnitude (px) |
|---|---:|---:|
| 0→1 | 1.1488971e-06 | 3.9297495e-09 |
| 1→2 | 5.787037e-07 | 4.9589242e-09 |
| 2→3 | 5.2764161e-07 | 6.3774793e-09 |
| 3→4 | 1.6169662e-07 | 7.2664372e-09 |
| 4→5 | 1.3105937e-06 | 7.5400814e-09 |
| 5→6 | 6.3827614e-07 | 7.7311991e-09 |
| 6→7 | 4.2551743e-07 | 6.8828242e-09 |
| 7→8 | 2.1275871e-07 | 8.1067417e-09 |

## Dense paths versus registered-anchor endpoints

| Interval | RGB-MAE adjacent path | RGB-MAE endpoint | Flow adjacent path (px) | Flow endpoint (px) |
|---|---:|---:|---:|---:|
| 0→2 | 1.7276008e-06 | 1.6424973e-06 | 8.8886737e-09 | 4.5683673e-09 |
| 2→4 | 6.8933824e-07 | 6.5529684e-07 | 1.3643916e-08 | 6.3856701e-09 |
| 4→6 | 1.9488698e-06 | 1.4382489e-06 | 1.527128e-08 | 7.5009008e-09 |
| 6→8 | 6.3827614e-07 | 5.1913126e-07 | 1.4989566e-08 | 7.1360803e-09 |

Across all eight pairs, the RGB-MAE dense path is **5.004085e-06** versus **4.2551743e-06** for the four endpoints; the portion of the dense path above endpoint contrasts is **15.0%**. Farneback path sum is **5.2793437e-08 px** versus **2.5591019e-08 px** at endpoints; the corresponding path fraction is **51.5%**.

The largest RGB-MAE transition is **4→5** (1.3105937e-06); the largest Farneback transition is **7→8** (8.1067417e-09 px). Farneback values are all extremely close to zero, so the relative path/endpoint percentage is sensitive to tiny absolute values.

These findings apply only to this saved clip. They do not support a cross-model or denoising-step claim, a human-label claim, or a sampling-policy claim.

## Artifacts

- Reproducer: [analyze_cogvideox2b_pilot_adjacent_motion.py](analyze_cogvideox2b_pilot_adjacent_motion.py)
- Machine-readable pair and interval values plus verified frame hashes: [cogvideox2b_pilot_adjacent_motion_v1.json](cogvideox2b_pilot_adjacent_motion_v1.json)
- Frame map and endpoint schedule: [registered input map](cogvideox2b_stepsensitivity_inputs_v1.json) and [sparse-readout preregistration](cogvideox_2b_sparse_frame_preregistration_v1.json)
