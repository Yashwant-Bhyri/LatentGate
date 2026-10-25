# CogVideoX-2B pilot_30steps adjacent RGB-MAE and Farneback diagnostic

**Date:** 2026-10-17  
**Scope:** CPU-only image-space measurement of the existing 9-frame CogVideoX-2B `pilot_30steps` clip. No inference, latent access, generation, GPU, cloud, downloads, installation, or tests.

## Integrity and prior coverage

All nine frame hashes match the registered step-sensitivity input map and sparse-readout preregistration; the source manifest hash and run metadata also match. The existing red-mask audit reports **0/9 scoreable frames** (the maximum across its threshold analysis is 203 pixels, below the 300-pixel floor). This diagnostic references that result and does not rerun or retune the mask gate.

Existing registered artifacts for this clip record mask scoreability and a stopped five-anchor readout, but not adjacent RGB-MAE or Farneback transition values. Those pixelwise measurements are computed here with the same definitions as the saved 8-step pilot diagnostic. The 8-step clip’s results are not compared or used to attribute an effect to denoising steps.

## Method

RGB MAE is `mean(abs(RGB_t - RGB_t+1))/255` over all pixels and channels. Farneback uses OpenCV 4.11.0 on grayscale converted from RGB, with `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, and `flags=0`; OpenCL was disabled and one CPU thread used. The registered anchor schedule is `[0,2,4,6,8]`. Endpoint metrics are direct estimates between anchor frames; dense paths sum adjacent-pair metrics.

## All eight adjacent transitions

| Pair | Normalized RGB MAE | Farneback mean magnitude (px) |
|---|---:|---:|
| 0→1 | 0.00083756298 | 0.18713317 |
| 1→2 | 3.3981822e-05 | 4.5796386e-06 |
| 2→3 | 1.733558e-05 | 7.743674e-07 |
| 3→4 | 8.910335e-06 | 7.0864481e-07 |
| 4→5 | 2.5199142e-05 | 6.4211809e-07 |
| 5→6 | 1.3829316e-05 | 7.5087958e-07 |
| 6→7 | 1.1863426e-05 | 5.599721e-07 |
| 7→8 | 5.8551198e-06 | 3.9487368e-07 |

## Dense paths versus registered-anchor endpoints

| Interval | RGB-MAE adjacent path | RGB-MAE direct endpoint | Flow adjacent path (px) | Flow direct endpoint (px) |
|---|---:|---:|---:|---:|
| 0→2 | 0.0008715448 | 0.00086162173 | 0.18713775 | 0.19118944 |
| 2→4 | 2.6245915e-05 | 2.1599265e-05 | 1.4830122e-06 | 9.275683e-07 |
| 4→6 | 3.9028459e-05 | 2.9905365e-05 | 1.3929977e-06 | 7.0426461e-07 |
| 6→8 | 1.7718546e-05 | 1.514842e-05 | 9.5484578e-07 | 5.3090317e-07 |

Across the eight transitions, the normalized RGB-MAE dense path is **0.00095453772** versus **0.00092827478** for the four direct endpoints; **2.75%** of the dense path is above the endpoint total. The Farneback path is **0.18714158 px** versus **0.1911916 px** at endpoints; the endpoint sum is slightly greater, by **2.16%** of the dense path.

Both metrics peak at **0→1**: RGB MAE **0.00083756298**, Farneback mean magnitude **0.18713317 px**. These summarize this saved sequence only; they are not a controlled cross-step inference comparison, ground-truth motion labels, or sampling-policy evidence.

For the peak-mean 0→1 pair, pixelwise Farneback magnitude has median **0 px**, P99 **3.91×10⁻⁵ px**, and maximum **36.68 px**. The mean is tail-sensitive on this transition and should not be read as broad, typical displacement.

## Artifacts

- Reproducer: [analyze_cogvideox2b_pilot_30steps_adjacent_motion.py](analyze_cogvideox2b_pilot_30steps_adjacent_motion.py)
- Machine-readable metrics and verified frame hashes: [cogvideox2b_pilot_30steps_adjacent_motion_v1.json](cogvideox2b_pilot_30steps_adjacent_motion_v1.json)
- Registered inputs and anchors: [step-sensitivity input map](cogvideox2b_stepsensitivity_inputs_v1.json), [sparse-readout preregistration](cogvideox_2b_sparse_frame_preregistration_v1.json)
