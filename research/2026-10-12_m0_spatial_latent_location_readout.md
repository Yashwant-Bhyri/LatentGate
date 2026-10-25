# M0 spatial-latent location readout

**Status:** preregistered offline exploratory analysis; no new generation, model weights, GPU, or human labels.  
**Experiment:** `EXP-LATENTGATE-M0-SPATIAL-LOCATION-READOUT-001`

## Question

Can a fixed linear readout recover the approximate image-space location of the red object from saved spatial latents or predicted-clean latents, better than a coordinate-only prior, on the one existing M0 clip? This checks whether the saved internal state retains a localizable signal over denoising. It is an exploratory same-clip proxy analysis only.

## Frozen inputs and labels

- Source run: `m0/runs/wan_t4_2026-10-02_retry5/run_001` (17 frames, 832×480, 30 denoising steps, one prompt/seed).
- Feature snapshots: all seven `latents/post_step_{000,004,009,014,019,024,029}.pt` tensors and all four `predicted_clean/pre_step_{004,014,024,029}.pt` x0-hat tensors. Load on CPU and verify file hashes against the source run manifest.
- Labels: the existing fixed RGB threshold (`R > 150`, `G < 125`, `B < 125`) on the saved final decoded frames at indices `[0, 4, 8, 12, 16]`, matching the 17-frame / five-latent-time index relation in the saved run config. This is a color proxy, not human annotation, segmentation truth, or a physical object track. Verify the recomputed proxy counts and centroids against the existing frame-motion audit before analysis.
- Spatial mapping: each latent snapshot has shape `[1,16,5,60,104]`. Flatten each 2×2 latent patch across its 16 channels to a 64-value feature vector on a 52×30 grid; pool the corresponding 16×16 RGB proxy pixels to a continuous occupancy target in `[0,1]`.

## Frozen readout and scoring

For each snapshot independently, use leave-one-latent-timepoint-out cross-validation: train on all 52×30 spatial patches from four latent times and score all patches at the held-out time. Standardize each feature using only training rows (zero variance maps to scale 1), fit an intercept plus ridge regression with `alpha=1.0` to continuous proxy occupancy, and do not tune features or hyperparameters. For the predicted location, subtract the training target mean, clamp negative residuals to zero, and take the positive residual-weighted center. If its mass is zero or non-finite, mark that fold unscorable. Compare against a coordinate-only prior: the training frames' mean occupancy map, used both as the heatmap ranking and as the weighted-center predictor.

Report each held-out latent time separately, center error in 52×30 control-token cells and pixels, and average precision for target patches with proxy occupancy ≥0.25. Also report the coordinate-only prior on the same held-out frame. Do not aggregate frames as independent samples; summarize only the five within-clip timepoint folds and show their full values.

## Stop conditions and interpretation boundary

Stop without a score if an expected file/hash/shape is missing, decoded frames do not match the registered 832×480 dimensions, the proxy count/centroid check fails, or the temporal/spatial mapping cannot be confirmed from the run metadata. Do not substitute pooled transformer summaries for spatial maps.

Five temporally related positions from one prompt/seed do not provide independent clips, a holdout scene, or a human-reference evaluation. A positive result would establish only that this fixed readout can localize the color proxy within this particular saved clip better than the fixed spatial prior. It would not show generalizable object understanding, geometry/contact representation, forecastability, causal controllability, policy efficacy, or M1/M2/M3 completion. A null result would not prove spatial information absent; feature choice and the probe's small sample may miss it.

## Implementation and result

- Analysis code: `research/run_m0_spatial_latent_location_readout.py`
- Machine-readable output: `m0/analysis/m0_spatial_latent_location_readout_v1.json`
- Input snapshots and frame PNG hashes are recorded in the output and cross-checked against the immutable M0 manifest.
- The frozen run completed on CPU without loading model weights or generating new samples. All 11 snapshot files passed the M0 manifest SHA-256 check; the five sampled decoded frames also matched the manifest, and the red-mask pixel counts/centroids matched the existing frame audit.

| Saved state | Mean held-out AP | Mean center error (52×30 token cells) | Coordinate-prior AP | Coordinate-prior center error |
|---|---:|---:|---:|---:|
| post-step 000 | 0.045 | 12.475 | 0.940 | 0.906 |
| post-step 004 | 0.082 | 11.748 | 0.940 | 0.906 |
| post-step 009 | 0.300 | 10.577 | 0.940 | 0.906 |
| post-step 014 | 0.737 | 8.981 | 0.940 | 0.906 |
| post-step 019 | 0.966 | 7.090 | 0.940 | 0.906 |
| post-step 024 | 0.994 | 3.902 | 0.940 | 0.906 |
| post-step 029 | 0.998 | 0.986 | 0.940 | 0.906 |
| x0-hat pre-step 004 | 0.987 | 3.046 | 0.940 | 0.906 |
| x0-hat pre-step 014 | 0.997 | 0.711 | 0.940 | 0.906 |
| x0-hat pre-step 024 | 0.997 | 0.927 | 0.940 | 0.906 |
| x0-hat pre-step 029 | 0.998 | 0.986 | 0.940 | 0.906 |

These are means across five correlated timepoint holdouts from one clip. The x0-hat at pre-step 014 was the only snapshot with a lower mean center error than the coordinate prior (0.711 vs 0.906 token cells, about 11.4 vs 14.5 px); the gain is small and one fold contains a large localization error at pre-step 004. Late snapshots rank proxy-positive patches highly, but the coordinate prior is already strong because the red object is large and moves only a few control-grid cells. The readout therefore shows late spatial features align with this clip's red proxy, but does **not** establish a robust localization improvement, object understanding, generalization, or causal/action usefulness. Retry4 and retry5 share byte-identical final frames and latents, so they are one generation, not a second sample.

Full per-timepoint results, hashes, and provenance are in the JSON output. The interpretation boundary above remains binding; do not promote this analysis into an M1 failure atlas or an M2 predictive geometry probe.
