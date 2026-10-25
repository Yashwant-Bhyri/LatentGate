# M0 temporal-change localization readout

**Experiment:** `EXP-LATENTGATE-M0-TEMPORAL-CHANGE-LOCALIZATION-001`  
**Status:** preregistered offline exploratory analysis; no new generation, model weights, GPU, or human labels.

## Question

Across adjacent temporal positions in the saved Wan M0 latents, do local latent changes align with visible image changes, especially the moving red-object proxy, or are they spread mainly over the background? This is a within-one-clip diagnostic of saved temporal features, separate from the earlier static object-location readout and five-anchor sampling audit.

## Frozen inputs and mapping

- Source run: `m0/runs/wan_t4_2026-10-02_retry5/run_001`, one 17-frame 832×480 clip and one prompt/seed.
- Inputs: the seven manifest-listed post-step latent tensors at steps `[0, 4, 9, 14, 19, 24, 29]`, final decoded RGB frames at `[0, 4, 8, 12, 16]`, the immutable M0 manifest, `m0/analysis/wan_t4_retry5_frame_motion_audit.json`, and `research/m0_temporal_sampling_audit_v1.json` to verify the nominal anchor mapping.
- The five latent-time slices map to RGB anchors `[0, 4, 8, 12, 16]`; each adjacent latent pair is therefore compared with a four-frame RGB interval, not a one-frame interval.
- Verify every tensor and sampled RGB frame against the M0 manifest. Recompute the fixed red proxy (`R > 150`, `G < 125`, `B < 125`) and require its pixel counts and centroids at each anchor to match the existing frame-motion audit before scoring.
- Pool each 16×16 RGB patch into a 52×30 grid. The corresponding model patch is a 2×2 block of the 16-channel, 60×104 latent map.

## Frozen features and targets

- For each denoising snapshot and each of the four adjacent latent-time pairs, compute the signed latent difference `z[t+1] - z[t]`. Flatten each aligned 2×2 spatial block across 16 channels into a 64-value feature vector. Do not mix post-step tensors with predicted-clean tensors.
- Primary target: fraction of pixels in each 16×16 patch whose red-proxy membership changes between the two RGB anchors (XOR mask). Score positive patches as those with target occupancy greater than zero.
- Descriptive targets: mean absolute RGB difference per pixel in each patch, plus the red-proxy swept region (union of the two anchor masks).
- Fit fixed, train-fold-standardized ridge regression (`alpha=1.0`, intercept) to predict red-XOR occupancy. Use leave-one-transition-pair-out evaluation: train on all spatial tokens from the other three temporal pairs and score the held-out pair. Do not tune features or hyperparameters.
- Compare against a spatial-site prior: mean red-XOR occupancy at each grid location across the three training transitions. Report average precision (AP) for changed patches and RMSE for fractional occupancy for both the latent readout and prior. Also report AP from the held-out mean absolute latent-difference energy map, Spearman correlation of that energy with patch RGB difference, and area-weighted latent-difference energy inside the red swept region versus outside it and inside the red-XOR region versus outside it.
- Report all four held-out transitions separately for every snapshot. No formal pass/fail threshold is set; interpret the values descriptively and do not treat the 28 snapshot-by-transition rows as independent samples.

## Stop conditions and interpretation boundary

Stop without a result if a required file is missing, any manifest hash differs, tensor/frame dimensions or temporal mapping disagree with the run record, or the fixed red-proxy check fails. Do not substitute pooled transformer summaries or attention statistics for spatial tensors.

Four correlated transitions from one generated clip are not independent samples. RGB differences and the red threshold are same-output proxies, not human labels or ground truth; red-mask XOR can include threshold shimmer, and RGB change also includes lighting and appearance changes. The analysis cannot establish object understanding, motion forecasting, physical geometry/contact, causal control, action-policy efficacy, cross-clip generalization, or completion of M1–M3. A positive within-clip alignment would only motivate an independently labeled, multi-clip follow-up.

## Implementation and result

- Preregistered runner: `research/run_m0_temporal_change_localization.py`
- Machine-readable result: `m0/analysis/m0_temporal_change_localization_v1.json`
- Run the saved-data analysis on CPU only, without loading model weights, running inference, or generating frames. Record source hashes and the complete per-snapshot/per-transition metrics in the JSON result.

The run completed after preregistration. At post-step 029, the 64-value ridge readout averaged AP 0.307 for red-XOR patches versus 0.671 for the spatial-site prior and beat that prior on only one of four held-out transitions. The untrained absolute-delta energy map averaged AP 0.546; its energy was enriched inside red-XOR regions on three of four transitions (mean inside/outside ratio 2.255), while its Spearman correlation with patch RGB change was positive on three of four transitions. Early snapshots were near chance by the direct-energy metric. This is mixed same-clip evidence: late latent changes sometimes concentrate around visible red-proxy changes, but the learned readout is less reliable than a prior based on where changes occurred in the other intervals, and the first interval does not show the same localization.

| Post-step | Ridge AP | Spatial-site prior AP | Absolute-delta energy AP | Mean red-XOR energy enrichment |
|---|---:|---:|---:|---:|
| 000 | 0.025 | 0.671 | 0.036 | 1.002 |
| 004 | 0.025 | 0.671 | 0.036 | 1.003 |
| 009 | 0.025 | 0.671 | 0.037 | 1.006 |
| 014 | 0.029 | 0.671 | 0.041 | 1.014 |
| 019 | 0.051 | 0.671 | 0.059 | 1.040 |
| 024 | 0.240 | 0.671 | 0.309 | 1.179 |
| 029 | 0.307 | 0.671 | 0.546 | 2.255 |

These averages are descriptive across four correlated intervals from one sample. The reported positive intervals and the weak/negative first-interval result are preserved in the JSON output; no milestone gate advances.
