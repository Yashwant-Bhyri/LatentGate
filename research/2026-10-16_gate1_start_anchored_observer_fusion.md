# Gate 1 start-anchored observer-fusion diagnostic

**Checked:** 2026-10-16 13:20 CST  
**Experiment:** `EXP-SAM2-RED-START-ANCHORED-FUSION-DIAGNOSTIC-001`  
**Scope:** CPU-only analysis of saved SAM 2 and red-centroid outputs; no new inference, no threshold fitting, and no human-label or policy test.

## Question and fixed rule

The 2026-10-15 stress report showed complementary failure patterns: SAM 2 continued to track through a static red distractor while the global red centroid failed closed, whereas partial pole occlusion left some red-centroid anchors confidently wrong. This diagnostic asks whether a fixed, start-anchored disagreement rule can combine their motion estimates without admitting the known distractor failure.

For the four saved stress composites (left/right paths × pole occlusion/static red distractor), I reused the framewise SAM 2 box centers and red-proxy centers and the existing Kubric projected-center truth. At controller anchors `[0, 4, 8, 12, 16]`, I scored future-anchor residuals in the 52×30 control grid:

`((observer_i - observer_0) - (truth_i - truth_0)) * [52/160, 30/96]`.

The red estimate was eligible only when its saved `scoreable` flag was true and the existing 0.65 stability heuristic passed at both frame 0 and the current anchor. The candidate selects the mean of SAM 2 and red start-relative residuals only when their L2 disagreement is at most 0.5 control-grid cell; otherwise it falls back to SAM 2. The 0.65 value is the existing in-sample heuristic. The 0.5-cell disagreement limit was fixed before running this diagnostic and was not tuned to these outcomes, but it was **not** previously registered as a production threshold.

## Results

Mean radial path-residual error is over four correlated future anchors per case; fusion counts are descriptive, not independent sample counts.

| Composite | SAM 2 mean L2 (cells) | Selected candidate mean L2 (cells) | Red eligible | Fused anchors |
|---|---:|---:|---:|---:|
| Left, fixed pole occlusion | 0.2640 | 0.2491 | 4/4 | 1/4 |
| Left, static red distractor | <0.0001 | <0.0001 | 0/4 | 0/4 |
| Right, fixed pole occlusion | 0.2173 | 0.1361 | 3/4 | 2/4 |
| Right, static red distractor | 0.1735 | 0.1735 | 0/4 | 0/4 |

On these four composites, the disagreement rule accepted three of sixteen future anchors. The two pole-occlusion cases had lower mean path residual under the selected candidate; the two distractor cases fused none and matched the SAM 2 fallback. This is a small development-set signal that the two estimates may be complementary. It does not establish that the rule is reliable or that it improves any generated video.

## Decision and limits

Keep this as an observer-preparation candidate only. Do not deploy it in the denoising loop or call it a validated observer. SAM 2 is not ground truth; both observers have biases, and the start anchor hides absolute first-frame localization error. The result covers two correlated path renders, two hand-designed nuisances, an oracle-prompted SAM 2 proposal, and four temporally correlated future anchors per case. It does not score identity, mask coverage, visibility labels, background stability, apparent size, or clean photometric shifts in this fusion comparison. It does not validate human-label reliability, transfer to Wan frames, actuator behavior, or policy efficacy. The generated Kubric artifact license remains unresolved as recorded in the fixture audit; this analysis adds no redistribution rights.

The earliest Gate 1 requirement remains two independent complete v5 human-label submissions, followed by exact packet-key validation, pooled and per-clip agreement, blind worst-frame review, and guarded truth scoring. If that gate passes, freeze this candidate before testing on newly rendered mirrored seeds with separate occlusion, lighting, and distractor conditions. Falsify the candidate if it increases held-out path error over SAM 2-only, accepts any wrong-identity/absent-object anchor, or falls below the registered clean-anchor coverage requirement. Report path error, visibility/identity, background stability, and apparent size separately.

## Reproducibility and sources

- Inputs: [`sam2_visibility_stress_v1/report.json`](sam2_visibility_stress_v1/report.json), SHA-256 `001c3f4d14be2c92e62c52edb3e62fc6c4e7c00b4e316c536dab364912aa69a6`; [`kubric_gate0_fixture_v1/ground_truth.json`](kubric_gate0_fixture_v1/ground_truth.json), SHA-256 `0dd6d5518de3c82d2e73d912c44fc712978d65dd93c38f8eb24d37500f8ad56b`.
- Analyzer: [`analyze_sam2_red_observer_fusion.py`](analyze_sam2_red_observer_fusion.py), SHA-256 `5f8c6f040815af93260794f519141acaee9cbe4466b2576360c95c452e3737e3`.
- Machine result: [`sam2_visibility_stress_v1/fusion_diagnostic_v1.json`](sam2_visibility_stress_v1/fusion_diagnostic_v1.json), SHA-256 `6d279f84c98f8a143b4a20a69a08ed422c4115edb20fe0e83c3b3c98e392ff73`.
- Runtime: 0.0252 s, CPU-only, macOS arm64, Python 3.12.3; peak VRAM not applicable. No model, network, external data, GPU, Colab/Kaggle runtime, or paid resource was used.
- The latest `Research LatentGate Methods` chat was read on 2026-10-16. Its October 11 primary-source report remains the latest; it is already reconciled in [`2026-10-11_full_research_chat_reconciliation.md`](2026-10-11_full_research_chat_reconciliation.md). It continues to recommend DiTraj as the backbone-matched starting point while explicitly noting that policy efficacy remains unmeasured.
