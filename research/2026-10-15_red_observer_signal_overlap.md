# Red-observer feature overlap after visibility stress

**Checked:** 2026-10-15 08:52 UTC  
**Experiment:** `EXP-RED-OBSERVER-FEATURE-DISCRIMINABILITY-001`  
**Scope:** CPU-only analysis of the prior synthetic visibility stress and clean fixture anchors. No SAM 2 rerun and no threshold change.

## Result

The ten clean control anchors from the same two Kubric path renders had red-threshold area-CV values from 0.3000 to 0.3331 and stability values from 0.6743 to 0.7678. The three pole-occlusion anchors previously accepted by the registered stability cutoff, despite >0.5-cell calibrated path error, had stability values 0.7070, 0.7539, and 0.8187. Their stability range overlaps the clean range, so raising the single stability cutoff enough to reject all three would also reject clean anchors.

Those three false-accept anchors had area-CV values 0.2477, 0.4072, and 0.2423. An interval copied from the clean anchors would flag all three in-sample, but a small global brightness sensitivity control undermines that idea: applying brightness factors 0.8, 0.9, 1.1, and 1.2 to the ten otherwise-clean anchors placed **all 40** measurements outside the original area-CV interval. All remained scoreable, but the existing stability cutoff accepted 0/10 anchors at 0.8–0.9 and 10/10 at 1.1–1.2. This feature therefore reacts to photometric changes as well as partial occlusion; an interval fit to two simple clips is not a safe visibility gate.

## Decision and next step

Do not replace the registered 0.65 heuristic with an area-CV range or a cutoff tuned on these frames. A single red-centroid confidence feature does not identify the partial-occlusion errors, and area-CV is photometrically brittle in this fixture. Keep the red observer secondary and fail closed when independent visibility evidence is absent. A candidate visibility-aware measurement module must be selected on separate development renders and then tested on newly rendered held-out paths with partial occlusion and lighting variation; it must reject every >0.5-cell false-accept anchor while retaining at least 90% of clean eligible anchors. Report path error, visibility/identity, background stability, and apparent size separately.

The first valid Gate 1 action remains two independent complete v5 human-label CSVs, followed by the registered pooled/per-clip agreement screen, blinded worst-frame review, and guarded truth score. This CPU diagnostic does not substitute for those labels or validate any observer on generated Wan frames.

## Reproducibility

- Analysis: [`analyze_red_observer_visibility_signals.py`](analyze_red_observer_visibility_signals.py)
- Result: [`red_feature_discriminability_v1.json`](sam2_visibility_stress_v1/red_feature_discriminability_v1.json)
- Inputs: pinned clean Kubric fixture, fixture truth, and [`SAM 2 visibility-stress report`](sam2_visibility_stress_v1/report.json)
- Runtime: 0.0599 s, Darwin arm64 / Python 3.12.3, CPU only; no GPU and peak VRAM N/A.
- No model, dataset, or external package was downloaded; no data or code outside the existing research workspace was reused. The fixture's existing Kubric/Blender output-license question remains open as recorded in the fixture license audit.
