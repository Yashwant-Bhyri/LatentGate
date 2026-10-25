# Gate 1 p95 tail sensitivity

**Audit time:** 2026-10-11 15:09 UTC  
**Component:** Gate 1 framewise human-reference agreement for fixed-camera 2D path measurement.

## Decision

Keep the preregistered p95 ≤0.5-cell threshold and do not add an unsupported numeric maximum-error cutoff. However, worst-frame review is required and its disposition must be recorded even when pooled and per-clip p95 values pass. The p95 is a process screen, not proof that every frame is localized within half a cell.

## Synthetic sensitivity case

`validate_gate1_agreement_scorer.py` builds two 17-frame mirrored clips with exact categorical agreement and zero box disagreement everywhere except `mirror_left`, frame 16. There, rater B's box is shifted horizontally by 2.5 control-grid cells. It runs the same agreement `summarize()` function and current registered thresholds used by the annotation scorer.

- `mirror_left` p95 x disagreement: 0.5 cell; maximum x disagreement: 2.5 cells at frame 16.
- Pooled p95 x disagreement: 0.0 cell.
- Pooled and both per-clip screens: pass.
- Runtime: 0.000426 s on Darwin arm64 / Python 3.12.3; CPU only, GPU false, peak VRAM null. No model inference, image rendering, external data, or human labels.

For 17 frames, the linear-interpolated 95th percentile lies 20% of the way from the second-largest value to the largest. One outlier therefore contributes only 0.2×its magnitude to that clip's p95. With 34 pooled frames, one outlier falls beyond the pooled p95 rank and contributes nothing. This is why reporting the maximum and reviewing its frame matters even when the p95 gate passes.

## Required review and next step

When the real two-rater CSVs arrive, report the maximum radial center disagreement frame for each clip and the per-axis maximum frames when they differ. Review those original frames and both boxes while still blind. Correct and rescore a data-entry/export mistake; if visibility, count, or identity is uncertain, preserve that categorical uncertainty and exclude the frame from path geometry. Record the disposition without changing the p95 threshold. A one-pair pass still requires the independently rendered second mirrored-seed block.

The new case validates percentile behavior only. It does not measure human reliability, generated-video observer accuracy, or steering efficacy.
