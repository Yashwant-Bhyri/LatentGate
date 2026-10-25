# Gate 1 annotation-coordinate scale audit

- **First identified:** 2026-10-11 13:01 UTC; CPU diagnostic rerun at 13:06 UTC.
- **Component:** Gate 1 human reference-label reliability (still the earliest open gate).
- **Decision:** preserve the v4 packets, but replace the scorer's raw-pixel-to-16-pixel-cell conversion with an explicit source-frame → target-frame → target-grid mapping. Do not score any returned CSV with the old scorer.

## Defect and evidence

The v4 synthetic Kubric packet embeds frames whose metadata declares `width=160`, `height=96`; `kubric_gate0_fixture_v1/fixture_report.json` independently records `frame_size_wh: [160, 96]`. The HTML `point()` function maps pointer positions through the SVG viewBox and clamps them to each frame's `width` and `height`, and the CSV stores those raw frame-pixel coordinates. By contrast, the intended Wan measurement grid is 52×30 at 832×480, or 16 target-image pixels per cell.

The prior scorer divided a raw annotation-coordinate delta by 16. That treated a source-pixel difference as if it were a target-image pixel difference. For the 160×96 Kubric images, an 8 px raw x disagreement maps to 41.6 px at target width, or 2.6 control cells; 8 raw y pixels map to 40 target px, or 2.5 cells. Thus the earlier “0.5 cell / 8 px” rule was about five times too permissive when applied directly to these low-resolution packets. The native-resolution M0 packet is 832×480 and does not have this scale mismatch.

No returned human annotations were present, so no agreement result was scored under the incorrect conversion. The v4 packet archive bytes remain unchanged. Its field-independence and export QA remain valid; only the agreement metric's resolution conversion was wrong.

## Correction

`score_annotation_agreement.py` now requires source width and height, maps x and y separately into target resolution, then reports disagreement in target pixels and 52×30 grid cells. It also rejects boxes outside the declared source-frame bounds. For the current synthetic packets, the half-cell p95 screen corresponds to source-image p95 limits of **1.5385 px on x** and **1.6 px on y**, not 8 raw pixels. For the native 832×480 M0 packet, the equivalent raw limits remain 8 px per axis.

The corrected command for the paired Kubric packets is:

```sh
python3 research/score_annotation_agreement.py rater_01.csv rater_02.csv \
  --source-width 160 --source-height 96 --expected-clips 2 \
  --output research/gate1_synthetic_agreement_screen_v2.json
```

For the separate native Wan M0 packet, pass `--source-width 832 --source-height 480 --expected-clips 1`. Keep the p95 ≤0.5 target-grid-cell threshold, ≥95% joint visibility/count/identity agreement, ≥90% unique visible center-pair coverage, and pooled plus per-clip pass requirement. These are project process-screen thresholds, not a published consensus standard; 17 frames per clip are correlated and do not form 17 independent trials.

## CPU validation

The updated diagnostic [`validate_gate1_agreement_scorer.py`](validate_gate1_agreement_scorer.py) writes [`gate1_scorer_tail_diagnostic_v3.json`](gate1_scorer_tail_diagnostic_v3.json). On synthetic 160×96 boxes mapped to 832×480 / 52×30, it verifies:

- one source pixel maps to 0.325 x-cells and 0.3125 y-cells;
- the old raw 8 px tolerance would have been 2.6 x-cells and 2.5 y-cells;
- a pooled p95 of 5.25 target px cannot hide a mirrored clip whose p95 is 15 target px / 0.9375 cell;
- a missing expected mirror and a 161 px box edge outside a 160 px frame are rejected.

The latest diagnostic completed in **0.000607 s** on Darwin arm64 / Python 3.12.3 at 13:06 UTC. It used CPU only; GPU false, peak VRAM not allocated (null). Inputs are synthetic scorer fixtures, not human labels. The run establishes unit conversion and validation behavior only; it does not measure annotator reliability, observer accuracy, or policy efficacy.

**Next falsifiable step:** collect two independent v4 annotations for both mirrored clips, verify the exported CSVs contain explicit fields, and run the command above. Require each clip and the pooled screen to pass in target-grid units. If labels fail the corrected threshold, improve annotation resolution/instructions or the source fixture and collect fresh labels; do not relax units to preserve a pass. A passing pair remains a process screen; full reference reliability still needs a second independently rendered mirrored-seed block.


Current source/result SHA-256: scorer `398cac964ea0c0cb94156192375616bf9c654eafc71ef3a4fa9169d86bcd2398`; diagnostic script `ce5918ede5a3d83a0ee7af4b8bd70c6eed93c7bcb0a0d0eba39c921ea9f97557`; result `44ecf673f79fef403cbc5fa2f39d35855eca859ab109df60bbcf5e8ff1157929`.
