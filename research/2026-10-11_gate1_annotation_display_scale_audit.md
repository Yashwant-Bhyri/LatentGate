# Gate 1 annotation display-scale audit

**Audit time:** 2026-10-11 13:12 UTC  
**Scope:** earliest open component only: human reference-label reliability for fixed-camera, single-object 2D paths.

## Finding

The v4 offline annotator sets `img.width` and `img.height` to the source frame dimensions. Its CSS then uses `width:auto;height:auto`, so the 160×96 Kubric frames render at their natural 160×96 CSS-pixel size on a sufficiently wide viewport. The CSS `max-width` only limits oversized images; it does not enlarge small images. The SVG overlay and pointer transform correctly preserve source-pixel coordinates, but a source-space tolerance of 1.5385 px horizontally and 1.6 px vertically (the corrected 0.5-cell p95 gate) was correspondingly only about 1.5 CSS pixels on screen. That is an avoidable human-labeling precision burden.

This finding came from the packet builder and embedded HTML source, not from human labels or an assumed agreement result. The v4 browser smoke covered independent fields, persistence, and CSV export; it did not test or change the image display size.

## Change and evidence

The builder now produces **v5** packets. It enlarges images to `min(960, max(640, source_width))` CSS pixels, subject to the existing viewport max-width, and enables pixelated display for the low-resolution synthetic frames. The SVG `viewBox` remains the declared source width and height and pointer coordinates still pass through the SVG screen transform; CSV boxes therefore remain in source-frame pixels. At the intended 640 CSS-pixel display width, the current half-cell screen corresponds to about 6.15 CSS pixels horizontally and 6.4 vertically. On narrow viewports, responsive CSS can shrink the image; desktop annotation is preferable.

The v5 build and local packet audit completed on Darwin arm64 / Python 3.12.3 in 0.346 s of builder-reported CPU wall time. GPU use was false; peak VRAM was null/not allocated; no model or vision inference ran. All four ZIPs match their manifest SHA-256 values and pass ZIP integrity. Each contains only `index.html`, a blank `annotation_template.csv`, and `README.txt`; the synthetic templates contain 34 frames each and the M0 convenience templates contain 17 each. Source dimensions (160×96 Kubric; 832×480 M0), source-coordinate mapping, independent label fields, and blank scored fields were verified.

A focused v5 browser smoke then loaded the synthetic packet in the Codex in-app browser over loopback. In the available pane, the 160-pixel source width displayed at 622 CSS pixels (3.8875×); the SVG retained `viewBox="0 0 160 96"`. A source coordinate `[37.25, 28.5]` made a screen-transform round-trip with 0 px error. Calling the browser export function while intercepting Blob/anchor handling produced one header plus 34 rows, all scored label fields blank; no file download occurred. The smoke report is [`gate1_v5_ui_roundtrip/report.json`](qa/gate1_v5_ui_roundtrip/report.json). This validates loading, coordinate transform math, and CSV serialization only; it does not measure visual box placement or human agreement.

At 2026-10-11 14:14 UTC, all v3, v4, and v5 human template CSVs were reopened: each synthetic packet remains 0/34 annotated rows and each M0 convenience packet remains 0/17. No label agreement result exists. V3 remains invalid due to coupled labels; v4 is retained as history but superseded for annotation because its low-resolution frames display at natural size.

## Decision and falsifiable next step

Stay at Gate 1. The v5 display/coordinate/export mechanics smoke passed. Next, use the v5 synthetic rater pair for two independent framewise annotations of both mirrored clips. Score with `--source-width 160 --source-height 96 --expected-clips 2`; require the existing pooled and per-clip rules, including p95 ≤0.5 target-grid cell per axis. If the screen fails, inspect its frame-level errors against Kubric truth and determine whether annotation instructions or image presentation need repair; do not relax the coordinate threshold. A pass remains a one-pair process screen and still requires a second independently rendered mirrored-seed block.

The separate v5 M0 packet pair remains a one-clip observer convenience check and must not be treated as observer validation or policy evidence. No generated-video observer or policy run is released by this audit.
