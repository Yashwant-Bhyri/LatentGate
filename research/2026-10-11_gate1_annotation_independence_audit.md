# Gate 1 annotation independence audit and v4 correction

- **Checked:** 2026-10-11 12:37 UTC.
- **Active component:** Gate 1 human reference-label reliability.
- **Decision:** supersede v3 packets and use the new v4 packets for any future human submissions. Keep Gate 1 open; no human agreement result exists.

## Defect found

The v3 page made several independently scored fields depend on one another. Its `pointerup` handler assigned `object_visible=true`, `instance_count=1`, and `identity_consistent=true` when a rater drew a box. Its visibility handler assigned `instance_count=0`, `identity_consistent=false`, and cleared the box when a rater selected “No.” This could inflate exact agreement on visibility/count/identity, especially when all raters used the same box-drawing workflow. The 2026-10-09 v3 UI smoke explicitly exercised those defaults; that smoke is now interpreted as evidence of the defect, not evidence of independent label entry.

The local return scan at 12:24 UTC found all v3 templates blank (synthetic 0/34 annotated rows per rater; M0 0/17 per rater) and no separate human-return CSV. Thus no received labels need to be rescored or discarded. The v3 archives are unchanged and their original SHA-256 values still match their archive manifest entries. Their README and manifest now mark them superseded so they are not distributed accidentally.

## Correction

The builder now writes `research/gate1_manual_annotation_packets_v4/`. The drawing handler writes only box coordinates. The visibility handler writes only visibility. Neither action modifies count, identity, or box. Validation requires explicit visibility, count, identity, and box decisions; an absent target must be explicitly recorded as visibility No, count 0, identity No/uncertain, with no box. V4 preserves the existing blinded left/right order variation and offline data-only pages.

No new external repository, dataset, model, simulator, or license was introduced. The page reuses the existing local Kubric fixture and the already-saved M0 sample. The ZIPs include no unblinding key, contain blank annotation templates, and the embedded HTML audit found no external network code or assets.

## Verification record

- The local v4 builder completed in **0.528 s** on Darwin arm64 / Python 3.12.3. No GPU was used; peak VRAM was not allocated or measured. No model inference ran.
- A browser smoke of isolated copies of the synthetic rater-01 page verified that selecting absent leaves count/identity blank; drawing does not fill categorical fields; selecting visible leaves count/identity blank; manually entering count and identity completes the frame; reload preserves the explicit values; and changing visibility back to absent leaves count, identity, and box unchanged while validation marks the row incomplete.
- A fresh packet key in standard Chrome exported a 34-row CSV with exactly one synthetic absent row (visibility false, count 0, identity false, no box) and 33 blank rows. That synthetic value deliberately contradicts the visible object in the frame; it tests serialization only and is invalid as an annotation. The file is named [`smoke_export_intentionally_invalid_not_for_scoring.csv`](qa/gate1_v4_ui_roundtrip/smoke_export_intentionally_invalid_not_for_scoring.csv), SHA-256 `85987434275b8ff399482edd02833deb983d86db4d4351cc2916baa9cef4b43b`. An earlier scratch-key export included two extra browser-local smoke rows; those downloads were removed and no packet ZIP or human return file was affected. Each future export should be checked for the expected row count and retained labels before scoring.
- Automated CSV import/resume could not be checked because the ChatGPT Chrome extension refused local-file selection while **Allow access to file URLs** is disabled; no permission was changed. The blank CSV fallback remains available, and the scorer enforces explicit absent-state semantics. The temporary annotation page was closed; the export artifact remains explicitly marked synthetic.
- The ZIP/CSV audit confirms all four v4 packet templates are blank (synthetic 0/34 rows each; M0 0/17 rows each); all four archive hashes match the v4 manifest; ZIP integrity passes; no private key is present in any ZIP; no external URL-backed code/assets were found. The same archive-hash checks pass for the preserved v3 ZIPs. See [`packet_integrity_audit`](qa/gate1_v4_packet_integrity_audit.json) and [`browser smoke record`](qa/gate1_v4_ui_roundtrip/report.json).
- A separate CPU-only check confirms the scorer accepts an explicitly absent row only with count 0, identity false, and no box, and rejects an absent row marked identity true. This protects manual CSV fallback input if the UI export fails: [`absent-identity guard result`](gate1_absent_identity_guard_v1.json). It uses synthetic rows, not human labels.
- Builder source SHA-256 at packet build: `58a0cb7c2729d59c438d384af1cc3e88e869caf5747500425a591ae3c2419825`.
- V4 packet manifest SHA-256 after post-build QA: `a1498dae7d9ce8cdcf11222a55fd21f96851811de0647caf3bab3201b763e58f`.
- Browser smoke report SHA-256: `f4b43449996c2799a8a98dda3e9f1446a018877a9bd2a670ac23b7fd193be9dc`.
- Static packet audit SHA-256: `87dfa50333f65b739f2178b290d1b2d47123a7057854bfe4dced3f1f485ba6bf`.

## Research implications and next falsifiable step

This corrects the measurement protocol, not the science result. It provides no evidence about human-human agreement, generated-video tracking, observer validity, path adherence, or policy efficacy. The existing global benchmark/tool review still motivates scoring tight-box centers, visibility/identity, and apparent extent separately; this UI correction makes the visibility/count/identity judgments independent of the box action.

The next step is two independent human submissions for both mirrored 17-frame Kubric clips using the v4 synthetic packets. Score with `--source-width 160 --source-height 96 --expected-clips 2`; require at least 95% joint visibility/count/identity agreement, at least 90% unique visible center-pair coverage, and p95 tight-box-midpoint disagreement at most 0.5 target-grid cell per axis both pooled and for each clip. A pass is only a process screen; full reliability still requires a second independently rendered mirrored-seed block. Any v3 return must be excluded. Keep the M0 convenience packets separate, and do not advance to the generated-video observer or policy experiment until the reliability gate passes.


## Later coordinate-scale correction (2026-10-11 13:01 UTC)

A follow-up audit found that the v4 synthetic images are 160×96 while the scorer's former 16 px-per-cell conversion referred to the 832×480 Wan target. The old raw 8 px rule therefore exceeded the intended half-cell agreement tolerance by about 5×. The v4 archives remain unchanged; the source-resolution mapping and corrected Kubric invocation are in the [coordinate-scale audit](2026-10-11_gate1_coordinate_scale_audit.md). No actual label results existed to invalidate.
