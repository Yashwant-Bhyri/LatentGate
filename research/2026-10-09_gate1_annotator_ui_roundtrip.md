# Gate 1 annotator UI round-trip smoke (not human labels)

- Checked: 2026-10-09 01:57 UTC
- Source: isolated temporary copy of `synthetic_reliability/rater_01.zip` `index.html`; only the browser `packetKey` was changed to `ui-roundtrip-smoke-20261009` to isolate localStorage. The source ZIP and all rater packets were not edited.
- Browser: local Codex in-app browser, page served from `127.0.0.1:18761`; no remote service or account.
- Device: project host inventory identifies a MacBook Pro with Apple M3; no model inference. GPU use: false; peak VRAM: not allocated (N/A). Interactive duration was not instrumented.
- Procedure: drew one synthetic box on frame 0; UI auto-filled visible/count/identity as true/1/true and advanced to 1/17 complete. Reload retained that annotation. Downloaded CSV, cleared the box (progress returned to 0/17), imported the downloaded CSV, and observed progress return to 1/17. Imported SVG overlay attributes were x=67, y=37.41, width=24, height=25 px, matching the exported box.
- CSV artifact: `research/qa/2026-10-09_gate1_annotator_roundtrip/smoke_not_human_rater_labels.csv`; 34 rows (two 17-frame clips), one nonblank synthetic annotation row; SHA-256 `62984ff39fd4e5b7dce10e248c7147da7358ad0ecd2f5cca80d5b13bb3404468`. This is a UI smoke annotation created by the agent, emphatically **not a human rater submission**; do not score it as Gate 1 evidence.
- Result: manual drawing, reload persistence, CSV export, and CSV resume/import work in this local browser smoke. This closes only the packet UI mechanics check. It does not establish inter-rater reliability, observer accuracy, or policy efficacy; the four actual rater ZIPs remain blank.
