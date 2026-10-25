# Gate 1 v5 synthetic-reliability packet: static audit

**Date:** 2026-10-16  
**Method:** read-only inspection of the two ZIPs in memory. No files were extracted, the annotator was not opened or executed, no tests were run, and `PRIVATE_UNBLIND_KEY.json` was not opened.

## Findings

Both archives contain exactly three members: `index.html`, `annotation_template.csv`, and `README.txt`. Their SHA-256 hashes are:

- `rater_01.zip`: `0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8`
- `rater_02.zip`: `8a660acb5bd77855b82c13b27cc951f8d2e4d6131e38e0144e83638e2f92e7ff`

The inspected HTML has no external `src`, `href`, or form-action references; no literal HTTP(S) URLs; and no direct `fetch`, XHR, WebSocket, `sendBeacon`, EventSource, or Axios references. All 34 frame images in each HTML are embedded `data:image/png;base64` URIs. The page persists labels in browser `localStorage`; CSV export creates a local Blob download. This is a static-source finding, not a runtime network trace.

The CSV template and embedded `columns` constant share this exact 11-column order:

`blind_clip_id, frame_index, image_file, object_visible, instance_count, identity_consistent, bbox_x1_px, bbox_y1_px, bbox_x2_px, bbox_y2_px, notes`

Each template has 34 blank data rows covering two blind clip IDs, 17 frames each (indices 0–16). In each packet, the embedded 34-frame array matches its template exactly on `(blind_clip_id, frame_index, image_file)`. Export writes the header from `columns`, then iterates the embedded frame array and looks up each annotation by blind clip and frame index, preserving that packet's mapping and order. The two packets use a different order for the blind clips.

Visibility, instance count, identity, and box are separately entered in the inspected JavaScript: visibility and identity handlers update only their own field; the count input updates only count; drawing a box writes only the four box coordinates. Import/export moves these columns by their names. There is no handler that fills a categorical value when a box is drawn or changes other fields when visibility changes. The page does impose completion/consistency checks: for an absent target, it requires count `0`, identity `false`, and no box; for a visible target, it requires a positive count and a box. These checks validate the separate entries rather than auto-populating them. The bundled instructions state the same separation explicitly.

## Limits and gate status

This audit did not exercise a browser, CSV download/import, or human annotation, so it cannot establish runtime behavior or annotator agreement. A fresh packet-bound [scan v5](gate1_label_intake_scan_v5.json) at 2026-10-16 05:47 UTC again found zero complete or partial human returns among 7,232 CSV/ZIP paths. The gate still needs two independent, full human CSVs before the agreement screen can run. No key, label values, GPU, cloud runtime, or paid compute was used here.
