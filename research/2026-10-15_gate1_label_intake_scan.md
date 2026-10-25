# Gate 1 label-intake scan

**Checked:** 2026-10-15 08:26 UTC  
**Component:** Gate 1 independent-label reliability, still the earliest open component.  
**Decision:** no new human return is available; retain the label gate and do not start the DiTraj policy comparison.

## Method and results

The reusable read-only scanner [`scan_gate1_label_intake.py`](scan_gate1_label_intake.py) scanned 7,232 loose CSV and ZIP paths under project research, Downloads, and Desktop. It accepted only the exact 11-column v5 schema, then compared each candidate's row keys and filenames to the frame manifest in the selected reference packet. ZIPs were inspected in memory, never extracted; archives containing a private/unblind-key-looking member were skipped. The scan records file paths, byte sizes, row counts, key-map booleans, and coarse intake classes. It emits no annotation values, blind aliases, alias-to-direction map, or private key.

There were 24 schema-matching candidates:

- Two are the archived v5 synthetic-reliability rater ZIPs. Their 34-row frame maps match the reference exactly, but all annotation fields are blank.
- Twenty other schema-matching CSV/ZIP files have blank annotation fields and do not match this v5 packet frame map; they are templates or other packet material, not usable returns.
- Two known synthetic/UI QA CSVs each have only one populated row; both are excluded.
- No complete packet-matched return or partial annotated return was found.

The [machine-readable scan](gate1_label_intake_scan_v1.json) binds the reference packet SHA-256 `0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8`. Runtime was 9.140115 s on Darwin arm64 / Python 3.12.3, CPU only; GPU false and peak VRAM null. No model, cloud runtime, external dataset, paid compute, or file extraction was used. The scanner only screens candidate intake; final acceptance still requires two separate complete CSVs to pass blinded agreement, worst-frame review, and guarded truth scoring.

## Scanner self-check and fresh rescan

At 2026-10-15 08:29 UTC, a synthetic CPU self-check verified that complete exact-map CSV and ZIP inputs are surfaced as candidates; a blank exact-map template is classified blank; a shared wrong filename in both submissions and duplicate frame keys are rejected; and an archive containing a private-key-looking member is skipped without reading that member. All six checks passed in 0.009451 s on Darwin arm64 / Python 3.12.3. The [self-check result](gate1_label_intake_scanner_selfcheck_v1.json) uses opaque synthetic clip IDs and emits no label values or aliases.

A fresh scan using the self-checked scanner again covered 7,232 CSV/ZIP paths and returned the same 24 candidates and zero complete/partial returns. See the [latest scan](gate1_label_intake_scan_v3.json), which records the scanner source hash. The type annotation mismatch found during review was corrected before this v3 run; it did not affect prior scan behavior.

## Falsifiable next step

When two rater CSVs arrive, the intake scanner must classify both as complete packet-matched candidates. Then verify they are distinct independent submissions, run the existing pooled and per-clip agreement screen, inspect maxima while still blind, and only then run the guarded truth scorer. The independent mirrored-seed block remains downstream of a passing first block.
