# Gate 1 packet-key fail-closed audit

**Checked:** 2026-10-15 08:17 UTC  
**Component:** earliest open component, Gate 1 independent-label reliability.  
**Decision:** require each submission's exact `(blind_clip_id, frame_index) -> image_file` mapping to match the archived v5 annotation packet before agreement scoring or any later truth/unblind stage. This closes a shared-wrong-filename hole in the scorer; it does not establish annotator agreement.

## Evidence

`score_annotation_agreement.py` now reads only the single annotation CSV member of a reference packet ZIP, refuses private-key-looking ZIP members, and validates both the exact frame-key set and corresponding image filename for each rater. It reports the reference ZIP SHA-256 and key-check booleans, without exposing alias direction. The known-truth scorer applies the same check before its blinded agreement gate; the blind worst-frame review builder checks both submissions before rendering review cards. The second-seed packet builder now requires exact mapping booleans in its referenced agreement and truth reports, matching their first-block packet digest, and requires the blind-review packet-key checks.

The v5 synthetic reliability packet maps contain 34 entries each and compare equal (`rater_01.zip` SHA-256 `0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8`; `rater_02.zip` SHA-256 `8a660acb5bd77855b82c13b27cc951f8d2e4d6131e38e0144e83638e2f92e7ff`). The earlier [pixel-provenance audit](2026-10-15_gate1_v5_packet_fixture_pixels_audit.md) independently checks that packet rows map one-to-one to the embedded frame manifests and that embedded PNGs match the local fixture. The current changes do not open the private unblind key or emit alias-to-direction mappings.

Synthetic checks passed:

- Agreement CLI accepts two exact-map submissions and rejects a pair where both raters use the same wrong filename for one frame; diagnostic: [v6](gate1_scorer_tail_diagnostic_v6.json), 0.409068 s.
- Truth scorer rejects the same common-mode wrong filename before any attempt to read the deliberately nonexistent key; exact-truth, shared +2 px bias, uncertainty, and duplicate-instance oracle cases remain consistent; diagnostic: [v3](gate1_truth_scorer_oracle_selfcheck_v3.json), 0.779671 s for seven isolated scorer processes.
- Blind maximum-error review still selects the injected frame, keeps cards offline and blind, and validates packet keys; diagnostic: [v2](gate1_blind_worst_frame_review_selfcheck_v2.json), 0.014575 s.

All runs were local CPU-only on Darwin arm64 / Python 3.12.3; GPU use was false, peak VRAM was not allocated, and there was no model inference, external download, paid resource, or cloud runtime. The validator scripts use Python standard-library modules plus the local scorer. No external code or data was added. Archive-key validation assumes the selected reference ZIP is the trusted, hash-recorded packet; the scorer cannot prove provenance if a caller deliberately substitutes another ZIP.

## Falsifiable next step and limits

Use the designated v5 `rater_01.zip` as the reference packet for each independent return. A submission pair with any missing, extra, or misbound frame must fail before agreement scoring and before key access; a pair with exact maps can proceed only if pooled **and each clip** pass the existing agreement screen and the blind worst-frame review. The truth scorer remains downstream of that gate. Two independently completed 34-frame CSVs are still missing, so no human reliability, generated-video observer validity, or path-policy efficacy claim is supported. Keep visibility/identity, box coverage, center path, and apparent size separate.
