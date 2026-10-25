# Gate 1 rater independence and intake provenance audit

**Checked:** 2026-10-16 08:19 UTC  
**Scope:** source and metadata audit only. No submitted annotation CSV or private key was opened, no test was run, no packet or v5 schema was changed, and no receipt was created.

## Decision

**The current pipeline does not establish that two complete files came from two independent raters.** It verifies packet/frame mapping and annotation completeness, and the truth scorer prevents passing the same resolved path twice. Those are useful integrity checks, but two copies saved under different paths can pass that path check. A matching pair of hashes could flag byte-identical files if compared, but the current handoff does not require or compare received-file hashes, and re-saving or editing a copy changes its hash. Neither file hashes nor two role labels (“Rater A/B”) authenticate two people.

## Source findings

- The v5 packet README instructs the coordinator to send the ZIPs separately to independent raters and keep returns separate. The packet build manifest records packet ZIP hashes and `label_fields_populated: 0`; it is not a delivery/receipt log and has no per-return rater token or attestation.
- `scan_gate1_label_intake.py` records a candidate's resolved path, input kind, byte count, and modified time, then classifies a complete file as `complete_candidate_needs_independence_and_blind_review`. It does not hash candidate submissions or record packet assignment, receipt, rater token, or attestation. Its packet hash authenticates only the reference packet.
- `score_annotation_agreement.py` validates each CSV's complete blind frame keys and exact frame-to-image map against a reference packet and records that packet's hash. It accepts two input paths but does not compare submission hashes or authenticate authorship.
- `score_gate1_labels_against_kubric_truth.py` rejects identical resolved input paths. After the blind agreement gate passes, it writes each input path and SHA-256 in the truth report. That later hash is useful audit evidence about the scored bytes, but neither it nor the same-path check proves separate authorship; two copies at different paths are not excluded.
- The latest packet-bound inventory, scan v9 (`2026-10-16T08:16:11Z`), covered 7,232 CSV/ZIP paths and found 24 exact-v5-header candidates: 2 blank canonical templates, 20 blank or wrong-map candidates, 2 known QA/synthetic files, and zero complete or partial human returns. Thus there are no current submissions or receipts to authenticate.

## Minimal separate receipt proposal

Keep the annotation CSV and v5 schema unchanged. For each genuine return, a coordinator should create a separate, access-controlled `gate1_submission_receipt_v1` sidecar with only:

```json
{
  "receipt_id": "random opaque ID",
  "packet_id": "opaque assignment ID",
  "packet_sha256": "hash of the exact assigned v5 ZIP",
  "rater_token": "random coordinator-issued token, unique per assigned rater",
  "received_utc": "UTC timestamp",
  "received_file_sha256": "hash of the exact original received CSV or ZIP bytes",
  "independence_attestation": "I completed this packet independently, did not inspect another submission, and did not coordinate labels."
}
```

Before scoring, require both receipts, distinct coordinator-issued tokens, the expected packet hash, and each received-file hash to match the exact bytes to be scored. Equal received-file hashes should be held as a duplicate pair before agreement scoring. Keep any token-to-person assignment roster with the coordinator, separate from blind IDs, labels, and the scorer; do not place names/emails or alias mappings in the receipt. The attestation and token process are evidence of procedure, not cryptographic proof that one person could not use two tokens or coordinate. Hash mismatch/copy checks alone cannot detect relabeling or a byte-modified copy, so independence still depends on controlled assignment and honest attestations.

## Falsifiable next step and blocker

No receipt or code change is warranted before actual returns exist. When two independent, complete v5 submissions arrive, preserve each original byte stream; obtain one receipt per assigned rater; verify token distinction and packet/file hashes before running the already-registered packet-map and blinded agreement gates. If either receipt is missing, tokens repeat, packet hashes differ, or the same received-file hash appears twice, stop and resolve custody while still blind.

**Current Gate 1 blocker remains two independent complete v5 human CSV submissions.** Scan v9 has zero complete or partial returns, so agreement, blind worst-frame review, and truth scoring remain unrun. This audit establishes a provenance gap in the present intake handoff; it does not establish that any existing candidate was copied or misattributed.

## Audited local artifacts

| Artifact | SHA-256 |
|---|---|
| `research/scan_gate1_label_intake.py` | `957db7e6be873ece8e472c488901df8577addd3cc7f9d701aea4ad2b9b13de8e` |
| `research/score_annotation_agreement.py` | `9a3bc5289ef7e89a88b26f20ffc8770b415044ef35cb8ced5ee590597fe8aaac` |
| `research/score_gate1_labels_against_kubric_truth.py` | `e170665880063bc5725c41f5316ac93275f360079ca0e4dc82ff21ffc5864957` |
| `research/gate1_manual_annotation_packets_v5/README.md` | `abec46899f8b7d40b5a5d3bd16d6cf0723b08ffafe5b46fd8b223861f8df0b1b` |
| `research/gate1_manual_annotation_packets_v5/packet_manifest.json` | `1a661d35a94c98f7ec8688d81d71c5cdd2f38c90b74e3ed738952cee9ed6fca5` |
| `research/gate1_label_intake_scan_v9.json` | `02632d15d60cb5122992686fb8d1b2ecff2053801a12d082a287573c6f7b215a` |
