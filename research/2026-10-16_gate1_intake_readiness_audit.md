# Gate 1 v5 intake readiness and provenance decision

**Checked:** 2026-10-16 21:43 Asia/Shanghai  
**Scope:** Source/document review of the v5 packet contract, owner handoff, coordinator intake SOP, scanner and its synthetic validator, agreement/truth scorers, and independence audit. No return CSV or ZIP candidate, label value, blind alias/direction map, or private key was opened. No archive was extracted; no test, scoring run, schema edit, contact, receipt, or compute was performed.

## Decision

Use a plain CSV from each of two distinct assigned people as the canonical return. The scanner is a coarse intake filter; it does not verify the truth of annotations or rater independence. Receipt, custody, and independence checks are coordinator work. Only after that manual gate should the existing packet-map and agreement checks run, followed by blind worst-frame review and guarded truth scoring. The current hard blocker remains **two independent, complete v5 human returns**.

## Acceptable return payload

Each person receives one of the two separately assigned v5 synthetic-reliability packets and returns its CSV export, not a screenshot. Each CSV must retain the exact ordered header:

```text
blind_clip_id,frame_index,image_file,object_visible,instance_count,identity_consistent,bbox_x1_px,bbox_y1_px,bbox_x2_px,bbox_y2_px,notes
```

It must contain all 34 packet rows (two blind 17-frame clips), with the exact original `(blind_clip_id, frame_index) → image_file` mapping. Each row needs separate, explicit visibility, instance-count, identity, and box judgments; label every frame independently without interpolation. Boxes are tight visible-object boxes in source-image pixel-edge coordinates, excluding shadows and detached glare. A visible object needs count ≥1 and a box; an absent object needs count 0, identity false, and no box. The agreement scorer checks boolean-like identity/visibility values, integer counts, box completeness, non-degenerate in-frame geometry, and these absent/visible consistency rules. `notes` is optional and not part of the categorical agreement metric.

The handoff says “identity No/uncertain,” while the schema/scorer provide a binary field and the scorer rejects the literal token `uncertain`. Recommended convention for unchanged v5: encode “No or uncertain” as false and use `notes` to retain that uncertainty for review; such a row is not eligible for path-center comparison. If the project intends a separately scored tri-state, v5 cannot express it and needs a deliberate future schema decision. Do not improvise a new token in a return.

The handoff prefers CSV. ZIP is an optional coordinator exception: the scanner accepts one CSV member up to 10 MB and can overlook extra non-CSV members, while the SOP requires exactly one regular unencrypted CSV member, no extras/path-like/duplicate/special members, a 12 MB envelope limit, actual streamed-byte counting, CRC completion, and separate envelope/member hashes. Those rules are compatible if the stricter SOP is applied before scanning; the scanner does not enforce them all. Scorers accept CSV paths, so ZIP submissions require a verified opaque working CSV.

## Automated versus human checks

- **Intake scanner:** enforces the exact ordered header; counts rows and nonempty fields; checks duplicate/invalid frame keys, complete frame-key set, exact packet image mapping, and coarse blank/partial/complete-candidate classes. It emits candidate paths, file kind, size, and modified time, so scan/scoring copies should use opaque receipt-based names in a restricted staging directory. It does not hash/authenticate a submitted file, validate all field semantics, check a receipt, or establish who annotated it. Its complete class explicitly still “needs independence and blind review.”
- **Synthetic validator:** the source builds disposable CSV/ZIP fixtures and checks scanner behavior. It is a self-check, not a validator of genuine returns or receipts. It was not executed during this audit.
- **Agreement scorer:** accepts two CSV paths; checks semantic field consistency and each submission's full frame/image map against the reference packet; reports pooled and per-clip screens. Keep the frozen gates: joint visibility/count/identity agreement ≥0.95, visible-center coverage ≥0.90, and p95 center disagreement ≤0.5 target-grid cell per axis. The blind worst-frame review remains required even after a pass.
- **Truth scorer:** repeats the blind map/agreement gate and only then reads protected truth/key material. It is not an independence verifier. No scorer was run here.
- **Coordinator/rater checks:** assign one packet per distinct person; issue distinct random tokens only at assignment; keep the token-to-person roster private and separate from labels and blind mapping; obtain an independence/no-cross-viewing attestation; preserve/hash each genuine original; verify the two receipt records, assignment packet hashes, returned-envelope hashes, token-to-person distinction, and (for ZIP) streamed CSV hash before scoring. Each separate receipt records an opaque receipt ID, manifest-relative packet ID, assigned-packet hash, anonymous rater token, UTC receipt time, format, received-envelope hash, and attestation; ZIP receipts also carry the CSV member ordinal and scored-CSV hash. Different tokens and hashes are procedural evidence, not cryptographic proof of authorship or non-collusion.

## Contradictions and operational gaps

1. **Stale inventory statements:** the owner handoff still calls scan v4 “latest”; the registry points to v14. The coordinator SOP was refreshed to v14 during this audit. Use the registry-bound v14 report as current: 7,232 paths, 24 exact-header candidates, two blank canonical packets, 20 blank/wrong-map candidates, two QA/synthetic candidates, and zero complete or partial returns.
2. **Original versus staged filenames:** the handoff gives `rater_01_completed.csv`/`rater_02_completed.csv` paths and scans the returns directory; the SOP requires opaque names for scan/scoring copies because absolute paths are emitted. Resolve this by preserving any named original only in restricted coordinator custody, then scanning/scoring opaque receipt-based copies from a separate restricted staging directory. Keep the identity-to-token/path map out of scanner-visible locations.
3. **Receipt enforcement is manual:** the sidecar fields and fail-closed checks are specified, but no receipt schema/validator is implemented. The coordinator must verify them explicitly before scanning. The SOP also refers to a controlled channel and restricted originals/staging without naming or verifying the actual storage/access-control location; establish that location before accepting returns.
4. **Uncertainty encoding:** “No/uncertain” is ambiguous against a binary CSV field and boolean parser. The recommended v5 convention is false plus a note, with uncertain identity excluded from geometry; this preserves a human-readable note but not a distinct machine-scored uncertainty category.
5. **Minor SOP edit:** the two consecutive receipt/custody actions in section 2 are both numbered “5”; the order is understandable, but renumbering would reduce handoff ambiguity.

There is no conflict in the canonical CSV-versus-optional-ZIP policy once the SOP's stricter ZIP gate is treated as coordinator-only and runs before the scanner. The source listing did not surface a newer **Research LatentGate Methods** report; the October 11 full-report reconciliation remains the available research synthesis, with no new claims added here.

## Falsifiable next step

After the coordinator establishes restricted original/staging locations, receive two genuine returns from the two distinct assigned people and create/verify one receipt per return without exposing blind mappings. Both opaque CSV staging copies must classify as complete packet-matched candidates, pass the registered pooled and per-clip agreement thresholds, and pass blind worst-frame review before guarded truth scoring. A zero-return scan remains a failure of the intake gate, not evidence about annotation quality or policy efficacy.

## Audited source hashes

| Artifact | SHA-256 |
|---|---|
| `research/gate1_label_intake_scan_v14.json` | `ce39cefcfdb77a6b7110fdbce8cfba36862be1febd0fe4d95de76a3cd6e51d6b` |
| `research/2026-10-16_gate1_owner_label_handoff_checklist.md` | `9b30bfcedf103f609025052761fe6d821deb83edb2a139ae60c75a0e577856b2` |
| `research/2026-10-16_gate1_assignment_intake_sop_draft.md` | `5a7458aebbcc13e1c6060b0cd2246e051979fb0553ce4b903ca2ace87dec8ace` |
| `research/scan_gate1_label_intake.py` | `957db7e6be873ece8e472c488901df8577addd3cc7f9d701aea4ad2b9b13de8e` |
| `research/validate_gate1_label_intake.py` | `0ea4a78124775a594d25a1e5f28ae51559da1c22ccc1f08d8caecc01e6b4bd49` |
| `research/score_annotation_agreement.py` | `9a3bc5289ef7e89a88b26f20ffc8770b415044ef35cb8ced5ee590597fe8aaac` |
| `research/score_gate1_labels_against_kubric_truth.py` | `e170665880063bc5725c41f5316ac93275f360079ca0e4dc82ff21ffc5864957` |
| `research/2026-10-16_gate1_rater_independence_provenance_audit.md` | `6d138c5480cfc2fd66f7d829537fe11b2a122fda8b6cc9168bd3fd91fd6e1829` |
