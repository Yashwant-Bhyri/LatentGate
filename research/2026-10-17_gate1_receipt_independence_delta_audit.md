# Gate 1 receipt and rater-independence delta audit

**Reviewed:** 2026-10-17 02:09 CST (2026-10-16 18:09 UTC)  
**Scope:** source-document comparison of the 2026-10-16 rater-independence provenance audit, current owner handoff checklist, and coordinator intake SOP.  
**Actions:** no real receipt, roster, assignment, submission, annotation, key, or staging file was opened. No scanner, scorer, test, or code was run or written.

## Delta from the earlier provenance audit

The earlier audit found that packet/frame validation was not evidence of authorship: the candidate scanner did not record candidate hashes, assignments, tokens, or attestations; agreement scoring did not compare submission hashes; and the truth scorer's same-resolved-path rejection could not prevent two byte-identical copies at different paths from being treated as a pair. It proposed a separate receipt and a controlled token-to-person roster while keeping the v5 CSV unchanged.

The current coordinator SOP adds the main missing controls:

- Two distinct people receive one different mirrored packet each; random opaque tokens are unique per assignment and mapped to people in a coordinator-only roster.
- The roster binds the manifest-relative packet ID and assigned packet ZIP hash. Receipts carry the packet ID/hash, token, receipt time/format, exact received-envelope hash, independence attestation, and—when the return is ZIP—the scored CSV member ordinal and hash.
- Immutable envelope bytes are copied and hashed into owner-only `originals/` before parsing/extraction. Opaque staged copies, receipts, roster, and derived outputs have separate owner-only locations. ZIP intake has bounded size/member/path/type/encryption checks and never extracts by member path.
- Before scoring, the SOP already required distinct tokens mapped to distinct people, packet/hash checks, original-envelope hash checks, staged-member hash checks, and duplicate-envelope hash comparison. The owner checklist routes genuine returns through this flow and explicitly says the broad scanner is not an authorship or independence check.

This closes the earlier *procedural* gaps if the coordinator follows the SOP. One relationship had been implicit: receipt `packet_id`, assigned packet ZIP hash, and token must all match the same roster assignment. I made that three-field tuple comparison explicit in a coordinator-only pre-score preflight paragraph. It adds no rater instruction, schema, threshold, or gate, and does not create a receipt or assignment.

## What a preflight can verify without interpreting labels

A strict metadata-only pass can verify directory/file paths, ownership, modes, and ACL metadata, but it cannot verify receipt fields or recompute file hashes. A safe pre-score check is better described as a **hash-only custody preflight**:

1. Check that the custody directories are still owner-only and each recorded original/staged path resolves under its designated directory.
2. Read only the two receipt sidecars and private roster; compare each `(rater token, packet ID, assigned packet ZIP hash)` tuple to its roster row, confirm two different tokens map to two different assigned people, and check each packet hash against the pinned packet value.
3. Stream-hash the exact original envelope and staged scored bytes without decoding or parsing CSV fields. For a ZIP, stream/hash only its single allowed CSV member and compare that digest with both the receipt's scored-member digest and the staged CSV digest. Require distinct received-envelope hashes.
4. Emit only pass/fail and an opaque run reference; keep names, tokens, paths, hashes, labels, and blind aliases out of shared logs/registry. Stop before any CSV parser or scorer on any mismatch.

Hash recomputation necessarily reads bytes from the CSV/ZIP, so this is not literally metadata-only. It does not interpret annotation values. Even a passing tuple/hash/attestation preflight cannot prove that two tokens correspond to two real people or rule out collusion or relabeling; that remains a controlled-assignment and honest-attestation limitation. No automated preflight helper exists, and none was added in this source-only audit.

## Current gate and minimum next action

The owner checklist and registry still report no returns/receipts and require **two independent, complete v5 human returns**. No actual assignment or custody event has occurred, so the new SOP is a ready procedure, not completed provenance evidence. The latest full inventory remains scan v16: 7,232 paths, 24 exact-header candidates, zero complete and zero partial human returns. A metadata-only path/stat inventory is not evidence of labels or human independence.

**Minimum safe next step:** wait for the two independently assigned, complete v5 CSV returns through the controlled channel. Before parsing either CSV, complete and privately record the custody/hash preflight above; fail closed and resolve any discrepancy while blind. Only after that pass should the unchanged packet-map, pooled/per-clip agreement, blinded worst-frame review, guarded truth, and held-out mirrored-seed gates proceed.

This audit is source/protocol evidence only. It is not human-label reliability evidence, observer validation, or policy-efficacy evidence.

## Source documents

- Earlier provenance audit: `research/2026-10-16_gate1_rater_independence_provenance_audit.md`
- Current owner handoff checklist: `research/2026-10-16_gate1_owner_label_handoff_checklist.md`
- Coordinator SOP with clarified pre-score check: `research/2026-10-16_gate1_assignment_intake_sop_draft.md`
