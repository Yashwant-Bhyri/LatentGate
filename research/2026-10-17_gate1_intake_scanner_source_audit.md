# Gate 1 intake scanner source audit

**Reviewed:** 2026-10-17 01:11 CST (2026-10-16 17:11 UTC)  
**Scope:** source-only review of `research/scan_gate1_label_intake.py` and the coordinator intake SOP.  
**Actions:** no scanner, scorer, test, browser, or compute was run. No candidate CSV/ZIP, annotation, key, receipt, or staging file was opened.

## What routine detection reads

The current scanner is a content validator, not a metadata-only arrival watcher. Its default roots are project `research/`, `~/Downloads`, and `~/Desktop`; it recursively enumerates only `.csv` and `.zip` paths and does not follow directory symlinks. The SOP excludes the owner-only `Application Support/LatentGate/Gate1/` custody tree from these roots.

For each CSV, `read_candidate` reads the entire UTF-8-with-BOM-compatible file into memory. It parses the exact 11-column v5 header and every row. In particular, it reads annotation cells in `object_visible`, `instance_count`, `identity_consistent`, all four box coordinates, and `notes` to derive counts of rows with any populated annotation field, rows with all three state fields populated, and rows with any box coordinate. It also reads blind clip IDs, frame indices, and image filenames to check duplicate/invalid keys and exact packet key-to-image mapping.

For each ZIP, the scanner reads the archive member directory and rejects archives with a member basename containing `UNBLIND` or `PRIVATE_KEY`. It then accepts only one CSV member, capped at 10,000,000 uncompressed bytes, reads that member into memory, and parses it as above. It does not extract members. The scanner also loads the registered reference packet's frame/image keys and hashes the reference packet; this source review did not open that packet.

## What it emits

The JSON report stores candidate-level resolved paths (including filenames), input kind, byte size, modification time, exact-header status, row count, the three annotation-derived counts, key/map/duplicate/invalid/full-packet flags, and an intake class. It also records aggregate counts, roots, reference/scanner hashes, and CPU/runtime metadata. The CLI prints only the output path, aggregate path/candidate/class counts, and runtime summary.

The report does **not** emit raw annotation strings, blind clip aliases, alias-to-direction mapping, or private-key contents. Therefore `annotation_values_emitted: false` is accurate for raw cell values, but it does not mean that no annotation-derived information is emitted: completeness/box counts and candidate paths are emitted. The current CLI default is `research/gate1_label_intake_scan_v1.json`; writing there would conflict with the SOP's rule that annotation-derived reports stay in owner-only staging. Any future full content scan must explicitly set `--output` to a fresh path in that restricted staging directory. Do not use the broad scanner on genuine returns. Existing historical inventory JSONs under `research/` were left untouched and not reopened during this source audit.

## Decision and next step

No separate metadata-only helper was added. A stat-only pass could report only path, size, and modification time for newly added/changed `.csv` or `.zip` paths, without reading their contents; it could not establish packet completeness, authorship, independence, or human validity. It would still expose path metadata and needs a persistent owner-only baseline/output. Genuine returns should arrive through the coordinator-controlled receipt flow into private custody, not the broad discovery roots, so a second public-root watcher does not materially improve the current safe return workflow. The prior state entry records a metadata-only inventory with no `.csv`/`.zip` changes since scan v16; this audit did not repeat it.

The latest full intake evidence remains scan v16 (7,232 paths, 24 exact-header candidates, zero complete and zero partial human returns). This source audit does not establish a new inventory result. **Gate 1 remains blocked on two independent complete v5 human returns.** The falsifiable next step remains receipt and independence verification for both returns, followed by exact packet-map validation, frozen agreement thresholds, blinded worst-frame review, guarded truth scoring, and held-out mirrored-seed replication. Rater instructions, schema, thresholds, and gate order are unchanged.

## Evidence pins

- Scanner source SHA-256: `957db7e6be873ece8e472c488901df8577addd3cc7f9d701aea4ad2b9b13de8e`
- Scan v16 SHA-256 (reported in the existing registry; not reopened in this audit): `51eca7508423b6fa6c6628b6ecfd0094d03e9798818e101e5e1d40d50b73d959`
- Coordinator SOP: `research/2026-10-16_gate1_assignment_intake_sop_draft.md`, SHA-256 `7278689382309a2794b4fa287ccca2a6a7692d429e2dad4773445896c77f6c52`

No candidate labels were inspected, and no human agreement, observer accuracy, or policy efficacy is claimed.

## Follow-up: fail-closed report destination (2026-10-17 06:23 UTC)

The scanner CLI now requires an explicit `--output`. Before scanning, it resolves the destination and requires an existing directory owned by the current user, with owner access and no group/other permission bits; it rejects paths inside the project tree and any existing output. It no longer creates directories. Report creation uses exclusive, no-follow open flags and mode `0600`, so it cannot replace a pre-existing file or follow an output symlink. The source does not inspect filesystem ACLs; the coordinator SOP's owner/mode/ACL preflight still applies to private staging.

This only hardens accidental output placement. The scanner still parses annotation cells and emits derived counts and candidate paths, so it remains a candidate-discovery tool and must not be run over genuine returns. Genuine returns continue through receipt/hash verification and the private staging workflow. The historical scan reports were neither changed nor reopened. This follow-up was source-only: no scanner, scorer, test, compile, candidate, annotation, receipt, key, staging file, or private custody tree was accessed.

- Current scanner source SHA-256: `a7db4518ef76d5efe761d0f05e014968ae32a94927a3ee8c849d89e8b30c360c`.
- Coordinator SOP remains `research/2026-10-16_gate1_assignment_intake_sop_draft.md`; current recorded SHA-256 is `30fc20c6f4696e993ab8fef4d29b2f6cb51cc3de939fa3e1556bec5045c9c4e4`.
