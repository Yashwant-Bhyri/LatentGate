# Gate 1 ZIP intake safety source review

**Checked:** 2026-10-16 21:29 CST  
**Scope:** static review of the owner handoff, intake scanner, agreement scorer, coordinator SOP, and the installed Python `zipfile` source. No return archive, annotation value, blind ID/map, or private key was opened. No code, tests, runtime, or compute were used.

## Source findings

- The [owner handoff checklist](2026-10-16_gate1_owner_label_handoff_checklist.md) asks raters for a plain CSV. `scan_gate1_label_intake.py` also recognizes `.zip` files: it rejects archives with a member whose basename contains `UNBLIND` or `PRIVATE_KEY`, requires exactly one `.csv` member, checks the central-directory `file_size` is at most 10,000,000 bytes, then calls `archive.read()` and decodes the returned bytes. It records the resolved outer archive path, size, and modified time in its report; it does not emit the internal member filename.
- The scanner uses `namelist()` followed by a filename-based `getinfo()` lookup. Two `.csv` entries, including duplicate CSV names, make the filtered count exceed one and are rejected. Other non-CSV entries are not rejected by that check. `ZipFile.read()` reads the member into a complete bytes object. The installed standard-library source shows `ZipExtFile` initializes its remaining-byte count from `ZipInfo.file_size`, slices decompressed output to that declared amount, and checks CRC as the stream reaches EOF. The scanner has no independent bounded-chunk byte counter, ZIP-envelope-size bound, or member-count bound. These source facts do not establish an exploit or show a real malicious input; the current declared-size guard is useful but is not the same as a streaming output cap.
- The agreement scorer's `load_rows()` opens a filesystem path as UTF-8 CSV, so a ZIP is not a direct scorer input. A ZIP that passes intake needs a CSV working file before agreement/truth scoring.
- The scanner does not extract archive members to disk. Therefore an internal `../` name or symlink metadata is not used as a destination by the current scanner; path traversal and symlink-following are not evidenced in this path. The earlier SOP's phrase “extract ... to a restricted working location” did not specify a safe API or destination, so a future ordinary `extract`/`extractall` workflow would add avoidable path and symlink handling risk.
- `ZipFile.open()` without a password raises on encrypted members; the scanner catches `RuntimeError` and omits that candidate. No password is requested. For future intake, reject encrypted entries before streaming and never prompt for credentials.
- The outer filename is included in scanner reports as a resolved path. Avoid putting real names or other personal identifiers in staged paths. The internal member name is not currently emitted by the scanner, and does not need to be copied into a receipt to identify its contents when the envelope hash, one-member constraint, ordinal, and member hash are recorded.

## Decision and SOP clarification

Keep plain CSV as the requested return format. If a ZIP is used, keep the coordinator's accepted shape deliberately narrower than the scanner's permissive intake filter: cap the outer envelope at 12,000,000 bytes, require exactly one archive member total and require it to be a regular, unencrypted `.csv` member with declared and streamed output no larger than 10,000,000 bytes; reject path-like names, duplicate names, symlink/special-file modes, and the existing key-name blacklist. The 12 MB envelope limit allows 2 MB over the existing CSV cap for ZIP structure; it is a proposed coordinator limit, not a scanner capability.

Hash and preserve the exact original CSV/ZIP envelope before parsing or extraction. For a ZIP, select the member through its `ZipInfo` object and stream bounded chunks into a securely created opaque temporary filename; count and hash actual bytes as they are written, stop if the 10 MB limit is exceeded, and read to EOF for CRC checking. Never use the member filename as a filesystem path. Record only member ordinal and hash in the receipt; keep original/internal filenames only in the coordinator-only custody record. Stage scanner/scorer copies under opaque receipt-derived names so the scanner's resolved-path field does not disclose a rater's filename. If any ZIP check fails, do not score it; request the plain CSV instead.

This is a procedural recommendation only. The v5 CSV/packet schema, scanner, and scorer remain unchanged. It establishes no annotation agreement and does not unblock Gate 1. Scan v12 remains the latest inventory: zero complete or partial human returns; two independent complete v5 CSVs remain necessary.

## Audited source hashes

| Artifact | SHA-256 |
|---|---|
| `research/scan_gate1_label_intake.py` | `957db7e6be873ece8e472c488901df8577addd3cc7f9d701aea4ad2b9b13de8e` |
| `research/score_annotation_agreement.py` | `9a3bc5289ef7e89a88b26f20ffc8770b415044ef35cb8ced5ee590597fe8aaac` |
| `research/2026-10-16_gate1_assignment_intake_sop_draft.md` before clarification | `fa2f3b51db7bd5b7d6776b7b085fdbf243fc26fc575f917ce50511eb9d84af9f` |
| `research/gate1_label_intake_scan_v12.json` | `261d153921a51438e62e4964065d2f89ce34054412ca9ef5630e573b4f59506b` |
