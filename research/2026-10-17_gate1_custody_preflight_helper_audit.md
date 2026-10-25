# Gate 1 custody preflight helper: source and design audit

**Completed:** 2026-10-17 07:37 CST  
**Scope:** implement a pre-score custody/hash helper while two complete v5 human returns are absent.  
**Execution:** source edits and static review only. The private Gate 1 tree was not inspected. No receipt, roster, assignment, submission, annotation, key, or staging file was opened or created. The helper, scanner, scorers, and tests were not run.

## Decision

Added `research/preflight_gate1_custody.py` as a **hash-only, fail-closed preflight**. It does not import a CSV parser, decode annotation fields, unblind, score, write output files, or print input paths, tokens, hashes, identities, or exceptions. On invocation it emits one JSON line with only `result` (`PASS` or `FAIL`) and a newly generated opaque `run_id`; it returns a nonzero exit status on failure. All failures collapse to the same result.

The script is intentionally **unverified**. It has not been executed, syntax-compiled, tested, or pointed at the private tree. Its successful path cannot be claimed until real receipts and staged files arrive and an authorized coordinator runs it before CSV parsing.

## Checks implemented

- Mac-only, fail-closed checks for the Gate 1 root and four custody directories: real directories, current-user ownership, exact `0700` mode, and no ACL entries. Private regular files must be current-user-owned, mode `0400` or `0600`, have no ACL entries, and are opened with `O_NOFOLLOW`.
- Bounded JSON reads for one exact `assignments.json` roster and exactly two receipt sidecars. Duplicate JSON keys, unknown fields, malformed records, unexpected files, and noncanonical IDs fail without diagnostic output.
- The two roster assignments must use distinct tokens and distinct private `assignee_ref` values, and bind the two pinned manifest-relative packet IDs to their registered packet hashes. Each receipt's `(rater_token, packet_id, packet_sha256)` must match the same roster row.
- Original envelopes are streamed to SHA-256 with a 10 MB CSV or 12 MB ZIP cap. Staged CSVs are streamed with a 10 MB cap. The exact received-envelope and scored-CSV hashes must match their receipts, and the two received-envelope hashes must be distinct. A later source-only contract review removed an extra staged-file-distinctness check because the current SOP does not require two independently produced CSVs to differ byte-for-byte.
- For a ZIP return, the helper requires one safe, unencrypted regular CSV member only, caps actual decompressed bytes at 10 MB, reads the member to EOF for CRC checking, and compares that member hash with both the receipt and staged CSV. It never writes/extracts the member or parses CSV contents.

The helper expects the documented names (`private_roster/assignments.json`, `<receipt_id>.json`, `<receipt_id>.envelope`, and `<receipt_id>.csv`) and exactly one initial two-rater pair with no extra direct entries in those directories. This first-pair scope must be adapted to a separate per-pair custody layout before the later mirrored-seed replication. The script's packet hash constants match the two pinned ZIP values already recorded in the coordinator SOP; it does not open the packet ZIPs.

## Limits and remaining human gate

The helper verifies that two private roster references differ; the coordinator must still ensure they identify two real, distinct people. Tokens, receipt hashes, and an attestation cannot cryptographically prove personhood, independence, or absence of collusion. Strict filesystem metadata alone cannot recompute content hashes: this procedure streams bytes but does not interpret annotation cells.

The latest full inventory remains scan v16 with zero complete and zero partial returns. **Two independent, complete v5 human returns are still required.** No label reliability, observer validity, or policy efficacy is established by this source/helper work.

## Source documents

- Earlier provenance analysis: `research/2026-10-16_gate1_rater_independence_provenance_audit.md`
- Current owner procedure: `research/2026-10-16_gate1_owner_label_handoff_checklist.md`
- Coordinator custody/receipt contract: `research/2026-10-16_gate1_assignment_intake_sop_draft.md`

No rater instruction, v5 CSV schema, scoring threshold, or label gate order changed.

## Follow-up static safety review

**Reviewed:** 2026-10-17 07:58 CST  
**Execution:** source and SOP text only. The helper was not executed, syntax-compiled, or tested. The private custody tree and every real receipt, roster, assignment, label, key, return, and staged file remained unopened and untouched.

### Findings and source-only corrections

- **Parent-path symlinks and descriptor anchoring:** the first draft checked only the Gate 1 root and direct custody directories, so an intermediate symlink could redirect absolute-path operations. The initial source-only follow-up added canonical-path checks; this follow-up replaces pathname-based data access with component-by-component traversal from `/`. It opens each directory relative to the previously opened descriptor with `O_DIRECTORY|O_NOFOLLOW`, retains those descriptors for the preflight, and checks ownership/mode from `fstat`. File opens, listings, and entry metadata checks are relative to the retained custody-directory descriptors; final file opens also use `O_NOFOLLOW`. Required `os.open`/`os.stat` dirfd, `os.listdir` fd, `O_DIRECTORY`, and `O_NOFOLLOW` capabilities are checked and unsupported platforms/capabilities fail closed. Ancestor ACL checks reject allow entries while permitting deny-only ACLs; the Gate 1 root and direct custody directories still require no ACL entries.
- **Residual same-UID race:** dirfd anchoring prevents an intermediate pathname swap from redirecting the content open after a directory descriptor is acquired. It does not provide an atomic filesystem snapshot or lock against another process running as the same owner. ACL inspection still calls `/bin/ls -lde` on an absolute pathname and brackets it with inode identity checks; an adversarial swap-out-and-back or same-UID change to files, directory entries, modes, or ACLs during/after checks is not ruled out. Run only in a quiescent owner-only custody window, and do not claim resistance to concurrent same-account mutation. The helper remains unverified.
- **ZIP entry type and encryption flags:** the original check used `ZipInfo.is_dir()` and Unix file-type bits, which did not reject a DOS-origin directory attribute when the member name lacked a trailing slash. It also checked traditional encryption bit 0 but not strong-encryption bit 6. Added rejection for DOS directory attribute `0x10` and either encryption bit. The helper still reads exactly one selected `ZipInfo`, rejects path-like names and non-regular Unix types, enforces the declared and actual 10 MB member cap, streams to EOF for CRC verification, and never extracts the member.

### Remaining review limits

- The owner/mode checks are explicit (`0700` custody directories, `0400`/`0600` regular files, current UID) and use `fstat` on opened descriptors; system ancestors must be root-owned and not group/world-writable, the home directory must match the SOP's `0750`, and the `Library`/`Application Support` parents must match `0700`. ACL inspection is still path-based and bracketed by descriptor/path inode identity checks; same-UID concurrent mutation is covered by the residual race above, not proven impossible by this source review.
- JSON is bounded to 64 KB, decoded as strict UTF-8, duplicate object keys are rejected, and schemas use exact key sets. Extreme nesting may still make the standard parser fail; failures are collapsed to the same generic result.
- The exact two-receipt/two-original/two-stage/one-roster file set matches only the first pair in the SOP. This helper must be adapted to isolated per-pair custody before the later mirrored-seed replication.
- Streamed hashes are compared to receipt values; file identity, size, and timestamps are checked on the open descriptor before and after reads. The code does not parse CSV cells. Output remains one opaque run ID plus `PASS`/`FAIL`; errors, paths, tokens, identities, hashes, and annotation values are not intentionally emitted.

**Decision:** the two concrete source gaps above received narrow source-only fixes. A later dirfd-anchored source revision is documented below. The helper is still unverified and must not be represented as a completed custody pass. No helper behavior has been validated by execution. Two independent complete v5 human returns remain the human-label blocker; scan v16 reports zero complete and zero partial returns. No label reliability or policy-efficacy evidence exists.

## Directory-fd-anchored follow-up

**Reviewed:** 2026-10-17 08:03 CST  
**Execution:** static source/document review only. No execution, syntax compilation, test, scanner/scorer run, or private custody/intake access.

### Design and correction

- Replaced `Path.home().resolve()` with the lexical `Path.home()` path. The helper now opens `/` and traverses every component to the home directory, `Library`, `Application Support`, `LatentGate`, and `Gate1` using `os.open(component, O_RDONLY|O_DIRECTORY|O_NOFOLLOW, dir_fd=parent_fd)`. It retains those descriptors during the preflight and opens each of the four custody directories relative to the verified `Gate1` descriptor.
- Owner and mode checks use `fstat` on each opened directory descriptor: system path ancestors must be root-owned and not group/world-writable; home is checked against the SOP's `0750`; `Library` and `Application Support` must be owner-owned `0700`; `LatentGate` must be owner-owned and not group/world-writable; and `Gate1` plus each direct custody directory must be owner-owned `0700`. ACL inspection rejects grants on the path ancestry, permits deny-only ACL entries there, and still rejects any ACL on the private root/children. Descriptor identity, owner, and mode are checked again before cleanup.
- Custody file opens now use `os.open(name, O_RDONLY|O_NOFOLLOW, dir_fd=the_verified_directory_fd)`. Directory listings and no-follow entry stats use the same descriptors. File type, owner, mode, size, ACL, streamed hash, and read-time identity/size/timestamp checks remain in place. macOS and required `dir_fd`, fd-listing, no-follow-stat, `O_DIRECTORY`, and `O_NOFOLLOW` capabilities are explicit fail-closed gates.

### Residual limit and decision

Directory descriptors prevent a parent-path swap from redirecting a content open after the directory was opened. They do **not** create an atomic snapshot or lock out another process running as the same UID: such a process can still mutate/replace entries, change modes/ACLs, or modify a file after it has been checked. ACL inspection still uses `/bin/ls -lde` on the absolute path and brackets the query with descriptor/path inode identity checks; a swap-out-and-back during that query is not ruled out. The helper must only be considered for a quiescent owner-only intake window, and it remains **unverified**. The source revision is not evidence that the implementation behaves correctly on this Mac or on any supported filesystem. A separate isolated validation/review must precede real use; no such validation was performed here.

The earliest project gate is unchanged: two independent complete v5 human returns, followed by the registered blinded agreement/review/truth checks. Scan v16 still has zero complete and zero partial returns. No labels, reliability result, compute, or policy-efficacy evidence were created.

**Version note:** the absolute-path ACL statements in the earlier review sections describe the helper at those review times and are superseded by this latest follow-up.

## Descriptor-based ACL follow-up

**Reviewed:** 2026-10-17 08:28 CST  
**Execution:** primary-document and source-only review. The helper was not executed, syntax-compiled, or tested; no custody-tree or intake-file access occurred.

### Finding and source correction

Apple documents `acl_get_fd_np(fd, type)` as retrieving any ACL type from the object referred to by an open descriptor, returning an independent copy that must be released with `acl_free`. Apple's public Libc header declares the function and defines `ACL_TYPE_EXTENDED = 0x00000100`, `ACL_FIRST_ENTRY = 0`, `ACL_NEXT_ENTRY = -1`, and the extended allow/deny tags. `acl_get_entry` and `acl_get_tag_type` provide descriptor-local iteration and entry classification. Python's standard `os` API supplies descriptor-relative filesystem operations but no ACL wrapper, so the helper binds the public libc calls through `ctypes` with explicit argument and return types.

The helper no longer invokes `/bin/ls -lde` or runs ACL queries by absolute pathname. Every ACL check now uses the already-open directory or file descriptor. A null ACL result, unavailable symbol, unsupported filesystem, unexpected return/errno, unknown ACE tag, or failed ACL release fails the preflight. Private directories and files still reject every ACE; ancestry checks permit only deny ACEs. For directory listings, the helper opens each no-follow regular-file entry, compares its descriptor identity to the prior no-follow stat, and checks ACLs on that descriptor.

### Limits and decision

- Apple documents `acl_get_entry` as returning `-1` with `EINVAL` for invalid ACL/entry arguments and after iteration has passed the final entry. The source uses `EINVAL` after `ACL_FIRST_ENTRY` on the fresh ACL object as the empty-list case. This exact empty-ACL behavior, the ctypes ABI bindings, symbol resolution through the process C library, and behavior on the target filesystem were not executable-validated here. Keep the helper **unverified** until a separate isolated fixture review confirms empty, allow, deny-only, unsupported-filesystem, and error cases. Any unexpected result remains a generic failure.
- The descriptor query removes the earlier pathname-swap window in ACL inspection. It does not lock against another same-UID process changing ACLs or contents after a check, and does not provide an atomic filesystem snapshot. Continue to require a quiescent owner-only intake window and do not treat this helper as proof against concurrent same-account mutation.
- Two independent complete v5 returns remain the earliest Gate 1 blocker; scan v16 has zero complete and zero partial returns. No label reliability, observer validity, or policy efficacy is established.

### Primary references

- Apple [`acl_get_fd_np(3)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/acl_get_fd_np.3.html): fd-based retrieval, independent ACL copy, `acl_free`, and unsupported-filesystem error.
- Apple [Libc `include/sys/acl.h`](https://github.com/apple-oss-distributions/Libc/blob/main/include/sys/acl.h): public function prototypes and Darwin ACL constants.
- Apple [`acl_get_entry(3)`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/acl_get_entry.3.html): entry iteration API and return/error contract.
- Python [`ctypes`](https://docs.python.org/3.13/library/ctypes.html) and [`os`](https://docs.python.org/3.13/library/os.html): C function bindings and descriptor-relative filesystem API.

## Post-refactor contract regression review

**Reviewed:** 2026-10-17 08:58 CST  
**Scope:** compare the current helper's file, schema, ZIP, and failure-path checks with the coordinator SOP and owner handoff checklist after the dirfd and descriptor-ACL refactors. Source/document review only; no execution, compilation, tests, scanner/scorer, private-tree access, or real return/label/key access.

### Result

One existing overconstraint was found during the focused contract comparison and corrected. It predates the dirfd/ACL refactors, so it is not a regression introduced by those refactors.

- The helper still requires exactly two distinct receipt IDs/tokens, a two-row roster with distinct assignee references, the exact two pinned packet IDs/hashes, matching receipt-to-roster tuples, and the documented independence attestation. It requires the exact roster/receipt/original/staging filename sets and rejects unlisted entries.
- Plain CSV envelope and staged CSV hashes must match. ZIP envelopes remain capped at 12 MB, have exactly one member total, and that member must be a bounded `.csv` with no path-like name, blacklist term, directory marker, Unix special-file type, or traditional/strong encryption bit. The member is streamed to EOF for CRC checking and its actual byte count/hash must match the ZIP metadata, receipt, and staged CSV. This is at least as strict as the coordinator SOP's ZIP exception; it does not parse annotation cells. The SOP requires the two received-envelope hashes to differ, but does not require different staged CSV hashes; the helper no longer rejects identical scored bytes when each envelope, receipt, roster assignment, and packet tuple independently passes.
- The directory-fd and ACL refactors preserve the SOP's owner/mode checks, strict no-ACL rule on the private root/directories/files, and deny-only ancestry exception. Every listed regular file is opened with `O_NOFOLLOW`, identity-matched to its no-follow stat, and checked through the descriptor ACL API. Missing symbols, null ACL results, unsupported ACL filesystems, unexpected errors/tags, schema/ZIP exceptions, and other runtime exceptions collapse to the same `FAIL` result; output remains only `PASS`/`FAIL` plus an opaque run ID.
- The exact v5 CSV header/frame-map/completeness checks remain outside this helper and belong to the later registered scanner/scorer sequence, as both SOP and owner checklist specify. The helper's initial-pair-only scope also remains explicit.

This review does not resolve the prior descriptor-ACL validation limit: empty-ACL `EINVAL`, ctypes ABI/symbol resolution, and target-filesystem behavior still need isolated fixture validation before any real intake. Same-UID concurrent mutation is not excluded. Two independent complete v5 human returns remain the earliest Gate 1 blocker; scan v16 still records zero complete and zero partial returns.
