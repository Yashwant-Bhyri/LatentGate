# Gate 1 return-storage control metadata audit

**Checked:** 2026-10-16 22:37 Asia/Shanghai  
**Scope:** Metadata and access-mode inspection only for the known Gate 1 parent and proposed return-storage paths. No directory entries or filenames were listed; no files, labels, CSVs, ZIPs, aliases, keys, or archive members were opened. No directories or permissions were changed.

## Finding

There is no existing restricted Gate 1 return location. The project `research/` directory and v5 packet directory both exist as `drwxr-xr-x` (`0755`), owned by `yash:staff`, with no ACL entries displayed. The known `returns/`, `returns/originals/`, `returns/staging/`, and `returns/receipts/` paths are absent. Thus the SOP's restricted immutable-originals area, opaque working area, and receipt custody location have not been provisioned or verified.

| Known path | Exists | POSIX mode | Owner/group | ACL metadata |
|---|---:|---:|---|---|
| `research/` | yes | `0755` | `yash:staff` | none displayed |
| `research/gate1_manual_annotation_packets_v5/` | yes | `0755` | `yash:staff` | none displayed |
| `research/gate1_manual_annotation_packets_v5/returns/` | no | — | — | — |
| `.../returns/originals/` | no | — | — | — |
| `.../returns/staging/` | no | — | — | — |
| `.../returns/receipts/` | no | — | — | — |

The existing `0755` parents are not restricted storage for returned labels. Their mode alone does not prevent creating a new owner-only child, but no such child exists yet and no inherited ACL boundary has been established for it.

## Required coordinator setup before accepting returns

Create a dedicated Gate 1 return root with separate `originals/`, `staging/`, `receipts/`, and private `roster/` locations. Restrict the return root and each child to the coordinator account (for example POSIX `0700`), then inspect both mode and ACL/inherited access before use. Preserve original envelopes only in `originals/`; after hashing, prevent accidental editing there. Put opaque receipt-based CSV working copies only in `staging/`. Keep receipt sidecars and the token-to-person roster coordinator-only, separately from label CSVs and scoring outputs. Record the actual paths and verified modes/ACLs in the custody log. Do not use the existing broad `research/` or v5 packet directory as the return destination.

This is a storage-readiness finding, not evidence of any exposure or a permission change. The last packet-bound inventory, scan v14, found zero complete or partial human returns; two independent complete v5 CSVs remain the Gate 1 prerequisite. No fresh content scan was performed in this metadata-only audit.
