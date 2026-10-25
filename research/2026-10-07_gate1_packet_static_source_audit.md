# Gate 1 packet static source audit

**Checked:** 2026-10-07 11:58:50 UTC  
**Scope:** Earliest incomplete component, two-rater reference annotation reliability. Static inspection only; no browser interaction, labels, model inference, or GPU allocation.

## Findings

- Reopened all four CSVs inside `research/gate1_manual_annotation_packets_v3/`: each synthetic Kubric rater CSV has 34 rows and zero nonblank annotation cells; each Wan M0 convenience CSV has 17 rows and zero nonblank annotation cells. No human labels have arrived.
- Each ZIP contains exactly `index.html`, `annotation_template.csv`, and `README.txt`. The private unblind key is outside the ZIPs. The HTML embeds frames as data URIs, and no machine observer proposals are included.
- Audited `research/build_blinded_box_packets.py` and all four generated HTML files. The page uses a restrictive `default-src 'none'` CSP with data images and inline CSS/JS; after removing embedded PNG payloads, the files contain no external script references, URLs, or network API calls. Frame labels initialize blank. Raters receive the same frame-to-ID mapping; the synthetic pair is ordered in opposite clip order for the two raters. The M0 convenience pair contains the same 17-frame sequence under the same blind ID.
- This confirms static packet construction and blinding metadata only. It does not establish that local browser drawing, persistence, or CSV export/import works; that round-trip remains unverified. Generic `left`/`right` string matches are CSS spacing/mobile layout and keyboard arrow names, while `policy` is a substring of the CSP directive; no condition identifiers were found outside embedded frame data.

## Decision and next falsifiable step

Gate 1 remains the active gate; do not advance to open-loop model inference. On the next label return, require both synthetic CSVs to contain complete annotations for both blind clip IDs, then run the existing scorer with `--expected-clips 2` and inspect pooled and per-clip thresholds plus worst-frame disagreement. A pass on this single mirrored pair is only a process screen; full reference reliability still requires the independent second mirrored-seed block. Keep the M0 labels separate as one-clip observer convenience evidence. If labels remain blank, retain the gate without claiming human-reference validation.

No source edit was justified by this audit. The unresolved operational check is browser round-trip; the independent blocker remains that two raters have not returned labels.
