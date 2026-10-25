# Gate 1 v5 owner handoff: independent reliability labels

**Prepared:** 2026-10-16 UTC  
**Scope:** CPU-only audit of the v5 synthetic-reliability packet, blank templates, intake scan, and scorer path. Shared `STATE.md` and the experiment registry were not edited.

## Current state and remaining human action

The latest inventory is [`gate1_label_intake_scan_v16.json`](gate1_label_intake_scan_v16.json), created at 2026-10-16 15:45:29 UTC. It searched 7,232 CSV/ZIP paths and found 24 exact-v5-header candidates: two blank canonical templates, 20 blank or wrong-map candidates, two known QA/synthetic exports, and **zero complete or partial human returns**. Its approved roots were project `research/`, Downloads, and Desktop; the private owner-only Application Support tree was excluded. Its reference packet hash is `0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8`.

**The gate is waiting on the owner to arrange two independent human annotations.** No annotators have been contacted by this audit. Give each person one ZIP, separately, and ask each to annotate every frame independently. Do not send either person the other person's work, any tracker/SAM2 proposals, or an unblinding key. Do not reveal the mirrored clip ordering or any hypothesized direction/condition.

## Materials to distribute and return

Use only the two v5 synthetic-reliability packages:

- `research/gate1_manual_annotation_packets_v5/synthetic_reliability/rater_01.zip` — SHA-256 `0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8`
- `research/gate1_manual_annotation_packets_v5/synthetic_reliability/rater_02.zip` — SHA-256 `8a660acb5bd77855b82c13b27cc951f8d2e4d6131e38e0144e83638e2f92e7ff`

Each archive is a distinct blind packet for the same two mirrored Kubric clips in a different order. Each contains an offline HTML annotator, blank CSV fallback, and instructions. Each blank template has 34 data rows (17 frames per blind clip), the exact 11-column v5 header, and no labels. The two archives are verified as distinct files; the packet mapping must stay intact.

Ask the two raters to download their completed CSV from the local page, close nothing until the CSV is saved, and return the plain CSV (not a screenshot) through the coordinator's controlled channel. Preserve each original export unchanged. The page is offline and does not upload annotations. Do not place genuine returns in the repository, Downloads, Desktop, or another shared project folder. The coordinator must preserve the received bytes under the verified owner-only `originals/` path and make opaque working copies under `staging/`, following the [coordinator intake SOP](2026-10-16_gate1_assignment_intake_sop_draft.md). Do not create or use the legacy project-tree `returns/` folder.

## Neutral annotation brief

For all 34 frames in their own packet, each rater should enter visibility, instance count, identity consistency, and bounding box as **separate judgments**. Draw the tight visible silhouette in source-frame pixel-edge coordinates; exclude shadows and detached glare. Label every frame independently; do not interpolate. If the object is absent, record visibility `false`, count `0`, identity consistency `false`, and leave the box empty. The v5 schema is binary for identity: record `false` when identity is No or uncertain; when the judgment is uncertain, write `identity uncertain` in the existing `notes` field rather than entering the literal value `uncertain` in `identity_consistent`. This preserves the existing schema and keeps uncertain identity out of path-center eligibility. Preserve blind clip IDs, frame indices, and image filenames. The task is label reliability on the Kubric fixture, not a test of path-following performance.

## Owner-side intake and scoring sequence

1. After two genuine, independent returns arrive, follow sections 2–3 of the [coordinator intake SOP](2026-10-16_gate1_assignment_intake_sop_draft.md) to preserve immutable originals, verify assignment receipts and hashes, confirm distinct people/tokens, and create opaque staged CSV copies. The broad candidate-inventory scanner is not run over private submissions and does not establish authorship or independence.
2. Follow section 4 of the SOP for scorer inputs, pinned packet hash verification, command arguments, and private output/log paths. Run the registered blinded agreement screen at the v5 source size. It requires pooled **and per blind clip** thresholds: joint visibility/count/identity agreement ≥0.95, visible-center pair coverage ≥0.90, and p95 center disagreement ≤0.5 target-grid cell on each axis. Leave defaults unchanged.
3. After agreement passes, use the still-blinded worst-frame review in the order and private staging paths specified by the SOP. Keep original CSVs immutable; do not correct them after viewing paired review. If the screen fails, keep the pair blinded, diagnose only packet/transcription or instruction problems without showing one rater the other's labels, and obtain a fresh independent pair if there is a genuine disagreement or process defect.
4. Only after both agreement and blind review pass, run the guarded truth scorer on the same unchanged staged submissions using the SOP's command contract. It recomputes the agreement gate before accessing truth. Do not open, copy, or send `PRIVATE_UNBLIND_KEY.json` manually.
5. Keep visibility, instance count, identity, box-center/path error, and apparent extent as separate measures. Passing this pair is only an annotation-process screen for one mirrored-seed block; it does not establish observer validity or policy efficacy. A second independently rendered mirrored-seed block is still needed for the full reliability check.

## Audit evidence and limits

- v5 packet instructions: [`README.md`](gate1_manual_annotation_packets_v5/README.md).
- The earlier independence audit documents the v3 auto-coupling defect and why v5's independent fields are required: [`2026-10-11_gate1_annotation_independence_audit.md`](2026-10-11_gate1_annotation_independence_audit.md). v5's README supersedes older packet versions for distribution.
- The agreement scorer's registered defaults are 0.95 categorical agreement, 0.5 grid-cell p95 center disagreement, and 0.90 visible-center coverage; the scorer evaluates each clip as well as the pooled set.
- The scan is point-in-time and only classifies files it can find in its configured roots. It cannot verify human independence. No label values were scored in this audit, no private key was read, and no model, cloud runtime, GPU, or paid compute was used. The coordinator SOP controls owner-side custody, scanning, scoring, and output storage.
- A mistaken `--help` call on `validate_gate1_label_intake.py` executed its synthetic scanner self-check instead of showing help and regenerated `gate1_label_intake_scanner_selfcheck_v1.json`. It reports six synthetic-only checks passing; it read no labels or private key and is not a human-label or scoring result.

**Falsifiable next step:** after two independent, full v5 CSVs arrive, both must appear as complete packet-matched candidates and pass the unchanged pooled and per-clip agreement thresholds before any truth unblinding or observer/policy experiment proceeds.
