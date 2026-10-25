# Gate 1 offline manual annotation packets

**Superseded — do not distribute these v3 ZIPs.** A source audit found that drawing a box auto-filled visibility, instance count, and identity, while selecting “No” visibility auto-filled count and identity. Those defaults could manufacture agreement on scored categorical fields. The v3 ZIPs are retained unchanged as provenance; use [`gate1_manual_annotation_packets_v4`](../gate1_manual_annotation_packets_v4/README.md), whose fields are independent.

Distribute each ZIP separately to an independent rater; never send `PRIVATE_UNBLIND_KEY.json`. Each ZIP contains one offline `index.html`, a blank CSV fallback, and instructions. The page embeds its frames, has no network calls, supports framewise tight-box drawing and CSV import/export, and does not interpolate.

`synthetic_reliability/` contains the paired Kubric mirror fixture in a different clip order for the two raters. `generated_sample_observer_check/` contains separate blinded aliases of the same saved Wan M0 sample so that human boxes can be compared with the red/SAM2 proposals without showing raters those proposals.

Keep returned CSVs separate and blinded until both are complete. Do not score blank templates. The Kubric labels measure annotation reliability only; the generated sample is a one-clip observer convenience check, not policy efficacy.

When comparing returned files, the scorer requires an explicit expected clip count: use `--expected-clips 2` for the paired Kubric process screen and `--expected-clips 1` for the one-clip M0 observer check. It rejects a submission missing a complete clip, even when both raters omitted the same clip.
