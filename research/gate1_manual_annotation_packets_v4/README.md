# Gate 1 offline manual annotation packets

Distribute each ZIP separately to an independent rater; never send `PRIVATE_UNBLIND_KEY.json`. These v4 packets correct a v3 UI flaw: v3 drawing/visibility actions auto-filled scored state labels and is superseded. Each ZIP contains one offline `index.html`, a blank CSV fallback, and instructions. The page embeds its frames, has no network calls, supports framewise tight-box drawing and CSV import/export, and does not interpolate. Visibility, instance count, identity, and box are entered independently; no action auto-fills another scored field.

`synthetic_reliability/` contains the paired Kubric mirror fixture in a different clip order for the two raters. `generated_sample_observer_check/` contains separate blinded aliases of the same saved Wan M0 sample so that human boxes can be compared with the red/SAM2 proposals without showing raters those proposals.

The synthetic Kubric frames are 160×96; their CSV boxes are raw source-frame pixel coordinates. When scoring this pair against the Wan control grid (832×480, 52×30), pass `--source-width 160 --source-height 96` to the agreement scorer; it maps x/y separately before applying the half-cell threshold. For the native 832×480 M0 sample, pass `--source-width 832 --source-height 480`. See [`coordinate-scale audit`](../2026-10-11_gate1_coordinate_scale_audit.md).

Keep returned CSVs separate and blinded until both are complete. Do not score blank templates. The Kubric labels measure annotation reliability only; the generated sample is a one-clip observer convenience check, not policy efficacy.
