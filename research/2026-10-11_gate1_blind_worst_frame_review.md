# Gate 1 blinded worst-frame visual review helper

**Updated:** 2026-10-11 15:23 UTC  
**Component:** Gate 1 inter-rater reliability and blind discrepancy review.

## Decision

Keep the existing pooled/per-clip agreement thresholds and the rule to review the worst frame while the path identities remain blind. The agreement scorer already reports per-frame errors and maximum frame indices, but it does not show the corresponding source frame with both submitted boxes. The new offline report builder fills that inspection gap. It does not replace the registered scorer or change the pass rule.

## Implementation

[`build_gate1_blind_worst_frame_review.py`](build_gate1_blind_worst_frame_review.py) accepts two complete rater CSVs and one blinded v5 packet ZIP. It calls the primary agreement scorer with its shared threshold defaults and generates an offline HTML page plus JSON sidecar. Per clip, it displays the maximum radial-center disagreement, nonzero horizontal/vertical maximum frames when they differ, categorical disagreements, and frames whose visibility/count/identity labels make center geometry ineligible. The UI keeps those categories separate from path error and overlays Rater A/B boxes on the actual packet frame.

The helper does not load the private direction key or Kubric truth. It requires matching image paths across both submissions, validates paths against traversal, parses embedded PNGs from the packet's `index.html` without executing JavaScript, rejects key-like ZIP filenames, and writes a static page with an offline-only content security policy. The human reviewer can inspect or correct labels while blind; any correction must be rescored before using the separate known-truth scorer.

## Synthetic CPU validation

[`validate_gate1_blind_worst_frame_review.py`](validate_gate1_blind_worst_frame_review.py) reads the public v5 packet ZIP and blank frame manifest, then creates temporary arbitrary complete label CSVs with a single +2.5 control-grid-cell x disagreement at frame 16 of one opaque blind clip. All other fields/boxes match. The same registered scorer still passes pooled and per-clip p95, as expected from the tail-sensitivity diagnostic. The HTML report selects the injected radial/x maximum, includes one representative zero-error worst frame from the other clip, and contains two offline cards with no script, network URL, direction alias, private key, or fixture truth.

- Result: [`gate1_blind_worst_frame_review_selfcheck_v1.json`](gate1_blind_worst_frame_review_selfcheck_v1.json)
- Builder SHA-256: `bde1f862dcb4b7cc89c536a4ded88066b5ba16aac6a0a580b85db9edc7ab02db`
- Validator SHA-256: `86f4379957ec81bef544e38ba00dda3d3fb4231e17169005937f235580e6ae1b`
- Result SHA-256: `f718a8785daa277fbcf193b2781770a60ee1b291b2b107d5c0247c795a03f55b`
- Runtime: 0.009979 s on Darwin arm64 / Python 3.12.3; CPU only, GPU false, peak VRAM null, no model inference.

## Limits and next step

This validates report selection, box overlay, offline rendering, and key/truth isolation only. The synthetic boxes are arbitrary, the temporary report/CSVs were deleted, and there are still no human labels. It is not annotation-precision evidence, generated-video observer validation, or policy efficacy. The v5 label templates remain blank.

After two independent raters return both mirrored 17-frame CSVs, run the pooled/per-clip agreement screen, then use the helper with either still-blinded v5 packet ZIP to inspect and record each clip's maximum radial frame and any distinct nonzero axis maxima. Preserve categorical visibility/identity disagreements separately. Only after this review and a passing blind screen may the key-based truth audit run. A one-pair pass still requires a second independently rendered mirrored-seed block.
