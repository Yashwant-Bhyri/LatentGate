# CogVideoX-2B five-anchor RGB readout: stopped at coverage gate

**Registered:** 2026-10-12 06:22 UTC  
**Status:** No interpolation metrics produced; fixed-proxy coverage gate failed.

## Preregistered question

Compare the five RGB anchors `[0,2,4,6,8]` with all nine frames in the saved 8-step and 30-step CogVideoX-2B pilots. Use piecewise-linear interpolation for red-mask centroid and pixel area at omitted frames `[1,3,5,7]`, plus schedule-only event-window coverage. The full plan, including all source and PNG hashes, is in [`cogvideox_2b_sparse_frame_preregistration_v1.json`](cogvideox_2b_sparse_frame_preregistration_v1.json), SHA-256 `a684caf7b9cb0a3170cc8803c0119a720e3e45f7df0e015880f298239022f4fd`.

## Integrity and proxy gate

Both runs passed manifest and RGB-file integrity: each has nine sequential 480×320 PNGs, and every manifest/frame SHA-256 matched the preregistration. Their recorded model, prompt, seed, guidance, resolution, and frame count match; the denoising-step values are 8 and 30.

The hash-pinned proxy uses the fixed RGB rule `R>150, G<125, B<125`, a 300-pixel centroid floor, and the existing 36-case threshold grid. The predeclared coverage requirement was 9/9 frames at or above that floor under the fixed threshold and every grid case. **Neither run passed:** each had 0/9 scoreable frames under the fixed rule and 0/36 grid cases with full-frame coverage. The coverage report records the failing frame indices for each case.

Per the preregistration, analysis stopped here. No five-anchor center or area interpolation errors and no event-window coverage numbers were computed; no threshold was changed. The existing [8/30-step scoreability audit](2026-10-12_cogvideox2b_stepsensitivity_rgb_report.md) independently records zero threshold-positive pixels in the 8-step clip and a 203-pixel maximum in the 30-step clip, below the same 300-pixel floor.

## Limits and handoff

The sparse-frame question remains unanswered for these two clips because the pinned red-mask proxy cannot score them. This is a failed measurement gate, not evidence for or against sparse sampling. Do not tune the threshold on these outputs. A future sparse-readout check needs RGB clips that pass a preregistered proxy-coverage gate.

These two 480×320 runs are sensitivity variants with the same recorded prompt and seed, not independent samples. Their difference from the separate 50-step 720×480 run is confounded by resolution and output length, so no causal denoising-step conclusion is supported.

The CPU-only runner is [`audit_cogvideox_2b_five_anchor_sparse_readout.py`](audit_cogvideox_2b_five_anchor_sparse_readout.py); the complete fail-closed coverage table is [`cogvideox_2b_five_anchor_readout_v1.json`](cogvideox_2b_five_anchor_readout_v1.json). No latent file was read, and no inference, GPU, human label, or generation was used.
