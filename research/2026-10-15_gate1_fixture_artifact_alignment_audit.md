# Gate 1 fixture artifact-alignment audit

**Checked:** 2026-10-15 07:49 UTC  
**Component:** Gate 1 — verify the known-truth fixture used to score independent annotations.  
**Decision:** saved Kubric mask/box/projection artifacts align; the human-label reliability gate remains open.

## Method and evidence

Read only the current `research/kubric_gate0_fixture_v1` files. The reproducible audit [`audit_gate1_fixture_artifact_alignment.py`](audit_gate1_fixture_artifact_alignment.py) walks the two 17-frame truth lists, checks that every referenced RGB and mask exists inside the fixture, validates 160×96 dimensions and PNG modes, recomputes each binary mask's pixel count, half-open box, and 8-connected component count, and compares those values with `ground_truth.json` and its embedded annotation record. It independently compares each mask-box midpoint with the stored continuous projected center. It records SHA-256 for all 68 image artifacts. The aggregate result is [`gate1_fixture_artifact_alignment_audit_v1.json`](gate1_fixture_artifact_alignment_audit_v1.json).

The pinned fixture implementation is the local Kubric/Blender renderer and source-mask scorer; the upstream method and source references are in the [Kubric paper](https://openaccess.thecvf.com/content/CVPR2022/html/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.html) and [official repository](https://github.com/google-research/kubric). This was an artifact consistency audit, not a new claim about Kubric's general annotation quality.

## Results

- **Pass:** 34 truth rows align with 34 RGB frames and 34 masks; no missing, duplicate, out-of-root, or extra indexed files were found. All are 160×96. RGB files are RGBA; masks are single-channel binary 0/255.
- **Pass:** all mask pixel counts and half-open boxes equal the JSON truth. Each target mask is one 8-connected component in every frame. Both mirrored paths have exactly 20×20 px target boxes and zero mask-pixel-area CV.
- **Pass:** across all frames, the mask-box-center versus projected-center gap has mean `1.63e-6 px`, p95 `3.81e-6 px`, and max `3.81e-6 px`, consistent with serialization precision.
- **Separate background diagnostic:** outside the union of each frame's target mask and frame 0's target mask, exact RGB change fraction averages 13.62% on the right path and 13.64% on the left path; maxima are 16.89% and 17.14%. Mean absolute channel deltas are 1.67/255 and 1.68/255. This region includes object-induced shadow changes; it is not a camera-motion estimate and must not be folded into the path score. The values are consistent with moving-shadow variation, but the audit did not spatially classify the changed pixels.
- The current cached `fixture_report.json` has no per-frame RGB hash map, so this run recorded hashes for future artifact continuity instead of claiming a comparison against a prior RGB hash baseline. The report and ground-truth JSON hashes are pinned in the result.

## Interpretation, limits, and next action

This closes only a fixture-integrity subcheck: it validates that the known-truth boxes are derived consistently from the saved masks and that their centers agree with continuous projected truth. It **does not** establish independent-rater agreement, validate a generated-video observer, or pass Gate 1. Keep the v5 human labels and blind-before-truth order unchanged.

For later background-stability scoring, raw pixel differences outside the object mask are inadequate on their own: they count moving shadows as background change. Use static landmark/feature tracks and separately document an object-shadow exclusion or handling rule. Keep that result separate from path adherence, visibility/identity, and apparent-size guardrails.

The exact next Gate 1 input remains two independent complete v5 CSVs; require the registered pooled and per-clip screen, blind worst-frame disposition, guarded truth score, and independent mirrored-seed replication. No labels or unblind key were read. No model, renderer, external dataset, cloud session, GPU, or human data ran. CPU-only Python/Pillow/NumPy file analysis took `0.068466 s` internally on Darwin arm64/Python 3.12.3; peak VRAM is not applicable. Dependency and generated-artifact distribution terms remain subject to the prior [Kubric/Blender license audit](2026-10-15_gate1_kubric_blender_license_audit.md); these files remain local.
