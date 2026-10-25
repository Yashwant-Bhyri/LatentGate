# Gate 1 observer stress: occlusion and same-color distractor

**Run:** 2026-10-15 08:37:40–08:44:46 UTC  
**Experiment:** `EXP-SAM2-VISIBILITY-DISTRACTOR-STRESS-001`  
**Component:** observer preparation within the still-open Gate 1; human-label reliability remains the earliest incomplete gate.

## Question and protocol

The clean SAM 2.1 Tiny probe already showed a repeatable downward box-center bias on the paired Kubric fixture. This follow-up asks whether a prompted mask tracker and the red-centroid observer behave sensibly under two controlled nuisances before either is allowed near the feedback controller.

Starting with the existing left/right 17-frame Kubric renders and saved masks, the probe created four deterministic image sequences: each path direction with (a) a fixed, full-height cyan pole placed across the target trajectory, or (b) a static red circular distractor placed away from the target path. The pole occludes 23.53% of target-mask area on average; the distractor leaves the target fully visible. SAM 2 received the canonical first-frame target-center positive point only, with no later prompts or corrections. Its predicted masks were scored against the visible target mask; tight-box centers were scored against the full projected sphere box center. The red observer used the existing nine-threshold ensemble and the registered 0.65 in-sample stability cutoff. Its relative-path residual was also recomputed after its existing start-anchor calibration.

The first local preflight invocation loaded the cached checkpoint but stopped before inference: the fixture reader called a nonexistent PIL `Image.__array__` method. The reader was corrected to use `np.asarray`; the reported run is the subsequent successful execution. The failed preflight produced no masks or experiment report and did not download data or weights.

This is a hand-designed synthetic stress case, not a held-out observer benchmark. The first-frame point is an oracle prompt; the RGB sequences and masks derive from the small Kubric fixture. It tests neither object discovery nor appearance diversity in generated videos.

## Results

| Perturbation | Direction | SAM 2 mask coverage | SAM 2 IoU to visible truth | SAM 2 mean full-box center error | Red-centroid mean center error | Current red gate at controller anchors |
|---|---:|---:|---:|---:|---:|---|
| Fixed pole occlusion | Left | 100% | 0.742 | 3.03 px | 2.65 px | 5/5 accepted; 2 future accepted anchors exceed 0.5 grid cell after start calibration |
| Fixed pole occlusion | Right | 100% | 0.745 | 3.02 px | 2.61 px | 4/5 accepted; 1 future accepted anchor exceeds 0.5 grid cell after start calibration |
| Static red distractor | Left | 100% | 0.753 | 2.68 px | 42.01 px | 0/5 accepted; start anchor fails closed |
| Static red distractor | Right | 100% | 0.752 | 2.70 px | 41.75 px | 0/5 accepted; start anchor fails closed |

The pole cases preserve the existing SAM 2 vertical localization bias (mean signed error about +0.74–0.75 control-grid cell in y). More importantly for path feedback, the red observer accepts some partially occluded anchors whose **post-anchor relative** x residual is about 0.86–1.15 grid cells on the left path and 0.90 cells on the right path. Threshold agreement alone therefore does not detect this synthetic dynamic occlusion error. The red distractor makes the global red-centroid shift by roughly 42 px, but in these two fixtures the stability heuristic rejects all five controller anchors, so the current adapter would fail closed.

The clean-fixture SAM 2 mean full-box-center errors were 2.64/2.73 px in the earlier probe; the distractor cases remain near that bias, while pole occlusion increases the mean to about 3.02 px and p95 to 4.47–4.56 px. Thus SAM 2 remains a useful prompted **mask proposal**, but a mask-box midpoint is not yet a valid center measurement under partial occlusion. The confidence heuristic rejects this particular red distractor but is not a general identity certificate.

## Evidence, resources, and limits

- Pinned checkpoint: `facebook/sam2.1-hiera-tiny`, revision `de431c4043854a71d8101e17995dfe596bf101a5`, Apache-2.0. It was loaded from the existing local Hugging Face cache with `local_files_only=True`; no weights or data were downloaded.
- Source fixture: `research/kubric_gate0_fixture_v1`, whose source code is pinned to Kubric commit `61f2422c84bab75006df33c6989e0b483db3ccfe`. The fixture audit records Kubric source as Apache-2.0, while generated-artifact licensing remains unresolved because the renderer also uses Blender's `bpy` API. Preserve that open licensing boundary; do not redistribute this derived set as Apache-only.
- Device: local Apple M3 / MPS, FP16; no CUDA or cloud runtime. Elapsed wall time was 426.21 s. Dedicated VRAM is not applicable to unified memory. A 0.2 s sampler measured peak MPS driver allocation of 15,275,196,416 bytes; this is an approximate sampled unified-memory figure, not dedicated VRAM. No paid compute was used.
- `report.json` stores per-frame outputs and synthetic input/truth hashes. [`artifact_sha256_v1.json`](sam2_visibility_stress_v1/artifact_sha256_v1.json) verifies all 208 frame, mask, and contact-sheet artifacts; no hash mismatch was found. The controller-anchor analysis is [`controller_anchor_analysis_v1.json`](sam2_visibility_stress_v1/controller_anchor_analysis_v1.json).
- Path error, visible-mask IoU/coverage, and red-observer confidence were analyzed separately. The fixed background/pole were compositor-controlled, not a camera-motion measurement. Apparent-size control, human identity labels, generated-video observer accuracy, and policy efficacy were not measured.

## Decision and next falsifiable step

Keep the red threshold and SAM 2 as secondary proposals. Do not let stability alone actuate on a path if the observed target may be occluded; require separate visibility/identity evidence and center-error evidence. Do not tune the 0.65 cutoff using these four cases. The next valid Gate 1 action is still to obtain two independent complete v5 human-label CSVs, pass the registered pooled/per-clip agreement screen and blinded worst-frame review, then run the guarded truth score. If that passes, validate any proposed generated-frame observer on fresh held-out clips with path error, visibility/identity, background stability, and apparent-size guardrails reported separately.

**Falsifiable prediction:** on an independently rendered partial-occlusion path, a red stability score alone will again permit at least one control anchor whose start-calibrated center residual exceeds 0.5 control-grid cell; a visibility-aware rejection gate should reject that anchor without rejecting the matched unoccluded anchor. Neither half of this prediction is established by the present four-case fixture.
