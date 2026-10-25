# Measuring path drift at Wan’s 832×480 / 52×30 control scale

**Checked:** 2026-10-02  
**Scope:** A defensible label and score contract for one moving target under a fixed camera, including the current Wan spatial-control grid, identity/visibility failures, and annotation uncertainty.  
**Evidence boundary:** Existing fixture data and reports were inspected; one bounded coordinate/quantization calculation was run on existing Kubric ground truth. No model, GPU, or cloud run was performed, and no shared code, notebook, or `STATE.md` was edited.

## Decision

For the current red-ball task, define the observed location as the **tight visible-object box midpoint**, annotated at the native 832×480 frame resolution. Score that continuous pixel-space location against the original requested trajectory at each matching output frame. Keep raw center error in pixels, normalize it by the frame diagonal, and also report it in spatial-cell units (`pixels / 16`). Do not round observed centers to the 52×30 grid: that would throw away precisely the sub-cell motion the measurement is meant to expose.

Treat target disappearance, identity ambiguity, duplicates, and camera movement as separate measured failures, not as opportunities to drop frames or reselect the closest object to the requested path. In the first efficacy set, keep paths and scenes unoccluded by design; record any generated disappearance or occlusion as a failure/guardrail. Use a separate explicitly labeled stress set before making claims about motion through occlusion, because a visible-part box midpoint no longer estimates the hidden object’s center reliably.

This is consistent with motion-control evaluation precedent: Tora uses average trajectory error between generated and requested tracks; Wan-Move’s MoveBench supplies point and mask annotations for 480p motion-control clips. The protocol here keeps that path-error idea and makes the target identity, observation uncertainty, and fixed-camera conditions explicit. [Tora CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Tora_Trajectory-oriented_Diffusion_Transformer_for_Video_Generation_CVPR_2025_paper.pdf), [Wan-Move paper](https://arxiv.org/abs/2512.08765), [MoveBench annotation description](https://huggingface.co/datasets/Ruihang/MoveBench)

## The current grid in measurable units

At 832×480 and a 52×30 spatial latent grid:

| Quantity | Pixels | Grid units | Frame-diagonal normalized |
|---|---:|---:|---:|
| One cell along either axis | 16 px | 1 cell | 0.01666 |
| Half cell along one axis | 8 px | 0.5 cell | 0.00833 |
| One-cell diagonal displacement | 22.63 px | 1.414 cells Euclidean | 0.02356 |
| Half-cell error on both axes (corner) | 11.31 px | 0.707 cells Euclidean | 0.01178 |
| Full frame diagonal | 960.53 px | 60.03 cells Euclidean | 1.0 |

`1 px = 0.0625 cell`; a one-pixel center-label difference is small compared with the control cell, but visual ambiguity and tracker bias can be much larger. The current human agreement rule of p95 ≤8 px **per axis** is half a cell on x and y; if both components simultaneously reach that bound, the radial difference is 11.31 px (0.707 cell). Thus that gate is sensible for detecting one-cell directional actuation, but it does not by itself establish sub-half-cell radial precision or support a claim whose entire gain is below this uncertainty. Report the effect in pixels and cells beside the rater-disagreement distribution.

**Resolution scope correction (2026-10-11):** the table and raw 8 px statement above are in native 832×480 target-image coordinates. The Gate 1 Kubric packet is 160×96, so its raw CSV pixel deltas must first be mapped per axis to 832×480 / 52×30. The half-cell p95 screen is 1.5385 raw x-px and 1.6 raw y-px on those Kubric images. The earlier scorer applied 8 px directly to those low-resolution coordinates; see the [coordinate-scale audit](2026-10-11_gate1_coordinate_scale_audit.md) for the correction and diagnostic.

Use pixel-edge coordinates with the existing annotation convention: `[x1,y1,x2,y2]`, right and bottom edges exclusive; center is `((x1+x2)/2, (y1+y2)/2)`. Keep x increasing right and y increasing down. For a displacement, `Δgx = Δx/16` and `Δgy = Δy/16`. If an absolute grid-center index is needed for a plot, cell `(0,0)` is centered at `(8,8)` px, so its coordinates are `(x/16−0.5, y/16−0.5)`. Keep actual labels in pixels; cell units are a scale conversion, not quantization.

Choose commanded path waypoints on cell centers for this first 52×30 pilot. That avoids asking the actuator to resolve sub-cell positions it cannot encode. Store both the original user-space path and the exact rasterized DiTraj boxes/masks. Score output against the **original requested path**, not against the policy’s own residual-corrected boxes; log the rasterized and corrected boxes separately so actuator mechanics remain auditable.

## Bounded local calibration on existing fixtures

I read the existing paired Kubric Gate 0 fixture: 34 RGB/mask frames, 17 per mirrored path, rendered at 160×96 with exact projected centers and a 20×20 px target. The scorer was validated against exact left/right truth; its report checks resolution-normalized error, missing/duplicate/identity penalties, and annotation CSV round trips. Its visual truth does not establish human reliability or generated-video observer accuracy. [Fixture description and report](kubric_gate0_fixture_v1/README.md), [scorer validation summary](gate0_scorer_validation_v1.json)

I performed a read-only coordinate calculation over those 34 exact projected centers: scaled their normalized locations to 832×480, then rounded each observed center to its nearest 16×16 cell center. This is a resolution/quantization diagnostic, not a model run.

| Result on existing 17-frame path | Left | Right |
|---|---:|---:|
| Distinct quantized x cells | 12 | 11 |
| Adjacent steps erased into repeated x cell | 5 of 16 | 6 of 16 |
| Center quantization RMSE | 9.23 px (0.577 cell) | 9.23 px (0.577 cell) |
| Maximum quantization error | 11.31 px (0.707 cell) | 11.31 px (0.707 cell) |

The source path advances about 0.65 cell per frame in this normalized mapping. Nearest-cell scoring turns part of its smooth motion into plateaus. Therefore: **calculate all center errors before any grid conversion**, report continuous cell units only for interpretation, and reserve cell quantization for the actual control-input construction. The existing 160×96 fixture itself has about a 5× scale-up to the Wan image size (`832/160=5.2`, `480/96=5.0`); it validates scoring geometry and mask ingestion, not the appearance-scale behavior or annotation precision of a native 832×480 Wan clip.

The existing synthetic red-centroid ensemble has mean center error 0.00927 frame diagonals on the fixture, p95 0.01001, and mean anchor-relative trajectory error 0.03794 cells. At Wan resolution those absolute normalized errors correspond to about 8.90 px mean and 9.61 px p95 (approximately 0.56 and 0.60 cell radially); the smaller anchor-relative result follows after cancelling the fixed initial offset. This supports a useful development observer on that simple red sphere, but it is not a calibrated probability, generated-Wan result, or evidence of absolute unbiased localization. Its development cutoff was selected on the same two paths. [Observer calibration artifact](ditraj_red_observer_kubric_calibration_v1.json)

The current annotation assets are still **blank**: the two Kubric reliability copies have 17 rows with no boxes, and the two-copy Wan M0 packet has 34 rows with no boxes. Human-human agreement and human-versus-observer error have not yet been measured. The Wan packet frames are native 832×480; use those for actual generated-video label precision. [Kubric reliability package manifest](kubric_gate0_annotation_reliability/package_manifest.json), [Wan M0 packet instructions](wan_m0_observer_labels_v1/README.md)

## Label contract for generated frames

1. **Blinding.** Each rater labels every output frame without seeing the target path, arm name, SAM/TAP/CoTracker output, color-threshold track, or another rater’s labels. Keep anonymized duplicates and the unblind key separate. This prevents target-directed tracker selection and visual confirmation bias.
2. **Identity and presence.** In every frame, record whether the designated target from frame 0 is visible, how many matching instances are visible, and whether identity remains unambiguous. For the red sphere, continuity is simple unless the model duplicates the object or introduces another similar ball. If identity is ambiguous, mark it ambiguous; do not guess from proximity to the requested path. Count duplicates even if one candidate is easy to track.
3. **Geometry.** For a fully visible target, draw a tight pixel-edge box around the target’s visible silhouette and record its midpoint as the scored center. Annotate in native resolution and do not draw a box around the commanded target region. For the current symmetric ball, this midpoint is a clear operational center proxy. For non-symmetric objects, first define the anchor separately (e.g., canonical point or simulator-projected object origin); do not assume a box midpoint is invariant to object rotation.
4. **Visibility quality.** Record partial occlusion, blur, clipping at the frame edge, shape changes, or uncertain boundaries in notes. The current template has boolean presence plus notes; for any later occlusion-focused study, add explicit visible/occluded/out-of-frame labels and, where ground truth permits, visible-area fraction. Do not infer a hidden geometric center from a partial tight box and score it as if fully observed.
5. **Adjudication.** Preserve both raw annotation files. Compute per-frame inter-rater `|Δx|`, `|Δy|`, and radial center difference in pixels and cells. If identity/presence differs or a center disagreement exceeds the gate, adjudicate from the original frames while still blind to treatment and proposed tracks. On known-GT Kubric, also compare each rater and the adjudicated labels to the projected simulator center: two raters can share the same systematic bias.
6. **Background.** Label at least three static background landmarks where visible; save their frame-0 locations and track them across the clip. Report how many remain measurable. A visually uniform tabletop may not supply enough landmarks; mark camera stability inconclusive in that case rather than interpreting low optical flow as proof of a static camera.

The currently documented observer release gate should be retained: double-label two mirrored seed blocks, require ≥95% agreement on identity/visibility/count and human-human p95 center disagreement ≤0.5 target-grid cell per axis (8 px **at 832×480**); for the 160×96 Kubric packet, first map the raw source coordinates per axis, which gives 1.5385 source x-px / 1.6 source y-px. Use three development blocks to freeze the observer and confidence cutoff, then five fresh held-out blocks for the locked observer check. Require ≥90% observer coverage on uniquely visible frames, at least 4/5 accepted control anchors per clip, median center error ≤0.5 cell per axis, p90 ≤1 cell per axis, and zero accepted wrong-ID/duplicate/absent detections at control anchors. The gate is in the existing [component plan](2d_path_policy_component_plan.md); this memo clarifies that rater agreement should also be checked against synthetic truth and interpreted against the effect size.

## Scorecard and failure accounting

For frame `t`, let `r_t` be the requested center and `c_t` the human-adjudicated generated center in the same pixel coordinate convention; let `D=sqrt(832²+480²)=960.53 px`.

- **Primary path score:** `ADE = (1/T) Σ e_t / D`, where `e_t=||c_t−r_t||₂`. Keep the native-resolution center; include every expected output frame. The current scorer assigns max normalized error `1.0` to a missing, duplicate, or identity-ambiguous center and reports coverage separately.
- **End point / direction:** report final-frame error, signed endpoint progress along the commanded direction, and monotonic-step fraction for straight horizontal paths. For reversals or curved paths, compare each frame to its time-aligned requested center; do not use dynamic time warping or re-time the video.
- **Presence and identity:** report unique-target visibility fraction, identity failures, duplicate frames, absent frames, and ambiguous frames separately. A clip with poor track coverage cannot look good by dropping difficult frames from ADE.
- **Camera and object guardrails:** report median and p95 landmark displacement; target box width/height and area variation; mask area if available; and blinded visual quality. Do not stabilize or camera-compensate the primary score. Keep path accuracy and appearance stability as separate outcomes.
- **Observer uncertainty:** report coverage-versus-error on held-out labels, not just a mean confidence score. A SAM predicted-IoU score or the red-threshold stability heuristic is not a calibrated probability until its coverage and error are measured on held-out generated clips. Keep proposals and final labels separate.
- **Inference over cases:** the independent unit is a paired case/seed, not each of its 17 correlated frames. Report each paired per-clip difference and confidence interval at the case level. Frames support the within-clip path trace; they are not 17 independent policy samples.

This matches the useful split in TAP-Vid: point localization is scored only on visible points, while visibility is reported separately and Average Jaccard combines position and visibility. For our generated-object benchmark, use those concepts for observer diagnostics but do not replace the full-frame-diagonal missing penalty or object-level identity score. [TAP-Vid paper and metrics](https://arxiv.org/abs/2211.03726), [official implementation](https://github.com/google-deepmind/tapnet)

## Observer roles and failure modes

- **Primary reference:** blind human box/ID/presence labels, anchored to a fixed frame-0 target. The existing 17-frame convenience packet can compare automated proposals to human boxes on one M0 sample only; the five-held-out-block gate remains necessary.
- **Mask proposal:** SAM 3.1 is the newest official candidate in the previous observer memo; SAM 2.1 tiny is already locally measured. Use a first-frame visual prompt and preserve raw per-frame masks/IDs. A segmentation mask estimates visible pixels and may drift, grow, shrink, merge, or split; mask centroid can move under asymmetric occlusion or holes. The existing SAM 2.1 probe’s repeatable vertical bias exceeds the mapped 8 px half-cell component limit, so raw SAM2 cannot become the control observer without a validated correction/replacement. [SAM 3.1/observer comparison](2026-10-02_observer_benchmark_research.md), [local SAM 2.1 probe](2026-10-02_sam2_observer_probe_report.md)
- **Motion cross-check:** use CoTracker3 offline or TAPNext++ on one or more first-frame points only as independent surface-motion checks. A surface point is not the object center; rotation of a sphere can move point tracks independently of the center. Point visibility/re-detection helps diagnose failure, but it does not certify instance identity or mask quality.
- **Flow and camera:** SEA-RAFT/RAFT can propose dense background motion, but pairwise flow does not identify the object and can be unreliable on occlusion, motion blur, disocclusion, and plain backgrounds. Prefer labeled static landmarks for the fixed-camera gate; report flow as a corroborating diagnostic.
- **Synthetic calibration:** Kubric/MOVi provides RGB, per-pixel instance segmentation, object image positions, boxes, visibility, and camera/flow annotations. Its variants have different camera regimes, so enforce static poses or use the `fixed_random` camera when rendering the calibration set. Synthetic results calibrate coordinates, masks, and scorer but do not certify observer transfer to Wan frames. [Kubric/MOVi specification](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md)

## Near-term measurement order

1. Complete both rater passes for the saved Kubric reliability pair and the two-copy native-resolution Wan M0 packet; retain labels blinded until both are returned.
2. On Kubric, compute each rater’s center/box errors against exact projected truth, inter-rater differences, mask IoU, component/identity errors, and background-landmark displacement. Confirm the image-coordinate transforms at 160×96 and 832×480 independently.
3. On the saved Wan M0 sample, compare SAM2 and the red-threshold proposal against adjudicated human boxes, including first-frame error, p50/p90/p95 x/y errors in px and cells, identity/visibility, duplicates, and confidence/coverage. This is a single-sample proposal check only.
4. Freeze observer, prompts, error thresholds, cell conversion, frame sampling, and missing-target policy on three development seed blocks. Then run the already specified five fresh held-out blocks before allowing an observer to control future masks.
5. If the held-out label noise or observer errors are near the action size being tested, increase rater review and sample count before interpreting an apparent small delta. If camera landmarks fail, do not claim fixed-camera path gains from that sample.

**Status:** The continuous path scorer and known-truth mirrored fixture pass their existing checks. The nearest-cell diagnostic shows why full-resolution scoring is required. Human reliability and generated-video observer accuracy remain open; this memo does not claim an observer has passed or a policy improved video.

## Primary references

- [Tora: trajectory-control paper and L1 trajectory error](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Tora_Trajectory-oriented_Diffusion_Transformer_for_Video_Generation_CVPR_2025_paper.pdf).
- [Wan-Move paper](https://arxiv.org/abs/2512.08765) and [MoveBench annotations](https://huggingface.co/datasets/Ruihang/MoveBench).
- [TAP-Vid: human and synthetic point tracks, visibility, and Average Jaccard](https://arxiv.org/abs/2211.03726); [official code](https://github.com/google-deepmind/tapnet).
- [Kubric/MOVi data and annotation specification](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md).
- [CoTracker3 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Karaev_CoTracker3_Simpler_and_Better_Point_Tracking_by_Pseudo-Labelling_Real_Videos_ICCV_2025_paper.pdf); [official repository](https://github.com/facebookresearch/co-tracker).
- [SAM 3 official code and release notes](https://github.com/facebookresearch/sam3); [SAM 2 official repository](https://github.com/facebookresearch/sam2).
- [SEA-RAFT paper and code](https://github.com/princeton-vl/SEA-RAFT).
