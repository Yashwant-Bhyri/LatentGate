# DiTraj background-stability protocol decision

**Date:** 2026-10-17 07:04 UTC  
**Status:** source-only method decision for the future six-output/102-frame packet. No packet, labels, scorer, or threshold exists. This does not advance Gate 1.

## Decision

Treat “background stability” as two distinct image-space signals measured on a fixed, target-excluded region:

1. **Geometric/background motion:** sparse Shi–Tomasi features tracked with pyramidal Lucas–Kanade on each adjacent pair, forward then backward. Report the signed median Δx/Δy and median/p95 displacement norm in **pixels per frame pair**, valid-track count and fraction, and spatial coverage. Do not estimate or remove a global transform, stabilize, warp, or subtract the measured motion: doing so could erase the fixed-camera failure this guardrail is meant to expose.
2. **Appearance/illumination change:** on the same region, report raw adjacent-pair and frame-0 RGB absolute difference (per-channel mean/median/p95, in 8-bit DN and divided by 255) plus the per-frame median luma change in DN. Define display-coded luma as `Y' = 0.2126 R' + 0.7152 G' + 0.0722 B'`, where `R'`, `G'`, and `B'` are the decoded 8-bit channel values. Keep the unnormalized values primary. A robust gain/offset fit, if later useful to diagnose exposure changes, is a separately labeled diagnostic and cannot replace the raw signal.

These are image-space scene-stability indicators, not proof of physical camera motion. Keep target path, visibility/count/identity, and apparent box size in their existing separate reports. If duplicate or unannotated moving content occupies the retained region, flag attribution as ambiguous; do not label its measured flow “camera motion.”

## Frozen inputs and masking rule

- Future input is the contract-bound set of six blind IDs × 17 frames, original 832×480 decoded RGB PNG pixels and immutable SHA-256/dimension manifest. Analyze all 16 adjacent pairs per clip and the frame-0 comparison. Use the same exported artifacts for every metric; no resizing, denoising, stabilization, color correction, or lossy re-encode after the packet images are made.
- If an upstream artifact is a video, record its container, video codec/profile, pixel format/chroma subsampling, frame rate/time base, encoder settings when available, and decoder/version. Decode once with one pinned decoder to the canonical PNGs, then hash those files. If arms can take different encoding paths, the evaluation is confounded and must stop pending a matched export path. PNG compression level does not alter decoded pixels; it is not a visual confound.
- After the two independent blind box returns are complete and frozen, form each clip’s exclusion mask from the union of every valid target box supplied by either rater across all 17 frames. Apply one fixed mask to every frame in that clip. Add a target/shadow halo only after estimating and freezing its radius on a separate calibration set; never tune it on the six evaluation clips. Record excluded and retained pixel fractions. Report background metrics as unscoreable if the calibration or annotations leave inadequate/ambiguous support; the minimum usable-area rule must be set during calibration, not invented here.
- A valid box can be included in this conservative exclusion union even when its identity/visibility label makes that frame ineligible for path geometry. Keep the raw rater rows and disagreement visible; do not form consensus boxes or read the arm key. If a rater has no usable target box for a frame, record that fact and the resulting mask limitation rather than filling a box.

## Measurement details and diagnostics

- For every adjacent pair, detect corners only inside the fixed exclusion mask on the first image; track them forward and backward with the same pinned pyramidal LK configuration. Accept/reject criteria (corner quality, border rule, LK status/error, forward-backward residual) are parameters to calibrate and freeze on separate data. Report counts at every stage and accepted-track fraction; a low displacement with near-zero coverage is not a pass.
- Report per-pair vector median Δx, Δy, median and p95 ∥Δp∥, in source pixels per frame pair. Also report tilewise coverage and tilewise medians/p95 so a local change cannot disappear inside one global median. Keep the 16 transition rows; summarize per clip, then show all six clip summaries. The six clips, not 96 adjacent pairs, are the independent experimental units.
- On exactly the same fixed ROI, report raw RGB difference between adjacent frames and from frame 0, with channel-wise mean/median/p95; report luma distribution shift separately. Large raw changes may be exposure, compression, scene change, or misregistration; flow, coverage, and calibration provide attribution clues, not an automatic correction.
- A secondary dense-flow sensitivity analysis may use the project’s existing Farnebäck/RAFT diagnostics, but it must share the same ROI and source pixels and remain a distinct report. VBench can be recorded as an external comparator only: its published/code background score compares whole-frame CLIP features to the previous and first frames, and its temporal-flicker implementation is whole-frame MAE explicitly intended for static videos. Neither localizes the moving target or separates object motion from background, so neither is the primary guardrail here.

## Calibration, blinding, and threshold rule

Before the 102-frame packet is created, use a **separate non-evaluation calibration set** processed through the same image-export/decode path. Include repeated static scenes, target motion with its moving shadow, illumination/exposure changes, codec-only changes if any lossy video path is used, and known small image-plane camera translations. Use known scene/camera conditions to estimate tracker coverage, displacement noise, illumination sensitivity, compression sensitivity, and halo/ROI usable-area behavior. Freeze ROI halo, corner/flow configuration, color conversion, aggregation, and any acceptance threshold before touching the six evaluation outputs. If calibration cannot distinguish static/illumination/codec variation from the camera-motion cases at useful coverage, keep this as descriptive evidence and do not issue a background pass/fail.

Keep all six aliases blind through measurement. Annotators see neutral IDs only; the measurement operator receives the hash-bound frames and blinded annotations, never arm/path/seed labels. Freeze the protocol/config hash and write per-blind-ID outputs before a separate custodian unblinds. Do not inspect private labels or any alias key in this source-only stage.

**No numeric background pass threshold is proposed.** The current pilot’s “no material worsening” wording is not made numerically testable by this memo. The threshold, usable-area floor, and calibration-derived uncertainty rule remain open until an independent calibration study; register them in a versioned preregistration amendment before packet creation, without changing the Gate 1 reliability thresholds or inventing them from six outputs.

## Evidence, uncertainty, reuse, and compute

- VBench (Huang et al., CVPR 2024) motivates a separate background dimension and human-aligned evaluation suite. The primary paper and official implementation use global CLIP image features for background consistency. The official `temporal_flickering.py` computes full-frame RGB MAE and documents a static-video assumption. This is useful comparative context, but object motion makes the full-frame metrics confounded for this fixed-camera task.
- Lucas–Kanade’s local brightness/patch assumptions and pyramidal approach are described by Bouguet’s tracker note; the official OpenCV optical-flow tutorial documents Shi–Tomasi corners, `calcOpticalFlowPyrLK`, status outputs, and recommends a backward check for robust tracking. These support the proposed inexpensive geometric signal, while illumination sensitivity and textureless/reflective backgrounds remain calibration risks.
- Official repositories: [VBench](https://github.com/Vchitect/VBench) reports a top-level Apache-2.0 license; [OpenCV 4.x](https://github.com/opencv/opencv/blob/4.x/LICENSE) reports Apache-2.0. No source code, model weights, dataset, or external artifacts were copied or run in this review. Recheck component/dependency licenses if code is later reused.
- Compute: source/repository review only; no GPU, model, dataset, test, fixture, scorer, or external runtime. The eventual metrics are CPU vision/image operations; runtime must be measured with the actual packet and device if needed.
- Limitations: 17 frames provide only 16 transitions per clip; sparse flow can fail on smooth, reflective, or photometrically changing surfaces. Background RGB differences are sensitive to both geometric misalignment and illumination; global flow does not identify the cause. Calibration and human ROI annotation are prerequisites, not evidence already collected.

## Falsifiable next step and blocker

Once the existing Gate 1 and no-paid/full-output prerequisites pass, calibrate this protocol on separate controls and freeze its ROI/flow settings and threshold before creating the six-clip packet. It is supported only if known fixed-background cases retain adequate coverage and known camera translations produce a separable displacement response under the same illumination/codec conditions. Otherwise report the measurements descriptively and do not call a clip stable/unstable by threshold.

**Current Gate 1 blocker is unchanged:** two independent complete Gate 1 v5 human returns are still missing (v16: zero complete and zero partial). No later output packet or background score is authorized by this protocol note.

## Sources

- Huang et al., [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982), CVPR 2024.
- Official VBench implementation: [`background_consistency.py`](https://github.com/Vchitect/VBench/blob/master/vbench/background_consistency.py), [`temporal_flickering.py`](https://github.com/Vchitect/VBench/blob/master/vbench/temporal_flickering.py), [repository license](https://github.com/Vchitect/VBench/blob/master/LICENSE).
- Bouguet, [Pyramidal Implementation of the Lucas Kanade Feature Tracker: Description of the Algorithm](https://web.stanford.edu/class/cs231m/references/pyr-lucas-kanade-feature-tracker-bouget.pdf).
- Official [OpenCV optical-flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) and [OpenCV 4.x license](https://github.com/opencv/opencv/blob/4.x/LICENSE).
