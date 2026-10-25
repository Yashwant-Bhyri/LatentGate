# DiTraj action path and Gate 1 measurement mapping

**Checked:** 2026-10-16 07:04 UTC / 15:04 Asia/Shanghai  
**Scope:** Read-only source audit of the pinned DiTraj open-loop path intervention, the local opt-in x0-hat adapter, and their measurement interface with Gate 1. No notebook cells, tests, model/weight downloads, inference, GPU, Colab, or Kaggle runtime were used. No human annotation values, private unblind key, or alias-to-direction mapping were inspected.

## Finding

The pinned DiTraj code is an **open-loop image-space box-conditioning method**. It consumes a user-specified normalized bounding-box path, separates foreground/background text guidance, and changes foreground spatial-temporal rotary position encodings during early denoising. It does not observe the generated object while the open-loop run is denoising, and it does not guarantee that a visible object instance follows the commanded path. The local x0-hat adapter adds a separate, unrun feedback route; its red-threshold observer does not establish visibility or identity.

The earliest release blocker remains human-label reliability. Scan v7 at `2026-10-16T06:48:03Z` covered 7,233 CSV/ZIP paths and found 24 exact-v5-header candidates: two blank canonical packets, 20 blank or packet-map-mismatched candidates, two known synthetic/UI QA exports, and **zero complete or partial human returns**. The scan emitted no annotation values, aliases, direction map, or private key. Two independent complete v5 submissions are still required before the registered agreement, blind worst-frame, and truth gates.

## Pinned intervention path

The source pin is DiTraj commit `6fa8e613c9e7c40ec2db456ab2ba34367a076886`. The paper describes foreground/background separation and inter-frame STD-RoPE as the training-free control components; the official README documents normalized box paths as `[frame_id, y1, y2, x1, x2]` and marks the repository CC BY-NC 4.0. This audit concerns the code path, not a reproduction of the paper's reported results.

1. `run.py` parses path anchors and delegates interpolation to `utils.plan_path`. The local pilot defines two mirrored paths with endpoints at decoded frames 0 and 16. For the 17-frame sequence, linear interpolation produces 17 normalized boxes; sampling `[::4]` selects frames `[0, 4, 8, 12, 16]` for the five Wan temporal-latent positions.
2. The local pilot rasterizes each sampled `[y1,y2,x1,x2]` rectangle into a binary mask with shape `[5,1,30,52]` for 832×480 output by dividing spatial dimensions by 16. Temporal sampling follows the local runner's 17-frame-to-five-mask convention; the separate M0 audit covers Wan VAE temporal support. Box starts and ends are converted with `int(normalized_coordinate * grid_size)` and used as `[start:end]` slices; right and bottom are therefore exclusive. The mask is a control region, not a segmentation of the object actually generated.
3. The pipeline encodes a base prompt plus foreground and background prompts. During the first 18 denoising calls in the local DiTraj arm, its cross-attention mask routes foreground grid queries to the foreground prompt tokens and background queries to the background prompt tokens. After that cutoff, it uses the base prompt without the foreground/background mask.
4. During the first three calls in the local full-DiTraj arm, the transformer modifies the rotary embeddings inside each frame's foreground box by copying/resampling the smallest-area box's spatial RoPE region. This changes positional encoding for the guided spatial cells; it does not track an object, estimate scene geometry, constrain depth or pose, or impose contact/rigid-body physics.

The local arm table is `base` (`mask_step=0`, `fixRope_step=0`), `sg_only` (`18`, `0`), and `ditraj` (`18`, `3`), each on both mirrored paths. The runner gives arms the same seed, prompt, model revision, and saved common initial latent. That is the planned six-case one-step smoke/full-run design; no DiTraj generation has run. The local zero-cutoff and CPU-corner-lookup source adaptations are separate from the upstream method and have not been exercised on the pinned stack.

The official paper's mechanism is open-loop. The separate local `RedPathFeedback` design captures UniPC's converted x0 estimate at zero-based step 4, decodes the 17-frame prediction, observes all red pixels with a threshold ensemble, and consumes observations at frames `[0,4,8,12,16]`. It locks the first box, subtracts a trusted start offset, then applies a bounded residual update (gain 0.5, at most one 52×30 grid cell per axis) to later boxes whose observer stability passes 0.65. The code labels that value an in-sample heuristic, not calibrated confidence. The patched pipe refreshes the later cross-attention mask after a changed box; the STD-RoPE interval has already ended at step 3, so this feedback update does not revise that early RoPE intervention. This adapter and its smoke runner remain unrun and are not evidence of policy efficacy.

## Mapping generated frames to the existing measures

The paired-path runner saves each decoded output as `frame_000.png` through `frame_016.png`, preserves the 832×480 pixels, and writes an annotation template with `frame_index` 0–16 and an `image_file` for each frame. For a generated DiTraj result, the exact frame index and image filename should remain bound together. Path error should use the annotated box center `(x1+x2)/2,(y1+y2)/2` on **all 17 decoded frames** against the target box-center path interpolated at those same frame indices. Scoring only the five latent-time anchors would miss deviations between anchors.

Coordinates must stay in source-frame pixels until explicitly transformed. The local paired-path output is already 832×480, matching the 832×480 target frame and 52×30 control grid (16 target pixels per grid cell on each axis). For the 160×96 Kubric reliability frames, Gate 1 maps each axis independently: `dx_cells = dx_source_px * 52/160`, `dy_cells = dy_source_px * 30/96`. The scorer uses box centers and half-open pixel-edge boxes. Do not use an isotropic scale or confuse the 30-row × 52-column grid order with x/y order.

The human-label constructs are related but not interchangeable:

| Question | Local DiTraj output template | Gate 1 v5 scorer | Interpretation |
|---|---|---|---|
| Visibility | `object_visible` | `object_visible` | A per-frame human judgment; report coverage/missing frames separately from path error. |
| Count | `instance_count` | `instance_count` | Keep as its own categorical measure. |
| Identity | `same_identity_as_frame_0` | `identity_consistent` | Closely related semantics, but the schema names differ. Any adapter must make this conversion explicit and preserve the original answer; do not infer identity from red color or box proximity. |
| Box | `bbox_x1_px` … `bbox_y2_px` | same four edge fields | Tight visible-object box in source pixels, not the DiTraj target-control rectangle. |
| Frame provenance | `frame_index`, `image_file` | exact packet-bound `(blind_clip_id, frame_index) → image_file` map | Preserve the precise map; the current v5 reference map is for its own packets. |
| Extra fields | visible/occlusion reason, center, confidence, blur/shape note | not required by the current agreement scorer | These do not replace categorical or box fields. |

The DiTraj template has six output clips × 17 rows and a custom image map; the current v5 Kubric reliability packet has two clips × 17 rows (34 rows), with a different archived image map. Thus the custom DiTraj CSV cannot be submitted directly to the current packet-bound v5 agreement/truth route. A later output study needs its own blind packet/reference map and explicit schema adapter. The current known-truth scorer compares v5 human boxes with Kubric mask boxes and projected centers; that validates annotation accuracy on the fixture, not DiTraj output quality. The separate v5 generated-sample packet concerns the saved unsteered Wan sample, not a DiTraj intervention.

The existing agreement scorer computes inter-rater center disagreement only when both labels say visible, count exactly one, and identity-consistent; its coverage is reported separately. The known-truth scorer likewise marks geometry scoreable only when both the annotation and fixture truth identify one visible, identity-consistent target. Apply the same eligibility rule when deriving generated-output path error, and do not replace ineligible frames with an invented box. The local runner's red-pixel proxy is explicitly secondary: it reports visible-pixel coverage, normalized box-center error, box IoU, red-mask area/box-size variation, and direction-aligned adjacent visible steps, but a red mask can merge objects or highlights and is not a semantic identity label.

Keep four outcome groups separate:

- **Path adherence:** signed and radial center error across visible, identity-consistent frames; include endpoint direction and all-frame path error.
- **Visibility/identity:** `object_visible`, `instance_count`, and `identity_consistent` agreement/coverage. The red threshold proxy groups all matching red pixels and cannot establish instance identity or count.
- **Apparent size and shape:** actual annotated box width, height, area, and their temporal changes. A prescribed DiTraj box is a guidance mask and must not be treated as the object's true extent.
- **Background/scene stability:** a separate background-region or blinded scene judgment. Center-path success says nothing about background drift or quality.

## Decision and next prerequisites

Keep DiTraj as the planned first open-loop Wan-compatible actuator candidate, with `base` as the no-intervention reference and `sg_only` as a component ablation. Treat the x0-hat adapter as a separate later mechanism. Neither component has generated a DiTraj video or measured a gain. The six-case pilot can screen whether the fixed path-conditioning code responds in mirrored directions under common noise; it cannot by itself establish semantic identity, reliable online observation, physical motion, or policy efficacy.

Next steps, in order:

1. Receive two independent complete v5 human-label submissions and pass the frozen packet-map, pooled/per-clip agreement, blind maximum-frame review, and guarded Kubric truth checks. The v7 scan shows none are present.
2. Only after that gate and fresh verified free-T4 admission, run the already-configured one-step open-loop six-case smoke; retain all 17 PNGs and record device, precision, elapsed time, peak VRAM, source hashes, and package versions.
3. Before scoring output quality, build a blinded per-output reference packet for the six generated clips; preserve exact image keys and explicitly adapt `same_identity_as_frame_0` to the Gate 1 identity field. Use human labels for visibility, count, identity, and boxes; report path, size, and background separately.
4. Do not enable state feedback until the generated-frame observer is independently validated and the open-loop action sign/sensitivity is identified. The efficacy comparison then needs the registered matched open-loop, decode-matched no-op, static, live-feedback, shuffled-residual, and sign-reversed arms on held-out seeds/paths.

## Local evidence and reproducibility

- Paired-path runner: `research/run_ditraj_paired_path_pilot.py`, SHA-256 `312d923f04af24a909082ca659a0d743e2066e2f970c71298eb971c594bc5a4e`.
- Feedback adapter: `research/ditraj_feedback_adapter.py`, SHA-256 `08fd6dd3d1b8acb11afec4432638211922da7f0e63ef745c96c1d7e3c4793d78`.
- Residual controller: `research/box_residual_feedback.py`, SHA-256 `c459c58104f04db8e31d09009daea3ec5ccba589f4d577d3c916edb556fafbb7`.
- Gate 1 agreement scorer: `research/score_annotation_agreement.py`, SHA-256 `9a3bc5289ef7e89a88b26f20ffc8770b415044ef35cb8ced5ee590597fe8aaac`.
- Guarded known-truth scorer: `research/score_gate1_labels_against_kubric_truth.py`, SHA-256 `e170665880063bc5725c41f5316ac93275f360079ca0e4dc82ff21ffc5864957`.
- Latest packet-bound intake: `research/gate1_label_intake_scan_v7.json`, SHA-256 `bc87aab892b6498149c83daaa845fe823e8411587fc3e2d08bb960587a1993c7`.
- Local v5 packet instructions: `research/gate1_manual_annotation_packets_v5/README.md`, SHA-256 `abec46899f8b7d40b5a5d3bd16d6cf0723b08ffafe5b46fd8b223861f8df0b1b`.
- Primary paper: [DiTraj arXiv v2](https://arxiv.org/abs/2509.21839).
- Pinned official source: [README](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/README.md), [run.py](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/run.py), [utils.py](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/utils.py), [pipeline](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/pipe.py), [Wan transformer](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/transformer_wan.py), and [attention processor](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/attention_processor.py).

The code was inspected statically from commit-specific primary sources and local files; no upstream checkout or weights were downloaded for this audit. No tests or experiments were run. Shared `STATE.md` and the experiment registry were left unchanged for the coordinator.
