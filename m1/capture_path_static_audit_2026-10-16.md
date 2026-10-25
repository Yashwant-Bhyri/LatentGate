# M1 capture-path static audit

**Reviewed:** 2026-10-16 (Asia/Shanghai)  
**Experiment:** `EXP-M1-COGVIDEOX5B-I2V-SMOKE-001`  
**Scope:** Original review statically checked `run_m1.py`, the local Colab notebook, `run_config.yaml`, the geometry/scale ticket, and the M1 smoke-scope amendment. At that review point, no notebook/runtime was opened, no packages or weights were installed or downloaded, and no inference or tests were run. The later packaging follow-up and its local hash reconciliation are recorded below and in the dated state entry.

## Summary

No capture-indexing or tuple-selection defect was found for the pinned Diffusers 0.40.0 CogVideoX route. The runner records six of the 50 denoising iterations at zero-based indices `[0, 9, 19, 29, 39, 49]`; callback latents are captured after the scheduler transition, transformer block summaries during the current-timestep forward pass, and predicted-clean samples from the scheduler's `pred_original_sample` output. The capture records and run summary now include machine-readable `capture_phase` labels, and block selection fails closed unless it resolves to exactly the configured six summary blocks and three spatial-map blocks.

One downstream interpretation detail remains important for feature/state alignment: the same `step_XXX` key names denote different points in a denoising iteration. At step `i`, the transformer summary describes the block output computed from the pre-transition noisy input; `predicted_clean_i` is the scheduler's x0 estimate from that same evaluation; `post_step_i` is the sample after applying the scheduler transition. Pairing records by step number is valid for an input/evaluation-to-transition analysis, but does not mean the hidden summary and post-step sample are the same latent state. At nonfinal steps the post-step sample becomes the next model input; after the final step it is the terminal latent passed to decoding.

## Findings

### Step indices and latent meanings

- `generation.num_inference_steps` is 50, so the pipeline's loop indices are `0..49`. The six configured capture indices are all valid and include the first and final transitions.
- The pipeline's `callback_on_step_end` is called after `scheduler.step` and receives the loop index `i`, the timestep `t`, and the current `latents`. Therefore the runner's callback value at index `i` is the scheduler-returned next sample for the transition applied at timestep `t`; it is not the pre-transition model input.
- The wrapper around `CogVideoXDDIMScheduler.step` maps the actual scheduler timestep back to its inference index. With the registered `v_prediction` setting, the scheduler's second return value is `pred_original_sample` (predicted x0); the runner saves this separately from the `prev_sample` used as the next denoising latent.
- The 49 decoded RGB frames and 50 denoising steps are distinct axes. The latent tensor's temporal length is a runtime/model property and should not be inferred from the requested decoded frame count.

### Transformer hook location and shapes

- The pinned Diffusers 0.40.0 transformer patch-embeds video latents and separates text tokens before iterating through `transformer_blocks`. Each `CogVideoXBlock` returns `(video_hidden_states, text_hidden_states)`. The runner's `output[0]` selection therefore captures video/image tokens, not the text branch.
- The transformer pre-hook reads the denoising timestep before the blocks run. Block hooks then summarize each selected block's output for that same timestep, before the scheduler transition.
- Let `B` be the CFG-preserved transformer batch, `L` the runtime latent-frame count, `D` the hidden width, and `G_h × G_w` the patch-token grid. The current mean-pooled summary has shape `[B, L, D]`. With spatial-map capture enabled, it first requires `patch_size_t is None` and checks the video token count equals `L × G_h × G_w`; the saved 8×12 map has shape `[B, L, 8, 12, D]`.
- At guidance scale 6.0, classifier-free guidance is enabled; both batch branches are retained without branch aggregation. The runner records the CFG batch size but intentionally does not interpret branch order.
- The config requests six summary blocks and three spatial blocks, selected evenly from the loaded transformer block count. The realized block indices and runtime model configuration are written to `model_map.json` and capture metadata; they are only known after loading the model.
- Count gate: `expected_block_indices()` now raises if the loaded transformer has fewer blocks than requested or if the resolved selection does not contain exactly the configured number of unique indices. Summary and spatial-map selection pass their separate configured counts through the same gate; `run_once()` derives expected keys from these now-exact selections. Missing or duplicate hook calls remain caught by the exactly-once coverage checks.

### Replay checks

- Each `run_once()` creates a fresh CUDA generator with seed 2026. The baseline and replay use the same pipeline object and runtime, with the same frozen config, image, and prompt.
- The comparison requires complete and exactly-once capture coverage in both runs, then applies bitwise equality (`torch.equal`) to post-step latents, pooled summaries, spatial maps, and predicted-clean tensors. Decoded RGB frames are compared pixel-for-pixel. It also requires exactly 50 transformer forwards in each pass.
- The machine gate requires 49 returned baseline frames and checks the replay has the same frame count and equal frames. Together those checks cover the requested frame count in both passes. The ticket's separate blinded review of all 49 frames remains required after the machine checks; the original eight samples are sentinel-only.
- This is a same-runtime replay check, not a cross-device or cross-version determinism claim. Equality results do not establish visual quality, geometry, or policy efficacy.

## Recommended handling

The local runner enforces the configured six summary-block and three spatial-map-block counts, preserves the resolved block indices, and labels transformer outputs, predicted x0, and post-step samples with explicit capture phases. The post-step phase label is valid for all six captured transitions, including the final transition. The pinned model is expected to satisfy the count gate; this is defensive hardening, not evidence of a model mismatch.

### Packaging review-contract synchronization

A follow-up member-by-member check found that the prior ZIP contained a stale `geometry_scale_drift_ticket.json` (member SHA-256 `1c24b788…`) while the current local ticket requires a complete ordered score table for all 49 decoded frames. The notebook correctly pinned that ZIP's binary digest, so it would have accepted the older review contract. The runner and generation settings were current; the runner does not read the ticket to control generation. The ZIP was rebuilt from the six current local inputs and all six members now byte-match their local sources. The notebook's pinned ZIP digest was updated accordingly. This fixes package/review-contract integrity without changing the registered experiment.

The separate [CogVideoX temporal-support audit](cogvideox_temporal_support_audit_2026-10-16.md) derives the 49→13→49 VAE time path and schedule-only missed-window counts. It does not transfer M0's Wan receptive-field result or change the all-frame scoring rule. The [full-sequence/frame-evidence audit](full_sequence_frame_audit_2026-10-16.md) traces the baseline/replay frame files and hash manifest and names the post-run frame-score artifact.

This audit does not clear M1's release gates: the designated-account Drive copy still needs refresh/readback, and current no-paid T4 admission must be clear before any runtime or package installation.

## Local artifact hashes

Candidate hashes after the local runner, notebook, and bundle refresh:

- `m1/run_m1.py`: `3ed1802b5cec198d94f7662ebca3e26531a50f8fc7691f9f2b7f01e73925ec0e`
- `m1/colab/LatentGate_M1_CogVideoX5B_I2V.ipynb`: `ab9d85acda9451ae2edc3a4509064cafc27c66573ef25f7e9158459e9a1e27ef`
- `m1/colab/latentgate_m1_bundle.zip`: `7f0764dd3e8730485650d1c80defc4e6024f621af03068dad18291e272daee1e`

These are local candidate hashes only; the designated-account Drive copy has not been refreshed or read back.

The refreshed ZIP contains the current ticket at SHA-256 `e7f09412381306749fa26e82915b24e148931494361007c3d8b052fb65f962ba`; the other five members also byte-match the current local runner, config, scene fixture, and two images.

## Source references

- Local capture implementation: `m1/run_m1.py`, especially `StepCapture.attach_scheduler_capture`, the transformer hooks and callback, `run_once`, and `compare_runs`.
- Local capture contract: `m1/run_config.yaml`, `m1/geometry_scale_drift_ticket.json`, and `research/2026-10-15_m1_smoke_gate_scope_amendment.md`.
- Pinned upstream implementation: [CogVideoX transformer, Diffusers v0.40.0](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/cogvideox_transformer_3d.py), [CogVideoX pipeline, Diffusers v0.40.0](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py), and [CogVideoX DDIM scheduler, Diffusers v0.40.0](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_ddim_cogvideox.py).
