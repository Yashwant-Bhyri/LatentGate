# DiTraj feedback adapter design

**Date:** 2026-10-02  
**Status:** implemented as a separate opt-in adapter and smoke runner; CPU mechanics/source-patch checks pass, but the pinned GPU runtime and any generated feedback result remain unrun.  
**Purpose:** define the smallest credible same-sample state-feedback adaptation for the 17-frame Wan 1.3B path pilot.

## Source behavior at the pinned revision

The reviewed source is DiTraj commit `6fa8e613c9e7c40ec2db456ab2ba34367a076886`.

- `module/pipe.py` reads `bbox_mask`, `mask_step`, and `fixRope_step`, then builds `cross_attention_mask` and `fixrope_self_attention_mask` once before denoising. Each denoising iteration creates a fresh `attention_kwargs` dictionary and uses the precomputed cross-attention mask when `i < mask_step`. The scheduler step runs before the optional end-of-step callback. See the pinned [pipeline source](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/pipe.py#L1613-L1725).
- `module/attention_processor.py` selects the spatial cross-attention mask only when `is_mask_step` is true. See the pinned [attention processor](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/attention_processor.py#L536-L581).
- The DiTraj transformer applies its bbox-derived RoPE modification only while `is_fixRope_step` is true. At the pilot setting `fixRope_step=3`, this means loop indices 0, 1, and 2. See the pinned [transformer source](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/transformer_wan.py#L2203-L2286).
- The pilot pins Diffusers 0.33.1. Its official UniPC source shows that `scheduler.step()` calls `convert_model_output()`, and with `predict_x0=True` plus `prediction_type="flow_prediction"`, the returned tensor is `sample - sigma * model_output` (after optional thresholding). See [UniPC v0.33.1 conversion](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_unipc_multistep.py#L3106-L3205) and its [step call site](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_unipc_multistep.py#L3769-L3815). The adapter now passes a CPU smoke against the actual 0.33.1 scheduler: at step 4 it captured the exact conversion with maximum absolute error 0.0 and the expected step index. The full DiTraj pipeline callback, GPU, and in-loop VAE integration remain untested.

This rules out a callback-only patch: the callback is after the scheduler update, and mutating the per-step `attention_kwargs` cannot update a mask that was built once outside the loop. The implemented adapter persists active boxes in the pipeline scope, arms exact x0 capture before step 4, rebuilds the spatial cross-attention mask only when the rasterized state changes, then reuses it for later eligible steps. It keeps the published open-loop runner untouched as the comparator. A local idempotent patch-shape check was run against a downloaded copy of the pinned `module/pipe.py`; actual import and scheduler behavior on the pinned GPU stack still need a smoke.

## Minimal feedback path

Use exactly one observation/update after zero-based denoising step 4 for the first causal probe. This is the fifth transformer evaluation. The existing saved M0 work confirms that the scheduler-converted `x0_hat` can be captured and decoded at that checkpoint, but only for one direction-revealing prompt and one seed. It is a feasibility signal, not observer validation.

1. Start with the same five target boxes and shared noise as the full DiTraj baseline. Keep the original `fixrope_self_attention_mask` and its first-three-step behavior fixed.
2. Wrap the pinned UniPC `convert_model_output` call during `scheduler.step` and retain its `predict_x0=true`, `flow_prediction` result for loop index 4. Do not derive `x0_hat` with a FlowMatch Euler formula. Preserve it in Wan-normalized latent coordinates until decoding.
3. Immediately after that scheduler step, inverse-normalize and VAE-decode this one 17-frame `x0_hat`. Run the preregistered red-object observer over every frame without cropping to the target box. Read centers and heuristic stability at RGB frames `[0, 4, 8, 12, 16]`, which map to the five latent-time control boxes. If the first observation is missing or unstable, fail closed and leave the whole path unchanged.
4. Use the observed first-frame center to calibrate a per-sample static offset against the locked commanded start: `observed_calibrated_i = observed_i - (observed_0 - target_center_0)`. Then apply the bounded residual to the **original target path**, not the previous corrected mask: `B_active_i = B_target_i + clip(gain * (center_target_i - observed_calibrated_i), cap)`. This makes the first probe respond to relative path error and cancels a stable color-centroid offset without moving the initial box. Preserve box extent, clamp to the image, and keep gain/cap/stability threshold fixed before the held-out run. Recomputing from the target avoids accumulating residuals across observations.
5. Rasterize the active boxes to the 5×30×52 mask and record mask IoU, changed-token count, and center shift per latent frame. Rebuild the cross-attention mask once from this active mask and let each subsequent loop iteration consume that cached value while `i < mask_step`. With `mask_step=18`, the step-4 observation can affect indices 5–17. The experiment can cap its active window at index 14 for a separate, predeclared schedule ablation.
6. Keep the prompt embeddings, negative-prompt handling, target path, hard RoPE geometry, initial noise, and scheduler state identical to the no-feedback comparator. The feedback intervention changes the spatial token-to-foreground-text assignment; it does not directly translate the object pixels.

The optional second observation after zero-based step 14 should be evaluated only after the single-read arm passes its memory and observer gates. It provides a separate update for indices 15–17 and requires a second VAE decode. Compare it against a decode/compute-matched no-op arm.

## Pseudocode boundary

This is an insertion map, not executable code:

```text
active_bbox_mask = original_bbox_mask
fixed_fixrope_mask = create_fixrope_self_attention_mask(original_bbox_mask)
captured_x0_by_step = {}

cross_mask = create_cross_attention_mask(active_bbox_mask, text_segment_mask)
for i, timestep in enumerate(timesteps):
    if i in feedback_read_steps:
        arm the UniPC conversion capture for this scheduler step

    noise = transformer(..., cross_attention_mask=cross_mask,
                        fixrope_mask=fixed_fixrope_mask)
    latents = scheduler.step(noise, timestep, latents)

    if i in feedback_read_steps:
        x0_hat = captured_x0_by_step.pop(i)
        rgb_preview = decode_wan_x0_hat(x0_hat)
        observations = observer(rgb_preview[0, 4, 8, 12, 16])
        if not trusted(observations[0]):
            continue  # no reliable start anchor: keep every box unchanged
        offset = observations[0].center - center(original_target_boxes[0])
        observations = [obs.center - offset for obs in observations]
        boxes = controller(original_target_boxes, observations,
                           lock_initial_box=True, preserve_extent=True,
                           bounded_residual=True)
        active_bbox_mask = rasterize(boxes, grid=(5, 30, 52))
        if active_bbox_mask != prior_active_bbox_mask:
            cross_mask = create_cross_attention_mask(active_bbox_mask, text_segment_mask)
            prior_active_bbox_mask = active_bbox_mask
```

The capture must be scoped to the active scheduler call and must verify the scheduler step index and `predict_x0`/prediction-type configuration. The CPU smoke against actual Diffusers 0.33.1 confirms the exact step-4 conversion; full DiTraj pipeline behavior still needs a GPU smoke. Fail closed if the expected conversion was not captured; do not silently substitute the raw model output.

## First test matrix after open-loop Gate 3

Keep one-seed results as mechanism screening only. All arms receive the same target path, noise, prompt, scheduler, and VAE decode checkpoints.

Before efficacy, validate the RGB observer on no-policy Wan previews at the exact step-4 checkpoint. Use five mirrored seed pairs (10 clips); have two raters independently and blindly label all 17 frames for visibility, identity/count, center, and box; score the five frames the controller reads. Keep these clips out of efficacy comparisons. Pass only if the expected object is uniquely visible in at least 16/17 frames per clip; the observer accepts at least 90% of human-visible controller-frame observations with no accepted wrong-object/background detections; accepted center-error median is no more than `max(1% of frame diagonal, human-human p95 center disagreement)` and p90 no more than twice that tolerance; and raw frame-0 localization against the commanded start independently meets that tolerance. Report coverage and error per clip/seed pair, not as independent frame samples. If the observer or cutoff is tuned, validate on fresh clips. The 0.65 cutoff from two Kubric paths is in-sample and cannot serve as this gate.

### Action-sign calibration before residual feedback

Do not infer the sign of the action from a changed mask. With one fixed prompt, path, seed, noise, and five-step prefix, compare the unshifted future boxes with static `+1` and `−1` token-cell horizontal offsets applied immediately after zero-based step 4. The pilot uses `mask_step=18` in a 30-step run, leaving 13 transformer iterations where the changed cross-attention mask is active; the earlier hard RoPE intervention stays fixed. Score the target against independent labels and check visibility, identity, apparent size, and background motion. The expected direction must be consistent across the mirrored path pair before introducing a live residual law. If response is absent, reversed, or paid for with scene/identity damage, stop this box-offset policy and try a spatial latent feature or Q/K state reader instead.

| Arm | Intervention | Question |
|---|---|---|
| A | Full open-loop DiTraj baseline | What path score does published spatial guidance achieve? |
| B | Decode-and-observe no-op, zero residual | Does the extra decode/observer alone change output? Require exact frame-array match with A before interpreting other comparisons. |
| C | Frozen static correction schedule, matched to D's token-shift budget | Does live state help beyond a predetermined offset? Freeze on separate calibration samples. |
| D | Correctly signed step-4 live residual | Does current-sample state feedback improve the later path? |
| E | Live residual vectors shuffled across unlocked frames | Does framewise correspondence matter? Preserve the residual distribution. |
| F | Sign-reversed live residual | Does action direction matter? |

B is the compute control for C–F; compare it with A to ensure observation and decoding alone do not change the output. C tests whether live state matters beyond adding a spatial offset. E tests temporal correspondence, and F tests directionality.

Run both mirrored paths under a neutral prompt. Keep both mirrored conditions, all arms, all frames, and all checkpoints from one seed in one group. Five paired seed groups are a directional screen; use at least ten seed blocks before a statistical improvement claim. Score final clips with blinded manual labels independent of the red observer: missing-penalized center ADE, endpoint FDE, visibility/identity, duplicates, box extent/apparent size, and background motion. Also log observer confidence and actual per-frame mask changes. A zero-token intervention is a no-op regardless of the controller's continuous residual.

## Compute and failure gates

- One extra full-resolution Wan VAE decode may increase runtime and peak T4 memory. The M0 generation's 7.91 GiB peak does not measure an in-loop decode. The staged adapter smoke runs three matched arms (baseline, decode-matched no-op, feedback) for one direction and eight denoising steps. It fails and saves an explicit gate-failure report unless the no-op output is frame-identical to baseline and the feedback arm changes at least one rasterized token; passing remains a resource/mechanism check, not an efficacy test.
- The red threshold is only suitable for the controlled red-ball fixture. The corrected known-truth CPU evaluation on 34 Kubric RGB/mask pairs gave 100% scoreable coverage and 1.73 px mean center error (about 0.00927 frame diagonals). The first-frame-anchored relative error averaged 0.0243 continuous control-grid units at the five controller keyframes; these units are not discrete token errors. Raw absolute first-frame error averaged 1.76 px and must pass independently because anchoring hides it. The 0.65 stability cutoff gives 33/34 accepted frames on these same two correlated development paths; it is an in-sample heuristic, not held-out validation or a calibrated probability. None of these results validate generated-video observation or policy gain; see [`ditraj_red_observer_kubric_calibration_v1.json`](ditraj_red_observer_kubric_calibration_v1.json).
- A corrected bbox can quantize to no change on the 52×30 grid or jump by multiple tokens. Log the rasterized intervention and include no-op/sign-reversed controls.
- If an in-loop VAE decode exceeds the resource gate, first defer the causal experiment. Later alternatives are a spatially retained internal observer (DiffTrack/DreamTraj-style QK reader) or a cheaper learned reader trained on the captured predicted-clean frames. Do not claim online feedback if observation happens only after sampling ends.
- The adapter is a separate opt-in source patch; it does not alter the baseline runner. Do not launch cloud inference until the open-loop result is reviewed, the exact Kaggle draft is reconciled, and a fresh free-T4 preflight passes.

## Evidence boundary

The implementation is in [`ditraj_feedback_adapter.py`](ditraj_feedback_adapter.py) and [`run_ditraj_feedback_pilot.py`](run_ditraj_feedback_pilot.py). It captures UniPC's exact converted x0 estimate at the selected step, verifies its sigma index and scheduler configuration, decodes with Wan inverse normalization, observes all 17 full RGB frames, anchors the observer to the fixed start, updates the five path boxes with a bounded one-token residual, and refreshes the cross-attention mask only if its rasterized state changes. Thirteen CPU unit checks pass across the policy and adapter, and a separate CPU smoke against actual Diffusers 0.33.1 UniPC captures the exact step-4 conversion with max absolute difference 0.0. The source patch was also checked for idempotence and compilation against the pinned pipe file. No DiTraj pipeline generation, GPU capture, or policy intervention has run. Any generated-video result must be judged with independent blinded labels; a successful code path alone is not evidence of improved control. See [`smoke_pinned_unipc_capture.py`](smoke_pinned_unipc_capture.py) and [`ditraj_unipc_capture_smoke_v1.json`](ditraj_unipc_capture_smoke_v1.json).
