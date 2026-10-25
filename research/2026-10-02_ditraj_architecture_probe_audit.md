# DiTraj architecture and signed-actuation probe audit

**Checked:** 2026-10-02  
**Scope:** pinned Wan2.1 T2V 1.3B + DiTraj pilot; deterministic signed path actuation; later same-sample latent feedback; execution and license feasibility.  
**Evidence class:** source inspection and local artifact inspection only. No model, GPU, or cloud job was run for this memo.

## Recommendation

Keep Wan2.1 T2V 1.3B + DiTraj for the first control experiment because it is the local pilot’s path-conditioned stack and it exposes a box-mask control surface. Before enabling the observer or training a policy, run a deterministic signed-actuation triplet: unchanged path, future boxes shifted one latent cell right, future boxes shifted one latent cell left. Apply the shift only after zero-based step 4, after the shared five-step prefix. This isolates the cross-attention actuator while preserving the baseline’s first three fixed-RoPE steps and the initial state.

This is the safest first probe because it requires no VAE preview, detector, gradient, or online action decision. It tests the critical unproven premise: whether a late one-cell change in DiTraj’s foreground/background routing produces a visible object displacement with the expected sign. A changed mask is not evidence of actuation. If no repeatable signed response appears, pause this controller design and compare a direct frame-loss actuator such as Frame Guidance.

Only after the signed response and observer reliability gates pass should the team run the current x0-hat feedback adapter. The longer-term native architecture should read spatially retained Wan features, estimate object state and confidence, and make bounded future-path mask updates inside the denoising loop. Keep the Wan generator frozen at first; train a small spatial state reader and controller only after the action-response map is measured.

## What is pinned locally

The current local [paired runner](run_ditraj_paired_path_pilot.py) pins:

| Component | Pinned value |
|---|---|
| DiTraj | Commit `6fa8e613c9e7c40ec2db456ab2ba34367a076886` |
| Wan Diffusers checkpoint | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision `0fad780a534b6463e45facd96134c9f345acfa5b` |
| Diffusers | `0.33.1`, matching DiTraj’s setup instructions |
| Video | 17 RGB frames, 832×480; five latent-time control boxes on a 52×30 spatial grid |
| Sampling | 30 steps, flow-prediction UniPC, flow shift 3, CFG 5, seed 2026 |
| DiTraj windows | `mask_step=18`, `fixRope_step=3` |
| T4 loading | FP16 transformer, FP32 VAE, sequential model CPU offload |

DiTraj’s README explicitly requires Diffusers v0.33.1 and replacing Diffusers’ Wan transformer with the project’s own transformer file. The local runner additionally applies a vectorized corner lookup and skips unused zero-window masks; it hashes patched source. Therefore this is not a plain `from_pretrained` Diffusers experiment. Use an isolated pinned environment and record the source hashes. [Pinned DiTraj setup and replacement instructions](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/README.md#L224-L243)

The existing M0 generation is a useful timing and preview precedent, but it is not the DiTraj baseline: its saved manifest identifies the same Wan checkpoint revision at 17 frames / 832×480 / 30 steps, while using Diffusers `0.40.0`, Torch `2.11.0+cu128`, Python 3.13, and a plain prompt-only Wan pipeline. It completed in 160.02 seconds on a Tesla T4 and reached 8,495,847,424 allocated bytes (7.91 GiB). It did not use DiTraj or its older pinned Diffusers stack. The step-4 x0 preview and this timing must not be described as a DiTraj feedback run.

The local [UniPC capture smoke artifact](ditraj_unipc_capture_smoke_v1.json) is CPU-only. It verifies exact Diffusers 0.33.1 `convert_model_output` capture at step index 4 with maximum absolute difference 0.0. The local [feedback runner](run_ditraj_feedback_pilot.py) and [adapter](ditraj_feedback_adapter.py) are source/CPU checked; neither has been executed through a DiTraj GPU pipeline. The runner’s default eight-step smoke is plumbing/resource validation, not an efficacy benchmark.

## Why the action has to be applied in the pipeline loop

At the pinned revision, `module/pipe.py` reads `bbox_mask`, then precomputes both the cross-attention mask and fixed-RoPE self-attention mask before denoising. The loop creates a fresh `attention_kwargs` dictionary each iteration. While `i < mask_step`, it passes the cached cross-attention mask; after that, it disables the mask. It applies the bbox-derived RoPE rewrite only while `i < fixRope_step`. CFG then runs an unconditional pass, the scheduler advances the latents, and the public callback runs afterward. [Pinned DiTraj mask setup and loop](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/pipe.py#L1613-L1725)

The attention processor consumes the cached spatial cross-attention mask only when `is_mask_step` is true. It uses the mask to route each video token to foreground or background text segments; it is not a direct pixel translation operation. The modified transformer consumes `bbox_mask` for RoPE only during the early `is_fixRope_step` calls. [Pinned attention processor](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/attention_processor.py#L536-L581) · [Pinned transformer RoPE branch](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/transformer_wan.py#L2196-L2286)

Consequences for the pilot:

- Merely changing `bbox_mask` in Diffusers’ ordinary end-of-step callback is not sufficient. The mask used by the next transformer evaluation was built earlier and stored in pipeline-scope variables; `attention_kwargs` is recreated each iteration.
- The current local adapter adds a pipeline-scope hook. It calls `before_scheduler_step`, runs the unchanged scheduler update, then calls `after_scheduler_step`. If a new mask is returned, the loop replaces `bbox_mask` and rebuilds the cached cross-attention mask. This is the correct narrow seam for the intervention.
- Keep the precomputed fixed-RoPE mask unchanged. In the proposed action probe, all boxes and RoPE behavior remain identical through step 4; only the cross-attention mask changes for later calls.
- With 30 steps and `mask_step=18`, an update after step 4 can affect loop indices 5–17: 13 subsequent conditional transformer passes. The unconditional CFG pass has its cross-attention mask disabled. An eight-step smoke has only indices 5–7, or three active conditional passes.

For the first triplet, implement a simple deterministic controller behind the existing optional post-step hook: at step 4, offset latent boxes 1–4 by exactly `−1`, `0`, or `+1` horizontal grid cell; keep box 0 fixed; preserve width/height; reject any box that would clip at the image boundary; rasterize with the same integer/floor convention as the baseline. Do not instantiate the x0 capture or observer in this triplet. The three arms must share the same prompt, original target path, initial latent tensor, scheduler configuration/history through step 4, and checkpoint. Log and compare the post-step-4 latent hashes (and scheduler index/history); if the prefix diverges, branch from one saved latent plus its full UniPC state or treat the run as noncausal. Keep the first three RoPE interventions identical in every arm. Later repeat with the mirrored target direction and paired seeds.

Read out the visible object from final frames using independent labels. The directional contrast is whether `x_center(+1) − x_center(−1)` is positive on the commanded axis across the unlocked future frames, not whether the mask pixels changed. Also report path error, visibility/missing frames, identity/duplicates, apparent extent, and fixed-background motion. This triplet is a mechanism screen; use the existing preregistered paired-seed plan before treating the sign as calibrated.

## Current feedback adapter and the next API boundary

The local [feedback adapter design](ditraj_feedback_adapter_design_v1.md) adds more than a callback:

1. It arms a scoped wrapper around pinned UniPC `convert_model_output` during the scheduler step at loop index 4. It requires `UniPCMultistepScheduler`, `prediction_type="flow_prediction"`, `predict_x0=True`, and `scheduler.step_index == 4`, then clones the converted clean estimate. The conversion is taken from the pinned scheduler implementation rather than recomputed using an assumed flow formula. [Diffusers v0.33.1 UniPC conversion](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_unipc_multistep.py#L3106-L3205)
2. After the scheduler update, it inverse-normalizes the latent estimate and decodes it with the Wan VAE. It observes all 17 decoded frames using a nine-threshold red-pixel centroid ensemble, reads indices `[0,4,8,12,16]`, and calibrates relative position to the locked starting box.
3. It applies a bounded residual to the original target boxes, not recursively to the prior corrected boxes. An unscoreable/untrusted anchor fails closed. Rasterized changes are logged at the 5×30×52 path grid, and the cached cross-attention mask is rebuilt only when the rasterized mask changes.

The red observer is specific to the bright-red-ball fixture. Its threshold agreement is a heuristic, not a calibrated probability. The `0.65` development cutoff is in-sample on correlated synthetic paths; it is not ready to gate generated clips. Independent label validation must precede live actuation. The exact output identity of the decode-matched no-op arm versus baseline is a required invariant, not an assumption.

The smallest maintainable pipeline API should make that control seam explicit instead of relying long term on brittle string substitution in `module/pipe.py`:

```text
PathController.before_scheduler_step(scheduler, step, timestep, sample)
PathController.after_scheduler_step(pipe, step, timestep, sample, bbox_mask)
    -> UpdatedPathMask | None
```

The loop owns the persistent mask and segment mask. It should rebuild the cross-attention mask only after a discrete mask change, leave fixed-RoPE state immutable after its first three steps, and publish a small telemetry object containing step index, checkpoint ID, actual rasterized action, observer state, confidence/scoreability, and fail-closed reason. Keep the scheduler model output and history untouched. For a state-based controller, capture either exact x0-hat or spatial latent features at a predeclared step; do not leak the target path into the observer.

For the eventual native policy, use a staged frozen-backbone path:

1. Calibrate the discrete mask-to-object response and validate an observer against blind human labels.
2. Save spatially indexed Wan hidden states at the observation step and selected layers. First test whether a linear per-frame feature probe can estimate center, extent, visibility, and identity on held-out seeds. Do not use pooled-only features if they erase spatial location.
3. Use the validated decoded observer as a teacher for a small spatial/temporal state reader. Keep the residual controller’s gain and shift cap fixed while evaluating the reader.
4. Feed the reader’s bounded state residual into the same post-step future-mask seam. Compare it against zero action, a fixed correction, shuffled per-frame residuals, and sign reversal. Evaluate held-out paths and seeds with independent labels.
5. Only if the controller shows robust gains should a LoRA or generator-side learned module be considered. At that stage, protect visual quality, object identity, visibility, extent, and background stability with explicit guard losses and held-out checks.

This makes “native” measurable: the state reader and controller operate on the active sample inside the denoising loop, and the returned mask changes subsequent transformer evaluations. A post-hoc edit or a detector overlay does not satisfy that criterion.

## Challenge to the current stack: Frame Guidance

The best direct alternative to DiTraj’s box-routing action is [Frame Guidance](https://arxiv.org/abs/2506.07177), an ICLR 2026 method for training-free frame-level controls using signals such as keyframes, sketches, color blocks, and depth. Its central alternative is closer to “optimize the sampled latent against a target-frame objective” than “move the object by changing foreground-token routing.” That is a stronger causal link to frame position and is the right comparator if ±1-cell DiTraj actions have no reliable visible response.

It is not the safer first run on this local stack. The official repository lists Wan-I2V notebooks, notes that its Wan2.1 version will be updated, and warns to pin `transformers==4.57.3` for a Wan loading issue with 5.0.0. The current local target is T2V 1.3B and a pinned Diffusers 0.33.1 DiTraj transformer, so Frame Guidance needs an explicit compatibility port and a task-specific differentiable object-center/mask loss. The paper describes memory-reduction methods because naive full-frame gradient guidance is expensive; its actual peak memory and speed on a T4 at our short resolution are unknown. No top-level code license was visible in the reviewed Frame Guidance repository, so do not vendor its implementation until the exact revision’s terms are checked. [Official implementation and compatibility notes](https://github.com/agwmon/frame-guidance)

For the controlled red-ball case, a differentiable soft-red spatial moment could be explored as a loss, with explicit penalties for object area and visibility. That would be a new adaptation, not a reproduction of the paper’s exact loss. It has a benefit—directly penalizing center error—and costs: gradient-enabled decoding, backpropagation through VAE and possibly the denoiser, increased memory/runtime, and potential appearance or motion damage. Use it as a matched same-backbone alternate after the simpler DiTraj action sign test.

The stack choice therefore remains a *test order*, not a claim that DiTraj is the better actuator. DiTraj is first because the existing 1.3B T2V path and local hook make it the most controlled minimal experiment. Frame Guidance is the stronger alternate if the mask’s causal response is absent or too weak.

## Execution environments and cost bounds

### Local Mac

The inspected machine is a MacBook Pro `Mac15,3`: Apple M3, 10-core GPU, 16 GB unified memory. Python has PyTorch 2.9.1 with MPS available, no CUDA, and no installed Diffusers package. The local runner explicitly refuses to proceed without CUDA and identifies T4 as its required device. It also calls `torch.cuda` memory APIs and installs a custom Wan transformer into Diffusers. So the exact DiTraj pilot is not runnable unchanged on this Mac.

The pinned DiTraj mask code also explicitly moves `overlap_token` to the string device `'cuda'`; that must become device-relative before MPS can work. The Mac remains useful for labels, rendering/evaluation, source inspection, CPU diagnostics, and small observer probes. An MPS port would need a separate dependency environment, replacement of the hard-coded CUDA move, removal of T4/CUDA guards and memory-meter calls, validation of CPU-offload behavior, and a check that the pinned custom transformer’s indexing/masks run correctly on MPS. No local DiTraj runtime or speed estimate has been measured. Marginal cloud spend would be $0, but electricity, install time, unified-memory pressure, and porting cost are nonzero; do not budget this as a validated generation route.

### T4 / Kaggle / Colab

The T4 is the practical first-generation target. The existing plain-Wan M0 sample used a Tesla T4 with 15.64 GB reported device memory and peaked at 7.91 GiB for one 17-frame, 832×480, 30-step generation. It took 160.02 seconds under Diffusers 0.40.0. As a rough lower-bound planning anchor only, three comparable static-action arms would be about 480 seconds (8 minutes, or 0.133 GPU-hours) before setup and save overhead. It is not a DiTraj benchmark: DiTraj’s older runtime, source patch, dense masks, CPU offload, and online VAE decode may change both time and peak memory. There is no measured dollar quote until the selected provider’s live GPU rate and session allowance are known.

At this pilot resolution, the dense fixed-RoPE mask is already material: the 5×30×52 token count is 7,800, whose square is about 60.8 million boolean entries (roughly 58 MiB before temporary tensors). The source’s three-way equality intermediate is about 183 MB. The same path at 81 RGB frames would have 32,760 tokens and a squared mask above 1 GiB, with a multi-gigabyte intermediate. This quadratic dependence is why the short 17-frame setup is the correct feasibility screen. Measure CUDA peak allocated/reserved memory and host RAM on the first actual smoke; do not infer DiTraj headroom from the plain-Wan M0 peak.

Kaggle’s documentation lists P100 and T4 configurations and warns that free GPU access is limited and may queue; a two-T4 configuration does not combine VRAM, and this runner uses only `cuda:0`. A fresh attached session and actual device name/memory must be confirmed before running. [Kaggle notebook docs](https://www.kaggle.com/docs/notebooks)

Colab’s current FAQ says eligible paid Google AI plans receive monthly Colab compute units and premium GPU/TPU access, but the exact allotment depends on plan and availability; it directs users to Settings → Subscription for the live balance. Earlier account inspection in this project showed 0 compute units and a free-T4 usage denial. Recheck before any run. If the monthly units or free GPU are available, direct cash cost may be zero; otherwise calculate paid cost from the provider’s current rate and the measured runtime. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html)

No GPU/cloud run was started for this audit. For the initial triplet, first run a short fail-fast T4 smoke that logs memory and validates exact source hashes, then run the 30-step action arms only if it fits. The online feedback arm adds one full decoded x0 preview and observer pass; that incremental T4 cost is unknown until measured. Avoid a larger sample sweep until no-op identity, memory margin, observer reliability, and signed response are all reviewed.

## License and reuse notes

- **DiTraj code:** CC BY-NC 4.0; the pinned README describes research use only and prohibits commercial use. The combined DiTraj implementation is noncommercial unless rights are clarified. [Pinned license statement](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/README.md#L295-L297)
- **Wan2.1:** official repository uses Apache-2.0, but keep the checkpoint’s own model-card terms in the experiment record as well. [Official Wan repository/license](https://github.com/Wan-Video/Wan2.1)
- **Frame Guidance:** paper and official code are public, but a top-level repo license was not visible in this source review; verify before code reuse. The paper’s license and repository license are separate matters.
- **Tora / Wan-Move:** not required for this probe. Tora’s released code is Apache-2.0 while the relevant checkpoint follows CogVideoX model terms; Wan-Move requires a 40-GB GPU according to its official instructions. Neither is a cheaper drop-in online feedback controller for the current stack.

## Stop/go conditions

1. **Before the T4 experiment:** verify exact commit, package versions, custom transformer hash, checkpoint revision, and actual GPU. The current M0 result is not a substitute for a pinned DiTraj baseline.
2. **Before signed arms:** run the zero-action hook against the unhooked DiTraj baseline with the same initial-latent hash; require identical output frames and matching post-prefix latents.
3. **Before online feedback:** exact decode-matched no-op output equals the no-preview baseline; the observer passes a fresh generated-video validation against blinded labels; confidence is calibrated or conservatively fail-closed.
4. **Before a signed-action claim:** ±1-cell interventions yield a repeatable, correctly signed object-center change on mirrored paths and paired seeds without visibility, identity, size, or background regressions.
5. **Before calling it a policy improvement:** held-out path ADE/FDE improves against baseline, no-op, fixed-shift, shuffled-residual, and wrong-sign controls using clip/seed as the evaluation unit.

If gate 3 fails, don’t tune a more elaborate residual on top of an uncalibrated actuator. Port Frame Guidance as the direct-loss alternate or test a spatially indexed latent/QK reader to locate the object, then choose a control signal from measured response.

## Sources and local implementation record

- Local [paired DiTraj runner](run_ditraj_paired_path_pilot.py), [feedback runner](run_ditraj_feedback_pilot.py), [feedback adapter](ditraj_feedback_adapter.py), and [adapter design](ditraj_feedback_adapter_design_v1.md).
- [Pinned DiTraj pipeline](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/pipe.py), [attention processor](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/attention_processor.py), [Wan transformer](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/transformer_wan.py), and [README](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/README.md).
- [Diffusers v0.33.1 UniPC scheduler](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_unipc_multistep.py) and [Wan VAE decode with forward-hook decorator](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L3196-L3225).
- [Frame Guidance paper](https://arxiv.org/abs/2506.07177) and [official implementation](https://github.com/agwmon/frame-guidance).
- Local timing/resource record: `m0/runs/wan_t4_2026-10-02_retry5/generation_packet.json`; exact scheduler conversion record: [ditraj_unipc_capture_smoke_v1.json](ditraj_unipc_capture_smoke_v1.json).
