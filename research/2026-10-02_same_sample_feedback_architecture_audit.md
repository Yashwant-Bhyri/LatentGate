# Same-sample feedback architecture audit for fixed-camera path control

**Date:** 2026-10-02  
**Scope:** Wan2.1-T2V-1.3B through the official DiTraj route; compare predicted-clean observation, spatial conditioning edits, native latent readers, distillation, and generator training.  
**Work performed:** Primary-paper and official-source review only. No code, notebook, or model was changed or run.

## Recommendation

Run a one-read, bounded state-feedback probe on the existing DiTraj path. After the first five denoising steps, derive the scheduler's current predicted-clean latent from the *guided* model output, decode it once, estimate the target object's center and visibility over the clip, then update the foreground/background cross-attention box mask for later denoising steps. Keep DiTraj's early RoPE intervention fixed. Compare the result with zero-update and sign-reversed or shuffled-residual controls.

This is the smallest causal experiment that asks whether the generated sample's measured path error can steer that same sample. It is not a demonstrated improvement or a new general paradigm. DiTraj already uses the supplied box path to control spatial text grounding and temporal self-attention; Frame Guidance already uses decoded predicted-clean frames for test-time gradient guidance; DiffTrack already intervenes on temporal attention maps to improve motion consistency. Any contribution must be narrowly stated as measured, same-sample path-residual feedback into later DiTraj box conditioning, and must beat matched controls.

First run a short **actuation check**: from a fixed prompt, seed, noise, and shared five-step prefix, compare unchanged boxes with ±1 latent-grid-cell box offsets during the remaining guided steps. If a one-cell box shift does not reliably move the measured object in the expected direction without harming identity/visibility, the proposed residual action has no known control sign; calibrate the mask-to-output response before adding a feedback law. An 8-step smoke can verify the hook, but is not an efficacy result: only three denoising steps remain after step 4.

## Exact Wan/DiTraj control surface

Wan is a video DiT. The official Diffusers model patchifies the latent video with a (1, 2, 2) spatiotemporal patch, uses 3D RoPE in video-token self-attention, and uses a separate cross-attention operation for text conditioning. In the pinned DiTraj implementation, trajectory control has two different levers:

1. **Foreground/background text cross-attention.** DiTraj encodes separate foreground and background prompts. It rasterizes each supplied box onto the latent video-token grid and restricts video tokens inside the box to foreground prompt tokens and tokens outside to background prompt tokens. This is video-token-to-text conditioning; it is not cross-frame attention and does not directly translate pixels.
2. **STD-RoPE and a repeated-token self-attention mask.** During early calls, spatial RoPE positions inside foreground boxes are aligned between frames while temporal positions are preserved. A mask suppresses interactions between repeated-position background tokens and the foreground. This changes video-token self-attention and supports stronger temporal feature transfer.

The official DiTraj runner uses 50 inference steps by default, with mask_step=30 and fixRope_step=5. Its Wan implementation builds the box-dependent cross-attention mask before denoising and selects the separate FG/BG prompt embeddings for the first mask_step denoising steps. After step 4, the cross-attention box mask can still affect steps 5–29 (25 later denoising steps under the defaults), while the RoPE modification has already ended. Classifier-free guidance typically makes two transformer forward passes per denoising step, one conditional and one unconditional. A live post-step-4 controller therefore changes only where foreground/background text is routed. It does not alter the existing sample's pixels or weights, and the object may not follow a shifted mask monotonically. This is why the ±1-cell actuation check is a useful prerequisite.

The original paper says foreground/background separation helps small movements but can miss large paths; its RoPE intervention was designed to strengthen cross-frame foreground correspondence. The paper's ablation reports improved path-control scores from combining the components. Thus the open-loop DiTraj baseline is necessary and box-path guidance itself cannot be presented as our new policy.

Sources: [DiTraj paper, method and ablations](https://arxiv.org/html/2509.21839v2); [DiTraj official repository](https://github.com/xduzhangjiayu/DiTraj); [released Wan transformer patch](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/transformer_wan.py); [released pipeline](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/pipe.py); [released attention processor](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/attention_processor.py); [runner defaults](https://github.com/xduzhangjiayu/DiTraj/blob/main/run.py).

## How to compute the observation without changing the sampler

With the DiTraj configuration, the model predicts a flow/velocity and classifier-free guidance combines conditional and unconditional predictions before the scheduler update. The pinned Diffusers v0.33.1 UniPC scheduler's flow-prediction conversion uses

$$
\hat z_{0|t}=z_t-\sigma_t\,v_{\mathrm{CFG}}(z_t,t,c).
$$

Capture that value for observation **before** UniPC advances the current latent; do not replace the scheduler's model output or perturb its history for this no-gradient experiment. Decode the preview using Wan's VAE latent mean/std convention. Feed the preview to an observer that returns, for every video frame, object center, extent, identity/count, visibility, and a scoreability flag. Estimate over the full frame, not a crop centered at the target box, so the observer cannot manufacture apparent adherence. Apply a predeclared bounded center residual only to later spatial masks:

$$
u^{\mathrm{next}}_f =
u^{\mathrm{base}}_f +
\mathbf{1}[\text{trusted}_f]\,
\operatorname{clip}\!\left(k(c^{*}_f-\hat c_f),-\delta_{\max},\delta_{\max}\right),
$$

where \(u\) is the box-mask center, \(c^{*}\) is the commanded center, and \(\hat c\) is the decoded preview's observed center. Preserve box size and lock the initial box. A missing or ambiguous object means zero action. Quantize and log the final rasterized mask because sub-cell movement may produce no model input change. Observer confidence thresholds must be calibrated against independent labels; a red-pixel heuristic is not a calibrated probability.

The exact capture point matters: Diffusers' standard callback exposes the post-scheduler latent, not the current pre-update predicted-clean estimate. The safest seam is inside the denoising loop after CFG and before scheduler.step, or a verified interception of that step's convert_model_output using the current scheduler step index. The base scheduler update should otherwise be identical across arms.

Sources: [pinned Diffusers v0.33.1 UniPC conversion](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_unipc_multistep.py); [Wan pipeline loop and callback](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/wan/pipeline_wan.py); [DiTraj pipeline loop](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/pipe.py).

## Candidate routes and evidence boundary

| Route | What is established in the source | Fit and decision for this experiment |
|---|---|---|
| **Decoded predicted-clean residual → DiTraj box mask** | DiTraj already exposes time-varying foreground box masks. UniPC provides a current clean estimate; using it to observe a sample and modify only future steps is mechanically well-defined. | **Run first as a falsifiable hypothesis.** Lowest intervention complexity and no autograd. The box mask only alters prompt routing; response direction is unknown until measured. |
| **Gradient guidance on decoded predicted-clean frames** | Frame Guidance explicitly derives clean estimates, decodes selected frames, differentiates a frame-level loss back to the current latent, and updates the sampling state. Its flow-matching form uses a clean-latent estimate. The authors test Wan-14B with 2× spatial downsampling and report limiting guidance so runtime stays within 4× base inference. | Strong prior art for predicted-clean guidance; it rules out a broad novelty claim. It needs a differentiable loss and repeated model/decoder passes. It is a later comparator, not the first T4 experiment, especially with a threshold detector. [Paper](https://arxiv.org/html/2506.07177v2); [official code](https://github.com/agwmon/frame-guidance). |
| **Latent trajectory optimization** | SG-I2V optimizes noisy latents at selected SVD steps to align intermediate features to a requested box trajectory. Its official inference uses five optimization iterations at selected timesteps; authors list an A6000 48GB test setup. | Established adjacent method, but it is a U-Net/SVD image-to-video route, not this Wan T2V DiT. Do not copy its multi-iteration gradient loop into the small run. [Paper](https://arxiv.org/abs/2411.04989); [official MIT repo](https://github.com/Kmcode1/SG-I2V). |
| **Attention/QK correspondence and attention guidance** | DiffTrack shows selected video-DiT query/key features act as point correspondences, and proposes Cross-Attention Guidance to perturb attention maps and improve temporal consistency without training. The released code supports CogVideoX-2B/5B and HunyuanVideo. | Good later observer/feature-control direction. This is a direct collision for generic “attention-guided temporal correction,” but it does not use an observed target-relative path residual and is not released for Wan 1.3B. [Paper](https://arxiv.org/abs/2506.17220); [official Apache-2.0 repo](https://github.com/cvlab-kaist/DiffTrack). |
| **Frozen backbone + trained latent reader** | DreamTraj reads q·k tracks and pooled hidden states from Wan2.2-I2V-A14B at an early denoising step, then uses a 7.1M-parameter flow-matching reader to predict a 6-DoF future path. Its paper uses 5,038 trajectories and both real/generated feature domains. | Strong precedent for a small learned reader over frozen video-diffusion features. It predicts an intended future from I2V context, not the observed state of this T2V sample, and does not feed error back into its denoising loop. Its public repo currently has docs only and says code/data will be added. Treat as architectural evidence, not an immediately reusable package. [Paper](https://arxiv.org/abs/2608.00486); [official repo status](https://github.com/whathappen0/DreamTraj). |
| **Joint correspondence supervision in generator** | Track4Gen adds a trainable feature refiner and point-correspondence loss alongside the video diffusion loss, feeding refined features back into Stable Video Diffusion. The paper reports improved appearance consistency, but also reduced camera motion and failure on fast blur/occlusion/similar objects. | The most relevant learned route for broad appearance drift, but it changes generator features/weights and was validated on SVD, not Wan. It needs a labeled trajectory corpus, a Wan port, and regression checks for motion and identity. Defer until the one-sample mechanism has repeatable value. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeong_Track4Gen_Teaching_Video_Diffusion_Models_to_Track_Points_Improves_Video_CVPR_2025_paper.pdf); [project page](https://hyeonho99.github.io/track4gen/). |
| **Simulator-in-the-loop physics** | PSIVG reconstructs a template video's 4D scene, simulates object motion, and uses simulated trajectories plus test-time texture consistency optimization to guide generation. | Relevant to a later physics/contact question, but far beyond a fixed-camera 2D path gate and a small T4 run. It is a pipeline with reconstruction/simulation, not a small latent controller. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Foo_Physical_Simulator_In-the-Loop_Video_Generation_CVPR_2026_paper.pdf); [official Apache-2.0 repo](https://github.com/MarkHershey/PSIVG). |

## Native policy and distillation path if the feedback probe works

Do not train the action rule first. Keep the bounded residual law fixed and make its *state estimate* native:

1. Save spatially indexed Wan hidden-state grids at the observation step and selected layers. Pooled per-frame feature summaries discard the location signal; they cannot localize an object.
2. Build labels on generated samples using independent tight boxes, identity, and visibility labels. Include varied object colors/textures and motion paths; keep all clips from one source seed/path in the same data split.
3. Fit a constant/prompt-only baseline and a linear per-frame spatial probe for center/extent/visibility. Evaluate on held-out seeds and trajectories. Only if the spatial probe wins should a small temporal head be considered.
4. Distill decoded-teacher state to that head; report state error and scoreability separately. Keep the action law and all feedback thresholds frozen for this comparison. At deployment, the head reads the latent grid and sends the same bounded action into later masks, avoiding an extra VAE preview.
5. Evaluate the complete latent observer + controller on held-out samples with blinded boxes. Then compare against decoded-state feedback, zero action, fixed calibration action, and wrong-sign/shuffled state. A reader's accuracy alone does not show an image-generation gain.

DreamTraj supports the premise that motion information is available in frozen video-DiT features, while Track4Gen supports adding spatial correspondence supervision to a generator. Neither establishes this specific Wan2.1-T2V feedback policy. Full LoRA/GRPO/reward training is a still later route; it needs more labeled data, compute, and a clean held-out path benchmark, and makes it harder to attribute any gain to the policy.

## Minimal decision experiment

1. Use the same prompt, path, seed, initial latent, CFG, scheduler, denoising-step budget, and checkpoint for every arm. Keep the first five DiTraj denoising steps identical. Branch on the changed box mask only afterward (or run deterministic matched full generations if checkpoint branching is unavailable).
2. First compare no box shift, one-cell shift in one horizontal direction, and the opposite shift. Use a mirrored path condition if available. This measures the box-to-object actuation sign without assuming that masks directly drag pixels.
3. If that signal is consistent, test the feedback law against decode-matched zero-update and sign-reversed or shuffled-residual arms. Choose gain/cap only on development data; freeze both before held-out confirmation.
4. Decode previews at matching checkpoints in every arm. Score final frames with independent labels. Primary metric: center path error (ADE plus endpoint error). Required guard metrics: object visibility, identity/duplicate count, box area/apparent size, and background/camera motion. A lower center error caused by shrinkage, disappearance, or camera movement is a failure.
5. Treat the first mirrored seed pair as a mechanism screen only. A small multi-seed screen can decide whether to continue; a claim of a real improvement needs more paired seed blocks and held-out paths. Never present mask changes, preview agreement, or one successful clip as policy efficacy.

## Bottom line for this stage

The feedback path is worth testing because it makes the sampling controller react to the sample it is actually producing, with minimal disruption to the chosen Wan/DiTraj stack. The first key falsifier is not whether the mask changed; it is whether changing that mask causes predictable object displacement while preserving the object. If it does, test residual feedback against causal controls. If it does not, move the policy search toward spatial Wan feature probes and Q/K tracking, rather than scaling up an unvalidated feedback heuristic.
