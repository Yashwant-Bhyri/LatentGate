# Challenge memo: is the one-cell DiTraj probe the right next experiment?

**Date:** 2 October 2026  
**Scope:** Fixed-camera, single-object, 2D path actuation in the Wan2.1-T2V-1.3B pilot.  
**Work performed:** Fresh review of primary papers and official repositories, plus a read-only check of the existing experiment plan. No model was run and no code, notebook, or `STATE.md` file was edited.

## Decision

Keep **Wan2.1-T2V-1.3B + DiTraj** as the first open-loop control route. It is the closest published method to the selected backbone: the DiTraj paper evaluates Wan2.1-T2V-1.3B, and its official runner builds per-frame boxes on the model's spatial token grid. But a lone “move the box by one token and check the sign” result would be too weak and could answer the wrong question.

The better next decision experiment is staged:

1. Establish the local **open-loop path-control baseline** (Wan base, foreground/background guidance, full DiTraj) on mirrored paths with a shared start and matched noise.
2. If full DiTraj is actually path-responsive and the independent observer passes its gate, run a **post-prefix action-dose sweep**: no-op and ±1/±2 grid-cell shifts to the future foreground/background mask, with the first five denoising calls and early RoPE intervention held fixed.

This preserves the original one-cell perturbation as the smallest-resolution probe, but adds a larger dose to distinguish “wrong sign/no control” from “one cell falls inside a dead band.” It also separates full DiTraj trajectory actuation from the narrower late mask action that the feedback adapter can actually change.

## Why the distinction matters

The DiTraj paper describes two connected but different controls. Foreground/background separation routes foreground text to tokens inside the box mask. Its STD-RoPE intervention aligns the spatial position embeddings of foreground tokens across frames, with an attention mask for repeated positions. The authors report that foreground/background separation alone handles smaller movements but struggles with larger ones; STD-RoPE was introduced to improve cross-frame foreground correspondence in early denoising.

In the official runner, `bbox_mask` is constructed before sampling, and the transformer uses that mask to form the early RoPE alignment and spatial text mask. The public example runs Wan2.1-T2V-1.3B at 480×832, 81 frames, with 50 denoising steps. The paper sets foreground/background guidance for 30 steps and RoPE alignment for 5; it reports 185 seconds for Wan base and 196 seconds for DiTraj on its A100 setup (+5.9%). These are paper settings, not measurements of the current 17-frame, 30-step T4 pilot.

The current pilot instead uses 17 RGB frames (five latent-time control slices), a 30-step schedule, `mask_step=18`, and an update after zero-based step 4. The project adapter can update the later cross-attention mask while keeping the original early RoPE action fixed; with this window, steps 5–17 leave 13 mask-guided calls after the update. The current pilot's `fixRope_step=3` is also shorter than the paper's five-step setting. **That update does not replay full DiTraj with a shifted path.** It changes which video tokens see the foreground versus background text in later calls; it does not directly translate pixels. Therefore:

- A full-run path-shift test measures the combined effect of foreground/background routing and early STD-RoPE.
- A shared-prefix post-step-4 test measures only the later mask action.
- A result from either one cannot be used as a proxy for the other.

At 480×832, the current spatial control mask is 30×52 cells. One mask-cell shift corresponds nominally to 16 output pixels per axis under the Wan VAE-plus-patch grid, subject to verification against the loaded configuration. The mask uses integer rasterization; moving a box changes a set of foreground/background assignments, not a continuous pixel-space actuator. The local offline diagnostic says a one-cell residual changed 71 rasterized cells in its five-slice example. That verifies an input change only; it does not establish rendered motion or a reliable actuator sign.

## Primary-source comparison of candidate actuators

| Route | Evidence and implementation fit | Why it is or is not the first test |
|---|---|---|
| **DiTraj, full open-loop path** | Training-free box paths; explicitly adapted to Wan2.1 DiT. The paper reports Wan2.1-1.3B evaluation, 480×832/81-frame/50-step settings, and a 5.9% overhead on A100. The official repo pins Diffusers v0.33.1 and currently states CC BY-NC 4.0, research/non-commercial use. [Paper](https://arxiv.org/html/2509.21839v2) · [official code](https://github.com/xduzhangjiayu/DiTraj) | **Best local open-loop positive control.** Run and score it before interpreting a feedback action. It is not state feedback, and the current pilot settings are a custom adaptation rather than a paper reproduction. |
| **DiTraj, late mask residual** | Uses the existing box-to-mask interface and avoids autograd. At the step-4 branch, only later cross-attention routing is changed; early RoPE remains based on the original path. | **Best first feedback actuator to calibrate** because it matches the implemented adapter and is small enough to isolate. A ±1-only null can be a quantization/dead-band result; include ±2. A positive response would validate only this late mask action, not all of DiTraj. |
| **Frame Guidance** | Training-free frame-level optimization; decodes selected predicted-clean frames, differentiates a task loss through the denoiser, and updates the latent. It demonstrates masked object-motion guidance and provides latent slicing plus spatial downsampling; the paper reports up to 60× lower gradient memory than full-sequence decoding. It also discusses flow-matching guidance. Its repo lists Wan-T2V/I2V examples but says the Wan2.1 version is to be updated, and the inspected repo page exposes no root license. [ICLR 2026 paper](https://arxiv.org/html/2506.07177v2) · [official code](https://github.com/agwmon/frame-guidance) | **Most direct alternative if a measured clean-frame center loss is desired**, but a generic differentiable object-center/identity loss is not already supplied. Its gradient path, detector loss, memory and runtime would add variables. Keep as a fallback or later comparator, not a silent replacement for the box-action test. |
| **SG-I2V** | Optimizes noisy latents against cross-frame features and target boxes, with a high-frequency restoration safeguard. [Paper](https://arxiv.org/html/2411.04989) · [MIT code](https://github.com/Kmcode1/SG-I2V) | Useful design precedent for closed-loop latent correction. It targets an SVD U-Net image-to-video model, not Wan T2V, and its repeated optimization is heavier than a mask-only calibration. |
| **Motion-Zero / FreeTraj / TrailBlazer** | Training-free object paths using noise priors, attention constraints, trajectory masks, or noise resampling. Motion-Zero's released example is ZeroScope v2 576w with U-Net-specific options and Python 3.8; its repository is Apache-2.0. [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/32198) · [code](https://github.com/vpx-ecnu/MotionZero) · [FreeTraj paper/code](https://arxiv.org/abs/2406.16863) · [TrailBlazer paper](https://arxiv.org/abs/2401.00896) | Good algorithmic comparison, but no drop-in Wan2.1-T2V-1.3B path for this pilot. Porting these U-Net methods would confound the action question with architecture migration. |
| **Tora / Motion Prompting / Wan-Move** | Learned trajectory inputs: Tora fuses compressed motion patches into a DiT; Motion Prompting trains a trajectory ControlNet; Wan-Move propagates first-frame features along dense tracks. [Tora CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Tora_Trajectory-oriented_Diffusion_Transformer_for_Video_Generation_CVPR_2025_paper.html) · [Tora code](https://github.com/alibaba/Tora) · [Motion Prompting paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.pdf) · [Wan-Move paper](https://papers.neurips.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) · [code](https://github.com/ali-vilab/Wan-Move) | Strong model-native future paths, but they require training and/or target different backbones/tasks. Wan-Move is Wan-I2V-14B; it is not a ready T2V-1.3B replacement. Too many changes for the first causal action check. |
| **DiffTrack correspondence/readout** | Video-DiT Q/K and intermediate features can expose temporal point correspondence; the paper adds cross-attention guidance for motion consistency. The official code supports CogVideoX-2B/5B and HunyuanVideo, not Wan2.1-1.3B. [NeurIPS 2025 paper](https://arxiv.org/html/2506.17220) · [Apache-2.0 code](https://github.com/cvlab-kaist/DiffTrack) | Better treated as an **observer/readout research lead** if the generated object's state cannot be reliably extracted from decoded previews. It does not currently provide a Wan-compatible residual controller. |

These sources show that box/trajectory control and inference-time video guidance are established. The useful unresolved question in this component is narrower: whether an independently measured object-state residual, observed during this Wan sample, can drive a repeatable same-sample correction beyond open-loop DiTraj and matched controls.

## Preregistered experiment

### Gate 0 — valid independent measurement

Before scoring any treatment, retain blinded framewise human tight boxes, visibility and identity/count labels as reference. Pass the existing observer gate: at least 95% agreement on identity/presence/count and human-human p95 center disagreement no greater than half a control cell per axis (8 px at native 832×480 output). If using the 160×96 Kubric reliability packets, map source coordinates per axis into that target grid first; do not use 8 px as the raw packet-coordinate tolerance. Validate any automatic observer on held-out generated frames; it must not select/crop detections using the target path. Do not use the policy-side color detector as the final scoreboard.

If this gate fails, fix labeling/observer quality first. A one-cell action is only interpretable if the measurement uncertainty is demonstrably below its nominal 16-pixel scale.

### Gate 1 — verify the local open-loop path interface

Use a direction-neutral prompt, one fixed-camera object, mirrored left/right target paths with the same first-frame box, identical initial noise per seed block, the same sampler/CFG/step budget, and the same exact checkpoint/dependencies. Run the planned base, foreground/background-only, and full DiTraj arms. Preserve every output, mask hash and run setting. Human-label all 17 RGB frames; do not infer success from the prompt, target-box overlay, a single attractive frame, or one M0 text-prompt sample.

**Fail closed** if a mask/path is silently misaligned to latent time, the start boxes differ, or matched inputs are not verified. If full DiTraj is not measurably more path-responsive than base on both mirrored directions, do not proceed to tune its feedback law. First debug the local reproduction and input-to-mask mapping.

### Gate 2 — distinguish total path control from the feedback action

After Gate 1, use two named comparisons; report them separately:

1. **Full-run open-loop sensitivity:** from the initial denoising call, shift only future target boxes by ±1 and ±2 spatial cells, keeping the first-frame box and box width/height fixed. This tests the combined published DiTraj interface (mask plus early RoPE).
2. **Shared-prefix feedback-action sensitivity:** branch after zero-based step 4 from the identical latent, scheduler history, prompt embeddings, and initial path. Freeze the original early RoPE intervention. Compare no-op, +1, −1, +2, −2 cell shifts applied only to future foreground/background masks (leave the first-frame mask unchanged). Rebuild the actual mask and log its changed-cell count/IoU before denoising resumes. This tests the action that the current feedback adapter can change.

Do not combine these into one “DiTraj moved the object” number. The first may pass while the second fails because late foreground/background routing is weaker than the early layout/RoPE intervention.

### Gate 3 — preregistered action falsification

Use five development seed blocks to verify the analysis and lock thresholds, then at least eight new paired seed blocks for the directional action calibration on the tested axis. Each block contains the same seed/noise, neutral prompt, start box and target path across no-op and action arms; run order is randomized. Start with x-axis mirrored paths. Test y only after x passes.

For every action, record the rasterized action, framewise manually adjudicated center, visibility/identity, box area, and background landmarks. Define the action response per clip as the mean displacement of the generated object center from the paired no-op, projected onto the commanded action direction over the frames where the shifted mask is active. Do not treat frames as independent samples.

**Calibration pass (screen only):** at one matched dose pair (±1 or ±2), at least 7 of 8 seed blocks show the correct sign for each action direction; the median signed displacement exceeds the larger of 8 px (half a token cell) and twice the measured paired no-op/observer repeatability; no accepted wrong-identity, duplicate or absent-object read occurs at action anchors. Report the full per-seed response and paired interval. This is a mechanism gate, not a population-level efficacy claim. Expand the calibration set if the interval is wide or results are borderline.

**Immediate falsifiers:**

- The no-op branch fails to reproduce the same-prefix comparator within the preregistered numerical tolerance.
- The control grid changes by zero cells, changes box size/start location unintentionally, or modifies the wrong latent-time slices.
- Full DiTraj has no observable signed target-path response on the mirrored paths.
- The ±1 action is null but ±2 is not: classify the one-cell setting as below the measurable action threshold; do not claim the actuator is absent.
- Neither ±1 nor ±2 yields a repeatable sign, or a correct sign depends on only one direction: reject this box-mask action for the current feedback controller.
- Full-run open-loop shifts pass while shared-prefix mask shifts fail: conclude that the late mask-only action is not sufficient; examine earlier actuation timing or a different mechanism, not a different controller gain.
- A path-error “gain” comes from object disappearance/shrinkage/replacement, a moving background/camera, or path-informed observer selection.

### Gate 4 — efficacy, only after actuation passes

Freeze the observer, action scale, feedback gain/cap and controller before held-out evaluation. Compare full open-loop DiTraj, compute-matched no-update, live state-feedback, and one preregistered falsification control (state shuffled across clips or sign reversed). Use at least ten held-out case-seed blocks over at least five distinct path cases; the independent unit is the case-seed block. Include the mirrored directions and a non-linear/reversal path. Retain all failed or unscoreable samples with a predeclared missing-target penalty.

**Policy claim gate:** paired held-out ADE is at least 20% lower than full open-loop DiTraj, its paired case-level confidence interval excludes zero improvement, and visibility, identity, apparent size, background stability and blinded visual quality do not materially regress. A no-op/shuffled/sign-reversed control must not retain the same gain. This threshold is a project decision, not a value established by the cited literature.

## Bottom line

The **smallest** one-cell signed check remains worth including, but it should not be the only next experiment and should not be described as testing “DiTraj” without naming which component is changed. The most informative next experiment is a matched DiTraj baseline followed by two separated sensitivity tests: full-run trajectory shifts and shared-prefix late-mask shifts, each with no-op, ±1 and ±2 cell actions. This keeps the first run small, makes a null result interpretable, and tells us whether to refine the existing box action or move to an outcome-based latent method such as Frame Guidance.

## Source ledger

- [DiTraj paper, v2](https://arxiv.org/html/2509.21839v2) and [official code/readme](https://github.com/xduzhangjiayu/DiTraj)
- [DiTraj official runner](https://github.com/xduzhangjiayu/DiTraj/blob/main/run.py) and [Wan transformer patch](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/transformer_wan.py)
- [Wan2.1 official model/code](https://github.com/Wan-Video/Wan2.1)
- [Frame Guidance, ICLR 2026 paper](https://arxiv.org/html/2506.07177v2) and [official repository](https://github.com/agwmon/frame-guidance)
- [SG-I2V paper](https://arxiv.org/html/2411.04989) and [official repository](https://github.com/Kmcode1/SG-I2V)
- [Motion-Zero, AAAI 2025 paper](https://ojs.aaai.org/index.php/AAAI/article/view/32198) and [official repository](https://github.com/vpx-ecnu/MotionZero)
- [FreeTraj paper](https://arxiv.org/abs/2406.16863) and [official repository](https://github.com/arthur-qiu/FreeTraj)
- [TrailBlazer paper](https://arxiv.org/abs/2401.00896)
- [Tora, CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Tora_Trajectory-oriented_Diffusion_Transformer_for_Video_Generation_CVPR_2025_paper.html) and [official repository](https://github.com/alibaba/Tora)
- [Motion Prompting, CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.pdf)
- [Wan-Move, NeurIPS 2025 paper](https://papers.neurips.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html) and [official repository](https://github.com/ali-vilab/Wan-Move)
- [DiffTrack, NeurIPS 2025 paper](https://arxiv.org/html/2506.17220) and [official repository](https://github.com/cvlab-kaist/DiffTrack)
- Existing [evaluation audit](2026-10-02_fixed_camera_path_evaluation_audit.md), [component plan](2d_path_policy_component_plan.md), and [DiTraj feedback adapter design](ditraj_feedback_adapter_design_v1.md) for project-specific gates and implementation details.
