# Native video-action policy: paper choice and feasibility

**Status:** design and source audit complete; no new model inference or policy training ran.

## Decision

Use **Unified World Models (UWM)** as the lead paper for the longer-term native video/action-policy branch. Keep the registered fixed-camera P4 / DiTraj experiment as the first Wan control-mechanism test after its human-reference gate passes. These answer different questions:

- P4 asks whether a measured image-space trajectory residual can change later denoising controls in the existing Wan/DiTraj pipeline.
- The native branch asks whether a state-conditioned video denoiser can jointly predict future video and executable robot actions, and whether that integration improves closed-loop task outcomes.

UWM is a design reference, not a drop-in Wan implementation and not evidence for the P4 feedback claim. The current T2V M0 model also has no current-observation input, so an action-producing controller cannot be built on that run alone.

## Preserve the project's scientific spine

This is a parallel architecture hypothesis, not a replacement for the original mechanistic program. The evidence chain remains:

```text
observability → forecastability → correctability → action specificity → downstream utility
```

We still need to show that a robotics-relevant signal is recoverable inside the generator, predicts a later failure before final RGB, can be changed by a causal intervention, and changes the intended action or trajectory without damaging unrelated capabilities. A joint action/video objective by itself proves none of those points.

The registered M1–M3 sequence remains active and binding: resolve the common generator/conditioning route and validate independent failure metrics on a small feasibility batch before scaling M1; use M2 to locate geometry emergence and establish a predictive failure probe; only then use M3 to test a causal geometry intervention. M2 and M3 remain gated on M1 evidence. The native simulator/data lane can prepare in parallel, but it does not unblock or satisfy those milestones.

The full policy portfolio is still P0–P9: temporal consistency; depth-order/3D spatial adherence; object identity/grounding; rigid geometry/orientation; trajectory adherence; camera motion; grasp/contact persistence; causal interaction/support; physical plausibility; and multi-view persistent world. Work one family at a time, measure isolated effects and interference, and compose only after those effects are understood. P4 is one family and one narrow fixed-camera study, not the whole program.

## Why UWM is the lead paper

UWM couples future-image and action diffusion in one transformer. It independently samples action and image diffusion timesteps so one model can represent a policy, forward dynamics, inverse dynamics, and future-image prediction. It trains on action-labeled robot trajectories and also studies action-free video co-training. The paper reports improved task success over its evaluated baselines, including gains from video co-training in its tested settings. Those are UWM paper results, not LatentGate results. [Paper](https://arxiv.org/html/2504.02792)

The published implementation is a 12-layer, 768-wide diffusion transformer. It uses a ResNet-18 current-observation encoder, a frozen SDXL VAE that maps 224×224 images to 28×28×4 latents, patchifies future-image latents with (4,4,2) patches, projects action chunks through an MLP, and uses register tokens to exchange information between modalities. The reported action chunk has length 16. The method uses DDPM noise prediction with independently selected modality timesteps. [Architecture and objective](https://arxiv.org/html/2504.02792)

That gives us a useful design principle: couple action and future-video objectives while preserving separate modality noise levels. It does not specify how to attach that branch to Wan's flow-matching transformer. A Wan adaptation would need its own action-time parameterization and loss scaling; it must not silently replace Wan's video objective with UWM's DDPM objective.

The paper reports 100,000 DROID pretraining steps in 24 hours on four A100 GPUs. Its LIBERO experiments use 4,500 trajectories for the 90-task pretraining set, then fine-tune on task-specific expert demonstrations. This scale is outside the currently demonstrated T4 lane. The official repository provides training code and LIBERO/DROID checkpoints, but the repository does not expose a software license; paper licensing does not establish code or checkpoint reuse terms. Inspect or run those artifacts only after their terms are clear. [Official repository](https://github.com/WEIRDLabUW/unified-world-model), [paper compute and benchmark details](https://arxiv.org/html/2504.02792)

UWM's stated limits matter for our transfer: it does not yet learn from large-scale human video, and the paper notes artifacts in forward-dynamics reconstructions. It predicts actions and future observations from robot observations; it does not probe or steer the internal states of a pretrained Wan video generator. [Limitations](https://arxiv.org/html/2504.02792)

## Where the nearby methods fit

**SUAVE** is a direct recent collision for unified video/action generation and closed-loop robot control. It uses a discrete masked-diffusion transformer with MMaDA/MAGViT video tokens and discretized action tokens. Its design is informative, but it is farther from Wan's continuous video latents and flow-matching objective; no author code or checkpoint was found in the paper materials reviewed here. [SUAVE paper](https://arxiv.org/html/2610.04009)

**VPDD** is an implementation reference for learning a policy from video-model features, but keeps video prediction and action diffusion in separate models. It is less direct for a shared native video/action denoiser. [Paper](https://arxiv.org/abs/2402.14407), [official repository](https://github.com/tinnerhrhe/VPDD)

These papers bound the claim. LatentGate should not claim novelty for unified action/video diffusion, action-free video co-training, or trajectory-conditioned video generation. A distinct contribution would require a well-controlled Wan-compatible integration and a measured improvement in generated-video behavior and closed-loop robot outcomes.

## Local evidence and exact gap

The full [Video Diffusion Roadmap chat](https://chatgpt.com/share/6abe4e3d-4570-83ea-8e63-5f0c3962fb9e) describes the broader project: inspect internal spatiotemporal states, find where geometry, identity, motion, camera, and contact information appears, intervene on a selected failure, then measure downstream synthetic-data value. The local [research OS](../LATENTGATE_RESEARCH_OS.md), [four-agent bootstrap](../CODEX_MULTI_AGENT_BOOTSTRAP.md), [context plan](../robotics_video_diffusion_context_and_plan.md), and experiment registry contain those source artifacts.

The current M0 Wan run gives 17 decoded frames and spatially retained video latents of shape `[1,16,5,60,104]`. With Wan's `[1,2,2]` patch size, that is 7,800 video tokens at width 1,536. Post-step latents and predicted-clean latents preserve spatial layout. The saved transformer summaries do not: they are spatially pooled to `[1,5,1536]`. The attention capture is sampled-query self-attention statistics, not a full spatial map. So the current packet supports latent-tensor analysis, but not a spatial hidden-state observer trained from the saved transformer summaries.

M0 demonstrates a 17-frame Wan T2V inference on a T4 in 160.02 seconds at 7.91 GiB peak allocation. It does not demonstrate training or joint action/video fine-tuning. The designated Colab profile's last recorded UI check showed a disconnected runtime, zero compute units, and no subscription; future T4 access remains unknown. The local Python environment has PyTorch, NumPy, and Pillow, but no Gymnasium, MuJoCo, LIBERO, Robomimic, PyBullet, or Pymunk installation.

The prepared M1 CogVideoX-5B-I2V path is an image-conditioned candidate, but it has not run. Its T4 fit is unmeasured. A state-conditioned backbone and executable action-labeled interactive data are prerequisites for a real visuomotor policy. The Kubric fixtures provide known image geometry, not executable actions or environment feedback. The P4 lane remains gated on independent human labels; its box-mask adapter has no GPU video or efficacy result.

### P4 Gate 1 remains open

The existing reference-reliability gate requires two independent raters to label all 34 frames across both v5 mirrored Kubric clips. The pooled result and each mirrored clip must independently meet all three thresholds: at least 95% joint visibility/count/identity agreement; at least 90% unique visible center-pair coverage; and p95 tight-box midpoint disagreement no greater than 0.5 target-grid cell on either axis. Regardless of numeric pass, review each clip's worst radial-disagreement frame and any distinct nonzero x/y maxima while still blind; correct entry/export errors and rescore. Score geometry only when both the rater and fixture truth identify one visible, identity-consistent target. Passing the first mirrored pair is not enough: repeat the screen on a second independently rendered mirrored-seed block. Blank templates and synthetic UI checks are not human evidence. This gate governs P4; it does not block independent simulator/data preparation for the native lane.

## Translation to a Wan-compatible model

The target architecture is a state-conditioned video model with a shared denoiser and two output streams:

1. Encode the current observation and task goal as context.
2. Denoise a future-video latent sequence using the pretrained video model.
3. Add a short action-token sequence with its own projection, modality/position embeddings, and noise-time embedding.
4. Let video and action tokens exchange information inside the denoiser.
5. Predict the action-stream target from its own continuous noise process while retaining Wan's video flow-matching loss.
6. Train on simulator/robot episodes with both objectives; separately compare an action-free-video path with video-only loss against UWM-style full-noise action marginalization.
7. At inference, predict a short video/action horizon, execute the first action, observe again, and replan.

This is an architecture hypothesis, not an implemented model. A small action head on frozen Wan features is an intermediate representation test; it is not the final native policy because it does not make the video generation itself action-aware. The full native branch requires an image-conditioned video denoiser, action tokens inside the denoiser, and a joint objective.

## 2026-10-15 architecture recommendation after scope clarification

The idea of examining frames or chunks is exploratory; it does not require independent-frame or chunk-by-chunk generation. Keep three decisions separate:

- **For current controlled experiments:** denoise one complete video sequence per arm and score every decoded frame. Measure internal states at the latent time positions the backbone actually provides. The saved Wan M0 sample has only five latent-time positions for 17 decoded frames; its sparse interpolation audit is one red-mask proxy clip and is not a general guarantee. Independent frames would discard temporal context and make identity, continuity, and event timing harder to interpret.
- **For a future closed-loop controller:** prefer a receding-horizon hybrid. At each new observation, jointly predict a temporally coherent future-video horizon and a short continuous action horizon, execute the first action or small action prefix, observe again, and replan. Here “hybrid” means feedback between full-sequence prediction calls; it does not mean generating unrelated frames. Choose the horizon from measured task latency and control needs.
- **For the eventual learned policy:** a shared video/action denoiser is a defensible research target, with state and goal conditioning, coupled video/action tokens, and separate modality noise processes. UWM provides a continuous video-plus-action diffusion precedent. The newer τ₀-WM reports a shared video-diffusion backbone with continuous action chunks, while SUAVE uses a discrete masked-diffusion vocabulary for video and action. DiTraj already covers training-free trajectory control. These works make generic “unified video/action diffusion” or “trajectory-controlled video” poor novelty claims; the sharper contribution must be the measured internal-state observer/intervention and its downstream robot value. See [UWM](https://arxiv.org/abs/2504.02792), [τ₀-WM](https://arxiv.org/abs/2606.01027), [SUAVE](https://arxiv.org/abs/2610.04009), and [DiTraj](https://arxiv.org/abs/2509.21839).

Do not implement or train the joint model yet. The broad feasibility lane still lacks verified RGB-to-action timing and an admitted current GPU runtime. The pinned LIBERO-Goal HDF5 has now been downloaded to temporary storage, hash-verified, and statically inspected. Its env_args records the Panda OSC_POSE configuration, seven action dimensions, 20 Hz control frequency, and two 128×128 RGB cameras. Across all 50 demos, saved robot joint/gripper observations match the next flattened simulator state row exactly for 4,619 adjacent pairs; this confirms the proprioception/state one-row offset, but does not prove the RGB phase. The HDF5 layout is consistent with the pinned LIBERO converter: same-row state/action is pre-action, while its source collects RGB/proprioception after the action. Subject to that converter having produced this artifact, a shifted view pairs obs[k-1] with actions[k] for k>=1, but the first stored action has no pre-action camera sample. The file does not record its converter revision, timestamps, or independent task-success labels; the pinned converter writes the final reward/done markers unconditionally. See the [all-episode static audit](2026-10-15_libero_episode_alignment_audit.md). Next verify converter provenance or regenerate an export with explicit source indices/phases, recover the initial policy observation, and validate the RGB mapping with source evidence or replay. Only then choose the image-conditioned backbone and action horizon against a valid sample. Colab/Kaggle status and the separate P4 human-label gate do not supply this missing data-path evidence.

## Minimal experiment ladder

### A. Establish action-labeled interactive data

Use **LIBERO-Goal / `put_the_bowl_on_the_plate`** as the provisional one-object placement task for the first data-path study. It is listed in LIBERO's official task map. The pinned Hugging Face file `libero_goal/put_the_bowl_on_the_plate_demo.hdf5` is at commit `97773100c1474cd0d686ebd173cc0e4fd5442466`, SHA-256 `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406` (468,246,288 bytes). Its payload was downloaded to temporary storage for a hash-verified static audit and is not stored in this repository. LIBERO's README directly links the Hugging Face dataset as its distribution route and lists dataset terms as CC BY 4.0. The Hugging Face Apache-2.0 tag applies at repository level; an open upstream issue asking whether Apache is authorized has no maintainer clarification. Use CC BY 4.0 attribution as the conservative project basis; do not rely on Apache to waive attribution. The official dataset page lists workspace/wrist RGB, proprioception, task language, and PDDL scene descriptions. The documented setup is old (Python 3.8.13 and PyTorch 1.11/CUDA 11.3) and no setup path has been tested on this machine. The [v2 episode contract](native_video_action_episode_schema_v2.md) records policy-visible observations separately from privileged simulator truth and now tracks each modality's source index and pre/post-action phase. The pinned wrapper defaults to Panda + OSC_POSE at 20 Hz; with its pinned robosuite 1.4.0 default config, the provisional arm action is six normalized pose-delta values and the Panda gripper adds one value (-1 open, +1 close). The [source audit](2026-10-12_libero_action_semantics_and_alignment_audit.md) records those defaults and their limits. The [single-episode audit](2026-10-15_libero_episode_alignment_audit.md) found that the HDF5 layout is consistent with the pinned `create_dataset.py` writer, which appends RGB/proprioception after `env.step(action)` while writing same-row `states` and `actions`. Conditional on that writer, build a transition from `obs[k-1]` to `actions[k]` for `k >= 1`; the first stored action lacks a pre-action camera observation. The HDF5 does not record the converter revision, so exact artifact lineage and replay validation remain open. Keep this simulator dataset independent of the Kubric human-label gate and do not treat simulator reward/done markers as human annotation or independent success evidence.

### B. Test representation access

After selecting an image-conditioned backbone, capture spatially retained video tokens at one or two blocks and one or two denoising times. Use episode-held-out splits to probe object state, target displacement, and action labels. Compare with constant, task-text-only, and shuffled-pair baselines across repeated held-out seed blocks. Fail this stage if the readout does not improve on held-out episodes or its calibration collapses under new initializations. Current M0 pooled summaries cannot serve as this feature set.

### C. Test a small action stream

With the video backbone frozen, train only a compact action projection/head on simulator labels. Verify it predicts executable actions in closed-loop simulation with repeated held-out seed blocks and an independent evaluator. This stage checks the action interface and data pipeline; it does not establish a video-generation gain.

### D. Test the native joint model

Add action tokens to the image-conditioned video denoiser and train low-rank/adaptor parameters plus the action head. Compare, at matched data and update budgets:

- frozen-video/action-only head;
- video-only adaptation;
- joint video/action adaptation with correct action pairs;
- joint adaptation with shuffled action pairs;
- joint adaptation with sign-reversed actions;
- P4/DiTraj only after its own human-reference, actuator, and compute gates pass.

Use repeated paired held-out seed blocks for closed-loop task success and target error, with an independent evaluator. Include unseen object instances and layouts before claiming generalization. Track video target path, visibility, object identity, apparent size, contact/task completion, background/camera stability, and non-target policy families as separate guardrails. Report collateral effects on the pretrained model's broader video prior. Keep the same initial conditions, action horizon, video sampling budget, training data/update budget, and inference compute across arms.

### E. Test synthetic-data value

Only after the native joint model beats the matched controls should we test downstream synthetic-data utility. First measure a real-only learning curve over several real-demonstration budgets. Build synthetic quality tiers (vanilla, one-policy improved, multi-policy, and high-confidence filtered), then test a mixing grid such as S/R ∈ {0, 0.5, 1, 2, 4, 8}; do not assume more synthetic samples help. At target performance P*, report the equivalent-real-data substitution fraction `1 - R_mix(P*, S, Q) / R_base(P*)` for each quality tier Q and synthetic amount S. Require positive marginal value over a useful range and generalization beyond training objects/layouts before claiming a useful threshold.

The downstream checks include inverse-dynamics recovery, policy learning, and real-data substitution. Evaluate unseen object instances, layouts and backgrounds, longer horizons, contact-rich tasks, precision placement, and failure-recovery trajectories. Match real-data and compute budgets, use an independent evaluator, and report uncertainty across repeated seeds. This is the project North Star for useful synthetic trajectories, not a video-quality score.

## Acceptance standards across branches

Every policy or model-improvement claim must use an independent evaluator; repeat across paired seeds; test unseen objects and layouts; record compute and match it where relevant; establish causal effect with matched, shuffled, sign-reversed, or other appropriate controls; and measure collateral effects on non-target metrics and the broad pretrained prior. Do not infer policy-bank-wide improvement from one task or one failure family. Any adapter or fine-tuning result must report what it preserves and what it damages in the base generator.

## Falsification and stop rules

- Stop the spatial-probe branch if state/action readouts fail on held-out episodes or do not beat task-text-only controls.
- Stop the action-stream branch if the small head cannot overfit a controlled subset and produce valid closed-loop actions.
- Reject the native-policy claim if joint training does not improve held-out closed-loop success/target error over video-only and action-only controls.
- Reject causal attribution if gains remain under shuffled or sign-reversed action labels.
- Reject a video-improvement claim if action metrics improve while target-path, visibility, identity, apparent-size, contact, or camera guardrails materially worsen.
- Make no robot-utility claim until a real-only learning curve and synthetic quality/mixing study establish positive marginal utility, a measured equivalent-real-data substitution fraction, and generalization on the downstream checks above.

## Immediate next action

Keep P4's open annotation gate explicit. The provisional LIBERO task and v2 episode contract are documented; a single hash-verified HDF5 copy was inspected in temporary storage and was not added to this repository. The original LIBERO README identifies CC BY 4.0 for datasets, so keep that attribution; the Hugging Face repository's Apache-2.0 label remains unexplained by the open upstream issue. One-episode static results and the conditional observation shift are in the [audit report](2026-10-15_libero_episode_alignment_audit.md). Before collection or policy training, verify the artifact's converter provenance, resolve the missing initial pre-action camera input, and validate phase alignment with explicit evidence or replay. Do not start an unlicensed UWM checkout or promise a Wan/T4 training run. After the simulator/data contract is frozen, choose the smallest image-conditioned backbone and measure its actual inference/training memory before implementing the joint stream.
