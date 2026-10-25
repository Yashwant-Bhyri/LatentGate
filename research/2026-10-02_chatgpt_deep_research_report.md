# LatentGate Deep Research Report: Closed-Loop Path Control for Wan2.1 Video Diffusion

## Executive assessment

**Scope.** This report narrows LatentGate to the smallest scientifically defensible experiment: fixed-camera text-to-video generation of one visible object following a commanded two-dimensional image-plane path, using **Wan2.1 T2V 1.3B, 832×480, 17 output frames, 30 denoising steps**, with a free NVIDIA T4 as the practical ceiling. The eventual target is not merely stronger trajectory conditioning; it is a **native denoising-time feedback loop** in which an intermediate predicted-clean video or spatial DiT state is observed, compared with the requested path, and used to modify a later control input.

The main conclusion is unchanged by the broader 2026 literature scan, but the implementation recommendation is now considerably sharper:

> **The most mature feasible first LatentGate probe is a DiTraj-derived controller that performs exactly one confidence-gated, signed bounding-box residual update after zero-based denoising step 4, applies the resulting persistent `active_bbox` state beginning at step 5, and rebuilds the cross-attention mask every subsequent step. DiTraj's first three hard-RoPE steps should remain frozen to the original commanded geometry. The step-14 intervention should remain a later ablation, not part of the first policy experiment.**

This recommendation is based primarily on artifact compatibility rather than paper leaderboard position. DiTraj is a **training-free T2V DiT method**, its released implementation directly instantiates `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, and its intervention is structurally a bbox-conditioned attention/RoPE mechanism rather than a new trained control network. Its official implementation is pinned to Diffusers v0.33.1 and is licensed CC BY-NC 4.0 for research. citeturn24search0turn24search4 fileciteturn27file0L2-L2

By contrast, several stronger-looking 2025–2026 systems are poor first-experiment fits. Wan-Move is a **Wan 14B I2V** system whose public inference contract requires an input image plus dense point tracks and visibility; its own documentation says the 14B model can be made to fit a 40 GB GPU with CPU/T5 offload. fileciteturn10file0L2-L2 MagicMotion is **CogVideoX-5B-I2V** and reports about 23 GB for released inference. fileciteturn11file0L2-L2 fileciteturn13file2L29-L38 Tora's public implementation is likewise CogVideoX-5B based; its original T2V configuration reports roughly 30 GiB, although the Diffusers port can reduce VRAM to about 5 GiB through very slow sequential CPU offload. fileciteturn20file0L2-L2 fileciteturn26file0L2-L2 MotionStream and 4DStreamCtrl are conceptually closer to online control but are distilled streaming systems rather than a small intervention on the public Wan2.1-1.3B T2V sampling loop. citeturn24search1turn24search5turn24search6

Most importantly, **there is still no LatentGate policy result**. The sole M0 clip and its red-pixel segmentation can serve as a smoke-test or screening artifact only. It is not evidence that path control works, that feedback helps, that step 4 is optimal, or that any feedback gain is appropriate. No feedback gain has yet been empirically identified.

The novelty claim also needs to be narrower than “closed-loop diffusion control.” By October 2026, the literature already contains trajectory conditioning, attention-based object localization, training-free inference-time refinement, latent geometric rewards, causal streaming control, and explicit “feedback control” interpretations of guidance. DiTraj itself establishes training-free bbox trajectory control in a DiT; TrajLoc uses per-object spatial cross-attention localization; VGGRPO includes latent-space geometric reward optimization and a test-time guidance variant; WorldForge develops training-free self-corrective/refinement mechanisms; MotionStream and 4DStreamCtrl perform online trajectory-conditioned streaming generation. citeturn24search0turn24search5turn24search6turn24search14

The defensible LatentGate novelty hypothesis is therefore more specific:

**Hypothesis, not an established contribution:** *during a single fixed-length T2V denoising trajectory, an explicitly observed object state from the model's own intermediate prediction contains actionable error information; feeding that measured error back into a later pre-existing trajectory-control channel causally reduces path error under matched compute, and the improvement disappears when the state is zeroed or shuffled.*

That claim is both narrower and more falsifiable than generic “self-correction.”

## Search methodology and primary-source ledger

The search was conducted as a **claim-versus-artifact audit**, not as a survey assembled from abstracts alone. Sources were prioritized in the following order: released source at an exact commit/tag; official model/configuration and inference scripts; official model and dataset cards; primary papers or their arXiv HTML; then project pages. README statements were treated as project claims unless corroborated by source/configuration. Secondary survey pages and aggregator claims were not used to establish compatibility.

For the crucial LatentGate runtime path, the audit pinned **DiTraj commit `6fa8e613c9e7c40ec2db456ab2ba34367a076886`** and **Diffusers tag `v0.33.1`**, then read the relevant pipeline and scheduler source rather than assuming callback semantics from generic Diffusers documentation. The pinned DiTraj runner explicitly selects `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, constructs a `UniPCMultistepScheduler` with `prediction_type='flow_prediction'`, `use_flow_sigmas=True`, and `flow_shift=3.0`, keeps the Wan VAE in FP32, loads the pipeline in BF16, and installs DiTraj's custom attention processors. fileciteturn9file0L2-L2

Citation tracing was both backward and forward. DiTraj belongs to a lineage that includes earlier zero/training-free trajectory methods such as Motion-Zero and TrailBlazer; later work increasingly moves toward learned point-track conditioning, causal distillation, three-dimensional tracks, and interactive world-model control. Current 2026 papers explicitly position MotionStream as a two-dimensional streaming predecessor to 4DStreamCtrl, while later systems such as PhysStream cite MotionStream, MagicMotion, and FlashMotion in the evolution toward interactive physical control. citeturn24search6turn24search11turn24search12 This matters because it makes “online control,” “trajectory feedback,” or “self-correction” by themselves unsafe novelty claims.

The non-English/Asian research ecosystem is especially important here rather than peripheral. DiTraj's authors are affiliated with BUPT, Lenovo, HKUST, and Tsinghua; Tora is released by Alibaba with both ModelScope and Hugging Face artifacts and Chinese/English demos; Wan-Move provides MoveBench in both English and Chinese; MagicMotion comes from a Fudan-led team; FlexTraj involves City University of Hong Kong, the University of Hong Kong, and Microsoft. Their official repositories expose implementation details that are often more informative for LatentGate than Western benchmark summaries. fileciteturn27file0L2-L2 fileciteturn20file0L2-L2 fileciteturn10file0L2-L2 fileciteturn15file0L2-L2

### Source ledger and what each source changes

| Source class | Primary evidence | Decision-relevant result |
|---|---|---|
| **Pinned DiTraj implementation** | `module/pipe.py`, `run.py`, exact commit fileciteturn8file0L2-L2 fileciteturn9file0L2-L2 | Changes the adapter design: callback-only bbox mutation is ineffective; control state must persist outside the fresh per-step kwargs and cross-attention must be rebuilt. |
| **Pinned UniPC implementation** | Diffusers v0.33.1 scheduler source fileciteturn6file0L2-L2 fileciteturn7file0L2-L2 | Establishes the exact predicted-clean formula used by the candidate state observer. |
| **DiTraj paper/repo** | Primary paper + official repo citeturn24search0turn24search4 fileciteturn27file0L2-L2 | Establishes a training-free, bbox-conditioned Wan2.1 T2V starting point with no new controller training. |
| **Learned trajectory controllers** | Tora, MagicMotion, Wan-Move, FlexTraj official artifacts fileciteturn20file0L2-L2 fileciteturn11file0L2-L2 fileciteturn10file0L2-L2 fileciteturn15file0L2-L2 | Establish that path conditioning itself is mature, but most stronger released systems change model/task/compute too much for the T4 experiment. |
| **Streaming / online control** | MotionStream, 4DStreamCtrl citeturn24search5turn24search6 | Establish that online trajectory updates and causal video control are not novel in isolation. |
| **Geometry/self-correction** | Geometry Forcing, VGGRPO, WorldForge lineage citeturn24search18turn24search14turn24search25 | Establish that feature-level geometric state and latent reward/self-correction precedents already exist. |
| **Tracking / known truth** | CoTracker/TAP-Vid/SAM family and Kubric primary work | Establish a principled separation between a tracked material point and object center, and provide a route to metric validation before evaluating generated clips. |

Two uncertainties remain especially decision-relevant. First, **the cost of decoding a predicted-clean Wan latent once inside the pinned DiTraj loop on the actual free T4 has not been measured**. Second, although the scheduler source is pinned, **the exact runtime tensors—step index, sigma, dtype, latent normalization, and memory cost—have not yet been captured in the LatentGate environment**. Those must remain “unmeasured,” not silently filled in from upstream documentation.

## Research landscape and artifact audit

The useful taxonomy is not simply “trajectory-control papers.” For LatentGate, the field separates into four mechanisms.

**Direct trajectory conditioning** encodes the desired motion before or during denoising. Motion-Zero, TrailBlazer, Tora, DiTraj, MagicMotion, TrajLoc, Wan-Move, and FlexTraj belong broadly here, with very different conditioning contracts. DiTraj is unusual for LatentGate because it is training-free, bbox based, T2V, DiT-native, and actually released on the target 1.3B Wan family. citeturn24search0turn24search4

**Inference-time steering and self-correction** modifies sampling according to a criterion evaluated during generation. WorldForge is an important collision for broad claims about generic denoising self-refinement; VGGRPO and later geometric-steering work show that rewards can operate in latent/geometric feature spaces rather than through RGB-only decoding. citeturn24search14turn24search25 This means “we compute a loss/reward on an intermediate latent and update denoising” is not sufficient novelty.

**Streaming/online controllers** move from fixed clips toward causal generation. MotionStream supports interactive tracks in streaming generation, while 4DStreamCtrl explicitly moves from two-dimensional image-plane tracks to a three-dimensional track interface for joint camera/object/depth control. citeturn24search5turn24search6 GeoStream goes even further toward an online state-estimation loop by reconstructing/updating a three-dimensional cache from recently generated output and reprojection. citeturn24search24 These are conceptually close to state feedback, but their *video-time* causal loop is not the same experiment as LatentGate's proposed *denoising-time* correction inside one bidirectionally generated 17-frame clip.

**Teacher/student and geometry-supervised world models** include MotionStream's causal distillation direction, 4DStreamCtrl's causal student, Geometry Forcing's alignment of video-model representations with geometry-foundation-model features, and feature-space planning/world-model approaches such as DINO-WM. Geometry Forcing has public code/checkpoints and explicitly aligns video features with VGGT-style geometric representations; its official repository also releases reprojection/revisit evaluation code. citeturn24search18 fileciteturn23file0L2-L2 These methods make “spatial DiT features contain geometry/state” a plausible LatentGate hypothesis, not a new premise.

### Strongest comparable methods

| Method | Paper claim and conditioning | Verified public artifact / actual contract | Fit to LatentGate |
|---|---|---|---|
| **DiTraj** | Training-free DiT trajectory control; foreground/background guidance plus modified spatial-temporal RoPE; bbox trajectory. citeturn24search0turn24search4 | **Verified at pinned source.** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, T2V, BF16 pipeline, FP32 VAE, UniPC flow-prediction; sparse bbox keyframes become latent masks. Repo pins Diffusers v0.33.1; CC BY-NC 4.0. fileciteturn9file0L2-L2 fileciteturn27file0L2-L2 | **Best first base.** Same public backbone and control variable; no controller training. |
| **Motion-Zero** | Zero-shot/training-free trajectory control using spatial constraints on generation rather than a newly trained trajectory network. | Paper-level precedent is strong; this audit did not verify a maintained public implementation matching Wan2.1-1.3B. | Important backward collision: training-free path steering is not new. |
| **Tora** | DiT with a learned trajectory extractor and motion-guidance fuser; supports text, visual, and trajectory conditions. | Public T2V and I2V weights/code are based on **CogVideoX-5B**. Original T2V reports ~30 GiB; Diffusers version can reach ~5 GiB using sequential CPU offload, explicitly described as much slower. Training code is public; weights follow CogVideoX licensing. fileciteturn20file0L2-L2 fileciteturn26file0L2-L2 | Strong mature trajectory baseline, but wrong backbone and much heavier. |
| **TrajLoc** | Strict per-object spatial localization through object-specific trajectory/depth conditioning and manipulation of object cross-attention around target positions; demonstrated on large CogVideoX/Wan variants. | No target-compatible 1.3B runtime was verified in this audit. | Direct collision with any proposed novelty around “per-object attention heatmaps/localization.” |
| **MagicMotion** | I2V with dense-to-sparse trajectory conditions: masks, boxes, sparse boxes; introduces MagicData/MagicBench. | **CogVideoX-5B-I2V** scripts verified. Released stage-specific controller/perception-head weights; reported inference is about **23 GB** on a 4090. Training code remains unreleased in the official README snapshot. fileciteturn12file0L2-L2 fileciteturn11file0L2-L2 fileciteturn13file2L29-L38 | Good benchmark/architecture comparator; unsuitable first T4/T2V implementation. |
| **Wan-Move / MoveBench** | Latent point-trajectory guidance for Wan; dense tracks, single/multi-object motion; benchmark for motion control. | Public checkpoint is **Wan-Move-14B-480P**, task flag is `wan-move-i2v`; inference requires an input image, track array, and visibility array. MoveBench ships English and Chinese variants. Repo says 14B can run on a **single 40 GB GPU** with T5/CPU offload. Apache-2.0 claimed for released models. fileciteturn10file0L2-L2 | Excellent evaluation/track-conditioning reference; not a T4 T2V drop-in. |
| **FlexTraj / FlexBench** | I2V with point trajectories carrying segmentation ID, temporally consistent track ID, and optional color; dense/sparse unification. | Public checkpoint and FlexBench. WAN setup downloads both **Wan2.2-TI2V-5B** and **Wan2.1-T2V-1.3B**; released inference uses `flextraj_wan.safetensors` with KV cache. The high-level setup does not by itself prove a 1.3B pure-T2V LatentGate-compatible inference contract, so that compatibility remains **unverified**. Repo metadata does not expose a standard license. fileciteturn15file0L2-L2 fileciteturn16file0L2-L2 fileciteturn17file0L2-L2 fileciteturn19file0L2-L2 | High-collision point-conditioning method; promising future comparator, not yet safer than DiTraj. |
| **MotionStream** | Causal/streaming interactive motion control from tracks, with real-time-style operation instead of fixed offline clips. citeturn24search1turn24search5 | Paper describes Wan-family teacher/student variants and substantial distillation/training; no public artifact was verified here that converts the free-T4 Wan2.1 T2V baseline into the claimed streaming system. | Strong collision for “online motion control”; strong architecture clue for future causal LatentGate, poor first probe. |
| **4DStreamCtrl** | Online 3D-track control combining object/camera/depth information; causal streaming student; reported ~20 FPS at 480p on a high-end GPU. citeturn24search2turn24search6 | Based around much larger current video backbones rather than the T4 target; no target-compatible released runtime was verified. | Future endpoint, not first experiment. |
| **WorldForge** | Training-free inference-time guidance toward world-consistent 3D/4D behavior, including self-refinement/corrective sampling ideas. citeturn24search25 | No Wan2.1-1.3B LatentGate-ready public implementation was verified in this audit. | Direct novelty collision for generic “denoising self-correction.” |
| **VGGRPO** | Geometry-aware latent representations/rewards for improving world consistency; later text describes test-time latent geometric guidance as well as training-time reinforcement. citeturn24search14 | No target-compatible public code/checkpoint was verified. | Direct collision for generic “latent reward gradient” claims. |
| **Geometry Forcing** | Align intermediate diffusion features with geometric foundation-model features; improves geometric/world consistency in paper evaluations. citeturn24search3turn24search18 | **Public code/checkpoint/evaluation verified.** Official repo supports single-image long/rotation video and RealEstate10K workflows; code/checkpoints released in 2025. Repository license metadata is `NOASSERTION`, so exact reuse rights require reading its custom license rather than assuming an SPDX license. fileciteturn23file0L2-L2 fileciteturn25file0L2-L2 | Strong evidence that spatial diffusion features can carry geometric state; not itself path-feedback control. |

A particularly important later collision is **GeoStream**: it maintains a self-refreshing three-dimensional cache estimated online from recent generated frames, then reprojects that state into future target views. citeturn24search24 That is much closer to an explicit observe-update-use loop than conventional trajectory conditioning. The remaining conceptual distinction for LatentGate is that it proposes **error feedback over denoising iterations of the same fixed clip**, rather than a history state updated as video time advances.

Consequently, LatentGate should not claim novelty for any of the following ingredients independently: bbox/path conditioning; point-track conditioning; per-object attention spatialization; predicted-clean decoding; latent/geometric rewards; feature-level geometry estimation; a later-step correction; online interactive motion input; or calling CFG/guidance “feedback.” The contribution would have to reside in the **causally demonstrated loop connecting measured intermediate object-state error to a later control action within a native pretrained T2V denoising trajectory**.

## Pinned DiTraj and scheduler audit

The newly completed pinned-source audit materially changes the implementation design and should be treated as the authoritative contract for the first adapter.

At commit `6fa8e613c9e7c40ec2db456ab2ba34367a076886`, DiTraj pulls `bbox_mask`, `fixRope_step`, and `mask_step` before entering the denoising loop. It then computes

`cross_attention_mask = self.create_cross_attention_mask(bbox_mask, ...)`

**once before the loop**, and similarly computes the hard-RoPE self-attention mask from that bbox once. fileciteturn8file0L2-L2

Inside every denoising iteration, however, it executes:

```python
attention_kwargs = {}
```

and repopulates that fresh dictionary using the already-computed masks. The transformer forward pass follows. DiTraj then executes

```python
latents = self.scheduler.step(...)
```

and **only after scheduler stepping** invokes `callback_on_step_end`. The callback return path can replace `latents`, `prompt_embeds`, and `negative_prompt_embeds`; it does not cause the precomputed bbox cross-attention mask to be reconstructed. fileciteturn8file0L2-L2

The consequence is exact, not speculative:

> **Mutating a bbox only inside `callback_on_step_end` cannot steer the next DiTraj step through its bbox-conditioned cross-attention path.** The next loop iteration throws away the local `attention_kwargs`, recreates it, and inserts the previously computed `cross_attention_mask`.

The LatentGate adapter therefore needs two distinct pieces of state:

```text
command_bbox        immutable desired trajectory
hard_rope_bbox       immutable copy used only for early hard-RoPE geometry
active_bbox          persistent mutable control state
```

and it must move mask construction inside the loop conceptually as:

```text
for denoise step i:
    cross_mask_i = make_cross_attention_mask(active_bbox, encoder_mask)
    hard_RoPE_mask = immutable_initial_hard_RoPE_mask, only while i < 3
    transformer(...)
    compute/capture state estimate as scheduled
    scheduler.step(...)
    after i == 4:
        active_bbox <- feedback_policy(state_estimate, command_bbox)
```

The cross-attention mask used at `i=5` and later is then rebuilt from the updated persistent `active_bbox`. The **hard-RoPE geometry should not be regenerated from that corrected bbox**. For the LatentGate experiment, the first three steps, `i = 0, 1, 2`, remain fixed to the original commanded geometry. This “three hard-RoPE steps” setting is a LatentGate experimental constraint; it should not be confused with DiTraj's released `run.py` default `fixRope_step=5`. The upstream default is verifiable in the pinned runner. fileciteturn9file0L2-L2

The same runner exposes another useful implementation fact for the 17-frame experiment. It computes

```python
latent_num_frame = num_frame // 4 + 1
PATHS = plan_path(..., video_length=num_frame)[::4]
```

so a 17-frame clip gives **five temporal bbox-mask slices**, corresponding naturally to output-frame positions approximately 0, 4, 8, 12, and 16. fileciteturn9file0L2-L2 The first controller therefore does not need to manipulate 17 separate latent masks. It should estimate the object's full 17-frame path for evaluation, but map its correction back onto these five latent control slices, with slice 0 frozen so that the starting location is unchanged.

### Predicted-clean state is source-supported

Diffusers v0.33.1 confirms the exact UniPC conversion needed for the first observer. When `predict_x0` is active and `prediction_type == "flow_prediction"`, `convert_model_output` computes

\[
\hat{x}_0 = x_t-\sigma_t\,v_\theta(x_t,t),
\]

where the implementation obtains `sigma_t = self.sigmas[self.step_index]`. If scheduler thresholding is configured, it subsequently applies `_threshold_sample` before returning that predicted clean sample. fileciteturn6file0L2-L2

The scheduler's `step()` implementation explicitly calls

```python
model_output_convert = self.convert_model_output(model_output, sample=sample)
```

before carrying out the UniPC corrector/predictor updates and incrementing the step index. fileciteturn7file0L2-L2

This validates the proposed **predicted-clean observer mathematically** but does not validate its usefulness as an object-state sensor. It also does not replace a runtime audit. Before any policy experiment, LatentGate still needs to record the actual `step_index`, `sigma`, scheduler configuration, dtype, latent range, and decoded output from the pinned environment at `i=4`. **Pinned-runtime capture remains unmeasured.**

Likewise, **in-loop VAE cost remains unmeasured**. The final DiTraj decode rescales latents according to Wan's VAE latent mean/std and calls `vae.decode`; source inspection proves the decode path exists, not that one extra predicted-clean decode will fit comfortably or cheaply on a free T4. fileciteturn8file0L2-L2

### The first controller should be deliberately weak

No feedback gain has been measured. Therefore the first policy should not contain a hand-tuned aggressive proportional controller presented as though it had empirical standing.

At step 4, let the observer produce centers \(\hat c_f\) for the visible object and the command supply \(c_f^\*\), all in normalized image coordinates. Define residual

\[
r_f=c_f^\*-\hat c_f.
\]

For the five latent temporal slices, aggregate/interpolate the residuals to frames \(0,4,8,12,16\), set \(r_0=0\), confidence-gate the update, smooth it over video time, and clamp both axes:

\[
\Delta c_j =
g\;\mathrm{clip}\!\left(\operatorname{smooth}(r_j),
-\Delta_{\max},+\Delta_{\max}\right).
\]

The updated bbox translates by \(\Delta c_j\) **without changing its width or height**. Translational control is the cleanest first actuator because otherwise size errors and center errors become inseparable.

Here \(g\) is intentionally **undefined today**. It should be measured from an open-loop actuator-identification experiment described below, not guessed from the M0 clip. The confidence gate should also be calibrated before policy use.

The timing should be precise: capture the step-4 model prediction while the scheduler is still at the matching sigma; compute the observation; allow the normal zero-based step-4 scheduler transition to finish; then commit the new `active_bbox`; use it when constructing the cross-attention mask for step 5. This leaves the denoising schedule itself unchanged.

## Measurement, independent tracking, and known-truth validation

The evaluation unit should be the **17-frame final RGB clip**, not the latent mask and not DiTraj's own attention map. The evaluator must be independent of the control mechanism or the experiment risks measuring compliance with its own internal representation.

Each frame should have a compact gold record:

\[
(f,\; visible,\; identity\_ok,\; x_1,y_1,x_2,y_2,\; c_x,c_y).
\]

Coordinates should be normalized to the 832×480 canvas. `visible` means the commanded object is sufficiently present for a human to identify it. `identity_ok` is false for duplication, substitution, or a morph severe enough that the target object cannot confidently be associated with that frame. The center should be derived from the adjudicated box or object mask according to a fixed rule; for this bbox-control experiment, **bbox center should be the preregistered primary center**, while mask centroid can be retained as a secondary measurement.

### Primary trajectory metric

Let \(D=\sqrt{832^2+480^2}\) and \(c_f^\*\) be the commanded center. Define normalized center error

\[
e_f=
\begin{cases}
\|c_f-c_f^\*\|_2/D,& \text{visible and identity-valid},\\[3pt]
1,& \text{missing, unidentifiable, or undetected}.
\end{cases}
\]

The clip's primary error is the mean or, preferably, a preregistered combination of **mean center error plus endpoint error**. A fixed penalty of 1 for a failed visible-frame detection ensures that disappearing objects cannot improve the metric by removing hard frames. This is an experiment-design choice, not a literature-standard constant, and it should be frozen before looking at policy outcomes.

Report at minimum center MAE, endpoint error, maximum center error, missing-frame count, identity-failure count, and bbox IoU against the commanded bbox where commanded size matters. For constant-size commands, also report

\[
E_{\text{size}} =
\operatorname{median}_{f>0}
\left|\log\frac{A_f}{A_0}\right|
\]

so a controller cannot “improve” center location merely by expanding the object across the target region.

### Mirrored path design

A particularly strong low-cost control is a **same-start, same-noise mirrored pair**. For a path \(p_f^{+}\) beginning at \(p_0\), define

\[
p_f^{-}=2p_0-p_f^{+}.
\]

Both commands therefore have exactly the same initial position, prompt, model, denoising schedule, generator seed and initial latent noise. Only the sign of the requested displacement changes.

This is substantially more informative than comparing arbitrary leftward and rightward generations. If the controller genuinely reads an error and responds to it, both the observed residual and applied update should reverse appropriately under this intervention. An uncontrolled content prior, camera drift, or fixed implementation bias will generally not exhibit that clean antisymmetry.

The same-noise pairing should be preserved through every critical comparison: open-loop DiTraj versus feedback, positive versus mirrored negative path, true-state versus shuffled-state, and step-4 versus step-14 timing.

### Rolling spheres: do not confuse a material track with object center

Point trackers answer a different geometric question from this experiment's controller. The TAP-Vid formulation defines a track as the temporal trajectory of a **physical point** together with visibility/occlusion. CoTracker-style systems likewise propagate point identities through time. For a rolling sphere, a marked surface point rotates around the projected object center, so its image-plane trajectory can deviate substantially even while the sphere's center follows the commanded path perfectly. A point-track error is therefore not, by itself, an object-center error.

The independent evaluation stack should consequently have two branches:

**Object branch.** Segment/detect the entire object in each frame, then derive bbox and center. A SAM2/SAM-family video mask seeded independently from frame 0 is a useful proposal generator, but every 17-frame research clip is small enough that masks/boxes should be human-reviewed rather than accepted blindly.

**Surface-track branch.** Run CoTracker or an equivalent point tracker on several interior object points as a diagnostic of appearance/identity continuity. Do not average those tracked surface points and call the result the object center on a rolling or deforming object unless a rigid-body model explicitly justifies that operation.

This distinction should be experimentally demonstrated rather than left as an annotation convention.

### Synthetic known-truth fixture

Before evaluating generated videos, validate the measurement code on a small **Kubric/PyBullet/Blender-style fixture**, where projected object center, segmentation mask, bbox, depth, pose, and material point locations are known exactly. Kubric was designed specifically to synthesize scenes with rich ground-truth annotations on top of Blender/PyBullet, making it better suited to validating a metric implementation than a generative benchmark whose ground truth is itself estimated.

The minimum fixture should contain:

1. a translating non-rotating cube or billboard following the same normalized paths planned for Wan;
2. a translating **rolling textured sphere** with known angular velocity;
3. fixed camera and static background;
4. exact mask, projected center, bbox, pose, and one marked surface-point projection for every frame.

The evaluator should recover object center independently of the marked point. A useful pre-policy engineering gate is **≤1% image-diagonal center MAE and ≥0.90 mean bbox IoU on the synthetic fixture**. The rolling-sphere case should deliberately show that the material-point trajectory can violate this center criterion while the segmentation/bbox center remains correct. These are proposed validation thresholds, not claims from Kubric.

### Benchmarks are diagnostic, not interchangeable leaderboards

MoveBench is useful because its public contract explicitly includes trajectory data and visibility and has English/Chinese versions, but it evaluates the Wan-Move 14B I2V regime, not LatentGate's 1.3B T2V regime. fileciteturn10file0L2-L2 MagicBench is likewise valuable for learning how dense/sparse trajectory accuracy and visual quality are separated, but MagicMotion itself is a CogVideoX-5B I2V controller. fileciteturn12file0L2-L2 FlexBench is a useful point-control benchmark artifact, but its published model stack does not by itself establish a directly comparable 1.3B T2V contract. fileciteturn15file0L2-L2

Thus these benchmarks should be used to **cross-check metric choices and failure categories**, not to claim that a 17-frame LatentGate run “beats” a model operating with a first-frame image, larger backbone, different trajectory representation, and different generation duration.

## T4-safe sequence and falsifiable feedback experiment

The shortest credible route is not “implement the controller and tune until the video looks better.” It is a sequence in which each run answers one binary technical question and avoids paying for expensive policy sweeps before observability and actuator direction have been established.

### Establish the pinned runtime before touching control

Create a machine-readable **R0 manifest** for a no-policy DiTraj run containing: DiTraj commit; Diffusers tag/commit; Wan model ID/revision if available; Torch/CUDA versions; GPU; prompt and negative prompt; seed; exact initial path; width/height/frame count; 30 timesteps; scheduler config; sigma vector; `mask_step`; local `fixRope_step=3`; guidance scale; dtype of VAE, transformer, latents and model output; wall time; and peak allocated/reserved CUDA memory.

This run is not another “M0 result.” It is a reproducibility/runtime artifact. DiTraj's released runner demonstrates what needs pinning but does not supply the free-T4 measurements. fileciteturn9file0L2-L2

### Verify predicted-clean capture without decoding it

In **R1**, make no control update. At zero-based step 4, record to CPU the current sample \(x_t\), model output \(v_\theta\), scheduler `step_index`, sigma, and independently computed

\[
\hat x_0=x_t-\sigma v_\theta.
\]

Compare the result numerically with the output of the pinned scheduler's conversion logic. This directly checks that LatentGate's observation hook is attached at the correct point. The Diffusers source justifies the formula; only R1 can verify the running configuration. fileciteturn6file0L2-L2

The instrumentation-only run should leave the final generation unchanged up to the deterministic numerical tolerance of the same environment. A material output change at R1 is a stop condition because the “observer” is then perturbing the plant.

### Measure one VAE observation, separately from control

In **R2**, decode exactly one captured step-4 predicted-clean tensor using the same Wan latent de-normalization and VAE contract as final output. Initially, the safest implementation is to **copy/capture the tensor and decode it after the main denoising run**. That establishes whether the intermediate prediction visually exposes the object's position without inserting extra peak memory into the denoising loop.

Only after that works should one time an actual in-loop step-4 decode. Record incremental wall time and peak memory. The result is the missing evidence for the currently unmeasured VAE cost.

If a single in-loop VAE observation causes T4 OOM or makes total runtime operationally unacceptable, do not shrink the experiment by silently changing resolution or backbone. Freeze the result as **“RGB observer infeasible under the current T4 budget”** and move the observer effort to spatial DiT features.

### Test observability before closing the loop

In **R3**, generate a small open-loop dataset containing mirrored commands and multiple seeds. Capture the step-4 predicted-clean state but make no control change. Independently annotate the final 17-frame clip.

The question is:

\[
\text{Does }\hat c_f^{(4)}
\text{ contain useful information about the path error that survives to the final video?}
\]

The first observer should be the most interpretable one: detect/segment the object in the decoded \(\hat x_0^{(4)}\) and estimate its box centers. Do not begin with a trained state head, because that adds a new training problem before the causal feedback question is answered.

A preregistered observability pass could require, over visible/confident frames, **median state-to-final center discrepancy ≤0.08 of image diagonal, signed horizontal/vertical error-direction agreement ≥80%, and confidence coverage ≥70%**. Those thresholds are LatentGate engineering criteria, not published benchmarks. What matters scientifically is that they are set before the feedback outputs are viewed.

If step 4 fails, inspect later observation points under the **same no-policy runs** before attempting feedback. This is where step 14 belongs initially: as an observability diagnostic, not immediately as a second control action.

### Identify the actuator without using state

Before choosing any feedback gain, run **R4**, an open-loop actuator experiment. After step 4, update `active_bbox` by a small predetermined \(+\delta_x\) in one run and \(-\delta_x\) in the matched run, rebuild the cross-attention mask from step 5 onward, and leave everything else identical.

This establishes three things at once:

1. the new persistent-state/mask-rebuild adapter actually affects generation;
2. the sign of the actuator is known;
3. a local empirical sensitivity \(K \approx \Delta c_{\rm final}/\delta\) can be estimated.

Only then choose a feedback gain. A conservative initial controller can target perhaps one-quarter to one-half of the measured residual correction while retaining a hard displacement clamp. That is a proposed design procedure, not an existing gain.

If \(+\delta\) and \(-\delta\) do not produce opposite signed effects reliably, **do not run a feedback policy**. The actuator path is not sufficiently identified.

### Run one step-4 policy intervention

Only **R5** is the actual LatentGate policy experiment.

For every seed/path:

```text
0. Initialize command_bbox, hard_rope_bbox and active_bbox identically.
1. Denoise steps 0..2 with immutable hard-RoPE geometry.
2. Continue normal DiTraj cross-attention from active_bbox.
3. At step 4, estimate predicted-clean object path and confidence.
4. Compute signed residual command_bbox - observed_bbox.
5. If confidence and deadband gates pass:
       translate active_bbox using fixed preregistered gain/clamp;
   otherwise:
       active_bbox remains unchanged.
6. Rebuild cross-attention mask from active_bbox at step 5 and every later step.
7. Perform no additional policy update.
8. Evaluate only the final 17-frame RGB clip with the independent evaluator.
```

This is a **denoising-time feedback controller**. It should not be described as streaming video control: every observation at step 4 is a prediction of the whole 17-frame clip, and the action changes a later denoising iteration of that same clip. That distinction separates it cleanly from MotionStream/4DStreamCtrl's video-time causal generation. citeturn24search5turn24search6

### Matched-compute controls

The causal test requires more than “policy versus baseline.”

**Matched open-loop control:** run the same state extraction/detection workload at step 4 but force the bbox update to zero. Thus any improvement cannot be attributed simply to an extra VAE decode, synchronization, or other computation.

**Zeroed-residual control:** execute the complete observer but replace the estimated residual with zero immediately before the policy map.

**Shuffled-state control:** permute state estimates among seeds while keeping each run's own command. The marginal state distribution and compute stay similar, but state/action causality is broken.

**Fixed-action control:** apply the same distribution of bbox shifts as the policy, but choose them independently of the observed state. This distinguishes “some late perturbation helps” from “feedback helps.”

**Direction-reversal control:** use the same start/noise and mirrored path. The residual and action should change sign in a directionally sensible manner.

**Fixed schedule:** all arms retain exactly 30 denoising steps, the same transformer-call count, scheduler timesteps, CFG settings, initial noise and early hard-RoPE schedule. A method that receives an extra transformer evaluation must be compared against a baseline receiving the same extra evaluation; the proposed first probe avoids that confound.

Only after R5 succeeds should **step 14** be tested as a timing ablation, using exactly the same one-update controller and matched compute. Do not add a step-4 plus step-14 two-update policy at the same time; that would conflate timing and update count.

## Pass/stop gates and required next artifacts

The following gates are proposed **before any LatentGate policy result exists**. They are intentionally hard enough to distinguish a causal controller from a visually attractive anecdote.

### Measurement gate

The independent evaluator must first pass the synthetic known-truth fixture with **center MAE ≤1% image diagonal, mean bbox IoU ≥0.90, and no systematic center bias under the rolling-sphere case**. A surface-point tracker is explicitly allowed to fail the center criterion on the rolling sphere; that is evidence that the fixture is testing the intended distinction.

**Stop:** if the evaluator cannot separate object-center translation from surface rotation, do not score generated clips.

### Runtime gate

R0/R1 must produce a complete pinned manifest, exact scheduler state at step 4, and a predicted-clean reconstruction numerically consistent with the v0.33.1 scheduler's flow-prediction conversion. fileciteturn6file0L2-L2 fileciteturn7file0L2-L2

**Stop:** if instrumentation changes the no-action final output materially, or if the runtime scheduler/config differs from the audited assumptions, repin before proceeding.

For the RGB observer, record actual incremental T4 memory and time rather than imposing a literature-derived estimate. **Stop the VAE-observer branch** if the single step-4 decode OOMs. If it more than doubles end-to-end wall time, retain it for observability research but treat spatial-feature observation as the preferred route for a practical native controller. The “2×” threshold is an engineering budget decision, not a claim about Wan.

### Observability gate

Before policy tuning, the step-4 state must predict final object-state error usefully across multiple seeds and both mirrored directions. A concrete preregistration is:

\[
\mathrm{median}\;\|\hat c^{(4)}-c_{\rm final}\|/D \le 0.08,
\]

with signed error direction agreement of at least **80%** and detector-confidence coverage of at least **70%** of relevant observations.

**Stop:** if step-4 state is not predictive, do not tune a feedback gain around it. Evaluate step 14 or spatial DiT features as an observer-only experiment.

This is where Geometry Forcing is useful conceptually: its success with geometry-aligned intermediate video-model representations supports investigating feature-level state when RGB predicted-clean decoding is inadequate, but it does not prove that Wan2.1's unmodified features already satisfy the LatentGate observability requirement. citeturn24search18

### Actuator gate

The open-loop \(+\delta/-\delta\) bbox perturbation after step 4 must produce **oppositely signed final center displacement in at least 75% of matched pairs**, with median induced displacement of at least **0.02 image diagonal** while preserving one-object visibility/identity.

**Stop:** if the sign is not reproducible. First debug persistent `active_bbox` state and per-step cross-attention rebuilding. Do not compensate by increasing gain; that would turn an unverified actuator into a destructive one.

This gate is especially important because the pinned DiTraj audit proves that a naive callback-only mutation would fail structurally. fileciteturn8file0L2-L2

### Feedback gate

The confirmatory set should contain at least **eight seeds with both mirrored directions**, yielding 16 matched command cases, subject to the T4 budget. Compare policy and matched open-loop runs seed-by-seed.

A persuasive first pass requires:

- **≥20% median reduction in penalized center MAE** relative to matched open-loop DiTraj;
- a **paired bootstrap 95% confidence interval for improvement entirely above zero**;
- no increase in missing/identity-failure frames;
- no more than a **10% relative worsening** in preregistered size-drift and background-drift guardrails;
- true-state feedback outperforming the zeroed- and shuffled-state controls, with **at least half of the nominal improvement disappearing under shuffled state**;
- the signed controller response reversing appropriately on at least **75% of same-noise mirrored pairs**.

These percentages are proposed project gates, not literature standards. Their purpose is to make the result falsifiable in advance.

For background drift, score pixels outside a dilated union of the object's gold boxes using a fixed temporal/background difference metric. For quality, retain both an automated artifact measure and blind pairwise human review; neither should be allowed to override the primary path metric post hoc. Size, background, visibility and identity guardrails prevent a controller from winning by enlarging the object, moving the camera/background, duplicating the target, or destroying it.

**Stop and reject the feedback claim** if the true-state advantage remains equally strong with shuffled state. That result would indicate that the benefit arises from the existence of a late perturbation or some correlated nuisance variable, not from measured state feedback.

Likewise, **stop and reject a control-improvement claim if only the existing M0/red-pixel proxy looks better**. The present single M0 clip is insufficient for any paired statistical comparison, and red-pixel segmentation is not an independent object-state measurement.

### Step-14 gate

Step 14 becomes a later timing ablation only after the step-4 policy passes the actuator and feedback gates. Run a **one-update-at-step-14** controller with the same observer, gain-selection rule, bounds, seeds, paths and compute accounting.

Three outcomes are informative:

- step 4 wins: early state already contains actionable geometry and leaves enough denoising horizon for correction;
- step 14 wins: later state is more observable and still sufficiently controllable;
- neither wins under matched controls: the core feedback hypothesis is unsupported for this actuator/observer, even if hand-selected videos occasionally improve.

Do not add both updates until that timing question is answered.

### Exact next artifacts

The next project artifacts should therefore be produced in this order:

| Artifact | Required content | What it unlocks |
|---|---|---|
| **`runtime_manifest.json`** | Exact DiTraj/Diffusers/Wan revisions, scheduler config/timesteps/sigmas, Torch/CUDA/GPU, prompt/path/seed, dtypes, peak VRAM, wall time | Establishes the experiment that actually ran. |
| **`ditraj_feedback_patch.diff`** | Persistent `active_bbox`; immutable hard-RoPE bbox; `fixRope_step=3`; cross-mask rebuilding every relevant step; explicit step-4 update hook | Makes the audited adapter inspectable. |
| **`step4_capture.npz`** | `x_t`, model output, sigma, step index, computed \(\hat x_0\), shape/dtype summaries | Verifies predicted-clean math against the pinned scheduler. |
| **`vae_cost.json`** | Offline and one-shot in-loop decode time/peak memory on the actual T4 | Resolves the largest current feasibility unknown. |
| **`synthetic_fixture/`** | Fixed camera cube + rolling sphere; exact masks, centers, bboxes, pose, surface-point tracks | Validates the evaluator and sphere center/point distinction. |
| **`labels_v1.jsonl`** | 17-frame visibility, identity, bbox and center gold labels, independent of controller | Replaces red-pixel screening with auditable measurements. |
| **`observability_step4.csv`** | Intermediate state versus final state over seeds and mirrored commands; confidence values | Determines whether a policy run is justified. |
| **`actuator_pm_delta.csv`** | Same-noise \(+\delta/-\delta\) open-loop perturbations and final signed displacement | Supplies the first empirical control sensitivity and permits selecting a gain. |
| **`policy_preregistration.yaml`** | Fixed gain derived from actuator test, confidence threshold, deadband, clamp, metrics, path set, seeds, all pass/stop gates | Prevents outcome-driven tuning. |
| **`R5_matched_results.csv`** | Open-loop, true-state, zeroed-state, shuffled-state, fixed-action and mirrored cases with compute accounting | Supports or falsifies the first actual LatentGate feedback claim. |

The evidence-weighted ordering is therefore **measurement → pinned observation → observability → actuator identification → one step-4 feedback update → state ablations → step-14 timing ablation**. Training a new policy, adding repeated feedback updates, substituting a 5B/14B controller, or moving to full streaming generation should come only after that chain succeeds.

The broader literature actually strengthens this conservative path. By October 2026, trajectory control itself is crowded; online and geometric control are advancing rapidly; and increasingly sophisticated systems already use learned point tracks, three-dimensional state, causal students, geometric feature alignment and inference-time guidance. citeturn24search6turn24search12turn24search18turn24search24 What remains experimentally valuable for LatentGate is not another trajectory-conditioned demo. It is a small, version-pinned experiment that can show—under same noise, same schedule, matched computation, independent state measurement, and destructive state ablations—that **the model's own intermediate estimated state contains information whose signed feedback causally improves later path adherence**. At present, that remains a hypothesis.