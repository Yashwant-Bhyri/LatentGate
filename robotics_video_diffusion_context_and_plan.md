# Robotics Synthetic Video Generation Project — Context + Experimental Roadmap
**Date:** 1 Oct 2026

## 1. Project north star

The project is not simply “make prettier robot videos.”

The core research question is:

> **Can we expose, measure, and intervene on the internal spatiotemporal representations of a general-purpose video diffusion model so that generated robot-manipulation videos become more geometrically, physically, and action consistent—without retraining the entire model into a narrow robotics generator?**

The intended downstream value is synthetic robot training data: robot-POV / egocentric manipulation video that is useful for policy learning, inverse-dynamics recovery, world-model training, or evaluation.

The project should preserve the broad priors of a pretrained video generator and add structure primarily through:
- diagnostics,
- verifiers,
- search,
- inference-time steering,
- small adapters/probes,
- and only later targeted fine-tuning if necessary.

## 2. Problem statement distilled from prior discussions

Current generative video models can make visually impressive clips but robotics exposes failures that ordinary video benchmarks often tolerate:

- object identity drift,
- object scale/orientation drift,
- inconsistent depth and scene geometry,
- implausible camera/object motion,
- jitter and blur,
- wrong gripper pose,
- grasping through objects,
- impossible contact,
- bad articulation/DoF behavior,
- tool/object interactions that look plausible frame-by-frame but are inconsistent in 3D,
- temporal “self-correction” where a failure disappears in later frames,
- weak causal grounding between intended action and generated state transition.

The proposed setting is especially laboratory manipulation, where more prior structure is available than in open-world video:
- object meshes,
- point clouds,
- RGB-D observations,
- camera calibration,
- robot kinematics,
- known gripper geometry,
- task descriptions,
- sometimes simulation ground truth.

That makes the lab setting an unusually good place to test whether a general video model can be constrained by structured verifiers.

## 3. What “dismantling the video model” actually means

A modern latent video diffusion / flow-matching model does **not** independently generate individual final pixels.

A simplified pipeline is:

text / image condition
→ text or image encoder
→ initial noisy video latent
→ many denoising / flow-matching steps
→ at each step, a stack of spatial-temporal Transformer blocks
→ predicted update / clean latent estimate
→ VAE decoder
→ RGB video

For Wan2.1-1.3B specifically:
- 3D causal VAE
- latent channels: 16
- spatial compression: ×8
- temporal compression: ×4
- Transformer patch size: 1 × 2 × 2
- model width: 1536
- 12 attention heads
- 30 Transformer layers
- text enters through cross-attention
- the generation loop is exposed through the Diffusers pipeline, including denoising-step callbacks.

A nominal 81-frame, 480×832 generation becomes roughly:
- 21 latent time steps
- 60 × 104 latent spatial grid
- 16 latent channels

After the 1×2×2 Transformer patching, that is about **32,760 spatiotemporal video tokens** per denoising step.

So the useful “atoms” for research are not output pixels. They are:

1. **video latent cells** z_t at each denoising step,
2. **Transformer hidden states** at each layer,
3. **self-attention patterns** between spatiotemporal video tokens,
4. **cross-attention patterns** between text tokens and video tokens,
5. **predicted denoised / clean latent estimates**,
6. **VAE-decoded previews** of intermediate states,
7. gradients or causal effects produced when we perturb any of the above.

This gives a natural 4D diagnostic cube:

**denoising time × Transformer depth × video time × spatial position**

plus channel / feature dimension.

## 4. The first scientific question

Before building steering logic, answer:

> **Where, when, and in what form do geometry, motion, object identity, and contact become represented inside the model?**

This is more important than immediately training another controller.

The first project artifact should therefore be a **Video Diffusion Autopsy / Observatory**.

## 5. Phase I — Diffusion Autopsy

### 5.1 Freeze the generator

Start with a completely frozen model.

Recommended research baseline:
- **Wan2.1-T2V-1.3B** for the lowest-cost, easiest-to-instrument Transformer baseline.
- **LTX-Video 2B** is a strong second baseline because its 2B model supports image-to-video / keyframe conditioning, which is closer to robot-world prediction.
- Later: **Wan2.2 TI2V-5B** for a stronger text+image-conditioned model once the instrumentation works.

Do not start with a 14B model.

### 5.2 Build a deterministic generation harness

For every run store:
- model version,
- prompt,
- conditioning image if used,
- random seed,
- scheduler,
- timesteps,
- guidance values,
- latent tensor checksum,
- output video,
- decoded intermediate previews.

Same prompt + same seed must be reproducible before any interpretation work begins.

### 5.3 Instrument four surfaces

#### A. Denoising-state capture
Save z_t every N denoising steps.

Also reconstruct / estimate an x0-like clean latent when the scheduler permits, and decode selected checkpoints through the VAE.

Goal:
- see when coarse layout appears,
- when object identity stabilizes,
- when motion becomes committed,
- when geometry breaks.

#### B. Block activation capture
Attach forward hooks to selected Transformer layers, e.g.:

0, 4, 8, 12, 16, 20, 24, 29.

Do not save all activations for all layers on every run at first—the storage explodes.

Store sampled / pooled representations for:
- target object region,
- robot/gripper region,
- background,
- selected temporal tracks.

#### C. Attention capture
Capture:
- text→video cross-attention,
- video→video self-attention.

Questions:
- Which prompt tokens bind to the mug / gripper / target location?
- Does an object maintain a coherent attention neighborhood over time?
- Do later frames attend to the earlier object instance or recreate it?
- Does contact cause a change in the interaction structure?

#### D. Causal perturbations
Observation alone is not enough.

Run interventions such as:
- zero one attention head,
- scale one attention head,
- mask a token region,
- swap selected hidden states between two seeds,
- replace the hidden state of an object region with the same region from a “good” generation,
- perturb one latent-time slice,
- strengthen/weaken selected prompt cross-attention.

Then finish generation and measure the effect.

This separates:
- correlation (“this feature lights up”)
from
- causal relevance (“changing this feature changes geometry/contact”).

## 6. Phase II — Build a failure atlas

Do not begin with a broad robotics benchmark.

Begin with one tightly controlled family, e.g.:

**robot POV: pick up one mug and place it into one holder**

Generate many seeds under the same starting condition.

Create good/bad pairs for individual failure axes:

### Geometry
- depth temporal consistency,
- pointmap consistency,
- reprojection error,
- object scale drift,
- camera trajectory consistency.

### Object identity
- feature correspondence across frames,
- segmentation-mask continuity,
- shape embedding stability.

### Motion
- optical flow smoothness,
- acceleration / jerk,
- trajectory deviation,
- sudden teleportation.

### Robot kinematics
- end-effector trajectory,
- joint/pose plausibility,
- wrist orientation,
- gripper aperture.

### Contact
- gripper-object distance,
- penetration,
- grasp-before-lift,
- object motion without contact,
- slip / detach after grasp.

### Visual-only nuisance
- blur,
- flicker,
- temporal texture instability.

The objective is a dataset of:
- generation internals,
- final outputs,
- external physical/geometry scores.

## 7. Phase III — Ask whether the model already “knows” geometry

This is where the project becomes scientifically interesting.

Train tiny probes from hidden states to structured targets.

Examples:

hidden state
→ depth

hidden state
→ pointmap

hidden state
→ object mask

hidden state
→ 2D / 3D keypoint location

hidden state
→ gripper pose

hidden state
→ future failure score

Use a hierarchy:

1. linear probe,
2. tiny MLP,
3. small temporal head.

If a linear probe predicts depth or object pose at an early layer/timestep, the information is already linearly accessible.

If only a nonlinear head succeeds, the information exists but is entangled.

If no probe succeeds until very late, then “verifying in latent space” may not be useful at early steps.

This produces a **layer × denoising-time emergence map**.

That map may become one of the central results of the project.

## 8. Phase IV — Inference-time steering ladder

Do not jump directly to backpropagation through every denoising step.

Use increasingly invasive methods.

### Level 0 — Best-of-N
Generate N complete samples.
Score with external verifier.
Keep the best.

This gives the upper-bound value of the verifier without touching the generator.

### Level 1 — Branch-and-prune during denoising
At selected denoising steps:
- fork K candidate next states,
- predict/decode a low-cost preview,
- score candidates,
- continue only the best branch(es).

This is directly aligned with recent latent-search work.

### Level 2 — Latent reward guidance
Create differentiable or approximately differentiable verifier objectives.

At selected steps:
- evaluate geometry/contact reward,
- obtain gradient with respect to latent,
- make a small update,
- resume normal denoising.

Only intervene at a few informative timesteps, not all of them.

### Level 3 — Attention intervention
Manipulate selected:
- cross-attention maps,
- self-attention neighborhoods,
- object-region token interactions,
- temporal correspondence structure.

This is especially suitable for:
- object persistence,
- prompt binding,
- trajectory anchoring.

### Level 4 — Learned lightweight controller
Train:
- LoRA,
- adapter,
- small latent controller,
- small verifier-conditioned residual.

Keep the base generator frozen.

The controller observes internal state and outputs a correction.

### Level 5 — Representation-level geometry supervision
Only if the previous levels plateau:
- align intermediate features with geometry features,
- jointly predict depth / pointmap,
- add geometry-aware auxiliary losses.

This is where ideas similar to Geometry Forcing or RGB+pointmap generation become relevant.

## 9. The most important research hypothesis

A strong project hypothesis is:

> **Robotics failures are often detectable in intermediate video-diffusion representations before they become visually obvious in the final decoded video; structured geometry/interaction verifiers can exploit those internal states to reject or repair bad denoising trajectories while preserving the pretrained generator’s general visual prior.**

This is stronger and more specific than:
“we will add physics to a video model.”

## 10. What is already known, so we should not claim as novelty

Several important pieces already exist:

### Synthetic robot data from video generators
**DreamGen (CoRL 2025)**:
video world model → synthetic robot videos → pseudo-actions → downstream policy training.

Therefore:
“use generated videos for robot learning” is not novel by itself.

### Geometry-aware robot video prediction
**Geometry-aware 4D Video Generation for Robot Manipulation (ICLR 2026)**:
joint RGB / pointmap forecasting and cross-view geometric consistency.

Therefore:
“add 3D geometry supervision to robot video generation” is not novel by itself.

### Geometry supervision inside a video diffusion model
**Geometry Forcing (ICLR 2026)**:
align intermediate diffusion representations with geometric foundation-model features.

Therefore:
“make diffusion hidden states more 3D-aware” is already an active research direction.

### Joint video + dense outputs
**Unified Dense Prediction of Video Diffusion (CVPR 2025)**:
jointly generate RGB, depth and segmentation.

Therefore:
“generate RGB and depth together” is not novel by itself.

### Training-free trajectory control
**FreeTraj (2024/2026)**:
change initial noise and attention to control trajectory without tuning.

### Learned trajectory conditioning
**Tora (CVPR 2025)**:
trajectory encoding + motion-guidance modules inside a DiT.

### Inference-time teacher guidance
**VideoGuide (CVPR 2025)**:
use a guiding video model during early denoising to improve temporal consistency.

### Latent inference-time search
**LatSearch (2026)**:
score partially denoised video latents and perform reward-guided resampling/pruning.

Therefore:
“score intermediate latents and search” is no longer novel by itself.

## 11. Where the stronger gap appears to be

The promising gap is the intersection:

**mechanistic autopsy**
+
**robotics-specific structured latent verifiers**
+
**causal intervention**
+
**multi-axis geometry/contact/kinematic constraints**
+
**downstream policy utility**

A credible research contribution would be something like:

1. Show where robotics-relevant state variables emerge inside a general video DiT.
2. Show that certain failure modes can be predicted from intermediate hidden states before final decoding.
3. Build an intervention method that operates only at the informative layers/timesteps.
4. Improve geometry/contact/trajectory metrics with minimal change to the generator.
5. Demonstrate that the corrected synthetic trajectories improve downstream robot learning or inverse-dynamics recovery.

## 12. Recommended first model choice

### First model: Wan2.1-T2V-1.3B
Why:
- 1.3B parameters,
- open weights,
- Diffusers integration,
- simple single-denoiser architecture,
- 30 Transformer layers,
- cross-attention,
- denoising-step callbacks,
- low enough cost for repeated ablations.

Use it for:
- instrumentation,
- layer/timestep probes,
- attention experiments,
- causal perturbation experiments.

### Second model: LTX-Video 2B
Why:
- still small,
- supports image-to-video,
- more appropriate for “current observation → future manipulation video.”

Use it after the autopsy code is stable.

### Stronger later model: Wan2.2 TI2V-5B
Use only after:
- logging works,
- metrics are automated,
- you know which timesteps/layers matter.

## 13. First concrete experiment

### Scene
One mug on a table.
One robot/gripper.
One target holder.

### Prompt/action
“Pick up the mug and place it into the holder.”

### Generate
50–100 seeds.

### For every seed collect
- final RGB video,
- denoising latents at ~8 checkpoints,
- hidden states from ~8 layers,
- selected cross-attention maps,
- selected self-attention maps,
- decoded intermediate previews.

### External measurements
- object segmentation,
- depth,
- feature tracks,
- object center trajectory,
- gripper trajectory,
- contact time,
- object-gripper distance,
- temporal jitter,
- optional pointmap / reprojection metrics.

### Label
For each run:
- good,
- geometry failure,
- trajectory failure,
- grasp/contact failure,
- identity failure,
- visual-only failure.

### First analysis
Ask:

> At which layer and denoising timestep can we best predict each failure class?

This is a clean result even before steering exists.

## 14. Second concrete experiment: causal token surgery

Take:
- one good seed,
- one bad seed.

Identify the object-region spatiotemporal tokens.

At one selected layer/timestep:
- transplant or interpolate the object-region hidden states from good → bad,
- then resume generation.

Measure:
- did object geometry improve?
- did trajectory improve?
- did unrelated appearance change?
- does the causal effect depend on layer/timestep?

This can reveal whether the latent representation contains locally controllable object state.

## 15. Third concrete experiment: branch-and-verify

At one early/mid denoising checkpoint:

1. duplicate the current latent K times,
2. make small stochastic continuations,
3. form a low-resolution or predicted-clean preview,
4. score with:
   - depth consistency,
   - object-track consistency,
   - trajectory plausibility,
5. keep the top branch,
6. finish normal sampling.

Baseline comparisons:
- vanilla,
- best-of-N final samples,
- branch-and-verify.

This gives an early test of whether inference-time intervention is worth pursuing.

## 16. What not to do yet

Do not yet:
- train a giant physics model,
- build a universal object ontology,
- collect every lab object,
- fine-tune a 14B model,
- add tactile/contact simulation immediately,
- attempt full downstream VLA training,
- build five verifiers at once,
- claim pixel-level mechanistic interpretability.

The first milestone is much narrower:

> **Can we identify and causally influence one robotics failure mode inside one small open video diffusion model?**

## 17. Suggested repository structure

```text
robot-video-autopsy/
  configs/
  models/
    wan/
    ltx/
  generation/
    baseline.py
    reproducibility.py
  hooks/
    latent_logger.py
    activation_logger.py
    attention_logger.py
  decode/
    intermediate_decode.py
  probes/
    depth_probe.py
    pose_probe.py
    failure_probe.py
  metrics/
    depth_consistency.py
    tracking.py
    trajectory.py
    contact.py
  interventions/
    token_mask.py
    activation_patch.py
    attention_control.py
    branch_and_verify.py
  datasets/
    tabletop_v0/
  analysis/
    emergence_maps.ipynb
    causal_effects.ipynb
  results/
```

## 18. Two-week execution plan

### Days 1–2
- run Wan2.1-1.3B reproducibly,
- understand tensor shapes,
- log scheduler states,
- decode baseline outputs.

### Days 3–4
- implement denoising callback,
- save intermediate latents,
- decode selected checkpoints.

### Days 5–6
- hook selected Transformer blocks,
- pool selected spatial-temporal activations,
- implement cross-attention logging.

### Day 7
- create first “generation autopsy” report for one clip.

### Days 8–9
- build controlled mug/holder dataset,
- generate 50–100 seeds.

### Days 10–11
- automate depth / tracking / trajectory metrics,
- manually validate failure labels.

### Day 12
- train simple failure probes from internal states.

### Day 13
- produce layer × timestep predictability heatmaps.

### Day 14
- perform first causal intervention:
  activation patch, token mask, or branch-and-verify.

Decision gate after Day 14:

**If intermediate states contain predictive signal:** continue to inference-time steering.

**If only final decoded frames contain the signal:** focus on search / rejection / model-based guidance rather than latent mechanistic steering.

**If the 1.3B T2V model is too weak for robot structure:** port the instrumentation to LTX-Video 2B I2V before scaling further.

## 19. Key literature to read first

1. **Wan: Open and Advanced Large-Scale Video Generative Models** — architecture / flow-matching DiT / Wan-VAE.
2. **Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence** — layer+timestep feature extraction.
3. **DIFT / Emergent Correspondence from Image Diffusion** — semantic and geometric correspondence from diffusion features.
4. **What the DAAM** — cross-attention attribution.
5. **Prompt-to-Prompt** — cross-attention intervention.
6. **Attend-and-Excite** — inference-time attention guidance.
7. **FreeTraj** — training-free trajectory steering in video diffusion.
8. **Tora** — explicit trajectory conditioning in a DiT.
9. **VideoGuide** — inference-time guidance during early denoising.
10. **Unified Dense Prediction of Video Diffusion** — RGB + depth + segmentation.
11. **DepthCrafter** and **Video Depth Anything** — external video-depth supervision.
12. **Geometry Forcing** — align video diffusion representations with geometric features.
13. **Geometry-aware 4D Video Generation for Robot Manipulation** — RGB/pointmap robot world prediction.
14. **DreamGen** — generated videos → pseudo-actions → robot policy.
15. **LatSearch** — reward-guided search over partially denoised video latents.

## 20. Working thesis

The project should begin as an **instrumentation and causal-analysis project**, not as a new video generator.

First understand the model.

Then predict failures from its internals.

Then intervene at the smallest effective interface.

Only then scale toward multi-verifier geometry/physics steering and downstream robot-policy training.
