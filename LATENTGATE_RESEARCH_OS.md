# LATENTGATE
## A Research Operating System for Latent-Steered Synthetic Robot Video Generation

**Version:** 0.4  
**Date:** 2 October 2026  
**Audience:** Codex / research agents / human project lead  
**Status:** hypothesis-driven research specification, not a fixed architecture

---

# 0. Executive statement

The project is not “make a better video generator.”

It is an experimental program for answering:

> **Can we expose, diagnose, and causally steer the internal spatiotemporal computation of a general-purpose video generator using modular policies distilled from specialist geometry, tracking, motion, camera, contact, and physics models—and does the resulting improvement make synthetic video materially more useful as robot-training data?**

The project should preserve the broad priors of strong pretrained video models and avoid prematurely turning the base generator into a narrowly trained robotics-only model.

The preferred progression is:

1. **Autopsy the generator.**
2. **Locate useful internal representations.**
3. **Build one policy at a time.**
4. **Measure each policy in isolation.**
5. **Measure policy interference.**
6. **Compose only policies whose isolated effects are understood.**
7. **Measure downstream robot-data utility.**
8. **Only then decide whether deeper architectural changes or retraining are justified.**

The research system must tolerate strange engineering ideas, but every idea must become a falsifiable experiment.

---

# 1. Consolidated problem definition

Robot-manipulation video generation exposes failure modes that ordinary video benchmarks can tolerate but robot learning cannot.

The core failure families are:

- frame-to-frame inconsistency,
- depth-order violations,
- object identity drift,
- object geometry drift,
- object orientation drift,
- robot geometry / pose drift,
- unstable or implausible trajectories,
- camera jitter,
- camera/object-motion entanglement,
- grasp-point drift,
- penetration / impossible contact,
- object motion without a corresponding cause,
- support violations,
- physically implausible collision response,
- temporal cause/effect reversal,
- blur and visual instability that corrupt tracking,
- action-video mismatch,
- visually plausible motion that is not robot-executable.

The downstream objective is **not** just photorealism.

The synthetic videos should eventually serve as one or more of:

- world-model training data,
- inverse-dynamics / latent-action training data,
- visuomotor policy data,
- trajectory priors,
- robot-policy augmentation,
- rare-state / edge-case augmentation,
- evaluation data,
- failure recovery data.

The ultimate research question is therefore:

> **At what quality, diversity, and physical-consistency threshold does synthetic video begin to replace or substantially reduce expensive real demonstrations?**

---

# 2. Three core hypotheses

## H1 — Latent observability

Robotics-relevant state variables are at least partially recoverable from intermediate generator representations.

Candidate variables include:

- depth,
- depth ordering,
- 2D/3D point tracks,
- camera pose,
- object identity,
- rigid object pose,
- end-effector pose,
- trajectory,
- impending contact,
- grasp stability,
- future failure probability.

If these variables cannot be recovered before final RGB decoding, latent steering may not be the correct mechanism.

---

## H2 — Latent correctability

Some internal representations are not merely correlated with final success/failure; intervening on them causally changes the generation outcome.

This must be established with:

- activation patching,
- token masking,
- latent perturbation,
- attention intervention,
- branch-and-prune,
- gradient steering,
- small learned residual controllers.

A good diagnostic representation is not automatically a good control surface.

---

## H3 — Synthetic substitution

As physical consistency, task fidelity, and diversity improve, synthetic trajectories should begin to substitute for a portion of real demonstrations at matched downstream robot performance.

This should be measured rather than assumed.

---

# 3. Correct mental model of “breaking the video model into chunks”

A modern latent video diffusion / flow model generally does not independently emit final pixels.

A useful abstraction is:

```text
text / first frame / other conditions
              ↓
      condition encoders
              ↓
 initial noisy video latent
              ↓
 denoising / flow step t
              ↓
 video transformer block 0
              ↓
            ...
              ↓
 video transformer block L
              ↓
 predicted update / clean estimate
              ↓
        next latent state
              ↓
            ...
              ↓
         video VAE
              ↓
          RGB video
```

The model therefore exposes several possible “chunks”:

In this section, “chunks” is a measurement metaphor for internal states and token groups. It does not require splitting a generated clip into independent frames, changing the model's output format, or adopting an autoregressive architecture. The first experiments use the model's normal full-sequence generation and inspect its internal trajectory. Literal chunk-by-chunk generation is an optional architecture branch only if a measured horizon, memory, or control need makes it useful.

### 3.1 Denoising latent
The spatiotemporal latent tensor \(z_t\) at a generation timestep.

### 3.2 Video tokens
Patchified latent cells processed by the transformer.

### 3.3 Hidden states
Representations after each transformer block.

### 3.4 Attention
- video-token self-attention,
- text/image-condition cross-attention,
- where supported, image-conditioning attention.

### 3.5 Predicted-clean latent
Scheduler-dependent estimate of the clean video latent at the current step.

### 3.6 Decoded preview
A selected predicted-clean latent decoded through the video VAE.

### 3.7 Structured interpretation
External models applied to the preview:
- depth,
- pointmaps,
- tracks,
- camera,
- optical flow,
- masks,
- pose,
- contact geometry.

The project's central diagnostic space should be thought of as:

```text
denoising timestep
× transformer depth
× video time
× spatial location
× feature channel
```

This is the microscope.

---

# 4. Do not immediately “change the output head”

The first version should not rewrite the foundation model.

Instead, build a **Generation Packet API** around it.

At selected denoising checkpoints, the generator should be able to emit:

```json
{
  "run_id": "...",
  "seed": 42,
  "denoise_step": 17,
  "sigma_or_timestep": 0.0,
  "latent": "...",
  "predicted_clean_latent": "...",
  "decoded_preview": "...",
  "selected_hidden_states": {
    "block_4": "...",
    "block_12": "...",
    "block_20": "..."
  },
  "attention_summary": "...",
  "teacher_outputs": {
    "depth": "...",
    "tracks": "...",
    "camera": "..."
  },
  "policy_scores": {
    "depth_gate": 0.0,
    "identity_gate": 0.0
  },
  "interventions": []
}
```

This gives us the user's desired “raw chunks” without committing to a new generative architecture.

Only later should we test native multi-output architectures such as:
- RGB + depth,
- RGB + XYZ,
- RGB + pointmaps,
- RGB + segmentation,
- RGB + explicit trajectories.

---

# 5. The universal policy contract

Every latent-steering policy should follow the same conceptual interface:

```text
specialist teacher / geometric source
                ↓
        measurable target state
                ↓
     latent readout / probe / predictor
                ↓
         policy score / diagnosis
                ↓
        intervention / gate / search
                ↓
             generator
                ↓
        independent verification
```

## 5.1 Teacher

Potentially expensive specialist model or privileged signal.

Examples:
- Video Depth Anything,
- VGGT,
- CoTracker3,
- optical flow,
- simulator state,
- known robot kinematics,
- first-frame RGB-D,
- known CAD / point cloud.

## 5.2 Measurable state

The teacher output must be converted into an explicit state:

- depth map,
- ordering relation,
- 3D point track,
- camera pose,
- SE(3) object transform,
- contact event,
- support graph,
- trajectory curvature,
- jerk,
- identity embedding.

## 5.3 Latent readout

Ask whether the video model already represents the target.

Use a hierarchy:
1. linear probe,
2. tiny MLP,
3. small temporal head.

Do not start with a large verifier.

## 5.4 Gate / intervention

Possible gate actions:

```text
PASS
DOWNWEIGHT
CORRECT
BRANCH
RESAMPLE
ANCHOR
PATCH
REPROJECT
BACKTRACK
STOP
```

## 5.5 Independent verifier

Avoid evaluating a policy solely with the same teacher used to train it.

When possible:

```text
teacher A → train policy
teacher B / simulator / ground truth → evaluate
```

---

# 6. Policy Bank

Every policy must first be tested in isolation.

---

## P0 — Temporal consistency

### Failure modes
- flicker,
- sudden local discontinuity,
- inconsistent short-term motion,
- unstable textures.

### Useful prior work
- VideoGuide,
- Track4Gen,
- CoTracker3,
- SEA-RAFT / optical flow.

### State
- point tracks,
- flow,
- feature displacement,
- temporal residual.

### Candidate gates
- temporal hidden-state consistency,
- track-consistency latent loss,
- flow-consistency preview guidance,
- branch-and-prune on track survival.

### Metrics
- point-tracking error,
- track survival,
- optical-flow residual,
- tracked-object acceleration/jerk.

---

## P1 — Depth-order / 3D spatial adherence

### Failure modes
- foreground/background inversion,
- implausible occlusion,
- depth jumps,
- object passes through scene geometry,
- perspective-inconsistent scale.

### Teachers / priors
- Video Depth Anything,
- Depth Anything V2,
- VGGT,
- Geometry Forcing,
- World-consistent Video Diffusion,
- GEN3C,
- Geometry-aware 4D Video Generation.

### State hierarchy

```text
relative depth
    <
metric depth
    <
depth + camera
    <
pointmaps
    <
persistent 3D tracks / scene coordinates
```

Depth alone should not be mistaken for full 3D consistency.

### Candidate gates
- depth-order inequality constraints,
- predicted-depth latent gradient,
- VGGT-feature alignment,
- pointmap reprojection consistency,
- first-frame RGB-D anchor preservation.

### Metrics
- temporal depth-gradient consistency,
- depth-order violations,
- scale drift,
- reprojection residual,
- 3D track residual.

---

## P2 — Object identity / grounding

### Failure modes
- object changes identity,
- texture / color drift,
- object token binds to wrong region,
- target object changes midway,
- manipulated object “melts.”

### Teachers / priors
- CoTracker3,
- Track4Gen,
- DINO-style features,
- segmentation / video segmentation,
- cross-attention analysis.

### State
- object mask,
- object tracks,
- instance embedding,
- token-to-region mapping.

### Candidate gates
- tracked-feature consistency,
- first-frame identity anchor,
- cross-attention spatial anchoring,
- hidden-state patching,
- self-attention neighborhood preservation.

### Metrics
- identity embedding variance,
- segmentation continuity,
- track survival,
- identity switch count.

---

## P3 — Rigid geometry / object orientation

### Failure modes
- mug stretches,
- handle migrates,
- robot link changes length,
- rigid object changes scale,
- impossible orientation changes.

### Teachers / priors
- VGGT,
- first-frame RGB-D,
- known mesh / point cloud,
- rigid tracking,
- GenVid2Robot-style sparse rigid consistency.

### State
- sparse 3D anchors,
- relative pairwise point distances,
- object frame,
- 6D pose,
- SE(3) transform.

### Important simplification

Do not require a complete mesh for the first experiment.

A sparse rigid state may suffice:

```text
anchor points
+ depth
+ visibility
+ relative SE(3)
```

### Candidate gates
- rigid point-constellation preservation,
- SE(3)-compatible hidden-state alignment,
- predicted pose correction,
- branch rejection when rigid fit is poor.

### Metrics
- rigid-fit residual,
- pairwise-distance drift,
- orientation continuity,
- scale drift.

---

## P4 — Trajectory smoothness / target adherence

### Failure modes
- jitter,
- teleportation,
- excessive jerk,
- trajectory misses target,
- implausible acceleration.

### Useful prior work
- Tora,
- FreeTraj,
- Motion Prompting,
- trajectory-conditioned video generation.

### State
- 2D / 3D tracks,
- velocity,
- acceleration,
- jerk,
- target waypoint sequence.

**Current first instantiation (2026-10-02):** fixed-camera 2D image-plane path drift of one red rigid object, with a commanded box trajectory and five control waypoints over a 17-frame Wan clip. This is the first P4 slice under the broader latent-steered synthetic-robot-video North Star; it does not replace later 3D pose, interaction, contact, and physical-validity policies. The current open-loop comparator is DiTraj's box/RoPE path guidance. The candidate intervention reads one UniPC step-4 predicted-clean sample, estimates relative path residual, and adjusts later spatial cross-attention boxes while leaving early hard RoPE fixed. See [`research/2d_path_policy_component_plan.md`](research/2d_path_policy_component_plan.md) and [`research/ditraj_feedback_adapter_design_v1.md`](research/ditraj_feedback_adapter_design_v1.md).

**Evidence boundary:** the existing text-prompt-only Wan sample moved its red-mask centroid 32.56 px while the numerical pilot path moves 332.8 px horizontally; this indicates a large visual-proxy mismatch but is not a path-conditioned DiTraj baseline. An offline counterfactual applied the step-4 observer residual to the five target boxes and changed 71 cells in the 52×30 mask (four unlocked boxes shifted right by one cell). Denoising was not resumed with that mask, so this demonstrates only that a real Wan preview can trigger the intended bounded action—not that the video responds or improves. Results: [`research/m0_step4_feedback_mechanism_diagnostic_v1.json`](research/m0_step4_feedback_mechanism_diagnostic_v1.json).

### Candidate gates
- trajectory token conditioning,
- attention intervention,
- latent gradient toward target path,
- branch-and-prune,
- minimum-jerk regularization when appropriate.

### Metrics
- path error,
- 3D trajectory error,
- acceleration spikes,
- jerk,
- DTW to intended trajectory.

---

## P5 — Camera motion

### Failure modes
- camera jitter,
- artificial zoom,
- camera motion confused with object motion,
- background flow inconsistent with camera pose.

### Useful prior work
- AC3D,
- Latent-Reframe,
- GEN3C,
- VGGT.

### Important clue

AC3D reports that an unconditional video DiT internally contains camera-pose information and that it is concentrated in only part of the network.

This is a template for this entire project:

> localize a representation → intervene only where it exists.

### State
- intrinsics,
- extrinsics,
- global camera trajectory,
- background flow.

### Candidate gates
- camera-carrying-layer conditioning,
- latent reprojection,
- time-aware point-cloud reframing,
- background-flow stabilization.

### Metrics
- camera-pose jitter,
- reprojection error,
- background-flow residual,
- camera acceleration.

---

## P6 — Grasp point / contact persistence

### Failure modes
- grasp point moves after contact,
- gripper penetrates object,
- object moves before contact,
- relative gripper/object pose drifts while carrying.

### State machine

```text
PRE_CONTACT
→ CONTACT_ESTABLISHED
→ CONSTRAINED_MOTION
→ RELEASE
```

### State
- gripper pose,
- object pose,
- surface distance,
- grasp frame,
- relative gripper-object transform.

### Candidate gates

Before contact:
- enforce approach consistency.

At contact:
- establish object↔gripper relation.

During carry:
- anchor relative transform.

After release:
- remove coupling.

### Metrics
- contact timing,
- penetration,
- grasp-point drift,
- post-contact relative-transform drift,
- unsupported pre-contact object motion.

---

## P7 — Causal interaction / support

### Failure modes
- object moves with no visible cause,
- effect precedes contact,
- unsupported objects float,
- collision produces no corresponding response.

### Useful prior work
- InterDyn,
- physics-guided object-interaction video work,
- explicit simulator signals.

### Key recommendation

Begin with coarse causal constraints rather than “perfect physics.”

Example:

```text
if |object_translation| > threshold:
    require at least one of:
        gripper contact
        supporting-surface motion
        collision event
        declared external actuator
```

This is easier to verify and steer than generic physical realism.

### State
- support graph,
- contact graph,
- event graph,
- object motion state.

### Metrics
- unsupported-motion count,
- cause/effect ordering,
- contact→motion delay,
- support violations.

---

## P8 — Physical plausibility

### Failure modes
- wrong collision response,
- gravity inconsistency,
- impossible momentum,
- physically impossible rigid-body evolution.

### Development ladder

1. explicit task-local constraints,
2. geometry-based constraints,
3. simulator supervision,
4. specialist learned physical verifier,
5. general learned physics gate only if residual value remains.

### Metrics
Task-specific:
- collision residual,
- trajectory residual,
- support / gravity constraints,
- simulator divergence.

---

## P9 — Multi-view persistent world

Useful once more than one laboratory camera is available.

### Priors
- Geometry-aware 4D Video Generation,
- GEN3C,
- WVD,
- VGGT.

### State
- shared pointmap,
- cross-view tracks,
- camera poses,
- persistent world coordinates.

### Metrics
- cross-view reprojection,
- pointmap alignment,
- object-pose agreement.

---

# 7. Three implementation levels per policy

Every policy should be evaluated at three increasingly ambitious levels.

## Level A — Post-hoc judge

```text
generate full video
→ specialist model
→ score
```

Purpose:
- establish a measurable baseline,
- learn distribution of failures,
- enable best-of-N rejection.

---

## Level B — Online RGB-preview teacher

At selected denoising steps:

```text
latent
→ predicted-clean estimate
→ low-resolution VAE decode
→ specialist teacher
→ reward / structured error
→ intervention
```

Purpose:
- determine the ceiling of teacher-guided inference,
- test whether corrections during generation outperform final rejection.

Cost:
- expensive,
- repeated decoding,
- repeated teacher inference.

---

## Level C — Latent surrogate

Train a small head:

```text
hidden state h(t,l)
→ approximate teacher state / reward
```

Then steer without decoding at every step.

Purpose:
- distill expensive teacher knowledge,
- reduce inference overhead,
- reveal whether teacher knowledge is represented internally.

---

# 8. Depth policy: exact first implementation

The phrase “use Depth Anything as a latent policy” should be made precise.

## D0 — Baseline output analysis

For identical generated clips, compute:
- framewise Depth Anything V2,
- Video Depth Anything,
- VGGT depth.

Compare:
- temporal smoothness,
- cross-model disagreement,
- sensitivity to generated artifacts.

Do not use a single depth model as unquestioned ground truth.

---

## D1 — Intermediate-preview emergence

At selected denoising steps:

1. recover predicted-clean latent where scheduler permits,
2. decode a low-resolution preview,
3. run Video Depth Anything / VGGT,
4. store estimated depth,
5. compare depth evolution over denoising.

Question:

> At what denoising stage does a stable scene-depth hypothesis appear?

---

## D2 — Hidden-state probe

Train:

```text
h(t,l) → depth or depth-order map
```

Sweep:
- denoising step,
- transformer layer.

Produce a matrix:

```text
             layer →
step ↓      0   4   8   12  ...
          --------------------
early
...
late
```

Each cell contains held-out depth prediction quality.

This becomes the **geometry emergence map**.

---

## D3 — Teacher-driven latent correction

At one selected timestep:
- decode preview,
- estimate target depth consistency,
- backpropagate or approximate correction into latent,
- take a small constrained step,
- continue denoising.

Sweep steering strength \(\lambda\).

Measure:
- depth improvement,
- identity damage,
- visual quality damage,
- trajectory effects.

---

## D4 — Latent depth surrogate

Replace repeated teacher calls with the best lightweight hidden-state probe.

If the surrogate reproduces most of D3's benefit, it becomes a real latent policy.

---

## D5 — Escalation criterion

Only if D0–D4 plateau should we test:
- Geometry Forcing-style representation alignment,
- native RGB+depth,
- RGB+XYZ / pointmap generation,
- direct 3D-conditioned model modification.

---

# 9. The policy-strength sweep

No latent gate should be treated as binary.

For policy \(P_i\), define intervention strength \(\lambda_i\).

Example sweep:

```text
λ ∈ {0, 0.05, 0.1, 0.25, 0.5, 1.0}
```

For each value, record:

- target metric,
- visual quality,
- prompt / task adherence,
- all other policy metrics,
- compute cost.

The key object is a **causal response curve**:

```text
policy strength → target improvement → collateral damage
```

The maximal score is not necessarily the correct setting.

The preferred point is usually a Pareto point:
- meaningful target improvement,
- minimal damage elsewhere.

---

# 10. Policy interference matrix

Before composing policies, estimate:

\[
M_{ij} = \Delta(\text{metric}_j \mid \text{policy}_i)
\]

Example:

| Active policy | depth | identity | trajectory | camera | contact | visual |
|---|---:|---:|---:|---:|---:|---:|
| depth | + | ? | ? | ? | ? | ? |
| identity | ? | + | ? | ? | ? | ? |
| trajectory | ? | ? | + | ? | ? | ? |
| camera | ? | ? | ? | + | ? | ? |
| contact | ? | ? | ? | ? | + | ? |

This should be experimentally populated.

It answers:
- does depth steering harm object identity?
- does trajectory steering distort camera motion?
- does identity anchoring prevent necessary rigid rotation?
- does contact anchoring create visual artifacts?

Only after this matrix exists should we compose the full stack.

---

# 11. Policy composition

The final system should probably not add all gradients together blindly.

Candidate composition strategies:

## 11.1 Scheduled policies

Different policies act at different denoising periods.

Example:
- early: layout / camera / global geometry,
- middle: trajectory / object identity,
- later: contact / local geometry / texture stability.

This is a hypothesis, not an assumption.

---

## 11.2 Layer-local policies

Different internal variables may exist in different transformer layers.

AC3D-style probing suggests this may be important.

---

## 11.3 Conditional routing

The orchestrator sees diagnostics and chooses only necessary gates.

```text
if camera_stable and depth_unstable:
    apply depth gate only
```

---

## 11.4 Sequential correction

Apply one gate, recompute diagnostics, then decide the next gate.

This is slower but easier to interpret.

---

# 12. First “autopsy” experiment

## Scene

Single tabletop setup:

- one robot / gripper,
- one mug-like rigid object,
- one target holder,
- fixed camera initially.

## Task

“Pick up the mug and place it into the holder.”

## Generate

Start with 50–100 seeds.

Keep:
- same prompt,
- same first frame where using I2V,
- same scheduler,
- controlled seed sweep.

## Save

At ~8 denoising checkpoints:
- latent,
- predicted-clean latent,
- low-res preview.

At ~6–8 transformer blocks:
- hidden state summary,
- selected region features,
- optionally attention summaries.

External:
- depth,
- tracks,
- camera,
- object mask,
- object trajectory,
- gripper trajectory.

## Label

Each clip may have multiple failure labels:

- depth,
- identity,
- geometry,
- trajectory,
- camera,
- grasp/contact,
- causal,
- visual-only.

## First scientific output

A layer × timestep map answering:

> **When can each future failure be predicted from internal state?**

---

# 13. Causal autopsy

Correlation is insufficient.

Run matched-seed interventions.

## 13.1 Token masking

Mask candidate object-region tokens and observe which output properties change.

## 13.2 Activation patching

Take:
- good run,
- bad run.

At the same prompt / initial condition:
- transplant selected hidden states from good → bad,
- continue generation.

Measure:
- target improvement,
- unrelated changes.

## 13.3 Layer-local perturbation

Inject controlled noise at one layer and one timestep.

Build causal sensitivity maps.

## 13.4 Attention editing

Manipulate:
- text→object binding,
- self-attention temporal correspondence.

## 13.5 Branch-and-prune

At a selected state:
- fork \(K\) continuations,
- cheaply score,
- retain top branches.

LatSearch makes this especially important as a baseline because latent reward-guided search already exists.

---

# 14. What counts as a successful policy?

A policy is accepted only if:

1. it improves its intended held-out metric,
2. improvement survives across seeds,
3. improvement survives unseen objects or layouts,
4. it causes limited degradation in unrelated metrics,
5. it works using an independent evaluator,
6. it has measurable causal effect, not just correlation,
7. its compute cost is recorded,
8. its benefit remains after matching inference budget where relevant.

---

# 15. Synthetic-data substitution experiment

The phrase “how much synthetic data can replace real data?” needs an operational definition.

## 15.1 Baseline real-data curve

Train downstream policy using:

```text
R ∈ {R1, R2, R3, R4}
```

real demonstrations.

Obtain:

```text
Performance_real(R)
```

---

## 15.2 Synthetic-quality tiers

Create synthetic datasets using:

- vanilla generator,
- one-policy improved generator,
- multi-policy generator,
- high-confidence filtered subset.

Keep diversity comparable where possible.

---

## 15.3 Mixing grid

For each real-data amount, add synthetic ratios such as:

```text
S/R ∈ {0, 0.5, 1, 2, 4, 8}
```

Do not assume more synthetic data is always better.

---

## 15.4 Equivalent-real-data substitution

For target performance \(P^*\):

```text
R_base(P*) = minimum real demonstrations required without synthetic data

R_mix(P*, S, Q) = minimum real demonstrations required
                  with S synthetic samples of quality tier Q
```

Define substitution fraction:

\[
Substitution(P^*, S, Q)
=
1 -
\frac{R_{mix}(P^*,S,Q)}{R_{base}(P^*)}
\]

This is more interpretable than saying “we generated 100,000 videos.”

---

## 15.5 Synthetic usefulness threshold

Estimate the quality level \(Q^*\) where:

- synthetic data consistently improves held-out robot success,
- marginal synthetic additions remain positive over a useful range,
- benefit generalizes beyond training objects/layouts.

This is the threshold the project ultimately cares about.

---

# 16. Downstream tests

Generation metrics are not terminal metrics.

Final evaluation should include:

### A. Inverse dynamics
Can actions recovered from improved video better reproduce the trajectory?

### B. Policy learning
Does training on improved synthetic trajectories increase robot success?

### C. Real-data substitution
At equal policy success, how many real demonstrations can be removed?

### D. Generalization
Evaluate:
- unseen object instances,
- unseen layouts,
- unseen backgrounds,
- longer horizon,
- contact-rich tasks,
- precision placement.

### E. Failure recovery
Synthetic data should include recoveries, not only perfect nominal trajectories.

---

# 17. Literature-derived lessons already established

The project should not claim novelty for ideas that already exist independently.

## DreamGen
Video world models can generate “neural trajectories”; pseudo-actions can be recovered with latent-action or inverse-dynamics models and used for robot policy learning. DreamGen also reports a positive relationship between world-model benchmark quality and downstream robot performance.

**Our use:** terminal synthetic-data benchmark and baseline pipeline.

---

## GenVid2Robot
Generated video is treated as an uncertain visual motion hypothesis rather than a direct executable demonstration. It validates sparse 2D motion against first-frame RGB-D under a relative rigid \(SE(3)\) model and includes grasp / robot execution constraints.

**Our use:** rigid-geometric verifier design and reminder that visual plausibility ≠ executability.

---

## Geometry-aware 4D Video Generation
Jointly reasons about geometry across views and produces spatiotemporally aligned RGB-D for robot manipulation.

**Our use:** high-end geometry baseline; pointmaps / multi-view consistency.

---

## Geometry Forcing
Aligns intermediate video diffusion features with a geometry foundation model and improves the geometry encoded in the generator.

**Our use:** escalation path if inference-time latent readout/guidance is insufficient.

---

## VGGT
Single model infers:
- camera,
- depth,
- pointmaps,
- 3D tracks.

**Our use:** exceptionally useful geometry teacher / evaluator.

---

## Track4Gen
Adding point-tracking supervision to diffusion features improves temporal consistency and appearance stability.

**Our use:** direct evidence that tracking supervision can improve generation, and a blueprint for identity/trajectory policy design.

---

## AC3D
Probing an unconditional video DiT reveals implicit camera pose, concentrated in only a subset of layers.

**Our use:** blueprint for representation localization and layer-specific gate injection.

---

## VideoGuide
A pretrained video diffusion model can guide another model during early denoising without ordinary fine-tuning.

**Our use:** evidence for teacher-guided inference and timestep-specific intervention.

---

## Tora / Motion Prompting
Explicit trajectories can condition video generation.

**Our use:** trajectory-policy upper bounds and representations for motion control.

---

## FreeTraj
Training-free motion control can be achieved by manipulating noise / attention during video diffusion inference.

**Our use:** baseline for inference-time motion intervention.

---

## Latent-Reframe
Camera trajectories can be enforced by reframing latent codes through time-aware point clouds without full fine-tuning.

**Our use:** strong precedent for geometry-aware direct latent intervention.

---

## GEN3C
Uses a 3D cache built from depth / point clouds to condition world-consistent future video and camera control.

**Our use:** alternative architecture if persistent explicit geometry outperforms implicit latent steering.

---

## World-consistent Video Diffusion
Joint RGB + XYZ diffusion is a direct explicit-3D alternative.

**Our use:** native structured-output baseline.

---

## InterDyn
Video diffusion can model controlled interactive dynamics between driving entities and surrounding objects.

**Our use:** interaction / causal-dynamics prior.

---

## MAGI-1
Autoregressively generates fixed-length video chunks and denoises chunks causally.

**Our use:** serious alternative to “whole-clip latent diffusion” for gate-at-chunk-boundary experiments.

---

## NOVA
Non-quantized autoregressive video generation.

**Our use:** architecture-skeptic branch exploring whether causal generation surfaces are more natural for policies than diffusion timesteps.

---

# 18. Architecture decision tree

Do not debate architectures abstractly forever.

Run evidence-based branch points.

## Branch A — standard latent video diffusion / flow model

Use if:
- instrumentation is easy,
- internal probes are predictive,
- interventions work.

Advantages:
- mature open tooling,
- natural denoising intervention axis,
- existing guidance literature.

This should be the initial path.

---

## Branch B — explicit 3D augmented diffusion

Consider if:
- geometry is not reliably recoverable,
- depth / pointmap teacher corrections help,
- but latent-only surrogate fails.

Candidates:
- Geometry Forcing-style feature alignment,
- RGB+XYZ,
- RGB+pointmap,
- 3D cache.

---

## Branch C — autoregressive / chunked generation

Consider if:
- long-horizon causal consistency is the major failure,
- policy decisions naturally occur between chunks,
- rollback / re-generation is important.

MAGI-1 is especially relevant because “video chunk” is a first-class generation unit.

---

## Branch D — removing the VAE

Do **not** make this an early objective.

The VAE may introduce:
- information compression,
- geometric distortion,
- local reconstruction limitations.

But removing it destroys a major efficiency advantage and changes the whole problem.

First measure:

```text
input video
→ VAE encode
→ VAE decode
```

and quantify how much:
- depth,
- rigid geometry,
- small-object detail,
- gripper geometry,
- point tracks

are already degraded by the VAE alone.

If the VAE is a dominant source of robotics error, then test:
- stronger VAE,
- geometry-aware VAE,
- less compression,
- alternative latent representation,
- pixel-space branch.

Do not assume the VAE is the bottleneck.

---

# 19. Model selection

## Microscope model — Wan2.1 T2V 1.3B

Use first because:
- open,
- relatively small,
- Diffusers support,
- transformer block access,
- self/cross-attention,
- enough depth for layer probing,
- practical repeated ablations.

Important implementation detail:
the 1.3B configuration in Diffusers conversion uses approximately:
- 30 transformer layers,
- 12 attention heads,
- 128 head dimension,
- 16 latent channels,
- patch size 1×2×2.

Do not accidentally use the default 14B-like configuration when building hooks.

---

## Second model — image-conditioned model

Robotics eventually needs:

```text
current observation → plausible future
```

Candidate routes:
- Wan I2V / TI2V,
- LTX family,
- Cosmos Predict video-to-world,
- another open I2V backbone.

Choose based on:
- accessible internals,
- compute,
- reproducibility,
- first-frame fidelity.

---

## Architecture-skeptic controls

At later milestones:
- MAGI-1 chunked autoregressive diffusion,
- NOVA non-quantized autoregression.

These are not initial implementation targets.

---

# 20. Four-agent research operating system

There should be **four concurrent slots**.

A fifth perspective—the architecture skeptic—must periodically replace one worker rather than becoming a permanent fifth agent.

This prevents uncontrolled parallelism while retaining adversarial review.

---

## Agent 0 — ORCHESTRATOR / SCIENTIFIC CONTROLLER

### Mission
Maintain the global project state and choose the next falsifiable experiment.

### Responsibilities
- hypothesis registry,
- experiment DAG,
- evidence ledger,
- ablation discipline,
- reproducibility,
- conflict resolution,
- stopping criteria,
- final decisions.

### It must never
- silently turn hypotheses into facts,
- merge policies before isolated testing,
- accept a new model because it is fashionable,
- use one metric as universal success.

### Main output
`STATE.md` + experiment tickets.

---

## Agent 1 — LITERATURE / OPEN-SOURCE RESEARCHER

### Mission
Continuously map the frontier around the currently active failure mode.

### Search dimensions
For every topic:
- papers,
- open code,
- released weights,
- datasets,
- licenses,
- exact architecture,
- training requirement,
- inference requirement,
- evaluation method,
- failure modes,
- reusable modules,
- novelty conflict.

### Evidence card

```json
{
  "paper": "",
  "year": 2026,
  "problem": "",
  "core_mechanism": "",
  "where_it_intervenes": "",
  "requires_training": true,
  "open_source": true,
  "weights_available": true,
  "license": "",
  "reusable_module": "",
  "relevant_to_policy": [],
  "evaluation": [],
  "limitations": [],
  "novelty_collision": "",
  "source_urls": []
}
```

### Main output
`research/cards/*.json`  
`research/literature_map.md`

---

## Agent 2 — AUTOPSY / INSTRUMENTATION ENGINEER

### Mission
Turn the base generator into a transparent experimental specimen.

### Responsibilities
- deterministic generation,
- tensor-shape map,
- denoising callbacks,
- hidden-state hooks,
- attention hooks,
- predicted-clean decoding,
- activation storage,
- patching / perturbation,
- visualization.

### Required output
Every experimental run must be reproducible from:
- config,
- prompt,
- seed,
- model revision,
- code commit.

### Main output
`autopsy/` + run artifacts.

---

## Agent 3 — POLICY TRANSLATION / INTEGRATION ENGINEER

### Mission
Translate literature mechanisms and specialist teachers into isolated LatentGate policies.

### Responsibilities
- define target state,
- build teacher adapter,
- train probe,
- build gate,
- sweep strength,
- run causal ablation,
- compare against post-hoc and best-of-N baselines.

### Main output
`policies/<policy_name>/`

Each policy folder must include:

```text
README.md
teacher.py
probe.py
gate.py
metrics.py
configs/
results/
```

---

# 21. Architecture skeptic / observer mode

Every major milestone triggers an independent observer pass.

The observer should temporarily replace Agent 1 or Agent 3.

It must ask:

1. Are we steering the wrong representation?
2. Is the VAE destroying the information first?
3. Would explicit geometry be simpler?
4. Would autoregressive chunk generation be easier to control?
5. Are we confusing diagnostic correlation with causal controllability?
6. Is the “policy” merely reproducing post-hoc filtering at higher cost?
7. Does the improvement survive matched compute?
8. Are evaluation models sharing the same failure biases as teachers?
9. Could a simple deterministic constraint outperform the learned gate?
10. Does any downstream robot metric improve?

Its job is to kill weak directions early.

---

# 22. Agent communication protocol

Do not use free-form “I looked at some papers” handoffs.

Each worker emits structured artifacts.

## Experiment proposal

```json
{
  "id": "EXP-D3-001",
  "hypothesis": "",
  "independent_variable": "",
  "controls": [],
  "metrics": [],
  "success_threshold": "",
  "failure_interpretation": "",
  "compute_budget": "",
  "required_artifacts": []
}
```

## Experiment result

```json
{
  "id": "EXP-D3-001",
  "status": "complete",
  "results": {},
  "confidence": "",
  "unexpected_observations": [],
  "supports": [],
  "falsifies": [],
  "next_recommended_experiment": ""
}
```

## Orchestrator rule

No major branch should proceed solely from qualitative video inspection.

---

# 23. Experiment registry

Every experiment must record:

- model + exact revision,
- scheduler,
- prompt,
- initial frame,
- seed,
- frame count,
- resolution,
- policy,
- policy strength,
- active timestep range,
- active layer range,
- teacher version,
- evaluator version,
- wall-clock time,
- peak VRAM,
- output hashes.

---

# 24. Repository layout

```text
latentgate/
├── README.md
├── STATE.md
├── hypotheses/
│   └── registry.yaml
├── research/
│   ├── literature_map.md
│   └── cards/
├── configs/
│   ├── models/
│   ├── scenes/
│   └── experiments/
├── generation/
│   ├── base_pipeline.py
│   ├── deterministic.py
│   └── generation_packet.py
├── autopsy/
│   ├── latent_hook.py
│   ├── hidden_state_hook.py
│   ├── attention_hook.py
│   ├── predicted_clean.py
│   ├── decode_preview.py
│   └── activation_patch.py
├── teachers/
│   ├── depth/
│   ├── geometry/
│   ├── tracking/
│   ├── flow/
│   └── robot/
├── policies/
│   ├── depth/
│   ├── identity/
│   ├── trajectory/
│   ├── camera/
│   ├── contact/
│   └── causal/
├── metrics/
│   ├── geometry/
│   ├── tracking/
│   ├── trajectory/
│   ├── contact/
│   └── video/
├── experiments/
│   ├── registry/
│   └── runners/
├── analysis/
│   ├── emergence_maps/
│   ├── causal_maps/
│   └── policy_interference/
├── synthetic_data/
│   ├── generation/
│   ├── filtering/
│   ├── pseudo_actions/
│   └── mixtures/
└── reports/
```

---

# 25. First execution program

## Phase 0 — Baseline sanity

- run base model,
- pin revisions,
- reproduce seeds,
- inspect VAE reconstruction separately,
- record memory and speed.

Exit criterion:
same seed/config gives identical or understood deterministic behavior.

---

## Phase 1 — Build the microscope

- denoising latent capture,
- predicted-clean decode,
- selected hidden-state capture,
- selected attention summaries,
- Generation Packet serialization.

Exit criterion:
one generation can be replayed and inspected over denoising time and transformer depth.

---

## Phase 2 — Failure atlas

Generate 50–100 controlled runs.

Create labels:
- geometry,
- identity,
- trajectory,
- camera,
- contact,
- causal.

Exit criterion:
enough good/bad examples for at least one binary failure probe.

---

## Phase 3 — Representation emergence

Start with one property:
**depth / geometry**.

Train linear probes over layer × timestep.

Exit criterion:
identify whether and where geometry is decodable.

---

## Phase 4 — Causal intervention

Perform:
- activation patch,
- latent perturbation,
- one guidance method.

Exit criterion:
show one reproducible causal change.

---

## Phase 5 — First policy

Implement the complete:
teacher → probe → gate → independent verifier
pipeline.

Depth or tracking are recommended first.

---

## Phase 6 — Policy strength and interference

For policy 1:
- sweep strength.

Then policy 2:
- sweep strength.

Then build first 2×N interference measurements.

---

## Phase 7 — Synthetic-data pilot

Generate a tiny high-confidence synthetic dataset.

Recover actions using:
- available IDM / latent action method,
- or simulator known actions for the first controlled test.

Train a small downstream policy.

The point is not SOTA.
The point is verifying the end-to-end causal chain:

```text
latent gate
→ better video
→ better pseudo trajectory
→ better robot policy
```

---

# 26. Recommended first three policies

## 1. Depth / geometry
Why:
- measurable,
- multiple strong teachers,
- rich literature,
- likely central to many other failures.

## 2. Tracking / identity
Why:
- direct relevance to appearance drift,
- CoTracker / Track4Gen provide strong priors,
- easy to quantify.

## 3. Trajectory
Why:
- directly robot-relevant,
- useful control literature,
- connects generation to downstream action.

Do not start with general physics.

Contact / causal physics should follow once geometry and tracking are reliable.

---

# 27. High-value negative experiments

Negative results are useful.

Run these deliberately:

### N1
Depth improves but robot trajectory does not.

Interpretation:
depth consistency alone may be insufficient.

### N2
Probe predicts depth but latent steering fails.

Interpretation:
observable ≠ controllable.

### N3
RGB-preview teacher guidance works but latent surrogate fails.

Interpretation:
internal state may not expose teacher knowledge in a simple usable form.

### N4
Best-of-N matches latent steering at equal compute.

Interpretation:
steering may not justify complexity yet.

### N5
VAE reconstruction causes most small-object / gripper corruption.

Interpretation:
move effort toward representation / codec.

### N6
Synthetic video metric improves but downstream robot policy worsens.

Interpretation:
current metric is misaligned with robot utility.

---

# 28. The core human-insight loop

The purpose of the human researcher is not to manually tune every parameter.

The human contribution should be:

1. notice a recurring failure,
2. propose a structured explanation,
3. find an external representation capable of measuring it,
4. ask whether that representation exists internally,
5. devise the smallest intervention,
6. test side effects,
7. decide whether to keep, reject, or escalate the idea.

Example:

```text
Observation:
The mug's handle changes orientation during lift.

Human hypothesis:
The model preserves semantic identity but not rigid object coordinates.

Research:
VGGT / rigid tracking / GenVid2Robot.

Experiment:
Can hidden states predict a sparse rigid object frame?

Intervention:
Preserve pairwise distances / SE(3) anchor across tracked object tokens.

Evaluation:
Does orientation drift fall without freezing legitimate object rotation?
```

This is the project’s intended style of research.

---

# 29. Key scientific distinction

Always separate:

## Observability
Can the internal state tell us something is going wrong?

## Forecastability
Can it tell us before the final failure is visible?

## Correctability
Does an intervention improve the outcome?

## Action specificity
Which intervention should be selected for which failure?

## Downstream utility
Does the corrected output improve robot learning?

A project can succeed at one and fail at the next.

---

# 30. Initial literature map

This is not exhaustive; the research agent must continuously expand it.

### Synthetic robot data
- DreamGen — synthetic neural trajectories from video world models.
- GR00T-Dreams — open pipeline around world-model synthetic trajectory generation.
- Video Generators are Robot Policies — action decoding / video generation for robot policy learning.
- GenVid2Robot — rigid geometric validation of generated-video motion.

### 3D / geometry
- VGGT — camera, depth, pointmaps, 3D tracks.
- Geometry Forcing — geometric feature alignment inside video diffusion.
- Geometry-aware 4D Video Generation for Robot Manipulation.
- World-consistent Video Diffusion — RGB+XYZ.
- GEN3C — 3D cache.
- Latent-Reframe — time-aware point-cloud latent reframing.

### Temporal correspondence
- Track4Gen.
- CoTracker3.
- DIFT / diffusion features.
- optical flow / SEA-RAFT.

### Motion control
- Tora.
- Motion Prompting.
- FreeTraj.

### Inference-time guidance
- VideoGuide.
- LatSearch.

### Camera
- AC3D.
- GEN3C.
- Latent-Reframe.

### Interaction / dynamics
- InterDyn.
- physics-guided object interaction video generation.

### Alternative generator families
- MAGI-1 — autoregressive video chunks.
- NOVA — non-quantized autoregressive video generation.

---

# 31. Reference links / starting sources

- DreamGen: https://research.nvidia.com/labs/gear/dreamgen/
- DreamGen PMLR: https://proceedings.mlr.press/v305/jang25a.html
- GenVid2Robot: https://arxiv.org/abs/2607.09191
- Geometry-aware 4D Video Generation: https://robot4dgen.github.io/
- Geometry Forcing: https://geometryforcing.github.io/
- Geometry Forcing code: https://github.com/CIntellifusion/GeometryForcing
- VGGT: https://vgg-t.github.io/
- VGGT code: https://github.com/facebookresearch/vggt
- Video Depth Anything: https://github.com/DepthAnything/Video-Depth-Anything
- CoTracker: https://github.com/facebookresearch/co-tracker
- Track4Gen: https://hyeonho99.github.io/track4gen/
- AC3D: https://arxiv.org/abs/2411.18673
- VideoGuide: https://videoguide2025.github.io/
- Tora: https://github.com/alibaba/Tora
- Motion Prompting: https://openaccess.thecvf.com/content/CVPR2025/html/Geng_Motion_Prompting_Controlling_Video_Generation_with_Motion_Trajectories_CVPR_2025_paper.html
- Latent-Reframe: https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Latent-Reframe_Enabling_Camera_Control_for_Video_Diffusion_Models_without_Training_ICCV_2025_paper.html
- GEN3C: https://research.nvidia.com/labs/toronto-ai/GEN3C/
- World-consistent Video Diffusion: https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_World-consistent_Video_Diffusion_with_Explicit_3D_Modeling_CVPR_2025_paper.html
- InterDyn: https://openaccess.thecvf.com/content/CVPR2025/html/Akkerman_InterDyn_Controllable_Interactive_Dynamics_with_Video_Diffusion_Models_CVPR_2025_paper.html
- LatSearch: https://arxiv.org/abs/2603.14526
- Wan Diffusers docs: https://huggingface.co/docs/diffusers/main/api/pipelines/wan
- Wan transformer docs: https://huggingface.co/docs/diffusers/main/api/models/wan_transformer_3d
- MAGI-1: https://github.com/SandAI-org/MAGI-1
- NOVA: https://github.com/baaivision/NOVA

---

# 32. Immediate instruction to Codex

Do not begin by implementing all policies.

Start with:

```text
MILESTONE M0:
Create the reproducible Wan2.1-1.3B autopsy harness.

Deliver:
1. deterministic generation script,
2. model/tensor architecture report,
3. VAE round-trip benchmark,
4. denoising latent logger,
5. predicted-clean preview decoder,
6. hooks for selected transformer blocks,
7. serialized Generation Packet,
8. one notebook/report visualizing a complete generation over time.
```

M0 is complete in `m0/runs/wan_t4_2026-10-02_retry5`; all ten recorded gates pass and packet/manifest hashes have been verified. The Orchestrator has reviewed this evidence. M1 planning may begin, but first resolve the generator/conditioning route for a common starting scene and validate independent failure metrics on a small feasibility batch before scaling to 50–100 seeds. M2 and M3 stay gated on M1 data and a predictive failure probe.

The project should grow from measured internal facts, not from architecture imagination alone.
