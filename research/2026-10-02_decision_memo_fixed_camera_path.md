# Research decision memo: fixed-camera object path control

**Date:** 2026-10-02  
**Scope:** Reduce image-plane trajectory drift for one visible object in a short, fixed-camera video diffusion sample. No claim is made about 3D pose, depth, contact, or physical validity.

## Decision

Keep **Wan2.1-T2V-1.3B + DiTraj** as the first controlled path. Test whether a small commanded box shift changes the rendered object's center in the expected direction before adding a feedback law. DiTraj is the closest open-source actuator to this problem and directly supports Wan; its action is still a spatial foreground-conditioning change, not a pixel translation. If signed actuation is absent, inconsistent, or damages object identity, stop tuning box feedback and move to a spatial latent/correspondence reader or a different actuator.

This choice agrees across the [global path-control survey](2026-10-02_global_motion_research.md), [open-source stack audit](2026-10-02_open_source_policy_stack.md), and [architecture audit](2026-10-02_same_sample_feedback_architecture_audit.md).

## What the literature contributes

| Research family | Useful contribution | Why it is not yet the first implementation |
|---|---|---|
| Open-loop box/trajectory control: DiTraj, Motion-Zero, TrailBlazer, Tora, Wan-Move | Explicit spatial paths, object/background separation, positional or motion conditioning | Most do not observe the generated state and correct its later trajectory. Tora/Wan-Move use different backbones or training/data regimes. |
| Test-time latent/self-guidance: SG-I2V, Frame Guidance, WorldForge | A state-dependent loss or feature alignment can steer a denoising trajectory; selective updates can protect appearance | The implementations target SVD I2V, other video backbones, camera/3D tasks, or costly gradient loops. They are useful design references but not drop-in Wan-1.3B controls. |
| Internal correspondence/readout: DiffTrack, DreamTraj | Video-DiT Q/K features or early latent features can expose motion state without a full perception pipeline | DiffTrack's released hooks target CogVideoX/Hunyuan, not Wan. DreamTraj's reader is a promising research result but uses a much larger Wan2.2 I2V setup and is not a commanded-path controller. |
| Trained trajectory adapters / teacher-student / reward learning: Tora, Motion Prompting, Wan-Move, DyMD | Could become a compact native module or distilled corrective policy after a reliable reward/state exists | Adds training data, architecture, licensing, and compute variables before the selected model's control action is even calibrated. |

The direct novelty collision is therefore **not** “trajectory guidance exists.” It clearly does. A potential contribution is a measured, confidence-gated state-to-action loop for a named object's path in this exact small T2V stack, with same-prefix controls and independent evaluation. That framing remains provisional until the experiment is run and compared carefully with adjacent work.

## Component order and gates

1. **Exact task and run artifact.** Preserve the neutral prompt, fixed camera, matched start box, mirrored paths, shared initial noise, fixed sampler, and pinned versions. The local [pilot notebook](ditraj_paired_path_pilot_v1.ipynb) is a 12-cell, unexecuted artifact with the 30-step stage disabled by default. The designated Kaggle draft is currently incomplete; do not start its session until its source matches the reviewed local notebook.
2. **Runtime smoke.** Freshly confirm free T4 availability, GPU identity, and memory. Run the bounded one-step six-case smoke first; inspect every output and resource measurement. This is an environment/code check, not a control result.
3. **Measurement reliability.** Score object center, visibility, and identity independently of the commanded path. Human frame labels remain the reference. SAM 2.1 tiny has been measured on the local Kubric mirrors but its ~2.6 px tight-box-center error includes a ~2.5 px vertical bias, which misses the illustrative 8 px Wan half-cell tolerance after scaling; it remains a proposal only. CoTracker3 is an independent surface-point cross-check, not an object-center estimator. SAM 3.1 is a newer gated-GPU candidate but has not been measured on our clips. The observer-specific requirements are in the [observer/benchmark audit](2026-10-02_observer_benchmark_research.md).
4. **Signed action calibration.** With one shared prefix through step 4, compare unshifted boxes and static ±1 latent-cell shifts. Start with horizontal motion, then test vertical motion if horizontal actuation is repeatable. Include both mirrored target directions. Verify the 52×30 control grid against the actual loaded model/VAE config before translating cells to pixels. Begin with a small directional screen; require at least eight paired seed blocks per tested axis before treating its response estimate as calibrated. Use the 95% paired interval for signed response and compare displacement with observer repeatability. Score with independent labels, not a path-selected tracker.
5. **Closed-loop mechanism.** Only after action sign and observer gates pass, run the bounded step-4 residual adapter with baseline, decode-matched no-op, and live feedback. It updates the future cross-attention mask; it does not directly move the subject. Verify exact no-op replay and that the intervention changes at least one rasterized token. An eight-step check is plumbing only because the intervention has few later calls.
6. **Policy efficacy.** Use the full 30-step schedule and matched controls: open loop, exact no-op, frozen static correction, live residual, temporally shuffled residual, and sign-reversed residual. Keep seed/path as the statistical unit and retain failed or unscoreable clips. Require a held-out observer gate, paired uncertainty, and no degradation to identity, visibility, object extent, background stability, or blind quality. A practical pilot target is at least 20% lower paired path error; this threshold is a project decision, not a published standard.
7. **If the action fails.** Do not tune an arbitrary controller. Investigate whether spatially retained hidden/QK features can read object position, using DiffTrack/SG-I2V/DreamTraj as design references; or evaluate Frame Guidance as a separate loss-based actuator. Re-check code and weight licenses before reusing unlicensed repositories.

## Measurement contract

- Primary outcome: time-aligned per-frame center ADE, with a predeclared penalty for missing target frames; report pixels and normalized image-diagonal units.
- Secondary outcomes: endpoint error and signed response to each action perturbation.
- Separate guardrails: visibility/identity/duplicates, apparent width/height or mask area, background landmark motion, and blinded appearance review.
- Calibrate the observer against simulator truth, then against blind human labels on generated frames. Do not select detections by proximity to the requested path; do not drop missing frames or use the tracker being optimized as ground truth.
- Report by clip/seed block, not by treating video frames as independent samples.

## Current evidence and limits

- The controlled Kubric scorer and observer code have CPU validation. The corrected red proxy scores the two mirrored Kubric paths at 34/34 frames, 1.7297 px mean center error and 1.8672 px p95; the 0.65 confidence cutoff is in-sample. This validates a narrow synthetic red-object measurement path, not transfer to generated Wan clips.
- SAM 2.1 MPS probing is complete; its systematic center bias is too large for direct uncorrected live control under the current half-cell criterion. A blinded human-label package exists for an existing M0 sample, but labels and held-out generated-video observer validation remain outstanding.
- The scheduler conversion smoke captured the exact step-4 UniPC x0 estimate on CPU with zero numerical difference. The feedback adapter/runner are source- and CPU-checked. No DiTraj GPU inference, online latent intervention, or policy gain has run.
- The local machine is MPS-capable but has no CUDA, no cached Wan weights, and no Diffusers installation. An earlier Wan M0 clip on free T4 peaked at 7.91 GiB, but DiTraj's mask overhead and in-loop VAE decode cost remain unmeasured. The T4 remains the target; no paid compute is authorized.
- DiTraj code is CC BY-NC 4.0, so this implementation path is limited to noncommercial research unless rights are separately resolved. The Frame Guidance and DiffTrack repositories need code-license verification before reuse.

## Research record

- [Global methods and papers](2026-10-02_global_motion_research.md)
- [Observer and benchmark choices](2026-10-02_observer_benchmark_research.md)
- [Open-source model/policy/runtime stack](2026-10-02_open_source_policy_stack.md)
- [Fixed-camera path evaluation audit](2026-10-02_fixed_camera_path_evaluation_audit.md)
- [DiTraj/feedback architecture audit](2026-10-02_same_sample_feedback_architecture_audit.md)
- [2D path component plan](2d_path_policy_component_plan.md)

