# LatentGate literature map

**Status:** the first experimental slice is fixed-camera 2D path adherence for one visible object. DiTraj is the closest runnable path-control baseline; it is not the LatentGate contribution.  
**Longer-term candidate failure:** rigid-object geometry consistency (apparent scale, shape, and orientation) during a fixed-camera tabletop pick-and-place.  
**Evidence caveat:** M0's single red-ball clip has a color-mask proxy showing 25.96% apparent area growth. This is a lead for selecting a measurable failure, not a robust failure-rate estimate or metric 3D measurement.

## Current project constraints

- The reproducible microscope is Wan2.1 T2V 1.3B on a Colab T4: 17 frames, 832×480, 30 steps, 7.91 GiB peak allocated VRAM, intermediate latent/clean-estimate previews, selected hidden/attention summaries, VAE round trip, and exact same-seed replay. The current run is M0 evidence, not a steering result.
- A separate CogVideoX-2B T2V feasibility sample ran on the free T4 in about 13.3 minutes. The MacBook Pro is M3 with 16 GiB unified memory.
- The draft M1 image fixture is an image-generated mug/holder scene without calibration or simulator truth. RGB masks, tracks, and monocular depth on it are proxies. Do not describe them as physical contact or metric 3D ground truth.
- The proposed 5B I2V smoke configuration and all policy ideas remain unrun. This reconnaissance does not establish T4 viability for that configuration.

## Paper and method families

### Geometry-aware 4D Video Generation for Robot Manipulation

[Paper (arXiv:2507.01099)](https://arxiv.org/abs/2507.01099) · [official code](https://github.com/lzylucy/4dgen) · [project page](https://robot4dgen.github.io/)

The paper's method is multi-view, RGB-D-conditioned video diffusion with cross-view pointmap alignment. It is not itself a 4D Gaussian-splatting generator. Its 4D-Gaussian comparison is a two-stage baseline: generate one RGB video with fine-tuned SVD, then reconstruct it with Shape of Motion. The official code describes SVD backbones and task-specific RGB/pointmap VAEs; it reports training on four 48-GB A6000 GPUs and roughly two days. The appendix also reports a 2.4B-parameter model, 47-GB training memory, and a 30-second inference time. These are strong transfer-risk flags for our current T4 and frozen-Wan setup. This paper is a useful robotics geometry benchmark and teacher-design reference; native latent-policy compatibility is not established.

### Geometry Forcing

[Paper (arXiv:2507.07982)](https://arxiv.org/abs/2507.07982) · [official code](https://github.com/CIntellifusion/GeometryForcing) · [project page](https://geometryforcing.github.io/)

This work aligns intermediate video-diffusion features with features from VGGT using angular and scale-alignment objectives. Of the candidates inspected so far, it has the closest conceptual match to teaching geometry inside a generator's latent trajectory. It is still a training recipe rather than a demonstrated drop-in inference-time gate for frozen Wan2.1; base-model, data, hardware, and code compatibility need a method/config audit.

### Gaussian Action Field (GAF)

[Paper (arXiv:2506.14135)](https://arxiv.org/abs/2506.14135) · [official code](https://github.com/ChaiYing1/GAF)

This is a likely match for the remembered “4D Gaussian” robotics paper. It extends Gaussian scene representations with motion attributes, predicts current/future scene states and an initial action, then uses a diffusion-based action-refinement stage. That is a 4D scene/action model, not a video-diffusion latent-steering method. Its potential role in LatentGate could be a structured motion/geometry teacher, but data requirements, license, compute, and whether it can supervise our chosen failure need review.

### DreamGen

[Paper (arXiv:2505.12705)](https://arxiv.org/abs/2505.12705) · [project page](https://research.nvidia.com/labs/gear/dreamgen/)

DreamGen studies synthetic robot trajectories from video world models, recovering pseudo-actions with latent-action or inverse-dynamics models and evaluating downstream robot-policy generalization. It is important context for the eventual data-value endpoint; it does not, by itself, identify or causally correct a geometry failure inside our video generator.

### Video Depth Anything

[Paper (arXiv:2501.12375)](https://arxiv.org/abs/2501.12375) · [official code](https://github.com/DepthAnything/Video-Depth-Anything)

A temporally consistent video-depth estimator is a possible teacher or post-hoc measurement source. Its predictions on generated RGB are not ground truth, and the depth model's own temporal consistency objective makes it unsuitable as the sole independent evaluator of a depth-consistency policy.

### Track4Gen

[CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Jeong_Track4Gen_Teaching_Video_Diffusion_Models_to_Track_Points_Improves_Video_CVPR_2025_paper.html)

Candidate evidence for tracking/appearance consistency. Its precise intervention point, training changes, released code/checkpoint, and relevance to rigid-object scale/orientation need to be checked before treating it as a reusable policy.

### RoboPearls and Shape of Motion

[RoboPearls paper (arXiv:2506.22756)](https://arxiv.org/abs/2506.22756) · [RoboPearls project page](https://tangtaogo.github.io/RoboPearls/)  
[Shape of Motion paper (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.html)

These distinguish two more 4D-Gaussian directions from video diffusion. RoboPearls reconstructs editable dynamic Gaussian scenes for simulation operators; Shape of Motion reconstructs persistent 3D motion trajectories from a monocular video. They may support synthetic-scene labels, analysis, or downstream simulation, but neither should be described as a video-diffusion latent gate without an explicit bridge experiment.

### Clean-space trajectory guidance and diffusion capacity routing

[Trajectory-Level Mode Guidance (arXiv:2609.36530)](https://arxiv.org/abs/2609.36530) directly guides reconstructed clean robot trajectories during diffusion sampling using planning-cost gradients and partial trajectory priors. This is prior art for clean-estimate guidance and test-time trajectory steering; do not claim priority for those broad ideas. Its low-dimensional trajectory costs do not establish an RGB video loop in which an external observer measures residuals on decoded intermediate estimates and routes them into later denoising calls. [TRACK (arXiv:2609.30096)](https://arxiv.org/abs/2609.30096) instead routes denoising steps between compatible large and small video models based on calibrated disagreement. It is an efficiency method, not geometry control, and no compatible pair has been audited for the current Wan 1.3B or CogVideoX-5B-I2V paths. See the [focused comparison](2026-10-11_clean_space_guidance_and_capacity_routing.md).

## Preliminary decision rubric

The anchor must satisfy all four conditions: (1) address one measurable failure; (2) expose a reusable teacher signal or intervention primitive; (3) map to the existing observable model without requiring an unmeasured multi-GPU training path; and (4) permit a held-out evaluator independent of the policy teacher. A paper can be the closest conceptual match yet still fail the compute/compatibility gate.

The fixed-camera 2D path slice has a first-pass literature and source audit in [`2026-10-02_path-control_landscape.md`](2026-10-02_path-control_landscape.md), plus a reconciled component plan. The completed account-side Deep Research reports and their project-gate reconciliation are summarized in [`2026-10-11_full_research_chat_reconciliation.md`](2026-10-11_full_research_chat_reconciliation.md). A focused follow-up covering three papers submitted October 2–8, 2026, appears in [`2026-10-11_post_cutoff_literature_update.md`](2026-10-11_post_cutoff_literature_update.md): RAFC strengthens the precedent for online reliability control over generated futures, SUAVE for unified masked-diffusion video/action control, and RobotAPO for denoising-space physical-preference optimization. A separate focused audit of September 24–29 work records clean-space robot-trajectory guidance as direct prior art and TRACK as a compatible-model capacity-routing method; see [`2026-10-11_clean_space_guidance_and_capacity_routing.md`](2026-10-11_clean_space_guidance_and_capacity_routing.md). The only defensible candidate distinction remains narrower and empirical: whether an external observer's independently measured RGB residual from a decoded, in-progress predicted-clean video sample can change later denoising calls for that same sample and improve it under matched controls. This is not an established novelty claim. The local Gate 1 human-reference gate remains open; no DiTraj feedback efficacy result exists. Maintain the later rigid-object geometry objective separately.

## Current open questions

1. Is GAF the paper the user recalls, and which other 4D-Gaussian manipulation/data-generation paper was intended?
2. Which target can be evaluated with actual rigid-pose/scale ground truth rather than generated-image proxies?
3. Can Geometry Forcing's representation-alignment idea be isolated in a small probe/controller on Wan, or does the contribution require full backbone training?
4. Is trajectory/point-track stability more observable and steerable than geometry/scale under the available image-conditioning setup?
5. Which teacher and held-out evaluator can be made independent enough to avoid rewarding a policy for copying its own teacher's errors?
