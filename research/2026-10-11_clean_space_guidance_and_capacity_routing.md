# Clean-space trajectory guidance and diffusion capacity routing

**Checked:** 2026-10-11  
**Scope:** Targeted audit of two recent primary arXiv papers that surfaced while checking the novelty boundary and compute alternatives. This is a focused comparison, not an exhaustive review or a reproduction.

## Trajectory-Level Mode Guidance for Controllable Diffusion-Based Multi-Robot Motion Planning

[Paper and full text](https://arxiv.org/abs/2609.36530) · [HTML](https://arxiv.org/html/2609.36530)

The v1 paper (submitted 2026-09-29) guides diffusion sampling in a reconstructed clean robot-trajectory space. At each reverse step, it applies gradients from planning costs and a timestep-dependent partial-trajectory prior to the clean estimate, then maps the guided estimate back to the preceding diffusion state; the prior is also injected at the corresponding noise level. The costs include velocity, signed-distance obstacle clearance, goal, inter-robot collision, and smoothness. The model uses a low-dimensional trajectory representation, and the paper states that code and the dataset-generation pipeline will be released upon publication.

**Relevance and boundary:** This is direct prior art for test-time clean-estimate guidance of diffusion-generated robot trajectories. LatentGate should not claim to be the first to guide diffusion through predicted-clean states, first to perform clean-space trajectory guidance, or broadly first to apply feedback or costs during trajectory denoising. The paper does not decode predicted-clean video latents to RGB, use an external object observer to measure image-plane residuals on the current generated sample, or use those residuals to alter later video denoising through spatial cross-attention masks. The narrower question—whether that observer-mediated, same-sample video feedback improves geometry under matched controls—remains distinct but unproven until the human-reference gate and controlled experiment pass.

## Accelerating Video Diffusion via Training-Free Trajectory Routing (TRACK)

[Paper and full text](https://arxiv.org/abs/2609.30096) · [HTML](https://arxiv.org/html/2609.30096)

The v1 paper (submitted 2026-09-24) calibrates disagreement between a large and small compatible video model along a reference latent trajectory, then routes each inference step to one model. Its reported setups require compatible latent representations, prediction targets, conditioning, and scheduler semantics; both model capacities are part of the setup. The paper reports results on Wan2.1, Cosmos 3, TurboDiffusion, and FastVideo.

**Relevance and boundary:** TRACK is a possible inference-efficiency direction, not a geometry controller or online residual-feedback method. It does not remove decode/observer work and is not an established substitute for the current T4 plan: the present project path uses Wan2.1 T2V 1.3B, while the M1 smoke is CogVideoX-5B-I2V; no compatible paired large/small checkpoint route has been audited or profiled for either active experiment. Treat it as a future benchmark candidate only if a compatible pair and memory/runtime cost can be demonstrated.

## Consequence for LatentGate

The most defensible contribution question is now narrower: **Can an independently measured geometric residual from decoded, in-progress predicted-clean video alter subsequent denoising controls and improve that same sample beyond matched no-op, fixed-action, shuffled-state, and sign-reversed controls?** Clean-space trajectory guidance is already established in low-dimensional planning; this memo makes no claim that the narrower video-loop distinction is novel across all literature. The current project still has no causal policy-efficacy result. Gate 1 human-reference reliability must pass before a video feedback experiment can support the claim.

## Source boundary

The summaries were checked against the v1 arXiv HTML pages linked above. Submission dates and method descriptions are author-reported. I did not audit released code/checkpoints, reproduce results, or verify acceptance status.
