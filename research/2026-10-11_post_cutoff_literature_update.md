# Post-cutoff literature update: generated futures and video-action control

**Checked:** 2026-10-11  
**Scope:** Focused screening of three primary arXiv papers submitted October 2–8, 2026, after the October 1 cutoff of the expanded Deep Research report. This is an update, not an exhaustive systematic review. The findings below describe the papers; they are not local reproductions.

## Papers to account for

### Reliability-Aware Future Conditioning (RAFC), October 8

[Paper and full text](https://arxiv.org/abs/2610.11956)

RAFC builds on Future-Experience Conditioning. The video future is generated once at task initialization. At each robot-control step, RAFC evaluates a static fallback and nearby temporal shifts of that same future through a frozen behavior-cloning policy. A learned gate estimates whether to trust the dynamic future and which shifted candidate fits, blends the policy features and base actions, and adds a residual-RL action. The paper reports temporal-mismatch experiments and a three-task Franka study.

**Relevance:** This is a close precedent for state-dependent, online reliability control over a generated future. Its live signal is the robot observation as interpreted by policy features; its action changes robot-policy conditioning and actions. It does not use an independently measured geometric residual from the current sample's intermediate predicted-clean frames to change later denoising calls for that same sample. That distinction supports a narrow experiment question, not a broad “generated futures have never been used in feedback” claim.

### SUAVE: Unified Video-Action Models via Masked Diffusion, October 2

[Paper and full text](https://arxiv.org/abs/2610.04009)

SUAVE uses one masked-diffusion transformer over a shared discrete token sequence for language, video, and actions. Which sequence positions are masked determines whether inference predicts a future video, actions, or both. The paper reports real-robot closed-loop manipulation at 2.5 actions per second on an RTX 5090.

**Relevance:** SUAVE is direct prior art for unified diffusion-based video/action prediction and closed-loop robot control. It jointly models and predicts video and action tokens; it does not feed a measured geometric residual from a partially denoised video back into later denoising steps of an already-running sample.

### RobotAPO: Adversarial Physics Preference Optimization, October 7

[Paper and full text](https://arxiv.org/abs/2610.09454)

RobotAPO uses condition-matched successful and physically failed manipulation videos, plus an adversarial counterfactual proposer, to optimize a Wan2.2 flow-matching video model in its denoising space. The paper describes a prompt-and-reference inference interface without external action, trajectory, mask, or depth conditioning, and reports downstream robot-replay results.

**Relevance:** This is important prior art for physical-video correction and optimization in denoising space. Its intervention is learned from a curated preference dataset during model optimization, rather than an online geometric residual measured from the current sample and fed into the remaining denoising calls.

## Consequence for LatentGate

Do not claim to be first in closed-loop video-based robot control, unified video-action diffusion, physical video correction, denoising-space optimization, or clean-estimate guidance broadly. The focused [September 24–29 audit](2026-10-11_clean_space_guidance_and_capacity_routing.md) records clean-space trajectory guidance as direct prior art. The most defensible candidate distinction remains conditional and empirical: **whether an independently measured geometric residual from decoded, in-progress predicted-clean video can alter later denoising calls for that same sample and improve it beyond matched no-op, fixed-action, shuffled-state, and sign-reversed controls.** This distinction still needs broader literature review and a passing experiment; it is not an established novelty claim.

The local experiment order does not change. Gate 1 human-reference reliability is still open because the v5 annotation templates are blank. The synthetic known-truth scorer check validates that the measurement path can expose common-mode annotation bias; it supplies no human reliability or policy-efficacy result. No model-generation or feedback experiment is established by these papers or by this literature update.

## Source boundary

The summaries were checked against the arXiv v1 full-text HTML linked above. Dates, methods, and reported results are author-reported. I did not audit released code/checkpoints, reproduce the benchmarks, or verify acceptance status.
