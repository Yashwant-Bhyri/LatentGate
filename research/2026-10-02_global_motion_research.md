# Fixed-camera 2D trajectory fidelity in video diffusion

**Research snapshot:** 2 October 2026  
**Question:** How can we measurably reduce a named object's path drift and identity loss in short text-to-video samples, with the camera fixed, using Wan2.1-T2V-1.3B as the pilot generator?

## Executive recommendation

The literature has many motion-control systems, but it has not converged on a universally reliable closed-loop object-state policy for a small T2V model. Most papers solve a neighboring problem: supplying an open-loop box/point path, making the model preserve correspondence, improving generic temporal coherence, or training a new conditioned generator. Those results are not interchangeable.

For the first small run, use **DiTraj on Wan2.1-T2V-1.3B as an actuator probe**, then decide whether it can support a state-feedback controller. DiTraj is the most direct fit in this survey: its official code explicitly targets Wan2.1 DiTs, edits foreground positional encodings and uses foreground/background separation guidance, and it does not require model training. Its paper/repository are new and small, however, and the source code is CC BY-NC 4.0. Treat the paper's claimed quality/control results as author-reported until our matched experiment reproduces them. [Paper](https://arxiv.org/abs/2509.21839), [official code and license](https://github.com/xduzhangjiayu/DiTraj).

The very first measurement should be a **signed one-cell actuation check**: with the same prompt, random seed and shared denoising prefix, nudge the requested object's bbox center by +1 and -1 Wan spatial-latent cells along x, then y. Track the rendered object's actual center. If the output does not move reliably in the requested direction by more than observer noise, do not build a feedback policy on top of that box action. Inspect a latent/attention correspondence reader next, drawing on SG-I2V's feature-alignment/latent-optimization principle. If the signed response is reliable, fit a small, bounded correction policy over that proven action interface and evaluate it against the same-seed open-loop baseline.

Wan's official repository reports 8.19 GB VRAM for T2V-1.3B and recommends 480p; its documented 4090 example is about four minutes for five seconds at 480p. This makes a T4-class trial plausible in principle, but it is not a measured T4 guarantee; text-encoder/VAE peaks, software versions and offloading still need a preflight. [Wan2.1 official repository](https://github.com/Wan-Video/Wan2.1).

## What “drift” means in this experiment

Keep four failure sources separate:

1. **Path error:** the target object center misses the requested 2D path.
2. **Object permanence/identity error:** the object disappears, changes identity, or reappears as a different instance.
3. **Control leakage:** the target moves, but unwanted scene/background regions move too.
4. **Observer error:** the tracker or segmentation mask is wrong. This can look like generator drift and needs its own calibration.

The initial scope is one visible, non-occluded object under a fixed camera, with a simple foreground/background scene. This isolates object trajectory from camera motion, depth reconstruction, contact physics and multi-object interactions. A result here would be a bounded claim about that setting, not proof of general physical consistency.

## Landscape: what each family contributes

| Family | Representative work and mechanism | Fit to this experiment | Main limitation |
|---|---|---|---|
| **Box/mask and positional-encoding guidance** | **DiTraj** uses separate foreground/background prompts and modifies foreground token positions with spatial-temporal decoupled 3D RoPE. **Motion-Zero** uses a position-biased initial noise prior, attention-map spatial constraints and shifted temporal attention. | DiTraj is the only surveyed method whose official repo directly names Wan2.1 and T2V/DiT support. It gives us a low-training-cost control action to test. Motion-Zero is useful evidence for noise priors and attention/position constraints. | These are mostly *open-loop* conditions. A requested box is not an observation of where the generated object actually went. Motion-Zero's released script targets ZeroScope U-Net, not Wan DiT; its transfer is architectural inspiration rather than a ready Wan patch. DiTraj's effectiveness at a one-cell shift is an empirical question. |
| **Latent feedback and self-guidance** | **SG-I2V** aligns per-frame self-attention K/V to the first frame, then optimizes the noisy latent to make target-box features correspond across frames; it restores high-frequency latent components after the update. **WorldForge** uses intra-step refinement, optical-flow-based channel selection and guided/unguided dual-path correction. | This family is closest to a genuine latent steering policy. SG-I2V supplies an especially concrete fallback if DiTraj's box action has no signed effect. WorldForge has useful safeguards against corrupting appearance while correcting motion. | SG-I2V is for SVD image-to-video U-Net and was demonstrated at 576×1024/14 frames with repeated gradient steps; the released setup was tested on a 48 GB A6000. WorldForge targets camera trajectories and 3D/4D warping, not a single foreground object under a static camera. Neither is a drop-in small Wan-T2V controller. |
| **Attention control / motion adapters** | **FreeTraj** changes initial noise and attention computation. **TrackGo** adds a TrackAdapter in temporal self-attention and conditions on free-form masks/arrows. | Establishes that attention and low-frequency latent structure can encode controllable motion. TrackGo's region+direction split is relevant if one bbox is too coarse. | FreeTraj's released base is VideoCrafter2 at 320×512 and the README limits use to research/personal/non-commercial despite GitHub license metadata. TrackGo is a trained adapter, not an inference-only state-feedback policy; no official source repository was located in this survey. |
| **Point tracks / flow-conditioned motion** | **Tora** compresses dense/sparse trajectories into motion patches and fuses them into a DiT. **Motion Prompting** trains a trajectory ControlNet using sparse or dense point tracks. **Wan-Move** propagates first-frame features along point trajectories in latent space. **ATI** adds trajectory instructions to Wan I2V. | These papers make trajectories and point tracks a natural conditioning representation, and offer a future model-native module path. Wan-Move/ATI provide Wan-specific design references. | They generally train or fine-tune a model and focus on I2V or larger backbones. They cost more than a T2V-1.3B pilot and do not establish that their gains transfer to a fixed-camera T2V-1.3B generator. Optical flow alone measures apparent motion, not object identity. |
| **Teacher/student distillation and reward post-training** | **VideoGuide** uses a stronger teacher's early denoising trajectory as inference-time guidance. **AR-Drag** fine-tunes an I2V generator then uses trajectory-based RL reward. **DyMD** (submitted 25 Sep 2026) reports distilling a 14B teacher into a 1.3B student while preserving robot-object dynamics. | Strong future direction if we can build a reliable trajectory reward/state estimator. DyMD is especially relevant to embodied interaction and weak-motion failure in students. | Teacher guidance is not automatically a policy for a particular target path. DyMD is a brand-new preprint with no public code link in its arXiv record; its robotics/world-model benchmark is not the same task as T2V fixed-camera object drift. Training before proving measurement and control would make failures hard to interpret. |

### Selected primary papers and repositories

**Direct small-model route**

- [DiTraj paper](https://arxiv.org/abs/2509.21839) and [official repository](https://github.com/xduzhangjiayu/DiTraj). The paper proposes foreground/background separation guidance plus foreground-only STD-RoPE changes. The repo says it adapts to Wan2.1 and CogVideoX; setup is Python 3.11 with Diffusers v0.33.1 and a replacement Wan transformer file. Code is **CC BY-NC 4.0 (research/non-commercial)**. The repository was released in Dec 2025 and is still a small research codebase, so pin and audit the exact commit/dependencies.
- [Wan2.1 official code/model card](https://github.com/Wan-Video/Wan2.1). It lists T2V-1.3B at 480p, reports 8.19 GB VRAM, and provides `--offload_model True --t5_cpu` for reducing peaks. Wan's repository is Apache-2.0; verify the exact selected checkpoint/model-card terms separately when redistributing weights or outputs.

**Open-loop and attention baselines**

- [Motion-Zero paper](https://ojs.aaai.org/index.php/AAAI/article/view/32198), [official code](https://github.com/vpx-ecnu/MotionZero). Zero-shot method; code is Apache-2.0. Its released inference example uses ZeroScope v2 576w, Python 3.8 and U-Net-specific controls. Good reference for factorizing initial noise, attention position losses and temporal copy/shift; not an out-of-box Wan baseline.
- [FreeTraj paper](https://arxiv.org/abs/2406.16863), [official code](https://github.com/arthur-qiu/FreeTraj). Noise construction plus attention changes, released for VideoCrafter2 320×512. README states research/personal/non-commercial use; do not infer commercial permission solely from GitHub's Apache metadata.
- [TrackGo paper](https://ojs.aaai.org/index.php/AAAI/article/view/33167), [project page](https://zhtjtcz.github.io/TrackGo-Page/). Its learned TrackAdapter goes into temporal self-attention and uses mask-and-arrow conditions. Its core evidence is learned feature control, but it requires a trained adapter and was not found as a drop-in Wan-1.3B package.

**Latent feedback and correspondence**

- [SG-I2V ICLR 2025 paper](https://arxiv.org/abs/2411.04989), [official code](https://github.com/Kmcode1/SG-I2V). It optimizes noisy latents against cross-frame semantically aligned features; the paper reports 14 frames, 576×1024, 50 denoising steps, and five latent-optimization iterations at each of 16 early steps. The repo is MIT and says it was tested on an RTX A6000 48 GB. The contribution to borrow is the *state reader + latent correction + noise-distribution safeguard*, not its SVD U-Net patch verbatim.
- [WorldForge CVPR 2026 paper](https://arxiv.org/abs/2509.15130), [official code](https://github.com/Westlake-AGI-Lab/WorldForge). It is a training-free camera/3D/4D method that includes intra-step refinement, flow-selected latent channels and guided-vs-unguided correction. The current repository labels itself Apache-2.0 and includes Wan2.1 inference; its setup uses CUDA, PyTorch 2.6, xformers and FlashAttention. Use its selective-fusion and protected-reference-path ideas as later design patterns; the task and code stack do not match our first T2V object-path run.
- [FlowMo paper](https://arxiv.org/abs/2506.01144), [official code](https://github.com/arielshaulov/FlowMo). It reduces patchwise temporal latent variance using the model's own predictions, without target-path supervision. Code is MIT; its README lists 2×H100 prerequisites while showing a Wan2.1-1.3B command and still has a TODO for Wan-specific code. This is a candidate for generic coherence ablation, not a direct target trajectory controller, and its listed hardware is not suitable for a small T4 pilot without adaptation.

**Trained control and data**

- [Tora CVPR 2025 paper](https://arxiv.org/abs/2407.21705), [official repository](https://github.com/alibaba/Tora). A learned trajectory extractor and motion-guidance fuser inject conditions into a DiT. The repo has Apache-2.0 code, but the released CogVideoX-5B weights are under the CogVideoX terms/academic-research positioning. The paper describes 630k training clips and 8 A100 GPUs; the repo reports roughly 30 GiB inference memory in its main path. This is a capable trained comparator, not the small run.
- [Motion Prompting CVPR 2025 oral paper](https://arxiv.org/abs/2412.02700), [project page](https://motion-prompting.github.io/). Google DeepMind et al. train a trajectory ControlNet; point tracks can represent object-specific/global, dense/sparse motion. It is an instructive representation/data strategy; no official reusable code/model package was found on the official project page in this survey.
- [Wan-Move NeurIPS 2025 paper](https://papers.neurips.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html), [official code, weights and MoveBench](https://github.com/ali-vilab/Wan-Move). It propagates first-frame features along dense tracks; released model is Wan-I2V-14B at 832×480, not T2V-1.3B. The repo states Apache-2.0 for its models and provides CPU/offload options; do not confuse its checkpoint size/task with the small T2V target.
- [ATI: Any Trajectory Instruction](https://arxiv.org/abs/2505.22944), [ByteDance repository](https://github.com/bytedance/ATI). Adds trajectory instruction modules to Wan2.1 I2V and unifies object/local/camera motion; repository is Apache-2.0, but its documented checkpoint is ATI-Wan-14B-I2V. It is a useful native-Wan architecture reference, not the pilot model.
- [MagicMotion ICCV 2025 paper](https://arxiv.org/abs/2503.16421), [official repository](https://github.com/quanhaol/MagicMotion). Progressive dense mask → box → sparse box control and the MagicData/MagicBench pipeline are valuable design/evaluation references. Their README says single-4090 inference under 24 GB; the repository page did not show a license file in the inspected tree, so confirm reuse terms before copying code or data.

**Teacher/student frontier**

- [VideoGuide CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_VideoGuide_Improving_Video_Diffusion_Models_without_Training_Through_a_Teachers_CVPR_2025_paper.pdf) uses a stronger video model's denoised output as early inference-time guidance for another generator. The technique targets temporal coherence, not explicit object-path error.
- [AR-Drag preprint](https://arxiv.org/abs/2510.08131) reports a 1.3B autoregressive I2V model fine-tuned for motion control and then optimized with a trajectory-based reward. It is evidence for a reward-driven downstream route, but not a ready Wan T2V implementation.
- [DyMD preprint](https://arxiv.org/abs/2609.31349), submitted 25 Sep 2026, is the freshest robotics-oriented distillation signal found. It reports a 14B → 1.3B student and +9.6 percentage points on R-Bench task adherence versus base DMD, with comparable visual quality. The preprint page has no code link and evaluates embodied interaction/world-model tasks, so regard it as a research lead rather than reproduced evidence for our metric.

## How to measure progress without confusing the tracker for the generator

### Primary outcome

For each visible frame, get target center \(\hat c_t=(\hat x_t,\hat y_t)\) from an object mask or median tracked object points; compare with requested center \(c_t\). Report:

- **Path ADE/RMSE:** mean and root-mean-square of \(\|\hat c_t-c_t\|_2\), in pixels and normalized by image diagonal; report separately in x and y.
- **Endpoint error:** final visible center versus final requested center.
- **Directional response:** for each control perturbation, project observed displacement onto the requested perturbation direction. This is the key calibration result.
- **Visible-track recall / permanence:** percentage of intended visible frames where the same target is found, plus number of identity swaps/reacquisitions. Mark intentional occlusion separately; the initial benchmark should avoid it.
- **Shape/appearance guardrails:** mask area variation, mask IoU where exact synthetic ground truth exists, and subject consistency. A controller that moves a different-looking replacement object has not solved permanence.
- **Background guardrail:** compare static-region consistency so a foreground correction is not credited if it drags the background.

Use [Kubric/MOVi](https://github.com/google-research/kubric) renders or another controllable renderer for the first synthetic observer set: object masks, object coordinates, depth and flow provide known ground truth. [TAP-Vid](https://arxiv.org/abs/2211.03726) defines any-point tracking and includes real and synthetic annotations; it is a useful tracking calibration reference. [CoTracker3](https://github.com/facebookresearch/co-tracker) supplies a released point tracker and TAP-Vid protocol; its repo documents a Kubric training checkpoint and its mixed licensing must be checked before reuse. [SAM 2](https://github.com/facebookresearch/sam2) provides video masks and Apache-2.0 code/weights. Neither tracker nor segmenter is ground truth: compare both to simulator masks and manually inspect a blinded subset before using them as the model scoreboard.

For broad quality guardrails, use [VBench/VBench++](https://github.com/Vchitect/VBench) dimensions such as subject consistency, background consistency, motion smoothness and temporal flicker. VBench is a quality suite, not a substitute for trajectory ADE or endpoint error. Trajectory-specific benchmarks such as [MoveBench](https://github.com/ali-vilab/Wan-Move) and [MagicBench](https://github.com/quanhaol/MagicMotion) can inform future holdouts, but their published generators/conditions are predominantly I2V and are not directly comparable to our T2V-1.3B prompt-only baseline.

### Observer protocol

1. First run the estimator on synthetic clips with exact masks/point tracks and measure center bias, missed-frame rate and uncertainty by direction, speed and object size.
2. On generated clips, use at least two independent views: (a) SAM2 mask centroid/area and (b) CoTracker3 points initialized inside the target object. Use color/instance-specific segmentation only as a third tool on simple colored-object stimuli.
3. Have blinded human annotators label a small balanced subset to estimate tracker-human disagreement. Mark occlusions and ambiguous object identity rather than silently dropping them.
4. Treat an intervention as a real gain only when paired path-error improvement exceeds the observer's calibrated uncertainty and static-background/subject-consistency guardrails do not regress materially.

## First intervention: DiTraj signed actuation → bounded feedback policy

### A. Causal actuation calibration

1. Fix a short prompt, resolution, sampler/settings and random seed. Disable prompt expansion for the pilot. Generate baseline and DiTraj controls with the same noise and compare identical frame indices.
2. After the same short denoising prefix, branch to bbox controls shifted by +1 and -1 Wan spatial latent cells in x, then y. Keep every other input identical. At the documented 832×480 setting, the latent grid used in the current pilot plan is 52×30, i.e. a cell is 16×16 output pixels; verify this mapping against the actual loaded VAE/model configuration before running.
3. Track the rendered object and compute the finite-difference response \(J_x=(\hat c_{x+}-\hat c_{x-})/2\), and likewise \(J_y\), plus signed projection and no-control repeatability.
4. Use three paired seeds per axis only as an exploratory smoke check. For the calibration gate, use at least eight paired seeds per axis. A concrete proposed gate is: the 95% paired bootstrap interval for the signed response is above zero on each axis, and median displacement exceeds tracker repeatability/noise by at least 2×. Freeze the gate before looking at the full benchmark. If the input perturbation only changes attention maps or prompt adherence but does not shift the rendered object, it fails. Eight seeds are still a pilot-scale screen, not strong population-level evidence; report the interval and expand seeds if it is wide or borderline.

### B. Small policy, only if A passes

Represent the tracked state per frame as \(s_t=(c_x,c_y,v_x,v_y,\log A)\); the first version only controls center position. At selected denoising checkpoints, estimate the current predicted object center from either a calibrated DiTraj foreground-token map or a decoded clean-latent estimate. With target \(r_t\), compute error \(e_t=r_t-\hat c_t\), map it through the measured local action response \(J\), and apply a damped, clipped box/RoPE correction:

\[
u_{k+1}=u_k+\mathrm{clip}(\lambda J^{\dagger} e_k,\; -u_{\max},u_{\max}).
\]

Use confidence gating: no object observation → no correction; low-confidence/occluded frame → hold the prior trajectory state. Begin with one correction at a small number of denoising checkpoints, not every step. Compare against DiTraj open-loop at the same seed and compute. If decoding an intermediate clean estimate is too expensive/noisy, test the token-map state reader separately; attention is only a proxy and must agree with the rendered observer before becoming the policy sensor.

### C. Evidence gates for a claim of real improvement

- **Run validity:** paired branches share the exact prefix/noise and same remaining generation settings; no missing outputs or silent fallback to base inference.
- **Action validity:** signed one-cell response passes the calibration gate above. If it fails, do not tune arbitrary gains; inspect latent/attention correspondence (SG-I2V principle) or change the action interface.
- **Small benchmark:** at least 8 held-out seeds across three simple path shapes (constant-velocity line, stop-go, smooth curve), with matched baseline/open-loop/feedback outputs. Keep the object and camera conditions fixed in this first phase.
- **Primary pass condition:** paired ADE reduction has a 95% bootstrap confidence interval above zero and a pre-registered material effect target (suggested initial target: ≥20% median ADE reduction). Report absolute pixel/cell changes too, not only relative percentages.
- **No-cheating guardrails:** subject-consistency and visibility do not worsen; mask area and static background remain within pre-registered tolerance; the model must obey both x and y controls, not only one trajectory direction.
- **Generalization:** after the first pass, test a different object appearance and path family, then an occlusion case. Do not claim robust policy behavior from a single red-object clip or a sample contact sheet.

The 20% target is a proposed project gate, not a literature-established threshold. The purpose is to prevent a statistically visible but practically tiny change from being presented as the North Star result.

## Why not train first?

Training-based systems have the strongest route to richer control: [Tora](https://github.com/alibaba/Tora) learns a trajectory encoder/fuser, [Motion Prompting](https://motion-prompting.github.io/) learns a trajectory ControlNet, [Wan-Move](https://github.com/ali-vilab/Wan-Move) trains motion-aware feature propagation, and [MagicMotion](https://github.com/quanhaol/MagicMotion) builds a dense-to-sparse curriculum. But they add dataset, optimization, architecture and licensing confounds before we know that our target action is observable and effective in the selected model. The mature sequence is:

1. Establish ground-truth evaluation and observer reliability.
2. Show a causal control-to-pixel response in the selected Wan checkpoint.
3. Show paired held-out path-error improvement from the inference-time policy.
4. Only then train a compact residual/adapter or student policy to imitate validated corrective actions, with path tracking reward plus a base-model appearance/identity preservation objective.
5. Evaluate against an open-loop DiTraj control and a no-controller Wan baseline; retain held-out path shapes and seeds.

Potential teacher sources can later include high-quality trajectory-conditioned I2V/video models or a larger model's corrected rollouts, but teacher output is not automatically ground truth. Keep simulator trajectories and measured object centers as the final judge. DyMD is promising for robotics interaction dynamics; it does not replace this task-specific evidence.

## Global research coverage

The recent trajectory-control literature is distributed across research groups rather than one settled stack: Google DeepMind/University of Michigan/Brown's motion prompts; Meta/Oxford tracking tools; University of Toronto/Vector's SG-I2V; Alibaba's Tora and Wan-Move; ByteDance's ATI; Fudan/Microsoft Research Asia's MagicMotion; Beihang/AIsphere's TrackGo; ECNU's Motion-Zero; Lenovo/HKUST/Tsinghua contributors to DiTraj; and Westlake/NTU's WorldForge. These clusters differ in control representation and base task (T2V vs I2V, U-Net vs DiT), so a headline result from one cannot be assumed to transfer to another. The narrowest practical common test is still: known 2D target path → named-object observation → paired tracking error → repeatable improvement.

## Source ledger

Primary papers, official project pages and official repositories were prioritized. Secondary summaries were not used to claim benchmark gains. New preprints without released code are marked as such above. Key evaluation sources:

- [Kubric paper/project](https://github.com/google-research/kubric) — synthetic videos with rich object-level truth.
- [TAP-Vid paper](https://arxiv.org/abs/2211.03726) and [CoTracker3 code](https://github.com/facebookresearch/co-tracker) — any-point tracking and visible-point metrics.
- [SAM 2 paper/code and license](https://github.com/facebookresearch/sam2) — video mask propagation; use as an observer, not truth.
- [VBench/VBench++ code](https://github.com/Vchitect/VBench) — broad quality guardrails, not a path-control metric.
- [Wan-Move/MoveBench](https://github.com/ali-vilab/Wan-Move) and [MagicMotion/MagicBench](https://github.com/quanhaol/MagicMotion) — emerging trajectory-control benchmarks, mainly I2V.
