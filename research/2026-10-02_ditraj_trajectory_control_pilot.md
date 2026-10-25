# First trajectory-control policy: DiTraj adaptation

**Updated:** 2026-10-02  
**Status:** method selection and measurable 17-frame pilot specified; no new generation started.  
**Supersedes:** the earlier provisional ranking in `2026-10-02_vigor_research_memo.md` for the *first* intervention. Frame Guidance remains a candidate for later selected-frame shape/appearance constraints.

## Decision

The first failure mode will be **2D trajectory drift of one rigid object under a fixed camera and approximately fixed depth**. The narrow fixture is the red ball in M0: prescribe a horizontal path from its observed start region toward the right side of frame and require the ball to remain visible with nearly constant apparent extent. Trajectory adherence is the primary outcome; apparent-size stability and scene quality are guardrails. This is a first controllability result, not a claim about 3D reconstruction, metric scale, object interaction, or physics.

The first method to replicate and adapt is **DiTraj**. Its inference-time intervention is on Wan's spatiotemporal positional embeddings and attention masks, which is more directly aligned with a known per-frame path than a generic video-geometry reward. It has a released Wan2.1-1.3B path and does not train weights. Use it first as a published-method baseline; any project-specific adaptive policy comes after we establish the baseline and failure cases.

## Why this paper is the right first experiment

DiTraj decomposes a prompt into foreground and background descriptions, maps the supplied bounding-box path onto video tokens, and applies a cross-attention mask during early layout formation. For the first few denoising steps, its STD-RoPE operation aligns the spatial RoPE values of foreground tokens across their per-frame boxes while leaving temporal RoPE unchanged. It adds a self-attention mask for repeated-position tokens to reduce artifacts. With different box sizes, it rescales the anchor's spatial RoPE by nearest-neighbor sampling; the authors describe this as 3D-aware control through box size, but the input remains a 2D box path and does not provide calibrated depth or 3D ground truth. [Paper, method](https://arxiv.org/html/2509.21839v2#S3) · [Official implementation](https://github.com/xduzhangjiayu/DiTraj)

The paper tests Wan2.1-1.3B and CogVideoX-5B. Reported Wan2.1 settings are 832×480, 81 frames, 50 steps, and an A100; its tested trajectory metrics use OWL-ViT-Large detections. In its ablation, the paper reports Wan baseline → foreground/background separation → full DiTraj mIoU of 23.7 → 35.4 → 47.3, center distance of 0.17 → 0.12 → 0.09, and AP50 of 7.7 → 25.6 → 50.5. Its reported 1.3B inference time changes from 185 to 196 seconds (+5.9%). These are author-reported numbers in their evaluation setup, not our reproduction or evidence for manipulation physics. The authors report runs on an A100; none of those figures establish T4 performance. [Experiments and ablations](https://arxiv.org/html/2509.21839v2#S4) · [Inference overhead](https://arxiv.org/html/2509.21839v2#A3)

The official implementation replaces the Wan transformer implementation and subclasses the Diffusers pipeline. The source snapshot inspected for this audit is commit `6fa8e613c9e7c40ec2db456ab2ba34367a076886`. Its README specifies Diffusers v0.33.1; the checked requirements pin PyTorch 2.7.1 and Transformers 4.52.4. The checked runner defaults to 81 frames and BF16. Our M0 microscope used Diffusers 0.40.0, PyTorch 2.11, FP16, 17 frames, and 30 steps. Therefore the pilot must use a separate pinned environment or an explicit port; it must use FP16 on a T4 and measure compatibility, speed, and memory. A result with the original DiTraj code and a result with M0's sampler cannot be called an exact replication unless the code and scheduler differences are reconciled. [Pinned source snapshot](https://github.com/xduzhangjiayu/DiTraj/tree/6fa8e613c9e7c40ec2db456ab2ba34367a076886) · [Official runner](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/run.py) · [Official requirements](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/requirements.txt)

The repository README states **CC BY-NC 4.0** for its code, but the inspected tree has no separate `LICENSE` file. Keep this adaptation in the research-only track unless the authors clarify terms for another use. The base Wan model and any evaluator weights have separate terms. [README license statement](https://github.com/xduzhangjiayu/DiTraj/tree/6fa8e613c9e7c40ec2db456ab2ba34367a076886#license)

### Source-level implementation audit

Two issues matter for a faithful, safe adaptation:

1. The reference runner uses BF16 and moves the complete pipeline to CUDA. T4 is compute capability 7.5 and should use FP16; text encoding, transformer loading, and the dense attention mask need measured peak-memory checks. The paper's A100 results are not evidence of T4 fit.
2. In both `module/transformer_wan.py` and `module/pipe.py`, `find_corners` returns the maximum occupied row/column as an inclusive index, but the subsequent Python slices use that value as an exclusive endpoint. The RoPE replacement and duplicate-coordinate map therefore omit the last row and column of each target box. The paper's method describes modifying all foreground box tokens. The local primitive uses consistent half-open bounds, fixing this likely implementation off-by-one; it is an explicit adaptation and will not be mislabeled as an exact upstream reproduction.

The reference repeated-position mask builds a pairwise coordinate-equality tensor, which scales quadratically with video-token count. At our 17-frame, 832×480 Wan token grid the final mask has 7,800×7,800 entries. The local helper [`trajectory_rope_gate.py`](trajectory_rope_gate.py) keeps a token-count guard, detects duplicate coordinates with `torch.unique`, preserves temporal phase, and can apply an optional unit-circle phase blend. This helper is currently a tested CPU geometry primitive, not an integrated Wan pipeline or a measured video-generation improvement.

For the first three-arm smoke, use the pinned upstream implementation so we establish the paper baseline. Do not quietly replace it with the local helper. If the smoke is valid, add a fourth controlled comparison later: upstream STD-RoPE vs the half-open full-box correction, then a soft/adaptive gate. This separates faithful reference behavior from the LatentGate extension.

## Alternatives and scope boundaries

- **Frame Guidance** directly optimizes decoded selected-frame losses through the video model and VAE, and supports RGB, depth, sketch, and masked-region conditions. It is a strong later route for keyframe geometry or shape constraints. Its paper identifies video-wide gradient memory as a primary challenge and uses latent slicing plus a stage-aware optimization schedule; it does not give as direct a first test of a prescribed object's full trajectory as DiTraj. Its Wan integration also needs pinned-version checking. [Paper](https://arxiv.org/html/2506.07177v2) · [Official code](https://github.com/agwmon/frame-guidance)
- **Wan-Move** provides motion-aware latent feature propagation and a benchmark, but the official released inference checkpoint is 14B. It is a useful future teacher if a suitable compute slot is available; the released 14B checkpoint and sample offload flags do not establish that it will fit or run usefully on our current T4. [Paper](https://arxiv.org/abs/2512.08765) · [Official code and checkpoint notes](https://github.com/ali-vilab/Wan-Move)
- **VIGOR** is retained as a possible static-background/camera-geometry score. Its static-world reprojection reward does not directly supervise an independently moving object, so it is not the primary trajectory policy or evaluator.
- This pilot controls a 2D box path. It does not establish contact, grasp success, object-relative pose, camera calibration, or rigid SE(3) motion. Those need a rendered or simulator-backed fixture with known object pose and separate evaluation.

DiTraj is an **open-loop trajectory-control baseline**, not yet the LatentGate contribution. It applies fixed box-conditioned prompt separation and a fixed early STD-RoPE window; it does not estimate the current object state and adapt its intervention to the resulting path error. The current candidate keeps the hard RoPE geometry fixed and uses a bounded signed residual to update the later foreground/background cross-attention boxes. This changes which spatial tokens receive foreground text; it does not directly translate object pixels. Exact insertion points and control gates are in [`ditraj_feedback_adapter_design_v1.md`](ditraj_feedback_adapter_design_v1.md). It must beat full DiTraj and compute-matched state ablations on held-out seeds and paths; otherwise the extra feedback module is not justified. [Independent method comparison](https://arxiv.org/html/2509.21839v2) · [Frame Guidance](https://arxiv.org/html/2506.07177v2)

## Current paired pilot protocol

This section supersedes the earlier one-way M0-prompt draft. The executable local source is [`ditraj_paired_path_pilot_v1.ipynb`](ditraj_paired_path_pilot_v1.ipynb); its embedded runner has not been run on a GPU. The older `run_ditraj_matched_pilot.py` and `ditraj_kaggle_pilot.ipynb` artifacts are one-way prototypes and are not the current causal protocol.

### Fixed inputs

- Backbone: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, M0 revision `0fad780a534b6463e45facd96134c9f345acfa5b`.
- Shared direction-neutral prompt: “A single glossy bright-red ball on a matte white tabletop, locked-off camera, soft even daylight. The ball moves smoothly and stays fully visible in frame.” Do not name direction.
- Seed 2026 and explicitly shared initial latent noise across all six cases. Use Wan2.1-T2V-1.3B at revision `0fad780a534b6463e45facd96134c9f345acfa5b`, 832×480, 17 frames, FP16 on a T4, and the same negative prompt/scheduler/settings in every arm.
- Mirrored box paths share the same start box: normalized y `[0.50, 0.76]`, x `[0.43, 0.57]` at frame 0; at frame 16, right ends at x `[0.63, 0.77]` and left at `[0.23, 0.37]`. Width/height and vertical center stay fixed. The local notebook records both path definitions and asserts a direction-neutral prompt.
- The one-step smoke runs three controls under both paths (six cases); the separate 30-step screen is disabled until the smoke passes and a fresh free-quota/device check succeeds. `mask_step=18` and `fixRope_step=3` are the initial 30-step adaptation. Record exact package versions, source hashes, model revision, initial-noise hash, scheduler, masks, generation time, and peak VRAM. This is a 17-frame adaptation, not a reproduction of the paper's 81-frame Wan setup.

### Arms

1. **Base:** unmodified Wan behavior through the same pinned DiTraj-compatible runner, with no foreground/background mask or RoPE change.
2. **SG:** foreground/background prompt separation and the box-conditioned cross-attention mask, no STD-RoPE.
3. **Full DiTraj:** SG plus early STD-RoPE and the repeated-token self-attention mask.

For a 30-step smoke run, start with the paper's relative schedule: `mask_step=18` (30/50 of the denoising steps) and `fixRope_step=3` (5/50). Treat those values as an adaptation to be swept only after the smoke run. Keep foreground/background text fixed between SG and full DiTraj; manually split the M0 prompt to avoid making an external LLM response another variable. Keep negative prompt, guidance scale, sampler, noise, and all other settings identical across arms.

### Measurements

The fast secondary measurement uses the same red-pixel threshold as the M0 audit (`R>150, G<125, B<125`) and the evaluator [`evaluate_trajectory_frames.py`](evaluate_trajectory_frames.py). It reports frame coverage, center-path error, bounding-box IoU against the prescribed path, monotonic rightward movement, and apparent mask-area/bbox-size variation. It groups all thresholded red pixels and is not an instance tracker; altered hue, reflections, or a second red region can fool it. Missing-frame coverage is a hard gate, and the pilot runner also reports a missing-frame-penalized score.

For confirmation, the pilot runner creates a **randomly blinded per-frame annotation package** for all six generated cases (102 frames at 17 frames per case). The annotator records visibility, instance count, identity consistency, center, box, confidence, and shape without seeing path direction or arm names. The private unblinding key is stored outside that package. Manual measurements are the primary evaluation; the red-mask score remains a quick secondary check.

Primary metric: mean detected object-center distance to the target center path, normalized by frame diagonal, over valid detections. Co-report target-box IoU and valid-frame coverage. Guardrails: apparent mask area and box width/height coefficient of variation; x-direction monotonicity; scene/background stability; blinded object-identity and video-quality preference. Use OWL-ViT-Large or manually verified boxes as a secondary detector if feasible; do not use only the detector that creates the policy signal.

### Predeclared decision gates

- **Runtime smoke:** all arms return 17 frames; no indexing failure; all use the same initial noise and generation settings; record peak VRAM/time. If it fails or cannot obtain a free GPU, stop before a seed batch and preserve logs.
- **Pilot signal:** full DiTraj should lower mean center error by at least 20% versus SG-only and the unsteered base, with object detection coverage at least 90%. Do not count this one-seed result as a confirmed improvement.
- **Guardrails:** reject a trajectory win if the ball disappears, freezes, changes identity, or if area/box-size drift or background/video-quality review materially worsens. A predicted path that is achieved by making the object a different size or changing the scene is not a pass.
- **Confirmation:** if the smoke passes, repeat paired arms for at least five matched seeds. Report per-seed paired changes and bootstrap confidence intervals; retain failed/unscoreable generations in the report. Only then decide whether to port the method to the M0 Diffusers 0.40.0 runner or build an adaptive latent gate that improves on fixed early-step STD-RoPE.

## M0 baseline anchor and current status

The existing M0 clip uses seed 2026, the same Wan 1.3B checkpoint, 832×480, 17 frames, and 30 steps. Running the new evaluator against the saved frames gives 17/17 valid frames, mean center error 0.148 of frame diagonal against the prescribed path, mean bbox IoU 0.186, 100% rightward center steps, 7.21% red-mask area coefficient of variation, and +25.96% red-mask area from first to last frame. The evaluator output is [`m0_trajectory_baseline_v1.json`](m0_trajectory_baseline_v1.json). These measurements motivate a test but are not a paired DiTraj result and do not prove 3D scale drift. The stated target path deliberately asks for greater rightward travel than that observed baseline while keeping box size fixed.

**No DiTraj generation has started.** The current local source is [`ditraj_paired_path_pilot_v1.ipynb`](ditraj_paired_path_pilot_v1.ipynb), which contains setup/version/preflight cells, the six-case mirrored-path runner, one-step smoke command, disabled-by-default 30-step screen, and result-review cell. A fresh read-only account check of the [Kaggle draft](https://www.kaggle.com/code/bvyashwantkumar/ditraj-wan-1-3b-pilot-research/edit) found only four saved cells (setup, blank, duplicated T4 preflight), so the paired runner is not staged there. The older one-way script and notebook are not the current causal experiment. Local notebook/runner/path-geometry and annotation-package checks pass; none establishes T4 compatibility, runtime, or successful generation. No quota has been consumed for DiTraj, and no paid compute was used.

The latest read-only check of the private [Kaggle draft](https://www.kaggle.com/code/bvyashwantkumar/ditraj-wan-1-3b-pilot-research/edit) found Draft saved, Draft Session off, T4×2 checked, and four saved cells: DiTraj setup, a blank cell, and two duplicate T4 preflights. The runner, bounded smoke, and results/review cells are absent. The local paired-path notebook [`ditraj_paired_path_pilot_v1.ipynb`](ditraj_paired_path_pilot_v1.ipynb) contains that flow but is not staged in the Kaggle draft. No Kaggle runtime has started, no DiTraj quota has been consumed, and no paid compute was used. Do not launch until the saved notebook is reconciled and a fresh free-quota/device check passes.

The CPU geometry smoke on the local helper passed: the target boxes map to a `(5,30,52)` Wan token mask; zero strength is identity; full strength aligns spatial phases while retaining per-frame temporal phase; fractional strength preserves complex RoPE magnitude; and the optimized repeated-token mask matched a direct pairwise reference on a small dynamic-box fixture. This validates the tensor geometry only, not model compatibility or generated output quality.
