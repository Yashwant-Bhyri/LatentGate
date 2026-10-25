# M1 — Controlled failure atlas

**Status:** execution materials prepared; no M1 generation has started. On 2026-10-15 the designated Colab profile allocated a T4 in notebook cells and confirmed PyTorch CUDA, but the session showed no billing receipt. The versioned [M1 package-stack decision](../research/2026-10-15_m1_package_stack_decision.md) now pins the observed `torch==2.11.0+cu130` and its official `torchvision==0.26.0+cu130` pair. M1 remains unrun pending explicit no-paid admission, refreshed/hash-checked Drive notebook and bundle, and guarded preflight. The single designated Kaggle T4×2 account remains assigned to the sibling DiTraj pilot; per the user's instruction and Kaggle's one-account Terms of Use, M1 will not use Kaggle or the separately observed handle. The one-clip M1 smoke is independent of DiTraj/Kubric Gate 1 under the [scope amendment](../research/2026-10-15_m1_smoke_gate_scope_amendment.md).

## Goal

Build a seed-controlled dataset for the fixed-camera tabletop task in the research specification: a robot or gripper picks up one mug-like object and places it into a holder. For each clip, preserve the final video and selected model internals, then attach independently measured geometry, identity, trajectory, camera, and contact outcomes. The first analysis asks when a future failure becomes predictable from the generator's intermediate state.

M0 establishes that the Wan2.1 T2V 1.3B microscope can produce and record a reproducible clip. It does not establish that its internal features encode these task variables, nor that the video is physically correct.

## Provisional generator choice

Use **CogVideoX-5B-I2V** for the first one-clip M1 smoke test, subject to the T4 and instrumentation gates below. The official model card provides a Diffusers image-to-video pipeline, a 720×480 / 49-frame example, VAE slicing and tiling, and sequential CPU offload. The CogVideoX repository links a free-Colab-T4 I2V notebook and documents quantization for smaller GPUs. Its published memory measurements are not a T4 guarantee, and the standard example uses BF16; run the T4 check in FP16 or a specifically validated quantized configuration, then record actual peak memory and time. The earlier CogVideoX-2B T2V pilot reduces framework setup risk, but this 5B I2V transformer is a different instrument target.

- First smoke test: one mug-and-holder starting image, fixed prompt, one seed, 720×480, 49 frames, and the official I2V step count unless the live T4 budget requires a documented reduction. Save the exact input image and checksum. Do not start the 8-seed batch until output quality, memory, runtime, and capture coverage are reviewed.
- A proposed common-start image is now saved at [`assets/mug_holder_start_v1_720x480.png`](assets/mug_holder_start_v1_720x480.png), with the original source and exact prompt/video condition recorded in [`scene_fixture_v1.json`](scene_fixture_v1.json). This generated still is a visual fixture, not simulator ground truth; review object visibility and gripper placement before using it as the frozen condition.
- The draft smoke settings are pinned in [`run_config.yaml`](run_config.yaml), including model commit `a6f0f4858a8395e7429d82493864ce92bf73af11`, the T4 FP16/offload strategy, seed 2026, and the planned replay. Status remains `planned_not_run`.
- Instrumentation gate: port the packet and captures to this pipeline, then verify callback timing, post-step latent meaning, scheduler-returned predicted-clean semantics, selected block outputs, attention-summary bounds, and same-seed replay. Wan's UniPC hook is not portable by assumption.
- If the CogVideoX smoke test cannot complete on the assigned T4 or cannot be instrumented reliably, assess **HunyuanVideo-1.5 480p I2V** next. Tencent's official repository lists a 14 GB minimum with model offloading, leaving little headroom on this T4, and its transformer requires a larger instrumentation adapter. Defer **Wan2.2 TI2V-5B** because its official single-GPU path requires at least 24 GB. Keep **LTX-Video 2B** on the watchlist until the exact I2V checkpoint, Diffusers route, and T4 memory use are verified.

Official references: [CogVideoX-5B-I2V model card](https://huggingface.co/zai-org/CogVideoX-5b-I2V), [Diffusers CogVideoX guide](https://huggingface.co/docs/diffusers/main/using-diffusers/cogvideox), [CogVideoX T4/I2V notebook index](https://github.com/zai-org/CogVideo), [HunyuanVideo-1.5 hardware and I2V models](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5), [Wan2.2 TI2V-5B model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), and [Diffusers LTX-Video API](https://huggingface.co/docs/diffusers/main/api/pipelines/ltx_video).

## Gate 0: lock the scene and conditioning contract

Choose one pinned generator and a supported scene setup before making a large batch.

- Prefer one shared starting image if an image-conditioned candidate fits the available compute and can be instrumented. Reuse the same image, prompt, scheduler, resolution, and frame count while varying only the seed.
- If starting with T2V, keep the prompt and camera description fixed, measure initial-scene variation, and decide whether that variation is acceptable for a controlled atlas. Wan T2V does not accept one common first-frame condition.
- Record model revision, package versions, sampler semantics, offload strategy, frame count, resolution, seed policy, and measured generation time/peak memory.
- Before scaling, verify the selected pipeline's callback timing, predicted-clean semantics, transformer-block access, and replay behavior. M0's UniPC hook is pipeline-specific and should not be assumed to transfer unchanged.

**Exit:** a short feasibility batch shows that the task is visibly present often enough to score and that the instrumentation produces valid packets for the selected model.

## Gate 1: define independent labels

Version the evaluator implementations and preregister thresholds before producing the full atlas. Save continuous scores as well as binary labels; use human adjudication for ambiguous cases.

| Failure family | Initial measurements | Candidate independent evidence |
|---|---|---|
| Geometry / scale | object size and depth consistency over time | object masks, depth or point maps, reprojection checks |
| Identity | mask/feature continuity and identity switches | segmentation and feature tracks |
| Trajectory | object and gripper path error, speed changes, jitter | point tracks and detected pose/keypoints |
| Camera | background motion and viewpoint drift | static-scene tracks or camera-motion estimate |
| Grasp / contact | grasp-before-lift, proximity, detach/slip | gripper/object detections and temporal contact heuristics |
| Visual-only | blur, flicker, or texture instability | image-quality measures reviewed against clips |

Treat teacher-model scores as measurements with known limitations, not ground truth. Keep a human-review set to estimate label error. Start with the task axes the chosen generator and evaluators can actually observe; defer unsupported labels.

### Smallest feasible label stack

Run teachers **after generation and sequentially**, with the generator unloaded. For the pilot, use SAM 2.1 Hiera Tiny to prompt and propagate mug and holder masks, CoTracker3 offline to track a few manually selected mug surface points plus background points, and human review for ambiguous frames and stage events. Keep raw continuous measurements and annotate visibility; do not force labels where the mug or gripper is occluded.

Initial usable measurements are mug-mask presence, centroid, bounding box and apparent area; point-track survival and 2D path/jitter; coherent background motion as a screen-space camera-drift proxy; and reviewed pick/lift/place stages with visible overlap or adjacency. A one-mug scene only supports mug-track continuity, not identity-switch rates across similar objects. Describe contact as **visible/apparent contact**. RGB-only clips cannot establish force, true 3D scale, or physical contact. Mask area is affected by pose and occlusion; background drift can be generator warping rather than camera motion.

Optional depth check: Video Depth Anything Small is a candidate only after the core labels work. Its official 6.8 GB FP16 figure is for a 32-frame 518×518 input on an A100, not a T4 benchmark; treat output as relative visual evidence, not geometry ground truth. Defer VGGT until a specific 3D question justifies its greater cost and licensing review. Check the current licenses of CoTracker and every selected checkpoint before distributing artifacts.

References: [SAM 2 video predictor and Hiera checkpoints](https://github.com/facebookresearch/sam2), [CoTracker3 offline tracking](https://github.com/facebookresearch/co-tracker), [Video Depth Anything memory table](https://github.com/DepthAnything/Video-Depth-Anything).

## Gate 2: pilot, then scale

1. Run an 8-seed feasibility batch with the final prompt, scene condition, runtime, instrumentation, and evaluator versions.
2. Review every clip and intermediate preview; check whether the chosen scene remains recognizable and whether each proposed metric agrees with human review.
3. Freeze the run config and evaluator thresholds. If a fixed-start or labeling requirement fails, revise the model/condition/evaluator choice and repeat the pilot.
4. Expand to the proposed 50–100 seeds only after the pilot passes. Split all training/validation/test data by seed, never by frame from the same clip.

Do not reuse M0's 2.2–4.4 GPU-hour estimate for the M1 generator. CogVideoX-5B-I2V with CPU offload is a different workload; estimate the batch from measured T4 smoke-test time, then add teacher-evaluation time separately.

For each seed, keep the prompt and scene condition, final RGB video, runtime packet, output hashes, approximately eight denoising checkpoints, predicted-clean previews where supported, selected hidden-state summaries across approximately six to eight blocks, and bounded attention diagnostics. Record teacher scores, labels, reviewer notes, and the exact evaluator versions alongside the packet.

## M1 exit criterion

At least one preregistered failure axis has reliable labels and enough independent positive and negative examples for a held-out-by-seed predictive probe. Report class balance, label agreement, and the held-out split. Do not interpret probe accuracy as causal control; M2 and M3 ask separate questions.

## Immediate decisions

**Project execution order:** DiTraj remains the only assigned Kaggle GPU lane. M1 may run only in the authorized Colab profile after Colab explicitly confirms a no-paid T4 path, the expected-versus-observed Torch pin is resolved by a versioned decision, the Drive notebook and bundle are refreshed/hash-checked, and the guarded T4/package preflight passes. The Oct 15 allocation confirmed hardware but not billing or M1 compatibility. Do not move M1 to either Kaggle handle or wait to reuse the DiTraj account. Review runtime, memory, captures, and the blinded scoreability review before sizing any seed batch. The VIGOR reward audit remains secondary and unscheduled. Expand M1 only after its smoke and evaluators pass review.

**M1 follow-up sequence:**

1. Review the proposed scene fixture and freeze the image and action prompt, or make a versioned replacement.
2. If an independently verified free Colab T4 is available, confirm the fixture and CogVideoX-5B-I2V T4 FP16/offload load path. If unavailable, keep M1 queued for the authorized Colab profile and continue CPU-only preparation; do not use the designated Kaggle DiTraj account. Then use the pinned pipeline-specific capture adapter.
3. Measure one smoke sample's runtime, peak memory, replay behavior, and full latent, hidden-summary, spatial-map, and predicted-clean capture coverage.
4. Review the smoke and candidate labels before authorizing the 8-seed feasibility pilot with SAM 2.1, CoTracker3, and human labels; freeze evaluator thresholds only after this review.
5. Produce a reviewed experiment ticket and config before starting the 50–100-seed batch.

## Prepared smoke-test adapter and historical preflight notes

The one-clip execution ticket is [`geometry_scale_drift_ticket.json`](geometry_scale_drift_ticket.json). The runner in [`run_m1.py`](run_m1.py) records callback latents as post-scheduler-step states, pools six evenly spaced transformer blocks by latent-time position, and retains the classifier-free-guidance batch without interpreting its branch order. It also saves three 8×12 spatial maps at three denoising steps from three evenly spaced blocks, which tests whether local mug/holder layout survives beyond global pooled summaries, and six scheduler-returned predicted-clean (x_0) snapshots, which test whether cleaner intermediate samples contain useful structure. The added spatial maps require roughly 131.6 MiB per pass; six predicted-clean snapshots require about 12.9 MiB in FP16 or 25.7 MiB in FP32, for approximately 144.5–157.3 MiB of added tensor artifacts per pass. Runtime overhead is not measured. Baseline/replay coverage and exact-equality checks now include these tensors as well as RGB frames, post-step latents, and pooled hidden summaries. Attention summaries remain deferred. The package pins, capture locations, and T4 gate are in [`run_config.yaml`](run_config.yaml).

A Colab notebook and upload bundle are prepared at [`colab/LatentGate_M1_CogVideoX5B_I2V.ipynb`](colab/LatentGate_M1_CogVideoX5B_I2V.ipynb) and [`colab/latentgate_m1_bundle.zip`](colab/latentgate_m1_bundle.zip). The notebook has been uploaded to the requested Colab account's Drive, but the bundle has not. Colab's Resources panel reports no subscription and zero compute units; T4 is selected. All three T4 allocation attempts on 2026-10-02 were denied due to usage limits, so no runtime or model run started. Do not proceed until Colab grants a T4 without paid units, then require compute capability 7.5, matching package pins, and the guarded preflight before loading model weights.

The first geometry readout is explicitly a triage layer: manually reviewed visibility/stage labels, image-space mug/holder mask ratios, shape proxies, tracked fixture landmarks, and optionally relative-depth-corrected area. Since the shared-start fixture has no calibration or physical ground truth, these measures cannot establish metric 3D scale, physical contact, or causal steering. Do not start the eight-seed atlas until the smoke sample, runtime, memory, capture coverage, replay, and evaluator behavior are reviewed.

### Current M1 release blockers — 2026-10-15

The designated account allocated a Tesla T4 in a notebook session and confirmed PyTorch CUDA, then disconnected; no project inference or billing receipt occurred. This confirms hardware availability at that time, not a no-paid billing path. The local Torch/TorchVision mismatch is resolved by the [versioned package-stack decision](../research/2026-10-15_m1_package_stack_decision.md). Refresh and hash-check the Drive notebook and bundle, then require explicit no-paid admission plus the T4/CC7.5/package preflight. The separate DiTraj/Kubric Gate 1 does not block this single instrumentation smoke under the [scope amendment](../research/2026-10-15_m1_smoke_gate_scope_amendment.md); it remains mandatory for the P4/DiTraj policy lane.

Pinned implementation references for the current adapter are the [Diffusers v0.40.0 CogVideoX I2V pipeline](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py), [CogVideoX transformer](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/cogvideox_transformer_3d.py), and [CogVideoX DDIM scheduler](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_ddim_cogvideox.py).
