# LatentGate research chat: full report recovery and project reconciliation

> **Historical decision memo (2026-10-11).** Current execution state is in [README.md](../README.md) and [STATE.md](../STATE.md). The later M1 scope amendment separates its instrumentation smoke from the DiTraj/Kubric Gate 1 lane.

**Reviewed:** 2026-10-11  
**Source:** ChatGPT project chat [Research LatentGate Methods](https://chatgpt.com/g/g-p-6abe460005e88191944a9d8c2587ab7e-robotics/c/6abec4eb-776c-83ea-a384-cd31853729cc), conversation `6abec4eb-776c-83ea-a384-cd31853729cc`  
**Status:** Full rendered content reviewed in the existing browser tab. The app's thread reader clips report items at 20,000 characters. The browser view contained two complete reports: **“LatentGate: Primary-Source Research Report”** and a later expanded **“LatentGate: Research Landscape, First-Experiment Decision, and Closed-Loop Validation Plan.”** The chat reports no attachments. The first report is dated October 11; the expanded report states a research cutoff of October 1, 2026.

This record supersedes the earlier “readable excerpt only” limitation in [the initial incorporation memo](2026-10-11_deep_research_report_incorporation.md). The report author could not inspect this local workspace, so its local implementation and calibration statements are context supplied to that chat, not independently rerun evidence. The repository artifacts remain authoritative for current status.

## Decision from the research

Keep **DiTraj as the first open-loop actuator foundation** for fixed-camera, single-object 2D path control on Wan2.1 T2V 1.3B. This is a fit-to-backbone and auditable-code decision, not a claim that DiTraj has closed-loop feedback or that it is the strongest controller in general.

The narrow research hypothesis is: during one frozen T2V denoising run, can an estimate of the generated object's current or predicted path residual change a later trajectory-control action and causally improve the final path? The report says this remains unanswered. It distinguishes:

1. **Observer mechanics:** a red-object observer scored well on a small synthetic Kubric calibration set; this does not establish transfer to Wan-generated videos.
2. **Actuator implementation:** an opt-in predicted-clean capture adapter and smoke runner are implemented; CPU equality is plumbing evidence, not a GPU pass.
3. **Policy efficacy:** no DiTraj feedback video, trajectory gain, or closed-loop efficacy result has been measured.

The first report recommends an exact no-op/actuator smoke, then independent observer validation on generated Wan clips, then matched causal testing. The expanded report instead lays out evaluator validation, M0 reproduction, open-loop DiTraj pairs, state observability, and only then feedback. Both reject broad claims based on trajectory conditioning alone.

## Source-level implementation findings

The first report identifies DiTraj commit [`6fa8e613c9e7c40ec2db456ab2ba34367a076886`](https://github.com/xduzhangjiayu/DiTraj/tree/6fa8e613c9e7c40ec2db456ab2ba34367a076886) and Diffusers [`v0.33.1` UniPC source](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/schedulers/scheduling_unipc_multistep.py) as its inspected implementation baseline.

- DiTraj builds cross-attention masks before the denoising loop. Its callback runs after `scheduler.step`, so editing a box in the callback alone does not change the mask for the next transformer call. A live update needs persistent active-box state and a mask rebuild before a subsequent controlled call; log before/after rasterized masks and quantized-token changes.
- Preserve and explicitly log the early RoPE geometry. The upstream `run.py` default for `fixRope_step` is 5; that is not the same as the local protocol's selected first-three-step hold.
- With 832×480 output, DiTraj's spatial grid is 52×30. A 17-frame clip maps to five latent-time control positions anchored at output frames 0, 4, 8, 12, and 16. Upstream examples use 81 frames, so the 17-frame adaptation needs its own runtime validation.
- For the cited flow-prediction UniPC path, the predicted-clean conversion is `x0_hat = x_t - sigma_t * v_theta(x_t,t)`. This makes capture plausible; it does not show that a decoded preview estimates final object location. The local CPU equality result does not establish GPU timing, dtype, memory, or observer validity.
- The report identifies DiTraj's code as **CC BY-NC 4.0** and the Wan Diffusers checkpoint as **Apache-2.0**. Keep code and model terms distinct.

## Literature map and novelty boundary

The report's primary-source comparison includes DiTraj, FreeTraj, Motion-Zero, Tora, SG-I2V, TrajLoc, Wan-Move, MagicMotion, FlexTraj, WorldForge, ATI, QWERTY, FlashMotion, MotionStream, 4DStreamCtrl, Geometry Forcing, VGGRPO, PSIVG, Track4Gen, and tracking/geometry sources such as CoTracker, TAP-Vid, and VGGT. It includes Chinese and Hong Kong work, including Motion-Zero, Tora, MagicMotion, FlexTraj, WorldForge, and DiTraj.

The report treats the following as occupied prior-art categories: generic path-conditioned video generation; training-free path control and attention steering; structured-noise motion control; test-time latent optimization/self-guidance; per-object attention heatmaps; geometry-aware diffusion features; teacher/student streaming motion control; and external-simulator-in-the-loop generation. In particular, it flags **SG-I2V** for latent correction, **TrajLoc** for per-object attention heatmaps, **FreeTraj/Motion-Zero/DiTraj** for training-free trajectory control, **PSIVG/WorldForge** for broader simulator or denoising self-correction claims, and **Geometry Forcing/VGGRPO** for geometry-aware representations or rewards.

The defensible candidate claim is narrower: an observed, generated-state residual is compared with a requested path and causally changes a later denoising action within the same run; its benefit must disappear or reverse under matched zeroed, shuffled, temporally mismatched, fixed-action, or sign-reversed controls. This would initially demonstrate state-dependent feedback for one constrained image-plane task, not general physical reasoning or robot control. Reported memory, speed, and benchmark results from other papers remain author results, not local reproductions; several projects have different I2V backbones, larger checkpoints, or incomplete/unreleased code.

## Measurement and proposed experiments

The reports recommend framewise independent labels across all 17 RGB frames: visibility, identity, tight box/center, size, and failure reason. Keep raw absolute start error separate from first-frame-anchored drift; penalize missing or wrong-identity frames; report visibility/identity coverage and size/background guardrails. A point tracker follows a surface point, not necessarily a rolling sphere's projected center, so it cannot replace object-center labels.

The report proposes ten no-policy Wan clips from five mirrored seed pairs for observer transfer, two blinded raters, and a fresh holdout if thresholds are tuned. Suggested observer criteria include ≥90% valid visible/identity observations and controller-keyframe coverage, median center error ≤5 px and p95 ≤10 px, raw start error ≤8 px in at least 9/10 clips, and frozen thresholds. These are proposed engineering cutoffs, not established field standards.

For efficacy, it recommends matched open-loop, exact no-op, fixed/static correction, live residual feedback, shuffled residual, and sign-reversed residual arms with the same model, prompt, initial noise, sampler, schedule, observations, and compute. The reports propose at least a 15% paired reduction in failure-penalized path error, mirrored-direction robustness, guardrails for visibility/identity/size/background, and uncertainty intervals. The expanded report further proposes at least 12 matched cases, a direction-reversal path, and explicit zeroed-state controls. These are prospective proposals, not measured effects.

## Reconcile with the local project gates

Keep the repository's registered gate order and thresholds until the project deliberately preregisters a replacement. The report is research input, not an instruction to bypass current gates:

- **At this memo date, Gate 1 was the first local DiTraj/P4 gate.** Two independent raters must label both mirrored 17-frame Kubric clips; a passing one-pair process screen still needs a second independently rendered mirrored-seed block. The later M1 scope amendment made its one-clip instrumentation smoke independent of Gate 1. Gate 1 remains open today pending two independent complete v5 human returns; consult the current registry for the latest scan and separate M1 release gates.
- **The reports use conflicting timing proposals.** One recommends a step-4 predicted-clean update; the expanded report suggests observing predicted-clean RGB at steps 10, 20, and 26 and treats feedback timing as a later choice. The local P4 adapter's step-4 schedule remains its own preregistered design; do not silently combine or retune these proposals.
- **The reports use different experiment names.** Their `M1` means open-loop DiTraj. The local registry uses **P4** for DiTraj and **M1** for a separate CogVideoX-5B-I2V diagnostic. Use P4/DiTraj when discussing the local active lane.
- **The local evidence is narrower than either proposed protocol.** M0 is reproducible; its image-space proxies are not metric geometry. The synthetic observer calibration is not generated-video validation. Adapter implementation and CPU checks are not a GPU smoke or policy gain. Preserve these distinctions in every status note.

## Handoff recorded on 2026-10-11

The next project action is still the local earliest open gate: return two independent annotations for the v5 mirrored Kubric packets, then score with the current resolution-aware scorer and registered requirements. The separate DiTraj chat completed the scorer correction and its CPU-only validation. The v5 UI enlarges 160×96 source frames for labeling while preserving source-coordinate scoring into 832×480 / 52×30 target units. A half-cell p95 threshold is therefore 1.5385 raw x-pixels and 1.6 raw y-pixels for Kubric. The synthetic diagnostic correctly rejects a weak left mirror (15 target px, 0.9375 cell) even though the pooled p95 is 5.25 target px; it also checks missing-clip and out-of-frame rejection. All four v5 archives pass hash and ZIP checks. A focused browser smoke displayed a frame at 622 CSS pixels, round-tripped an exact source coordinate through the SVG, and exported the expected 35-row CSV; it did not save CSV data or enter labels. All v3–v5 packet templates remain blank. This is scorer and packet-mechanics validation, not human reliability, generated-video observer accuracy, or policy evidence. See [the completed Gate 1 coordinate-scale audit](2026-10-11_gate1_coordinate_scale_audit.md).

The designated `y` Chrome profile is now directly verified as `yashwant's tech tales (yashwantkumarbhyri@gmail.com)`. On 2026-10-11 14:32 UTC, Colab showed no subscription, zero compute units, no active sessions, and no connected runtime; the runtime selector had T4 GPU selected, while free-resource availability was explicitly not guaranteed. At 14:35 UTC, one connection-only check succeeded and the backend identified itself as `T4 (Python 3)` with 0/15 GB GPU RAM in use. The UI still showed zero units and did not explicitly identify this allocation's billing class. No notebook cell ran, no bundle was uploaded, and no weights were loaded; the runtime was immediately disconnected. This proves one T4 allocation was available at that time, not future availability, free billing, or permission to run the gated experiment. The currently open Drive copy was last edited October 2 and predates the local manifest-hash wrapper update; refresh it from the local notebook before any eventual M1 run. See [the M1 Colab preflight log](../m1/colab_preflight_2026-10-02.md).

No report attachment was present in the source chat. The primary report and expanded version remain accessible there; this file is a full decision-and-gate reconciliation, not a verbatim export.
