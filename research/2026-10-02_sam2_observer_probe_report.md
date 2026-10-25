# SAM 2.1 tiny observer feasibility probe

**Experiment:** `EXP-SAM2-OBSERVER-TRANSFER-PROBE-001`  
**Status:** completed on the local M3/MPS; this is an observer-proposal benchmark, not a video-policy experiment.  
**Implementation:** [`sam2_observer_probe.py`](sam2_observer_probe.py)  
**Machine-readable report and masks:** [`sam2_observer_probe_v1/report.json`](sam2_observer_probe_v1/report.json)

## Why this probe

The current path-feedback plan needs an object-state estimate independent of the commanded trajectory. SAM 2 has a streaming-memory video predictor that propagates a prompted object mask across frames; the official implementation and model checkpoints are Apache-2.0. The tiny checkpoint is available through Transformers, including a video inference API. That makes it practical to check whether it can provide stable masks on our local M3 before spending scarce accelerator quota. ([SAM 2 paper](https://arxiv.org/abs/2408.00714), [official implementation/license](https://github.com/facebookresearch/sam2), [Transformers video API](https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/sam2_video), [tiny checkpoint](https://huggingface.co/facebook/sam2.1-hiera-tiny))

## Setup

- Checkpoint: `facebook/sam2.1-hiera-tiny`, revision `de431c4043854a71d8101e17995dfe596bf101a5`.
- Runtime: PyTorch 2.9.1, Transformers 5.3.0, Apple M3 via MPS, float16. No CUDA, Colab, or Kaggle quota was used.
- One positive point was supplied only on frame 0: the known target-mask centroid on Kubric, and the existing RGB red-threshold centroid on the Wan clip. There were no later prompts or corrections.
- 17 frames were scored in each of two mirrored Kubric renders with exact segmentation masks and the saved 17-frame Wan M0 red-ball sample. The Wan sample has no independent human labels.

## Results

| Sequence | Mask coverage | Mean IoU to known truth | Mean tight-box-center error | p95 center error | Other |
|---|---:|---:|---:|---:|---|
| Kubric left | 100% | 0.750 | 2.64 px | 2.74 px | Predicted mask area 1.30× truth |
| Kubric right | 100% | 0.750 | 2.73 px | 2.92 px | Predicted mask area 1.30× truth |
| Wan M0 | 100% | Not available | Not available | Not available | SAM2 vs red-threshold centers differ by 4.91 px on average; mask IoU to that red proxy is 0.870 |

On the known-truth synthetic sequences, the propagated masks cover every frame, but the center is consistently biased downward by about 2.5 px; mean horizontal bias is about −1 px. For a normalized mapping from the 160×96 fixture to Wan's 832×480 / 52×30 token grid, this vertical bias is about 12.5 px in Wan pixels, greater than the planned half-cell tolerance of 8 px. The mask also includes roughly 30% more area than the ground truth. Thus this run does **not** pass the current strict observer localization gate, despite visually stable tracks and respectable IoU.

The saved Wan sample demonstrates that the checkpoint runs on our generated frames and returns full-frame masks. Its center trajectory closely agrees with the red threshold across frames (Pearson correlation 0.9999 in x and 0.9977 in y; endpoint displacement +32.0,+5.7 px versus +32.6,+6.3 px). This is still proposal-to-proposal agreement on the same unlabelled clip; it cannot establish that either proposal marks the true object boundary or center. The output contact sheets are [`Kubric left`](sam2_observer_probe_v1/kubric_left/contact_sheet.png), [`Kubric right`](sam2_observer_probe_v1/kubric_right/contact_sheet.png), and [`Wan M0`](sam2_observer_probe_v1/wan_m0_red_ball/contact_sheet.png).

Measured end-to-end tracking time ranged from 15 to 77 seconds per 17-frame sequence across the three runs. The timing varied substantially and is a local feasibility measurement, not a stable throughput estimate. The first uncached model fetch was about 156 MB; subsequent loads were from local cache.

## Decision

Keep SAM 2.1 tiny as an optional **mask proposal** for blinded human correction and for stress-testing measurement pipelines. Do not use its raw center as the live controller observer yet: it misses the current half-cell localization threshold on the low-resolution known-truth fixture and has systematic vertical bias. Do not use its high apparent agreement with the M0 red mask as validation. The next observer step is still independent human tight-box/identity/visibility labels on generated Wan previews, followed by a frozen observer comparison on held-out seed blocks.

This experiment does not test DiTraj box actuation, latent feedback, generation quality, or policy efficacy. The architecture audit identifies an even earlier causal prerequisite: test whether a one-token-cell cross-attention-mask offset moves the generated object predictably before interpreting any residual-feedback law. See [`2026-10-02_same_sample_feedback_architecture_audit.md`](2026-10-02_same_sample_feedback_architecture_audit.md).
