# Runnable stack audit: fixed-camera object-path control

**Checked:** 2026-10-02. This is an implementation-feasibility audit, not evidence of a steering gain.

## Recommendation

Use **Wan2.1-T2V-1.3B + DiTraj** for the first matched baseline, then add the predicted-clean feedback intervention to the same Wan sampling loop. Use a short **17-frame, 832×480, 30-step** clip, a manually reviewed fixed foreground/background prompt split, and same-seed, same-start mirrored paths. Save decoded PNG frames plus the short MP4. This reuses the project’s Wan M0 and the only directly runnable DiTraj stack.

This is a research adaptation, not an untouched reproduction: DiTraj’s public entry point defaults to 81 frames/50 steps, its released implementation is Wan-specific, and its fixed-RoPE mask creates a large quadratic intermediate. Keep Tora as a later independent trajectory-control baseline. Keep Wan-Move as a later high-memory point-control reference.

## Candidate stacks

| Stack | Runnable path / compute | License and practical fit |
|---|---|---|
| **Wan2.1-T2V-1.3B** | Official code and Diffusers weights; 480p T2V. Wan reports 8.19 GB VRAM and ~4 minutes for a 5-second 480p clip on RTX 4090 without quantization. This is base-model guidance, not a DiTraj peak. | Code and model weights Apache-2.0. Best match for existing M0. DiTraj hardcodes CUDA, so the local M3/MPS machine is not the intended execution lane. [Official repo](https://github.com/Wan-Video/Wan2.1), [1.3B model card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |
| **DiTraj on Wan** | Official run script loads Wan2.1-T2V-1.3B, custom Wan pipeline/attention processor, BF16, CUDA, 832×480, 81 frames. It leaves inference steps at pipeline default 50; mask conditioning defaults to the first 30 steps and RoPE adjustment to first 5. | Repo says CC BY-NC 4.0, research use only/no commercial use. README says it can adapt to CogVideoX, but current released run path is Wan-specific: Wan classes, Wan transformer patch, Wan RoPE/masks. Treat Cog support as a port, not an available switch. [Repo](https://github.com/xduzhangjiayu/DiTraj), [run.py](https://github.com/xduzhangjiayu/DiTraj/blob/main/run.py), [pipeline](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/pipe.py) |
| **CogVideoX-2B** | Official Diffusers inference for T2V/V2V and supported I2V; supports sequential CPU offload, VAE slicing, and tiling. No universal peak-VRAM claim in the official runner. Offload is slower. | 2B weights/modules Apache-2.0. Good for a separate 2B smoke run, but an intervention would need a CogVideo-specific attention/conditioning port and validation; not a ready DiTraj alternative. [Official code](https://github.com/zai-org/CogVideo), [model card](https://huggingface.co/zai-org/CogVideoX-2b), [runner](https://github.com/zai-org/CogVideo/blob/main/inference/cli_demo.py) |
| **Tora** | Official trajectory-conditioned code and weights. README gives ~30 GiB for SAT inference. Separate Diffusers path says sequential CPU offload can reduce GPU memory to ~5 GiB, much slower. A possible later T4 comparison. | Repository code Apache-2.0; weight download is subject to CogVideoX license and README says its CogVideoX-5B release is for academic research. Strong independent open-loop trajectory baseline, but different model/pipeline. [Repo](https://github.com/alibaba/Tora), [Diffusers README](https://github.com/alibaba/Tora/blob/main/diffusers-version/README.md) |
| **Wan-Move** | Official Wan I2V extension, point-level trajectories, model, MoveBench labels and eval code. Official README says its 14B model needs one 40-GB GPU even with CPU/T5 offload. | Apache-2.0 model terms. Useful later on larger research compute; I2V/14B is not an apples-to-apples extension of this 1.3B T2V pilot. [Official repo/benchmark](https://github.com/ali-vilab/Wan-Move) |
| **SG-I2V** | Official zero-shot I2V trajectory-control code and notebook; authors report testing on RTX A6000 48 GB. Quantitative evaluation code was still in progress in the checked README. | MIT. Important related work, but not the smallest documented stack for this pilot. [Paper](https://arxiv.org/abs/2411.04989), [code](https://github.com/Kmcode1/SG-I2V) |

## DiTraj risks that affect the run

1. The released script imports Wan-specific classes and hard-selects CUDA. The README’s CogVideoX adaptability statement is broader than the runnable repository. The pipeline is pinned to a custom Diffusers 0.33.1 setup, with instructions to replace the Diffusers Wan transformer file. Keep that environment isolated and record its exact revision.
2. The pipeline computes fixed-RoPE masks before denoising and rebuilds a local attention kwargs dictionary in each step. A live feedback intervention must update the future path and refresh derived masks inside the denoising loop; changing the initial box argument after initialization will not feed back.
3. *run.py* accepts frame count but does not pass a CLI step count to the pipeline, so it defaults to 50. Add explicit step-count control before claiming a 30-step comparison.
4. The code’s RoPE mask uses dense pairwise equality over all spatial-temporal tokens. At 832×480 its box grid is 30×52. With 81 output frames, Wan latent time is 21, so N=32,760; a dense N×N boolean mask is ~1.07 billion entries (~1.0 GiB at one byte/entry), and the intermediate N×N×3 equality tensor is ~3.22 billion booleans (~3.0 GiB), before other buffers. The code builds coordinate/equality tensors on CPU and moves the overlap mask to CUDA. This is a source-level size estimate, not a profiled peak.
5. With 17 frames, latent time is 5 and N=7,800; the two corresponding dense arrays are ~61 MB and ~183 MB. Shortening the clip makes the T4 lane substantially more plausible, but does not guarantee fit or speed. If it still OOMs, replace all-token pairwise equality with unique/index grouping of repeated coordinates.
6. The run script reads a foreground/background prompt JSON; its optional refinement helper loads Qwen3-8B and requests up to 32,768 new tokens. Skip that second model for this fixed-object prompt and manually supply the reviewed prompt split.
7. Because the pipeline returns decoded frames, PNG-frame inspection does not require generating a long video. The model still denoises the short clip jointly; these are temporally conditioned frames, not independent single-image runs.

Code checked: [sampling entry point](https://github.com/xduzhangjiayu/DiTraj/blob/main/run.py), [mask construction and loop](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/pipe.py#L77-L145), [attention processor](https://github.com/xduzhangjiayu/DiTraj/blob/main/module/attention_processor.py), [prompt helper](https://github.com/xduzhangjiayu/DiTraj/blob/main/prompt_refine.py).

## Measurement and public tools

Keep hand-labeled boxes, identity/presence/visibility, and fixed background landmarks as generated-video ground truth. A tracker’s output is a proposal or cross-check, not ground truth.

- **SAM 2.1** can propagate a promptable video mask from a first-frame point/box; use its mask centroid as a second observer and have a human audit splits, merges, and disappearance. The optimized install uses CUDA (the extension may be skipped with limited post-processing). Repo lists Apache-2.0 code/checkpoints. [Official repo](https://github.com/facebookresearch/sam2)
- **TAPNext++/TAPIR** tracks surface points and predicts visibility/occlusion. A few points can independently flag tracking failures, but do not replace the object center: surface points move with rotation/deformation. The TAP repo provides Colab demos and TAP-Vid metrics; it states Apache-2.0 for code and listed weights, and CC BY for TAP-Vid annotations. [TAP repo](https://github.com/google-deepmind/tapnet), [TAP-Vid](https://tapvid.github.io/)
- **Kubric** provides synthetic RGB with instance masks, depth, and flow; code is Apache-2.0 and uses Blender/PyBullet. Use fixed-camera renders with exact projection/mask centers to validate the scorer/observer, not as proof of performance on generated Wan samples. [Repo](https://github.com/google-research/kubric), [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.pdf)
- Check camera stability primarily with labeled background landmarks. RAFT-style background flow is a secondary diagnostic; near-zero flow in textureless regions does not prove a stable camera.

The existing [fixed-camera evaluation audit](2026-10-02_fixed_camera_path_evaluation_audit.md), Kubric fixture, and planned human reliability labels already cover the scorer/observer gates. Complete those before adding more benchmark infrastructure.

## Execution lane and gate order

Reuse the project’s existing NVIDIA T4 runtime for a short run after checking GPU, host RAM/disk, and cached weights. Colab is suitable when it allocates a GPU, but Google states GPU type and usage limits vary and are not guaranteed. Its current FAQ says eligible paid Google AI plans include monthly Colab compute units and access to stronger GPUs; confirm actual balance and hardware in the signed-in Colab settings. The plan name alone does not determine available runtime or cost. [Colab resource FAQ](https://research.google.com/colaboratory/faq.html)

1. **Smoke:** one 17-frame Wan sample and one 17-frame open-loop DiTraj sample; save PNGs, timing, peak GPU memory, host RAM, versions, prompt, and seed.
2. **Mechanism:** one same-seed DiTraj run with one predicted-clean observation and bounded update to future path conditioning; verify denoising continues after the intervention.
3. **Pilot:** five paired seed/case blocks. Match starting latent, prompts, path start, sampler, step count, and scene; mirror path direction. Include shuffled/sign-reversed feedback on a fixed subset.
4. **Confirm:** only after observer reliability passes, freeze observer/controller and run at least ten new paired blocks over multiple scene/path cases. Analyze per-block paired deltas, not frames as independent samples.

**No model was run, no Colab account was accessed, and no compute was consumed for this audit.** The mask figures are derived from the public source dimensions and allocation pattern; peak memory and run time still require a smoke profile.

