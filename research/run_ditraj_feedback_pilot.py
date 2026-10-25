#!/usr/bin/env python3
"""Small one-path probe of a step-4 x0-hat feedback gate in DiTraj/Wan.

This keeps the paired open-loop runner unchanged. It runs a full DiTraj
baseline, a decode-matched zero-update control, and one feedback arm from the
same noise. The default eight-step screen is for checking the intervention
path, mask change, runtime, and VRAM only; it is not an efficacy benchmark.

Run from the exact pinned DiTraj checkout after the paired-path setup has
installed DiTraj's Diffusers transformer copy and required dependencies:

    python run_ditraj_feedback_pilot.py \
        --output-dir /kaggle/working/feedback_smoke \
        --admission-receipt /path/to/fresh-receipt.json \
        --notebook-sha256 <reviewed-local-notebook-sha256>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # flat imports on Kaggle; package imports in the local research workspace
    from . import run_ditraj_paired_path_pilot as baseline
    from .ditraj_admission import validate_admission_receipt
except ImportError:
    import run_ditraj_paired_path_pilot as baseline
    from ditraj_admission import validate_admission_receipt


ARM_NAMES = ("ditraj_baseline", "decode_matched_noop", "x0_hat_feedback")


def make_path_mask(path_name: str, *, plan_path: Any, device: torch.device) -> torch.Tensor:
    height, width = baseline.HEIGHT, baseline.WIDTH
    anchors = baseline.PATHS_YXYX[path_name]
    latent_frames = baseline.NUM_FRAMES // 4 + 1
    box_h, box_w = height // 16, width // 16
    pixel_path = plan_path(anchors, video_length=baseline.NUM_FRAMES)[::4]
    if len(pixel_path) != latent_frames:
        raise RuntimeError(f"Expected {latent_frames} latent boxes for {path_name}, got {len(pixel_path)}")
    bbox_mask = torch.zeros((latent_frames, 1, box_h, box_w), device=device)
    for index, (y1, y2, x1, x2) in enumerate(pixel_path):
        y1i, y2i = int(y1 * box_h), int(y2 * box_h)
        x1i, x2i = int(x1 * box_w), int(x2 * box_w)
        bbox_mask[index, :, y1i:y2i, x1i:x2i] = 1
    if not bool(torch.all(bbox_mask.flatten(1).sum(dim=1) > 0)):
        raise RuntimeError(f"At least one latent-frame {path_name} box is empty")
    return bbox_mask


def run(
    output_dir: Path,
    *,
    path_name: str = "right",
    num_steps: int = 8,
    admission_receipt: Path,
    notebook_sha256: str,
) -> None:
    admission_metadata = validate_admission_receipt(
        admission_receipt,
        intended_arm=f"feedback_path_{path_name}_steps_{num_steps}",
        expected_notebook_sha256=notebook_sha256,
        runner_path=Path(__file__).resolve(),
    )

    global np, torch
    import numpy as np
    import torch
    try:
        from .ditraj_feedback_adapter import FEEDBACK_STEP, RedPathFeedback, patch_pinned_pipeline
    except ImportError:
        from ditraj_feedback_adapter import FEEDBACK_STEP, RedPathFeedback, patch_pinned_pipeline

    if path_name not in baseline.PATHS_YXYX:
        raise ValueError(f"path_name must be one of {tuple(baseline.PATHS_YXYX)}")
    if not FEEDBACK_STEP < num_steps <= baseline.NUM_STEPS:
        raise ValueError(f"num_steps must be {FEEDBACK_STEP + 1}..{baseline.NUM_STEPS}")
    repo_root = Path.cwd()
    source_hashes = baseline.patch_upstream_source(repo_root)
    feedback_pipe_hash = patch_pinned_pipeline(repo_root / "module/pipe.py")
    source_hashes["module/pipe.py_feedback_patch"] = feedback_pipe_hash

    # Imports happen only after the notebook's setup and source patching.
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from module.attention_processor import MyWanAttnProcessor2_0
    from module.pipe import myWanPipeline
    from utils import plan_path

    runtime = baseline.runtime_metadata()
    started = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    print(json.dumps(runtime, indent=2), flush=True)

    vae = AutoencoderKLWan.from_pretrained(
        baseline.MODEL_ID,
        revision=baseline.MODEL_REVISION,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = myWanPipeline.from_pretrained(
        baseline.MODEL_ID,
        revision=baseline.MODEL_REVISION,
        vae=vae,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    processors = {name: MyWanAttnProcessor2_0() for name in pipe.transformer.attn_processors.keys()}
    pipe.transformer.set_attn_processor(processors)
    pipe.scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=baseline.FLOW_SHIFT,
    )
    pipe.model_cpu_offload_seq = "text_encoder->transformer->vae"
    pipe.enable_model_cpu_offload(gpu_id=0)

    target_mask = make_path_mask(path_name, plan_path=plan_path, device=device)
    initial_generator = torch.Generator(device=device).manual_seed(baseline.SEED)
    common_latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=int(pipe.transformer.config.in_channels),
        height=baseline.HEIGHT,
        width=baseline.WIDTH,
        num_frames=baseline.NUM_FRAMES,
        dtype=torch.float32,
        device=device,
        generator=initial_generator,
        latents=None,
    )
    noise_sha256 = baseline.sha256_tensor(common_latents)
    torch.save({"latents": common_latents.detach().cpu(), "seed": baseline.SEED}, output_dir / "common_initial_noise.pt")

    arms: dict[str, dict[str, Any]] = {}
    frame_dirs: dict[str, Path] = {}
    frame_files: dict[str, list[str]] = {}
    for arm_name in ARM_NAMES:
        pipe.scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=baseline.FLOW_SHIFT,
        )
        feedback = None
        if arm_name == "decode_matched_noop":
            feedback = RedPathFeedback(initial_bbox_mask=target_mask, gain=0.0)
        elif arm_name == "x0_hat_feedback":
            feedback = RedPathFeedback(initial_bbox_mask=target_mask, gain=0.5)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        generator = torch.Generator(device=device).manual_seed(baseline.SEED)
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                kwargs = {
                    "prompt": baseline.PROMPT,
                    "negative_prompt": baseline.NEGATIVE_PROMPT,
                    "height": baseline.HEIGHT,
                    "width": baseline.WIDTH,
                    "num_frames": baseline.NUM_FRAMES,
                    "num_inference_steps": num_steps,
                    "guidance_scale": baseline.GUIDANCE_SCALE,
                    "generator": generator,
                    "latents": common_latents.clone(),
                    "output_type": "np",
                    "attention_kwargs": {
                        "bbox_mask": target_mask.clone(),
                        "encoder_attention_mask": torch.tensor([False] * 512 + [True] * 512, device=device),
                        "bg_prompt": baseline.BG_PROMPT,
                        "fg_prompt": baseline.FG_PROMPT,
                        "fixRope_step": 3,
                        "mask_step": 18,
                    },
                }
                if feedback is not None:
                    kwargs["attention_kwargs"]["latentgate_feedback"] = feedback
                output = pipe(**kwargs)
        finally:
            if feedback is not None:
                feedback.close()
        elapsed = time.perf_counter() - start
        if arm_name != "ditraj_baseline" and feedback.report is None:
            raise RuntimeError(f"{arm_name} completed without a step-{FEEDBACK_STEP} observer report")
        frames = np.asarray(output.frames[0], dtype=np.float32)
        if frames.shape != (baseline.NUM_FRAMES, baseline.HEIGHT, baseline.WIDTH, 3):
            raise RuntimeError(f"Unexpected {arm_name} frame shape: {frames.shape}")
        frame_dir = output_dir / arm_name
        names = baseline.save_frames(frames, frame_dir)
        frames_sha256 = hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()
        frame_dirs[arm_name] = frame_dir
        frame_files[arm_name] = names
        metrics = baseline.evaluate_red_proxy(frame_dir, baseline.PATHS_YXYX[path_name])
        arms[arm_name] = {
            "path_condition": path_name,
            "same_initial_noise_sha256": noise_sha256,
            "decoded_frame_array_sha256": frames_sha256,
            "generation_seconds": elapsed,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "frame_paths": [f"{arm_name}/{name}" for name in names],
            "secondary_proxy_metrics": metrics,
            "feedback_report": feedback.report if feedback is not None else None,
        }
        del output, frames
        torch.cuda.empty_cache()
        print(
            f"Completed {arm_name}: {elapsed:.1f}s; "
            f"{metrics['valid_frame_coverage']:.0%} red-proxy coverage",
            flush=True,
        )

    noop_matches_baseline = (
        arms["ditraj_baseline"]["decoded_frame_array_sha256"]
        == arms["decode_matched_noop"]["decoded_frame_array_sha256"]
    )
    feedback_mask_changed = (
        int(
            arms["x0_hat_feedback"]["feedback_report"]["rasterization"]["changed_token_total"]
        )
        > 0
    )
    baseline.save_contact_sheet(frame_dirs, output_dir / "contact_sheet.png")
    baseline.save_annotation_template(output_dir, frame_files)
    report = {
        "experiment_id": "EXP-DITRAJ-WAN13B-X0-BOX-FEEDBACK-MECHANISM-SMOKE-001",
        "status": "mechanism_smoke_only_not_an_efficacy_test",
        "run_started_at_utc": started,
        "run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {"repo_id": baseline.MODEL_ID, "revision": baseline.MODEL_REVISION},
        "ditraj": {"commit": baseline.DITRAJ_COMMIT, "source_repo": "https://github.com/xduzhangjiayu/DiTraj"},
        "local_source_sha256": source_hashes,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "admission": admission_metadata,
        "account_quota": {
            "quota_hours_before": "not queried; no Kaggle quota API is available to this runner",
            "quota_hours_after": "not queried; no Kaggle quota API is available to this runner",
        },
        "annotation_packet": {
            "status": "preliminary_not_scoreable",
            "scoreable": False,
            "exportable": False,
            "missing_requirements": [
                "registered per-frame SHA-256 and dimension manifest",
                "separate background-stability measure",
            ],
        },
        "runtime": runtime,
        "generation": {
            "prompt": baseline.PROMPT,
            "negative_prompt": baseline.NEGATIVE_PROMPT,
            "direction_words_in_prompt": False,
            "path_condition": path_name,
            "seed": baseline.SEED,
            "width": baseline.WIDTH,
            "height": baseline.HEIGHT,
            "num_frames": baseline.NUM_FRAMES,
            "num_inference_steps": num_steps,
            "feedback_step_zero_based": FEEDBACK_STEP,
            "guidance_scale": baseline.GUIDANCE_SCALE,
            "scheduler": "UniPCMultistepScheduler(flow_prediction, flow_shift=3.0)",
            "target_paths_yxyx": baseline.PATHS_YXYX[path_name],
            "initial_noise_sha256_shared_across_arms": noise_sha256,
            "arms": list(ARM_NAMES),
        },
        "decision_boundary": [
            "This screen checks whether exact x0_hat capture, VAE decoding, observer output, bounded box update, and cached cross-attention-mask refresh work together.",
            "The decode-matched no-op control estimates effects from the extra VAE decode and observer schedule.",
            "A single path and seed with at most a few post-update steps cannot establish an efficacy claim.",
            "The threshold observer is red-object-specific and is not independent ground truth.",
            "Review local-only preliminary frames; do not export or score them until the registered frame manifest and separate background-stability measure exist.",
        ],
        "arms": arms,
        "control_checks": {
            "decode_matched_noop_exactly_matches_baseline_frames": noop_matches_baseline,
            "feedback_changed_any_rasterized_attention_tokens": feedback_mask_changed,
            "all_mechanism_gates_passed": noop_matches_baseline and feedback_mask_changed,
            "failures": [
                name
                for name, passed in (
                    ("decode_matched_noop_exactly_matches_baseline_frames", noop_matches_baseline),
                    ("feedback_changed_any_rasterized_attention_tokens", feedback_mask_changed),
                )
                if not passed
            ],
            "note": "A changed mask verifies actuation plumbing only; it does not prove that generated object motion improves.",
        },
    }
    if not report["control_checks"]["all_mechanism_gates_passed"]:
        report["status"] = "mechanism_smoke_gate_failed_not_an_efficacy_test"
    (output_dir / "feedback_smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Artifacts saved to {output_dir}", flush=True)
    if not report["control_checks"]["all_mechanism_gates_passed"]:
        failed = ", ".join(report["control_checks"]["failures"])
        raise RuntimeError(f"Feedback mechanism smoke gate failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/ditraj_feedback_smoke"))
    parser.add_argument("--path", choices=("right", "left"), default="right")
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--notebook-sha256", required=True)
    args = parser.parse_args()
    run(
        args.output_dir.resolve(),
        path_name=args.path,
        num_steps=args.num_steps,
        admission_receipt=args.admission_receipt.resolve(),
        notebook_sha256=args.notebook_sha256,
    )


if __name__ == "__main__":
    main()
