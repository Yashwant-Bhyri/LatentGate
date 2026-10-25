#!/usr/bin/env python3
"""Direction-balanced six-case 17-frame DiTraj pilot for Wan2.1-T2V-1.3B.

Run from the pinned DiTraj checkout after installing its pinned Diffusers
environment and replacing Diffusers' Wan transformer with DiTraj's copy:

    python run_ditraj_paired_path_pilot.py --num-steps 1 \
        --output-dir /kaggle/working/ditraj_paired_path_smoke \
        --admission-receipt /path/to/fresh-receipt.json \
        --notebook-sha256 <reviewed-local-notebook-sha256>

This is an adapted small pilot, not an exact reproduction of the paper. It
keeps DiTraj's official pipeline/attention code and source behavior, uses the
one direction-neutral prompt, mirrored paths, a 17-frame shape, a 30-step sampler,
and common initial latent noise across Base, SG-only, and full DiTraj arms. Research-only:
the upstream DiTraj repository states CC BY-NC 4.0.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import secrets
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # works as both `research.*` and a staged flat runner
    from .ditraj_admission import validate_admission_receipt
except ImportError:
    from ditraj_admission import validate_admission_receipt


MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
DITRAJ_COMMIT = "6fa8e613c9e7c40ec2db456ab2ba34367a076886"
WIDTH, HEIGHT, NUM_FRAMES, NUM_STEPS = 832, 480, 17, 30
SEED, GUIDANCE_SCALE, FLOW_SHIFT = 2026, 5.0, 3.0
PROMPT = (
    "A single glossy bright-red ball on a matte white tabletop, locked-off camera, "
    "soft even daylight. The ball moves smoothly and stays fully visible in frame."
)
FG_PROMPT = "A single glossy bright-red ball moves smoothly and stays fully visible in frame."
BG_PROMPT = "A locked-off camera, matte white tabletop, soft even daylight, simple clean background."
NEGATIVE_PROMPT = "text, watermark, blurry image, JPEG artifacts"
PATHS_YXYX = {
    "right": [[0, 0.50, 0.76, 0.43, 0.57], [16, 0.50, 0.76, 0.63, 0.77]],
    "left": [[0, 0.50, 0.76, 0.43, 0.57], [16, 0.50, 0.76, 0.23, 0.37]],
}
ARM_CONFIGS = {
    "base": {"mask_step": 0, "fixRope_step": 0},
    "sg_only": {"mask_step": 18, "fixRope_step": 0},
    "ditraj": {"mask_step": 18, "fixRope_step": 3},
}


def sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to("cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def normalized_target_boxes(n_frames: int, anchors: list[list[float]]) -> list[list[float]]:
    """Interpolate anchors and return normalized [x1,y1,x2,y2] boxes."""
    out: list[list[float]] = []
    for frame in range(n_frames):
        if frame <= anchors[0][0]:
            row = anchors[0]
            y1, y2, x1, x2 = row[1:]
        elif frame >= anchors[-1][0]:
            row = anchors[-1]
            y1, y2, x1, x2 = row[1:]
        else:
            right = next(i for i, row in enumerate(anchors[1:], start=1) if frame <= row[0])
            a, b = anchors[right - 1], anchors[right]
            alpha = (frame - a[0]) / (b[0] - a[0])
            y1, y2, x1, x2 = [a[i] + alpha * (b[i] - a[i]) for i in range(1, 5)]
        out.append([x1, y1, x2, y2])
    return out


def evaluate_red_proxy(frames_dir: Path, anchors: list[list[float]]) -> dict[str, Any]:
    """Secondary red-pixel screen; blinded manual labels remain the primary score."""
    paths = sorted(frames_dir.glob("frame_*.png"))
    if len(paths) != NUM_FRAMES:
        raise RuntimeError(f"Expected {NUM_FRAMES} saved frames, found {len(paths)} in {frames_dir}")
    target_boxes = normalized_target_boxes(len(paths), anchors)
    centers: list[tuple[float, float] | None] = []
    areas: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    valid_errors: list[float] = []
    penalized_errors: list[float] = []
    ious: list[float] = []
    width_px = height_px = 0

    for index, path in enumerate(paths):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        h, w = rgb.shape[:2]
        height_px, width_px = h, w
        mask = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 125) & (rgb[:, :, 2] < 125)
        ys, xs = np.nonzero(mask)
        if len(xs) < 300:
            centers.append(None)
            penalized_errors.append(1.0)
            continue

        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        cx, cy = float(xs.mean()), float(ys.mean())
        centers.append((cx, cy))
        areas.append(float(len(xs)))
        widths.append(x2 - x1)
        heights.append(y2 - y1)

        tx1n, ty1n, tx2n, ty2n = target_boxes[index]
        tx1, ty1, tx2, ty2 = tx1n * w, ty1n * h, tx2n * w, ty2n * h
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        error = math.hypot(cx - tcx, cy - tcy)
        diag = math.hypot(w, h)
        valid_errors.append(error / diag)
        penalized_errors.append(error / diag)
        ix1, iy1 = max(x1, tx1), max(y1, ty1)
        ix2, iy2 = min(x2, tx2), min(y2, ty2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - intersection
        ious.append(intersection / union if union else 0.0)

    adjacent = [(a, b) for a, b in zip(centers, centers[1:]) if a is not None and b is not None]
    target_dx = ((target_boxes[-1][0] + target_boxes[-1][2]) - (target_boxes[0][0] + target_boxes[0][2])) / 2
    expected_sign = 1 if target_dx > 0 else -1
    aligned = [expected_sign * (b[0] - a[0]) > 0 for a, b in adjacent]

    def cv(values: list[float]) -> float | None:
        return statistics.pstdev(values) / statistics.mean(values) if values and statistics.mean(values) else None

    return {
        "evaluator": "red_rgb_threshold_proxy_secondary_only",
        "threshold": {"r_gt": 150, "g_lt": 125, "b_lt": 125, "minimum_pixels": 300},
        "frame_count": len(paths),
        "frame_size_wh": [width_px, height_px],
        "valid_frame_coverage": len([center for center in centers if center is not None]) / len(paths),
        "mean_center_error_frame_diagonals_valid_only": statistics.mean(valid_errors) if valid_errors else None,
        "mean_center_error_frame_diagonals_missing_penalty_1": statistics.mean(penalized_errors),
        "mean_bbox_iou_valid_only": statistics.mean(ious) if ious else None,
        "red_mask_area_cv": cv(areas),
        "bbox_width_cv": cv(widths),
        "bbox_height_cv": cv(heights),
        "direction_aligned_adjacent_visible_step_fraction": sum(aligned) / len(aligned) if aligned else None,
        "adjacent_visible_step_count": len(aligned),
        "limitations": [
            "All red pixels are grouped; this does not track a semantic object instance.",
            "Image-plane scoring does not verify depth, 3D pose, contact, or physical rolling.",
            "Specular highlights can split or remove threshold pixels; blinded manual labels are primary.",
        ],
    }

def save_frames(frames: np.ndarray, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for index, frame in enumerate(frames):
        pixels = (np.clip(frame, 0.0, 1.0) * 255).round().astype(np.uint8)
        path = out_dir / f"frame_{index:03d}.png"
        Image.fromarray(pixels, mode="RGB").save(path, optimize=True)
        files.append(path.name)
    return files


def save_contact_sheet(frame_dirs: dict[str, Path], target: Path) -> None:
    arms = list(frame_dirs)
    frames = [[Image.open(frame_dirs[arm] / f"frame_{i:03d}.png").convert("RGB") for i in range(NUM_FRAMES)] for arm in arms]
    thumb_w, thumb_h = 208, 120
    gap, label_h = 6, 22
    columns = 5
    rows = math.ceil(NUM_FRAMES / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumb_w + (columns + 1) * gap, len(arms) * (rows * (thumb_h + label_h) + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for arm_i, arm in enumerate(arms):
        row_y = gap + arm_i * (rows * (thumb_h + label_h) + gap)
        for frame_i, image in enumerate(frames[arm_i]):
            col, row = frame_i % columns, frame_i // columns
            x = gap + col * (thumb_w + gap)
            y = row_y + row * (thumb_h + label_h)
            image.thumbnail((thumb_w, thumb_h))
            sheet.paste(image, (x, y + label_h))
            draw.text((x + 2, y + 3), f"{arm} · frame {frame_i:02d}", fill="black")
    sheet.save(target)


def save_annotation_template(output_dir: Path, frame_files: dict[str, list[str]]) -> None:
    package_dir = output_dir / "blinded_annotation_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    import csv

    blind_ids = [f"clip_{chr(ord('A') + index)}" for index in range(len(frame_files))]
    secrets.SystemRandom().shuffle(blind_ids)
    blind_map = dict(zip(frame_files, blind_ids, strict=True))
    for arm, blind_id in blind_map.items():
        dest = package_dir / blind_id
        dest.mkdir(exist_ok=True)
        for name in frame_files[arm]:
            shutil.copy2(output_dir / arm / name, dest / name)
    path = package_dir / "annotation_template.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "blind_clip_id", "frame_index", "image_file", "object_visible", "visible_reason_or_occlusion",
            "instance_count", "same_identity_as_frame_0", "center_x_px", "center_y_px",
            "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
            "annotation_confidence_1_to_5", "blur_or_shape_note",
        ])
        for run_key, blind_id in blind_map.items():
            for index, name in enumerate(frame_files[run_key]):
                writer.writerow([blind_id, index, f"{blind_id}/{name}", "", "", "", "", "", "", "", "", "", "", "", ""])
    key_path = output_dir.parent / "ditraj_private_unblind_key.json"
    key_path.write_text(json.dumps({blind: arm for arm, blind in blind_map.items()}, indent=2) + "\n")
    # Preliminary local-only directory: do not archive/export until both the
    # registered per-frame manifest and separate background measure exist.


def patch_upstream_source(repo_root: Path) -> dict[str, str]:
    """Apply behavior-preserving runtime fixes to the pinned source checkout.

    The CPU bound lookup returns the same inclusive corners for each filled
    rectangular mask but avoids thousands of Python-level CUDA scalar reads.
    The two zero-cutoff checks skip masks that cannot be consumed by that arm.
    The published inclusive-corner/exclusive-slice behavior is intentionally
    preserved for this first source-faithful comparison.
    """
    if not (repo_root / "module/pipe.py").is_file() or not (repo_root / "module/transformer_wan.py").is_file():
        raise FileNotFoundError(f"Run from the pinned DiTraj checkout; expected module/ under {repo_root}")
    head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    if head != DITRAJ_COMMIT:
        raise RuntimeError(f"Expected DiTraj commit {DITRAJ_COMMIT}, found {head}")
    diffusers_version = importlib.metadata.version("diffusers")
    if diffusers_version != "0.33.1":
        raise RuntimeError(f"Expected pinned Diffusers 0.33.1, found {diffusers_version}")

    finder_pattern = re.compile(
        r"(?m)^(?P<indent> *)def find_corners\(matrix\):\n.*?^(?P=indent)    return top_left, bottom_right\n",
        re.DOTALL,
    )

    def vectorized_finder(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}def find_corners(matrix):\n"
            f"{indent}    coords = torch.nonzero(matrix.detach().to(device='cpu') == 1, as_tuple=False)\n"
            f"{indent}    if coords.numel() == 0:\n"
            f"{indent}        return None, None\n"
            f"{indent}    lo = coords.min(dim=0).values\n"
            f"{indent}    hi = coords.max(dim=0).values\n"
            f"{indent}    return (int(lo[0]), int(lo[1])), (int(hi[0]), int(hi[1]))\n"
        )

    patched_hashes: dict[str, str] = {}
    for relative in ("module/pipe.py", "module/transformer_wan.py"):
        path = repo_root / relative
        source = path.read_text()
        if "coords = torch.nonzero(matrix.detach().to(device='cpu') == 1" not in source:
            source, count = finder_pattern.subn(vectorized_finder, source)
            if count != 1:
                raise RuntimeError(f"Expected one find_corners block in {relative}, replaced {count}")
        if relative == "module/pipe.py":
            source = source.replace(
                "cross_attention_mask = self.create_cross_attention_mask(bbox_mask, attention_kwargs['encoder_attention_mask'].flatten())",
                "cross_attention_mask = (self.create_cross_attention_mask(bbox_mask, attention_kwargs['encoder_attention_mask'].flatten()) if mask_step > 0 else None)",
                1,
            )
            if "if mask_step > 0 else None" not in source:
                raise RuntimeError("Could not apply the zero-mask-step cross-attention optimization")
            source = source.replace(
                "fixrope_self_attention_mask = self.create_fixrope_self_attention_mask(latents.shape[-3:], bbox_mask)",
                "fixrope_self_attention_mask = (self.create_fixrope_self_attention_mask(latents.shape[-3:], bbox_mask) if fixRope_step > 0 else None)",
                1,
            )
            if "if fixRope_step > 0 else None" not in source:
                raise RuntimeError("Could not apply the zero-RoPE-step mask optimization")
        path.write_text(source)
        patched_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    # The published setup replaces Diffusers' Wan transformer with DiTraj's
    # implementation. Copy the CPU-bound lookup optimization into that active
    # module as well; otherwise the pipeline file would be fast but the actual
    # loaded model would retain the per-pixel CUDA synchronization loop.
    transformer_target = Path(
        importlib.metadata.distribution("diffusers").locate_file(
            "diffusers/models/transformers/transformer_wan.py"
        )
    )
    shutil.copy2(repo_root / "module/transformer_wan.py", transformer_target)
    patched_hashes["diffusers/models/transformers/transformer_wan.py"] = hashlib.sha256(
        transformer_target.read_bytes()
    ).hexdigest()
    return patched_hashes


def runtime_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; refusing to download model weights on CPU.")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    if "T4" not in name:
        raise RuntimeError(f"This ticket is scoped to a T4; observed device is {name!r}.")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "diffusers": package_version("diffusers"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "cuda_available": True,
        "gpu": name,
        "gpu_capability": list(capability),
        "gpu_total_memory_bytes": total_memory,
        "active_gpu_index": 0,
        "torch_dtype": "float16",
    }


def run(
    output_dir: Path,
    num_steps: int = NUM_STEPS,
    *,
    admission_receipt: Path,
    notebook_sha256: str,
) -> None:
    admission_metadata = validate_admission_receipt(
        admission_receipt,
        intended_arm=f"paired_path_steps_{num_steps}",
        expected_notebook_sha256=notebook_sha256,
        runner_path=Path(__file__).resolve(),
    )

    # No model/CUDA-facing imports occur before the fresh receipt is checked.
    global np, torch, Image, ImageDraw
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    repo_root = Path.cwd()
    source_hashes = patch_upstream_source(repo_root)

    # Delayed imports allow the notebook to install and patch Diffusers first.
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from module.attention_processor import MyWanAttnProcessor2_0
    from module.pipe import myWanPipeline
    from utils import plan_path

    runtime = runtime_metadata()
    run_started_at = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    print(json.dumps(runtime, indent=2), flush=True)

    # Load the released DiTraj pipeline and use T4-safe FP16 transformer weights.
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = myWanPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
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
        flow_shift=FLOW_SHIFT,
    )
    pipe.model_cpu_offload_seq = "text_encoder->transformer->vae"
    pipe.enable_model_cpu_offload(gpu_id=0)

    # Wan's temporal compression maps 17 frames to five path masks. Build one
    # mask per mirrored target below while holding prompt and initial noise fixed.
    latent_frames = NUM_FRAMES // 4 + 1
    box_h, box_w = HEIGHT // 16, WIDTH // 16

    # One initial noise tensor is explicitly shared across all three arms.
    init_generator = torch.Generator(device=device).manual_seed(SEED)
    common_latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=int(pipe.transformer.config.in_channels),
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        dtype=torch.float32,
        device=device,
        generator=init_generator,
        latents=None,
    )
    common_noise_sha = sha256_tensor(common_latents)
    torch.save({"latents": common_latents.detach().cpu(), "seed": SEED}, output_dir / "common_initial_noise.pt")

    arms: dict[str, dict[str, Any]] = {}
    frame_dirs: dict[str, Path] = {}
    frame_files: dict[str, list[str]] = {}
    for path_name, anchors in PATHS_YXYX.items():
        bbox_mask = torch.zeros((latent_frames, 1, box_h, box_w), device=device)
        pixel_path = plan_path(anchors, video_length=NUM_FRAMES)[::4]
        if len(pixel_path) != latent_frames:
            raise RuntimeError(f"Expected {latent_frames} latent boxes for {path_name}, got {len(pixel_path)}")
        for index, (y1, y2, x1, x2) in enumerate(pixel_path):
            y1i, y2i = int(y1 * box_h), int(y2 * box_h)
            x1i, x2i = int(x1 * box_w), int(x2 * box_w)
            bbox_mask[index, :, y1i:y2i, x1i:x2i] = 1
        if not bool(torch.all(bbox_mask.flatten(1).sum(dim=1) > 0)):
            raise RuntimeError(f"At least one latent-frame {path_name} box is empty")

        for arm_name, control in ARM_CONFIGS.items():
            run_key = f"{path_name}_{arm_name}"
            # Recreate scheduler state; every arm and direction receives the
            # same prompt, model revision, seed, and explicitly shared noise.
            pipe.scheduler = UniPCMultistepScheduler(
                prediction_type="flow_prediction",
                use_flow_sigmas=True,
                num_train_timesteps=1000,
                flow_shift=FLOW_SHIFT,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            arm_generator = torch.Generator(device=device).manual_seed(SEED)
            start = time.perf_counter()
            with torch.inference_mode():
                output = pipe(
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=HEIGHT,
                    width=WIDTH,
                    num_frames=NUM_FRAMES,
                    num_inference_steps=num_steps,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=arm_generator,
                    latents=common_latents.clone(),
                    output_type="np",
                    attention_kwargs={
                        "bbox_mask": bbox_mask,
                        "encoder_attention_mask": torch.tensor(
                            [False] * 512 + [True] * 512, device=device
                        ),
                        "bg_prompt": BG_PROMPT,
                        "fg_prompt": FG_PROMPT,
                        "fixRope_step": control["fixRope_step"],
                        "mask_step": control["mask_step"],
                    },
                )
            elapsed = time.perf_counter() - start
            frames = np.asarray(output.frames[0], dtype=np.float32)
            if frames.shape != (NUM_FRAMES, HEIGHT, WIDTH, 3):
                raise RuntimeError(f"Unexpected {run_key} output frame shape: {frames.shape}")
            arm_dir = output_dir / run_key
            names = save_frames(frames, arm_dir)
            frame_dirs[run_key] = arm_dir
            frame_files[run_key] = names
            metrics = evaluate_red_proxy(arm_dir, anchors)
            arms[run_key] = {
                "path_condition": path_name,
                "settings": control,
                "initial_noise_sha256": common_noise_sha,
                "generation_seconds": elapsed,
                "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "frame_paths": [f"{run_key}/{name}" for name in names],
                "secondary_proxy_metrics": metrics,
            }
            del output, frames
            torch.cuda.empty_cache()
            print(f"Completed {run_key}: {elapsed:.1f}s, {metrics['valid_frame_coverage']:.0%} proxy coverage", flush=True)

    save_contact_sheet(frame_dirs, output_dir / "contact_sheet.png")
    save_annotation_template(output_dir, frame_files)
    report = {
        "experiment_id": "EXP-DITRAJ-WAN13B-PAIRED-PATH-PILOT-002",
        "status": "completed_single_seed_screen" if num_steps == NUM_STEPS else "runtime_smoke_only",
        "run_started_at_utc": run_started_at,
        "run_completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "adaptation_note": "17-frame/30-step DiTraj adaptation; not an exact reproduction of the 81-frame paper setup.",
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "ditraj": {"commit": DITRAJ_COMMIT, "source_repo": "https://github.com/xduzhangjiayu/DiTraj"},
        "local_source_adaptations": {
            "patched_file_sha256": source_hashes,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "changes": [
                "Vectorized the filled-box corner lookup on CPU while preserving upstream inclusive bounds and exclusive slicing.",
                "Skipped cross-attention and repeated-token masks only when their corresponding cutoff is zero.",
            ],
        },
        "runtime": runtime,
        "account_quota": {
            **admission_metadata,
            "quota_hours_before": "not queried; this runner has no Kaggle quota API",
            "quota_hours_after": "not queried; this runner has no Kaggle quota API",
        },
        "generation": {
            "prompt": PROMPT,
            "foreground_prompt": FG_PROMPT,
            "background_prompt": BG_PROMPT,
            "direction_words_in_prompt": False,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": SEED,
            "width": WIDTH,
            "height": HEIGHT,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": num_steps,
            "guidance_scale": GUIDANCE_SCALE,
            "scheduler": "UniPCMultistepScheduler(flow_prediction, flow_shift=3.0)",
            "same_initial_noise_sha256_across_arms": common_noise_sha,
            "target_paths_yxyx": PATHS_YXYX,
        },
        "arms": arms,
        "primary_evaluation": {
            "method": "blinded_manual per-frame visibility, identity/count, center, bbox, confidence, and shape labels",
            "primary_metrics": ["center_ADE_normalized_by_frame_diagonal", "center_FDE_normalized_by_frame_diagonal"],
            "missing_detection_center_error_penalty": 1.0,
            "template_directory": "blinded_annotation_package/",
            "status": "preliminary_not_scoreable",
            "scoreable": False,
            "exportable": False,
            "missing_requirements": [
                "registered per-frame SHA-256 and dimension manifest",
                "separate background-stability measure",
            ],
            "unblinding_key": "../ditraj_private_unblind_key.json (keep separate until labels are complete)",
            "rule": "Do not export or score this preliminary directory; a later registered packet must hide path direction and arm names and treat missing detections as failures.",
        },
        "decision_gate": {
            "runtime": "all arms produce 17 frames with identical initial noise and recorded runtime/memory",
            "pilot_signal": "full DiTraj lowers blinded center ADE versus both controls when averaged over mirrored paths, with >=90% visibility and the correct signed endpoint response in both directions",
            "guardrails": "no identity loss/freeze/disappearance; no material apparent-size, background, or human-rated quality regression",
            "caveat": "single seed is a screening result only; repeat >=5 paired seeds before claiming improvement",
        },
        "licenses": {
            "ditraj": "CC BY-NC 4.0 per upstream README; research-only pending any separate commercial-use review",
            "wan": "See the pinned model card and its terms",
        },
    }
    (output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Artifacts saved to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/ditraj_paired_path_pilot"))
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--notebook-sha256", required=True)
    args = parser.parse_args()
    if not 1 <= args.num_steps <= NUM_STEPS:
        parser.error(f"--num-steps must be between 1 and {NUM_STEPS}")
    run(
        args.output_dir.resolve(),
        num_steps=args.num_steps,
        admission_receipt=args.admission_receipt.resolve(),
        notebook_sha256=args.notebook_sha256,
    )


if __name__ == "__main__":
    main()
