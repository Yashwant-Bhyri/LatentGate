#!/usr/bin/env python3
"""Run the first instrumented CogVideoX-5B-I2V LatentGate smoke sample.

This is an M1 feasibility runner, not a steering method. It refuses to load
weights unless a CUDA Tesla T4 is attached. Confirm the free billing option
in Colab before connecting; the hardware API cannot identify billing class.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageOps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if cuda_available else None
    free_bytes, total_bytes = torch.cuda.mem_get_info(0) if cuda_available else (None, None)
    capability = torch.cuda.get_device_capability(0) if cuda_available else None
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": package_version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "diffusers": package_version("diffusers"),
        "transformers": package_version("transformers"),
        "accelerate": package_version("accelerate"),
        "huggingface_hub": package_version("huggingface-hub"),
        "safetensors": package_version("safetensors"),
        "sentencepiece": package_version("sentencepiece"),
        "pyyaml": package_version("PyYAML"),
        "numpy": np.__version__,
        "pillow": package_version("Pillow"),
        "imageio": package_version("imageio"),
        "imageio_ffmpeg": package_version("imageio-ffmpeg"),
        "cuda_available": cuda_available,
        "device_name": properties.name if properties else None,
        "compute_capability": list(capability) if capability else None,
        "total_memory_bytes": total_bytes,
        "free_memory_bytes": free_bytes,
        "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(0) if cuda_available else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def expected_block_indices(num_blocks: int, capture_cfg: dict[str, Any]) -> list[int]:
    configured = capture_cfg.get("transformer_block_indices", "discover_evenly_spaced")
    requested_count = int(capture_cfg.get("transformer_block_count", 6))
    if requested_count < 1:
        raise ValueError(f"Transformer capture block count must be positive, got {requested_count}")
    if num_blocks < requested_count:
        raise RuntimeError(
            "Loaded CogVideoX transformer cannot satisfy the configured capture coverage: "
            f"requested {requested_count} block summaries, but only {num_blocks} blocks are exposed."
        )
    if configured == "discover_evenly_spaced":
        if requested_count == 1:
            indices = [0]
        else:
            indices = sorted({
                round(i * (num_blocks - 1) / (requested_count - 1))
                for i in range(requested_count)
            })
    elif isinstance(configured, (list, tuple)):
        indices = sorted({int(index) for index in configured})
    else:
        raise ValueError(
            "transformer_block_indices must be 'discover_evenly_spaced' or an explicit list "
            f"of indices; got {configured!r}"
        )
    if len(indices) != requested_count:
        raise ValueError(
            "Transformer capture selection must contain exactly the configured number of "
            f"unique blocks: expected {requested_count}, got {len(indices)}"
        )
    invalid = [index for index in indices if index < 0 or index >= num_blocks]
    if invalid:
        raise ValueError(f"Transformer block indices out of range: {invalid}; count={num_blocks}")
    return indices


class StepCapture:
    """Stream bounded latent and transformer summaries to CPU during generation."""

    def __init__(self, pipe: Any, capture_cfg: dict[str, Any]):
        self.pipe = pipe
        self.capture_cfg = capture_cfg
        self.steps = {int(step) for step in capture_cfg.get("latent_steps_zero_based", [])}
        self.block_modules = getattr(pipe.transformer, "transformer_blocks", None)
        if self.block_modules is None:
            raise RuntimeError("CogVideoX transformer exposes no transformer_blocks ModuleList")
        self.block_indices = expected_block_indices(len(self.block_modules), capture_cfg)
        spatial_indices_cfg = {
            "transformer_block_indices": capture_cfg.get(
                "spatial_map_block_indices", "discover_evenly_spaced"
            ),
            "transformer_block_count": capture_cfg.get("spatial_map_block_count", 3),
        }
        self.spatial_block_indices = expected_block_indices(
            len(self.block_modules), spatial_indices_cfg
        )
        self.spatial_steps = {
            int(step) for step in capture_cfg.get("spatial_map_steps_zero_based", [])
        }
        spatial_size = capture_cfg.get("spatial_map_size", [8, 12])
        if not isinstance(spatial_size, (list, tuple)) or len(spatial_size) != 2:
            raise ValueError("spatial_map_size must contain [height, width]")
        self.spatial_map_size = tuple(int(value) for value in spatial_size)
        if min(self.spatial_map_size) <= 0:
            raise ValueError(f"spatial_map_size must be positive, got {self.spatial_map_size}")
        self.predicted_clean_enabled = bool(capture_cfg.get("predicted_clean_capture", False))
        self.predicted_clean_steps = {
            int(step) for step in capture_cfg.get("predicted_clean_steps_zero_based", [])
        }
        self.scheduler = pipe.scheduler
        self.original_scheduler_step: Any | None = None
        self.current_step = 0
        self.current_latent_frames: int | None = None
        self.current_spatial_grid: tuple[int, int] | None = None
        self.transformer_forward_calls = 0
        self.block_call_counts: dict[str, int] = {}
        self.spatial_map_call_counts: dict[str, int] = {}
        self.predicted_clean_call_counts: dict[str, int] = {}
        self.post_step_latents: dict[int, dict[str, Any]] = {}
        self.predicted_clean_latents: dict[int, dict[str, Any]] = {}
        self.latent_callback_counts: dict[str, int] = {}
        self.hidden_summaries: dict[str, dict[str, Any]] = {}
        self.spatial_summaries: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def scheduler_step_index(self, timestep: Any) -> int:
        scheduler_steps = getattr(self.scheduler, "timesteps", None)
        if not isinstance(scheduler_steps, torch.Tensor) or scheduler_steps.numel() == 0:
            raise RuntimeError("Scheduler timesteps are unavailable during capture")
        scalar_t = float(torch.as_tensor(timestep).reshape(-1)[0].detach().cpu().item())
        values = scheduler_steps.detach().cpu().reshape(-1).tolist()
        distances = [abs(float(value) - scalar_t) for value in values]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        if distances[nearest] >= 1e-4:
            raise RuntimeError(f"Could not map scheduler timestep {scalar_t} to an inference step")
        return nearest

    def attach_scheduler_capture(self) -> None:
        if not self.predicted_clean_enabled:
            return
        scheduler_config = getattr(self.scheduler, "config", {})
        scheduler_class = type(self.scheduler).__name__
        prediction_type = (
            scheduler_config.get("prediction_type")
            if hasattr(scheduler_config, "get") else getattr(scheduler_config, "prediction_type", None)
        )
        expected_class = str(self.capture_cfg.get(
            "predicted_clean_scheduler_class", "CogVideoXDDIMScheduler"
        ))
        expected_prediction_type = str(self.capture_cfg.get(
            "predicted_clean_prediction_type", "v_prediction"
        ))
        if scheduler_class != expected_class or prediction_type != expected_prediction_type:
            raise RuntimeError(
                "Predicted-clean capture is validated only for the pinned scheduler route; "
                f"expected {expected_class}/{expected_prediction_type}, "
                f"got {scheduler_class}/{prediction_type}"
            )
        self.original_scheduler_step = self.scheduler.step
        original_step = self.original_scheduler_step

        @wraps(original_step)
        def capture_step_result(*args: Any, **kwargs: Any) -> Any:
            result = original_step(*args, **kwargs)
            timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
            if timestep is None:
                raise RuntimeError("Scheduler step did not expose its timestep")
            step = self.scheduler_step_index(timestep)
            if step not in self.predicted_clean_steps:
                return result
            key = f"step_{step:03d}"
            self.predicted_clean_call_counts[key] = self.predicted_clean_call_counts.get(key, 0) + 1
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                predicted_clean = result[1]
            else:
                predicted_clean = getattr(result, "pred_original_sample", None)
            if not isinstance(predicted_clean, torch.Tensor):
                raise RuntimeError(
                    "Pinned CogVideoXDDIMScheduler did not return predicted x_0 as its second result"
                )
            if step not in self.predicted_clean_latents:
                self.predicted_clean_latents[step] = {
                    "tensor": predicted_clean.detach().to(device="cpu").contiguous(),
                    "step_zero_based": step,
                    "scheduler_timestep_applied": float(
                        torch.as_tensor(timestep).reshape(-1)[0].detach().cpu().item()
                    ),
                    "capture_phase": "scheduler_predicted_clean_x0_from_current_model_evaluation",
                    "meaning": (
                        "predicted original sample x_0 returned by the pinned CogVideoXDDIMScheduler "
                        "from the current model evaluation; distinct from the post-step sample; "
                        "native scheduler output dtype is preserved"
                    ),
                    "dtype": str(predicted_clean.dtype),
                    "shape": list(predicted_clean.shape),
                }
            return result

        self.scheduler.step = capture_step_result

    def attach(self) -> None:
        self.attach_scheduler_capture()

        def before_transformer(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            self.transformer_forward_calls += 1
            raw_hidden = kwargs.get("hidden_states", args[0] if args else None)
            if not isinstance(raw_hidden, torch.Tensor) or raw_hidden.ndim != 5:
                shape = list(raw_hidden.shape) if isinstance(raw_hidden, torch.Tensor) else None
                raise RuntimeError(
                    "Expected CogVideoX transformer input latents with shape "
                    f"[batch, latent_frames, channels, height, width], got {shape}"
                )
            self.current_latent_frames = int(raw_hidden.shape[1])
            if self.spatial_steps:
                transformer_config = self.pipe.transformer.config
                patch_size = int(getattr(transformer_config, "patch_size"))
                patch_size_t = getattr(transformer_config, "patch_size_t", None)
                if patch_size_t is not None:
                    raise RuntimeError(
                        "Spatial map capture requires patch_size_t=None for per-frame 2D token grids"
                    )
                height, width = int(raw_hidden.shape[-2]), int(raw_hidden.shape[-1])
                if height % patch_size or width % patch_size:
                    raise RuntimeError(
                        f"Latent frame size {(height, width)} is not divisible by patch size {patch_size}"
                    )
                self.current_spatial_grid = (height // patch_size, width // patch_size)
            timestep = kwargs.get("timestep", args[2] if len(args) > 2 else None)
            self.current_step = (
                self.transformer_forward_calls - 1
                if timestep is None else self.scheduler_step_index(timestep)
            )

        self.handles.append(
            self.pipe.transformer.register_forward_pre_hook(before_transformer, with_kwargs=True)
        )
        hooked_block_indices = sorted(set(self.block_indices) | set(self.spatial_block_indices))
        for block_id in hooked_block_indices:
            def after_block(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                block_id: int = block_id,
            ) -> None:
                step = int(self.current_step)
                capture_hidden = step in self.steps and block_id in self.block_indices
                capture_spatial = step in self.spatial_steps and block_id in self.spatial_block_indices
                if not capture_hidden and not capture_spatial:
                    return
                key = f"step_{step:03d}_block_{block_id:02d}"
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                    if capture_hidden:
                        self.block_call_counts[f"{key}_bad_output"] = 1
                    if capture_spatial:
                        self.spatial_map_call_counts[f"{key}_bad_output"] = 1
                    return
                latent_frames = self.current_latent_frames
                if not latent_frames or hidden.shape[1] % latent_frames != 0:
                    if capture_hidden:
                        self.block_call_counts[f"{key}_bad_shape"] = 1
                    if capture_spatial:
                        self.spatial_map_call_counts[f"{key}_bad_shape"] = 1
                    return
                batch, token_count, hidden_width = hidden.shape
                tokens_per_frame = token_count // latent_frames
                current_timestep = getattr(self.pipe, "_current_timestep", None)
                timestep_value = (
                    float(torch.as_tensor(current_timestep).reshape(-1)[0].detach().cpu().item())
                    if current_timestep is not None else None
                )
                if capture_hidden:
                    self.block_call_counts[key] = self.block_call_counts.get(key, 0) + 1
                    if key not in self.hidden_summaries:
                        pooled = hidden.reshape(
                            batch, latent_frames, tokens_per_frame, hidden_width
                        ).mean(dim=2)
                        self.hidden_summaries[key] = {
                            "summary": pooled.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                            "step_zero_based": step,
                            "timestep": timestep_value,
                            "capture_phase": "transformer_block_output_before_scheduler_transition",
                            "block_index": block_id,
                            "pooling": "mean over spatial tokens separately for each latent video frame",
                            "tensor_shape": list(pooled.shape),
                            "cfg_batch_size": int(batch),
                            "cfg_branch_semantics": "batch preserved without aggregation; branch order not interpreted",
                        }
                if capture_spatial:
                    self.spatial_map_call_counts[key] = self.spatial_map_call_counts.get(key, 0) + 1
                    grid = self.current_spatial_grid
                    if grid is None or token_count != latent_frames * grid[0] * grid[1]:
                        raise RuntimeError(
                            "CogVideoX image-token sequence does not match latent frames × patch grid: "
                            f"tokens={token_count}, frames={latent_frames}, grid={grid}"
                        )
                    if key not in self.spatial_summaries:
                        import torch.nn.functional as F

                        spatial = hidden.reshape(
                            batch, latent_frames, grid[0], grid[1], hidden_width
                        ).permute(0, 1, 4, 2, 3)
                        reduced = torch.stack(
                            [
                                F.adaptive_avg_pool2d(spatial[:, frame], self.spatial_map_size)
                                for frame in range(latent_frames)
                            ],
                            dim=1,
                        ).permute(0, 1, 3, 4, 2)
                        self.spatial_summaries[key] = {
                            "summary": reduced.detach().to(device="cpu", dtype=torch.float16).contiguous(),
                            "step_zero_based": step,
                            "scheduler_timestep": timestep_value,
                            "capture_phase": "transformer_block_output_before_scheduler_transition",
                            "block_index": block_id,
                            "source_spatial_grid": list(grid),
                            "pooled_spatial_grid": list(self.spatial_map_size),
                            "patch_size": int(getattr(self.pipe.transformer.config, "patch_size")),
                            "pooling": "adaptive average pooling over image-token rows and columns per frame",
                            "tensor_shape": list(reduced.shape),
                            "cfg_batch_size": int(batch),
                            "cfg_branch_semantics": "batch preserved without aggregation; branch order not interpreted",
                        }

            self.handles.append(self.block_modules[block_id].register_forward_hook(after_block))

    def callback(self, _pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        step = int(step)
        if step in self.steps:
            key = str(step)
            self.latent_callback_counts[key] = self.latent_callback_counts.get(key, 0) + 1
            latents = callback_kwargs.get("latents")
            if not isinstance(latents, torch.Tensor):
                raise RuntimeError("CogVideoX callback did not expose latents")
            self.post_step_latents[step] = {
                "tensor": latents.detach().to(device="cpu").contiguous(),
                "step_zero_based": step,
                "scheduler_timestep_applied": float(
                    torch.as_tensor(timestep).reshape(-1)[0].detach().cpu().item()
                ),
                "capture_phase": "scheduler_step_result_after_current_model_evaluation",
                "meaning": (
                    "sample returned by scheduler.step for scheduler_timestep_applied after "
                    "the current model evaluation; at nonfinal steps this is the next latent "
                    "state, while after the final transition it is the terminal latent state "
                    "passed to output decoding; it is not the transformer input at the same "
                    "step index"
                ),
                "dtype": str(latents.dtype),
                "shape": list(latents.shape),
            }
        return callback_kwargs

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.original_scheduler_step is not None:
            self.scheduler.step = self.original_scheduler_step
            self.original_scheduler_step = None


def as_pil_frames(output: Any) -> list[Image.Image]:
    frames = output.frames[0] if hasattr(output, "frames") else output[0]
    converted: list[Image.Image] = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            converted.append(frame.convert("RGB"))
            continue
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            if array.min() < 0:
                array = (array + 1.0) / 2.0
            array = np.clip(array * (255 if array.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        converted.append(Image.fromarray(array).convert("RGB"))
    if not converted:
        raise RuntimeError("The pipeline returned no output frames")
    return converted


def write_contact_sheet(frames: list[Image.Image], path: Path, columns: int = 7) -> None:
    thumb_w, thumb_h = 180, 120
    rows = (len(frames) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 22)), "white")
    drawer = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        thumb = ImageOps.fit(frame, (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w
        y = (index // columns) * (thumb_h + 22)
        canvas.paste(thumb, (x, y))
        drawer.text((x + 4, y + thumb_h + 3), f"frame {index:03d}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_frames(frames: list[Image.Image], run_dir: Path) -> list[str]:
    frame_dir = run_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = frame_dir / f"frame_{index:03d}.png"
        frame.save(path)
        paths.append(path.relative_to(run_dir).as_posix())
    return paths


def run_once(pipe: Any, image: Image.Image, prompt: str, config: dict[str, Any], run_dir: Path, run_name: str) -> dict[str, Any]:
    from diffusers.utils import export_to_video

    generation = config["generation"]
    capture = StepCapture(pipe, config["capture"])
    started = time.perf_counter()
    try:
        capture.attach()
        generator = torch.Generator(device="cuda").manual_seed(int(generation["seed"]))
        torch.cuda.reset_peak_memory_stats(0)
        output = pipe(
            image=image,
            prompt=prompt,
            height=int(generation["height"]),
            width=int(generation["width"]),
            num_frames=int(generation["num_frames"]),
            num_inference_steps=int(generation["num_inference_steps"]),
            guidance_scale=float(generation["guidance_scale"]),
            use_dynamic_cfg=bool(generation.get("use_dynamic_cfg", True)),
            generator=generator,
            callback_on_step_end=capture.callback,
            callback_on_step_end_tensor_inputs=["latents"],
            output_type="pil",
        )
        torch.cuda.synchronize()
    finally:
        capture.close()
    elapsed = time.perf_counter() - started
    frames = as_pil_frames(output)
    run_dir.mkdir(parents=True, exist_ok=False)
    frame_paths = save_frames(frames, run_dir)
    write_contact_sheet(frames, run_dir / "frames_contact_sheet.png")
    video_path = run_dir / "output.mp4"
    export_to_video(frames, str(video_path), fps=int(generation.get("fps", 8)))
    torch.save(
        {f"post_step_{step:03d}": item for step, item in sorted(capture.post_step_latents.items())},
        run_dir / "post_step_latents.pt",
    )
    torch.save(
        {
            f"predicted_clean_{step:03d}": item
            for step, item in sorted(capture.predicted_clean_latents.items())
        },
        run_dir / "predicted_clean_latents.pt",
    )
    torch.save(capture.hidden_summaries, run_dir / "transformer_hidden_summaries.pt")
    torch.save(capture.spatial_summaries, run_dir / "transformer_spatial_summaries.pt")
    expected_steps = sorted({int(step) for step in config["capture"].get("latent_steps_zero_based", [])})
    expected_hidden = len(expected_steps) * len(capture.block_indices)
    expected_hidden_keys = {
        f"step_{step:03d}_block_{block_id:02d}"
        for step in expected_steps
        for block_id in capture.block_indices
    }
    expected_spatial_steps = sorted(capture.spatial_steps)
    expected_spatial_keys = {
        f"step_{step:03d}_block_{block_id:02d}"
        for step in expected_spatial_steps
        for block_id in capture.spatial_block_indices
    }
    expected_predicted_clean_steps = sorted(capture.predicted_clean_steps)
    expected_predicted_clean_keys = {
        f"step_{step:03d}" for step in expected_predicted_clean_steps
    }
    return {
        "run_name": run_name,
        "status": "generation_complete",
        "capture_phase_semantics": {
            "post_step_latents": "scheduler_step_result_after_current_model_evaluation",
            "predicted_clean_latents": "scheduler_predicted_clean_x0_from_current_model_evaluation",
            "hidden_summaries": "transformer_block_output_before_scheduler_transition",
            "spatial_summaries": "transformer_block_output_before_scheduler_transition",
        },
        "elapsed_seconds": elapsed,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(0),
        "transformer_forward_calls": capture.transformer_forward_calls,
        "expected_inference_steps": int(generation["num_inference_steps"]),
        "post_step_latent_steps_expected": expected_steps,
        "post_step_latent_steps_captured": sorted(capture.post_step_latents),
        "post_step_latent_callback_counts": capture.latent_callback_counts,
        "post_step_latents_captured_exactly_once": (
            set(capture.latent_callback_counts) == {str(step) for step in expected_steps}
            and all(count == 1 for count in capture.latent_callback_counts.values())
        ),
        "hidden_summary_blocks": capture.block_indices,
        "hidden_summary_expected": expected_hidden,
        "hidden_summary_captured": len(capture.hidden_summaries),
        "hidden_summary_expected_keys": sorted(expected_hidden_keys),
        "hidden_summary_captured_keys": sorted(capture.hidden_summaries),
        "hidden_summary_call_counts": capture.block_call_counts,
        "hidden_summaries_captured_exactly_once": (
            set(capture.block_call_counts) == expected_hidden_keys
            and all(count == 1 for count in capture.block_call_counts.values())
        ),
        "spatial_map_blocks": capture.spatial_block_indices,
        "spatial_map_size": list(capture.spatial_map_size),
        "spatial_map_expected_keys": sorted(expected_spatial_keys),
        "spatial_map_captured_keys": sorted(capture.spatial_summaries),
        "spatial_map_call_counts": capture.spatial_map_call_counts,
        "spatial_maps_captured_exactly_once": (
            set(capture.spatial_map_call_counts) == expected_spatial_keys
            and all(count == 1 for count in capture.spatial_map_call_counts.values())
        ),
        "predicted_clean_steps_expected": expected_predicted_clean_steps,
        "predicted_clean_steps_captured": sorted(capture.predicted_clean_latents),
        "predicted_clean_expected_keys": sorted(expected_predicted_clean_keys),
        "predicted_clean_captured_keys": sorted(
            f"step_{step:03d}" for step in capture.predicted_clean_latents
        ),
        "predicted_clean_call_counts": capture.predicted_clean_call_counts,
        "predicted_clean_captured_exactly_once": (
            set(capture.predicted_clean_call_counts) == expected_predicted_clean_keys
            and all(count == 1 for count in capture.predicted_clean_call_counts.values())
        ),
        "transformer_forward_calls_exact": capture.transformer_forward_calls == int(generation["num_inference_steps"]),
        "frame_count_returned": len(frames),
        "frames": frame_paths,
        "video": "output.mp4",
        "contact_sheet": "frames_contact_sheet.png",
        "latent_snapshots": "post_step_latents.pt",
        "predicted_clean_latents": "predicted_clean_latents.pt",
        "hidden_summaries": "transformer_hidden_summaries.pt",
        "spatial_summaries": "transformer_spatial_summaries.pt",
        "_frames_for_comparison": frames,
        "_latent_payloads_for_comparison": capture.post_step_latents,
        "_predicted_clean_payloads_for_comparison": capture.predicted_clean_latents,
        "_hidden_payloads_for_comparison": capture.hidden_summaries,
        "_spatial_payloads_for_comparison": capture.spatial_summaries,
    }


def compare_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    latents_a = first["_latent_payloads_for_comparison"]
    latents_b = second["_latent_payloads_for_comparison"]
    common_steps = sorted(set(latents_a) & set(latents_b))
    latent_equal = {
        str(step): torch.equal(latents_a[step]["tensor"], latents_b[step]["tensor"])
        for step in common_steps
    }
    expected_steps = first["post_step_latent_steps_expected"]
    latent_coverage_equal = (
        first["post_step_latent_steps_captured"] == expected_steps
        and second["post_step_latent_steps_captured"] == expected_steps
        and first["post_step_latents_captured_exactly_once"]
        and second["post_step_latents_captured_exactly_once"]
    )
    hidden_a = first["_hidden_payloads_for_comparison"]
    hidden_b = second["_hidden_payloads_for_comparison"]
    expected_hidden_keys = first["hidden_summary_expected_keys"]
    hidden_coverage_equal = (
        first["hidden_summary_captured_keys"] == expected_hidden_keys
        and second["hidden_summary_captured_keys"] == expected_hidden_keys
        and first["hidden_summaries_captured_exactly_once"]
        and second["hidden_summaries_captured_exactly_once"]
    )
    common_hidden_keys = sorted(set(hidden_a) & set(hidden_b))
    hidden_equal = {
        key: torch.equal(hidden_a[key]["summary"], hidden_b[key]["summary"])
        for key in common_hidden_keys
    }
    spatial_a = first["_spatial_payloads_for_comparison"]
    spatial_b = second["_spatial_payloads_for_comparison"]
    expected_spatial_keys = first["spatial_map_expected_keys"]
    spatial_coverage_equal = (
        first["spatial_map_captured_keys"] == expected_spatial_keys
        and second["spatial_map_captured_keys"] == expected_spatial_keys
        and first["spatial_maps_captured_exactly_once"]
        and second["spatial_maps_captured_exactly_once"]
    )
    common_spatial_keys = sorted(set(spatial_a) & set(spatial_b))
    spatial_equal = {
        key: torch.equal(spatial_a[key]["summary"], spatial_b[key]["summary"])
        for key in common_spatial_keys
    }
    predicted_clean_a = first["_predicted_clean_payloads_for_comparison"]
    predicted_clean_b = second["_predicted_clean_payloads_for_comparison"]
    expected_predicted_clean_keys = first["predicted_clean_expected_keys"]
    predicted_clean_captured_keys_a = first["predicted_clean_captured_keys"]
    predicted_clean_captured_keys_b = second["predicted_clean_captured_keys"]
    predicted_clean_coverage_equal = (
        predicted_clean_captured_keys_a == expected_predicted_clean_keys
        and predicted_clean_captured_keys_b == expected_predicted_clean_keys
        and first["predicted_clean_captured_exactly_once"]
        and second["predicted_clean_captured_exactly_once"]
    )
    common_predicted_clean_steps = sorted(set(predicted_clean_a) & set(predicted_clean_b))
    predicted_clean_equal = {
        str(step): torch.equal(
            predicted_clean_a[step]["tensor"], predicted_clean_b[step]["tensor"]
        )
        for step in common_predicted_clean_steps
    }
    frames_a = first["_frames_for_comparison"]
    frames_b = second["_frames_for_comparison"]
    frames_equal = len(frames_a) == len(frames_b) and all(
        np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(frames_a, frames_b)
    )
    return {
        "same_seed": True,
        "expected_latent_steps": expected_steps,
        "both_runs_have_complete_latent_coverage": latent_coverage_equal,
        "captured_latent_steps_equal": latent_equal,
        "all_captured_latents_equal": latent_coverage_equal and bool(latent_equal) and all(latent_equal.values()),
        "expected_hidden_summary_keys": expected_hidden_keys,
        "both_runs_have_complete_hidden_summary_coverage": hidden_coverage_equal,
        "captured_hidden_summary_keys_equal": hidden_equal,
        "all_captured_hidden_summaries_equal": hidden_coverage_equal and bool(hidden_equal) and all(hidden_equal.values()),
        "expected_spatial_map_keys": expected_spatial_keys,
        "both_runs_have_complete_spatial_map_coverage": spatial_coverage_equal,
        "captured_spatial_map_keys_equal": spatial_equal,
        "all_captured_spatial_maps_equal": spatial_coverage_equal and bool(spatial_equal) and all(spatial_equal.values()),
        "expected_predicted_clean_keys": expected_predicted_clean_keys,
        "both_runs_have_complete_predicted_clean_coverage": predicted_clean_coverage_equal,
        "captured_predicted_clean_keys_equal": predicted_clean_equal,
        "all_captured_predicted_clean_latents_equal": (
            predicted_clean_coverage_equal
            and bool(predicted_clean_equal)
            and all(predicted_clean_equal.values())
        ),
        "transformer_forward_calls_exact_both_runs": (
            first["transformer_forward_calls_exact"] and second["transformer_forward_calls_exact"]
        ),
        "frame_count_equal": len(frames_a) == len(frames_b),
        "all_decoded_frames_equal": frames_equal,
        "interpretation": "bitwise replay within this runtime only; no cross-device determinism claim",
    }


def public_run_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def create_baseline_checkpoint(
    output_dir: Path,
    baseline: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Package the frozen baseline before replay, explicitly marking it incomplete."""
    checkpoint_path = output_dir.parent / f"{output_dir.name}_baseline_checkpoint.zip"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite baseline checkpoint: {checkpoint_path}")

    checkpoint_name = f"{output_dir.name}_baseline_checkpoint"
    source_names = ("runtime_preflight.json", "config_snapshot.yaml", "model_map.json")
    missing = [name for name in source_names if not (output_dir / name).is_file()]
    if missing or not (output_dir / "baseline").is_dir():
        raise FileNotFoundError(
            "Cannot checkpoint the completed baseline; missing artifacts: "
            f"{missing + ([] if (output_dir / 'baseline').is_dir() else ['baseline/'])}"
        )

    with tempfile.TemporaryDirectory(prefix="latentgate_m1_checkpoint_", dir=output_dir.parent) as temp_dir:
        stage = Path(temp_dir) / checkpoint_name
        stage.mkdir()
        for name in source_names:
            shutil.copy2(output_dir / name, stage / name)
        shutil.copytree(output_dir / "baseline", stage / "baseline")

        manifest = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "run_id": output_dir.name,
            "checkpoint_status": "baseline_complete_replay_pending",
            "checkpoint_phase": "after_baseline_before_same_seed_replay",
            "baseline_only": True,
            "passing_run": False,
            "baseline_complete": True,
            "same_seed_replay": {"status": "pending", "started": False},
            "replay_comparison_available": False,
            "machine_gate_result": "incomplete",
            "smoke_gate_status": "not_passed",
            "download_handoff": {
                "required_before_replay": True,
                "notebook_operator_confirmation": (
                    "The notebook asks the operator to confirm the ZIP appears in local Downloads."
                ),
                "limitation": (
                    "Colab cannot inspect the browser download directory or independently prove "
                    "that the browser finished saving the ZIP."
                ),
            },
            "interpretation": (
                "Baseline artifacts only. This checkpoint is intentionally incomplete: "
                "the exact same-seed replay and replay equality checks have not run. "
                "It cannot satisfy or pass the registered smoke gate."
            ),
            "baseline": public_run_result(baseline),
            "artifact_sha256": artifact_hashes(stage),
        }
        write_json(stage / "baseline_checkpoint_manifest.json", manifest)

        temporary_zip = Path(temp_dir) / f"{checkpoint_name}.zip"
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent).as_posix())
        os.replace(temporary_zip, checkpoint_path)

    return {
        "zip_path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "status": "baseline_complete_replay_pending",
    }


def await_baseline_checkpoint_ack(checkpoint: dict[str, Any]) -> None:
    """Do not start replay until the notebook has initiated the checkpoint download."""
    prefix = "ACK_BASELINE_CHECKPOINT "
    expected = f"{prefix}{checkpoint['sha256']}"
    received = sys.stdin.readline()
    if received.strip() != expected:
        raise RuntimeError(
            "Baseline checkpoint handoff was not acknowledged; the same-seed replay "
            f"was not started (expected {expected!r}, received {received.strip()!r})."
        )
    print(
        "LATENTGATE_M1_BASELINE_CHECKPOINT_ACKNOWLEDGED="
        + json.dumps(
            {
                "sha256": checkpoint["sha256"],
                "acknowledgement_scope": (
                    "Colab files.download call returned; browser save completion is not observable"
                ),
                "replay_may_start": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def execute(
    config_path: Path,
    output_dir: Path,
    preflight_only: bool,
    baseline_checkpoint_handshake: bool,
) -> None:
    config = yaml.safe_load(config_path.read_text())
    if baseline_checkpoint_handshake and not bool(
        config["generation"].get("deterministic_repeat", False)
    ):
        raise ValueError(
            "Baseline checkpoint handshake requires the configured same-seed replay."
        )
    runtime = runtime_metadata()
    print(json.dumps(runtime, indent=2, sort_keys=True))
    if preflight_only:
        write_json(output_dir / "runtime_preflight.json", runtime)
        return
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse an existing output directory: {output_dir}")
    if not runtime["cuda_available"]:
        raise RuntimeError("No CUDA GPU is attached. Refusing to load weights on CPU.")
    if "T4" not in str(runtime["device_name"]):
        raise RuntimeError(
            f"Expected a Tesla T4, got {runtime['device_name']!r}; "
            "refusing to use a different accelerator. Confirm free billing in the Colab UI."
        )
    if runtime["compute_capability"] != [7, 5]:
        raise RuntimeError(f"Unexpected T4 compute capability: {runtime['compute_capability']}")

    expected_versions = config["runtime"].get("expected_packages", {})
    version_mismatches = {
        name: {"expected": expected, "actual": runtime.get(name)}
        for name, expected in expected_versions.items()
        if runtime.get(name) != expected
    }
    if version_mismatches:
        raise RuntimeError(f"Package pins do not match the experiment config: {version_mismatches}")

    project_root = config_path.resolve().parents[1]
    image_path = project_root / Path(config["conditioning"]["image_path"])
    fixture_path = project_root / Path(config["conditioning"]["video_prompt_source"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Conditioning image is missing: {image_path}")
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Prompt fixture is missing: {fixture_path}")
    image_hash = sha256_file(image_path)
    expected_hash = config["conditioning"]["image_sha256"]
    if image_hash != expected_hash:
        raise RuntimeError(f"Fixture image hash mismatch: expected {expected_hash}, got {image_hash}")
    fixture = json.loads(fixture_path.read_text())
    prompt = fixture["video_prompt"]
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    expected_size = (int(config["generation"]["width"]), int(config["generation"]["height"]))
    if image.size != expected_size:
        raise ValueError(f"Fixture has size {image.size}, expected {expected_size}")

    seed = int(config["generation"]["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    from diffusers import CogVideoXImageToVideoPipeline

    model = config["model"]
    runtime_cfg = config["runtime"]
    scheduler_name = config["generation"].get("scheduler", "use_pretrained_config")
    if scheduler_name not in {"use_pretrained_config", "CogVideoXDPMScheduler"}:
        raise ValueError(f"Unsupported scheduler in run config: {scheduler_name!r}")
    dtype_names = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype_name = str(runtime_cfg.get("torch_dtype", "float16"))
    if dtype_name not in dtype_names:
        raise ValueError(f"Unsupported torch_dtype in run config: {dtype_name!r}")
    if runtime["compute_capability"] == [7, 5] and dtype_name != "float16":
        raise ValueError("The configured Tesla T4 route requires torch_dtype=float16")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model["repo_id"],
        revision=model["revision"],
        torch_dtype=dtype_names[dtype_name],
        use_safetensors=True,
    )
    if scheduler_name == "CogVideoXDPMScheduler":
        from diffusers import CogVideoXDPMScheduler

        pipe.scheduler = CogVideoXDPMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing=config["generation"].get("scheduler_timestep_spacing", "trailing"),
        )
    offload_mode = str(runtime_cfg.get("offload", "sequential_cpu_offload"))
    offload_methods = {
        "sequential_cpu_offload": pipe.enable_sequential_cpu_offload,
        "model_cpu_offload": pipe.enable_model_cpu_offload,
    }
    if offload_mode not in offload_methods:
        raise ValueError(f"Unsupported offload mode in run config: {offload_mode!r}")
    offload_methods[offload_mode]()
    if bool(runtime_cfg.get("vae_slicing", False)):
        pipe.vae.enable_slicing()
    if bool(runtime_cfg.get("vae_tiling", False)):
        pipe.vae.enable_tiling()
    num_blocks = len(getattr(pipe.transformer, "transformer_blocks", []))
    if num_blocks <= 0:
        raise RuntimeError("Loaded CogVideoX transformer has no transformer_blocks")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "runtime_preflight.json", runtime)
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    )
    write_json(
        output_dir / "model_map.json",
        {
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "pipeline_class": type(pipe).__name__,
            "transformer_class": type(pipe.transformer).__name__,
            "transformer_config": dict(pipe.transformer.config),
            "transformer_block_count": num_blocks,
            "vae_class": type(pipe.vae).__name__,
            "vae_config": dict(pipe.vae.config),
            "scheduler_class": type(pipe.scheduler).__name__,
            "scheduler_config": dict(pipe.scheduler.config),
            "transformer_parameter_elements_by_dtype": {
                str(dtype): sum(
                    parameter.numel()
                    for parameter in pipe.transformer.parameters()
                    if parameter.dtype == dtype
                )
                for dtype in sorted({parameter.dtype for parameter in pipe.transformer.parameters()}, key=str)
            },
        },
    )

    baseline = run_once(pipe, image, prompt, config, output_dir / "baseline", "baseline")
    replay = None
    replay_comparison = None
    checkpoint_handoff = None
    if bool(config["generation"].get("deterministic_repeat", False)):
        if baseline_checkpoint_handshake:
            checkpoint_handoff = create_baseline_checkpoint(output_dir, baseline, config)
            print(
                "LATENTGATE_M1_BASELINE_CHECKPOINT_READY="
                + json.dumps(checkpoint_handoff, sort_keys=True),
                flush=True,
            )
            await_baseline_checkpoint_ack(checkpoint_handoff)
        gc.collect()
        torch.cuda.empty_cache()
        replay = run_once(pipe, image, prompt, config, output_dir / "replay", "same_seed_replay")
        replay_comparison = compare_runs(baseline, replay)

    baseline_public = public_run_result(baseline)
    replay_public = public_run_result(replay) if replay is not None else None
    gates = {
        "t4_runtime": "T4" in str(runtime["device_name"]) and runtime["compute_capability"] == [7, 5],
        "fixture_hash_matches": image_hash == expected_hash,
        "baseline_completed": baseline["status"] == "generation_complete",
        "baseline_frame_count_matches_request": baseline["frame_count_returned"] == int(config["generation"]["num_frames"]),
        "post_step_latent_capture_complete": baseline["post_step_latent_steps_captured"] == baseline["post_step_latent_steps_expected"],
        "post_step_latent_capture_exactly_once": baseline["post_step_latents_captured_exactly_once"],
        "transformer_hidden_summary_capture_complete": baseline["hidden_summary_captured"] == baseline["hidden_summary_expected"],
        "transformer_hidden_summary_keys_complete": baseline["hidden_summary_captured_keys"] == baseline["hidden_summary_expected_keys"],
        "one_hidden_summary_call_per_block_step": baseline["hidden_summaries_captured_exactly_once"],
        "transformer_spatial_map_capture_complete": (
            baseline["spatial_map_captured_keys"] == baseline["spatial_map_expected_keys"]
        ),
        "one_spatial_map_call_per_block_step": baseline["spatial_maps_captured_exactly_once"],
        "predicted_clean_capture_complete": (
            baseline["predicted_clean_captured_keys"] == baseline["predicted_clean_expected_keys"]
        ),
        "predicted_clean_capture_exactly_once": baseline["predicted_clean_captured_exactly_once"],
        "single_transformer_forward_per_step": baseline["transformer_forward_calls_exact"],
        "same_seed_replay_exact": bool(
            replay_comparison
            and replay_comparison["all_captured_latents_equal"]
            and replay_comparison["all_captured_hidden_summaries_equal"]
            and replay_comparison["all_captured_spatial_maps_equal"]
            and replay_comparison["all_captured_predicted_clean_latents_equal"]
            and replay_comparison["transformer_forward_calls_exact_both_runs"]
            and replay_comparison["all_decoded_frames_equal"]
        ) if replay_comparison is not None else False,
        "attention_summary_capture": "deferred; no bounded CogVideoX attention adapter in this smoke runner",
    }
    bool_gates = [value for value in gates.values() if isinstance(value, bool)]
    machine_gates_passed = all(bool_gates)
    packet = {
        "experiment_id": config["experiment_id"],
        "run_id": output_dir.name,
        "status": "run_completed" if machine_gates_passed else "run_completed_with_gaps",
        "machine_gate_result": "passed" if machine_gates_passed else "failed_or_incomplete",
        "human_review_status": "pending_blinded_review",
        "smoke_gate_status": (
            "awaiting_blinded_human_review" if machine_gates_passed else "not_passed_machine_gates"
        ),
        "status_scope": (
            "run_completed reports machine-side generation and capture gates only; the ticket smoke gate "
            "cannot pass until the required blinded human review is recorded."
        ),
        "model": model,
        "runtime": runtime,
        "conditioning": {
            "image_path": str(image_path),
            "image_sha256": image_hash,
            "prompt": prompt,
            "note": "Image conditioning does not guarantee a pixel-identical first output frame.",
        },
        "generation": config["generation"],
        "baseline": baseline_public,
        "same_seed_replay": replay_public,
        "replay_comparison": replay_comparison,
        "result_persistence": (
            {
                **checkpoint_handoff,
                "status": "notebook_download_call_acknowledged_before_replay",
                "acknowledgement_scope": (
                    "Colab files.download call returned; browser save completion is not observable"
                ),
            }
            if checkpoint_handoff is not None else None
        ),
        "completion_gates": gates,
    }
    write_json(output_dir / "generation_packet.json", packet)
    write_json(
        output_dir / "capture_status.json",
        {
            "post_step_latent_steps_expected": baseline["post_step_latent_steps_expected"],
            "post_step_latent_steps_captured": baseline["post_step_latent_steps_captured"],
            "post_step_latent_callback_counts": baseline["post_step_latent_callback_counts"],
            "hidden_summary_expected": baseline["hidden_summary_expected"],
            "hidden_summary_captured": baseline["hidden_summary_captured"],
            "hidden_summary_expected_keys": baseline["hidden_summary_expected_keys"],
            "hidden_summary_captured_keys": baseline["hidden_summary_captured_keys"],
            "hidden_summary_blocks": baseline["hidden_summary_blocks"],
            "hidden_summary_call_counts": baseline["hidden_summary_call_counts"],
            "spatial_map_expected_keys": baseline["spatial_map_expected_keys"],
            "spatial_map_captured_keys": baseline["spatial_map_captured_keys"],
            "spatial_map_call_counts": baseline["spatial_map_call_counts"],
            "predicted_clean_steps_expected": baseline["predicted_clean_steps_expected"],
            "predicted_clean_steps_captured": baseline["predicted_clean_steps_captured"],
            "predicted_clean_expected_keys": baseline["predicted_clean_expected_keys"],
            "predicted_clean_captured_keys": baseline["predicted_clean_captured_keys"],
            "predicted_clean_call_counts": baseline["predicted_clean_call_counts"],
            "transformer_forward_calls_exact": baseline["transformer_forward_calls_exact"],
            "transformer_forward_calls": baseline["transformer_forward_calls"],
            "expected_inference_steps": baseline["expected_inference_steps"],
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "experiment_id": config["experiment_id"],
            "status": packet["status"],
            "artifact_sha256": artifact_hashes(output_dir),
        },
    )
    print(json.dumps(packet, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--baseline-checkpoint-handshake",
        action="store_true",
        help="Package the completed baseline and wait for notebook download acknowledgment before replay.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    project_root = config_path.parents[1]
    configured_output_dir = project_root / Path(config["capture"]["output_dir"])
    output_dir = args.output_dir.resolve() if args.output_dir else configured_output_dir.resolve()
    output_dir_preexisted = output_dir.exists()
    try:
        execute(
            config_path,
            output_dir,
            args.preflight_only,
            args.baseline_checkpoint_handshake,
        )
    except Exception as error:
        if not output_dir_preexisted:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure_path = output_dir / "failure.json"
            if not failure_path.exists():
                write_json(
                    failure_path,
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                        "runtime": runtime_metadata(),
                    },
                )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
