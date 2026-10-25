#!/usr/bin/env python3
"""Run the pinned LatentGate M0 Wan2.1 T2V microscope experiment.

Intended for a CUDA Colab runtime with the package versions recorded in
``run_config.yaml``. This is an instrumentation runner, not a steering method.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Set before importing torch. PyTorch may already be imported by a notebook,
# but setting this here still helps for direct script invocation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = SCRIPT_PATH.with_name("run_config.yaml")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel not in excluded:
            result[rel] = sha256_file(path)
    return result


def cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def save_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def version_of(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_metadata() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    capability = torch.cuda.get_device_capability(0) if cuda_available else None
    return {
        "python": sys.version,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "diffusers": version_of("diffusers"),
        "transformers": version_of("transformers"),
        "accelerate": version_of("accelerate"),
        "huggingface_hub": version_of("huggingface-hub"),
        "safetensors": version_of("safetensors"),
        "sentencepiece": version_of("sentencepiece"),
        "pyyaml": version_of("PyYAML"),
        "numpy": np.__version__,
        "pillow": version_of("Pillow"),
        "cuda_available": cuda_available,
        "device_name": device_name,
        "compute_capability": list(capability) if capability else None,
        "device_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda_available else None
        ),
        "determinism_flags": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
    }


def parameter_dtype_profile(module: Any) -> dict[str, Any]:
    profile: dict[str, dict[str, Any]] = {}
    for name, parameter in module.named_parameters():
        dtype = str(parameter.dtype).replace("torch.", "")
        entry = profile.setdefault(dtype, {"tensor_count": 0, "element_count": 0, "examples": []})
        entry["tensor_count"] += 1
        entry["element_count"] += parameter.numel()
        if len(entry["examples"]) < 6:
            entry["examples"].append(name)
    return {
        "by_dtype": profile,
        "total_tensor_count": sum(entry["tensor_count"] for entry in profile.values()),
        "total_element_count": sum(entry["element_count"] for entry in profile.values()),
    }


def model_map(pipe: Any, initial_latents: torch.Tensor) -> dict[str, Any]:
    # Diffusers 0.40 exposes ConfigMixin configs as FrozenDict mappings; older
    # releases provided a ``to_dict`` helper. Normalize both forms here.
    transformer_config = dict(pipe.transformer.config)
    vae_config = dict(pipe.vae.config)
    scheduler_config = dict(pipe.scheduler.config)
    transformer_first_parameter_dtype = str(next(pipe.transformer.parameters()).dtype)
    transformer_compute_dtype = str(getattr(pipe.transformer, "dtype", next(pipe.transformer.parameters()).dtype))
    vae_first_parameter_dtype = str(next(pipe.vae.parameters()).dtype)
    vae_compute_dtype = str(getattr(pipe.vae, "dtype", next(pipe.vae.parameters()).dtype))
    return {
        "initial_latents": {
            "shape": list(initial_latents.shape),
            "dtype": str(initial_latents.dtype),
            "meaning": "initial Gaussian noise / first flow state",
        },
        "transformer": {
            "class": type(pipe.transformer).__name__,
            "blocks_attribute": "blocks",
            "block_count": len(pipe.transformer.blocks),
            "in_channels": transformer_config.get("in_channels"),
            "out_channels": transformer_config.get("out_channels"),
            "num_attention_heads": transformer_config.get("num_attention_heads"),
            "attention_head_dim": transformer_config.get("attention_head_dim"),
            "patch_size": transformer_config.get("patch_size"),
            "text_dim": transformer_config.get("text_dim"),
            "first_parameter_dtype": transformer_first_parameter_dtype,
            "compute_dtype": transformer_compute_dtype,
            "parameter_dtype_profile": parameter_dtype_profile(pipe.transformer),
        },
        "vae": {
            "class": type(pipe.vae).__name__,
            "z_dim": vae_config.get("z_dim"),
            "in_channels": vae_config.get("in_channels"),
            "out_channels": vae_config.get("out_channels"),
            "spatial_compression_ratio": getattr(pipe, "vae_scale_factor_spatial", None),
            "temporal_compression_ratio": getattr(pipe, "vae_scale_factor_temporal", None),
            "latents_mean": vae_config.get("latents_mean"),
            "latents_std": vae_config.get("latents_std"),
            "first_parameter_dtype": vae_first_parameter_dtype,
            "compute_dtype": vae_compute_dtype,
            "parameter_dtype_profile": parameter_dtype_profile(pipe.vae),
        },
        "scheduler": {
            "class": type(pipe.scheduler).__name__,
            "order": getattr(pipe.scheduler, "order", None),
            "config": scheduler_config,
        },
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def get_sigma(scheduler: Any, index: int, sample: torch.Tensor) -> torch.Tensor:
    if not hasattr(scheduler, "sigmas"):
        raise RuntimeError(f"{type(scheduler).__name__} has no `sigmas`; cannot compute flow x0 estimate")
    if index < 0 or index >= len(scheduler.sigmas) - 1:
        raise IndexError(f"Sigma index {index} is outside the active schedule ({len(scheduler.sigmas)} values)")
    return scheduler.sigmas[index].to(device=sample.device, dtype=torch.float32)


class CaptureController:
    """Per-generation capture state shared by scheduler and module hooks."""

    def __init__(self, config: dict[str, Any], run_dir: Path, is_repeat: bool = False):
        self.config = config
        self.generation = config["generation"]
        self.capture = config["capture"]
        self.run_dir = run_dir
        self.is_repeat = is_repeat
        self.full_instrumentation = not is_repeat
        self.post_steps = {int(v) for v in self.capture.get("latent_steps_zero_based", [])}
        self.clean_steps = {int(v) for v in self.capture.get("predicted_clean_steps_zero_based", [])}
        self.block_ids = {int(v) for v in self.capture.get("transformer_blocks", [])}
        self.transformer_steps = {int(v) for v in self.capture.get("transformer_steps_zero_based", [])}
        self.blocks = {int(v): [] for v in self.capture.get("transformer_blocks", [])}
        self.branches_per_step = 2 if float(self.generation["guidance_scale"]) > 1.0 else 1
        self.transformer_call_count = 0
        self.scheduler_step_count = 0
        self.active_model_call: dict[str, Any] | None = None
        self.model_input_map: dict[str, Any] = {}
        self.latent_checkpoints: dict[int, torch.Tensor] = {}
        self.clean_payloads: dict[int, dict[str, Any]] = {}
        self.hidden_payloads: dict[str, dict[str, torch.Tensor]] = {}
        self.attention_summaries: list[dict[str, Any]] = []
        self.attention_enabled = self.full_instrumentation
        self.attention_failure: str | None = None
        self.predicted_clean_failure: str | None = None
        self.active_scheduler_step: int | None = None
        self.active_scheduler_timestep: Any = None
        self.hook_handles: list[Any] = []
        self.attention_processors: list[tuple[Any, Any]] = []
        self.extra_shapes: dict[str, Any] = {}

    def reset_for_call(self) -> None:
        self.transformer_call_count = 0
        self.scheduler_step_count = 0
        self.active_model_call = None
        self.latent_checkpoints.clear()
        self.clean_payloads.clear()
        self.hidden_payloads.clear()
        self.attention_summaries.clear()

    def transformer_pre_hook(self, _module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        call_number = self.transformer_call_count
        self.transformer_call_count += 1
        step = call_number // self.branches_per_step
        branch_index = call_number % self.branches_per_step
        branch = "positive_prompt" if branch_index == 0 else "negative_prompt"
        self.active_model_call = {"step": step, "branch": branch, "call_index": call_number}
        if self.model_input_map:
            return
        for key in ("hidden_states", "encoder_hidden_states", "timestep"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                self.model_input_map[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}

    def block_hook(self, block_id: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            context = self.active_model_call
            if not self.full_instrumentation or context is None:
                return
            step = int(context["step"])
            if block_id not in self.block_ids or step not in self.transformer_steps:
                return
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                return
            generation = self.generation
            latent_frames = (int(generation["num_frames"]) - 1) // 4 + 1
            vae_spatial = 8
            patch = getattr(self, "patch_size", (1, 2, 2))
            patch_h, patch_w = int(patch[-2]), int(patch[-1])
            tokens_per_frame = (int(generation["height"]) // vae_spatial // patch_h) * (
                int(generation["width"]) // vae_spatial // patch_w
            )
            expected = latent_frames * tokens_per_frame
            if hidden.shape[1] != expected:
                self.extra_shapes[f"block_{block_id}_step_{step}_sequence_length"] = int(hidden.shape[1])
                return
            grouped = hidden.detach().float().reshape(
                hidden.shape[0], latent_frames, tokens_per_frame, hidden.shape[-1]
            )
            key = f"block_{block_id:02d}_step_{step:03d}_{context['branch']}"
            self.hidden_payloads[key] = {
                "spatial_mean": cpu_tensor(grouped.mean(dim=2)),
                "spatial_std": cpu_tensor(grouped.std(dim=2, unbiased=False)),
            }

        return hook

    def capture_post_step(self, step: int, timestep: torch.Tensor, latents: torch.Tensor) -> None:
        if step not in self.post_steps:
            return
        snapshot = cpu_tensor(latents)
        if self.full_instrumentation:
            self.latent_checkpoints[step] = snapshot
        else:
            self.latent_checkpoints[step] = snapshot
        self.extra_shapes.setdefault("post_step_latent", {"shape": list(latents.shape), "dtype": str(latents.dtype)})

    def scheduler_wrapper(self, scheduler: Any, original_step: Any):
        if hasattr(scheduler, "convert_model_output"):
            if not hasattr(scheduler, "_latentgate_original_convert_model_output"):
                scheduler._latentgate_original_convert_model_output = scheduler.convert_model_output

            def convert_with_capture(model_output: torch.Tensor, *args: Any, **kwargs: Any):
                converted = scheduler._latentgate_original_convert_model_output(model_output, *args, **kwargs)
                controller = getattr(scheduler, "_latentgate_active_controller", None)
                if controller is not None:
                    sample = kwargs.get("sample")
                    if sample is None and args:
                        sample = args[0]
                    controller.capture_unipc_conversion(
                        scheduler=scheduler,
                        model_output=model_output,
                        sample=sample,
                        converted=converted,
                    )
                return converted

            scheduler.convert_model_output = convert_with_capture

        def wrapped_step(model_output: torch.Tensor, timestep: Any, sample: torch.Tensor, *args: Any, **kwargs: Any):
            step = self.scheduler_step_count
            self.scheduler_step_count += 1
            if self.full_instrumentation and step in self.clean_steps:
                scheduler_config = dict(scheduler.config) if hasattr(scheduler, "config") else {}
                if type(scheduler).__name__ == "FlowMatchEulerDiscreteScheduler":
                    sigma = get_sigma(scheduler, step, sample)
                    # For FlowMatch Euler, v is the guided model output and
                    # x0_hat = x_i - sigma_i * v_i before the scheduler update.
                    sample32 = sample.detach().float()
                    velocity32 = model_output.detach().float()
                    x0_hat = sample32 - sigma * velocity32
                    self.clean_payloads[step] = {
                        "step_zero_based": step,
                        "timestep": float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep),
                        "sigma": float(sigma.item()),
                        "x_i": cpu_tensor(sample32),
                        "guided_velocity": cpu_tensor(model_output),
                        "x0_hat": cpu_tensor(x0_hat),
                        "formula": "x0_hat = x_i - sigma_i * guided_velocity (FlowMatch Euler)",
                    }
                elif not (
                    type(scheduler).__name__ == "UniPCMultistepScheduler"
                    and scheduler_config.get("prediction_type") == "flow_prediction"
                    and scheduler_config.get("predict_x0") is True
                ):
                    self.predicted_clean_failure = self.predicted_clean_failure or (
                        "Predicted-clean estimate omitted: unsupported scheduler conversion "
                        f"{type(scheduler).__name__} with prediction_type="
                        f"{scheduler_config.get('prediction_type')!r} and predict_x0="
                        f"{scheduler_config.get('predict_x0')!r}"
                    )
            self.active_scheduler_step = step
            self.active_scheduler_timestep = timestep
            scheduler._latentgate_active_controller = self
            try:
                result = original_step(model_output, timestep, sample, *args, **kwargs)
            finally:
                self.active_scheduler_step = None
                self.active_scheduler_timestep = None
                scheduler._latentgate_active_controller = None
            if (
                self.full_instrumentation
                and step in self.clean_steps
                and type(scheduler).__name__ == "UniPCMultistepScheduler"
                and step not in self.clean_payloads
                and self.predicted_clean_failure is None
            ):
                self.predicted_clean_failure = (
                    "UniPCMultistepScheduler did not call its captured convert_model_output "
                    "for a requested predicted-clean step"
                )
            return result

        return wrapped_step

    def capture_unipc_conversion(
        self,
        scheduler: Any,
        model_output: torch.Tensor,
        sample: torch.Tensor | None,
        converted: torch.Tensor,
    ) -> None:
        step = self.active_scheduler_step
        if (
            not self.full_instrumentation
            or step is None
            or step not in self.clean_steps
            or type(scheduler).__name__ != "UniPCMultistepScheduler"
        ):
            return
        scheduler_config = dict(scheduler.config)
        if scheduler_config.get("prediction_type") != "flow_prediction" or scheduler_config.get("predict_x0") is not True:
            return
        if sample is None:
            self.predicted_clean_failure = "UniPC convert_model_output was called without its pre-step sample"
            return
        sigma_index = getattr(scheduler, "step_index", None)
        if sigma_index is None or int(sigma_index) != step:
            self.predicted_clean_failure = (
                f"UniPC sigma index mismatch at requested step {step}: step_index={sigma_index!r}"
            )
            return
        sigma = get_sigma(scheduler, step, sample)
        timestep = self.active_scheduler_timestep
        self.clean_payloads[step] = {
            "step_zero_based": step,
            "timestep": float(timestep.item()) if isinstance(timestep, torch.Tensor) else float(timestep),
            "sigma": float(sigma.item()),
            "x_i": cpu_tensor(sample),
            "guided_velocity": cpu_tensor(model_output),
            "x0_hat": cpu_tensor(converted),
            "formula": "Diffusers UniPC convert_model_output (predict_x0=true, prediction_type=flow_prediction)",
            "scheduler_step_index": int(sigma_index),
            "scheduler_thresholding": bool(scheduler_config.get("thresholding", False)),
        }

    def callback(self, _pipe: Any, step: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        self.capture_post_step(step, timestep, callback_kwargs["latents"])
        return {}

    def attention_record(self, record: dict[str, Any]) -> None:
        self.attention_summaries.append(record)

    def disable_attention(self, exc: BaseException) -> None:
        if self.attention_failure is None:
            self.attention_failure = f"{type(exc).__name__}: {exc}"
        self.attention_enabled = False


class BoundedWanSelfAttentionProcessor:
    """Wan SDPA delegate plus sampled-query, bounded self-attention statistics.

    The probability matrix is computed only for a deterministic set of query
    tokens, never for all video-token pairs. Normal model attention remains in
    the stock Diffusers processor.
    """

    def __init__(self, controller: CaptureController, block_id: int, delegate: Any, query_count: int = 16):
        self.controller = controller
        self.block_id = block_id
        self.delegate = delegate
        self.query_count = query_count

    @staticmethod
    def _rotary(hidden: torch.Tensor, rotary_emb: Any) -> torch.Tensor:
        if rotary_emb is None:
            return hidden
        freqs_cos, freqs_sin = rotary_emb
        x1, x2 = hidden.unflatten(-1, (-1, 2)).unbind(-1)
        cos = freqs_cos[..., 0::2]
        sin = freqs_sin[..., 1::2]
        out = torch.empty_like(hidden)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out.type_as(hidden)

    def _summarize(self, attn: Any, hidden_states: torch.Tensor, rotary_emb: Any) -> None:
        context = self.controller.active_model_call
        if (
            not self.controller.attention_enabled
            or not self.controller.full_instrumentation
            or context is None
            or int(context["step"]) not in self.controller.transformer_steps
        ):
            return
        step = int(context["step"])
        if self.block_id not in self.controller.block_ids:
            return

        if getattr(attn, "fused_projections", False):
            qkv = attn.to_qkv(hidden_states)
            query, key, _value = qkv.chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        query = self._rotary(query, rotary_emb)
        key = self._rotary(key, rotary_emb)

        batch, sequence, heads, head_dim = query.shape
        query_count = min(self.query_count, sequence)
        indices = torch.linspace(0, sequence - 1, steps=query_count, device=query.device).round().long()
        # Only [batch, heads, sampled_queries, all_keys] is materialized.
        query_sample = query.index_select(1, indices).permute(0, 2, 1, 3).float()
        keys = key.permute(0, 2, 3, 1).float()
        scores = torch.matmul(query_sample, keys) / math.sqrt(head_dim)
        probabilities = torch.softmax(scores, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        max_probability = probabilities.max(dim=-1).values
        self_probability = probabilities.gather(
            -1, indices.view(1, 1, -1, 1).expand(batch, heads, -1, 1)
        )

        generation = self.controller.generation
        latent_frames = (int(generation["num_frames"]) - 1) // 4 + 1
        vae_spatial = 8
        patch = getattr(self.controller, "patch_size", (1, 2, 2))
        patch_h, patch_w = int(patch[-2]), int(patch[-1])
        tokens_per_frame = (int(generation["height"]) // vae_spatial // patch_h) * (
            int(generation["width"]) // vae_spatial // patch_w
        )
        frame_mass = None
        if sequence == latent_frames * tokens_per_frame:
            frame_mass = probabilities.reshape(batch, heads, query_count, latent_frames, tokens_per_frame)
            frame_mass = frame_mass.sum(dim=-1).mean(dim=(0, 1, 2)).detach().cpu().tolist()

        self.controller.attention_record(
            {
                "step_zero_based": step,
                "block": self.block_id,
                "branch": context["branch"],
                "attention": "attn1 self-attention",
                "metric": "sampled_query_self_attention_probability_summary",
                "query_token_indices": indices.detach().cpu().tolist(),
                "query_sampling": "uniform indices from first through last transformer token",
                "key_tokens": int(sequence),
                "attention_entropy_nats_mean": float(entropy.mean().item()),
                "attention_entropy_nats_per_head_mean": entropy.mean(dim=(0, 2)).detach().cpu().tolist(),
                "max_attention_probability_mean": float(max_probability.mean().item()),
                "same_token_probability_mean": float(self_probability.mean().item()),
                "mean_probability_mass_by_latent_frame": frame_mass,
                "limitations": "self-attention only; 16 sampled query tokens; all keys; no cross-attention map",
            }
        )

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: Any = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            try:
                self._summarize(attn, hidden_states, rotary_emb)
            except Exception as exc:  # instrumentation must not break generation
                self.controller.disable_attention(exc)
        return self.delegate(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


def install_hooks(pipe: Any, controller: CaptureController, patch_size: Any) -> None:
    controller.patch_size = patch_size
    controller.hook_handles.append(
        pipe.transformer.register_forward_pre_hook(controller.transformer_pre_hook, with_kwargs=True)
    )
    for block_id in sorted(controller.block_ids):
        if block_id >= len(pipe.transformer.blocks):
            raise IndexError(f"Configured transformer block {block_id} exceeds block count {len(pipe.transformer.blocks)}")
        controller.hook_handles.append(
            pipe.transformer.blocks[block_id].register_forward_hook(controller.block_hook(block_id))
        )
        if not controller.full_instrumentation or block_id not in controller.block_ids:
            continue
        attn = pipe.transformer.blocks[block_id].attn1
        original = attn.processor
        controller.attention_processors.append((attn, original))
        attn.set_processor(BoundedWanSelfAttentionProcessor(controller, block_id, original))


def remove_hooks(controller: CaptureController) -> None:
    for handle in controller.hook_handles:
        handle.remove()
    controller.hook_handles.clear()
    for attn, original in controller.attention_processors:
        attn.set_processor(original)
    controller.attention_processors.clear()


def make_initial_noise(pipe: Any, generation: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Generator]:
    device = pipe._execution_device
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=int(pipe.transformer.config.in_channels),
        height=int(generation["height"]),
        width=int(generation["width"]),
        num_frames=int(generation["num_frames"]),
        dtype=torch.float32,
        device=device,
        generator=generator,
        latents=None,
    )
    return initial, generator


def save_video_frames(frames: np.ndarray, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    images = []
    for idx, frame in enumerate(frames):
        pixels = (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        path = target / f"frame_{idx:03d}.png"
        image.save(path, optimize=True)
        frame_paths.append(path.as_posix())
        images.append(image.copy())

    if images:
        cell_w, cell_h = images[0].size
        columns = min(5, len(images))
        rows = math.ceil(len(images) / columns)
        sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + 22)), "white")
        draw = ImageDraw.Draw(sheet)
        for idx, image in enumerate(images):
            x = (idx % columns) * cell_w
            y = (idx // columns) * (cell_h + 22)
            sheet.paste(image, (x, y))
            draw.text((x + 6, y + cell_h + 3), f"frame {idx:02d}", fill="black")
        sheet.save(target.parent / f"{target.name}_contact_sheet.png", optimize=True)
    return frame_paths


def vae_roundtrip(pipe: Any, frames: np.ndarray, output_dir: Path) -> dict[str, Any]:
    """Encode and decode generated RGB frames using the Wan VAE's latent stats."""
    if len(pipe.vae.config.latents_mean) != int(pipe.vae.config.z_dim):
        raise RuntimeError("VAE latents_mean length differs from z_dim")
    device = pipe._execution_device
    input_pixels = torch.from_numpy(np.asarray(frames, dtype=np.float32)).permute(3, 0, 1, 2).unsqueeze(0)
    input_tensor = input_pixels.to(device=device, dtype=torch.float32).mul(2.0).sub(1.0)
    mean = torch.tensor(pipe.vae.config.latents_mean, dtype=torch.float32, device=device).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    )
    std = torch.tensor(pipe.vae.config.latents_std, dtype=torch.float32, device=device).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    )
    with torch.inference_mode():
        encoded_raw = pipe.vae.encode(input_tensor).latent_dist.mode().float()
        normalized = (encoded_raw - mean) / std
        decoded_raw = normalized * std + mean
        reconstructed = pipe.vae.decode(decoded_raw, return_dict=False)[0].float()
    recon = ((reconstructed.clamp(-1.0, 1.0) + 1.0) * 0.5).permute(0, 2, 3, 4, 1)[0].cpu().numpy()
    reference = np.clip(frames.astype(np.float32), 0.0, 1.0)
    if recon.shape != reference.shape:
        return {
            "status": "shape_mismatch",
            "input_video_shape_bfchw": list(input_tensor.shape),
            "encoded_raw_shape": list(encoded_raw.shape),
            "decoded_video_shape_bfhwc": list(recon.shape),
            "reference_video_shape_bfhwc": list(reference.shape),
        }
    error = recon - reference
    mse = float(np.mean(np.square(error)))
    per_frame = []
    for idx in range(len(reference)):
        frame_mse = float(np.mean(np.square(error[idx])))
        per_frame.append(
            {
                "frame": idx,
                "mae": float(np.mean(np.abs(error[idx]))),
                "rmse": math.sqrt(frame_mse),
                "psnr_db": float("inf") if frame_mse == 0 else -10.0 * math.log10(frame_mse),
            }
        )
    frame_dir = output_dir / "vae_roundtrip_frames"
    save_video_frames(recon, frame_dir)
    return {
        "status": "ok",
        "method": "Wan VAE encode(mode) -> (raw_latent - latents_mean) / latents_std -> inverse normalization -> decode",
        "input_video_shape_bfchw": list(input_tensor.shape),
        "encoded_raw_shape": list(encoded_raw.shape),
        "normalized_latent_shape": list(normalized.shape),
        "decoded_video_shape_bfhwc": list(recon.shape),
        "pixel_range": "both reference and round-trip frames in [0, 1] RGB",
        "mae": float(np.mean(np.abs(error))),
        "rmse": math.sqrt(mse),
        "psnr_db": float("inf") if mse == 0 else -10.0 * math.log10(mse),
        "max_absolute_error": float(np.max(np.abs(error))),
        "per_frame": per_frame,
        "roundtrip_frames_dir": frame_dir.name,
        "limitations": "VAE reconstruction error includes RGB preprocessing and deterministic posterior mode; it is not model-generation error.",
    }


def decode_predicted_clean(pipe: Any, normalized_latents: torch.Tensor) -> np.ndarray:
    """Decode Wan's normalized predicted-clean latent into [frames, H, W, RGB]."""
    vae = pipe.vae
    device = pipe._execution_device
    dtype = next(vae.parameters()).dtype
    latents = normalized_latents.to(device=device, dtype=torch.float32)
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32, device=device).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32, device=device).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    raw_latents = (latents * std + mean).to(dtype=dtype)
    with torch.inference_mode():
        decoded = vae.decode(raw_latents, return_dict=False)[0]
    video = ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 0.5)
    return video.permute(0, 2, 3, 4, 1)[0].cpu().numpy()


def write_autopsy_report(
    path: Path,
    *,
    config: dict[str, Any],
    packet: dict[str, Any],
    preview_report: dict[str, Any],
    roundtrip_report: dict[str, Any],
    hook_status: dict[str, Any],
) -> None:
    generation = config["generation"]
    runtime = packet["runtime"]
    repeat = packet["determinism_repeat"]
    lines = [
        f"# M0 Autopsy: {packet['experiment_id']}",
        "",
        f"- **Model:** `{packet['model']['repo_id']}`",
        f"- **Revision:** `{packet['model']['revision']}`",
        f"- **Prompt:** {generation['prompt']}",
        f"- **Seed / size / steps:** {generation['seed']} / {generation['width']}×{generation['height']} / {generation['num_inference_steps']}",
        f"- **Frames requested:** {generation['num_frames']}",
        f"- **Runtime:** {runtime.get('device_name')} (Python {runtime.get('python_version')}, PyTorch {runtime.get('torch')}, CUDA {runtime.get('cuda_runtime')})",
        f"- **Generation time:** {packet['generation_seconds']:.2f} s",
        f"- **Peak allocated GPU memory:** {packet['peak_cuda_allocated_bytes'] / (1024**3):.2f} GiB",
        f"- **VAE round trip:** {roundtrip_report.get('status')}; RMSE {roundtrip_report.get('rmse', 'n/a')}",
        f"- **Same-seed replay:** {repeat.get('status')}",
        f"- **M0 completion assessment:** {packet['m0_completion_assessment']['status']}",
        "",
        "## Model checks",
        "",
        f"- Transformer architecture matches expected config: {packet['model_config_check'].get('matches')}",
        f"- Runtime tensor dtypes match expected config: {packet['dtype_check'].get('matches')}",
        "",
        "## M0 gates",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'INCOMPLETE'} — {name.replace('_', ' ')}"
        for name, passed in packet["m0_completion_assessment"]["gates"].items()
    )
    lines.extend([
        "",
        "## Visual outputs",
        "",
        "- Final video frames: `run_001/frames/`",
        "- Final frame contact sheet: `run_001/frames_contact_sheet.png`",
        "- VAE round-trip frames: `run_001/vae_roundtrip_frames/`",
        "- VAE round-trip report: `run_001/vae_roundtrip_report.json`",
    ])
    preview_steps = preview_report.get("steps", {})
    if preview_steps:
        lines.extend(["", "## Predicted-clean previews", ""])
        lines.extend(
            f"- Step {step}: `{record['contact_sheet']}` ({record['decoded_video_shape_bfhwc'][0]} frames)"
            for step, record in sorted(preview_steps.items(), key=lambda pair: int(pair[0]))
        )
    else:
        lines.extend(["", "## Predicted-clean previews", "", "No previews were decoded."])
    lines.extend(
        [
            "",
            "## Capture status",
            "",
            f"- Post-step latents: {hook_status['post_step_latent_steps_captured']} / {hook_status['post_step_latent_steps_expected']}",
            f"- Predicted-clean latents: {hook_status['predicted_clean_steps_captured']} / {hook_status['predicted_clean_steps_expected']}",
            f"- Transformer summaries: {hook_status['hidden_state_summary_count']} / {hook_status['hidden_state_summary_expected']}",
            f"- Attention summaries: {hook_status['attention_summary_count']} / {hook_status['attention_summary_expected']}",
            f"- Attention limitation/failure: {hook_status['attention_limitation_or_failure'] or 'none reported'}",
            f"- Predicted-clean limitation/failure: {hook_status['predicted_clean_limitation_or_failure'] or 'none reported'}",
            "",
            "## Interpretation boundary",
            "",
            "This run instruments one prompt and one seed. It is a pipeline feasibility result, not evidence that hidden states encode a target variable or that an intervention improves generation. The attention output is a sampled-query self-attention summary; transformer outputs are spatially pooled per latent frame.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def save_predicted_clean_previews(
    pipe: Any, clean_payloads: dict[int, dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    """Decode selected normalized Wan x0 estimates into RGB frame previews."""
    result: dict[str, Any] = {"status": "ok", "steps": {}}
    for step, payload in sorted(clean_payloads.items()):
        frames = decode_predicted_clean(pipe, payload["x0_hat"])
        preview_dir = output_dir / "predicted_clean_previews" / f"step_{step:03d}" / "frames"
        save_video_frames(frames, preview_dir)
        result["steps"][str(step)] = {
            "status": "ok",
            "decoded_video_shape_bfhwc": list(frames.shape),
            "frame_dir": preview_dir.relative_to(output_dir).as_posix(),
            "contact_sheet": (preview_dir.parent / "frames_contact_sheet.png").relative_to(output_dir).as_posix(),
            "semantics": "VAE decode of x0_hat = x_i - sigma_i * guided_velocity after inverse Wan latent normalization",
        }
        del frames
        torch.cuda.empty_cache()
    if not clean_payloads:
        result["status"] = "no_predicted_clean_snapshots"
    return result


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if tuple(a.shape) != tuple(b.shape):
        return {"shape_match": False, "left_shape": list(a.shape), "right_shape": list(b.shape)}
    difference = a.float() - b.float()
    abs_diff = difference.abs()
    return {
        "shape_match": True,
        "exact_equal": bool(torch.equal(a, b)),
        "max_absolute_difference": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_absolute_difference": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()) if difference.numel() else 0.0,
    }


def prepare_wan_prompt_embeddings(
    repo_id: str,
    revision: str,
    generation: dict[str, Any],
    conditioning_config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Encode both CFG branches before loading the rest of Wan into host RAM.

    Wan's UMT5-XXL encoder dominates the checkpoint size. Loading it separately
    with Accelerate's device map lets its weight shards span available CPU/GPU
    memory; after the two prompt embeddings are materialized, the encoder is
    released and the generation pipeline is loaded without that component.
    """
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import T5Tokenizer, UMT5EncoderModel

    max_sequence_length = int(generation.get("max_sequence_length", 226))
    dtype_name = str(conditioning_config.get("text_encoder_dtype", "float16"))
    if dtype_name != "float16":
        raise ValueError(f"The Colab T4 prompt encoder must use float16, found {dtype_name!r}")
    device_map = str(conditioning_config.get("text_encoder_device_map", "auto"))
    configured_max_memory = conditioning_config.get("text_encoder_max_memory", {})
    max_memory = {
        (int(key) if str(key).isdigit() else str(key)): str(value)
        for key, value in configured_max_memory.items()
    }
    tokenizer = T5Tokenizer.from_pretrained(
        repo_id,
        subfolder="tokenizer",
        revision=revision,
        local_files_only=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    text_encoder = UMT5EncoderModel.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        revision=revision,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory=max_memory or None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    text_encoder.eval()
    input_device = text_encoder.get_input_embeddings().weight.device

    def encode_one(text: str) -> torch.Tensor:
        cleaned = prompt_clean(text)
        tokens = tokenizer(
            [cleaned],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(input_device)
        attention_mask = tokens.attention_mask.to(input_device)
        sequence_length = int(attention_mask.gt(0).sum(dim=1)[0].item())
        with torch.inference_mode():
            hidden = text_encoder(input_ids, attention_mask).last_hidden_state[0, :sequence_length]
        hidden = hidden.to(device="cpu", dtype=torch.float16)
        if sequence_length < max_sequence_length:
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, max_sequence_length - sequence_length))
        return hidden.unsqueeze(0).contiguous()

    started = time.perf_counter()
    prompt_embeds = encode_one(str(generation["prompt"]))
    negative_embeds = encode_one(str(generation.get("negative_prompt") or ""))
    encoder_metadata = {
        "class": type(text_encoder).__name__,
        "repo_id": repo_id,
        "revision": revision,
        "parameter_count": sum(parameter.numel() for parameter in text_encoder.parameters()),
        "parameter_dtype": str(next(text_encoder.parameters()).dtype),
        "device_map": getattr(text_encoder, "hf_device_map", {"default": str(input_device)}),
        "max_memory": max_memory,
        "prompt_embedding_shape": list(prompt_embeds.shape),
        "negative_prompt_embedding_shape": list(negative_embeds.shape),
        "embedding_dtype": str(prompt_embeds.dtype),
        "max_sequence_length": max_sequence_length,
        "encode_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else None,
        "pipeline_text_encoder_loaded": False,
    }
    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt_embeds, negative_embeds, encoder_metadata


def repeat_is_exact(repeat_report: dict[str, Any], expected_post_steps: list[int]) -> bool:
    if repeat_report.get("status") != "completed":
        return False
    comparisons = repeat_report.get("comparisons", {})
    if not all(name in comparisons for name in ("initial_noise", "final_latent", "post_step_latents")):
        return False
    post_steps = comparisons.get("post_step_latents", {})
    if set(post_steps) != {str(step) for step in expected_post_steps}:
        return False
    records = [comparisons["initial_noise"], comparisons["final_latent"], *post_steps.values()]
    return all(record.get("shape_match") is True and record.get("exact_equal") is True for record in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to m0/run_config.yaml")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: m0/runs/<experiment_id>)")
    parser.add_argument("--single-run", action="store_true", help="Skip the same-seed repeat even if configured")
    parser.add_argument("--no-attention-summary", action="store_true", help="Use stock attention; record that no attention statistics were captured")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Config file does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    experiment_id = config["experiment_id"]
    output_root = (args.output_dir or SCRIPT_PATH.parent / "runs" / experiment_id).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_root}. Choose a new --output-dir.")
    output_root.mkdir(parents=True, exist_ok=True)
    full_run_dir = output_root / "run_001"
    full_run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_root / "config_snapshot.yaml")

    generation = config["generation"]
    if (int(generation["num_frames"]) - 1) % 4 != 0:
        raise SystemExit("Wan2.1 requires num_frames = 4n + 1; update the config.")
    revision = str(config["model"]["revision"])
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise SystemExit("The model revision must be an immutable 40-character commit SHA.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Select a Colab GPU runtime before running M0.")

    seed_everything(int(generation["seed"]))
    runtime = runtime_metadata()
    expected_runtime = config.get("runtime", {})
    runtime_differences = {}
    runtime_key_map = {
        "python": "python_version",
        "torch": "torch",
        "cuda": "cuda_runtime",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "huggingface_hub": "huggingface_hub",
        "safetensors": "safetensors",
        "sentencepiece": "sentencepiece",
        "accelerator": "device_name",
        "compute_capability": "compute_capability",
    }
    for expected_key, observed_key in runtime_key_map.items():
        observed = runtime.get(observed_key)
        expected = expected_runtime.get(expected_key)
        if expected is None or observed is None:
            continue
        if expected_key == "compute_capability" and isinstance(observed, list):
            observed = ".".join(str(part) for part in observed)
        if expected_key == "accelerator":
            expected = str(expected).replace("NVIDIA ", "").strip()
            observed = str(observed).replace("NVIDIA ", "").strip()
        if str(expected) != str(observed):
            runtime_differences[expected_key] = {"expected": str(expected), "observed": str(observed)}

    from diffusers import WanPipeline

    conditioning = config.get("conditioning", {})
    if not conditioning.get("precompute_text_embeddings", False):
        raise SystemExit("M0 on the constrained Colab host requires precomputed Wan prompt embeddings.")
    print("Precomputing positive and negative prompt embeddings with the Wan text encoder …", flush=True)
    prompt_embeds_cpu, negative_prompt_embeds_cpu, conditioning_metadata = prepare_wan_prompt_embeddings(
        config["model"]["repo_id"],
        revision,
        generation,
        conditioning,
    )
    conditioning_path = full_run_dir / "prompt_conditioning.pt"
    save_torch(
        conditioning_path,
        {
            "prompt_embeds": prompt_embeds_cpu,
            "negative_prompt_embeds": negative_prompt_embeds_cpu,
            "metadata": conditioning_metadata,
        },
    )

    print(f"Loading {config['model']['repo_id']} at {revision} …", flush=True)
    pipe = WanPipeline.from_pretrained(
        config["model"]["repo_id"],
        revision=revision,
        torch_dtype=torch.float16,
        # The real UMT5-XXL encoder was loaded separately to compute the exact
        # prompt embeddings. Diffusers skips a component explicitly passed as
        # None, and every pipeline call below supplies both embedding tensors.
        text_encoder=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    pipe.vae.to(dtype=torch.float32)
    pipe.model_cpu_offload_seq = "transformer->vae"
    pipe.enable_model_cpu_offload(gpu_id=0)
    prompt_embeds = prompt_embeds_cpu.to(device=pipe._execution_device, dtype=torch.float16)
    negative_prompt_embeds = negative_prompt_embeds_cpu.to(device=pipe._execution_device, dtype=torch.float16)

    initial_noise, generator = make_initial_noise(pipe, generation, int(generation["seed"]))
    tensor_map = model_map(pipe, initial_noise)
    expected_transformer = config["model"].get("expected_transformer", {})
    observed_transformer = tensor_map["transformer"]
    model_config_checks = {}
    for key in ("block_count", "num_attention_heads", "attention_head_dim", "patch_size"):
        if key not in expected_transformer:
            continue
        expected = expected_transformer[key]
        observed = observed_transformer.get(key)
        if isinstance(observed, tuple):
            observed = list(observed)
        model_config_checks[key] = {"expected": expected, "observed": observed, "matches": expected == observed}
    model_config_check = {
        "checks": model_config_checks,
        "matches": bool(model_config_checks) and all(item["matches"] for item in model_config_checks.values()),
    }
    dtype_check = {}
    for name, expected, observed in (
        ("transformer", expected_runtime.get("transformer_dtype"), observed_transformer["compute_dtype"]),
        ("vae", expected_runtime.get("vae_dtype"), tensor_map["vae"]["compute_dtype"]),
    ):
        def normalize_dtype(value: Any) -> str | None:
            return str(value).replace("torch.", "") if value is not None else None

        dtype_check[name] = {
            "expected": expected,
            "observed": observed,
            "basis": "effective_model_compute_dtype; individual retained parameter dtypes are in tensor_map",
            "matches": expected is None or normalize_dtype(expected) == normalize_dtype(observed),
        }
    dtype_check["matches"] = all(item["matches"] for item in dtype_check.values())
    save_torch(full_run_dir / "initial_noise.pt", {"latents": cpu_tensor(initial_noise), "seed": int(generation["seed"])})

    controller = CaptureController(config, full_run_dir, is_repeat=False)
    if args.no_attention_summary:
        controller.attention_enabled = False
        controller.attention_failure = "disabled_by_cli_flag"
    patch_size = pipe.transformer.config.patch_size
    install_hooks(pipe, controller, patch_size)
    original_scheduler_step = pipe.scheduler.step
    pipe.scheduler.step = controller.scheduler_wrapper(pipe.scheduler, original_scheduler_step)

    print("Running M0 baseline …", flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    start_time = time.perf_counter()
    with torch.inference_mode():
        output = pipe(
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=int(generation["height"]),
            width=int(generation["width"]),
            num_frames=int(generation["num_frames"]),
            num_inference_steps=int(generation["num_inference_steps"]),
            guidance_scale=float(generation["guidance_scale"]),
            generator=generator,
            latents=initial_noise,
            output_type="np",
            callback_on_step_end=controller.callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    generation_seconds = time.perf_counter() - start_time
    baseline_input_map = dict(controller.model_input_map)
    peak_memory_bytes = torch.cuda.max_memory_allocated(0)
    video = np.asarray(output.frames[0], dtype=np.float32)
    del initial_noise
    torch.cuda.empty_cache()
    frame_paths = save_video_frames(video, full_run_dir / "frames")
    try:
        roundtrip_report = vae_roundtrip(pipe, video, full_run_dir)
    except Exception as exc:
        roundtrip_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        torch.cuda.empty_cache()
    write_json(full_run_dir / "vae_roundtrip_report.json", roundtrip_report)
    del output
    torch.cuda.empty_cache()

    # Checkpoint callback captures post-update states x_(i+1). Save selected
    # pre-update x_i and guided velocity with x0_hat as separate forensic data.
    for step, latent in sorted(controller.latent_checkpoints.items()):
        save_torch(
            full_run_dir / "latents" / f"post_step_{step:03d}.pt",
            {"step_zero_based": step, "latent_after_scheduler_step": latent},
        )
    for step, payload in sorted(controller.clean_payloads.items()):
        save_torch(full_run_dir / "predicted_clean" / f"pre_step_{step:03d}.pt", payload)
    try:
        clean_preview_report = save_predicted_clean_previews(pipe, controller.clean_payloads, full_run_dir)
    except Exception as exc:
        clean_preview_report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "steps": {}}
        torch.cuda.empty_cache()
    peak_memory_bytes = max(peak_memory_bytes, torch.cuda.max_memory_allocated(0))
    hidden_path = full_run_dir / "transformer_summaries.pt"
    if controller.hidden_payloads:
        save_torch(hidden_path, controller.hidden_payloads)
    attention_path = full_run_dir / "attention_summaries.json"
    write_json(attention_path, controller.attention_summaries)
    expected_post_steps = sorted({int(s) for s in config["capture"].get("latent_steps_zero_based", [])})
    expected_clean_steps = sorted({int(s) for s in config["capture"].get("predicted_clean_steps_zero_based", [])})
    captured_post_steps = sorted(int(s) for s in controller.latent_checkpoints)
    captured_clean_steps = sorted(int(s) for s in controller.clean_payloads)
    expected_summary_count = (
        len(controller.block_ids) * len(controller.transformer_steps) * controller.branches_per_step
    )
    hook_capture_status = {
        "post_step_latent_steps_expected": expected_post_steps,
        "post_step_latent_steps_captured": captured_post_steps,
        "predicted_clean_steps_expected": expected_clean_steps,
        "predicted_clean_steps_captured": captured_clean_steps,
        "hidden_state_summary_count": len(controller.hidden_payloads),
        "hidden_state_summary_expected": expected_summary_count,
        "attention_summary_count": len(controller.attention_summaries),
        "attention_summary_expected": expected_summary_count,
        "attention_summary_status": "captured" if len(controller.attention_summaries) == expected_summary_count else "partial_or_unavailable",
        "attention_limitation_or_failure": controller.attention_failure,
        "predicted_clean_limitation_or_failure": controller.predicted_clean_failure,
        "predicted_clean_preview_status": clean_preview_report.get("status"),
        "predicted_clean_preview_count": len(clean_preview_report.get("steps", {})),
        "hidden_state_shape_notes": controller.extra_shapes,
    }
    write_json(full_run_dir / "capture_status.json", hook_capture_status)
    remove_hooks(controller)
    pipe.scheduler.step = original_scheduler_step

    # Release full-run CPU snapshots before repeat to keep Colab host RAM lower.
    del controller
    torch.cuda.empty_cache()

    repeat_requested = bool(generation.get("deterministic_repeat", False)) and not args.single_run
    repeat_report: dict[str, Any] = {"requested": repeat_requested, "status": "not_requested"}
    if repeat_requested:
        repeat_dir = output_root / "run_002_replay"
        replay = CaptureController(config, repeat_dir, is_repeat=True)
        replay.patch_size = patch_size
        replay_hook = pipe.transformer.register_forward_pre_hook(replay.transformer_pre_hook, with_kwargs=True)
        replay.scheduler_step_count = 0
        original_scheduler_step = pipe.scheduler.step
        pipe.scheduler.step = replay.scheduler_wrapper(pipe.scheduler, original_scheduler_step)
        seed_everything(int(generation["seed"]))
        replay_noise, replay_generator = make_initial_noise(pipe, generation, int(generation["seed"]))
        replay_noise_cpu = cpu_tensor(replay_noise)
        torch.cuda.reset_peak_memory_stats(0)
        replay_start = time.perf_counter()
        with torch.inference_mode():
            replay_output = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                height=int(generation["height"]),
                width=int(generation["width"]),
                num_frames=int(generation["num_frames"]),
                num_inference_steps=int(generation["num_inference_steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                generator=replay_generator,
                latents=replay_noise,
                output_type="latent",
                callback_on_step_end=replay.callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        replay_seconds = time.perf_counter() - replay_start
        replay_peak_memory = torch.cuda.max_memory_allocated(0)
        final_replay_latent = cpu_tensor(replay_output.frames)
        replay_hook.remove()
        pipe.scheduler.step = original_scheduler_step

        initial_ref = torch.load(full_run_dir / "initial_noise.pt", map_location="cpu", weights_only=True)["latents"]
        comparisons = {"initial_noise": compare_tensors(initial_ref, replay_noise_cpu), "post_step_latents": {}}
        for step, latent in sorted(replay.latent_checkpoints.items()):
            reference_file = full_run_dir / "latents" / f"post_step_{step:03d}.pt"
            if reference_file.exists():
                reference = torch.load(reference_file, map_location="cpu", weights_only=True)["latent_after_scheduler_step"]
                comparisons["post_step_latents"][str(step)] = compare_tensors(reference, latent)
        last_step = int(generation["num_inference_steps"]) - 1
        final_reference_file = full_run_dir / "latents" / f"post_step_{last_step:03d}.pt"
        if final_reference_file.exists():
            final_reference = torch.load(final_reference_file, map_location="cpu", weights_only=True)["latent_after_scheduler_step"]
            comparisons["final_latent"] = compare_tensors(final_reference, final_replay_latent)
        repeat_report = {
            "requested": True,
            "status": "completed",
            "same_model_revision_and_generation_parameters": True,
            "instrumentation_on_repeat": "selected post-step latents and initial noise only; full baseline captures are in run_001",
            "duration_seconds": replay_seconds,
            "peak_cuda_allocated_bytes": replay_peak_memory,
            "comparisons": comparisons,
            "note": "Bitwise equality is reported; CUDA kernels may be nondeterministic even with fixed seeds and deterministic flags.",
        }
        del replay_output, replay_noise, replay_noise_cpu, final_replay_latent, replay
        torch.cuda.empty_cache()

    replay_exact = repeat_is_exact(repeat_report, expected_post_steps) if repeat_requested else False
    preview_steps = {str(step) for step in clean_preview_report.get("steps", {})}
    completion_gates = {
        "runtime_matches_config": not runtime_differences,
        "transformer_architecture_matches": model_config_check.get("matches") is True,
        "runtime_dtypes_match": dtype_check.get("matches") is True,
        "post_step_latents_complete": captured_post_steps == expected_post_steps,
        "predicted_clean_latents_complete": captured_clean_steps == expected_clean_steps,
        "predicted_clean_previews_complete": (
            clean_preview_report.get("status") == "ok"
            and preview_steps == {str(step) for step in expected_clean_steps}
        ),
        "hidden_state_summaries_complete": hook_capture_status["hidden_state_summary_count"] == expected_summary_count,
        "attention_summaries_complete": hook_capture_status["attention_summary_count"] == expected_summary_count,
        "vae_roundtrip_complete": roundtrip_report.get("status") == "ok",
        "same_seed_replay_exact": replay_exact,
    }
    m0_completion_assessment = {
        "status": "complete" if all(completion_gates.values()) else "incomplete",
        "gates": completion_gates,
    }

    packet = {
        "packet_schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": "M0_WAN21_T2V13B_T4_001_run_001",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {
            "repo_id": config["model"]["repo_id"],
            "revision": revision,
            "source_url": config["model"].get("source_url"),
        },
        "runtime": runtime,
        "runtime_differences_from_config": runtime_differences,
        "model_config_check": model_config_check,
        "dtype_check": dtype_check,
        "prompt_conditioning": conditioning_metadata,
        "generation": generation,
        "tensor_map": tensor_map,
        "transformer_inputs_observed": baseline_input_map,
        "artifacts": {
            "frame_paths": [Path(p).relative_to(output_root).as_posix() for p in frame_paths],
            "post_step_latent_steps": hook_capture_status["post_step_latent_steps_captured"],
            "predicted_clean_steps": hook_capture_status["predicted_clean_steps_captured"],
            "transformer_summary_path": (hidden_path.relative_to(output_root).as_posix() if hidden_path.exists() else None),
            "attention_summary_path": attention_path.relative_to(output_root).as_posix(),
            "vae_roundtrip_report": roundtrip_report,
            "predicted_clean_previews": clean_preview_report,
            "autopsy_report": "autopsy_report.md",
            "prompt_conditioning_path": conditioning_path.relative_to(output_root).as_posix(),
        },
        "capture_semantics": {
            "post_step_latent": "callback after scheduler.step; x_(i+1)",
            "pre_step_state": "captured inside scheduler.step before update; x_i",
            "guided_velocity": "scheduler input after conditional/unconditional classifier-free guidance combination",
            "predicted_clean": "Scheduler-converted guided x0 before update; UniPC uses its exact convert_model_output result",
            "transformer_hidden": "selected block outputs pooled over spatial tokens per latent frame; positive and negative prompt calls kept separate",
            "attention": "sampled-query self-attention summary only; does not save full attention maps or cross-attention",
        },
        "generation_seconds": generation_seconds,
        "peak_cuda_allocated_bytes": peak_memory_bytes,
        "determinism_repeat": repeat_report,
        "capture_status": hook_capture_status,
        "m0_completion_assessment": m0_completion_assessment,
    }
    autopsy_path = output_root / "autopsy_report.md"
    write_autopsy_report(
        autopsy_path,
        config=config,
        packet=packet,
        preview_report=clean_preview_report,
        roundtrip_report=roundtrip_report,
        hook_status=hook_capture_status,
    )
    packet["artifact_sha256_before_packet"] = hash_tree(
        output_root,
        exclude={"generation_packet.json", "manifest.json"},
    )
    packet_path = output_root / "generation_packet.json"
    write_json(packet_path, packet)
    manifest = {
        "manifest_schema_version": 1,
        "experiment_id": experiment_id,
        "status": "run_completed" if all(completion_gates.values()) else "run_completed_with_capture_gaps",
        "m0_completion_assessment": m0_completion_assessment,
        "model_revision": revision,
        "runtime": runtime,
        "runtime_differences_from_config": runtime_differences,
        "model_config_check": model_config_check,
        "dtype_check": dtype_check,
        "observed_transformer_inputs": baseline_input_map,
        "tensor_map": tensor_map,
        "generation_seconds": generation_seconds,
        "peak_cuda_allocated_bytes": peak_memory_bytes,
        "repeat": repeat_report,
        "capture_status": hook_capture_status,
        "predicted_clean_previews": clean_preview_report,
        "files_sha256": hash_tree(output_root, exclude={"manifest.json"}),
        "generation_packet_sha256": sha256_file(packet_path),
        "limitations": [
            "A same-seed replay checks selected latents; the full capture set is written only for run_001.",
            "Attention metrics are sampled-query self-attention statistics, not a full attention map and not cross-attention.",
            "Transformer summaries are spatially pooled channel vectors per latent frame, not token-level activations.",
        ],
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"M0 artifacts: {output_root}", flush=True)
    print(f"Generation packet: {packet_path}", flush=True)
    return 0


def controller_clean_count(config: dict[str, Any], run_dir: Path) -> int:
    return len(list((run_dir / "predicted_clean").glob("pre_step_*.pt"))) if (run_dir / "predicted_clean").exists() else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Partial run artifacts, if any, remain in the selected output directory.")
