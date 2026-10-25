#!/usr/bin/env python3
"""Safe CPU-only diagnostics for the saved Wan M0 latent checkpoints.

Run from any directory. Only the named checkpoint tensors are loaded, using
torch.load(map_location="cpu", weights_only=True). There is deliberately no
pickle fallback, model construction, inference, or VAE decoding.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import platform
import sys
from typing import Any

import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "research/m0_latent_slot_dynamics_v1.json"
STEPS = ["000", "004", "009", "014", "019", "024", "029"]
X0_STEPS = ["004", "014", "024", "029"]
RUN_DIRS = {
    "retry4": ROOT / "m0/runs/wan_t4_2026-10-02_retry4",
    "retry5": ROOT / "m0/runs/wan_t4_2026-10-02_retry5",
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_safe_loader() -> None:
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify torch.load safety options: {exc}") from exc
    if "weights_only" not in signature.parameters:
        raise RuntimeError("This torch.load does not expose weights_only; refusing to load any checkpoint.")


def tensor_inventory(obj: Any, prefix: str = "root") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if isinstance(obj, torch.Tensor):
        result[prefix] = {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
            "numel": obj.numel(),
        }
    elif isinstance(obj, dict):
        for key, value in obj.items():
            result.update(tensor_inventory(value, f"{prefix}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            result.update(tensor_inventory(value, f"{prefix}[{index}]"))
    return result


def expected_tensor(obj: Any, key: str, file_label: str) -> torch.Tensor:
    value = obj
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"Missing tensor key {key!r} in {file_label}")
        value = value[part]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Expected tensor at {key!r} in {file_label}, got {type(value).__name__}")
    if value.device.type != "cpu":
        raise RuntimeError(f"Expected CPU tensor at {key!r} in {file_label}, got {value.device}")
    if value.ndim != 5 or value.shape[2] != 5:
        raise RuntimeError(f"Unexpected latent shape at {key!r} in {file_label}: {list(value.shape)}")
    return value


def describe_artifact(run: str, relative: str, obj: Any, digest: str, manifest_digest: str) -> dict[str, Any]:
    return {
        "run": run,
        "path": str((RUN_DIRS[run] / relative).relative_to(ROOT)),
        "sha256": digest,
        "manifest_sha256": manifest_digest,
        "manifest_hash_matches": digest == manifest_digest,
        "top_level_keys": list(obj.keys()) if isinstance(obj, dict) else None,
        "tensors": tensor_inventory(obj),
    }


def per_slot_delta(previous: torch.Tensor, current: torch.Tensor) -> list[dict[str, Any]]:
    if previous.shape != current.shape or previous.dtype != current.dtype:
        raise RuntimeError("Checkpoint tensors differ in shape or dtype; refusing an ambiguous delta comparison.")
    rows = []
    for slot in range(previous.shape[2]):
        before = previous[0, :, slot].to(torch.float64)
        after = current[0, :, slot].to(torch.float64)
        delta = after - before
        prev_rms = torch.sqrt(torch.mean(before.square())).item()
        delta_rms = torch.sqrt(torch.mean(delta.square())).item()
        rows.append(
            {
                "latent_time_index": slot,
                "nominal_rgb_anchor": slot * 4,
                "mean_abs_delta": torch.mean(torch.abs(delta)).item(),
                "rms_delta": delta_rms,
                "max_abs_delta": torch.max(torch.abs(delta)).item(),
                "previous_rms": prev_rms,
                "relative_rms_delta": delta_rms / prev_rms if prev_rms > 0 else None,
            }
        )
    return rows


def main() -> None:
    require_safe_loader()

    manifests: dict[str, dict[str, Any]] = {}
    manifest_hashes: dict[str, str] = {}
    for run, run_dir in RUN_DIRS.items():
        manifest_path = run_dir / "manifest.json"
        manifests[run] = json.loads(manifest_path.read_text())
        manifest_hashes[run] = sha256(manifest_path)

    artifact_specs: list[tuple[str, str]] = []
    for run in ("retry4", "retry5"):
        artifact_specs.extend((run, f"run_001/latents/post_step_{step}.pt") for step in STEPS)
        artifact_specs.append((run, "run_001/initial_noise.pt"))
    artifact_specs.extend(
        ("retry5", f"run_001/predicted_clean/pre_step_{step}.pt") for step in X0_STEPS
    )

    # Check every registered file hash before any torch.load call.
    verified: list[tuple[str, str, pathlib.Path, str]] = []
    for run, relative in artifact_specs:
        file_path = RUN_DIRS[run] / relative
        manifest_digest = manifests[run]["files_sha256"].get(relative)
        if not file_path.is_file() or not manifest_digest:
            raise RuntimeError(f"Missing checkpoint or manifest hash: {run}/{relative}")
        digest = sha256(file_path)
        if digest != manifest_digest:
            raise RuntimeError(
                f"Manifest hash mismatch for {run}/{relative}: expected {manifest_digest}, got {digest}"
            )
        verified.append((run, relative, file_path, digest))

    objects: dict[tuple[str, str], Any] = {}
    artifact_rows: list[dict[str, Any]] = []
    for run, relative, file_path, digest in verified:
        # Intentional: safe weights-only unpickler, CPU map location, no fallback.
        try:
            obj = torch.load(file_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Safe torch.load failed for {run}/{relative}; stopped without pickle fallback: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        objects[(run, relative)] = obj
        artifact_rows.append(
            describe_artifact(
                run,
                relative,
                obj,
                digest,
                manifests[run]["files_sha256"][relative],
            )
        )

    retry_comparisons = []
    shared_relatives = [f"run_001/latents/post_step_{step}.pt" for step in STEPS]
    shared_relatives.append("run_001/initial_noise.pt")
    for relative in shared_relatives:
        key = "latents" if relative.endswith("initial_noise.pt") else "latent_after_scheduler_step"
        a = expected_tensor(objects[("retry4", relative)], key, f"retry4/{relative}")
        b = expected_tensor(objects[("retry5", relative)], key, f"retry5/{relative}")
        retry_comparisons.append(
            {
                "path_suffix": relative,
                "tensor_key": key,
                "torch_equal": bool(torch.equal(a, b)),
                "retry4_sha256": sha256(RUN_DIRS["retry4"] / relative),
                "retry5_sha256": sha256(RUN_DIRS["retry5"] / relative),
            }
        )

    post_changes = []
    for previous_step, current_step in zip(STEPS, STEPS[1:]):
        relative_previous = f"run_001/latents/post_step_{previous_step}.pt"
        relative_current = f"run_001/latents/post_step_{current_step}.pt"
        previous = expected_tensor(
            objects[("retry5", relative_previous)], "latent_after_scheduler_step", relative_previous
        )
        current = expected_tensor(
            objects[("retry5", relative_current)], "latent_after_scheduler_step", relative_current
        )
        post_changes.append(
            {
                "from_step": int(previous_step),
                "to_step": int(current_step),
                "tensor_key": "latent_after_scheduler_step",
                "per_temporal_slot": per_slot_delta(previous, current),
            }
        )

    x0_changes = []
    for previous_step, current_step in zip(X0_STEPS, X0_STEPS[1:]):
        relative_previous = f"run_001/predicted_clean/pre_step_{previous_step}.pt"
        relative_current = f"run_001/predicted_clean/pre_step_{current_step}.pt"
        previous = expected_tensor(objects[("retry5", relative_previous)], "x0_hat", relative_previous)
        current = expected_tensor(objects[("retry5", relative_current)], "x0_hat", relative_current)
        x0_changes.append(
            {
                "from_snapshot_step": int(previous_step),
                "to_snapshot_step": int(current_step),
                "tensor_key": "x0_hat",
                "per_temporal_slot": per_slot_delta(previous, current),
            }
        )

    support_audit = ROOT / "research/m0_wan_vae_temporal_support_audit_v1.json"
    result = {
        "audit_id": "EXP-M0-LATENT-SLOT-DYNAMICS-001",
        "status": "completed_cpu_safe_tensor_diagnostics",
        "created_date": "2026-10-16",
        "scope": "Exploratory arithmetic on existing serialized latent tensors only. No weights, inference, VAE decode, image inspection, RGB proxy scoring, GPU, or cloud runtime.",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "loader": "torch.load(map_location='cpu', weights_only=True)",
            "pickle_fallback": False,
        },
        "inputs": {
            "run_manifests": {
                run: {
                    "path": str((RUN_DIRS[run] / "manifest.json").relative_to(ROOT)),
                    "sha256": manifest_hashes[run],
                }
                for run in RUN_DIRS
            },
            "source_temporal_support_audit": {
                "path": str(support_audit.relative_to(ROOT)),
                "sha256": sha256(support_audit),
            },
        },
        "temporal_slot_mapping": [
            {
                "latent_time_index": slot,
                "nominal_rgb_anchor": slot * 4,
                "structural_causal_prefix_support_inclusive": [0, slot * 4],
            }
            for slot in range(5)
        ],
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "retry4_retry5_exact_tensor_comparison": retry_comparisons,
        "post_step_adjacent_snapshot_changes": post_changes,
        "predicted_clean_adjacent_x0_hat_changes": x0_changes,
        "metric_definitions": {
            "per_slot_population": "C×H×W elements for one latent temporal index; tensors cast to float64 only for metric arithmetic.",
            "mean_abs_delta": "mean(abs(current - previous))",
            "rms_delta": "sqrt(mean((current - previous)^2))",
            "relative_rms_delta": "rms_delta / RMS(previous) when previous RMS is nonzero",
            "max_abs_delta": "max(abs(current - previous))",
        },
        "limitations": [
            "Latent-time indices map to nominal RGB anchors [0,4,8,12,16], but the VAE source audit shows causal-prefix structural support; a slot is not an isolated RGB frame.",
            "Tensor changes are exploratory latent-space diagnostics. They are not decoded-frame evidence, object/path influence, semantics, controllability, or policy efficacy.",
            "Retry4 has no saved predicted-clean snapshots, so predicted-clean dynamics are computed only within retry5.",
            "The measurements describe this one saved run and do not establish generality across prompts, seeds, or models.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(artifact_rows)} verified artifacts.")
    print("Post-step adjacent comparisons:", len(post_changes), "Predicted-clean x0_hat comparisons:", len(x0_changes))
    print("Retry equality checks passed:", sum(r["torch_equal"] for r in retry_comparisons), "/", len(retry_comparisons))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
