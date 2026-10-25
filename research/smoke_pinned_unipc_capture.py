#!/usr/bin/env python3
"""Run the adapter capture against actual Diffusers 0.33.1 UniPC on CPU tensors.

From this workspace, this can use an isolated package install without changing
the host environment:

    PYTHONPATH=/tmp/latentgate-diffusers-0.33.1 python -m research.smoke_pinned_unipc_capture
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch

try:
    from .ditraj_feedback_adapter import UniPCX0Capture
except ImportError:  # flat import after notebook staging into DiTraj's root
    from ditraj_feedback_adapter import UniPCX0Capture


def run() -> dict[str, object]:
    version = importlib.metadata.version("diffusers")
    if version != "0.33.1":
        raise RuntimeError(f"Expected Diffusers 0.33.1, found {version}")
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    torch.manual_seed(2026)
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=3.0,
    )
    if scheduler.config.thresholding:
        raise RuntimeError("Smoke expects thresholding=False so the direct conversion formula is checkable")
    scheduler.set_timesteps(8, device="cpu")
    sample = torch.randn((1, 2, 5, 4, 6), dtype=torch.float32)
    result: dict[str, object] | None = None
    for step, timestep in enumerate(scheduler.timesteps):
        model_output = torch.randn_like(sample)
        capture = None
        expected_x0 = None
        if step == 4:
            capture = UniPCX0Capture(scheduler, capture_step=step)
            capture.begin(step, timestep)
            sigma = scheduler.sigmas[scheduler.step_index].to(sample.device)
            expected_x0 = sample - sigma * model_output
        try:
            sample = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
            if capture is not None:
                packet = capture.consume(step)
                assert expected_x0 is not None
                torch.testing.assert_close(packet["x0_hat"], expected_x0, rtol=1e-6, atol=1e-6)
                result = {
                    "experiment_id": "EXP-DITRAJ-UNIPC-EXACT-CAPTURE-CPU-001",
                    "status": "passed",
                    "diffusers_version": version,
                    "device": "cpu",
                    "seed": 2026,
                    "num_inference_steps": 8,
                    "captured_step_zero_based": packet["step_zero_based"],
                    "scheduler_step_index": packet["scheduler_step_index"],
                    "captured_tensor_shape": packet["shape"],
                    "captured_tensor_dtype": packet["dtype"],
                    "max_abs_error_vs_flow_prediction_conversion": float(
                        (packet["x0_hat"] - expected_x0).abs().max().item()
                    ),
                    "scheduler_index_after_update": int(scheduler.step_index),
                    "final_sample_finite": bool(torch.isfinite(sample).all().item()),
                }
        finally:
            if capture is not None:
                capture.close()
    if result is None or result["final_sample_finite"] is not True:
        raise RuntimeError("Pinned UniPC CPU smoke did not capture a finite step-4 estimate")
    return result


def main() -> None:
    result = run()
    output = Path(__file__).with_name("ditraj_unipc_capture_smoke_v1.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), **result}, indent=2))


if __name__ == "__main__":
    main()
