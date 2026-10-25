#!/usr/bin/env python3
"""Stress SAM 2 and the red-threshold proxy with known synthetic occlusion.

The scenes are deterministic 2D composites of the existing Kubric Gate 0
fixture. A fixed vertical pole partially occludes the sphere in one arm; a
separate same-color static distractor is added in another. The projected
object center remains the primary target truth, while the visible mask changes
under occlusion. This is an observer stress test, not human-label reliability
or video-diffusion policy evidence.

The pinned SAM 2.1 checkpoint is loaded from the local Hugging Face cache only;
the script refuses network downloads.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import Sam2VideoModel, Sam2VideoProcessor

try:
    from . import sam2_observer_probe as sam2
    from .ditraj_feedback_adapter import observe_red_frame
except ImportError:
    import sam2_observer_probe as sam2
    from ditraj_feedback_adapter import observe_red_frame


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research/kubric_gate0_fixture_v1"
OUT = ROOT / "research/sam2_visibility_stress_v1"
MODEL_ID = "facebook/sam2.1-hiera-tiny"
MODEL_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
WIDTH, HEIGHT = 160, 96
CONTROL_GRID = (52, 30)
SAMPLE_SECONDS = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class MPSMemorySampler(threading.Thread):
    """Sample MPS driver allocation; Apple unified memory has no VRAM metric."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.peak_bytes: int | None = None
        self.samples = 0
        self.api_available = callable(getattr(torch.mps, "driver_allocated_memory", None))

    def run(self) -> None:
        if not self.api_available:
            return
        while not self.stop_event.is_set():
            try:
                current = int(torch.mps.driver_allocated_memory())
                self.peak_bytes = current if self.peak_bytes is None else max(self.peak_bytes, current)
                self.samples += 1
            except Exception:
                self.api_available = False
                return
            self.stop_event.wait(SAMPLE_SECONDS)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=2)


def make_case(
    direction: str,
    disturbance: str,
    output_dir: Path,
    ground_truth: dict,
) -> tuple[list[Image.Image], list[Path], list[np.ndarray], list[list[float]], dict]:
    frame_dir = FIXTURE / direction / "frames"
    mask_paths = sorted((FIXTURE / direction / "masks").glob("mask_*.png"))
    source_frames = sorted(frame_dir.glob("frame_*.png"))
    if len(source_frames) != 17 or len(mask_paths) != 17:
        raise RuntimeError(f"Expected 17 frames and masks for {direction}")

    rows = ground_truth[direction]
    centers = [
        [row["projected_target_center_xy_normalized"][0] * WIDTH,
         row["projected_target_center_xy_normalized"][1] * HEIGHT]
        for row in rows
    ]
    full_boxes = [row["annotation_from_mask"]["bbox_xyxy_px"] for row in rows]
    case_name = f"{direction}_{disturbance}"
    case_dir = output_dir / case_name
    input_dir = case_dir / "input"
    visible_truth_dir = case_dir / "visible_truth"
    input_dir.mkdir(parents=True, exist_ok=True)
    visible_truth_dir.mkdir(parents=True, exist_ok=True)

    images: list[Image.Image] = []
    image_paths: list[Path] = []
    visible_masks: list[np.ndarray] = []
    visible_area_ratios: list[float] = []
    fixed_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    settings: dict = {"kind": disturbance}
    if disturbance == "fixed_pole_occlusion":
        pole_center_x = 100 if direction == "right" else 60
        x0, x1 = pole_center_x - 4, pole_center_x + 4
        fixed_mask[:, x0:x1] = True
        settings.update({"pole_x_interval_px_half_open": [x0, x1], "pole_rgb": [40, 165, 220]})
    elif disturbance == "static_red_distractor":
        distractor_center = [140, 20] if direction == "left" else [20, 20]
        settings.update({"circle_center_xy_px": distractor_center, "radius_px": 8, "rgb": [220, 22, 22]})
    else:
        raise ValueError(f"Unknown disturbance: {disturbance}")

    for index, (frame_path, mask_path) in enumerate(zip(source_frames, mask_paths, strict=True)):
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        original_mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        visible = original_mask.copy()
        if disturbance == "fixed_pole_occlusion":
            x0, x1 = settings["pole_x_interval_px_half_open"]
            draw.rectangle((x0, 0, x1 - 1, HEIGHT - 1), fill=tuple(settings["pole_rgb"]))
            visible = np.logical_and(visible, np.logical_not(fixed_mask))
        else:
            cx, cy = settings["circle_center_xy_px"]
            radius = settings["radius_px"]
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=tuple(settings["rgb"]))

        image_path = input_dir / f"frame_{index:03d}.png"
        visible_path = visible_truth_dir / f"mask_{index:03d}.png"
        image.save(image_path)
        Image.fromarray(visible.astype(np.uint8) * 255).save(visible_path)
        images.append(image)
        image_paths.append(image_path)
        visible_masks.append(visible)
        visible_area_ratios.append(float(visible.sum() / max(int(original_mask.sum()), 1)))

    return images, image_paths, visible_masks, full_boxes, {
        **settings,
        "direction": direction,
        "mean_visible_target_area_fraction": statistics.mean(visible_area_ratios),
        "minimum_visible_target_area_fraction": min(visible_area_ratios),
        "synthetic_input_sha256": {p.name: sha256(p) for p in image_paths},
        "visible_truth_sha256": {p.name: sha256(p) for p in sorted(visible_truth_dir.glob("mask_*.png"))},
    }


def red_proxy_metrics(frames: list[Image.Image], target_centers: list[list[float]]) -> dict:
    rows = [observe_red_frame(np.asarray(frame, dtype=np.float32) / 255.0) for frame in frames]
    scoreable: list[tuple[int, list[float], list[float]]] = []
    for index, row in enumerate(rows):
        normalized = row["center_xy_normalized"]
        if normalized is None:
            continue
        center_px = [normalized[0] * WIDTH, normalized[1] * HEIGHT]
        dx = (center_px[0] - target_centers[index][0]) * CONTROL_GRID[0] / WIDTH
        dy = (center_px[1] - target_centers[index][1]) * CONTROL_GRID[1] / HEIGHT
        scoreable.append((index, center_px, [dx, dy]))
    errors = [float(np.hypot(*row[2])) for row in scoreable]
    return {
        "scoreable_frame_fraction": len(scoreable) / len(frames),
        "scoreable_frames": len(scoreable),
        "mean_center_error_px": statistics.mean(float(np.hypot(
            row[1][0] - target_centers[row[0]][0], row[1][1] - target_centers[row[0]][1]
        )) for row in scoreable) if scoreable else None,
        "p95_center_error_control_grid_units_l2": float(np.percentile(errors, 95)) if errors else None,
        "max_abs_center_error_control_grid_units_xy": [
            max((abs(row[2][axis]) for row in scoreable), default=None) for axis in (0, 1)
        ],
        "framewise": rows,
        "interpretation": "color-threshold proxy only; a synthetic same-color distractor violates its single-target color assumption",
    }


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    ground_truth = json.loads((FIXTURE / "ground_truth.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "mps" else (torch.bfloat16 if device == "cuda" else torch.float32)
    memory = MPSMemorySampler() if device == "mps" else None
    if memory is not None:
        memory.start()

    load_start = time.perf_counter()
    model = Sam2VideoModel.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    ).to(device=device, dtype=dtype).eval()
    processor = Sam2VideoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    load_seconds = time.perf_counter() - load_start

    results = []
    for direction in ("left", "right"):
        source_masks = [np.asarray(Image.open(path).convert("L")) > 0 for path in sorted((FIXTURE / direction / "masks").glob("mask_*.png"))]
        seed = sam2.seed_from_mask(source_masks[0])
        target_centers = [
            [row["projected_target_center_xy_normalized"][0] * WIDTH,
             row["projected_target_center_xy_normalized"][1] * HEIGHT]
            for row in ground_truth[direction]
        ]
        for disturbance in ("fixed_pole_occlusion", "static_red_distractor"):
            frames, paths, visible_masks, full_boxes, disturbance_metadata = make_case(
                direction, disturbance, OUT, ground_truth
            )
            infer_start = time.perf_counter()
            predictions = sam2.run_tracker(model, processor, frames, seed, device, dtype)
            infer_seconds = time.perf_counter() - infer_start
            case = sam2.summarize_case(
                f"{direction}_{disturbance}", frames, paths, predictions, seed,
                OUT, visible_masks, full_boxes,
            )
            case["runtime_seconds"] = infer_seconds
            case["disturbance"] = disturbance_metadata
            case["projected_center_xy_px_sha256"] = hashlib.sha256(
                json.dumps(target_centers, separators=(",", ":")).encode()
            ).hexdigest()
            case["red_proxy"] = red_proxy_metrics(frames, target_centers)
            results.append(case)
            print(json.dumps({
                "case": case["name"],
                "runtime_seconds": round(infer_seconds, 2),
                "mean_visible_mask_iou": case["mean_mask_iou_to_kubric_gt"],
                "mean_full_box_center_error_px": case["mean_bbox_center_error_to_gt_px"],
                "red_proxy_mean_center_error_px": case["red_proxy"]["mean_center_error_px"],
                "visible_fraction": disturbance_metadata["mean_visible_target_area_fraction"],
            }), flush=True)

    if memory is not None:
        memory.stop()
    report = {
        "experiment_id": "EXP-SAM2-VISIBILITY-DISTRACTOR-STRESS-001",
        "status": "completed_synthetic_observer_stress_not_human_or_policy_efficacy",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0"},
        "software": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "hardware": {
            "device": device,
            "dtype": str(dtype),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "peak_vram_bytes": None,
            "peak_memory_note": (
                "Apple M3 uses unified memory; no dedicated VRAM. Peak sampled MPS driver allocation is reported below."
                if memory is not None else "No discrete VRAM metric on selected device."
            ),
            "peak_sampled_mps_driver_allocated_bytes": memory.peak_bytes if memory is not None else None,
            "mps_memory_samples": memory.samples if memory is not None else 0,
            "mps_memory_sample_period_seconds": SAMPLE_SECONDS if memory is not None else None,
        },
        "fixture": {
            "source": "research/kubric_gate0_fixture_v1",
            "rgb_and_projected_center_truth": "existing pinned Kubric fixture; target center remains the full sphere projected center",
            "visible_mask_truth": "original binary object mask minus the synthetic pole pixels",
            "directions": ["left", "right"],
            "frames_per_case": 17,
            "resolution_wh": [WIDTH, HEIGHT],
            "reference_control_grid_wh": list(CONTROL_GRID),
            "prompting": "one canonical first-frame projected-center positive point; no future-frame prompting or correction",
            "case_design": "mirrored path directions crossed with either a fixed partial occlusion or a fixed static same-color distractor",
        },
        "cases": results,
        "artifacts": {
            "runner": "research/sam2_visibility_stress_probe.py",
            "runner_sha256": sha256(Path(__file__).resolve()),
            "output_dir": "research/sam2_visibility_stress_v1",
            "fixture_ground_truth_sha256": sha256(FIXTURE / "ground_truth.json"),
        },
        "decision": (
            "Use this only as controlled observer failure analysis. Even a passing synthetic case cannot establish human-label reliability or accuracy on generated Wan frames; a failure rules out using that observer configuration as the sole measurement source."
        ),
        "limitations": [
            "Only two correlated simple synthetic paths and two hand-designed nuisance factors are used.",
            "The one-point prompt is oracle initialized from fixture truth; object discovery is not tested.",
            "SAM 2 masks are proposals; occlusion truth is synthetic compositing, not independent human labels.",
            "The red observer intentionally assumes a single red target and is not expected to resolve a same-color distractor.",
            "No generated-video inference, policy action, or efficacy test was performed.",
        ],
    }
    out_path = OUT / "report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(out_path.relative_to(ROOT)),
        "elapsed_seconds": round(report["elapsed_seconds"], 2),
        "peak_sampled_mps_driver_allocated_bytes": report["hardware"]["peak_sampled_mps_driver_allocated_bytes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
