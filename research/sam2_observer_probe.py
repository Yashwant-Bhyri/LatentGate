#!/usr/bin/env python3
"""Probe SAM 2.1 tiny as a mask proposer on Kubric truth and one Wan clip.

This is an observer feasibility experiment only. Kubric segmentation masks are
the synthetic ground truth; the Wan run has no independent labels, so SAM2 and
the existing red threshold are compared only as two proposals there.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import PIL
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import Sam2VideoModel, Sam2VideoProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "facebook/sam2.1-hiera-tiny"
MODEL_REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
M0_ROOT = ROOT / "m0/runs/wan_t4_2026-10-02_retry5/run_001"
KUBRIC_ROOT = ROOT / "research/kubric_gate0_fixture_v1"
OUT = ROOT / "research/sam2_observer_probe_v1"
SAMPLE_INDICES = (0, 4, 8, 12, 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frames(frame_dir: Path) -> tuple[list[Image.Image], list[Path]]:
    paths = sorted(frame_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(frame_dir)
    return [Image.open(path).convert("RGB") for path in paths], paths


def bbox_xyxy(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def center_from_mask(mask: np.ndarray, mode: str) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    if mode == "centroid":
        return [float(xs.mean()), float(ys.mean())]
    box = bbox_xyxy(mask)
    assert box is not None
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def seed_from_mask(mask: np.ndarray) -> list[float]:
    point = center_from_mask(mask, "centroid")
    if point is None:
        raise ValueError("Cannot seed a video mask from an empty mask")
    return point


def run_tracker(
    model: Sam2VideoModel,
    processor: Sam2VideoProcessor,
    frames: list[Image.Image],
    seed_xy: list[float],
    device: str,
    dtype: torch.dtype,
) -> dict[int, np.ndarray]:
    session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=0,
        obj_ids=1,
        input_points=[[[seed_xy]]],
        input_labels=[[[1]]],
    )
    output: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        first = model(inference_session=session, frame_idx=0)
        first_mask = processor.post_process_masks(
            [first.pred_masks],
            original_sizes=[[session.video_height, session.video_width]],
            binarize=True,
        )[0]
        output[0] = first_mask.detach().cpu().numpy()[0, 0].astype(bool)
        for item in model.propagate_in_video_iterator(session):
            mask = processor.post_process_masks(
                [item.pred_masks],
                original_sizes=[[session.video_height, session.video_width]],
                binarize=True,
            )[0]
            output[int(item.frame_idx)] = mask.detach().cpu().numpy()[0, 0].astype(bool)
    if set(output) != set(range(len(frames))):
        raise RuntimeError(f"SAM2 returned frame indices {sorted(output)}, expected 0..{len(frames)-1}")
    return output


def percentile95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def summarize_case(
    name: str,
    frames: list[Image.Image],
    frame_paths: list[Path],
    masks: dict[int, np.ndarray],
    seed_xy: list[float],
    out_dir: Path,
    gt_masks: list[np.ndarray] | None = None,
    gt_boxes: list[list[float] | None] | None = None,
    compare_red: bool = False,
) -> dict:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    ious: list[float] = []
    centroid_errors: list[float] = []
    box_center_errors: list[float] = []
    area_ratios: list[float] = []
    red_center_errors: list[float] = []
    red_ious: list[float] = []
    records = []
    for idx, image in enumerate(frames):
        pred = masks[idx]
        pred_box = bbox_xyxy(pred)
        pred_centroid = center_from_mask(pred, "centroid")
        pred_box_center = center_from_mask(pred, "bbox")
        rec = {
            "frame": idx,
            "pred_mask_pixels": int(pred.sum()),
            "pred_bbox_xyxy_px": pred_box,
            "pred_centroid_xy_px": pred_centroid,
            "pred_bbox_center_xy_px": pred_box_center,
        }
        if gt_masks is not None:
            gt = gt_masks[idx]
            union = int(np.logical_or(pred, gt).sum())
            iou = int(np.logical_and(pred, gt).sum()) / union if union else 1.0
            ious.append(iou)
            gt_centroid = center_from_mask(gt, "centroid")
            gt_bbox_center = center_from_mask(gt, "bbox")
            gt_box = gt_boxes[idx] if gt_boxes is not None else bbox_xyxy(gt)
            if pred_centroid and gt_centroid:
                centroid_errors.append(math.dist(pred_centroid, gt_centroid))
            if pred_box_center and gt_box:
                truth_box_center = [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]
                box_center_errors.append(math.dist(pred_box_center, truth_box_center))
            rec.update({
                "gt_mask_pixels": int(gt.sum()),
                "mask_iou": iou,
                "gt_centroid_xy_px": gt_centroid,
                "gt_bbox_xyxy_px": gt_box,
                "centroid_error_px": math.dist(pred_centroid, gt_centroid) if pred_centroid and gt_centroid else None,
                "bbox_center_error_px": math.dist(pred_box_center, [(gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2]) if pred_box_center and gt_box else None,
                "area_ratio_pred_over_gt": int(pred.sum()) / max(int(gt.sum()), 1),
            })
            area_ratios.append(int(pred.sum()) / max(int(gt.sum()), 1))
        if compare_red:
            rgb = np.asarray(image.convert("RGB"))
            red = (rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 125) & (rgb[:, :, 2] < 125)
            red_center = center_from_mask(red, "centroid")
            if red_center and pred_centroid:
                red_center_errors.append(math.dist(pred_centroid, red_center))
            union = int(np.logical_or(pred, red).sum())
            red_ious.append(int(np.logical_and(pred, red).sum()) / union if union else 1.0)
            rec.update({
                "red_threshold_pixels": int(red.sum()),
                "red_threshold_centroid_xy_px": red_center,
                "sam2_to_red_center_error_px": math.dist(pred_centroid, red_center) if pred_centroid and red_center else None,
                "sam2_red_mask_iou": red_ious[-1],
            })
        records.append(rec)
        mask_img = Image.fromarray(pred.astype(np.uint8) * 255)
        mask_img.save(case_dir / f"mask_{idx:03d}.png")

    thumb_w, thumb_h, label_h = 320, 190, 30
    sheet = Image.new("RGB", (len(SAMPLE_INDICES) * thumb_w, thumb_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for col, idx in enumerate(SAMPLE_INDICES):
        if idx >= len(frames):
            continue
        image = frames[idx].copy().convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        oa = np.asarray(masks[idx], dtype=np.uint8) * 95
        overlay_np = np.zeros((oa.shape[0], oa.shape[1], 4), dtype=np.uint8)
        overlay_np[:, :, 1] = 255
        overlay_np[:, :, 3] = oa
        mask_overlay = Image.fromarray(overlay_np)
        image = Image.alpha_composite(image, mask_overlay).convert("RGB")
        draw_img = ImageDraw.Draw(image)
        center = center_from_mask(masks[idx], "bbox")
        if center:
            x, y = center
            rad = max(2, round(image.width / 80))
            draw_img.ellipse((x-rad, y-rad, x+rad, y+rad), outline="magenta", width=max(1, rad//2))
        image.thumbnail((thumb_w, thumb_h))
        sheet.paste(image, (col * thumb_w, label_h))
        draw.text((col * thumb_w + 8, 8), f"frame {idx}", fill="black")
    sheet.save(case_dir / "contact_sheet.png")

    return {
        "name": name,
        "frame_count": len(frames),
        "seed_point_xy_px": seed_xy,
        "mask_coverage": sum(bool(masks[i].any()) for i in range(len(frames))) / len(frames),
        "mean_pred_mask_pixels": statistics.mean(int(masks[i].sum()) for i in range(len(frames))),
        "pred_mask_area_cv": statistics.pstdev(int(masks[i].sum()) for i in range(len(frames))) / statistics.mean(int(masks[i].sum()) for i in range(len(frames))),
        "mean_mask_iou_to_kubric_gt": statistics.mean(ious) if ious else None,
        "p05_mask_iou_to_kubric_gt": float(np.percentile(ious, 5)) if ious else None,
        "mean_centroid_error_to_gt_px": statistics.mean(centroid_errors) if centroid_errors else None,
        "p95_centroid_error_to_gt_px": percentile95(centroid_errors),
        "mean_bbox_center_error_to_gt_px": statistics.mean(box_center_errors) if box_center_errors else None,
        "p95_bbox_center_error_to_gt_px": percentile95(box_center_errors),
        "mean_area_ratio_pred_over_gt": statistics.mean(area_ratios) if area_ratios else None,
        "mean_sam2_to_red_center_error_px": statistics.mean(red_center_errors) if red_center_errors else None,
        "mean_sam2_red_mask_iou": statistics.mean(red_ious) if red_ious else None,
        "inputs_sha256": {path.name: sha256(path) for path in frame_paths},
        "mask_output_dir": str(case_dir.relative_to(ROOT)),
        "contact_sheet": str((case_dir / "contact_sheet.png").relative_to(ROOT)),
        "records": records,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "mps" else (torch.bfloat16 if device == "cuda" else torch.float32)
    load_start = time.perf_counter()
    model = Sam2VideoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device=device, dtype=dtype).eval()
    processor = Sam2VideoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    load_seconds = time.perf_counter() - load_start

    results = []
    kubric_gt = json.loads((KUBRIC_ROOT / "ground_truth.json").read_text())
    for direction in ("left", "right"):
        frames, paths = load_frames(KUBRIC_ROOT / direction / "frames")
        mask_paths = sorted((KUBRIC_ROOT / direction / "masks").glob("mask_*.png"))
        gt_masks = [np.asarray(Image.open(p).convert("L")) > 0 for p in mask_paths]
        gt_boxes = [row["annotation_from_mask"]["bbox_xyxy_px"] for row in kubric_gt[direction]]
        seed = seed_from_mask(gt_masks[0])
        tic = time.perf_counter()
        predictions = run_tracker(model, processor, frames, seed, device, dtype)
        seconds = time.perf_counter() - tic
        case = summarize_case(f"kubric_{direction}", frames, paths, predictions, seed, OUT, gt_masks, gt_boxes)
        case["runtime_seconds"] = seconds
        results.append(case)

    m0_frames, m0_paths = load_frames(M0_ROOT / "frames")
    first_rgb = np.asarray(m0_frames[0])
    m0_red_seed = (first_rgb[:, :, 0] > 150) & (first_rgb[:, :, 1] < 125) & (first_rgb[:, :, 2] < 125)
    seed = seed_from_mask(m0_red_seed)
    tic = time.perf_counter()
    predictions = run_tracker(model, processor, m0_frames, seed, device, dtype)
    seconds = time.perf_counter() - tic
    case = summarize_case("wan_m0_red_ball", m0_frames, m0_paths, predictions, seed, OUT, compare_red=True)
    case["runtime_seconds"] = seconds
    case["interpretation"] = "No independent labels exist for this saved Wan sample; SAM2 and red threshold are two machine proposals, not accuracy evidence."
    results.append(case)

    report = {
        "experiment_id": "EXP-SAM2-OBSERVER-TRANSFER-PROBE-001",
        "status": "completed_local_mps_observer_probe",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "license": "Apache-2.0"},
        "software": {"python_torch": torch.__version__, "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": np.__version__},
        "hardware": {"device": device, "dtype": str(dtype), "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available()},
        "model_load_seconds": load_seconds,
        "prompting": "one positive point at first-frame target-mask centroid for Kubric, and first-frame red-threshold centroid for Wan; no later-frame prompts or corrections",
        "cases": results,
        "limits": [
            "SAM2 is a measurement proposal and is not independent ground truth.",
            "The Kubric fixture is one simple synthetic red sphere on a uniform background and does not represent Wan appearance diversity.",
            "The Wan sample has no human labels; agreement with its red color mask is not observer validation.",
            "One prompt per clip does not measure sensitivity to prompt placement or occlusion.",
            "No diffusion generation or policy intervention was performed.",
        ],
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "report": str(report_path.relative_to(ROOT)),
        "model_load_seconds": round(load_seconds, 2),
        "device": device,
        "summary": [{
            "name": case["name"],
            "runtime_seconds": round(case["runtime_seconds"], 2),
            "mean_mask_iou_to_kubric_gt": case["mean_mask_iou_to_kubric_gt"],
            "mean_centroid_error_to_gt_px": case["mean_centroid_error_to_gt_px"],
            "mean_bbox_center_error_to_gt_px": case["mean_bbox_center_error_to_gt_px"],
            "mean_sam2_to_red_center_error_px": case["mean_sam2_to_red_center_error_px"],
        } for case in results],
    }, indent=2))


if __name__ == "__main__":
    main()
