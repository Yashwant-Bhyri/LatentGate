"""Rebuild the Kubric Gate 0 scores from saved mask images and projected truth."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from PIL import Image

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from path_annotation_scoring import evaluate_annotations


FRAME_SIZE_WH = (160, 96)
FRAME_COUNT = 17
TARGET_BOX_SIZE_PX = (20.0, 20.0)
PATH_NAMES = ("right", "left")


def target_boxes_from_centers(
    centers_xy_normalized: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
) -> list[list[float]]:
    """Build 20x20-pixel target boxes around projected object centers."""
    box_width, box_height = TARGET_BOX_SIZE_PX
    half_width = box_width / (2 * width)
    half_height = box_height / (2 * height)
    boxes = []
    for center in centers_xy_normalized:
        if len(center) != 2:
            raise ValueError(f"Projected center must have two coordinates: {center}")
        center_x, center_y = map(float, center)
        box = [
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ]
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"Projected target box is outside frame: {box}")
        boxes.append(box)
    return boxes


def _connected_component_count(mask: Image.Image) -> int:
    """Count 8-connected foreground components in a small binary mask."""
    pixels = mask.load()
    width, height = mask.size
    remaining = {
        (x, y)
        for y in range(height)
        for x in range(width)
        if pixels[x, y] > 0
    }
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return components


def annotation_from_source_mask(
    fixture_root: Path,
    truth_row: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Read visibility, component count, and box directly from a saved mask."""
    mask_path = (fixture_root / str(truth_row["mask_file"])).resolve()
    root = fixture_root.resolve()
    if not mask_path.is_relative_to(root) or not mask_path.is_file():
        raise ValueError(f"Missing or out-of-fixture mask path: {mask_path}")
    with Image.open(mask_path) as image:
        mask = image.convert("L")
    if mask.size != (width, height):
        raise ValueError(f"Mask {mask_path} has size {mask.size}, expected {(width, height)}")

    bbox = mask.getbbox()
    component_count = _connected_component_count(mask)
    pixel_count = sum(1 for value in mask.getdata() if value > 0)
    if "mask_pixel_count" in truth_row and int(truth_row["mask_pixel_count"]) != pixel_count:
        raise ValueError(f"Mask pixel count disagrees with ground truth for {mask_path}")
    if bbox is None:
        return {
            "frame_index": int(truth_row["frame_index"]),
            "object_visible": False,
            "instance_count": 0,
            "identity_consistent": False,
            "bbox_xyxy_px": None,
        }
    if component_count < 1:
        raise ValueError(f"Nonempty mask has no connected components: {mask_path}")
    return {
        "frame_index": int(truth_row["frame_index"]),
        "object_visible": True,
        "instance_count": component_count,
        "identity_consistent": component_count == 1,
        "bbox_xyxy_px": [int(value) for value in bbox],
    }


def recompute_fixture_scores(
    fixture_root: Path,
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Recompute both path scores using mask PNGs, never cached score records."""
    fixture_root = fixture_root.resolve()
    if ground_truth is None:
        import json

        ground_truth = json.loads((fixture_root / "ground_truth.json").read_text(encoding="utf-8"))

    path_scores: dict[str, Any] = {}
    width, height = FRAME_SIZE_WH
    for path_name in PATH_NAMES:
        truth_rows = list(ground_truth[path_name])
        if len(truth_rows) != FRAME_COUNT:
            raise ValueError(f"{path_name} ground truth must contain {FRAME_COUNT} frames")
        truth_rows.sort(key=lambda row: int(row["frame_index"]))
        if [int(row["frame_index"]) for row in truth_rows] != list(range(FRAME_COUNT)):
            raise ValueError(f"{path_name} ground-truth frame indices are incomplete or duplicated")

        annotations = [
            annotation_from_source_mask(fixture_root, row, width=width, height=height)
            for row in truth_rows
        ]
        targets = target_boxes_from_centers(
            [row["projected_target_center_xy_normalized"] for row in truth_rows],
            width=width,
            height=height,
        )
        score = evaluate_annotations(
            annotations,
            frame_count=FRAME_COUNT,
            width=width,
            height=height,
            target_boxes_xyxy_normalized=targets,
        )
        path_scores[path_name] = {
            "scorer_report": score,
            "target_boxes_xyxy_normalized": targets,
            "annotations_from_source_masks": annotations,
        }

    return {
        "score_source": "binary mask PNGs plus projected centers in ground_truth.json",
        "frame_size_wh": list(FRAME_SIZE_WH),
        "frames_per_path": FRAME_COUNT,
        "target_box_size_px": list(TARGET_BOX_SIZE_PX),
        "metric_scope": {
            "path_metric": "center error and signed path response",
            "extent_metric": "bbox IoU is valid for the size-matched 20x20 fixture box",
            "duplicate_policy": "hard failure; ambiguous duplicate-frame centers are unscoreable",
        },
        "path_scores": path_scores,
    }


def refresh_fixture_report(fixture_root: Path) -> dict[str, Any]:
    """Refresh the cached report from the saved ground-truth centers and masks."""
    fixture_root = fixture_root.resolve()
    report_path = fixture_root / "fixture_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_scores = recompute_fixture_scores(fixture_root)
    report.update({
        "target_box_size_px": source_scores["target_box_size_px"],
        "score_contract": "blinded_manual_boxes_v2; scores recomputed from saved binary mask PNGs and projected ground-truth centers",
        "score_source": source_scores["score_source"],
        "metric_scope": source_scores["metric_scope"],
        "path_scores": source_scores["path_scores"],
    })
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    fixture = Path(__file__).resolve().parent
    report = refresh_fixture_report(fixture)
    print(json.dumps({
        "report": str(fixture / "fixture_report.json"),
        "score_source": report["score_source"],
        "target_box_size_px": report["target_box_size_px"],
        "path_scores": {
            arm: {
                "center_ade": score["scorer_report"]["mean_center_error_frame_diagonals_missing_penalized"],
                "bbox_iou": score["scorer_report"]["mean_bbox_iou_all_frames_missing_as_zero"],
                "hard_fail": score["scorer_report"]["hard_fail"],
            }
            for arm, score in report["path_scores"].items()
        },
    }, indent=2))
