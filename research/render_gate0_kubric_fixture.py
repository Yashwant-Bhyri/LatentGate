#!/usr/bin/env python3
"""Render paired known-truth path clips with Kubric/Blender on CPU.

This is a research fixture builder, not the video-diffusion pilot runner. It
expects Kubric's current source and its Blender runtime to be installed in the
invoking Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import bpy
import kubric as kb
import numpy as np
from PIL import Image
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
from kubric.renderer.blender import Blender

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from kubric_gate0_fixture_v1.scoring import (  # noqa: E402
    FRAME_COUNT,
    FRAME_SIZE_WH,
    TARGET_BOX_SIZE_PX,
    recompute_fixture_scores,
)


WIDTH, HEIGHT = FRAME_SIZE_WH
FIRST_FRAME = 1
LAST_FRAME = FIRST_FRAME + 2 * FRAME_COUNT - 1
OBJECT_SEGMENTATION_ID = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_to_image_xy(camera: Any, point: np.ndarray) -> tuple[float, float]:
    """Use Blender's render camera projection, expressed top-left normalized."""
    bpy.context.view_layer.update()
    blender_camera = bpy.data.objects[camera.uid]
    coordinates = world_to_camera_view(
        bpy.context.scene, blender_camera, Vector(np.asarray(point, dtype=np.float64))
    )
    return float(coordinates.x), 1.0 - float(coordinates.y)


def make_path(
    start_x: float,
    end_x: float,
    path_y: float = 0.0,
) -> list[tuple[float, float, float]]:
    return [
        (start_x + (end_x - start_x) * index / (FRAME_COUNT - 1), path_y, 0.3)
        for index in range(FRAME_COUNT)
    ]


def annotation_from_mask(frame_idx: int, mask: np.ndarray) -> tuple[dict[str, Any], list[int] | None]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return ({
            "frame_index": frame_idx,
            "object_visible": False,
            "instance_count": 0,
            "identity_consistent": False,
            "bbox_xyxy_px": None,
        }, None)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return ({
        "frame_index": frame_idx,
        "object_visible": True,
        "instance_count": 1,
        "identity_consistent": True,
        "bbox_xyxy_px": bbox,
    }, bbox)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        help="Create a deterministic nuisance-varied mirrored block; omit to reproduce the original Gate 0 scene.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New output directory. Seeded runs default to kubric_gate1_seed_block_<seed>.",
    )
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be a non-negative integer")

    if args.output_dir is not None:
        output_root = args.output_dir.expanduser().resolve()
    elif args.seed is not None:
        output_root = ROOT / "research" / f"kubric_gate1_seed_block_{args.seed}"
    else:
        output_root = ROOT / "research" / "kubric_gate0_fixture_v1"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing fixture: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True)

    if args.seed is None:
        scene_parameters = {
            "start_x_world": 0.0,
            "travel_distance_world": 1.0,
            "path_y_world": 0.0,
            "background_gray": 0.45,
        }
    else:
        rng = np.random.default_rng(args.seed)
        scene_parameters = {
            "start_x_world": float(rng.uniform(-0.10, 0.10)),
            "travel_distance_world": float(rng.uniform(0.85, 1.15)),
            "path_y_world": float(rng.uniform(-0.18, 0.18)),
            "background_gray": float(rng.uniform(0.40, 0.50)),
        }
    scene = kb.Scene(
        frame_start=FIRST_FRAME,
        frame_end=LAST_FRAME,
        frame_rate=24,
        step_rate=240,
        resolution=(WIDTH, HEIGHT),
        background=(
            scene_parameters["background_gray"],
            scene_parameters["background_gray"],
            scene_parameters["background_gray"],
            1.0,
        ),
    )
    renderer = Blender(scene, samples_per_pixel=8, use_denoising=False)
    if args.seed is not None:
        # Keep Cycles sampling deterministic for the same pinned runtime and seed.
        bpy.context.scene.cycles.seed = args.seed % 2_147_483_647
        bpy.context.scene.cycles.use_animated_seed = False

    floor = kb.Cube(
        name="static_table",
        scale=(8.0, 8.0, 0.05),
        position=(0.0, 0.0, -0.05),
        segmentation_id=1,
    )
    ball = kb.Sphere(
        name="tracked_red_ball",
        scale=(0.3, 0.3, 0.3),
        position=(0.0, 0.0, 0.3),
        segmentation_id=OBJECT_SEGMENTATION_ID,
        material=kb.PrincipledBSDFMaterial(
            color=(0.8, 0.035, 0.02, 1.0),
            roughness=0.8,
        ),
    )
    light = kb.DirectionalLight(
        name="fixed_key_light",
        position=(-1.0, -2.0, 5.0),
        look_at=(0.0, 0.0, 0.0),
        intensity=2.0,
    )
    camera = kb.OrthographicCamera(
        name="fixed_camera",
        orthographic_scale=5.0,
        position=(0.0, 0.0, 8.0),
        look_at=(0.0, 0.0, 0.0),
    )
    scene += [floor, ball, light, camera]

    # Kubric attaches Blender observers when assets are added, but those
    # observers only propagate subsequent changes. Touch and restore each
    # initial transform/scale so the Blender scene matches the Kubric scene.
    for asset in scene.assets:
        position = np.asarray(asset.position, dtype=np.float64)
        asset.position = tuple(position + np.asarray([1e-4, 0.0, 0.0]))
        asset.position = tuple(position)
        quaternion = np.asarray(asset.quaternion, dtype=np.float64)
        changed_quaternion = quaternion.copy()
        changed_quaternion[0] += 1e-4
        changed_quaternion /= np.linalg.norm(changed_quaternion)
        asset.quaternion = tuple(changed_quaternion)
        asset.quaternion = tuple(quaternion)
        if asset.has_trait("scale"):
            scale = np.asarray(asset.scale, dtype=np.float64)
            changed_scale = scale.copy()
            changed_scale[0] += 1e-4
            asset.scale = tuple(changed_scale)
            asset.scale = tuple(scale)
        if asset.has_trait("orthographic_scale"):
            scale = asset.orthographic_scale
            asset.orthographic_scale = scale + 1e-4
            asset.orthographic_scale = scale
        if asset.has_trait("intensity"):
            intensity = asset.intensity
            asset.intensity = intensity + 1e-4
            asset.intensity = intensity
    bpy.context.view_layer.update()

    center_x = project_to_image_xy(camera, np.asarray((0.0, 0.0, 0.3)))[0]
    positive_world_x = project_to_image_xy(camera, np.asarray((1.0, 0.0, 0.3)))[0]
    world_x_to_screen_sign = 1.0 if positive_world_x > center_x else -1.0
    start_x = scene_parameters["start_x_world"]
    travel_distance = scene_parameters["travel_distance_world"]
    path_y = scene_parameters["path_y_world"]
    path_positions = {
        "right": make_path(
            start_x,
            start_x + world_x_to_screen_sign * travel_distance,
            path_y,
        ),
        "left": make_path(
            start_x,
            start_x - world_x_to_screen_sign * travel_distance,
            path_y,
        ),
    }
    frame_to_path: dict[int, tuple[str, int, tuple[float, float, float]]] = {}
    for path_name, positions in path_positions.items():
        for path_idx, position in enumerate(positions):
            frame = FIRST_FRAME + (0 if path_name == "right" else FRAME_COUNT) + path_idx
            ball.position = position
            ball.keyframe_insert("position", frame=frame)
            frame_to_path[frame] = (path_name, path_idx, position)

    # Use linear interpolation for the commanded path. The discontinuity
    # between the two independent 17-frame arms is intentionally not scored.
    for action in bpy.data.actions:
        for curve in action.fcurves:
            for key in curve.keyframe_points:
                key.interpolation = "LINEAR"

    rendered = renderer.render(
        frames=list(range(FIRST_FRAME, LAST_FRAME + 1)),
        return_layers=("rgba", "segmentation"),
    )
    (output_root / "right" / "frames").mkdir(parents=True)
    (output_root / "right" / "masks").mkdir()
    (output_root / "left" / "frames").mkdir(parents=True)
    (output_root / "left" / "masks").mkdir()

    frame_truth = {"right": [], "left": []}
    for rendered_index, frame in enumerate(range(FIRST_FRAME, LAST_FRAME + 1)):
        path_name, path_idx, position = frame_to_path[frame]
        rgb = rendered["rgba"][rendered_index]
        segmentation = rendered["segmentation"][rendered_index, ..., 0]
        mask = segmentation == OBJECT_SEGMENTATION_ID
        annotation, bbox = annotation_from_mask(path_idx, mask)
        frame_path = output_root / path_name / "frames" / f"frame_{path_idx:03d}.png"
        mask_path = output_root / path_name / "masks" / f"mask_{path_idx:03d}.png"
        Image.fromarray(rgb, mode="RGBA").save(frame_path)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)

        target_cx, target_cy = project_to_image_xy(camera, np.asarray(position, dtype=np.float64))
        frame_truth[path_name].append({
            "frame_index": path_idx,
            "kubric_frame": frame,
            "world_center_xyz": list(position),
            "projected_target_center_xy_normalized": [target_cx, target_cy],
            "mask_pixel_count": int(mask.sum()),
            "mask_bbox_xyxy_px": bbox,
            "annotation_from_mask": annotation,
            "rgb_file": f"{path_name}/frames/{frame_path.name}",
            "mask_file": f"{path_name}/masks/{mask_path.name}",
        })

    ground_truth = {"right": frame_truth["right"], "left": frame_truth["left"]}
    ground_truth_path = output_root / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps(ground_truth, indent=2) + "\n",
        encoding="utf-8",
    )
    source_scores = recompute_fixture_scores(output_root, ground_truth)
    path_reports = source_scores["path_scores"]
    for path_name, report in path_reports.items():
        score = report["scorer_report"]
        expected_sign = 1 if path_name == "right" else -1
        if score["hard_fail"]:
            raise AssertionError(f"{path_name} fixture contains duplicate object components")
        if score["scoreable_identity_coverage"] != 1.0:
            raise AssertionError(f"{path_name} fixture must keep the object visible in all frames")
        if score["target_direction_sign_x"] != expected_sign:
            raise AssertionError(f"{path_name} projected path has an incorrect direction sign")
        if score["target_aligned_endpoint_progress_frame_diagonals"] <= 0:
            raise AssertionError(f"{path_name} endpoint response must align with its target")
        if score["center_x_monotonic_step_fraction_target_direction"] != 1.0:
            raise AssertionError(f"{path_name} rendered path must be monotonic")
        if score["center_x_monotonic_valid_adjacent_pair_count"] != FRAME_COUNT - 1:
            raise AssertionError(f"{path_name} monotonicity denominator must be 16")
        if score["mean_center_error_frame_diagonals_missing_penalized"] >= 0.03:
            raise AssertionError(f"{path_name} mask centers are too far from projected object centers")
    kubric_source = Path(kb.__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "-C", str(kubric_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    renderer_metadata = {
        "fixture_id": (
            "EXP-LATENTGATE-KUBRIC-GATE0-RENDER-001"
            if args.seed is None
            else f"EXP-LATENTGATE-KUBRIC-GATE1-SEED-BLOCK-{args.seed}"
        ),
        "status": "kubric_rendered_truth_scored",
        "seed": args.seed,
        "scene_parameters": scene_parameters,
        "mirrored_pair_contract": (
            "same seeded start position and path_y; equal-magnitude opposite horizontal travel"
        ),
        "python": sys.version.split()[0],
        "kubric": kb.__version__,
        "kubric_source_commit_sha": revision,
        "blender_python": bpy.app.version_string,
        "runtime_packages": {
            name: importlib.metadata.version(name)
            for name in ("bpy", "tensorflow", "pybullet-arm64", "numpy", "pillow")
        },
        "frame_size_wh": [WIDTH, HEIGHT],
        "frames_per_path": FRAME_COUNT,
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "rgb_frame_sha256": {
            f"{direction}/frames/{frame_path.name}": sha256_file(frame_path)
            for direction in ("left", "right")
            for frame_path in sorted((output_root / direction / "frames").glob("frame_*.png"))
        },
        "target_box_size_px": list(TARGET_BOX_SIZE_PX),
        "total_rendered_frames": LAST_FRAME - FIRST_FRAME + 1,
        "fixed_camera": {
            "type": "orthographic",
            "orthographic_scale": 5.0,
            "position_xyz": [0.0, 0.0, 8.0],
            "look_at_xyz": [0.0, 0.0, 0.0],
        },
        "object": {
            "name": "tracked_red_ball",
            "shape": "Kubric ico-sphere",
            "radius_world_units": 0.3,
            "segmentation_id": OBJECT_SEGMENTATION_ID,
        },
        "world_x_to_screen_x_sign": world_x_to_screen_sign,
        "render": {
            "engine": "Kubric Blender/Cycles",
            "samples_per_pixel": 8,
            "denoising": False,
            "device": "CPU",
        },
        "score_contract": "blinded_manual_boxes_v2; scores recomputed from saved binary mask PNGs and projected ground-truth centers",
        "score_source": source_scores["score_source"],
        "metric_scope": source_scores["metric_scope"],
        "path_scores": path_reports,
        "limitations": [
            "This fixture validates camera projection, segmentation ingestion, and the box scorer on rendered known truth; it is not a generated video or a manual-label reliability study.",
            "The two path arms are deterministic render checks, not a video-diffusion control experiment.",
        ],
    }
    (output_root / "fixture_report.json").write_text(
        json.dumps(renderer_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_root),
        "status": renderer_metadata["status"],
        "versions": {"kubric": kb.__version__, "blender": bpy.app.version_string},
        "scores": {
            name: {
                "penalized_ade": report["scorer_report"]["mean_center_error_frame_diagonals_missing_penalized"],
                "signed_endpoint_dx": report["scorer_report"]["endpoint_delta_x_frame_diagonals"],
                "monotonic_pairs": report["scorer_report"]["center_x_monotonic_valid_adjacent_pair_count"],
            }
            for name, report in path_reports.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
