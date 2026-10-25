#!/usr/bin/env python3
"""Inspect the pinned LIBERO HDF5 layout and emit a source-aware audit record.

This is a static file audit. It does not import LIBERO, replay an episode,
render images, or read the RGB payloads into memory. It checks whether saved
robot joint/gripper observations match the same or next flattened state row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
REVISION = "97773100c1474cd0d686ebd173cc0e4fd5442466"
FILENAME = "libero_goal/put_the_bowl_on_the_plate_demo.hdf5"
EXPECTED_SIZE = 468_246_288
EXPECTED_SHA256 = "e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406"
CONVERTER_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
HUB_UPLOAD_TITLE = "Uploading libero_goal libero_object and libero_spatial"
HUB_UPLOAD_DATE = "2025-03-15"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def json_attr(value: Any, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(native(value))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON metadata") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object")
    return parsed


def episode_number(name: str) -> int:
    try:
        return int(name.removeprefix("demo_"))
    except ValueError as error:
        raise ValueError(f"Unexpected episode name: {name}") from error


def inspect_episode(group: h5py.Group) -> dict[str, Any]:
    required = ("actions", "states", "robot_states", "rewards", "dones", "obs")
    missing = [name for name in required if name not in group]
    if missing:
        raise ValueError(f"{group.name} is missing required items: {missing}")
    actions = group["actions"]
    states = group["states"]
    obs = group["obs"]
    action_count = int(actions.shape[0])
    observation_shapes = {
        name: {
            "shape": list(dataset.shape),
            "dtype": str(dataset.dtype),
        }
        for name, dataset in obs.items()
        if isinstance(dataset, h5py.Dataset)
    }
    all_observation_lengths_match = all(
        item["shape"][0] == action_count for item in observation_shapes.values()
    )
    action_values = actions[()]
    reward_values = group["rewards"][()]
    done_values = group["dones"][()]
    terminal_marker_only = (
        action_count > 0
        and np.flatnonzero(reward_values).tolist() == [action_count - 1]
        and np.flatnonzero(done_values).tolist() == [action_count - 1]
        and int(reward_values[-1]) == 1
        and int(done_values[-1]) == 1
    )
    timestamp_datasets = []

    def find_timestamps(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset) and any(
            marker in name.lower() for marker in ("timestamp", "sim_time", "time_s")
        ):
            timestamp_datasets.append(name)

    group.visititems(find_timestamps)
    model_xml = native(group.attrs.get("model_file", ""))
    init_state = group.attrs.get("init_state", [])
    state_phase_alignment = inspect_state_phase_alignment(group)
    return {
        "episode_id": group.name.rsplit("/", 1)[-1],
        "num_samples_attribute": int(group.attrs.get("num_samples", -1)),
        "action_count": action_count,
        "states_shape": list(states.shape),
        "robot_states_shape": list(group["robot_states"].shape),
        "actions_shape": list(actions.shape),
        "actions_dtype": str(actions.dtype),
        "actions_all_finite": bool(np.isfinite(action_values).all()),
        "actions_min": float(np.min(action_values)),
        "actions_max": float(np.max(action_values)),
        "observation_streams": observation_shapes,
        "observation_lengths_match_actions": all_observation_lengths_match,
        "reward_nonzero_indices": np.flatnonzero(reward_values).tolist(),
        "done_nonzero_indices": np.flatnonzero(done_values).tolist(),
        "terminal_reward_and_done_are_last_row_only": bool(terminal_marker_only),
        "timestamp_datasets": timestamp_datasets,
        "robot_observation_state_phase_alignment": state_phase_alignment,
        "init_state_length": len(init_state),
        "model_xml_characters": len(model_xml),
    }


def inspect_state_phase_alignment(group: h5py.Group) -> dict[str, Any]:
    """Compare robot proprioception with adjacent flattened simulator states."""
    states = group["states"][()]
    observations = group["obs"]
    if "joint_states" not in observations or "gripper_states" not in observations:
        return {"available": False, "reason": "joint/gripper observations missing"}

    joints = observations["joint_states"][()]
    grippers = observations["gripper_states"][()]
    if (
        states.ndim != 2
        or states.shape[1] < 10
        or joints.ndim != 2
        or joints.shape[1] != 7
        or grippers.ndim != 2
        or grippers.shape[1] != 2
        or len(states) != len(joints)
        or len(states) != len(grippers)
        or len(states) < 2
    ):
        return {"available": False, "reason": "unexpected state/proprioception shapes"}

    # LIBERO's flattened MuJoCo state stores time, seven Panda joints, then
    # the two gripper joints. Exact matches identify these slices in this file.
    shifted_joint_error = joints[:-1] - states[1:, 1:8]
    shifted_gripper_error = grippers[:-1] - states[1:, 8:10]
    same_row_joint_error = joints[:-1] - states[:-1, 1:8]
    same_row_gripper_error = grippers[:-1] - states[:-1, 8:10]
    return {
        "available": True,
        "adjacent_pairs_compared": len(states) - 1,
        "flattened_state_joint_slice": "states[:, 1:8]",
        "flattened_state_gripper_slice": "states[:, 8:10]",
        "joint_observation_matches_next_state_row_exactly": bool(
            np.array_equal(shifted_joint_error, np.zeros_like(shifted_joint_error))
        ),
        "gripper_observation_matches_next_state_row_exactly": bool(
            np.array_equal(shifted_gripper_error, np.zeros_like(shifted_gripper_error))
        ),
        "next_state_row_joint_max_abs_error": float(np.abs(shifted_joint_error).max()),
        "next_state_row_gripper_max_abs_error": float(
            np.abs(shifted_gripper_error).max()
        ),
        "same_row_joint_mean_abs_error": float(np.abs(same_row_joint_error).mean()),
        "same_row_gripper_mean_abs_error": float(
            np.abs(same_row_gripper_error).mean()
        ),
    }


def audit(path: Path, episode_id: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"HDF5 file does not exist: {path}")
    actual_size = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_size != EXPECTED_SIZE or actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            "Artifact does not match the pinned LIBERO file "
            f"(size={actual_size}, sha256={actual_sha256})"
        )

    with h5py.File(path, "r") as source:
        if "data" not in source:
            raise ValueError("Expected a top-level data group")
        data = source["data"]
        env_args = json_attr(data.attrs.get("env_args"), "data.env_args")
        problem_info = json_attr(data.attrs.get("problem_info"), "data.problem_info")
        env_kwargs = env_args.get("env_kwargs", {})
        demo_names = sorted(
            (name for name in data if name.startswith("demo_")),
            key=episode_number,
        )
        if episode_id not in demo_names:
            raise ValueError(f"Episode {episode_id!r} is not in the HDF5 file")

        episode_lengths = [
            int(data[name].attrs.get("num_samples", data[name]["actions"].shape[0]))
            for name in demo_names
        ]
        selected = inspect_episode(data[episode_id])
        data_environment = native(data.attrs.get("env_name"))
        data_bddl_path = native(data.attrs.get("bddl_file_name"))
        data_tag = native(data.attrs.get("tag"))
        length_sum = sum(episode_lengths)
        root_total = int(data.attrs.get("total", -1))
        if root_total != length_sum:
            raise ValueError(
                f"Root total={root_total} does not equal episode sample sum={length_sum}"
            )
        all_terminal_markers_only = True
        all_episode_array_lengths_match = True
        all_joint_obs_match_next_state = True
        all_gripper_obs_match_next_state = True
        adjacent_pairs_checked = 0
        joint_next_state_max_errors: list[float] = []
        gripper_next_state_max_errors: list[float] = []
        joint_same_row_mean_errors: list[float] = []
        gripper_same_row_mean_errors: list[float] = []
        for name in demo_names:
            group = data[name]
            count = int(group["actions"].shape[0])
            if (
                int(group.attrs.get("num_samples", -1)) != count
                or int(group["states"].shape[0]) != count
                or int(group["robot_states"].shape[0]) != count
                or int(group["rewards"].shape[0]) != count
                or int(group["dones"].shape[0]) != count
                or any(
                    dataset.shape[0] != count
                    for dataset in group["obs"].values()
                    if isinstance(dataset, h5py.Dataset)
                )
            ):
                all_episode_array_lengths_match = False
            phase = inspect_state_phase_alignment(group)
            if not phase.get("available"):
                raise ValueError(
                    f"{name} cannot be checked for state/proprioception phase: "
                    f"{phase.get('reason')}"
                )
            if not phase["joint_observation_matches_next_state_row_exactly"]:
                all_joint_obs_match_next_state = False
            if not phase["gripper_observation_matches_next_state_row_exactly"]:
                all_gripper_obs_match_next_state = False
            adjacent_pairs_checked += phase["adjacent_pairs_compared"]
            joint_next_state_max_errors.append(
                phase["next_state_row_joint_max_abs_error"]
            )
            gripper_next_state_max_errors.append(
                phase["next_state_row_gripper_max_abs_error"]
            )
            joint_same_row_mean_errors.append(
                phase["same_row_joint_mean_abs_error"]
            )
            gripper_same_row_mean_errors.append(
                phase["same_row_gripper_mean_abs_error"]
            )
            rewards = group["rewards"][()]
            dones = group["dones"][()]
            if not (
                count > 0
                and np.flatnonzero(rewards).tolist() == [count - 1]
                and np.flatnonzero(dones).tolist() == [count - 1]
                and int(rewards[-1]) == 1
                and int(dones[-1]) == 1
            ):
                all_terminal_markers_only = False

        robot_state_phase_summary = {
            "episodes_checked": len(demo_names),
            "adjacent_pairs_checked": adjacent_pairs_checked,
            "all_joint_observation_rows_match_next_state_rows_exactly": (
                all_joint_obs_match_next_state
            ),
            "all_gripper_observation_rows_match_next_state_rows_exactly": (
                all_gripper_obs_match_next_state
            ),
            "max_next_state_joint_abs_error": max(joint_next_state_max_errors),
            "max_next_state_gripper_abs_error": max(gripper_next_state_max_errors),
            "same_row_joint_mean_abs_error_min": min(joint_same_row_mean_errors),
            "same_row_joint_mean_abs_error_max": max(joint_same_row_mean_errors),
            "same_row_gripper_mean_abs_error_min": min(gripper_same_row_mean_errors),
            "same_row_gripper_mean_abs_error_max": max(
                gripper_same_row_mean_errors
            ),
            "interpretation": (
                "Across every episode, saved joint/gripper observation row k is "
                "bitwise equal to the corresponding flattened state at row k+1. "
                "This validates a one-row offset for robot proprioception versus "
                "the stored state array. It does not by itself prove camera phase "
                "or identify the converter revision."
            ),
        }

    return {
        "record_version": "latentgate.libero_episode_alignment_audit/v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "repository": REPO_ID,
            "revision": REVISION,
            "filename": FILENAME,
            "hub_upload_commit": REVISION,
            "hub_upload_commit_title": HUB_UPLOAD_TITLE,
            "hub_upload_commit_date": HUB_UPLOAD_DATE,
            "hub_upload_commit_identifies_converter": False,
            "size_bytes": actual_size,
            "sha256": actual_sha256,
            "license_basis": (
                "LIBERO's official repository lists datasets under CC BY 4.0; "
                "preserve that attribution. Hugging Face repository-level "
                "Apache-2.0 metadata remains unresolved."
            ),
        },
        "dataset": {
            "environment": data_environment,
            "goal": native(problem_info.get("language_instruction")),
            "bddl_path": data_bddl_path,
            "tag": data_tag,
            "demo_count": len(demo_names),
            "sample_count_total": length_sum,
            "episode_sample_count_min": min(episode_lengths),
            "episode_sample_count_max": max(episode_lengths),
            "camera_names": env_kwargs.get("camera_names", []),
            "camera_height": env_kwargs.get("camera_heights"),
            "camera_width": env_kwargs.get("camera_widths"),
            "control_frequency_hz": env_kwargs.get("control_freq"),
            "robot_models": env_kwargs.get("robots", []),
            "controller_config": env_kwargs.get("controller_configs"),
        },
        "selected_episode": selected,
        "converter_source": {
            "repository": "Lifelong-Robot-Learning/LIBERO",
            "commit": CONVERTER_COMMIT,
            "path": "scripts/create_dataset.py",
            "artifact_records_converter_commit": False,
            "hdf5_layout_consistent_with_converter_writer": True,
            "phase_mapping": {
                "states_and_actions_same_stored_row": (
                    "Source converter selects states[j] and actions[j]; "
                    "its playback initializes from states[0] and compares the "
                    "state after action[j] against states[j+1]."
                ),
                "rgb_and_proprioception": (
                    "Source converter calls env.step(action[j]) before appending "
                    "the returned observation, so stored obs[k] is post-action."
                ),
                "pre_action_policy_input": (
                    "For stored action rows k>=1, obs[k-1] is the previous "
                    "post-action observation and is the candidate pre-action "
                    "RGB/proprioception input for action[k]."
                ),
                "missing_endpoint": (
                    "The first stored action lacks its pre-action camera/proprio "
                    "sample. The converter also skips raw indices 0 through 4 "
                    "with cap_index=5."
                ),
                "artifact_internal_state_check": (
                    "Across all 50 episodes, saved joint/gripper observation row k "
                    "matches flattened state row k+1 exactly; same-row comparisons "
                    "differ. This confirms the proprioception/state offset in the "
                    "artifact but does not by itself prove camera phase or converter "
                    "provenance."
                ),
                "confidence": (
                    "Static source-code mapping, conditional on this artifact "
                    "having been produced by the matching converter. The HDF5 "
                    "does not record the converter revision and no simulator "
                    "replay or timestamp comparison was performed."
                ),
            },
        },
        "outcome_semantics": {
            "all_episodes_have_only_final_reward_and_done_markers": (
                all_terminal_markers_only
            ),
            "interpretation": (
                "The pinned converter sets rewards[-1] and dones[-1] to 1 "
                "unconditionally; these fields are not independent task-success "
                "labels."
            ),
        },
        "all_episode_array_lengths_match": all_episode_array_lengths_match,
        "robot_observation_state_phase_alignment": robot_state_phase_summary,
        "decision": {
            "status": "partial_static_alignment_audit_complete",
            "safe_derived_transition_view": (
                "Subject to converter provenance, pair obs[k-1] with actions[k] "
                "and obs[k] for k=1..N-1; do not pair obs[k] with actions[k]."
            ),
            "collection_or_training_gate_passed": False,
            "remaining": [
                "Confirm the exact converter and source revision for this artifact.",
                "Record the CC BY 4.0 attribution; do not rely on unresolved "
                "Apache-2.0 metadata.",
                "Recover or explicitly accept the missing initial pre-action "
                "camera sample.",
                "Do not treat terminal reward/done markers as measured success.",
                "Pin the simulator/assets and validate the transition mapping "
                "before collection or policy training. The numeric proprioception "
                "check supports the one-row shift but does not clear camera timing "
                "or exporter provenance.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_path", type=Path)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = audit(args.hdf5_path, args.episode)
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
