# Native video/action episode contract v2

**Status:** v2 contract; one hash-pinned LIBERO-Goal artifact has received a structural audit across all 50 episodes and a static proprioception/state phase comparison. No simulator was installed and no episode was collected or replayed. Version 2 corrects the source-specific phase ambiguity identified in the v1 draft.

## Purpose and scope

This contract supports a future image-conditioned video/action experiment. It stores source observations, applied actions, outcomes, and simulator truth while keeping privileged fields separate from model inputs. It is independent of P4's human-label gate and does not satisfy or unblock M1–M3.

The provisional first task remains **LIBERO-Goal / `put_the_bowl_on_the_plate`**, a one-object placement task candidate, not a final benchmark. The pinned `yifengzhu-hf/LIBERO-datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5` artifact is at repository commit `97773100c1474cd0d686ebd173cc0e4fd5442466`, 468,246,288 bytes, SHA-256 `e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406`. A temporary copy was hash-verified and used for the [static all-episode audit](2026-10-15_libero_episode_alignment_audit.md); the dataset is not stored in this repository. LIBERO's README lists datasets as CC BY 4.0 and points users to this Hugging Face repository; Hugging Face's Apache-2.0 tag is repository-level and the upstream license issue remains unanswered. Conservatively preserve CC BY attribution and do not treat the Apache tag as permission to waive it. Pin the conversion path, environment, applicable license, and attribution before collecting data.

## Why v2 replaces v1

The v1 draft treated each `observation[t]` as pre-action without requiring that a source exporter preserve that phase. That assumption is unsafe. In LIBERO's `scripts/create_dataset.py` at commit [`8f1084e3132a39270c3a13ebe37270a43ece2a01`](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/create_dataset.py), the conversion loop calls `env.step(action)` and then appends the returned RGB and robot observations, while `states` and `actions` are selected separately through `valid_index`. The stored arrays can have equal lengths while referring to different phases: selected `states[k]` are pre-action, while the same-row returned RGB/proprioception are post-action. The converter also processes and appends the final returned post-action observation. Its `cap_index = 5` skips source indices 0–4. This finding is specific to that converter and commit; it does not establish the layout of every LIBERO artifact. The HDF5's `env_args` exposes its stored controller parameters; the file's producing converter and full environment/asset provenance remain unverified.

Therefore, raw streams must retain their own source indices and verified phase. A same-row join is never evidence of temporal alignment. Across all 50 demos, robot joint/gripper observation row k matches flattened state row k+1 exactly for 4,619 adjacent pairs. This verifies the one-row shift for robot proprioception versus the state array, but it does not independently verify RGB timing. The HDF5 layout is consistent with the pinned converter output, but the artifact does not record its converter revision; camera-phase mapping remains conditional on that source. No simulator replay or timestamp comparison was performed. The [audit report](2026-10-15_libero_episode_alignment_audit.md) records the evidence and the missing initial pre-action camera sample.

## Core temporal model

An episode has independent, ordered sensor streams and an ordered action stream. Do not assume all streams have equal lengths or that row `k` across files describes one instant. Each observation record carries a simulation timestamp when available, source index, and phase (`pre_action`, `post_action`, or `unknown`). Each action records its source index and references the pre- and post-observation records it actually connects, modality by modality.

For a complete transition, `action[t]` maps verified policy inputs at time `t` to verified post-step observations at time `t+1`. A canonical episode may have `T` actions and `T+1` observation events, but only when the source truly contains those endpoints. Missing endpoints remain null with a reason. Do not synthesize, shift, or infer a pre-action image from an array row. Any reconstructed or resampled view is derived data with explicit provenance and a separately verified mapping.

## Versioned layout

```text
native_video_action_v2/
  manifest.json
  splits.json
  episodes/
    <episode_id>/
      episode.json
      streams/<stream_name>/<source_index>.<ext>
      privileged_state/<source_index>.npz
```

Source files are immutable and referenced by relative path and SHA-256. Resized, re-encoded, temporally sampled, normalized, or phase-shifted training views are derived artifacts; record the transform, source references, and validation result separately.

## Dataset manifest

`manifest.json` records:

| Field | Required contents |
|---|---|
| `schema_version` | `latentgate.native_video_action_episode/v2` |
| `experiment_id` | `EXP-LATENTGATE-NATIVE-VIDEO-ACTION-DATASET-001` |
| `task` | Suite, exact task, language goal, BDDL/PDDL path and hash |
| `simulator` | Name/version, source commit, physics timestep, renderer/backend |
| `source_data` | Exact demonstration artifact/version/URL, applicable license, attribution, hashes, and conversion code/commit; mark generated rollouts as such |
| `robot` | Robot model, controller name/config, gripper type |
| `observation_streams` | Stream names/modalities, camera poses/calibration or proprioception field definitions, native dimensions/encoding, source clock, and expected phase if verified |
| `action` | Raw vector dimension and order, names, units, frame, bounds, scaling, controller semantics, control frequency/hold duration, and preprocessing |
| `alignment_audit` | Reviewed episode IDs, source indices, phase evidence, action-to-observation references, dropped/capped indices, missing endpoints, and reviewer/source revision |
| `goal_and_success` | Language goal, structured predicate, success test, and termination rule |
| `randomization` | Seed policy and randomized object, pose, lighting, texture, and layout fields |
| `split_manifest` | Path/hash to frozen episode assignments; no episode/window leakage |
| `provenance` | Collector/version, timestamp, source revision, config hash, and human curation |

Unverified values remain `null` with a reason. Do not silently fill them from a default or another wrapper.

## Episode record

Each modality has its own observation stream. Transitions reference the records that precede and follow an action rather than relying on positional array joins.

```json
{
  "schema_version": "latentgate.native_video_action_episode/v2",
  "episode_id": "<stable-id>",
  "experiment_id": "EXP-LATENTGATE-NATIVE-VIDEO-ACTION-DATASET-001",
  "task_id": "libero_goal/put_the_bowl_on_the_plate",
  "split": "train | validation | test_seen | test_unseen_object | test_unseen_layout",
  "seed": 0,
  "object_instance_ids": [],
  "layout_id": "<stable-layout-id>",
  "goal": {"language": "<task instruction>", "predicate": "<structured goal>"},
  "observation_streams": {
    "agentview_rgb": [
      {
        "source_index": 5,
        "sim_time_s": null,
        "phase": "post_action",
        "path": "streams/agentview_rgb/000005.png",
        "sha256": "<sha256>",
        "valid": true
      }
    ],
    "proprioception": [],
    "robot_state": []
  },
  "transitions": [
    {
      "step": 0,
      "source_action_index": 5,
      "applied_at_sim_time_s": null,
      "pre_observations": {
        "agentview_rgb": null,
        "proprioception": null,
        "robot_state": {"stream": "robot_state", "source_index": 5}
      },
      "action": {
        "raw": [],
        "semantics_ref": "manifest.action",
        "duration_s": null
      },
      "post_observations": {
        "agentview_rgb": {"stream": "agentview_rgb", "source_index": 5},
        "proprioception": null,
        "robot_state": null
      },
      "privileged_truth": {
        "pre": null,
        "post": {"path": "privileged_state/000005.npz", "sha256": "<sha256>"}
      },
      "outcome": {
        "reward": null,
        "success": null,
        "terminated": false,
        "truncated": false,
        "termination_reason": null
      },
      "alignment": {
        "status": "incomplete_missing_pre_action_rgb",
        "evidence": "<source audit reference>",
        "notes": "Do not pair this action with the same-row post-action RGB as its input."
      },
      "validity": {"action": true, "simulator_step": true}
    }
  ]
}
```

`pre_observations` and `post_observations` are modality-specific references. A null reference must have an explicit missingness/alignment reason. Keep policy-visible RGB/proprioception distinct from privileged object poses, contact/support labels, and evaluator-only success state. Missing or invalid values are never encoded as zeros without a validity flag.

## Source-alignment gate before collection or training

1. Pin the exact simulator commit, task/BDDL, converter, demonstration artifact, and file hashes. Record terms and attribution for the exact artifact.
2. Inspect the controller and action decoder. Establish action dimension/order, units, coordinate frame, scaling, delta-vs-absolute semantics, gripper convention, control rate, and hold duration from the pinned code/configuration.
3. Trace one complete episode through reset, each action application, each sensor capture, termination, and export. Match source indices and timestamps for RGB, proprioception, robot state, and privileged truth. Record all dropped/capped indices and endpoint coverage.
4. Verify the derived pre/post transition links against the source timeline. An equal number of rows or a successful file parse does not pass this gate.
5. If required policy-input observations are absent, collect a source that records them or label the episode incomplete. Do not start model training with an assumed shift.

## Split and comparison rules

- Assign complete episodes before extracting frames or windows. No episode may cross splits.
- Freeze the split manifest and hash before model training. Keep repeated paired-seed evaluation blocks distinct.
- Provide unseen-object/layout tests only when the simulator configuration actually holds out those instances/layouts; the standard LIBERO task list does not establish such splits.
- Match initial conditions, goals, demonstrations, updates, video samples, action horizon, and inference compute across model arms. Pair arms on the same evaluation episodes.
- Preserve exact replay/no-op, correct-pair, shuffled-pair, and sign-reversed-action controls where the action transform makes reversal meaningful.

## Required per-episode outputs

Retain task success and termination reason; target pose/error; action validity; object and end-effector state; contact/support/collision events; forecast error/calibration where a failure probe exists; video path, visibility, identity, apparent size, and camera/background stability; recovery/completion outcome; and all run/config/source/data hashes. Report paired-seed uncertainty and per-episode metrics, not only aggregate success.

This contract creates a measurable data foundation. It does not establish latent observability, forecastability, causal correctability, action specificity, native video/action improvement, or synthetic-data substitution.

## Primary sources reviewed

- [LIBERO datasets](https://libero-project.github.io/datasets)
- [LIBERO repository and component licenses](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Pinned LIBERO `create_dataset.py` converter](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/create_dataset.py)
- [Pinned LIBERO task map](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/benchmark/libero_suite_task_map.py)
- [LIBERO collection script](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/libero_100_collect_demonstrations.py)
