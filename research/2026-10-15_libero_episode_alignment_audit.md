# LIBERO-Goal all-episode static alignment audit

**Checked:** 2026-10-15 06:37 UTC  
**Status:** all-episode structural and proprioception/state phase audit; RGB/action alignment remains conditional. No simulator, model, or GPU run.

## Artifact and attribution

The pinned candidate is yifengzhu-hf/LIBERO-datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5, revision 97773100c1474cd0d686ebd173cc0e4fd5442466. Its 468,246,288-byte payload was downloaded to a temporary path for this audit and its SHA-256 matched the registered value e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406. The data file is not stored in this repository.

The official LIBERO README lists its datasets under CC BY 4.0 and points to this Hugging Face distribution. The pinned Hugging Face dataset card labels its repository Apache-2.0, while upstream issue #150 asking which grant applies remains open without a maintainer clarification. This audit preserves CC BY 4.0 attribution and does not rely on the Apache label or redistribute the data. See the [LIBERO license table](https://github.com/Lifelong-Robot-Learning/LIBERO#L338-L353), [pinned dataset card](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/blob/97773100c1474cd0d686ebd173cc0e4fd5442466/README.md), and [open clarification issue](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/150).

The pinned Hugging Face commit is an upload record titled “Uploading libero_goal libero_object and libero_spatial,” dated 2025-03-15. It pins the published file's bytes but does not identify the converter revision, command, or build environment. The pinned README contains no converter details; the current README points to LIBERO's download script, which does not establish how this artifact was generated. See the [upload commit](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/commit/97773100c1474cd0d686ebd173cc0e4fd5442466), [pinned README](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/blob/97773100c1474cd0d686ebd173cc0e4fd5442466/README.md), and [current README](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/blob/main/README.md). This confirms exact artifact publication, not exporter provenance.

## HDF5 contents

The file contains 50 demonstrations and 4,669 stored rows in total, with episode lengths from 79 to 126. The inspected demo_0 has 90 rows:

- actions: 90 × 7 float64.
- states: 90 × 79 float64.
- robot_states: 90 × 9 float64.
- Agent-view and eye-in-hand RGB: 90 × 128 × 128 × 3 uint8 each.
- End-effector, gripper, and joint observation arrays: 90 rows each.
- No timestamp datasets are present.

The file records a Panda robot, OSC_POSE, control_delta=true, seven-dimensional actions, action inputs bounded at [-1, 1], output ranges of ±0.05 m for xyz and ±0.5 for the three orientation components, and a 20 Hz control frequency. It identifies the goal as “put the bowl on the plate.” Those values come from the HDF5 env_args; the earlier [action semantics audit](2026-10-12_libero_action_semantics_and_alignment_audit.md) documents the controller-source interpretation and its limits.

All episode arrays have equal row counts. In every episode, the only nonzero reward and done entries are at the final row. The pinned converter itself writes those final markers unconditionally, so they are not independent task-success labels.

The HDF5 attributes and dataset layout are consistent with the output writer in pinned LIBERO scripts/create_dataset.py; the file does not record the converter revision. The model XML is embedded, but it contains source-machine asset paths, so the file alone does not provide a portable simulator setup.

## Alignment result

The pinned [converter source](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/create_dataset.py#L170-L267) initializes playback from states[0], steps action[j], then compares the resulting simulator state with states[j+1]. It also appends the returned observation after env.step(action[j]), while storing states[j] and actions[j] under the same selected index. It skips source indices 0–4 with cap_index = 5.

An artifact-internal check compared each saved `obs/joint_states` and `obs/gripper_states` row with adjacent flattened `states` rows. Across all 50 demonstrations, all 4,619 adjacent pairs matched the next state row exactly: `joint_states[k] == states[k+1, 1:8]` and `gripper_states[k] == states[k+1, 8:10]`, with zero maximum absolute difference. Same-row comparisons had positive mean errors overall; that does not imply every individual row differs. This independently confirms a one-row offset between stored robot proprioception and stored simulator states. It does not by itself establish RGB timing or identify which converter revision produced the artifact.

If this artifact was produced by that converter:

- Same-row states[k] is the pre-action state for actions[k].
- Same-row RGB and proprioception are post-action observations.
- For stored rows k >= 1, obs[k-1] is the candidate pre-action RGB/proprioception input for actions[k], and obs[k] is that transition's post-action observation.
- The first stored action has no pre-action camera/proprioception sample. The first five source actions were also omitted.

This is a source-code-derived one-row mapping, not a replay or timestamp verification. The HDF5 does not identify its converter commit, so exact artifact lineage remains open. Do not pair same-row obs[k] with actions[k] as a pre-action policy input.

## Decision and next gate

**Decision:** partial pass for the static proprioception/state phase check; the collection/training gate remains closed because RGB timing and exporter provenance are unresolved.

Before training or collecting more data:

1. Confirm the exact exporter/version that produced this HDF5, or regenerate a traceable export.
2. Keep the CC BY 4.0 attribution and resolve the missing initial pre-action camera sample.
3. Pin the LIBERO/robosuite assets and simulator configuration, then validate the shift against a replay or explicit source timestamps.
4. Replace the converter's terminal reward/done markers with an independently defined success record.
5. Freeze episode-level splits and keep privileged simulator state out of policy inputs.

The reproducible audit is [audit_libero_episode_alignment.py](audit_libero_episode_alignment.py); its structural and proprioception/state results are in [libero_goal_episode_alignment_audit_v2.json](libero_goal_episode_alignment_audit_v2.json). The source video/action lane remains separate from P4 Gate 1 and does not advance M1–M3.
