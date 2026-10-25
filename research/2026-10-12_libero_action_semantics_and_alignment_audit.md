# LIBERO action semantics and observation alignment audit

**Checked:** source audit on 2026-10-12; artifact review updated 2026-10-15  
**Status:** source audit plus hash-verified structural inspection of all 50 HDF5 episodes and static phase comparison; no package, simulator, model, or GPU was run. See the [2026-10-15 episode audit](2026-10-15_libero_episode_alignment_audit.md).

## Findings from the pinned software path

The LIBERO commit used for this audit is `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Its `ControlEnv` wrapper defaults to a Panda arm, `OSC_POSE`, the robot's default gripper, and `control_freq=20`; it asks robosuite to load its default `OSC_POSE` controller configuration. The pinned LIBERO requirements specify `robosuite==1.4.0`.

The matching robosuite v1.4.0 default `osc_pose.json` defines six arm-control dimensions with input range `[-1, 1]`, output ranges of ±0.05 m for xyz and ±0.5 for the three orientation components, `control_delta=true`, fixed impedance, and uncoupled position/orientation torque control. The `OSC_POSE` controller code documents absolute inputs as world-frame commands; in this default configuration inputs are deltas. The default Panda gripper has one action dimension: -1 opens and +1 closes. `SingleArm.control` splits the vector after the controller's six dimensions and sends the remainder to the gripper. This yields a provisional seven-dimensional action layout for the default Panda + gripper path.

The wrapper's 20 Hz control frequency implies a 0.05 s policy interval when that default is used. The exact demonstration file's controller override, action preprocessing, action vector, and episode-level sampling must still be checked. These source defaults do not by themselves prove that a particular published dataset artifact was collected with them.

## Exporter alignment finding

In the same pinned LIBERO source, `scripts/create_dataset.py` calls `env.step(action)` before appending returned RGB and proprioceptive observations. It selects `states` and `actions` separately with `valid_index`, then asserts equal lengths between actions and agent-view RGB. Equal row counts do not imply equal phase: selected `states[k]` are pre-action, while the returned RGB and robot observations in the corresponding exported row are post-action. The loop appends the final returned post-action observation. `cap_index = 5` skips source action indices 0–4, so the first retained action has no same-exporter pre-action RGB. A derived training view may link a previous post-action image as the next pre-action image only after timeline continuity is verified; it must not relabel the same-row image as pre-action.

This source audit is scoped to that converter and commit. The exact task artifact was subsequently downloaded to temporary storage and SHA-256 verified. Its HDF5 attributes and layout are consistent with the converter writer; the file does not record the converter revision. Across all 50 episodes, the saved joint and gripper observations exactly match the next flattened state row for all 4,619 adjacent pairs, confirming the one-row proprioception/state offset inside the artifact. The source writer collects RGB from that same post-step observation, but the numeric state check alone does not verify image timing or converter provenance. Therefore the RGB shift remains conditional on the converter match, and the first stored action lacks a pre-action camera sample. See the [detailed episode audit](2026-10-15_libero_episode_alignment_audit.md). The [v2 episode contract](native_video_action_episode_schema_v2.md) preserves raw indices/phases and keeps collection/training gated until image timing and provenance are verified.

## License and data boundary

The LIBERO README lists MIT for code and CC BY 4.0 for datasets, and explicitly directs dataset users to this Hugging Face repository. The selected file is therefore traceable to LIBERO's documented distribution route. The Hugging Face repository labels itself Apache-2.0, but that is repository-level metadata, not a separate license declaration for this HDF5. An open issue in the original LIBERO repo asks which dataset license governs and whether the Apache tag was authorized; it remains unanswered. For this static inspection, conservatively preserve CC BY 4.0 attribution and do not rely on the Apache tag to waive it. Resolve the Apache discrepancy before relying on additional Apache-specific permissions or redistributing under Apache terms. The inspected copy was stored temporarily outside the repository; the data is not redistributed.

## Remaining gates

1. Record the CC BY 4.0 attribution from LIBERO's official distribution route for any use; resolve whether the Hugging Face Apache-2.0 tag is an authorized additional grant before relying on it or redistributing under it.
2. Confirm the exact converter/version for this artifact, or regenerate a traceable export. The HDF5 records a seven-dimensional Panda OSC_POSE action configuration at 20 Hz, but not the converter commit or timestamps.
3. Validate the conditional one-row phase mapping with explicit source provenance or replay, and recover or explicitly accept the missing initial pre-action camera sample.
4. Replace terminal reward/done markers with independently verified success labels; then freeze episode-level splits and keep privileged state/contact/success fields out of policy inputs.

Until these gates pass, treat LIBERO as a candidate data path, not a collected training set. This lane does not alter P4 Gate 1, M1–M3, or establish policy efficacy.

## Primary sources

- [Pinned LIBERO environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/envs/env_wrapper.py)
- [Pinned LIBERO requirements](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/requirements.txt)
- [Pinned LIBERO dataset converter](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/create_dataset.py)
- [robosuite v1.4.0 OSC controller config](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/config/osc_pose.json)
- [robosuite v1.4.0 OSC controller source](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/controllers/osc.py)
- [robosuite v1.4.0 Panda gripper](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/models/grippers/panda_gripper.py)
- [robosuite v1.4.0 single-arm action splitting](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/robosuite/robots/single_arm.py)
- [LIBERO repository license table](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO datasets page](https://libero-project.github.io/datasets)
- [Pinned Hugging Face artifact metadata and LFS SHA-256](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/blob/97773100c1474cd0d686ebd173cc0e4fd5442466/libero_goal/put_the_bowl_on_the_plate_demo.hdf5)
- [Hugging Face dataset repository license metadata](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)
- [Pinned Hugging Face dataset card](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/blob/97773100c1474cd0d686ebd173cc0e4fd5442466/README.md)
- [Open LIBERO issue asking about the dataset-license discrepancy](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/150)
