# LatentGate

LatentGate studies whether internal states of general-purpose video generators can be measured and steered to improve robot-manipulation geometry, object identity, motion, camera stability, and contact. A later goal is to test whether better synthetic trajectories reduce the amount of real robot data needed at matched policy performance.

The scientific sequence is:

```text
observe → localize → intervene → ablate → compose → measure downstream value
```

## Current research position — 2026-10-17

- **Keep full-sequence generation and score every decoded RGB frame.** Latent-time anchors are secondary summaries. Literal chunked generation remains a later architecture branch unless measured memory, latency, or control-horizon evidence motivates it.
- **M0 microscope — complete.** Wan2.1-T2V-1.3B retry5 produced 17 frames with saved latents, selected hidden/attention summaries, predicted-clean previews, a VAE round-trip, and exact same-seed replay. The completion is an instrumentation milestone; the one-clip proxy analyses do not establish geometry or policy efficacy.
- **M1 failure atlas — prepared, not run.** Its smallest registered experiment is a 49-frame CogVideoX-5B-I2V baseline followed by an exact same-runtime replay, with all frames and configured captures reviewed. It is waiting for same-ID Drive notebook/bundle hash readback and a fresh, explicit no-paid T4 path under the owner-designated account, followed by the guarded T4/CC7.5 and package preflight. The separate nine-frame CogVideoX-2B analysis is not a substitute.
- **M2 geometry emergence — blocked by M1. M3 causal geometry intervention — blocked by M2.** No emergence map or causal geometry result exists yet.
- **Parallel DiTraj / Gate 1 workstream — blocked.** This is a separate path-control workstream, distinct from registry milestone M2. The latest label scan (v16) found zero complete or partial human returns; two independent complete v5 returns are required. A source crosswalk found a bypassable quota gate and gaps in the preliminary scoring packet. Local source remediation now adds a fresh, exact-arm human admission receipt before notebook or CLI execution and disables export of the preliminary packet until its frame hashes, dimensions, and background measure are complete. Static review only: the receipt is an operator attestation and cannot verify Kaggle billing itself, so no cloud run is authorized. See [the remediation memo](research/2026-10-17_ditraj_admission_export_remediation.md).
- **Saved-clip CPU queue — exhausted for the current inventory.** The latest inventory contains four unique complete RGB sequences; every unique sequence already has adjacent RGB-MAE and Farneback summaries. Wan retry4/retry5 have byte-identical final RGB sequences, while their complete artifact sets differ. No further diagnostic should recompute a covered clip.

Details and timestamps are maintained in [STATE.md](STATE.md), with the registered experiment decisions in [latentgate_experiment_registry.yaml](latentgate_experiment_registry.yaml).

## Milestone ladder

| Milestone | Objective | Status |
| --- | --- | --- |
| M0 | Instrument and replay a video generator | Complete; evidence is a microscope milestone, not policy efficacy. |
| M1 | Build a controlled failure atlas | Prepared; first CogVideoX-5B-I2V baseline/replay has not run. |
| M2 | Map geometry information over model depth and denoising time | Blocked by M1. |
| M3 | Test one causal geometry intervention | Blocked by M2. |
| M4–M6 | Build teacher→probe→gate→independent-verifier policies, including tracking/identity, then measure interference | Planned; detailed experiments and release gates are not registered yet. |
| M7 | Test whether improved videos help pseudo-action learning or downstream robot policy performance | Future research; no result exists. |

The DiTraj/Kubric Gate 1 annotation work is a parallel path-control lane. Its “M2” chat/task label is separate from LatentGate's geometry-emergence milestone above.

## Research rules

1. Treat hypotheses and proxy measurements as clip- and model-specific until independent evidence supports a broader claim.
2. Separate observability, prediction, causal correction, and downstream utility.
3. Give each policy an explicit teacher, measurable target, latent readout, intervention, independent verifier, control, and falsifier.
4. Evaluate each policy alone and record collateral effects before composition.
5. Use fresh no-paid compute admission, a reviewed hash-matched notebook, and exact device/package preflight before any remote inference. Do not use paid compute or rotate accounts to work around provider limits.
6. Keep private annotation returns in the approved owner-only flow; do not put them in the project tree.

## Start here

1. [Research operating system](LATENTGATE_RESEARCH_OS.md) — thesis, failure families, policy contract, and staged research design.
2. [Multi-agent bootstrap](CODEX_MULTI_AGENT_BOOTSTRAP.md) — roles, evidence standards, and serialized shared-state writes.
3. [Current state](STATE.md) — chronological decisions and evidence; start at the latest dated entry.
4. [Experiment registry](latentgate_experiment_registry.yaml) — hypotheses, milestone gates, tickets, artifact hashes, and current blockers.
5. [M1 readiness report](m1/experiment_readiness_2026-10-16.md) and [registered smoke ticket](m1/geometry_scale_drift_ticket.json) — current full-sequence run contract.
6. [Original context and two-week roadmap](robotics_video_diffusion_context_and_plan.md) — the complete source plan from the shared ChatGPT discussion.
7. [README history](README_HISTORY.md) — the previous long-form landing page, retained for its detailed chronological notes and links.

## Current M0 and M1 evidence

- [M0 30-step saved-clip diagnostic](research/2026-10-17_cogvideox2b_pilot_30steps_adjacent_motion.md) and [machine-readable result](research/cogvideox2b_pilot_30steps_adjacent_motion_v1.json): nine hash-verified CogVideoX-2B T2V frames, 480×320, seed 2026, 30 steps. Its fixed red-mask gate scored 0/9. Whole-frame RGB-MAE and Farneback summaries describe this clip only and do not establish an object track, event recall, sparse-sampling sufficiency, or a denoising-step effect.
- [M1 sequence synthesis](m1/cogvideox2b_30step_m0_and_m1_sequence_synthesis_2026-10-17.md): separates that clip-local result from the pinned CogVideoX-5B-I2V source audit and states the smallest valid M1 run and pass conditions.
- [M1 smoke ticket](m1/geometry_scale_drift_ticket.json): defines the all-49-frame JSONL review artifact. The legacy eight sentinel indices are not preserved and must not be guessed; attention capture is deferred from the smoke and must be resolved before the later atlas batch.
- [M1 sparse RGB decode source audit](m1/cogvideox_sparse_rgb_decode_audit_2026-10-17.md): the pipeline decodes the full 49-frame sequence from 13 temporal latents; selecting RGB anchors afterward does not avoid VAE decode.

## Repository map

| Path | Contents |
| --- | --- |
| `m0/` | Wan microscope runner, notebook, captures, manifests, and saved-run reports |
| `m1/` | Mug/holder failure-atlas plan, CogVideoX-5B-I2V smoke, readiness gates, and source audits |
| `research/` | DiTraj/Gate 1 artifacts, saved-clip analyses, source audits, protocols, and reports |
| `STATE.md` | Chronological evidence and project handoffs |
| `latentgate_experiment_registry.yaml` | Machine-readable hypotheses, milestones, experiment states, and hashes |
| `LATENTGATE_RESEARCH_OS.md` | Full scientific design and policy bank |
| `CODEX_MULTI_AGENT_BOOTSTRAP.md` | Orchestration and contributor instructions |

## Workspace notes

This local directory is on branch `main`, has no configured Git remote, and currently has no tracked files. The project has not been committed or published from this workspace.
