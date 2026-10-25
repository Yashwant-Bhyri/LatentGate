# M1 smoke gate-scope amendment

**Recorded:** 2026-10-15 07:42 UTC  
**Amendment:** `LG-M1-SCOPE-2026-10-15-01`  
**Applies to:** `EXP-M1-COGVIDEOX5B-I2V-SMOKE-001` only

## Decision

The single unsteered CogVideoX-5B-I2V instrumentation smoke may proceed before the separate DiTraj/Kubric annotation Gate 1, but only after every M1-specific release guard below passes. This is a narrow sequencing correction: the M1 smoke has no policy intervention, does not score the DiTraj path-control experiment, and does not use the Kubric reference labels. Its own preregistered blinded scoreability review remains required after generation.

The DiTraj/Kubric Gate 1 remains open and continues to block DiTraj path-control and feedback experiments that depend on reliable human reference labels. The independent second mirrored-seed label block remains required before accepting that reference reliability. M1's separate failure-atlas label gate and any multi-seed M1 expansion also remain closed until the one-clip feasibility smoke has been reviewed.

## Why this amendment is needed

The M1 Colab runbook's previous resume procedure imported the DiTraj/Kubric human-label gate as a prerequisite for any M1 experiment runtime. The Oct 15 project-state summary repeated that broad stop. This conflicts with the M1 ticket and registry: they define a one-clip, unsteered feasibility smoke with its own free-T4, capture, exact-replay, and blinded sample-scoreability criteria. No threshold, dataset, prompt, model revision, runtime route, or metric is changed here; only the unrelated cross-lane dependency is removed.

## M1 release guards (all required)

1. **Authorized, no-paid compute:** use only the designated `yashwant's tech tales` Colab account. Before connecting, the UI must make clear that the selected T4 uses no paid subscription or compute units. If the billing path is ambiguous or Colab requests payment, stop. The Oct 15 T4/PyTorch preflight proves hardware availability at that time, not the billing class; it does not satisfy this guard.
2. **Current source and integrity:** refresh the Drive notebook from the current local notebook, upload the pinned bundle through the notebook workflow, and confirm its registered SHA-256 before extraction. The stale Drive copy is not runnable.
3. **Guarded preflight before weights:** use notebook cells. Before model weights load, verify Tesla T4, compute capability 7.5, and every package version in `m1/run_config.yaml`. Stop on any mismatch. Do not use Colab Terminal for project work.
4. **One baseline and exact replay only:** run the frozen seed-2026, 49-frame, 50-step baseline once, then repeat it with the same seed in the same runtime. Capture each configured post-step latent, pooled hidden summary, spatial map, predicted-clean latent, and decoded frame exactly once; require the registered coverage and bitwise-replay checks.
5. **Blinded sample review:** after generation, have a reviewer who did not inspect teacher outputs score the eight preregistered temporal samples for visibility, scene identity, and task-stage plausibility. Treat this as an n=1 scoreability screen, not a held-out evaluation or a geometry/policy result.
6. **No automatic expansion:** do not start the eight-seed feasibility batch until the baseline, replay, runtime, memory, captures, and blinded scoreability review pass and the next batch has its own reviewed release decision. The 50–100-seed atlas remains out of scope for this amendment.

## Known package-pin blocker

Historical blocker, superseded by the [versioned package-stack decision](2026-10-15_m1_package_stack_decision.md): M1 originally expected `torch==2.11.0+cu128`, while the Oct 15 T4 notebook-cell preflight recorded `torch==2.11.0+cu130`. The local config and notebook now explicitly pin the observed CUDA 13.0 Torch/TorchVision pair. The guarded preflight remains mandatory before model weights load.

**Dated supersession (2026-10-16):** This Oct 15 amendment's eight-sample blinded review detail is historical. The later [M1 readiness report](../m1/experiment_readiness_2026-10-16.md) and current ticket supersede it with a complete ordered score table for all 49 decoded frames; the original eight remain tagged as a sentinel subset.

## Scope boundary

This amendment does not pass DiTraj/Kubric Gate 1, change its annotation thresholds, make M1 evidence of path control, validate metric 3D scale or physical contact, or authorize paid compute. It supersedes only the cross-lane prerequisite in the M1 resume instructions and broad project-summary wording. Historical entries remain intact as records of the earlier gate interpretation.

## Record updates

The current M1 resume procedure, README, registry release gate, and latest state summary should point to this amendment. The underlying experiment ticket and capture thresholds remain unchanged.
