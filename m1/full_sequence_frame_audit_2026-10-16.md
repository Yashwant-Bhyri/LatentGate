# M1 full-sequence and frame-evidence audit — 2026-10-16

**Experiment:** `EXP-M1-COGVIDEOX5B-I2V-SMOKE-001`  
**Scope:** Static review of the local runner, notebook source, six-file upload bundle, current ticket, readiness report, and registry. No tests, model weights, inference, notebook cells, uploads, GPU, or Colab runtime were used.

## Findings

- `run_config.yaml` requests one 49-frame, 50-step sequence. Each `run_once()` calls the image-to-video pipeline once with `num_frames=49`; `execute()` invokes it once for the baseline and once for the exact same-seed replay on the same pipeline, image, prompt, and frozen config. The runner has no per-frame generation loop or temporal chunking path. This is source evidence only; no generated sequence exists from this audit.
- Each pass saves every returned decoded frame in sequence as `frames/frame_000.png` through `frames/frame_048.png` when 49 frames are returned. The run result lists each relative path. The baseline count must equal the requested 49; replay must match the baseline count and every decoded RGB frame is compared pixel-by-pixel. Thus a passing baseline plus replay count/equality gates implies 49 equal replay frames.
- `artifact_hashes()` recursively SHA-256 hashes files below the run directory, including both `baseline/frames/` and `replay/frames/`. The notebook refreshes the manifest after adding its log and invocation record, then archives the run. The baseline-only checkpoint also hashes the baseline images; it remains explicitly incomplete and cannot pass the smoke gate.
- The runner/notebook preserve generation artifacts but do not create the human/evaluator score table. The ticket and readiness report require that post-run review, so the contract now names `m1/runs/cogvideox_5b_i2v_t4_smoke/frame_score_table.jsonl`: one ordered record per index 0–48 with baseline/replay paths, image hashes, pixel equality, review labels, evaluator measures, and explicit uncertainty/missingness. After review, its hash must be added to the final manifest and the reviewed run repackaged. The pre-replay baseline checkpoint stays unchanged.
- The current ticket and registry require all 49 decoded frames. The original eight samples are a sentinel subset only. Their exact legacy indices do not appear in the local M1 records; no indices were inferred from latent anchors or another schedule. Tag that legacy subset only if its exact prior index list is recovered. This does not reduce or block the all-49 scoring requirement.
- The Oct. 2 preflight note still described an eight-sample-only review. Its resume step now points to the current full-frame table and marks the old instruction superseded.

## Package and source hashes

- Runner: `3ed1802b5cec198d94f7662ebca3e26531a50f8fc7691f9f2b7f01e73925ec0e`
- Ticket: `e7f09412381306749fa26e82915b24e148931494361007c3d8b052fb65f962ba`
- Bundle: `7f0764dd3e8730485650d1c80defc4e6024f621af03068dad18291e272daee1e`
- Local notebook: `ab9d85acda9451ae2edc3a4509064cafc27c66573ef25f7e9158459e9a1e27ef`

The six ZIP members byte-match the local runner, config, scene fixture, ticket, and both fixture images. The notebook pins the current bundle hash. The local Drive notebook still needs refreshing and SHA-256 readback; fresh no-paid T4 admission in the designated profile also remains unresolved. No remote copy or compute state was changed.

## 2026-10-17 follow-up: legacy sentinel index search

Searched the local M1 notes, ticket, readiness report, runner/config, notebook source, scope amendment, registry, state log, and the saved M1 bundle inventory for an explicit legacy eight-frame index list. The project contains only the current M1 bundle ZIP; no older M1 ZIP copy or M1 commit history is present locally. No exact indices or source were found. The ticket's `sentinel_indices` remains `null`; no indices were inferred. Score indices 0–48 regardless of whether the legacy sentinel mapping is later recovered.
