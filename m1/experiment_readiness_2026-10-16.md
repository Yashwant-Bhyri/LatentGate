# M1 experiment readiness — 2026-10-16

**Status:** local/source-only preparation. EXP-M1-COGVIDEOX5B-I2V-SMOKE-001 remains planned and unrun. No model weights, tests, GPU, Colab runtime, or notebook cells were used for this readiness pass.

## Decision: one next experiment

Run the pinned CogVideoX-5B-I2V instrumented smoke on a verified no-paid Tesla T4, then repeat the exact same generation in the same runtime. Generate the entire 49-frame sequence in both passes. The first pass is the baseline; the second is a same-runtime deterministic replay. These are matched for model revision, condition image, prompt, seed, scheduler, dimensions, frame count, inference steps, guidance, package stack, and capture hooks. The replay is a reproducibility check, not an intervention arm.

The experiment asks whether this pinned I2V route can complete on the allowed T4 path, produce the frozen mug-and-holder scene, capture the configured internal states with unambiguous phase/index labels, and reproduce its outputs exactly within one runtime. It does not test causal action control, metric 3D geometry, physical contact, or held-out predictive performance.

## Frozen run and captures

- Model: zai-org/CogVideoX-5b-I2V at revision a6f0f4858a8395e7429d82493864ce92bf73af11; Diffusers CogVideoXImageToVideoPipeline.
- Condition: fixed 720×480 start image (SHA-256 69cf5a5b1385ee0efdb525b4e164b35624c5d6d3db4d7a2da00fe2a5e27d69ea) and frozen prompt in m1/scene_fixture_v1.json.
- Generation: seed 2026, 720×480, 49 frames at 8 fps, 50 steps, guidance 6.0, pretrained CogVideoX DDIM scheduler, dynamic CFG off.
- Runtime: Tesla T4, compute capability 7.5, FP16 transformer pipeline, sequential CPU offload, VAE slicing/tiling, and the exact package versions in m1/run_config.yaml. The notebook must stop before install or weight loading unless the operator confirms the no-paid T4 path.
- Captures: post-step latents, predicted-clean x0, six evenly spaced pooled transformer blocks at steps 0, 9, 19, 29, 39, 49, and three evenly spaced 8×12 spatial summaries at steps 0, 29, 49. The configured capture-phase tags distinguish current transformer evaluation outputs, predicted-clean samples, and scheduler post-step samples.

Attention capture is intentionally deferred for this one-clip smoke and remains labeled as deferred in its packet. The broader [failure-atlas plan](failure_atlas_plan.md) calls for bounded attention diagnostics per seed. After reviewing the smoke, freeze the exact attention hooks, summarized tensor/schema, and storage/runtime budget before authorizing the eight-seed feasibility pilot. This smoke does not close the later atlas attention requirement.

## Dense decoded-frame scoring

Score every decoded RGB frame, indices 0–48, in the baseline output and save the post-run artifact at `m1/runs/cogvideox_5b_i2v_t4_smoke/frame_score_table.jsonl`. Use one ordered JSON Lines record per frame index, with baseline and replay PNG paths and SHA-256 values plus pixel equality. When the pixels are bitwise equal, the same review payload may be shared across both paths; if any frame differs, record its index and score the differing output separately. Run post-generation evaluators sequentially after unloading the video generator. After review, add the score-table SHA-256 to the final run manifest and repackage the reviewed run; leave the pre-replay baseline checkpoint baseline-only. The frame record should contain:

- Reviewer labels for mug visibility (visible, partial/occluded, absent, or uncertain), scene/object identity continuity, and task stage (initial, reach, grasp, lift, transport, lower, release, withdraw, or uncertain).
- Mug, holder, and gripper image-space masks or boxes when visible, with centroids and apparent pixel area; record missingness and its reason rather than interpolating across an unobserved frame.
- Mug-to-holder and gripper-to-mug image-space displacement/proximity, visible overlap, and apparent-contact judgment. These are visual proxies, not physical contact or calibrated geometry.
- Tracked mug points and static-background landmarks across all frames, with track survival, per-frame displacement/jitter, and coherent background-motion estimates as a camera/warp proxy.
- Frame-level evaluator confidence, reviewer uncertainty, and any occlusion or segmentation failure flags.

The original eight preselected temporal samples are a sentinel subset only; they do not replace the 49-row framewise review. Their exact legacy indices are not present in the local source records, so do not guess or derive them from latent anchors. Tag them only if the prior index list is recovered; this does not change the all-49 review gate. Summaries may report visible-frame fraction, stage order and transitions, track survival/jitter, apparent-area and object-separation trajectories, and background-motion statistics. Keep raw per-frame values and reviewer labels with the summary; do not convert these proxies into claims of 3D scale, force, or true contact.

## Outputs and pass evidence

For each pass, preserve the runtime/package preflight, elapsed time, peak allocated and reserved GPU memory, 49 RGB frames, MP4 and contact sheet, all configured tensor captures, capture call/key coverage, and SHA-256 manifests. The exact replay gate compares all selected latents, predicted-clean samples, pooled summaries, spatial maps, and decoded RGB frames bitwise within the same runtime. Record exactness as runtime-local; do not generalize it across hosts or software environments.

Save the baseline checkpoint archive and verify its hash before starting replay. The notebook can request operator confirmation that the download appears in local Downloads, but it cannot independently verify the local saved copy. The extra tensor artifacts are estimated at 144.5–157.3 MiB per pass; this is storage size, not a peak-VRAM estimate. Do not authorize a larger seed batch until runtime, peak memory, capture coverage, replay, and full-frame scoreability have been reviewed.

## Prerequisites and current blockers

1. Refresh repository-canonical Drive notebook ID 1tbDohIHBeHeKF5SD9_9Nkwf22LaJfMmk from the current local notebook, reopen that same ID, and verify the notebook source and bundle pin against local notebook SHA-256 ab9d85acda9451ae2edc3a4509064cafc27c66573ef25f7e9158459e9a1e27ef and bundle SHA-256 7f0764dd3e8730485650d1c80defc4e6024f621af03068dad18291e272daee1e. The local six-file bundle includes the current all-frame review contract; the October 15 duplicate is stale and is not the refresh target.
2. The attempted in-place refresh was blocked before transfer because Chrome's ChatGPT extension denied local-file selection while Allow access to file URLs is disabled. No permission changed and no Drive version or duplicate was created. The connected Drive API is signed in under a different profile and returned 404. The canonical copy therefore remains stale.
3. In the designated profile yashwantkumarbhyri@gmail.com, verify a fresh no-paid admission path before connecting. The latest recorded Resources view showed not subscribed, zero compute units, free resources not guaranteed, and no connected runtime. A prior idle T4 and October 15 hardware smoke do not prove current free admission.
4. Before model weights load, pass the notebook's guarded checks for bundle hash, T4 identity, exact compute capability (7,5), CUDA availability, and all package pins. Stop on any mismatch or paid-resource prompt.

No upload, permission change, runtime connection, cell execution, model download, or paid-compute action was taken in this readiness pass. The separate DiTraj/Kubric human-label Gate 1 does not gate this single M1 instrumentation smoke under the registered scope amendment; it remains a requirement for that policy lane.

## Temporal measurement and chunking

The exact CogVideoX-5B-I2V VAE path maps the 49-frame output to 13 temporal latent positions and decodes those positions back to 49 frames. Its two temporal compression stages preserve the first position and pool pairs from the rest; nominal ancestry is frame 0, then groups 1–4, 5–8, …, 45–48. Causal convolutions make these groups overlapping context summaries rather than isolated frame samples. Schedule-only enumeration finds that RGB inspection at `[0,4,8,…,48]` would miss 36/49 possible one-frame event windows, 24/48 two-frame windows, and 12/47 three-frame windows. The derivation and edge handling are in the [CogVideoX temporal-support audit](cogvideox_temporal_support_audit_2026-10-16.md).

M0's Wan VAE source audit found a maximum past reach of 112 RGB frames. That exact receptive-field result is specific to Wan and must not be transferred to CogVideoX. Both source audits support the registered requirement to score every decoded RGB frame.

Generate each CogVideoX arm as one jointly denoised full sequence. There is no measured CogVideoX memory, latency, or feedback-horizon evidence that justifies chunking. Consider chunked generation only as a later matched ablation if the completed smoke records a concrete resource or control-horizon constraint; do not change this smoke's sequence mode to make an unmeasured fit assumption.

### Sparse RGB readout from one full-sequence result — source-only follow-up, 2026-10-17

The pinned pipeline has no RGB frame-index selector. With the same full-sequence call, selecting anchor frames from the returned 49 PIL frames after the call leaves denoising unchanged, but it still performs full VAE decoding and therefore does not reduce decode compute or peak decode activation memory. `output_type="latent"` skips RGB decoding and returns the complete 13-position final latent sequence; it is not a sparse RGB mode. The VAE internally uses two-latent temporal batches, with a special three-latent first batch in the 13-latent case, carries causal cache, and concatenates every decoded chunk into 49 frames. VAE slicing is batch-only and has no effect for M1's single-video batch. Enabled VAE tiling is spatial and should lower per-tile activations while adding overlapping decode/blend work; M1 peak-memory and latency effects remain unmeasured. Full-frame review remains required. See the [sparse RGB decode audit](cogvideox_sparse_rgb_decode_audit_2026-10-17.md).

### M0 30-step clip and M1 sequence architecture synthesis — 2026-10-17

The saved CogVideoX-2B `pilot_30steps` clip has nine hash-verified frames and no scoreable red-mask track (0/9). On this clip, adjacent RGB-MAE totals are 2.75% above direct anchor endpoint contrasts, while Farneback endpoint estimates exceed its adjacent-flow sum by 2.16%; both peak at 0→1. These are clip-local whole-frame proxies and do not validate object motion, event coverage, sparse evaluation, or a denoising-step effect. They are not combined with the Wan retry5 or CogVideoX-2B `chunk13` clips.

For M1, source supports one jointly denoised 49-frame pipeline call per pass, a 49→13→49 temporal path, and no sparse RGB-index API. Post-call anchor selection does not skip VAE decode; internal VAE temporal batches are not generation chunks. M1 T4 fit, runtime, peak memory, decode latency, and any chunked-generation effect remain unmeasured. The smallest valid run remains the registered seed-2026 full-sequence baseline plus exact same-runtime replay, with all 49 frames scored, after Drive hash readback, same-account no-paid T4 admission, and guarded package/T4 preflight. Pass only on the ticket's full capture, 49-row scoreability, bitwise replay, and blinded scene-scoreability criteria. Details and source links are in the [architecture synthesis](cogvideox2b_30step_m0_and_m1_sequence_synthesis_2026-10-17.md).

## Source records

- Local configuration and runner: m1/run_config.yaml and m1/run_m1.py.
- Capture-path static audit: m1/capture_path_static_audit_2026-10-16.md.
- Full-sequence/frame-evidence static audit: m1/full_sequence_frame_audit_2026-10-16.md.
- CogVideoX temporal-support audit: m1/cogvideox_temporal_support_audit_2026-10-16.md.
- CogVideoX sparse RGB decode/API and VAE memory-path audit: m1/cogvideox_sparse_rgb_decode_audit_2026-10-17.md.
- Smoke gate: m1/geometry_scale_drift_ticket.json.
- Canonical Drive inventory and failed refresh: research/2026-10-16_m1_drive_notebook_copy_inventory.md.
- Wan VAE temporal-support derivation: research/2026-10-16_m0_wan_vae_temporal_support.md.
