# DiTraj output contract / notebook / runner crosswalk

**Date:** 2026-10-17 21:28 CST  
**Scope:** read-only comparison of the registered generated-output contract and background preregistration against the current local notebook/runner source. No notebook cells, runner, tests, fixtures, scorer, model, or cloud runtime were executed. No human-label values, private key contents, receipts, or custody data were opened.

## Source pins reviewed

- Notebook `research/ditraj_paired_path_pilot_v1.ipynb`: SHA-256 `fc066dc732ca1213487181fe38f7b2a5d54038b39aa14d16f8569c8548c56e15`, matching the registry pin.
- Runner `research/run_ditraj_paired_path_pilot.py`: SHA-256 `c00e1470c3e67da3573f61614f90e3ec9ab525135040ca3928a4bc3db4432984`, matching the current registry pin.
- Requirements cross-checked: `research/ditraj_generated_output_annotation_adapter_contract_v1.md`, `research/2026-10-17_ditraj_background_stability_protocol.md`, and `research/2026-10-17_ditraj_background_calibration_prereg_v1.md`.

## Concrete crosswalk

| Contract requirement | Current source behavior | Finding |
|---|---|---|
| Six complete 17-frame outputs at 832×480 | Runner asserts each pipeline return has shape `(17, 480, 832, 3)` and saves all 17 RGB PNGs for each arm. The paired notebook keeps the 30-step screen disabled by default; its one-step run is explicitly a smoke. | Shape/full-frame mechanics are present. A one-step smoke is not eligible for the registered packet. |
| Immutable 102-row frame map with unique `(blind_clip_id, frame_index)` and image path, SHA-256, width, height | The runner returns frame filenames/paths and proxy dimensions, but has no per-frame SHA-256/dimension manifest writer. The dimension summary is not bound to each individual PNG hash. | **Missing required artifact.** Generation shape validation does not substitute for a hash-bound manifest. |
| Blind packet contains only neutral IDs and frame images | The preliminary local template builder creates a CSV/image directory and `clip_A`… style IDs; the run report also contains arm/run metadata. The updated notebook/report prohibit archiving/export/scoring this preliminary directory and mark it `scoreable: false`, `exportable: false`. | **Not a contract-conformant packet.** Its ID scheme and local report are for smoke diagnostics; do not reuse or export this directory. A later packet builder must assign fresh opaque random IDs and package only allowed files/fields. |
| Separate background-stability measurement frozen before packet creation | The runner's `primary_evaluation.missing_requirements` names the separate background measure; the current source computes no target-excluded LK flow, ROI coverage, raw RGB/luma appearance, or calibration result. The notebook says the background metric is required before export. | **Missing required evidence.** The current fail-closed status correctly prevents scoring/export. The preregistered method still lacks numeric settings/tolerances until calibration data exist. |
| Path, visibility/count/identity, apparent size, and background kept separate | The runner's red-pixel screen is explicitly secondary. The annotation template includes visibility/count/identity and box/shape fields, while future path scoring derives center from the annotated box. | The proxy is not a substitute for labels or the background measure. Any future adapter must retain raw labels and keep these reports separate as registered. |

## Decision and next gate

No code correction is warranted: the current source documents the two missing artifacts and fails closed by keeping preliminary outputs local, non-scoreable, and non-exportable. The actual mismatch is between what the current smoke artifacts contain and what the later registered scoreable packet requires; the guard already identifies that gap. Do not treat `pilot_report.json`, its annotation template, or a one-step output as the final packet.

Before any later packet scoring, the implementation must produce and independently verify all 102 PNG hashes/dimensions and exact key-to-image rows, use fresh random blind IDs with no arm/run metadata in the packet, and attach a separate background-stability report using the calibrated/frozen method. Keep all 17 decoded frames from each jointly generated sequence; do not replace them with chunks or sparse RGB samples.

**Blockers remain:** (1) two independent complete Gate 1 v5 human returns are absent; (2) the task has not supplied the material camera-motion or acceptable illumination/codec tolerances, and there is no independent calibration dataset/result from which to freeze those thresholds. Fresh verified no-paid admission and six complete full-schedule outputs remain required before the later output packet stage. This crosswalk changes no gate and establishes no policy efficacy.

## Output-artifact schema follow-up (2026-10-17 22:17 CST)

The contract now contains an implementation-ready, specification-only schema for the eventual 102-row frame manifest and separate background-stability evidence artifact: [`ditraj_generated_output_annotation_adapter_contract_v1.md`](ditraj_generated_output_annotation_adapter_contract_v1.md), section “Implementation-ready artifact schemas.” It fixes six newly generated opaque UUIDv4 clip IDs × 17 indices, exact relative PNG paths, per-file SHA-256 and RGB8 dimensions, detached manifest/archive hashes, exact-key/path/archive checks, and fail-closed behavior. The background artifact is blind and separate, records provenance and all 16 transitions per clip, and keeps G/A/L/C distinct; threshold and classification fields stay null until the missing task tolerances and independent development/holdout calibration support a versioned freeze.

This is a contract clarification only. The notebook and runner remain unchanged; no manifest, archive, evidence artifact, labels, or thresholds were created. The two independent complete Gate 1 v5 returns remain the earliest blocker, with fresh verified no-paid admission, six complete full-schedule outputs, and background calibration also still required.

## Source-only manifest helper draft (2026-10-17 22:35 CST)

Added [`build_ditraj_output_frame_manifest.py`](build_ditraj_output_frame_manifest.py), SHA-256 `8f00716ca622ce1e57ae96a2569be3d95022a301eb5ae6409cadf46cb5fd7015`. Its build path requires the registered six mirrored arms and the pinned 30-step run report, checks model/DiTraj/runner pins, shared initial-noise hash, all 17 exact frame names, PNG RGB8/IHDR metadata, per-PNG byte hashes, and the manifest schema. If later invoked, it writes only `frame_manifest.json` and a separate owner-only `blind_run_map.json`; it does not copy images, create a ZIP, or score outputs. Its read-only validator checks a finalized ZIP against the exact 102-row manifest and detached integrity receipt and emits only duplicate-frame diagnostics, not quality scores or background classifications.

This is an unexecuted source draft: no syntax compilation, tests, run report, generated frame, manifest, key, archive, annotation, or calibration artifact was opened or created. It does not itself attest Gate 1 human-return acceptance, task tolerances, independent calibration, or no-paid admission. Keep invocation closed until all registered release prerequisites pass; the two independent complete Gate 1 v5 returns remain absent.

## Source-only contract and pin audit follow-up (2026-10-17 23:01 CST)

Static comparison of the helper against the frozen manifest/integrity schemas and current runner pins found three corrections:

- The detached integrity-receipt schema requires the exact five-key set but does not prescribe key order. Receipt validation now checks the exact set without rejecting a valid key permutation; the manifest and frame rows continue to enforce their explicitly frozen order.
- Output paths are now canonicalized through their existing parent directories before checking separation and source-tree containment. This closes symlinked-ancestor aliases that could make separate path spellings resolve into a shared output directory or source tree.
- The report's own `local_source_adaptations.runner_sha256` field and caller input are insufficient to establish the report bytes' provenance. Build mode now requires an exact report-byte SHA-256 pin from the project registry and offers no CLI override. The registry pin is deliberately `null` because no full-schedule report exists or has been independently reviewed; build therefore stays fail-closed. Record a future report digest only after independent review.

The helper SHA-256 is now `370dcf2168cfcdf52ff33aea92e0f7f464d5f295690140028a0ed7ff0da4fb19`; the frozen contract SHA-256 remains `82db035eaefbb4a1384e1e296cefc99e42d3e1240b64e124ac1ba16d3e2fd7f7`. No code was executed, compiled, or tested. No report, frames, labels, private mapping, manifest, archive, keys, or custody material was accessed or created.

**Remaining blockers:** no reviewed full-schedule report/hash pin; two independent complete Gate 1 v5 returns remain absent; fresh verified no-paid admission, six complete full-schedule outputs, task tolerances, and independent background calibration are unresolved. This source audit changes no experiment gate and establishes no policy efficacy.
