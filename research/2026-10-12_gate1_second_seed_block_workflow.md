# Gate 1 second-seed block workflow

The first v5 Kubric mirror pair is still awaiting two independent human CSVs. Do not render or distribute a second block until the first pair passes its pooled and per-clip agreement screen, the blind worst-frame review is complete, corrections (if any) are rescored, and the known-truth diagnostic has been recorded. The second block extends annotation-reference reliability only; it is not a video-diffusion sample or policy-efficacy test.

## Prepared renderer

`research/render_gate0_kubric_fixture.py` keeps its no-argument behavior for the original fixed Gate 0 fixture. Supplying `--seed` creates a distinct candidate block at `research/kubric_gate1_seed_block_<seed>` (or at `--output-dir`). A NumPy generator maps the seed to a shared starting x, travel distance, path y, and grayscale background. The left/right paths use the same start and equal, opposite horizontal travel. Blender Cycles also receives the seed. The report records the seed and sampled scene parameters.

The candidate uses the pinned 160×96 camera and object dimensions, so the current annotation/scoring coordinate contract remains applicable. It varies the path placement and background while keeping one continuously visible red sphere. Its report records SHA-256 hashes for the ground truth and all 34 RGB frames; the packet builder checks those hashes and each PNG's dimensions before using the frames. This is a narrow repeat block in the existing fixture family, not evidence of reliability across object classes, cameras, or occlusion.

Run only after the first-block review gate passes:

```sh
python research/render_gate0_kubric_fixture.py --seed 20261012
```

The renderer refuses to overwrite an existing output. It writes no second fixture during this preparation step.

## Required first-block pass record

`research/build_gate1_seed_block_packets.py` refuses to build packets without a JSON record bound to the exact first-block files. The record has this shape; use real file paths and SHA-256 digests, and create it only after the gate has passed. Put the record in `research/` for these relative paths to resolve as shown. The review artifact must be the machine-readable JSON sidecar from the blind-review helper. The named reviewer must personally inspect every selected card while the aliases remain blind:

```json
{
  "record_version": "gate1_first_seed_block_pass_v2",
  "decision": "pass",
  "first_block_packet_zip": {
    "path": "gate1_manual_annotation_packets_v5/synthetic_reliability/rater_01.zip",
    "sha256": "0ab9ebeebc24e9badd58c53b1d9076522e926d69dc47067b6f957450a56533d8"
  },
  "rater_csvs": [
    {"path": "rater_01.csv", "sha256": "<64 lowercase hex characters>"},
    {"path": "rater_02.csv", "sha256": "<64 lowercase hex characters>"}
  ],
  "agreement_report": {
    "path": "first_block_agreement.json",
    "sha256": "<64 lowercase hex characters>"
  },
  "known_truth_report": {
    "path": "first_block_known_truth.json",
    "sha256": "<64 lowercase hex characters>"
  },
  "blind_review": {
    "artifact": {
      "path": "first_block_blind_review.json",
      "sha256": "<64 lowercase hex characters>"
    },
    "reviewer_attestation": {
      "reviewer": "<human reviewer name or stable identifier>",
      "reviewed_utc": "<ISO-8601 UTC timestamp>",
      "decision": "pass"
    },
    "human_review_complete": true,
    "corrections_rescored": true
  }
}
```

Before creating packets, the builder verifies the registered first-block packet ZIP hash, the known-truth report's first-block key/truth hashes, and all other referenced hashes. It recomputes the registered agreement screen from the two CSVs and checks that the cited agreement JSON matches those metrics. It also validates that the blind-review JSON records the same CSV hashes, the registered packet ZIP hash, a passing screen for both clip IDs, and no unblinding; the human attestation must postdate that review artifact. The known-truth report must bind the same CSVs and postdate the human review. If corrections are made, regenerate the agreement and blind-review reports on the corrected CSVs, inspect the updated review, and then run known-truth scoring. Relative paths resolve from the pass-record file.

After the pass record exists, build the second pair with:

```sh
python research/build_gate1_seed_block_packets.py \
  --seed 20261012 \
  --first-block-pass-record path/to/first_block_pass_record.json
```

Each rater ZIP contains the same two clip aliases with opposite sequence ordering. `PRIVATE_UNBLIND_KEY.json` stays separate and must not be sent to raters. The builder does not upload or distribute files. Score returned CSVs with `research/score_annotation_agreement.py`, then build and inspect the blind-review page and its JSON sidecar before unblinding:

```sh
python research/build_gate1_blind_worst_frame_review.py \
  path/to/returned/rater_01.csv \
  path/to/returned/rater_02.csv \
  --packet-zip research/gate1_seed_block_packets_20261012/synthetic_reliability/rater_01.zip \
  --output research/gate1_seed_block_packets_20261012/second_block_blind_review.html
```

After the blinded review and any corrections have been completed, run the known-truth diagnostic with the **second block's** private key and truth file:

```sh
python research/score_gate1_labels_against_kubric_truth.py \
  path/to/returned/rater_01.csv \
  path/to/returned/rater_02.csv \
  --unblind-key research/gate1_seed_block_packets_20261012/PRIVATE_UNBLIND_KEY.json \
  --ground-truth research/kubric_gate1_seed_block_20261012/ground_truth.json \
  --output research/gate1_seed_block_packets_20261012/second_block_known_truth.json
```

No second fixture, packet, model, runtime, or human-label result was created as part of preparing this workflow.
