# Gate 1 packet QA provenance reconciliation

- Checked: 2026-10-11 12:16 UTC.
- Gate 1 status: all four prepared packets still have blank annotation fields (synthetic mirrored packets 0/34 per rater; Wan M0 convenience packets 0/17 per rater). No separate human-return CSV candidate exists under `research/`.
- The active `Research LatentGate Methods` thread still says the global report is continuing; no final report or attachment is available.
- Provenance issue found: the packet manifest's `browser_interaction_verified: false` was an as-built fact, but did not say that later smoke results were tracked separately. Clarified its scope as build-time-only in the manifest and generator, added post-build QA details from the 2026-10-09 UI smoke memo, and revised the generator's limitation wording so a future build will not imply later QA did not happen.
- Integrity: all four packet ZIP SHA-256 values still match the manifest. The synthetic smoke CSV SHA-256 matches the value in its memo and post-build QA record. Manifest JSON and generator syntax parse successfully. No packet archive was edited or rebuilt.
- Decision: UI mechanics are verified only by an agent-created synthetic smoke. This is not human usability, agreement, observer-accuracy, or policy-efficacy evidence. Gate 1 remains open until two independent raters return all 34 mirrored Kubric annotations; use `--expected-clips 2` and then add a second independently rendered mirror block for the full reliability gate.
- Resource record: local file/hash/source inspection only; no inference, GPU, cloud runtime, or paid compute.

