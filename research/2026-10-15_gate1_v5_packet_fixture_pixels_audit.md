# Gate 1 v5 packet-to-fixture pixel audit

**Checked:** 2026-10-15 07:55 UTC  
**Component:** Gate 1 — validate the exact visual samples delivered in both blind rater packets.  
**Decision:** both packets contain the canonical 34 Kubric fixture frames byte-for-byte; human-label reliability remains unmeasured.

## Evidence and procedure

The pinned Kubric fixture is documented in the [Kubric paper](https://openaccess.thecvf.com/content/CVPR2022/html/Greff_Kubric_A_Scalable_Dataset_Generator_CVPR_2022_paper.html) and [official repository](https://github.com/google-research/kubric). This audit makes no new claim about Kubric beyond checking local artifact provenance.

[`audit_gate1_v5_packet_fixture_pixels.py`](audit_gate1_v5_packet_fixture_pixels.py) reads the two archived v5 `index.html` files as data, extracts their embedded PNG data URIs, and compares the unordered SHA-256 multiset against the 34 canonical RGB PNGs. It checks two blind IDs × 17 frame indices, 160×96 metadata, the 34-row blank annotation CSVs, expected ZIP members, and CRC integrity. It also requires each template's `(blind_clip_id, frame_index, image_file)` triplets to be unique and to match the embedded frame manifest exactly. It does **not** read or infer from `PRIVATE_UNBLIND_KEY.json`; it emits no mapping from a blind ID to a direction. The reproducible output is [`gate1_v5_packet_fixture_pixels_audit_v1.json`](gate1_v5_packet_fixture_pixels_audit_v1.json).

## Results

- Pass: each rater packet's embedded PNG multiset exactly matches the canonical fixture RGB file multiset; the two rater packets also match each other.
- Each packet has exactly two blind groups with 17 distinct frame indices each, and 34 complete rows with every scored field blank.
- Both CSV templates' 34 row keys match the embedded frame manifest one-to-one; no missing, duplicate, or cross-indexed image key was found.
- There are 34 frame records and 33 unique source PNG byte hashes. The report withholds per-frame hashes and source-path mapping to preserve the blind alias boundary.
- The audit read no rater submissions, unblind key, human labels, or truth-to-alias mapping. It did not change any packet or account state.

## Limits and next step

This closes packet/source visual-content provenance only. It does not show that raters are independent or agree, and cannot pass Gate 1. The prior fixture-alignment audit verifies mask/truth consistency; this audit adds the missing byte-level link from that fixture to the archived v5 packets. The next empirical input is still two independent, complete v5 CSVs. Apply the registered pooled and per-clip screens, inspect worst disagreements while blind, then run guarded truth scoring and the independent mirrored-seed block.

The packet frames are generated Kubric artifacts whose distribution terms remain unspecified in the inspected Kubric README/LICENSE; keep the packets local pending a complete provenance review. The verifier uses Python standard-library modules only. The final run took `0.020742 s` on Darwin arm64/Python 3.12.3 with no model, renderer, GPU, cloud runtime, or external data; peak VRAM is not applicable.
