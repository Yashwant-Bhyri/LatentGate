# Gate 1 v5 packet integrity and blinding audit

**Audit time:** 2026-10-11 14:23 UTC  
**Component:** Gate 1, human reference labels for fixed-camera 2D object-path control.

## Decision

The v5 synthetic pair is structurally ready for independent annotation: both rater ZIPs contain the same two clips, with identical embedded frame pixels under each shared blind ID and the clip order reversed between raters. The private unblind key is outside both archives. Each archive contains exactly `index.html`, `annotation_template.csv`, and `README.txt`; the templates are blank. The local page restricts content with CSP, uses embedded image data, and contains no external scripts or network APIs. Source-pixel dimensions and the SVG pointer-coordinate transform are present.

The reproducible audit is [`audit_gate1_v5_packet_integrity.py`](audit_gate1_v5_packet_integrity.py); result: [`gate1_v5_packet_integrity_audit.json`](gate1_v5_packet_integrity_audit.json). Script SHA-256: `202c2036577e5199ba50b7f1cf4b3bf0d878a81274a5781b954540bb958c41fd`. Result SHA-256: `179988a70d5a83cb29a104ec3f40cb0dea0a63d843be01d84cf66582a4b69f9e`.

The audit took 0.039848 seconds on Darwin arm64 / Python 3.12.3; GPU use was false and peak VRAM was null/not allocated. It validates packet construction only. It does **not** establish that human raters agree, that the observer transfers to generated Wan clips, or that any steering policy improves trajectory adherence. All v5 human templates remain blank.

## Next falsifiable step

Use the v5 synthetic packets for two independent annotations of both 17-frame clips. Score with `--source-width 160 --source-height 96 --expected-clips 2`, require pooled and per-clip thresholds, and retain worst-frame errors. If the one-pair screen passes, obtain a second independently rendered mirrored-seed block before accepting reference reliability.
