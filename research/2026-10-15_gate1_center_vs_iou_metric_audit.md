# Gate 1 metric audit: center-path agreement versus box IoU

**Checked:** 2026-10-15 06:11 UTC  
**Component:** earliest open object-measurement gate; human-reference agreement for fixed-camera image-plane path centers.

## Relevant primary evidence

- Murrugarra-Llerena, Kirsten, and Jung, *Can We Trust Bounding Box Annotations for Object Detection?* (CVPR Workshops 2022), derive and measure how small box annotation discrepancies can disproportionately change IoU/AP, particularly for small objects and strict IoU thresholds. They also discuss that “tight box”/visible-part conventions can be subjective. The target in our fixture is a small 20×20 px sphere at 160×96, so box IoU is sensitive to edge choices; the study is adjacent evidence from object-detection datasets, not a calibration for our path-center threshold. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2022W/VDU/html/Murrugarra-Llerena_Can_We_Trust_Bounding_Box_Annotations_for_Object_Detection_CVPRW_2022_paper.html)
- Strafforello et al., *Humans Disagree With the IoU for Measuring Object Detector Localization Error* (ICIP 2022), ran a preference/acceptance study with more than 70 participants. Participants sometimes preferred one localization over another at the same IoU. This concerns human evaluation of displayed detections, not inter-annotator reliability or a task-state ground truth. The [official repository](https://github.com/ombretta/humans_vs_IoU) contains study materials/notebook, displays BSD-3-Clause, and was not downloaded or run.
- Llerena and Jung, *Noise-Aware Evaluation of Object Detectors* (WACV 2025), estimate corner-wise annotation noise on COCO and report AP sensitivity for small objects at tight IoU thresholds. The [official Error-Aware repository](https://github.com/Artcs1/Error-Aware) is visible and links the paper and a Drive-hosted annotation JSON archive, but the repository tree has no visible license file. No data or code was retrieved or reused.

## Decision for the registered measurement

Keep the primary Gate 1 numerical screen on per-axis center disagreement in control-grid cells, alongside separate categorical agreement and unique-visible-center coverage. That is aligned with the eventual 2D path objective, while IoU answers a different box-overlap question and is scale-sensitive for this small target. The current scorer already reports box IoU diagnostically without using it as a pass threshold; retain that split. Preserve raw box edges so the later apparent-size guardrail can measure width, height, and area separately. Keep the known-truth comparison for common-mode center bias and the mask-derived box discrepancy; neither replaces independent human labels.

No change to the 0.5-cell p95 limit, blind review ordering, or v5 schema is justified. A center-only pass does not certify size stability; when generated outputs are eventually scored, report the registered size guardrail as a separate outcome rather than folding it into path accuracy.

## Evidence boundary and next falsifiable action

The papers show metric sensitivity and subjective detection judgments, not how precisely people label this synthetic sphere or generated Wan frames. The local Kubric/Blender fixture remains the only relevant known-truth simulator for this gate; no external dataset is suitable for transferring the numeric tolerance. The Gate 1 p95 value remains a finite-packet process screen, not an empirical population bound.

Return two independent complete v5 CSVs. They must pass the existing pooled/per-clip center, categorical, and coverage criteria; then pass blinded worst-frame review and the guarded truth audit. Repeat with the second mirrored-seed block before a reliability claim. Keep IoU and box dimensions descriptive/guardrail measurements; do not invent an IoU cutoff after seeing labels.

**Resources / uncertainty:** primary-paper and official-repository review only; no repository cloned, dataset downloaded, package installed, or code executed. No compute; device and peak VRAM are not applicable. Repository terms are verified as BSD-3-Clause for `humans_vs_IoU`; no license was visible for `Error-Aware` in the inspected tree.
