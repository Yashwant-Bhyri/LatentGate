# Gate 1 annotation protocol: reflected-object benchmark review

**Checked:** 2026-10-15 06:06 UTC  
**Component:** earliest open component, independent human reference reliability for fixed-camera 2D object paths.

## Primary-source and repository findings

Guo et al., *Tracking Reflected Objects: A Benchmark* (ACCV 2024), introduce TRO with about 200 sequences and 70,000 frames. Their annotation section describes tight axis-aligned boxes around the visible object, separate visible / absent / out-of-view / occluded states, SAM-assisted point annotations, visual inspection and manual correction, then a second annotator's review with discrepancies discussed to consensus. The paper is useful adjacent evidence for separating presence states and layering model proposals under human review. It does **not** report an inter-annotator error distribution or a numeric center tolerance in the annotation description, so its “high-quality” process claim cannot set our p95 threshold. The reflected-object domain is also not our simple synthetic rolling sphere.

The [official paper page](https://openaccess.thecvf.com/content/ACCV2024/html/Guo_Tracking_Reflected_Objects_A_Benchmark_ACCV_2024_paper.html) links the [authors' repository](https://github.com/OpenCodeGithub/HIP-HaTrack). At this check, the repository README says code and data will be made public after acceptance; the visible repository contains only that README and has no license file or downloadable dataset/checkpoint. Do not reuse its data or code unless availability and terms change and are verified.

## Decision for this gate

Keep the v5 blind, independent box labels as the human-reference measurement. Do not expose SAM/tracker proposals to raters or change the preregistered pooled/per-clip screen based on TRO: proposals can share errors, and TRO supplies no numeric reliability estimate. The existing post-label sequence remains: scorer agreement → blind worst-frame visual disposition → truth-gated comparison. For the 160×96 fixture, the unchanged half-cell limit maps to 1.538 source pixels horizontally and 1.600 vertically at the registered 832×480 / 52×30 scale.

TRO's more detailed absent / out-of-view / occluded taxonomy is a useful future schema consideration for generated-video observer evaluation. The present Kubric reference fixture keeps one object continuously visible, so do not change the v5 CSV schema or conflate this future issue with the current reliability gate.

## Evidence boundary and next falsifiable action

The applicable evaluator and known-truth simulator remain the local pinned Kubric/Blender fixture; public tracking datasets cannot calibrate the project's target-specific pixel-to-grid tolerance. Current labels remain absent, and the 34 frames in the first mirrored pair are correlated observations. No reliability pass or broader population claim is available.

Return two independent, complete v5 CSVs for the same 34 blind frames. They must pass the existing pooled and per-clip categorical, coverage, and axis-wise p95 limits; then complete blind worst-frame review and the guarded truth comparison. Repeat on the independently rendered mirrored-seed block before treating this annotation process as repeatable. If any generated-video observer schema is later widened to distinguish out-of-view, occlusion, and disappearance, preregister those states separately before labeling that sample.

**Resources / uncertainty:** paper and official-repository source review only; no data or software downloaded, no package/model run, no compute. Device and peak VRAM are not applicable. Main uncertainties are the unmeasured human agreement, small correlated first block, and weak transfer from TRO's reflection domain to our fixture.
