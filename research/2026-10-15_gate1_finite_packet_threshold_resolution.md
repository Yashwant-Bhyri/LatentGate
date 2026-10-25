# Gate 1 finite-packet threshold resolution

**Checked:** 2026-10-15 06:16 UTC  
**Component:** Gate 1 — independent human-label agreement for the paired 17-frame Kubric fixture  
**Purpose:** Make the preregistered pass criteria's exact finite-sample implications explicit before any human labels are inspected. This is a protocol interpretation, not a threshold change.

## Frozen inputs and method

Read `research/2d_path_policy_component_plan.md`, `latentgate_experiment_registry.yaml`, and the current `research/score_annotation_agreement.py`. The scorer requires two complete matched submissions, computes the joint visibility/count/identity exact rate, computes unique visible-center coverage, and uses the local linearly interpolated quantile `pos=(n-1)q`. It requires the pooled screen and each blind clip's screen to pass. The packet has two clips with 17 frames each; registered limits remain 95% joint categorical agreement, 90% unique-center coverage, and axis-wise center p95 ≤0.5 control-grid cell.

The arithmetic below uses only those frozen values. No labels, clip key, or truth files were read. No code, thresholds, packet fields, or score outputs were changed.

## Exact count implications

| Criterion | Per 17-frame clip | Pooled 34 frames | Effective constraint after per-clip rule |
|---|---:|---:|---|
| Joint visibility/count/identity exact rate ≥95% | `ceil(0.95×17)=17` matches; 16/17 = 94.12% fails | `ceil(0.95×34)=33` matches; 33/34 = 97.06% passes | Both clips must be 17/17, hence 34/34 overall. Any one joint categorical mismatch fails this packet screen. |
| Unique visible-center pair coverage ≥90% | `ceil(0.90×17)=16` comparable frames; 15/17 = 88.24% fails | `ceil(0.90×34)=31` comparable frames | Per-clip constraints require at least 16 in each clip (≥32/34 total), so at most one frame per clip can be non-comparable. |

The categorical condition is on the **joint** visibility, instance-count, and identity fields. The report should still preserve the scorer's component-wise rates and frame records so a mismatch can be diagnosed; a pooled 95% display must not obscure that the per-clip screen requires exact categorical agreement on all 34 rows.

## Center-tail interpretation

At the minimum allowed 16 center pairs in a clip, this scorer's p95 uses sorted ranks 15 and 16 (one-based), weighted 0.75 and 0.25. With all 17 center pairs, it uses ranks 16 and 17, weighted 0.8 and 0.2. Thus the p95 rule alone is not a maximum-error bound; the already-registered blind worst-frame review remains necessary even when p95 passes. This audit does not add a maximum threshold or replace the existing worst-frame disposition.

## Literature, tools, and scope boundary

- Klie, Eckart de Castilho, and Gurevych's 2024 empirical study annotated 591 dataset-introduction papers. It found recurring problems in interpreting agreement and in making statistically sound claims from small annotation samples; it recommends confidence intervals and manual inspection, and explicitly cautions that agreement by itself does not guarantee annotation quality. This is adjacent NLP corpus evidence, not a visual tracking or temporal-frame calibration study. Because our 34 rows are temporally correlated and comprise one mirrored seed block, the count screen is descriptive process evidence, not a population reliability estimate. [Paper](https://aclanthology.org/2024.cl-3.1/).
- The authors' [official `qanno` repository](https://github.com/UKPLab/qanno) contains code for the paper's publication-corpus preprocessing and analysis. Its root license is GPL-3.0; the repository and its NLP publication corpus do not provide a suitable video-box evaluator or simulator. Nothing was downloaded, run, or reused.
- The existing Kubric/Blender fixture remains the only known-truth environment for this gate. This pass did not use a learned tracker, segmenter, model, cloud runtime, or GPU; device time and peak VRAM are not applicable.

## Decision and falsifiable next step

Keep the frozen criteria and blind-before-truth order. Make the effective 34/34 categorical requirement explicit when labels are returned. On the two v5 CSVs, require 17/17 joint categorical matches in each clip, at least 16 unique comparable centers in each clip, both axis-wise p95 values ≤0.5 cell per clip and pooled, and the registered blind review of the radial maximum and distinct axis maxima. Only after that blind screen passes may the guarded truth scorer open the key; a first-block pass still requires replication on the independently rendered mirrored-seed block. Preserve genuine ambiguity as disagreement/uncertainty rather than editing independent labels into consensus. Do not generalize this finite-packet result beyond this fixture.

