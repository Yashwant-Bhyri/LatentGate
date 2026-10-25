# M0 saved RGB sequence inventory

**Date:** 2026-10-16  
**Status:** Completed CPU-only inventory. No interpolation readout was repeated.

## Inventory

Five saved run records contain complete, sequential, readable RGB frame sets with dimensions and frame hashes matching their existing manifests or frozen input maps. The two Wan records are an exact byte-for-byte replay of the same 17-frame output, so the inventory contains **four unique RGB sequences**, not five independent samples.

| Saved run | Backbone / steps | RGB frames | RGB file set | Fixed red proxy, baseline | Full coverage across 36 thresholds |
|---|---|---:|---|---:|---:|
| Wan retry4 | Wan2.1 T2V 1.3B / 30 | 17 × 832×480 | Complete; instrumentation attempt has capture gaps | 17/17 | 36/36 |
| Wan retry5 | Wan2.1 T2V 1.3B / 30 | 17 × 832×480 | Complete; M0 run passes | 17/17 | 36/36 |
| CogVideoX-2B `chunk13` | 50 | 16 × 720×480 | Complete | 16/16 | 36/36 |
| CogVideoX-2B `pilot` | 8 | 9 × 480×320 | Complete | 0/9 | 0/36 |
| CogVideoX-2B `pilot_30steps` | 30 | 9 × 480×320 | Complete | 0/9 | 0/36 |

“Complete RGB file set” means the declared PNGs exist in order, decode as RGB at the recorded size, and match the bound hashes. “Proxy scoreable” applies the already registered red mask and 300-pixel minimum to every frame. It does not mean that the clips have human ground-truth labels. Across the full threshold grid, the 8-step output has zero qualifying red pixels and the 30-step output peaks at 203 pixels in one frame, still below the fixed floor. Do not infer motion from either low-resolution run.

## What five anchors can miss

These are schedule-only counts: a window is “seen” if it intersects an anchor. They are not observed event labels or event rates.

| Existing schedule | 1-frame windows missed | 2-frame windows missed | 3-frame windows missed | 4-frame windows missed |
|---|---:|---:|---:|---:|
| Wan 17-frame `[0,4,8,12,16]` | 12/17 | 8/16 | 4/15 | 0/14 |
| CogVideoX 16-frame `[0,4,8,12,15]` | 11/16 | 7/15 | 3/14 | 0/13 |
| CogVideoX 9-frame `[0,2,4,6,8]` | 4/9 | 0/8 | 0/7 | 0/6 |

The schedules make brief one-frame events easy to miss; the every-other-frame schedule on the 9-frame clips intersects all 2–4-frame windows. That says nothing about whether those events occurred. The existing Wan and CogVideoX five-anchor interpolation metrics were not recomputed here.

## Matched comparison and next evaluation

There is no informative existing matched multi-clip motion comparison. The CogVideoX 8- and 30-step files share the recorded prompt, seed, resolution, and frame count, but neither passes the fixed proxy coverage gate, both are below the model's documented 720×480 resolution, and checkpoint revision, runner, and effective initial-noise provenance are missing. The scoreable 50-step run changes resolution and frame count. Wan retry4/retry5 are exact replay duplicates, and Wan versus CogVideoX changes the backbone and generation setup.

The smallest useful **exploratory** replication after the current two-rater labels for the saved Wan clip and a fresh free-T4 admission is two more unique seeds at the exact pinned M0 setup. Together with the existing seed-2026 sequence, that yields three unique 17-frame Wan clips at 832×480 and 30 steps under one fixed prompt, revision, scheduler, and guidance setting. Jointly generate each whole sequence; save and hash all decoded frames; double-label and score every frame with the existing visibility/count/identity and tight-box-center contract. Keep clip/seed as the unit of reporting. Evaluate the five-anchor schedule only as a secondary summary over those same dense labels.

This three-clip screen is exploratory and does not pass or replace the existing observer-release gate: three development seed blocks followed by five fresh held-out blocks, with the registered agreement, coverage, center-error, and wrong-identity requirements unchanged. Do not add a chunked-generation arm unless memory, latency, or feedback-horizon pressure is measured. Two additional M0-like generations are about **320 seconds (5.3 minutes) of generation-only T4 time**, based on the measured 160.02 seconds per baseline run; setup, model download, annotation, and artifact I/O are extra. Free runtime admission remains unconfirmed, so this is a conditional plan, not an authorized or started run.

The complete per-run frame inventory, hashes, proxy counts, and missed window start positions are in [`m0_saved_rgb_sequence_inventory_v1.json`](m0_saved_rgb_sequence_inventory_v1.json). Reproducer: [`inventory_m0_saved_rgb_sequences.py`](inventory_m0_saved_rgb_sequences.py).
