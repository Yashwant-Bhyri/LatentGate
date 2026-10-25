# VIGOR and object-motion geometry steering

**Updated:** 2026-10-02  
**Status:** focused primary-source audit; Frame Guidance is the lead latent-intervention mechanism, VIGOR is the reward-audit baseline, and GEM-4D is the longer-term feature-distillation reference. No new Colab run has started.

## Decision in brief

The first failure mode remains **rigid-object shape and correspondence consistency while the object moves**. The M0 red-ball area change motivates this target but does not establish a 3D failure: it is a single clip and a color-threshold image proxy.

For the North-Star intervention, Frame Guidance is the lead mechanism: it updates a video latent using a selected-frame objective with gradients through the denoiser and VAE. Its Wan code uses a custom flow-matching loop, so this is a task-specific UniPC adaptation, not plug-and-play. VIGOR is the best open geometry-reward audit baseline we have found, but its public reward targets static-scene/camera consistency. GEM-4D remains the longer-term robot-geometry feature-distillation reference, but has no public implementation/checkpoint at the time of this audit. None is a direct reproduction for our task.

The key gap is that VIGOR's reprojection reward treats selected image points as static world points transformed only by the estimated camera. That is a sensible signal for camera/background consistency, but it can mark a correctly moving manipulated object as inconsistent. The first project contribution should therefore be an object-motion-aware reward or latent policy that separates static background from a rigid object with its own time-varying SE(3) pose. It must preserve intended motion while holding object shape/correspondence together.

## What VIGOR contributes

VIGOR estimates camera, depth, and 3D structure with VGGT. Its Geometry-Aware Sampling selects high-attention patches; a point tracker follows each selected point; the reward back-projects points to 3D in a reference frame, reprojects them under another frame's camera, and measures pixel distance to the tracked location. It provides pointwise reprojection, pixel-warp, and epipolar variants. The pointwise objective avoids direct pixel-intensity error and uses estimated geometry as a reward teacher. [Paper](https://arxiv.org/html/2603.16271v1) · [Official code](https://github.com/Ethan2k04/VIGOR)

The repository exposes three pathways:

| Path | What changes | Relevance to our Wan baseline | Main caveat |
|---|---|---|---|
| Best-of-N | Generate several full clips, score, keep one | The clean low-risk control for a bidirectional Wan sampler | Extra sampling cost; does not change model weights |
| DPO/SFT LoRA | Fine-tune Wan attention projections from geometry-ranked preferences | Direct precedent for a persistent model-weight change | Published data are mostly real-estate/landmark scenes, not robot manipulation; SFT reduces motion substantially |
| Search on Start/Path/Beam | Use geometry reward to choose autoregressive chunk paths | Useful for causal streaming models | The released path-search code is not a direct fit for M0's bidirectional Wan + UniPC loop |

The paper's GB3DV-25k pairs come from RealEstate10k and Google Landmarks prompts, with ten CausVid samples per prompt and best/worst reward ranking. The reported bidirectional Best-of-10 experiment is on CausVid; Wan2.1-T2V-1.3B is the post-training backbone. In the paper's Wan table, DPO improves several reported metrics (for example, SSIM, LPIPS, EPI, and pointwise reprojection) while the pixelwise reprojection value moves slightly worse; SFT drops VBench dynamic degree from 27.34 to 15.62, and DPO recovers it to 25.78. These are useful published results, not independent evidence for robot-object manipulation. [Paper experiments](https://arxiv.org/html/2603.16271v1#S6)

The VIGOR implementation's reference evaluator defaults to at most 10 frames, 448-pixel short side, up to 2,048 query points, and VGGT-derived attention maps. The sampler code defaults to 40 steps and 81 frames, but these are configurable. Our M0 T4 result establishes a much smaller, concrete operating point: Wan 1.3B, FP16, 832x480, 17 frames, 30 steps, about 160 s generation time and 7.91 GiB peak allocated GPU memory. It does **not** establish VIGOR compatibility or evaluator memory. The VIGOR inference path uses DiffSynth, has a different loading/scheduler stack, and defaults to BF16; T4 is compute capability 7.5, so any port or adapter check should use FP16 and be measured. [VIGOR reward config](https://github.com/Ethan2k04/VIGOR/blob/master/rewards/config/config.json) · [VIGOR Best-of-N code](https://github.com/Ethan2k04/VIGOR/blob/master/bon/best_of_n.py) · [M0 config](../m0/run_config.yaml) · [M0 report](../m0/runs/wan_t4_2026-10-02_retry5/autopsy_report.md)

The project README claims MIT and announces pretrained LoRA adapters, but the tracked repository has no `LICENSE` file, GitHub Releases is empty, the author profile exposes no public model repository, and the training download script fetches the dataset and base Wan weights rather than a LoRA. Treat the adapter as unavailable until an official checkpoint URL, base revision, and license are verifiable. The Hugging Face dataset page reports 21.2 GB and a generic `cc` license field; inspect the card and per-asset terms before reuse or redistribution. The evaluator hard-codes original `facebook/VGGT-1B`, whose checkpoint has a non-commercial license; the separately gated commercial VGGT checkpoint has different terms and is not wired into VIGOR. The repository's code claim, dataset terms, checkpoint terms, and downstream use must be checked separately. [Dataset card](https://huggingface.co/datasets/Ethan2k04/GB3DV-25k) · [VIGOR training download script](https://github.com/Ethan2k04/VIGOR/blob/master/dpo/model_training/download.sh) · [VGGT-1B model card](https://huggingface.co/facebook/VGGT-1B)

## Why the stock reward is not yet our object-motion objective

In the released reprojection evaluator, a selected point's reference-frame 3D coordinate is projected into another frame using that frame's camera pose, then compared with the tracker's image location. It does not first estimate a separate rigid transform for a manipulated object's surface. This is an inferred mismatch from the paper equations and code, not a failure result reported by VIGOR. Therefore:

- Background points should receive a camera-compensated static-world consistency score.
- Object points should receive a per-frame rigid-body score: fit one time-varying SE(3) transform to object correspondences, then measure residual shape deformation/reprojection after that transform.
- The object trajectory should be assessed separately against the prompt or task phase. A policy must not win by freezing the mug/ball or by suppressing the grasp/placement.
- The scoring teacher must not be the only evaluator. Use CAD pose and camera ground truth on synthetic held-out clips where possible, plus an independent track/mask evaluator and human review on generated videos.

This also separates our target from generic scene reconstruction. GEM-4D's dense temporal geometry features and geometry-flow auxiliary branch are a good conceptual route for a robot-specific geometry teacher, but GEM-4D is not presently released as runnable code/checkpoints and its paper omits core reproduction details. Geometry Forcing remains a useful public feature-alignment comparator but focuses primarily on camera/world geometry rather than a moving robot object. [GEM-4D](https://arxiv.org/html/2605.22882v4) · [Geometry Forcing](https://arxiv.org/html/2507.07982v2)

## Candidate intervention paths

1. **Reward-ranked Best-of-N diagnostic.** Score a small set of M0-matched clips and compare VIGOR's winner with a predeclared first/random choice from the same candidates. This tests whether the score contains signal for the prompt; it is a comparator, not a model improvement.
2. **Object-aware latent guidance.** Extend Frame Guidance with a loss that is invariant to the allowed object translation/rotation but penalizes changing object shape/correspondences; add a separate trajectory/task loss so the object cannot freeze. The current Wan integration uses FlowMatch Euler; M0 uses UniPC, so preserve one sampler in both arms or label a change of sampler as a new paired control. A discrete VGGT/tracker score cannot be inserted as a direct gradient; it needs a differentiable surrogate or separate teacher-distillation stage.
3. **Object-aware preference LoRA.** If latent guidance passes its feasibility and independent-evaluator checks, use generated/teacher-ranked task pairs to train a compact adapter on held-out objects/prompts/seeds. VIGOR's claimed adapter is not currently discoverable, and its published 10,000-step BF16/rank-64 setup is not a T4 recipe; a reduced FP16 run would be a new adaptation needing its own memory/quality validation.
4. **GEM-style auxiliary geometry branch.** Add a train-time geometry-flow head supervised by dense object-aware geometry features, then discard it at inference. This is the longer-term route to internalized geometry but needs a controlled dataset and backbone/LoRA adaptation. Use GEM-4D as design inspiration, not a reproduction claim.

## First controlled pilot after runtime coordination

Keep the first pilot small and answer a single question: **does geometry-based selection or a geometry-informed intervention reduce object-shape drift while preserving requested movement?**

1. After runtime coordination, smoke-test VIGOR scoring on the saved M0 clip with 4–5 sampled frames at the documented 448-pixel short side, after unloading Wan. VGGT reconstructs once and its released evaluator makes additional tracker/backbone passes per reference frame, so record wall time, peak VRAM, and valid point counts before using the 10-frame default.
2. If scoring fits and returns valid pairs, reuse M0 seed 2026 plus one matched-config second-seed output. Compare the VIGOR-selected clip to a predeclared first/random choice from the same pair. This is a two-candidate reward-signal smoke test, not a performance claim.
3. In both clips, separately record object presence, mask area/shape, point tracks, and motion displacement. Treat VIGOR's score as a scene/camera comparator only; its attention may emphasize background and its static-world assumption may penalize object motion.
4. Add an object-aware rigid residual on a fixed-camera synthetic CAD fixture with known camera and object poses. The existing image-generated mug/holder fixture has no such ground truth, so do not claim metric 3D from it.
5. If the reward ranks samples in a way that tracks independent object metrics without freezing/suppressing motion, expand to several seed pairs. Then implement the first Frame-Guidance-inspired intervention on the same sampler and paired seed, including zero-guidance and norm-matched random-gradient controls. Evaluate every frame, including unguided frames, for rigid shape and intended motion.
6. If latent guidance passes feasibility and independent-evaluator checks, use task-specific preference pairs for a persistent adapter/gate. VIGOR's released adapter is not presently reproducible from public links, and its published DPO training needs more GPU memory than our T4. Treat any short FP16 LoRA run as an adaptation, not a VIGOR reproduction.

## Outstanding checks

- Ask the VIGOR authors for or locate the announced LoRA artifact, base model revision, and license; public GitHub Releases and the linked author HF profile currently expose no model checkpoint.
- Verify whether its evaluator can run on a 15 GB T4 after unloading Wan; start with 4–5 frames because the default scoring path repeats a VGGT tracking/backbone call for each reference frame.
- Resolve VGGT checkpoint and GB3DV dataset terms before any redistribution or commercial use; the repo's MIT claim does not cover these assets.
- Review frame tracking, validity filtering, and score behavior on independently moving objects; create a known-pose synthetic counterexample.
- Choose a CAD/simulator fixture and a task intent label that permit correct rigid motion rather than favoring stillness.
- Decide whether the first in-model route is a LoRA/DPO update, learned latent reward gate, or Frame-Guidance-inspired differentiable latent correction after one feasibility pass.

## Current evidence boundary

No VIGOR model, teacher, DPO adapter, or latent guidance has run in this project yet. The M0 area-growth number remains an image-space proxy. No Colab generation or evaluation was started during this audit, and no existing planning file was edited. The Colab coordination report and user-approved notebook link were sent to the other diffusion task; the waiting upload cell was left unchanged.
