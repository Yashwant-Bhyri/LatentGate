# DiTraj Kaggle staging and source audit

**Checked:** 2026-10-12 04:52 UTC  
**Scope:** static local notebook/source review, designated Kaggle draft inspection, and read-only accelerator menu check. No cell was executed.

## Local notebook review

The intended source is `research/ditraj_paired_path_pilot_v1.ipynb`, SHA-256 `2e697fb5942472a0e6dd71a6c67975345f9fb7970aae08609221cd8940583e4e`. It has 12 cells, no execution counts, and no saved outputs. The manual free-quota gate defaults to `False`; the full 30-step and feedback runs also default to `False`. The one-step command is a six-case plumbing smoke, not path-control evidence. The T4 check precedes model-weight loading. The notebook was reviewed locally but was not staged.

The runner sets a direction-neutral prompt, mirrored targets, seed 2026, and a single initial latent tensor whose SHA-256 is logged and cloned for each arm. Each call requests 17 frames and rejects any other returned shape; it saves every decoded RGB frame for blinded labeling. The 17-frame target is mapped to five temporal boxes via `[::4]`. These are source-level protocol facts, not results from an inference run.

## Pinned implementation audit

At the pinned [DiTraj `pipe.py`](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/pipe.py), `mask_step=0` routes the base prompt on all denoising calls and disables the cross-attention mask; `fixRope_step=0` disables the RoPE correction branch. The matching [attention processor](https://github.com/xduzhangjiayu/DiTraj/blob/6fa8e613c9e7c40ec2db456ab2ba34367a076886/module/attention_processor.py) passes no attention mask when the mask flag is false. The [pinned Diffusers 0.33.1 transformer](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_wan.py) has the corresponding base operations. This supports the intended zero-action behavior at source level; a decoded, matched no-op replay has not been measured, so bitwise equivalence is not claimed.

## Designated remote draft and upload blocker

The open Kaggle URL is under the designated `bvyashwantkumar` account. It shows a saved four-cell draft with Draft Session off and no executed cells. Its first code cell clones DiTraj and Diffusers and installs requirements; a blank cell follows, then two duplicate T4/CUDA preflight cells. It is the stale draft, not the reviewed 12-cell source, and must not be run.

The current Settings → Accelerator menu lists `None`, `GPU T4 x2` (selected), and `TPU v5e-8`; it exposes no free-quota or admission value. Selection does not establish that a free T4 is available. No runtime was started.

File → Import Notebook opened its dialog. The browser's file chooser rejected the local notebook because Chrome's ChatGPT extension does not have **Allow access to file URLs** enabled. The dialog was cancelled. No file was transmitted, no permission was changed, and no account or terms prompt was accepted. Staging remains blocked until that browser permission is enabled or another approved transfer route is available.

## Gate decision

Gate 1 remains earliest: the repeat scan still finds no human annotations. Keep the cloud run off. After the file-transfer blocker is cleared, import and verify the exact 12-cell hash on the designated draft without starting a session. Before a later runtime, obtain a fresh affirmative free-T4/quota check. After Gate 1 passes, run the six-case one-step smoke; keep it separate from the 30-step open-loop screen and the later feedback arms. Any efficacy claim still requires independent labels and the matched open-loop, decode-matched no-op, frozen-static, live-feedback, shuffled-residual, and sign-reversed comparison.
