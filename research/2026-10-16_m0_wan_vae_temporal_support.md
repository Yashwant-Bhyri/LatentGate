# M0 Wan VAE temporal support: local source audit

**Date:** 2026-10-16  
**Scope:** CPU/source and saved-metadata inspection only. The exact pinned VAE config and matching Diffusers source text were fetched from their official primary-source pages and hashed. No VAE/model weights were fetched or loaded; no inference, generation, GPU, or cloud runtime was used.

## What the saved run establishes

The M0 run is pinned to `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` revision `0fad780a534b6463e45facd96134c9f345acfa5b`; the run environment used Diffusers 0.40.0. Its saved tensor map labels the VAE `AutoencoderKLWan` and reports temporal compression 4. The temporal audit records 17 RGB frames and latent shape `[B,C,T,H,W]=[1,16,5,60,104]`.

Applying the recorded length relation `T_latent = (T_RGB−1)/4 + 1` gives five **nominal stride anchors**:

| Latent time index | Nominal RGB frame index |
|---:|---:|
| 0 | 0 |
| 1 | 4 |
| 2 | 8 |
| 3 | 12 |
| 4 | 16 |

These indices identify the stride grid implied by the clip length and compression ratio. They do not say that a latent position uses only that RGB frame or that its receptive field is centered there.

## Exact receptive field and boundaries

The official config at the run's exact Hugging Face revision identifies `AutoencoderKLWan`, with `temperal_downsample=[false,true,true]`, `num_res_blocks=2`, and no down-block attention scales. The installed runtime was Diffusers 0.40.0; the matching official v0.40.0 `AutoencoderKLWan` source was inspected. The VAE encoder has four stages with two residual blocks each, two temporal downsampling operations, a two-block middle section, and a causal 3D output convolution. Each residual block has two temporal kernel-3 causal convolutions. The middle attention is applied independently to each time index, so it does not expand temporal support.

For the encoder's maximum causal receptive field, each kernel-3 stride-1 convolution adds two positions at its current temporal stride. A residual block therefore adds four current-grid positions. Accounting for the temporal stride at each stage gives the maximum past reach below:

| Encoder component | Temporal stride in RGB-frame units | Maximum additional past reach |
|---|---:|---:|
| `conv_in` | 1 | 2 frames |
| Stage 0: two residual blocks | 1 | 8 frames |
| Stage 1: two residual blocks | 1 | 8 frames |
| Temporal downsample 1, kernel 3 / stride 2 | 1 | 2 frames |
| Stage 2: two residual blocks | 2 | 16 frames |
| Temporal downsample 2, kernel 3 / stride 2 | 2 | 4 frames |
| Stage 3: two residual blocks | 4 | 32 frames |
| Middle: two residual blocks | 4 | 32 frames |
| `conv_out` | 4 | 8 frames |
| **Total maximum past reach** |  | **112 frames** |

Thus the structural receptive-field support for a latent position anchored at RGB frame `a` is the valid causal interval `[max(0, a−112), a]`. For this 17-frame run, that support reaches all available earlier frames at every position after the first. The five-position mapping is:

| Latent time index | Nominal RGB anchor | Structural RGB-frame support |
|---:|---:|---|
| 0 | 0 | `{0}` |
| 1 | 4 | `0–4` |
| 2 | 8 | `0–8` |
| 3 | 12 | `0–12` |
| 4 | 16 | `0–16` |

This is the architecture's possible dependency support, not proof that every listed frame has a nonzero numerical influence for the particular learned weights and input. The sequence-level encoding path processes the first frame alone and then groups of four, using cached causal features. Its two stride-2 temporal convolutions produce nominal endpoints `0, 2, 4, ...` and then `0, 4, 8, ...`; this matches the five anchors above. At the initial edge, causal convolutions left-pad with zeros and the first temporal-downsample call preserves the first feature position. At the final edge, the 17-frame sequence ends on a complete four-frame group; the encoder is causal and adds no future/right temporal padding. The final latent therefore depends on frames 0–16, with no frames beyond 16.

The local Python environment still lacks the `diffusers` package, and the unrelated cached non-Diffusers Wan VAE and separate MLX implementation were not used to infer this result.

## Primary-source provenance and hashes

- Exact model VAE config: [Hugging Face config at revision `0fad780a534b6463e45facd96134c9f345acfa5b`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/blob/0fad780a534b6463e45facd96134c9f345acfa5b/vae/config.json), SHA-256 `f0c1cc1d7decb5badc384f54691746a27a9aeff49f7ebca974e583389342d527`.
- Matching installed-runtime implementation: [Hugging Face Diffusers `autoencoder_kl_wan.py` at v0.40.0](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py), SHA-256 `a38de0c53adcb81887a0b48d44e3fb59737d7b1e4779a666febe83be666e8fa8`.

The hashes are for the fetched text files, not model weights. This update supersedes the initial local-only statement below that exact support was unavailable.

## Recommendation

Continue generating each video as one full sequence and score every decoded RGB frame for visibility, identity, boxes, and trajectory. Treat `[0,4,8,12,16]` only as nominal latent-time anchors for secondary internal-state summaries. This audit provides no reason to switch to temporal chunks; consider chunks only if measured memory, latency, or feedback-horizon pressure justifies a controlled comparison.

Machine-readable evidence, input hashes, source URLs, source hashes, and the per-position support calculation are in [`m0_wan_vae_temporal_support_audit_v1.json`](m0_wan_vae_temporal_support_audit_v1.json).
