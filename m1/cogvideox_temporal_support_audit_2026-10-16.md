# CogVideoX temporal VAE support audit

**Reviewed:** 2026-10-16 (Asia/Shanghai)  
**Experiment:** `EXP-M1-COGVIDEOX5B-I2V-SMOKE-001`  
**Scope:** Source-only temporal-architecture derivation for the exact CogVideoX-5B-I2V checkpoint revision and Diffusers package version already pinned by M1. No weights, inference, tests, GPU/runtime, notebook cells, upload, or browser-profile changes were used.

## Pinned architecture

- Model: `zai-org/CogVideoX-5b-I2V`, revision `a6f0f4858a8395e7429d82493864ce92bf73af11`.
- Diffusers: `0.40.0`, from the M1 package stack.
- The checkpoint VAE config selects `AutoencoderKLCogVideoX`, 16 latent channels, four down/up blocks, three residual layers per block, spatial scale factor 8, and temporal compression ratio 4. The checkpoint transformer config has 49 sample frames and no `patch_size_t`; Diffusers 0.40.0's transformer default for the missing field is `None`.

Sources: [pinned VAE config](https://huggingface.co/zai-org/CogVideoX-5b-I2V/blob/a6f0f4858a8395e7429d82493864ce92bf73af11/vae/config.json#L128-L166), [pinned transformer config](https://huggingface.co/zai-org/CogVideoX-5b-I2V/blob/a6f0f4858a8395e7429d82493864ce92bf73af11/transformer/config.json), [Diffusers 0.40.0 transformer defaults](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/transformers/cogvideox_transformer_3d.py#L220-L241).

## 49 RGB frames to 13 latent positions and back

The I2V pipeline computes latent time length as `floor((F - 1) / 4) + 1`. For `F=49`, this is 13 positions. At 720×480, the spatial VAE scale of 8 gives latent maps of 60×90, with 16 channels per denoised latent position. The pipeline separately encodes the single conditioning image to one VAE latent and appends 12 zero conditioning latents; this conditioning tensor is distinct from the 13-position noise/sample sequence.

The encoder sets `temporal_compress_level = int(log2(4)) = 2`, so only its first two down blocks compress time. Each temporal downsample preserves position 0 and average-pools adjacent pairs from the remaining positions; the other two down blocks downsample space only. The time lengths therefore go `49 → 25 → 13`:

| Latent position | Nominal input-position ancestry after the two pooling stages |
|---:|---|
| 0 | frame 0 |
| 1 | frames 1–4 |
| 2 | frames 5–8 |
| … | … |
| 12 | frames 45–48 |

These are pooling ancestry groups, not averages of raw RGB frames or exclusive receptive fields. Causal temporal convolutions in the encoder can also carry earlier-frame context into later positions, so `{0,4,8,…,48}` is a useful nominal inspection grid, not proof that each latent represents only one sampled RGB frame or one disjoint four-frame window.

The decoder uses two temporal upsample blocks. Each preserves the first position and doubles the remaining sequence, giving `13 → 25 → 49`. The nominal output alignment is one first frame from latent position 0, then four output frames per later latent position: position 1 aligns to frames 1–4, position 2 to 5–8, and position 12 to 45–48. Decoder causal convolutions can mix earlier latent context into later RGB frames.

Source: [Diffusers 0.40.0 I2V latent preparation and length formula](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L363-L417), [VAE encoder block setup](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L668-L719), [temporal downsample implementation](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/downsampling.py#L286-L351), [VAE decoder block setup](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L820-L895), [temporal upsample implementation](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/upsampling.py#L357-L415).

## Temporal edges and padding

The encoder and decoder default to `pad_mode="first"`. For a temporal kernel of 3, each causal convolution with no cache prepends two copies of its first temporal feature, pads on the left, and adds no right-side temporal context. The two compression stages receive odd lengths 49 and 25: each preserves the first position and pairs all 48 then 24 remaining positions, so the last source frame (48) is retained and no tail frame is dropped.

The pipeline's optional temporal patch padding is for checkpoints with non-null `patch_size_t`. This pinned 5B-I2V config omits that field and Diffusers defaults it to `None`, so no extra latent position is added and the corresponding output crop removes zero positions. The normal decode length remains 49.

Sources: [causal convolution edge handling](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L67-L142), [VAE temporal-padding defaults](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L668-L719), [I2V patch padding](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L778-L785) and [output crop](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L894-L898).

## Schedule-only short-event coverage

For the nominal sparse RGB inspection schedule `[0,4,8,…,48]`, enumerate every possible contiguous window of `d` frames in the 49-frame output. A window is counted as missed when it contains no inspected index. The percentages are schedule arithmetic only:

| Window length | Duration at 8 fps | Possible start positions | Windows with no anchor | Missed |
|---:|---:|---:|---:|---:|
| 1 frame | 0.125 s | 49 | 36 | 73.5% |
| 2 frames | 0.250 s | 48 | 24 | 50.0% |
| 3 frames | 0.375 s | 47 | 12 | 25.5% |
| 4 frames | 0.500 s | 46 | 0 | 0.0% |

These counts say when a sparse frame-review schedule would contain no sample from a possible event window. They are not event labels, event-rate estimates, or evidence that CogVideoX's internal latent lost the event; causal temporal context may carry information across these nominal groups. Dense scoring of all 49 decoded RGB frames remains the M1 requirement.

## Experiment boundary and interpretation

This audit is architecture guidance, not efficacy evidence. It does not estimate a total VAE receptive field, confirm M1 runtime fit, measure output quality, or show that any event survives generation. M0's Wan2.1 causal-support result, including its 112-frame past reach, is specific to Wan and is not transferred here. M1 remains the registered seed-2026, full-sequence baseline plus exact same-runtime replay, with all 49 decoded frames scored. No run setting, arm, or all-frame gate changed.

## Packaging integrity follow-up

During this audit, a cross-check found the local upload ZIP included the older `geometry_scale_drift_ticket.json` even though its ZIP digest was correctly pinned by the notebook. The mismatch affected the packaged review contract: the bundled ticket omitted the current all-49-frame score-table requirement. The runner and generation settings were current and the runner does not read the ticket to control generation. The ZIP has now been rebuilt from the six current local inputs, including the current ticket; the notebook pin and project hashes are updated in the readiness report, static capture audit, registry, and state log. The designated Drive notebook is still stale and has not been refreshed.
