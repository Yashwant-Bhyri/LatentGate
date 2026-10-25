# M0 Autopsy: M0_WAN21_T2V13B_T4_001

- **Model:** `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- **Revision:** `0fad780a534b6463e45facd96134c9f345acfa5b`
- **Prompt:** A single bright red ball rolling slowly from left to right across a smooth white tabletop, fixed camera, soft daylight, simple clean background.
- **Seed / size / steps:** 2026 / 832×480 / 30
- **Frames requested:** 17
- **Runtime:** Tesla T4 (Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8)
- **Generation time:** 160.02 s
- **Peak allocated GPU memory:** 7.91 GiB
- **VAE round trip:** ok; RMSE 0.0019180503076321842
- **Same-seed replay:** completed
- **M0 completion assessment:** complete

## Model checks

- Transformer architecture matches expected config: True
- Runtime tensor dtypes match expected config: True

## M0 gates

- PASS — runtime matches config
- PASS — transformer architecture matches
- PASS — runtime dtypes match
- PASS — post step latents complete
- PASS — predicted clean latents complete
- PASS — predicted clean previews complete
- PASS — hidden state summaries complete
- PASS — attention summaries complete
- PASS — vae roundtrip complete
- PASS — same seed replay exact

## Visual outputs

- Final video frames: `run_001/frames/`
- Final frame contact sheet: `run_001/frames_contact_sheet.png`
- VAE round-trip frames: `run_001/vae_roundtrip_frames/`
- VAE round-trip report: `run_001/vae_roundtrip_report.json`

## Predicted-clean previews

- Step 4: `predicted_clean_previews/step_004/frames_contact_sheet.png` (17 frames)
- Step 14: `predicted_clean_previews/step_014/frames_contact_sheet.png` (17 frames)
- Step 24: `predicted_clean_previews/step_024/frames_contact_sheet.png` (17 frames)
- Step 29: `predicted_clean_previews/step_029/frames_contact_sheet.png` (17 frames)

## Capture status

- Post-step latents: [0, 4, 9, 14, 19, 24, 29] / [0, 4, 9, 14, 19, 24, 29]
- Predicted-clean latents: [4, 14, 24, 29] / [4, 14, 24, 29]
- Transformer summaries: 18 / 18
- Attention summaries: 18 / 18
- Attention limitation/failure: none reported
- Predicted-clean limitation/failure: none reported

## Interpretation boundary

This run instruments one prompt and one seed. It is a pipeline feasibility result, not evidence that hidden states encode a target variable or that an intervention improves generation. The attention output is a sampled-query self-attention summary; transformer outputs are spatially pooled per latent frame.
