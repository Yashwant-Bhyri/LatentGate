# M0 Wan microscope

`run_config.yaml` pins the Wan2.1 T2V 1.3B Diffusers experiment to an immutable model revision and records the Colab T4 runtime target. **M0 is complete in retry5:** all ten configured gates pass, including predicted-clean estimates and their decoded previews, which were missing from retry4. The corrected runner records effective FP16 transformer compute separately from Wan's intentionally retained FP32 parameters.

The adjacent **Run a small diffusion model** task completed a CogVideoX-2B feasibility pilot and created an incomplete Wan2.1 MLX script outside this repository. Neither is this M0 run. See [../STATE.md](../STATE.md) for the artifact audit.

The run uses 17 frames at 832×480 for 30 steps. Wan requires `4n+1` frame counts. Retry5 saved post-step latents at steps 0, 4, 9, 14, 19, 24, and 29; predicted-clean latents and decoded previews at steps 4, 14, 24, and 29; selected transformer-block summaries pooled by latent frame; and bounded sampled self-attention summaries. Predicted-clean estimates use the pinned `UniPCMultistepScheduler` conversion path for `flow_prediction`. Self-attention capture is bounded to a deterministic sample of query tokens; it does not materialize full attention matrices or record cross-attention.

The pinned transformer architecture, effective runtime dtypes, all required captures, VAE round trip, and exact same-seed replay pass. Retry5 took 160.02 seconds for generation and 142.10 seconds for replay, with 7.91 GiB peak allocated GPU memory. All 128 paths hashed in the manifest are present and match; the manifest itself is not self-hashed. Retry5's 17 final frame PNGs are byte-for-byte identical to retry4, as are all seven saved post-step latent tensors. Retry4 remains in [`runs/wan_t4_2026-10-02_retry4`](runs/wan_t4_2026-10-02_retry4) as the historical first attempt with predicted-clean capture gaps. The latest frames and previews are linked below; the motion audit is in [`analysis/wan_t4_retry5_frame_motion_audit.json`](analysis/wan_t4_retry5_frame_motion_audit.json).

## Latest artifacts

- [Colab execution notebook export](notebooks/LatentGate_M0_Wan_Microscope.ipynb) — 18-cell notebook with the prior attempts, retry5 execution, artifact inspection, and runner hash verification.
- [Colab notebook](https://colab.research.google.com/drive/1jmYmwpV70RonpQUuQV-nbEcJ0Bzby1wI)
- [M0 autopsy report](runs/wan_t4_2026-10-02_retry5/autopsy_report.md)
- [Generation Packet](runs/wan_t4_2026-10-02_retry5/generation_packet.json)
- [Final 17-frame contact sheet](runs/wan_t4_2026-10-02_retry5/run_001/frames_contact_sheet.png)
- [Step 4 predicted-clean preview](runs/wan_t4_2026-10-02_retry5/run_001/predicted_clean_previews/step_004/frames_contact_sheet.png)
- [Step 14 predicted-clean preview](runs/wan_t4_2026-10-02_retry5/run_001/predicted_clean_previews/step_014/frames_contact_sheet.png)
- [Step 24 predicted-clean preview](runs/wan_t4_2026-10-02_retry5/run_001/predicted_clean_previews/step_024/frames_contact_sheet.png)
- [Step 29 predicted-clean preview](runs/wan_t4_2026-10-02_retry5/run_001/predicted_clean_previews/step_029/frames_contact_sheet.png)
- [SHA-256-verified retry5 runner source](runner_snapshots/run_m0_retry5_2026-10-02.py)
- [Post-hoc frame motion audit](analysis/wan_t4_retry5_frame_motion_audit.json)
- [Offline spatial-latent location readout protocol](../research/2026-10-12_m0_spatial_latent_location_readout.md) and [machine-readable result](analysis/m0_spatial_latent_location_readout_v1.json) — CPU-only same-clip analysis of saved latents; late snapshots rank the red-color proxy, but center-error improvement over the coordinate prior is small and not robust.

These predicted-clean sheets show the model's current clean-video estimate during denoising, decoded for inspection. They are not separate frame-by-frame generations. This single prompt and seed establish instrumentation feasibility, not representation or intervention hypotheses. M1 planning is next; the [failure-atlas plan](../m1/failure_atlas_plan.md) sets the common-start/conditioning gate, independent teacher metrics, and pilot before the proposed 50–100-seed batch.
