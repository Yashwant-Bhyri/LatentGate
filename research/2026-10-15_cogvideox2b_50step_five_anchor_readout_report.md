# CogVideoX-2B 50-step five-anchor RGB readout: completed

## Result

The preregistered integrity gate passed: the saved manifest and all 16 RGB frame hashes matched, and each frame was 720×480. The fixed 300-pixel coverage gate also passed in all 36 registered red-threshold settings.

Using anchors `[0,4,8,12,15]`, the baseline threshold `(R>150, G<125, B<125)` gave a mean omitted-frame centroid error of **1.14 px** (RMSE 1.31 px; maximum 1.89 px). Its mask-area interpolation error averaged **12.0 pixels** (maximum 34.5 pixels), or **0.62% mean / 1.83% maximum** relative error. Across all 36 threshold settings, mean center error ranged from **0.95 to 1.15 px**, maximum center error from **1.64 to 2.03 px**, and mean relative area error from **0.45% to 3.24%**.

The anchor schedule intersects 5/16 one-frame event windows, 8/15 two-frame windows, 11/14 three-frame windows, and 13/13 four-frame windows. These are schedule counts, not events observed in the clip.

## Interpretation

Five points summarize this one clip's smooth red-mask path closely. They miss 11/16 possible one-frame and 7/15 two-frame event placements, so the result supports sparse anchors only as a coarse summary for this sample. Keep dense decoded-frame scoring for brief events, visibility, identity, and trajectory evaluation. This is an RGB proxy on one clip, not ground truth, a general sampling claim, or a chunked-generation comparison. No latent files were read and no inference, GPU, Colab, Kaggle, or new generation was used.

The full preregistration, runner, and machine-readable output are [`preregistration`](cogvideox2b_50step_five_anchor_preregistration_v1.json), [`runner`](audit_cogvideox2b_50step_five_anchor_readout.py), and [`result`](cogvideox2b_50step_five_anchor_readout_v1.json).
