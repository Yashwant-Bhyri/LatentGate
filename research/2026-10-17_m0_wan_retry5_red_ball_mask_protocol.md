# Wan retry5 red-ball image-mask protocol

**Status:** fixed before scoring this diagnostic, 2026-10-17.

## Pinned input

Use only the 17 decoded RGB PNGs in `m0/runs/wan_t4_2026-10-02_retry5/run_001/frames/`. Before reading pixels for the diagnostic, verify each file's SHA-256 against the source run manifest (`a190ba398f7db0bca7ae0e7b4471050bfd3af581dd38cdfb67c59ee6dcffbfd8`). If any frame is absent or differs, stop.

## Fixed mask

Reuse the already registered M0 RGB rule, without adjusting it: a pixel is foreground exactly when `R > 150`, `G < 125`, and `B < 125` on the decoded 8-bit RGB image. No color-space conversion, morphology, threshold sweep, manual ROI, or temporal association is allowed.

Label foreground using 8-connectivity. The sole candidate in a frame is the largest foreground connected component. The mask is considered sufficiently isolated only if, in **every** frame, (a) that component has at least 300 pixels, matching the existing M0 minimum-mask-pixel criterion, and (b) all other components combined contain fewer than 300 pixels. If either condition fails in any frame, stop the object-track analysis; record component counts/areas as the failure evidence and request independently specified or human-reviewed object masks. Do not tune the color thresholds or select components using motion continuity.

## Measurements if the isolation check passes

For the selected component, report its pixel area, centroid (mean foreground-pixel `x,y`), and half-open bounding box `[xmin,ymin,xmax+1,ymax+1]` in source-image pixels. For each adjacent pair, report centroid displacement `(dx,dy)` and Euclidean distance. Join those 16 rows by pair index to the existing hash-pinned Wan Farneback and RGB-MAE rows; report their previously measured peak pairs for context. These are image-space proxies from one clip, not ground-truth object tracking or sampling-policy evidence.

No model or latent arrays, inference, generation, GPU/cloud compute, dependency installation, or tests are in scope.
