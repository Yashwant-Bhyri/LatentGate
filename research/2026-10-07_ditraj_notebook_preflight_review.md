# DiTraj notebook preflight and staging review

**Checked:** 2026-10-07 12:09 UTC  
**Scope:** Review the local paired-path notebook and stage it to the designated Kaggle draft without starting a runtime.

## Local notebook review

- `research/ditraj_paired_path_pilot_v1.ipynb` is nbformat 4.5 with 12 cells, no execution counts, and no saved outputs. Current SHA-256: `2e697fb5942472a0e6dd71a6c67975345f9fb7970aae08609221cd8940583e4e`.
- The six-case pilot retains mirrored left/right paths, a direction-neutral prompt, shared initial noise, and Base / foreground-background / full DiTraj arms. The six-case one-step smoke remains the intended initial generation check; the 30-step run flag remains `False`.
- Review found the later three-arm feedback smoke command was unconditional: `Run All` could reach it before the open-loop result, observer, actuator, and quota gates passed. The feedback result-viewer cell also assumed outputs existed. I changed the local notebook so a manual `FREE_KAGGLE_GPU_QUOTA_CONFIRMED = True` is required before setup, `RUN_FEEDBACK_SMOKE` defaults to `False`, the feedback command and result viewer are gated, and the overview tells the operator to run cells individually. No notebook cells were executed.
- A static pass found no absolute user-home paths or credential-like tokens in the notebook. This is source review only; the notebook has not been staged remotely.

## Designated Kaggle state and blocker

- The open draft is `https://www.kaggle.com/code/bvyashwantkumar/ditraj-wan-1-3b-pilot-research/edit`; the visible account handle matches the designated `bvyashwantkumar` lane. The remote draft remains four cells and `Draft Session off`.
- The fresh Settings → Accelerator menu lists `GPU T4 x2` and `TPU v5e-8`, but does not expose quota/admission. Free T4 availability remains unknown; no runtime was started.
- File → Import Notebook opened its local file chooser, but Chrome blocked file selection because the ChatGPT extension's persistent **Allow access to file URLs** permission is disabled. I did not change that permission, select a file, or upload anything. The local notebook is ready to stage when this permission is enabled or an approved file-transfer path is available.

## Decision and next gate

Keep the revised source local, preserve the remote draft unchanged, and do not start cloud compute. After the file-transfer blocker is cleared, import this exact hash into the designated draft and verify the remote 12-cell notebook plus all three default gates. Before any runtime, obtain a fresh, affirmative free-T4/quota check; the current selector alone is insufficient.

Gate 1 remains open: the latest packet scan still found 0/34 labeled rows in each synthetic rater CSV and 0/17 in each M0 convenience CSV. No observer or policy efficacy is established.
