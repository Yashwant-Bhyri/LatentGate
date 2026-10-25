# M1 Colab recheck — 2026-10-15

## Source refresh

- Uploaded the current local notebook to the designated `y` Chrome profile (`yashwantkumarbhyri@gmail.com`) as a fresh Colab Drive notebook: `1QctKinHmdeFM7Rdj5oDke_UsqdCjxFBb`.
- The visible notebook title is `LatentGate_M1_CogVideoX5B_I2V.ipynb`. Its first code cell contains the no-paid prompt before package installation, and its next cell checks the upload bundle hash before extraction and preflight.
- Local notebook SHA-256: `d989b641ae7801645db61993b71a1ba930f7f54fe5eddb974996dce86f5b7164`.
- Local bundle SHA-256: `bc11a0745f676575bcc4bdbffe8b3e31690da0ad824ec3121e5a6720c6b0da2d`.
- The Oct 13 Drive notebook was left untouched. No cell in the refreshed notebook was executed.

## Runtime and quota observation

- Colab showed the designated account and `You are not subscribed.` The runtime selector offered `T4 GPU`; H100, G4, A100, and L4 were disabled. T4 was selected for this notebook.
- The same account's earlier Resources view showed zero compute units. During this recheck, Resources instead displayed an error fetching compute-unit information; the current value could not be independently re-read.
- Two connection attempts for the selected T4 ended with `Unable to connect to the runtime`. The Active sessions panel showed `Runtime disconnected`.
- No checkout, upgrade, paid accelerator, package install, bundle upload, model download, or model inference occurred. The failed allocation is not evidence of a completed experiment or of current billing status.
- During the final status check, Chrome switched to a different signed-in profile. Browser interaction stopped there to avoid continuing in the wrong account.

## Run-retrieval risk found in source review

- The frozen experiment remains one 50-step seed-2026 baseline followed by the same-seed replay in the same T4 runtime. Its full runtime is unmeasured.
- `run_once()` writes a pass's outputs only after its generation call returns. The local runner now packages a baseline-only checkpoint after baseline artifact writes and before same-seed replay. The checkpoint includes a hash manifest and is explicitly marked replay-pending, incomplete, and not passing.
- The updated notebook validates the checkpoint path and SHA-256, initiates `files.download`, then asks the operator to confirm the ZIP is visible in local Downloads before sending the matching acknowledgement that permits replay. Colab cannot independently inspect or hash the local downloaded file, and browser save completion is not observable to the notebook.
- The runner still performs the exact replay in the same Python process and pipeline object, then writes the combined generation packet and manifest. The notebook archives available run diagnostics after the subprocess exits. A hard cutoff before checkpoint creation or before the operator saves it can still lose the baseline; a cutoff after the saved checkpoint leaves an incomplete baseline artifact, not a passing smoke result. This change does not alter model, prompt/image, seed, step count, captures, equality criteria, or blinded-review gate.

## Local baseline-checkpoint handoff update

- Runner SHA-256: `5aa71897d43fbf8dfbbbdf783d400a8487c4f2c44bc2ea84c892946e2a286310`.
- Notebook SHA-256: `4c0b6e06dff23c85ab1c2d8b8eb01d3a88c01b52b441387184114170d0089b46`.
- Upload bundle SHA-256: `1c47c467d4fb97d194741cb8272a5e44cee43bfb1dc49f358a2a2be9b53e2adb`; the notebook pins this digest and the bundle contains the matching runner/config/fixture/ticket/assets.
- This is a local source update only. The Drive notebook has not yet been refreshed to these hashes, and no Colab runtime or model inference was used.

## Next step

Restore the designated `y` profile in the browser, verify the current no-paid resource state, connect a free T4, and refresh the Drive notebook and bundle to the hashes above. Run the notebook's guarded hardware/package preflight before starting the frozen baseline-plus-exact-replay smoke. Confirm the baseline checkpoint is visible in local Downloads when the notebook prompts; the full smoke still requires replay equality and blinded human review.
