# DiTraj admission and preliminary-packet remediation

**Recorded:** 2026-10-17 06:10 UTC  
**Scope:** local source/notebook remediation only. No notebook cell, runner, model, CUDA call, install, test, compile, upload, or cloud runtime was executed.

## Admission design

Added `ditraj_admission.py`, a standard-library-only fail-closed receipt validator. Each accepted receipt must identify the designated Kaggle account (`bvyashwantkumar`), a UTC check time no more than 15 minutes old (with a two-minute future-skew allowance), exact local notebook and runner SHA-256 values, the exact intended arm, affirmative free-T4 admission, an explicit statement that no paid/upgrade prompt was seen, a provenance description, and the SHA-256 of the reviewed evidence. The direct runners independently re-hash their executing source before importing NumPy, Torch, Diffusers, patching upstream source, probing CUDA, or loading weights. Their run records now retain the account, UTC check, both source hashes, intended arm, evidence hash, provenance, and the explicit statement that no Kaggle quota API was queried; quota-hour placeholders are strings saying “not queried,” not nulls.

The current notebook asks for a fresh `setup_only` receipt before clone/install, another receipt before CUDA preflight, and a run-specific receipt before each enabled generation arm. The paired, feedback, and legacy direct CLIs all require both a receipt file and notebook SHA; the receipt must match the CLI's intended arm, and the runner bytes must match the receipt. The stale four-cell notebook has every code cell replaced with an immediate retirement error so running cells out of order cannot reach its old install, CUDA, model, or archive paths. `ditraj_admission_receipt_template.json` is intentionally invalid until a human completes it.

## Honest boundary of the manual receipt

No Kaggle billing/quota API is available to these sources. The receipt is a strict manual operator attestation, not provider-signed proof: the kernel cannot reopen/re-hash the staged `.ipynb`, independently authenticate the account UI, or inspect the external evidence named by `evidence_sha256`. The notebook therefore requires the exact local notebook pin to be transcribed from the reviewed local registry. Run metadata marks that distinction explicitly (`manual_operator_attestation_no_provider_quota_api`, `operator_verified_local_pin_not_kernel_rehashed`, and `provider_quota_api_queried: false`). A fabricated receipt could still lie about what the UI showed. No valid fresh receipt exists in this workspace, Kaggle admission remains unknown, and the current known Colab profile is unsubscribed with zero units and is not a substitute. The code is default-closed, but actual execution remains blocked until an authorized operator can review fresh designated-account evidence and supply the corresponding receipt; this remediation is not evidence of free admission.

## Preliminary annotation artifacts

Removed automatic `blinded_annotation_package.zip` creation from the paired and legacy runners. The current notebook no longer zips or exports the smoke directory. Runner reports call the annotation directory `preliminary_not_scoreable`, set `scoreable: false` and `exportable: false`, and enumerate the missing registered per-frame SHA-256/dimension manifest and separate background-stability measure. The feedback smoke report carries the same non-scoreable/non-exportable status. No thresholds, control arms, paths, or metrics were changed. Do not give this preliminary directory or any manually made archive to raters.

## Static source record

- Canonical notebook, 12 cells, all code cells with null execution counts and empty outputs: SHA-256 `fc066dc732ca1213487181fe38f7b2a5d54038b39aa14d16f8569c8548c56e15`.
- Retired notebook: SHA-256 `ebcebad652cb69daed5f4ccef3dcdccdc95a2cdf97643860bd7040b67079720f`.
- Admission validator: SHA-256 `b40f3d230dcdf9e47ebbedfe853e6d26090933db199959b96c79a4612d6193d9`.
- Receipt template: SHA-256 `48c80e420932aec436ee832bee311c4a8e39415431d364cbb4334111d394fc11`.
- Paired runner: SHA-256 `c00e1470c3e67da3573f61614f90e3ec9ab525135040ca3928a4bc3db4432984`.
- Feedback runner: SHA-256 `8b00b8c413b6e8dcfd4b61ceaf206ec4a3f45351557c3b21aa52d06c75dac25f`.
- Legacy matched runner: SHA-256 `fcd286cc91ee1a8122374a800464f38c7ea9d36235ec022fa085c6eb0e931bd7`.

Static source/JSON review confirms the old notebook no longer contains an executable path; the canonical notebook still has its six matched mirrored-path/Base/SG-only/full-DiTraj arms and its 30-step and feedback switches remain false. `jq` confirmed the canonical notebook is valid JSON metadata, remains 12 cells, and every code cell has null execution count and no outputs. Grep review found no remaining `make_archive`, preliminary ZIP creation, old quota Boolean, or null quota-hour placeholder in these entry paths. This was not a syntax compile, test run, inference, or runtime validation.

## Decision and next falsifiable gate

Keep all execution blocked. The next requirement is a manually reviewed, fresh receipt from the designated Kaggle account tied to the exact local notebook pin and one explicit arm; on a future authorized source stage, confirm the embedded runner hash and repeat the receipt check immediately before setup and run. If the account UI cannot provide unambiguous free-T4/no-paid evidence, do not run. Independently, packet export remains disabled until a registered per-frame hash/dimension manifest and separate background-stability measure exist. Gate 1 still awaits two independent complete v5 human returns; no policy-efficacy claim is made.
