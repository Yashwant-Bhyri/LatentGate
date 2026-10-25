# M1 Drive notebook copy inventory — 2026-10-16

## Scope

Read-only inspection of the exact-title search and both Colab notebooks in the designated Chrome profile `yashwantkumarbhyri@gmail.com`. No notebook was edited or uploaded; no permissions were changed; no cell or runtime was started.

## Copies found

An exact-title Google Drive search for `LatentGate_M1_CogVideoX5B_I2V.ipynb` found two files in `Colab Notebooks`, both owned by the signed-in user:

| Role | Drive ID | Drive / Colab date | Pinned bundle SHA-256 | Saved execution state |
|---|---|---|---|---|
| Repository-designated canonical copy | `1tbDohIHBeHeKF5SD9_9Nkwf22LaJfMmk` | Modified / last edited October 13 | `73af0539fd76957217f4d817560315d7c319321fa2f7bdc2cfe0305b13fc5564` | Colab currently shows `Connect T4`. A package-install output is retained in its first code cell; the bundle/preflight and runner cells have no visible output. This is not evidence of a completed M1 model run or free billing. |
| Duplicate copy | `1QctKinHmdeFM7Rdj5oDke_UsqdCjxFBb` | Modified / last saved October 15 | `bc11a0745f676575bcc4bdbffe8b3e31690da0ad824ec3121e5a6720c6b0da2d` | Colab currently shows `Connect T4`; its code cells have no saved output. |

The repository identifies `1tbDohIHBeHeKF5SD9_9Nkwf22LaJfMmk` as canonical. Keep that ID as the one refresh target; the October 15 file is a second stale copy.

## Comparison with the local artifact

The current local notebook SHA-256 is `c8275000d67e85e369f0834434eb2fa0c52425e417320ef037ed894587cc8c23`, and its pinned bundle SHA-256 is `34777621d1572ca75e40103cd5a222df8da76faecfe921f5edbf9229d8133f29`. The runner SHA-256 is `3ed1802b5cec198d94f7662ebca3e26531a50f8fc7691f9f2b7f01e73925ec0e`. Neither Drive copy pins the current bundle digest.

## Current blocker and exact readback

The blocked step is to replace the contents of the repository-designated Drive file `1tbDohIHBeHeKF5SD9_9Nkwf22LaJfMmk` with the current local `.ipynb`, preserving that Drive ID, then reopen that same ID in Colab and read back the notebook source to confirm `expected_bundle_sha256` is exactly `34777621d1572ca75e40103cd5a222df8da76faecfe921f5edbf9229d8133f29`. The notebook bytes should also be compared with the local file SHA above if a raw readback is available. The separate `1Qct...` duplicate need not become another refresh target.

Before any later preflight or model cell, obtain a fresh no-paid admission check in the designated profile. The same-day project record reports no subscription and zero compute units, but an idle T4 showed no billing class and was disconnected; both notebook pages now show `Connect T4`. This does not establish a free allocation or guarantee future admission. If a guarded preflight is later authorized, it must verify the bundle SHA before extraction and stop unless the UI confirms the no-paid path.

## Evidence limits

This inventory reads Drive titles, ownership, modified dates, Colab notebook source, saved outputs, and the visible connection control. The retained package-install output in the October 13 copy is historical notebook output; it does not identify its billing class or establish that the M1 runner ran. No GPU or cloud computation was initiated during this inspection.

## Canonical refresh attempt — 2026-10-16 06:51 UTC

I selected the October 13 canonical file and opened **File information → Manage versions**. The dialog offers **Upload new version**, which is the in-place path for preserving the Drive ID. Selecting the local notebook at `/Users/yash/Documents/ChatGPT/ROBOTICS/m1/colab/LatentGate_M1_CogVideoX5B_I2V.ipynb` was blocked by the ChatGPT Chrome extension: its `Allow access to file URLs` permission is disabled. The file chooser rejected `setFiles` before any file was transferred.

I left that permission unchanged and stopped. The Manage versions dialog still lists the same current version and versions 1–7, with no new version; no duplicate was created. Therefore the canonical notebook remains stale at bundle hash `73af0539fd76957217f4d817560315d7c319321fa2f7bdc2cfe0305b13fc5564`; the requested readback of the current local notebook hash and bundle pin could not be completed. The current Drive connector is signed in to a different Google account and returns 404 for this file ID, so it cannot serve as an alternate updater. No notebook/runtime action occurred. To use this Chrome path, local-file access would need to be enabled for the extension; I did not change it.
