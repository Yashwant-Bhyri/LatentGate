# Kubric Gate 0 fixture

This fixture contains paired 17-frame CPU renders of a red sphere moving in opposite image-plane directions under one fixed orthographic camera. RGB frames and binary object masks are saved separately; `ground_truth.json` records each 3D center, its Blender-projected image center, and the mask-derived box. The commanded target box is 20×20 px to match the rendered sphere mask. Fixture scoring is recomputed from the saved binary masks and projected centers; `fixture_report.json` is a cache checked against those source-derived scores by the validator.

Duplicate mask components are a hard failure. Their ambiguous centers are excluded from valid-path ADE and monotonicity calculations, while the missing-frame penalty remains in the missing-penalized ADE denominator. This fixture validates image-plane center and box scoring from known synthetic truth; it does not measure human annotation reliability or video-diffusion behavior.

The checked run used Kubric source commit `61f2422c84bab75006df33c6989e0b483db3ccfe`, Python 3.11.14, Blender's `bpy` 4.3.0 wheel, TensorFlow 2.16.2, and CPU rendering. The pinned non-Kubric Python dependencies are in `requirements-py311.lock`.

To reproduce in an isolated environment:

```sh
uv python install 3.11.14
uv venv --python 3.11.14 /tmp/latentgate-kubric-env
uv pip install --python /tmp/latentgate-kubric-env/bin/python -r research/kubric_gate0_fixture_v1/requirements-py311.lock
git clone https://github.com/google-research/kubric.git /tmp/latentgate-kubric-src
git -C /tmp/latentgate-kubric-src checkout 61f2422c84bab75006df33c6989e0b483db3ccfe
uv pip install --python /tmp/latentgate-kubric-env/bin/python --no-deps --editable /tmp/latentgate-kubric-src
/tmp/latentgate-kubric-env/bin/python research/render_gate0_kubric_fixture.py
```

The renderer refuses to overwrite this output directory. Move or rename it before regenerating. The fixture validates rendered mask ingestion and known path scoring; it does not measure human annotation agreement or video-diffusion behavior.

The renderer also accepts `--seed <non-negative-integer>` to prepare a deterministic, nuisance-varied mirrored candidate at `research/kubric_gate1_seed_block_<seed>`. It changes the shared path start/offset, travel distance, background level, and Cycles seed while preserving the 160×96 scorer contract. Do not render or distribute the second block until the first v5 pair passes its pooled/per-clip screen, blind worst-frame review, and known-truth diagnostic. See `research/2026-10-12_gate1_second_seed_block_workflow.md` for the pass-record-gated packet command. This second block is prepared but has not been rendered.
