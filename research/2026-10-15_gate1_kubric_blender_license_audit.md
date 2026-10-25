# Gate 1 Kubric/Blender fixture license audit

**Checked:** 2026-10-15 07:40 UTC  
**Component:** Gate 1 synthetic fixture provenance and the reliability-measurement dependency chain.  
**Scope:** Source terms for the exact pinned Kubric revision and the Blender Python API used by the local fixture renderer. No software was installed and no fixture was rendered.

## Findings

- The local fixture report and README pin Kubric to commit `61f2422c84bab75006df33c6989e0b483db3ccfe`. The run's lockfile names `bpy==4.3.0`, and `research/render_gate0_kubric_fixture.py` directly imports `bpy`, `bpy_extras`, and `mathutils` as well as Kubric. The renderer is therefore a Blender-Python script, not merely an external caller of Kubric.
- The exact pinned Kubric `LICENSE` is Apache License 2.0: [LICENSE at the recorded commit](https://raw.githubusercontent.com/google-research/kubric/61f2422c84bab75006df33c6989e0b483db3ccfe/LICENSE). The corresponding [README at that commit](https://raw.githubusercontent.com/google-research/kubric/61f2422c84bab75006df33c6989e0b483db3ccfe/README.md) describes Kubric as a modular synthetic-video/data-generation pipeline with rich annotations. A root `NOTICE` path returned 404 at this commit; this says nothing about notices elsewhere in the source tree.
- Blender's [official FAQ](https://www.blender.org/support/faq/) says Blender is GPL-licensed and that Python scripts using Blender Python API calls must be made available under GPL when shared or published. The local renderer imports that API, so the Kubric Apache-2.0 license alone is not a sufficient basis to label the entire renderer Apache-only. The official FAQ also distinguishes internal use from external distribution. This report records the upstream statements; it is not a legal determination about the combined work.
- Blender's [official 5.4 manual](https://docs.blender.org/manual/id/dev/getting_started/about/license.html) says the GPL applies to the Blender application, not the artwork created with it; the FAQ likewise describes Blender output as user-created work. This is good evidence that Blender's GPL does not automatically cover rendered artwork, but the inspected Kubric repository README/LICENSE does not state a separate license for generated frames, masks, or truth JSON. Keep those terms recorded as unspecified by the reviewed Kubric sources rather than assuming Apache or GPL automatically applies. No external assets are listed in this fixture's README; the scene uses primitives, but dependency and generated-output terms remain a separate publication check.

## Decision and next step

Keep the fixture and renderer local for this research run; do not publish the Blender-API renderer under an Apache-only notice. This finding does **not** change the v5 annotation protocol, coordinate transform, 0.5-cell p95 process threshold, blinding, or first-block label gate. Local reliability work can continue because nothing is being distributed.

Before any public source release, audit the pinned Blender/bpy and remaining lockfile dependency notices as a set, and decide whether to release the renderer under GPL-compatible terms or separate Blender scene inputs from an external non-`bpy` controller. Separately document the terms chosen for RGB/mask/truth artifacts. The next empirical Gate 1 action is unchanged: obtain two independent complete v5 CSVs, pass pooled and per-clip agreement, complete blind worst-frame review, then run the guarded truth diagnostic.

## Local evidence inspected

- `research/kubric_gate0_fixture_v1/README.md` and `fixture_report.json` for the exact Kubric commit and runtime.
- `research/kubric_gate0_fixture_v1/requirements-py311.lock` for the recorded `bpy==4.3.0` pin.
- `research/render_gate0_kubric_fixture.py:20-26` for direct Blender/Kubric API imports.

No package, dataset, or model was downloaded; no fixture or test ran. GPU use: false; device time: not applicable; peak VRAM: not applicable.
