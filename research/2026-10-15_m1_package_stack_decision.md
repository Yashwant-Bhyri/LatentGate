# M1 package-stack decision

**Recorded:** 2026-10-15 08:00 UTC  
**Decision:** `LG-M1-STACK-2026-10-15-01`  
**Experiment:** `EXP-M1-COGVIDEOX5B-I2V-SMOKE-001`

## Decision

Pin the M1 Colab stack to `torch==2.11.0+cu130` and `torchvision==0.26.0+cu130`. Keep the other package versions in `m1/run_config.yaml`. The notebook installs these exact wheels from the official PyTorch CUDA 13.0 index, then its isolated runner preflight compares installed package metadata with the config before loading model weights.

The Oct 15 designated-account preflight observed `torch==2.11.0+cu130`, CUDA runtime 13.0, Tesla T4, compute capability 7.5, and driver 580.82.07. PyTorch reported CUDA available. The chosen stack therefore matches the observed runtime rather than replacing its CUDA wheel. PyTorch's official installation matrix lists Torch 2.11.0 with TorchVision 0.26.0 for both CUDA 12.8 and CUDA 13.0. NVIDIA's CUDA 13.0 release notes include Turing, and its compatibility table gives driver 580 as the minimum for CUDA 13.x; the observed driver is 580.82.07.

Sources: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html), and [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## Scope and limits

This versions the runtime package stack only. It does not change the model revision, scheduler, image, prompt, seed, capture plan, replay threshold, or M1 release gates. CUDA visibility and package availability do not establish that CogVideoX-5B-I2V fits or completes on a T4; that remains the purpose of the one baseline plus exact-replay smoke.

At decision time on 2026-10-15, after the baseline-checkpoint handoff patch, the six-file upload bundle had SHA-256 `1c47c467d4fb97d194741cb8272a5e44cee43bfb1dc49f358a2a2be9b53e2adb`, and the local notebook pinned that digest. The runner SHA-256 was `5aa71897d43fbf8dfbbbdf783d400a8487c4f2c44bc2ea84c892946e2a286310`; the notebook SHA-256 was `4c0b6e06dff23c85ab1c2d8b8eb01d3a88c01b52b441387184114170d0089b46`. Those hashes document the 2026-10-15 local state; the earlier bundle hash in the first registry snapshot predates the checkpoint patch and is superseded. The existing Drive notebook remains stale until refreshed and verified. Before any weight load, still require the designated account's explicit no-paid T4 confirmation, current notebook and bundle hashes, T4/CC 7.5, and a passing package preflight. No model, checkpoint, dataset, or paid compute was used for this decision.

**2026-10-16 local-source update:** The M1 ticket was later amended to require a complete score table for all 49 decoded frames. A bundle audit found the old ZIP still held the prior ticket even though its pinned ZIP digest matched the notebook. The local bundle was rebuilt with the current ticket; its new SHA-256 is `5239a42a7846e08ba659096f80e19344f708c30cd3f7411d7071eb71365b3058`, the local notebook SHA-256 is `c94c234733683bc6c43518b1f12fc81f86d8b818cc678e486499ca67dfbd10be`, and the runner is unchanged. This packaging correction does not alter the package pins or experiment settings. The designated Drive copy has not been refreshed.
