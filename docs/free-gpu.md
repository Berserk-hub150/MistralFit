# Zero-cost remote GPU preparation

**Budget: zero. Personal PC: browser only. GPU benchmark status: not yet run.**

The selected route is a free Kaggle notebook with one P100, if available on the account. Kaggle documents free P100 access subject to GPU quota and availability. The actual assignment must be inspected; no GPU or quota is guaranteed by this repository.

## Start with the small inventory notebook

1. Download [MistralFit-Kaggle-Preflight.ipynb](../notebooks/MistralFit-Kaggle-Preflight.ipynb) using GitHub's **Download raw file** button. It is a small notebook, not the model.
2. Open [a new Kaggle notebook](https://www.kaggle.com/code/new), sign in, and import the downloaded notebook. Complete any account verification Kaggle requires for GPU access.
3. In the notebook's **Settings → Accelerator**, select **GPU P100** if offered and check the remaining free GPU quota. The current study protocol requires a single detected NVIDIA GPU, so **T4 x2 is not supported**.
4. Run the inventory cell. It checks RAM, GPU/VRAM, scratch space and whether the CUDA compiler supports the assigned GPU.
5. Download `mistralfit-free-gpu-preflight.json` through the link shown by the cell, then **stop the Kaggle session** to preserve quota.

The notebook refuses to run outside Kaggle. It performs no model download, CUDA compilation or inference. It invokes only local inventory commands and writes a small JSON file. It contains no provisioning, billing, payment, credential collection or session keep-alive code.

## Why inspect before the large run?

The pinned Small 4 IQ1_S GGUF is **26,882,944,256 bytes**. Its complete checksum and earlier CPU measurements are recorded in the [published experiment](../benchmarks/2026-10-15-small4-iq1s-cpu/README.md). The inventory uses that target size but does not download or claim to inspect its weights.

The notebook conservatively checks combined free RAM/VRAM and scratch headroom. A passing inventory does not prove that every configuration fits. A P100 also needs a compatible CUDA compiler: NVIDIA removed offline compilation for architectures earlier than compute capability 7.5 in CUDA 13. The cell checks the actual compiler's advertised architectures rather than assuming every Kaggle image supports the assigned card.

The free quota must cover the eventual study, which can be long. The inventory does not query account quota, alter platform limits or keep an idle session running. If a required resource is unavailable, retain the `blocked` report and stop; this workflow does not switch to a paid service.

## After a passing inventory

Use the actual JSON to choose the runtime build and workload for that remote session. The [controlled study guide](controlled-study.md) describes the pinned reference autotuner, calibration, 128/256-token holdout comparisons, memory observations and separate output checks. That setup still needs execution and hardware validation; the inventory notebook is not the complete benchmark runner.

Only small reports need to reach the personal PC. Keep model weights and compilation inside the remote session, and export the results before its temporary storage disappears. No results are uploaded or published automatically.

## Official references checked on 2026-10-15

- [Kaggle GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle notebooks](https://www.kaggle.com/docs/notebooks)
- [NVIDIA: Kaggle GPU notebook setup](https://docs.nvidia.com/datascience/deployment/26.10/platforms/kaggle/)
- [NVIDIA: CUDA architecture support](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)
