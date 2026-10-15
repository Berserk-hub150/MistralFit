# Roadmap

The next release should be driven by reproducible reports. These are contribution areas, not delivery-date promises.

Published evidence: [Small 4 IQ1_S in an 8 GiB CPU container](benchmarks/2026-10-15-small4-iq1s-cpu/README.md). This validates real generation on one constrained setup; resident-model and GPU placement comparisons remain open.

| Priority | Milestone | Evidence needed |
|---|---|---|
| 1 | Validate Small 4 on target GPU hardware | Resident-model experiments, full model checksums, runtime build and separate output checks; extend the preliminary CPU evidence. |
| 1 | Establish a reliable upstream-auto baseline | Repeated whole experiments with identical context, weights and workload. |
| 2 | Improve MLA KV estimates | Compare estimates with actual backend allocations for several context lengths. |
| 2 | Map GPU identities explicitly | Validate NVIDIA/llama.cpp device ordering and mixed-device cases. |
| 2 | Validate process/GPU sampling on a workstation | `study` implements observed RSS/NVIDIA VRAM sampling; validate actual CUDA counters and overhead on target hardware. |
| 3 | Add versioned tool-use/quality checks | Small reproducible tasks, clear pass criteria and reference outputs. |
| 3 | Broaden expert-cache search | Backend feature/version checks and measured cache-size sweeps. |
| 4 | Implement a focused upstream or native optimization | Profiling evidence, numerical correctness tests and comparable before/after runs. |

Useful first pull requests can improve error diagnostics, add malformed GGUF cases or document a reproducible backend mismatch. See [CONTRIBUTING.md](CONTRIBUTING.md) before adding hardware claims.
