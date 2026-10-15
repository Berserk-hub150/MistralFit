# Roadmap

The next release should be driven by reproducible reports. These are contribution areas, not delivery-date promises.

| Priority | Milestone | Evidence needed |
|---|---|---|
| 1 | Validate full Small 4 inference on real hardware | Complete raw experiment, model checksums, runtime build and separate output checks. |
| 1 | Establish a reliable upstream-auto baseline | Repeated whole experiments with identical context, weights and workload. |
| 2 | Improve MLA KV estimates | Compare estimates with actual backend allocations for several context lengths. |
| 2 | Map GPU identities explicitly | Validate NVIDIA/llama.cpp device ordering and mixed-device cases. |
| 2 | Sample process/GPU memory | Record sampling interval and distinguish sampled peaks from guaranteed maxima. |
| 3 | Add versioned tool-use/quality checks | Small reproducible tasks, clear pass criteria and reference outputs. |
| 3 | Broaden expert-cache search | Backend feature/version checks and measured cache-size sweeps. |
| 4 | Implement a focused upstream or native optimization | Profiling evidence, numerical correctness tests and comparable before/after runs. |

Useful first pull requests can improve error diagnostics, add malformed GGUF cases or document a reproducible backend mismatch. See [CONTRIBUTING.md](CONTRIBUTING.md) before adding hardware claims.
