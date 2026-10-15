# Validation — 2026-10-15

Local environment: Linux x86_64, Python 3.12.14, Intel Xeon Platinum 8573C, no GPU detected, 8 GiB cgroup RAM limit and 8-core CPU quota. A complete Small 4 IQ1_S GGUF has generated tokens through the official llama.cpp b11662 CPU runtime. Because its weights exceed RAM, this setup measures a short, heavily memory-constrained CPU workload; it cannot establish representative GPU, resident-model or sustained-generation performance.

| Check | Result |
|---|---|
| `python -m unittest discover -v` | 57 tests, including protocol deadlines, repeated study rounds, external-profile controls, interruption checkpoints and PID-namespace sampling. See CI for each platform's current result. |
| GGUF single/split inspection | Synthetic structural fixtures passed, including missing/mixed shards and malformed/truncated data. |
| CPU/GPU/unified-memory planning | Candidate and budget tests passed on synthetic layouts. |
| HTTP/SSE and process supervision | Real local subprocess and HTTP fixtures passed; these do not run inference. |
| Trial comparability and selection | Wrong context, cached prompts, truncated outputs and invalid token/timing contracts excluded in tests. |
| Report rendering | Generated HTML and untrusted-string escaping tested. |
| Python wheel | Built without dependencies or network using `pip wheel --no-deps --no-build-isolation --no-index .`. |
| Installed-artifact import | Wheel imported and `doctor --json` executed from outside the source directory. |
| Windows PowerShell scripts | Syntax parsed on Windows CI. Interactive setup/publication is not exercised by automated tests. |
| GitHub CI matrix | Linux, Windows and macOS with Python 3.10/3.13, plus a package build/install job. Live results: [GitHub Actions](https://github.com/Berserk-hub150/MistralFit/actions/workflows/test.yml). |
| Real llama.cpp / Small 4 generation | Complete 26,882,944,256-byte IQ1_S GGUF, independently verified full SHA-256; an initial 4-token probe and an 8-token, three-repetition comparison. [Raw data and exact setup](../benchmarks/2026-10-15-small4-iq1s-cpu/README.md). |
| Native decode metric | A real-model run exposed an incorrect first-token interval assumption. Reporting now uses the backend's `predicted_per_second`; original collector files and correction checksums are preserved. |
| Controlled workstation protocol | Implemented: holdout prompts, 128/256-token budgets, balanced repeated process starts, an external autotuner adapter and separate JSON functional checks. End-to-end orchestration tested with HTTP/subprocess fixtures only. [Protocol and run instructions](controlled-study.md). |
| HTTP deadline and final-event correction | Reproduced before/after using actual loopback subprocesses. This is protocol behavior, not model performance. [Evidence and reproduction](../validation/2026-10-15-protocol/README.md). |
| Process-memory sampling | Linux/Windows implementations, lifecycle and unavailable-counter behavior tested. A host procfs namespace mismatch disables child-PID sampling in this environment. Real CUDA process-memory behavior remains unvalidated. |
| GPU/expert-offload speedup, quality, peak memory | Not established or measured by this CPU experiment. |

Performance values in automated protocol tests belong to a deterministic HTTP fixture and are not model/hardware measurements. They are separate from the actual model runs published under `benchmarks/`.

The extended GPU study has not been executed: [current capacity/readiness inspection](../validation/2026-10-15-protocol/gpu-study-readiness.json) records no GPU and an 8 GiB RAM limit. No GPU or multi-prompt Small 4 performance result is inferred from the fixtures.
