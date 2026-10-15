# Changelog

## 0.2.0 — 2026-10-15

- Add `study`: holdout workloads with 128/256-token budgets, separate calibration, three selected configurations, balanced process-start rounds and per-case native timing/hash results.
- Import a validated Najafu/llama-autotune launch profile and record context normalization; reject different KV precision, unsupported flags and multi-GPU settings.
- Add a pinned-reference orchestration script, full model/executable checksums and external-search provenance.
- Sample server RSS and NVIDIA process VRAM when supported; collect Linux CPU/I/O/page-fault deltas and separate startup, prompt and decode observations. Missing counters remain null.
- Enforce completion deadlines through slow response headers and partial SSE lines, and return on the final event without waiting for connection closure.
- Detect procfs PID namespace mismatches before sampling, avoiding attribution to an unrelated host process.
- Add three separate JSON functional checks. These do not establish general quality parity.

The workstation GPU study has **not** been executed in the available 8 GiB/no-GPU environment. The previous constrained CPU data remains the only real Small 4 experiment. Protocol regression timings are separate from model measurements.

## 0.1.1 — 2026-10-15

- Add an explicit `--no-repack` experiment control, validate backend support, record it and preserve it when launching the measured winner.
- Prepare constrained-memory real-model validation without changing quantization between candidates.
- Estimate reclaimable clean inactive file cache in Linux cgroup memory budgets instead of counting the entire downloaded-model cache as unavailable RAM.
- Use the backend-reported decode rate, preserving llama.cpp's first-token/interval convention instead of recomputing an inflated N/time value on short generations.

## 0.1.0 — 2026-10-15

Initial experimental source release.

- Inspect GGUF v2/v3 metadata and tensor directories without loading weights; handle complete split-model sets.
- Discover host RAM and NVIDIA memory; generate CPU/GPU layer and MoE expert placement candidates.
- Compare upstream automatic fitting with explicit placements, and optional expert caching when the backend supports it.
- Enforce fixed context and generation budget; collect streaming TTFT and backend decode timing.
- Preserve individual trials, failures, exclusions, model identity and settings in JSON and offline HTML.
- Replay the selected configuration and offer a local API tool-call smoke check.
- Include Windows entry scripts, a dependency-free automated test suite and package/CI workflows.
- Include startup health diagnostics and a loopback protocol fixture that avoids reverse-DNS dependencies.

Full Mistral Small 4 inference and performance await real hardware validation. This release contains no measured speedup or model-quality claim.
