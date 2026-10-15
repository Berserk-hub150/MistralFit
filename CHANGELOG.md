# Changelog

## 0.1.1 — 2026-10-15

- Add an explicit `--no-repack` experiment control, validate backend support, record it and preserve it when launching the measured winner.
- Prepare constrained-memory real-model validation without changing quantization between candidates.
- Estimate reclaimable clean inactive file cache in Linux cgroup memory budgets instead of counting the entire downloaded-model cache as unavailable RAM.

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
