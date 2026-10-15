# Contributing

Start with `python -m unittest discover -v`. No external Python dependencies or model downloads are needed for these tests. New parser cases should use tiny generated files; do not commit weights or copied proprietary data.

For a backend failure, include the executable version, relevant capabilities, GGUF architecture/quantization, platform, exact command and reproducible symptom. For a performance result, follow `docs/benchmark-method.md`; attach the raw experiment and identify estimates separately from measurements.

The most useful first issues are:

- Validate a full Small 4 run on a Windows NVIDIA machine and provide both successful and excluded trials.
- Compare actual MLA KV allocations with the planner's expanded estimate and propose an evidence-backed formula.
- Add explicit mapping between discovered NVIDIA devices and llama.cpp device identifiers.
- Add a measured memory sampler without confusing current usage with a true sampled peak.
- Add versioned tool-use/quality checks that run only when a real model is available.

Do not submit fabricated timings, hardware claims, copied benchmark results, or promotional-only changes. Small, reviewable changes with a reproducer are welcome.
