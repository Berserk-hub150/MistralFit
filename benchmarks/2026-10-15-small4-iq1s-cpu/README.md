# Mistral Small 4 IQ1_S: first constrained-memory CPU measurements

These are real generations from the complete Mistral Small 4 model through an official llama.cpp CPU build. They validate the MistralFit integration on one specific setup. **They do not demonstrate a GPU/expert-offload advantage or establish general model speed or quality.**

The 25.04 GiB GGUF is larger than this container's 8 GiB RAM cap. The OS must use file-backed memory mapping and reclaim model pages. This is a slow, I/O-constrained setup rather than a recommended Small 4 workstation configuration.

## Measured comparison

One fixed public prompt, 512-token context, 8 generated tokens per request, three measured requests after one warmup per candidate, 8 CPU threads, f16 KV types, temperature 0, seed 42, prompt cache disabled and weight repacking disabled. Candidate order is recorded in the JSON. Eight generated tokens is a short microbenchmark; these results do not describe sustained long generation or coding-agent performance.

| Candidate | Status | Median decode tok/s | Median TTFT (s) | Decode min–max (tok/s) |
|---|---|---:|---:|---:|
| cpu-only | ok | 0.1040 | 48.406 | 0.0887–0.1402 |
| upstream-auto | ok | 0.1417 | 37.794 | 0.1392–0.2367 |

The selected candidate is **upstream-auto**. MistralFit found no faster alternative in this two-profile run.

Both configurations use the CPU on this machine. Differences are subject to disk/cache state and shared-container load. Three requests in one process are not independent thermal or cold-start runs; no statistical significance or acceleration claim is made.

## Files

- [Comparison JSON](comparison.json): individual samples, native timings, hashes, configuration and exclusions.
- [Comparison HTML](comparison.html): standalone offline report; download and open it locally.
- [Original comparison collector output](comparison.collector.json): preserved before the reporting correction described below.
- [First 4-token inference probe](probe.json) and [its collector output](probe.collector.json): integration smoke test, not a throughput study.
- [Public prompt](prompt.txt) and [full provenance](provenance.json).
- [Report rebuild script](rebuild_reports.py): reproduces the corrected JSON and HTML from the unchanged collector files, without model weights or inference.

The earlier inference probe used four generated tokens, one request and no warmup. Its collector's `full_model_sha256` field is null because the checksum had been verified separately before launching it; that verification is recorded in the shared provenance. The subsequent comparison also carries the checksum in its own collector JSON.

## Exact setup

- CPU: Intel Xeon Platinum 8573C; 9 visible logical CPUs and an 8-core cgroup quota.
- Memory: 8,589,934,592-byte cgroup limit. This is a limit, not a measurement of peak RSS.
- GPU: none detected; an official CPU-only runtime asset was used.
- Model: `bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF`, revision `f569d04efea9546de10ec040819908fa4e8f1c92`, file `mistralai_Mistral-Small-4-119B-2603-IQ1_S.gguf`.
- Model bytes: 26,882,944,256. Entire-file SHA-256 verified: `e6363fa561991421a01c665e18b573f3affdef8c1eb039e63da8fcfcb8f826d5`.
- Runtime: [llama.cpp b11662](https://github.com/ggml-org/llama.cpp/releases/tag/b11662), commit `c0cb53ac4`, GNU 11.4 Linux x86_64 CPU build. Runtime archive SHA-256 is recorded in provenance.
- Collection code: [8bb51d0](https://github.com/Berserk-hub150/MistralFit/commit/8bb51d0626bb51bb8532014afa34bf2d34f246a3).
- Metric reporting code: [c844bbf5406225e07c8792ff883a45a827aa5d9e](https://github.com/Berserk-hub150/MistralFit/commit/c844bbf5406225e07c8792ff883a45a827aa5d9e).

## Reporting correction discovered with the real model

The initial collector computed `predicted_n / predicted_ms`. In this backend build, N emitted tokens have N-1 timed decode intervals because the first token is accounted for separately. That formula overstated the short-generation rate. The client now uses `backend_timings.predicted_per_second` directly.

The displayed rates above are the original native backend values. Native timing counters, token counts, output hashes, TTFT and request durations were not changed. Original collector JSON is retained, with a checksum linked from each corrected report. The four-token probe's correct native decode rate was 0.1021 tok/s, rather than the old derived 0.1362 tok/s.

To audit the correction from a repository checkout, run `python benchmarks/2026-10-15-small4-iq1s-cpu/rebuild_reports.py`. It reads the preserved collector files and provenance, checks the public prompt hash and regenerates the derived JSON/HTML. It does not rerun the model.

## Reproduce

Download the exact model revision/file, verify its checksum, and obtain the exact CPU runtime release. From this repository, substitute your local paths:

```bash
python -m mistralfit tune /models/mistralai_Mistral-Small-4-119B-2603-IQ1_S.gguf \
  --server /opt/llama-b11662/llama-server \
  --profiles upstream-auto,cpu-only --ctx 512 --tokens 8 \
  --repeats 3 --warmups 1 --threads 8 --no-repack --allow-paging \
  --timeout 600 --startup-timeout 600 --sha256 --out results/small4-cpu.json
```

Use the [same prompt](prompt.txt), which is also this version's default prompt. Reproducing the cgroup limit is necessary to compare the memory-constrained environment, and storage/host behavior can still change results. `--allow-paging` is an explicit opt-in; the planner does not claim that all weights fit in resident RAM.

## Next validation

Test a resident model and CPU/GPU expert placement on an appropriate workstation, use substantially longer generations and several prompts, repeat full experiments, measure memory with a defined sampler, and evaluate outputs separately. A meaningful performance claim needs that evidence.
