# Controlled Small 4 workstation study

**Status: implemented and tested with protocol fixtures; the real GPU study has not run.** The available development environment has 8 GiB of RAM and no NVIDIA GPU. The [published CPU experiment](../benchmarks/2026-10-15-small4-iq1s-cpu/README.md) remains preliminary model evidence.

This protocol compares three **launch configurations** on shared holdout workloads:

1. llama.cpp automatic fitting.
2. The configuration selected by MistralFit on a separate calibration prompt.
3. A configuration exported by [Najafu/llama-autotune](https://github.com/Najafu/llama-autotune), retested through the same server/HTTP client.

The tools use different search workloads and budgets. These measurements cannot establish that one tuning algorithm is better than another. The external tool's own `llama-bench` numbers are retained separately and are never mixed into the holdout results.

## Required machine and files

- One NVIDIA GPU, compatible llama.cpp CUDA binaries (`llama-server` and `llama-bench` from the same build), and local Small 4 GGUF weights.
- Enough available RAM/VRAM for the chosen weights and runtime headroom. Preflight checks are conservative estimates, not proof of residency. No paging opt-in is offered for this study.
- Python 3.12+ for the optional reference autotuner; MistralFit itself still supports Python 3.10+.
- A clean checkout of the reference tool at commit `b03fe68e476181b4acf9f3dec1e80274426e3877` and its dependencies in a separate Python environment.

The scripts do not acquire a GPU, rent services, download weights, or install dependencies. Use the model/runtime licenses and choose a quantization appropriate to your machine. The first protocol supports a single detected NVIDIA GPU; other backends and multiple physical GPUs remain outside this validation.

## Check capacity first

From the MistralFit checkout, this runs no inference and does not need a server binary:

```powershell
py -3 -m mistralfit study "D:\Models\small4.gguf" --preflight --out results\preflight.json
```

Inspect the blockers and HTML report. A missing GPU or insufficient estimated memory returns exit code 2. `--ram-gib` and `--gpu-gib` mean measured **available** budgets, not advertised capacities. A small model fixture is not a valid substitute for Small 4 evidence.

## Prepare the independent reference tool

These are commands for a new reference checkout on Windows PowerShell. Activate a Python 3.12+ installation before creating the environment:

```powershell
git clone https://github.com/Najafu/llama-autotune.git ..\llama-autotune-reference
git -C ..\llama-autotune-reference checkout b03fe68e476181b4acf9f3dec1e80274426e3877
py -3 -m venv ..\autotune-venv
& ..\autotune-venv\Scripts\python.exe -m pip install -e ..\llama-autotune-reference
```

Run the pipeline with your own paths and an empty output directory:

```powershell
py -3 scripts\run_gpu_study.py "D:\Models\small4.gguf" `
  --server "C:\llama.cpp\llama-server.exe" `
  --autotune-root ..\llama-autotune-reference `
  --autotune-python ..\autotune-venv\Scripts\python.exe `
  --threads 8 --out-dir results\small4-workstation-01
```

On Linux the same script works with `python3`, `/path/to/llama-server`, a reference checkout, and the virtual environment's `bin/python`.

The pipeline runs hardware preflight before starting any search, verifies the pinned reference checkout is clean, executes the reference search, records its elapsed time/revision/profile checksum, and starts MistralFit calibration followed by the common holdout measurements. Reference search failures stop the pipeline and preserve the log; they do not become invented scores.

Defaults are substantial: **162 timed requests, 54 warmups and 27 small functional checks**, plus both tools' calibration/search work. Runtime depends on the model and hardware. `--rounds` and `--repeats` can be lowered for a smoke run, but preserve that limitation when publishing it.

## Use an existing reference export

```bash
python -m mistralfit study /models/small4.gguf \
  --server /opt/llama/llama-server \
  --competitor-profile results/autotune-profile.json \
  --ctx 4096 --tokens 128,256 --rounds 3 --repeats 3 \
  --out results/study.json
```

The adapter reads JSON argv data and accepts a documented subset of launch flags; it executes no embedded command. It normalizes the exported **context** to the shared evaluation context and records the original value. It retains supported selected settings such as thread count, batches, GPU layers, mmap and flash attention. Non-f16 KV precision, a different main GPU, multiple slots, multi-GPU split modes and unsupported flags are rejected before measurement. A rejection is not evidence that the external tool is slower. This is a controlled re-evaluation of an export, not a claim that its original training conditions were identical.

Repacking is shared across evaluation configurations; `--no-repack` changes that control explicitly. An external search may have used its own default, which is another reason to keep search and evaluation results separate.

## Workloads, ordering and output checks

The versioned public prompts in [study.py](../mistralfit/study.py) cover a Python LRU implementation, incident diagnosis and document reasoning. Each is measured with 128 and 256 generated tokens, fixed context, f16 KV, seed 42, temperature 0, disabled prompt cache and ignored EOS. The calibration prompt is separate.

One warmup precedes the repeated requests for every workload/budget. Each round starts a fresh server per configuration. The three-round order puts each configuration first, second and third once. Workload order changes by round but is shared across configurations in that round. OS caches, power and thermal history remain coupled; process restarts are not independent thermal resets.

Timing comes from the same native endpoint and client for every configuration. The report contains each sample, native counters, prompt and output hashes, context checks, full model/executable checksums, failures and per-case ratios. Missing or failed rounds cannot receive a complete comparison ratio. There is no global winner or general speedup claim generated by this command.

Separate checks ask for three exact JSON answers with normal EOS behavior. Their pass/fail results and output hashes are recorded. A pass covers those small tasks only. Matching holdout output hashes also do not prove broad quality preservation; add a separately versioned domain evaluation before making that claim.

## Locate the cost before optimizing

For each server start, the JSON records:

| Observation | Interpretation and limit |
|---|---|
| Startup to readiness | Process launch through authenticated readiness; includes orchestration overhead. |
| Native prompt/decode times and streamed TTFT | Distinguishes the larger reported phase for these requests. It does not identify the underlying kernel bottleneck. |
| Observed RSS/VRAM peaks | One server PID, including startup, warmups, requests and functional checks. RSS includes resident mapped pages. Samples are lower bounds, not guaranteed peaks. |
| Linux CPU time, read bytes and major-fault deltas | Recorded around timed requests after warmup. Can expose paging/I/O activity; does not measure disk bandwidth or PCIe expert-transfer latency. |
| Unsupported counters | `null`, with sampling limits. A host procfs/PID namespace mismatch disables server counters to avoid measuring an unrelated process. |

Sampling defaults to a requested one-second interval. NVIDIA process-memory queries have overhead, may extend that interval and may be unavailable under WDDM. Linux uses procfs; Windows supports working-set RSS; missing CPU/I/O counters stay null. Measure with the same sampling policy for every configuration.

If expert transfer is suspected, collect an appropriate GPU profiler trace on the workstation. This package does not infer a CUDA bottleneck from CPU page faults. Implement a targeted change only after observing it, then repeat the same protocol and domain checks before claiming an improvement.

## Publish

Retain `study.json`, its offline HTML, calibration JSON, external training metadata and profile, exact model/runtime provenance, OS/driver and power settings, and the separate domain evaluation. The reference tool's export/log can contain absolute model paths; review those before sharing. MistralFit never uploads the results automatically.

`serve` continues to accept ordinary `tune`/calibration reports, not study reports. A study report can be re-rendered with `python -m mistralfit report results/study.json --out results/study.html`.
