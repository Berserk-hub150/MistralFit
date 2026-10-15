# Architecture

The package uses Python's standard library and controls a separately installed llama.cpp server. No tensors are evaluated by Python.

| Module | Responsibility |
|---|---|
| `gguf.py` | Read v2/v3 metadata and tensor directories, check shard presence/layout, measure tensor spans, compute identities. |
| `hardware.py` | Discover host memory, logical CPUs, NVIDIA free memory and Apple shared-memory status. |
| `planner.py` | Estimate memory and generate baseline, layer-offload, expert-offload and cache candidates. |
| `runtime.py` | Discover backend capabilities, build argv without a shell, supervise processes, consume SSE timings. |
| `tuning.py` | Enforce comparability, collect repeated trials, checkpoint JSON, select the declared objective. |
| `report.py` | Generate a standalone, escaped offline HTML report. |
| `cli.py` | User commands, winner replay, local tool-call smoke check. |

GGUF metadata arrays such as vocabulary entries are skipped instead of loaded into memory. Weight payloads are not read unless full checksums are requested. Split model files are kept local and must be complete. F32/F16 payload spans are checked against their shapes; quantized types receive structural offset/span checks, not a complete format-specific tensor validation.

Explicit placement estimates follow llama.cpp's layer offload convention: last transformer blocks first, with the output layer offloaded when the requested count exceeds the number of blocks. `--n-cpu-moe N` keeps routed experts in the first N blocks on CPU. Unknown tensor naming and backend allocation can reduce estimate accuracy. The backend's actual ability to load and generate is authoritative.

Benchmarks bind to loopback on an available local port, use a fresh ephemeral API key, and require a successful authenticated `/props` probe. The environment passed to the child removes `LLAMA_ARG_*` presets that could silently change the experiment. Standard HTTP/SSE is used; no shell command interpolation, cloud calls, external telemetry or model downloads occur. The small port-selection race can cause a trial to fail; an unrelated server is not intentionally reused.

Test fixtures create tiny structural GGUF files and a local HTTP protocol emulator. They check logic, lifecycle and contracts; they cannot establish actual Mistral compatibility or performance. The CI workflow is provided for Linux, Windows and macOS; the package's initial local test run occurred on Linux/Python 3.12. Running the workflow on GitHub is still required to establish its other matrix results.

Future native work should be chosen after real profiling identifies a bottleneck. A replacement kernel or expert-streaming implementation must have its own numerical tests and benchmark evidence. This package does not claim to implement either.
