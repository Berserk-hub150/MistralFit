# Validation — 2026-10-15

Local environment: Linux x86_64, Python 3.12.14, no NVIDIA GPU detected, 8 GiB cgroup RAM limit. This environment cannot establish full Mistral Small 4 inference performance.

| Check | Result |
|---|---|
| `python -m unittest discover -v` | 37 tests, including a reverse-DNS-independent loopback regression. See CI for each platform's current result. |
| GGUF single/split inspection | Synthetic structural fixtures passed, including missing/mixed shards and malformed/truncated data. |
| CPU/GPU/unified-memory planning | Candidate and budget tests passed on synthetic layouts. |
| HTTP/SSE and process supervision | Real local subprocess and HTTP fixtures passed; these do not run inference. |
| Trial comparability and selection | Wrong context, cached prompts, truncated outputs and invalid token/timing contracts excluded in tests. |
| Report rendering | Generated HTML and untrusted-string escaping tested. |
| Python wheel | Built without dependencies or network using `pip wheel --no-deps --no-build-isolation --no-index .`. |
| Installed-artifact import | Wheel imported and `doctor --json` executed from outside the source directory. |
| Windows PowerShell scripts | Syntax parsed on Windows CI. Interactive setup/publication is not exercised by automated tests. |
| GitHub CI matrix | Linux, Windows and macOS with Python 3.10/3.13, plus a package build/install job. Live results: [GitHub Actions](https://github.com/Berserk-hub150/MistralFit/actions/workflows/test.yml). |
| Real llama.cpp / Small 4 generation | Not run. Requires compatible runtime, model files and sufficient hardware. |
| Speedup, quality, peak memory | No measured claims. |

The only performance values used by tests belong to a deterministic HTTP contract fixture and are not model/hardware measurements. No such values are included as product benchmark results.
