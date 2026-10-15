# HTTP protocol regression evidence

**This is not an inference benchmark. No model or GPU was used.** These timings belong to a deliberately delayed loopback HTTP fixture and must never be presented as Small 4 token-generation speed.

The client previously checked its deadline after a complete line arrived. A backend slowly sending response headers or a partial line could keep the socket active beyond the configured wall-clock limit. It also waited for stream closure after receiving valid final timing data.

The fix uses a monotonic request deadline with transport shutdown, covering headers and partial SSE lines, and returns after a valid final event. Cleanup stops the timer and closes the connection. The existing token/count/cache/timing checks remain enforced.

## Observations

Three subprocess/HTTP requests per case, same fixture before and after. The fixture emits one byte every 10 ms in the slow cases and waits 600 ms after its final event in the lingering case. Native fixture counters are synthetic and are deliberately omitted from this report.

| Client | Fixture | Configured deadline (s) | Median observed request wall time (s) |
|---|---|---:|---:|
| before | slow-headers | 0.08 | 0.5350 |
| before | slow-line | 0.08 | 0.6796 |
| before | linger-after-final | 2.00 | 0.6112 |
| after | slow-headers | 0.08 | 0.0806 |
| after | slow-line | 0.08 | 0.0804 |
| after | linger-after-final | 2.00 | 0.0109 |

The lingering case completed with the same four fixture tokens and output hash before/after. This demonstrates the protocol correction under these conditions, not a general application latency or model-throughput improvement. Scheduling and machine load can change the exact timings.

## Reproduce and audit

- [Raw timing observations](protocol-regression.json) include every sample and hashes of the baseline client, fixed client and fixture source.
- The measured fixed-client source is preserved in [commit 63b4654](https://github.com/Berserk-hub150/MistralFit/commit/63b465431743cb65652ffe79b48755feb7a621d3). Later high-resolution clock changes do not rewrite these observations; a current-checkout rerun records its own source hash.
- [Reproduction script](reproduce.py) loads the baseline runtime source from commit `52554f55b6e87755a6f91f5385b0393584a1d61e` and compares it with the current checkout. Run from a normal Git checkout containing that historical commit:

```bash
python validation/2026-10-15-protocol/reproduce.py --out results/protocol-regression.json
```

If using a shallow checkout, fetch the baseline commit first. Future client/fixture changes produce a different source hash and a new observation, rather than reproducing an identical timing file.

## GPU study status

The [readiness inspection](gpu-study-readiness.json) records actual development-environment capacity and target-model identity taken from the earlier published CPU experiment. It performs no new inference. The machine has an 8 GiB RAM limit and no GPU; the complete target GGUF is approximately 25 GiB. The [workstation study](../../docs/controlled-study.md) therefore remains unexecuted on GPU.
