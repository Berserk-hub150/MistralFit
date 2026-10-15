# Benchmark method

## Experimental unit

One candidate is one explicit tensor-placement configuration of the same llama-server executable and GGUF shard set. The initial release searches a small set of candidates rather than every possible GPU layer count, CPU expert count or cache size. Candidate order is shuffled deterministically and saved.

Every candidate starts a fresh local server process. After the server is healthy, MistralFit checks the reported effective slot context. A candidate that silently lowers context is excluded. Prompt tokenization is checked before generation, so the prompt and fixed generation budget must fit together.

Default run: one warmup request and three measured requests, 128 generated tokens, a 4096-token context, f16 KV data types, temperature zero and seed 42. The token budget ignores EOS. This is a throughput workload rather than representative free-form conversational behavior. Use a longer output and representative local workload for a stronger experiment.

Weight repacking is left at its backend default unless `--no-repack` is requested. The setting is recorded and applied to every candidate and to winner replay. A constrained-memory run that disables repacking is a different setup from an ordinary resident-memory run and must be identified as such.

## Baseline

`upstream-auto` uses llama.cpp's own automatic fitting, with the same explicitly requested context, batch sizes, KV types, threads, prompt and token budget as the other profiles. It receives no invented penalty or handicap. If it is the fastest passing candidate, it wins.

Explicit candidates disable automatic fitting and set their own GPU-layer and expert-placement arguments. All candidates are single-slot, single-device experiments. Expert caching is tried only if the executable advertises its flag. CPU-only experiments explicitly disable device offload.

## Timing

The client starts a monotonic clock immediately before making the local native `/completion` request. TTFT is measured when the first SSE event containing generated content or token IDs arrives. Decode speed comes from the backend's `predicted_n / predicted_ms`; it is not estimated from text length. TTFT includes client/HTTP overhead and the processing needed before the first emitted event.

`cache_prompt` is false and the prompt-cache RAM allocation is disabled when the installed build supports it. A trial reporting nonzero `timings.cache_n` is excluded. Unsupported flags, loading failures, completion timeouts, truncated generations, missing timing fields and wrong generated token counts all remain visible in the report. Failed candidates cannot win. Ctrl+C stops the child and preserves the partial JSON experiment.

## Selection and interpretation

Selection maximizes median decode tokens/second among passing candidates, optionally requiring their median TTFT to satisfy `--max-ttft`. Exact throughput ties are broken by median TTFT. This selects a winner for the tested prompt and candidates; it is not a global optimum. When both upstream-auto and a winner succeed, the report divides their measured medians. A 1.0 ratio is a valid result.

Min/max throughput and relative median absolute deviation expose variability. Repeated requests reuse the loaded process and warm OS/model state, so they are not independent thermal or cold-start replications. Repeat whole experiments, change candidate order in the source when studying order effects, keep background load low, and record power/thermal policy separately. No statistical significance is asserted.

Generated text and prompts are not saved in the experiment. Output hashes can show that placement changed deterministic results. A matching hash on one workload does not establish language quality. A separate versioned evaluation is required before claiming quality preservation or agent reliability.

## Model and runtime identity

The structural fingerprint hashes the GGUF header bytes and file sizes for every shard. It detects metadata/layout changes quickly but intentionally does not hash weight payloads. Changing a weight without changing its layout can evade this fingerprint. `--sha256` explicitly reads and hashes every file, and those hashes are verified by `serve` when present. Use them in published experiments.

The runtime's `--version` output is recorded and compared on launch. Version strings are not binary checksums. Full executable integrity and reproducible builds are outside v0.1.

## Memory

Weight storage is measured from GGUF tensor offsets. Those spans include padding and do not equal actual resident memory. Host and GPU estimates use tensor placement, an expanded f16 KV calculation when metadata is available, and scratch headroom. MLA and backend allocation can differ substantially; incomplete metadata falls back to a documented heuristic. The planner can reject feasible candidates conservatively or miss an OOM. It never certifies a fit.

Available RAM and free GPU memory are sampled before the experiment, not monitored throughout it. Apple Silicon CPU/GPU allocations share one pool and are not added together. Peak RSS, peak VRAM, actual offloaded tensor allocation, disk bandwidth and expert-cache hit rates are not measured in this release.

## Publication checklist

Include model source and quantization, full shard checksums, runtime build/commit, OS/backend/driver, CPU/GPU/RAM, power policy, fixed context, prompt source/hash, tokens, repeats, warmups, complete raw JSON, exclusions and a separate quality check. Review the report for private model names or machine details. Do not publish the protocol fixture's timings as model results. Do not compare results using different models or shorter contexts as if they were the same experiment.
