# Reproducible experiment recipes

These commands require a real, compatible GGUF and llama-server executable. Paths are placeholders. No result or performance number is supplied by these examples.

## Inspect before loading

```bash
python -m mistralfit doctor --server /opt/llama.cpp/llama-server
python -m mistralfit inspect /models/model-00001-of-00003.gguf --sha256
python -m mistralfit plan /models/model-00001-of-00003.gguf --ctx 4096
```

All model shards must be present. Hashing full weights can take minutes. The planner uses available memory; manual overrides must be available budgets, not the capacity printed on the hardware's box.

## Fixed-workload search

```bash
python -m mistralfit tune /models/model-00001-of-00003.gguf \
  --server /opt/llama.cpp/llama-server \
  --prompt-file examples/workload.txt \
  --ctx 4096 --tokens 256 --repeats 5 --warmups 1 \
  --sha256 --out results/workload-4k.json
```

The planner may exclude candidates that exceed estimates. If none succeeds, the report keeps the reasons and no winner is selected. Investigate compatibility and memory instead of treating a failed trial as a low score.

## Compare a longer context

Repeat the whole command with `--ctx 8192 --out results/workload-8k.json`. Compare candidates **within each run** first. A run at a shorter context is not an equivalent speedup baseline. Repeat entire experiments to check order and thermal effects.

## Apply a response-latency constraint

Add `--max-ttft 5` to select only a candidate whose measured median TTFT is at most five seconds. This does not bound worst-case latency. If every candidate exceeds the limit, no winner is selected.

## Launch a recorded winner

```bash
python -m mistralfit serve /models/model-00001-of-00003.gguf \
  --server /opt/llama.cpp/llama-server \
  --report results/workload-4k.json --dry-run
```

Remove `--dry-run` to start the local server. Review the experiment's output-difference flags and run separate quality checks before adoption. See [methodology](../docs/benchmark-method.md) for publication requirements.
