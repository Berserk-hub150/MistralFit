# Published measurements

These are actual model runs. Automated server fixtures and the synthetic memory demo are excluded.

| Experiment | Setup and workload | What it establishes |
|---|---|---|
| [2026-10-15: Small 4 IQ1_S, constrained CPU](2026-10-15-small4-iq1s-cpu/README.md) | Complete 25.04 GiB GGUF; 8 GiB RAM limit; CPU only; 8-token generations, one warmup and three measured requests per candidate. | Real generation and native timing collection on this setup. No GPU/expert-offload advantage or quality result. |

Each experiment records model and runtime identity, individual samples, controls, limitations and reproduction commands. Do not compare numbers across different quantizations, hardware or workloads as if they were a ranking.

To contribute a run, follow the [benchmark methodology](../docs/benchmark-method.md) and use the repository's benchmark issue template.
