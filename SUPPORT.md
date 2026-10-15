# Getting help

For MistralFit parser, planner, reporting or process-supervision bugs, [open an issue](https://github.com/Berserk-hub150/MistralFit/issues/new/choose) with a minimal reproducer. Include OS, Python and backend versions, exact command and the error text after removing private information.

If `llama-server` cannot load the same GGUF when invoked directly, investigate the model/backend compatibility first using the upstream project. MistralFit does not implement model kernels or repair incompatible quantizations.

For performance findings, use the experiment issue template and follow [the methodology](docs/benchmark-method.md). Reports where upstream-auto wins, or every candidate fails, are useful too.

MistralFit does not provide Mistral cloud account support, model weights, GPU drivers or hosted inference. No response-time guarantee is offered.
