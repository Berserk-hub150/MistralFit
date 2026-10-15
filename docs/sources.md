# Primary references

Checked on 2026-10-15. Model availability and backend flags can change; use the installed executable's help and model documentation for your run.

- [antirez/ds4](https://github.com/antirez/ds4): DwarfStar's model-focused local inference approach, quantization, serving and hardware-specific engineering.
- [Mistral Small 4 official model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603): available weights; 119B total / 6.5B active parameters; MoE, reasoning, tool calling and existing inference-library support. Links to GGUF distributions are maintained by the model publisher.
- [Mistral Small 4 documentation](https://docs.mistral.ai/models/mistral-small-4-0-26-03): model overview and context.
- [Mistral Large 4 documentation](https://docs.mistral.ai/models/mistral-large-4-0) and [changelog](https://docs.mistral.ai/resources/changelogs): October 6 public preview and open weights indicated as coming soon at the time checked. Large 4 local execution is not a feature of this release.
- [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): fitting, layer/expert placement, optional expert cache, single-device mode, props/context checks, tokenization, streaming completion timings and tool calling.
- [llama.cpp Mistral Small 4 support PR](https://github.com/ggml-org/llama.cpp/pull/20649): existing model implementation; MistralFit does not claim first model support.
- [llama.cpp build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md): external backend installation.

The package does not bundle or copy implementation code, weights, benchmark tables, logos or screenshots from these sources.
