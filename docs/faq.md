# Frequently asked questions

## Is this a new inference engine?

MistralFit is a configuration-search and measurement tool. llama.cpp runs the model. Native kernels or expert-streaming implementations would be separate future engineering work chosen from profiling evidence.

## Does it make Small 4 fit in any laptop?

No. CPU/GPU placement cannot remove model weights or create physical memory. The planner screens candidates using estimates and can still miss a backend allocation problem. A low-memory machine may only support inspection and planning for Small 4. The memory demo is explicitly synthetic.

## How is this different from llama.cpp's automatic fit?

Automatic fit is the baseline. MistralFit compares a limited set of explicit placements using repeated generation trials at one fixed context. Automatic fit is allowed to win. Results depend on the workload, build and machine.

## Which Mistral 4 is targeted?

Small 4. The original implementation work checked its available weights and existing backend support. Large 4 local execution is not implemented here. Model availability and runtime compatibility should be checked against current primary documentation before use.

## Does it change quantization or model quality?

The search does not rewrite weights or choose a different quantization. Different placements can still produce numerical/output differences. The report exposes output hashes but does not establish language-quality equivalence. Run a separate evaluation before making quality claims.

## Can I install it from PyPI?

No PyPI release is currently advertised. Clone this repository and optionally use `python -m pip install -e .`. GitHub Actions builds distributions that can be downloaded as workflow artifacts after a successful package job.

## Are performance or compatibility claims validated?

See [validation](validation.md), [GitHub Actions](https://github.com/Berserk-hub150/MistralFit/actions) and any raw reports accompanying a claim. Protocol fixtures test orchestration; they are not model execution. A green CI badge does not prove full Small 4 inference performance.

## Does the tool send my prompts anywhere?

Experiments contact the locally started llama-server. There is no telemetry or automatic report upload. Prompts and generated text are not intentionally stored in the portable experiment; their hashes are. Review model names, machine details and errors before sharing reports.
