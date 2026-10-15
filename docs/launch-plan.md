# Launch plan — Luca Magrini

## Positioning hypothesis

Make Mistral Small 4 setup easier to measure on a personal workstation. The first audience to test is developers with substantial host RAM and a consumer NVIDIA GPU who currently choose MoE offload settings by trial and error. A practical first test target is a 64–128 GiB RAM machine with 12–24 GiB VRAM, using an appropriately sized quantization. This is a target for validation, not a supported-hardware guarantee.

MistralFit should earn attention through a reproducible improvement or a useful failure diagnosis. The current package supplies the experiment tooling; the performance result remains to be earned. There is no basis to promise DwarfStar's adoption or a number of stars.

The [first real CPU measurements](../benchmarks/2026-10-15-small4-iq1s-cpu/README.md) are published with checksums and native timings. They also exposed and helped fix an overstated decode-rate calculation. Their short, paging-heavy workload validates integration; it does not yet supply the target-hardware performance advantage described below.

## Before the first broad announcement

1. Put the experimental source on a fresh, clearly named repository. Preserve the MIT license, upstream acknowledgement and status note.
2. Run GitHub CI and fix actual Windows/macOS failures before presenting those platforms as validated.
3. Obtain one full Small 4 run on real target hardware. Run `doctor`, inspect all shards, record full checksums, then tune with a representative workload.
4. Compare the selected candidate with upstream-auto at identical context, quantization, token budget and runtime build. Re-run the full experiment to check reproducibility.
5. Perform a separate correctness/quality check. Investigate different output hashes; do not equate a single matching result with quality preservation.
6. Publish the raw JSON and a readable report with exact hardware and limitations. If no alternative wins, publish that finding and seek another bottleneck.

## The most valuable first milestone

One documented case where a Mistral developer can reproduce a better tradeoff between decode speed and time to first token using a command from the README. A result showing that an apparently sensible setting is slower can also save users time. The claim must match the measured workload.

## A small launch sequence

Start with an experimental GitHub release and invite two or three hardware owners to run the protocol. File concrete, reproducible issues in this project. Where profiling identifies an upstream issue, prepare a focused fix with a minimal reproducer; coordinate with maintainers only when there is substantive evidence to share.

After repeatable results exist, publish one technical write-up showing baseline and candidate settings, raw runs, variability and quality checks. Share it in a small number of relevant Mistral/local-inference communities while following their rules. Invite reports and technical criticism. Do not claim affiliation with Mistral AI.

The source repository is [Berserk-hub150/MistralFit](https://github.com/Berserk-hub150/MistralFit). Public code and CI make the implementation reviewable; performance evidence requires real model experiments. Promotion should follow validated results rather than replace them.

## Next engineering investment

Use measured reports to choose between improving the search space, making backend/device detection reliable, or optimizing a specific expert-placement path in upstream llama.cpp. A native runtime becomes justified when profiling and a reference implementation make its target advantage concrete.
