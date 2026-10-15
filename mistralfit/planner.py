"""Generate placement candidates; estimates are screening hints, never measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .gguf import Model
from .hardware import GIB, Hardware


@dataclass
class Profile:
    name: str
    gpu_layers: int | None
    cpu_moe_layers: int = 0
    expert_cache_mib: int = 0
    estimated_gpu_bytes: int | None = None
    estimated_ram_bytes: int | None = None
    eligible: bool = True
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def kv_estimate(model: Model, context: int) -> tuple[int, str]:
    md, arch = model.metadata, model.architecture
    heads = md.get(arch + ".attention.head_count")
    kvheads = md.get(arch + ".attention.head_count_kv", heads)
    embedding = md.get(arch + ".embedding_length")
    key = md.get(arch + ".attention.key_length")
    value = md.get(arch + ".attention.value_length", key)
    if key is None and isinstance(heads, int) and heads and isinstance(embedding, int):
        key = value = embedding // heads
    if all(isinstance(x, int) and x > 0 for x in (kvheads, key, value, model.blocks)):
        return context * model.blocks * kvheads * (key + value) * 2, "Expanded f16 KV estimate; MLA/SWA backend allocation may differ."
    return context * max(1, model.blocks) * 1024, "KV metadata incomplete: heuristic of 1 KiB per token per block."


def budgets(hw: Hardware, ram_gib: float | None, gpu_gib: float | None, gpu_index: int):
    # Overrides are explicit AVAILABLE budgets, not advertised physical capacities.
    ram = int(ram_gib * GIB) if ram_gib is not None else hw.ram_available
    gpu = int(gpu_gib * GIB) if gpu_gib is not None else next(
        (g.free_bytes for g in hw.gpus if g.index == gpu_index), 0)
    if hw.unified_memory and gpu_gib is None:
        gpu = ram
    return ram, gpu


def placement(model: Model, layers: int, cpu_moe_layers: int) -> int:
    """llama.cpp places the last N transformer blocks on GPU, then output."""
    first_gpu = max(0, model.blocks - layers)
    result = 0
    for tensor in model.tensors:
        if tensor.block is not None:
            on_gpu = layers > 0 and tensor.block >= first_gpu
            on_cpu_expert = tensor.routed_expert and tensor.block < cpu_moe_layers
            if on_gpu and not on_cpu_expert:
                result += tensor.stored_bytes
        elif layers > model.blocks and tensor.name.startswith("output"):
            result += tensor.stored_bytes
    return result


def plan(model: Model, hw: Hardware, *, context: int = 4096, ram_gib: float | None = None,
         gpu_gib: float | None = None, gpu_index: int = 0, allow_paging: bool = False) -> dict:
    if context < 256:
        raise ValueError("Context must be at least 256")
    ram, gpu = budgets(hw, ram_gib, gpu_gib, gpu_index)
    kv, kv_note = kv_estimate(model, context)
    scratch = max(512 * 1024 ** 2, int(model.weight_bytes * 0.015))
    gpu_reserve = 768 * 1024 ** 2
    profiles = []

    def add(name, layers, cpu_moe=0, cache_mib=0):
        if layers is None:
            # Baseline is the installed upstream automatic fit; placement is unknown beforehand.
            minimum_host = max(0, model.weight_bytes - max(0, gpu - gpu_reserve - kv)) + scratch
            host_need = model.weight_bytes + kv + scratch if hw.unified_memory else minimum_host
            p = Profile(name, None, estimated_ram_bytes=host_need)
        else:
            gpu_weights = placement(model, layers, cpu_moe)
            gpu_need = gpu_weights + cache_mib * 1024 ** 2 + (kv + gpu_reserve if layers > 0 else 0)
            host_need = model.weight_bytes - gpu_weights + scratch + (kv if layers == 0 else 0)
            if hw.unified_memory:
                # CPU and Metal refer to the same pool. Cache candidates are omitted below.
                host_need = model.weight_bytes + kv + scratch + (gpu_reserve if layers else 0)
            p = Profile(name, layers, cpu_moe, cache_mib, gpu_need, host_need)
            if layers > 0 and gpu <= 0:
                p.eligible, p.reason = False, "GPU budget unknown or unavailable."
            elif layers > 0 and not hw.unified_memory and gpu_need > gpu:
                p.eligible, p.reason = False, "Estimated GPU requirement exceeds the available GPU budget."
        if ram <= 0:
            p.eligible, p.reason = False, "Available RAM unknown. Supply --ram-gib with an available-memory budget."
        elif host_need > max(0, ram - 512 * 1024 ** 2) and not allow_paging:
            p.eligible, p.reason = False, "Estimated host/shared RAM requirement exceeds budget; --allow-paging opts in."
        profiles.append(p)

    add("upstream-auto", None)
    if gpu > 0:
        add("gpu-all", model.blocks + 1)
        fitting = [n for n in range(1, model.blocks + 1)
                   if placement(model, n, 0) + kv + gpu_reserve <= gpu]
        if fitting:
            add("gpu-layer-fit", max(fitting))
            if max(fitting) >= 4:
                add("gpu-layer-margin", max(1, max(fitting) - max(1, model.blocks // 8)))
        if model.expert_bytes:
            add("cpu-experts", model.blocks + 1, model.blocks)
            fitting_experts = [n for n in range(model.blocks + 1)
                               if placement(model, model.blocks + 1, n) + kv + gpu_reserve <= gpu]
            if fitting_experts and min(fitting_experts) not in (0, model.blocks):
                add("hybrid-experts", model.blocks + 1, min(fitting_experts))
            if not hw.unified_memory:
                residual = gpu - placement(model, model.blocks + 1, model.blocks) - kv - gpu_reserve
                cache = int(max(0, residual * 0.65) / (1024 ** 2))
                if cache >= 256:
                    add("cached-experts", model.blocks + 1, model.blocks, cache)
    add("cpu-only", 0)
    notes = [kv_note, "Weights are measured from tensor spans, including padding; KV and scratch are estimates.",
             "These are candidate configurations, not proof of compatibility, speed, or memory fit.",
             "No quantization is changed. One context, prompt and generation budget are used across candidates."]
    if hw.unified_memory:
        notes.append("Unified memory is counted once; moving tensors to CPU does not expand its capacity.")
    if not model.expert_bytes:
        notes.append("No routed-expert tensors recognized; expert placement candidates omitted.")
    return {"schema_version": 1, "model": model.summary(), "hardware": hw.to_dict(),
            "context": context, "ram_budget_bytes": ram, "gpu_budget_bytes": gpu,
            "gpu_index": gpu_index, "kv_estimate_bytes": kv, "notes": notes,
            "profiles": [p.to_dict() for p in profiles]}
