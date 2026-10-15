from __future__ import annotations

import hashlib
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .planner import Profile
from .runtime import RuntimeError_, Server, command, json_request, runtime_info, stream_completion, unsupported

DEFAULT_PROMPT = (
    "Write a Python function that merges sorted iterators lazily with a heap. "
    "Explain the complexity and give examples with empty inputs, duplicates, and three input streams.\n"
)


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def effective_context(props: dict) -> int | None:
    settings = props.get("default_generation_settings", {})
    for source in (settings, settings.get("params", {}), props):
        for key in ("n_ctx", "n_ctx_slot"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


def aggregate(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("At least one sample is required")
    speeds = [s["decode_tokens_per_second"] for s in samples]
    median = statistics.median(speeds)
    return {"median_decode_tokens_per_second": median,
            "median_ttft_seconds": statistics.median(s["ttft_seconds"] for s in samples),
            "min_decode_tokens_per_second": min(speeds), "max_decode_tokens_per_second": max(speeds),
            "relative_mad": statistics.median(abs(value - median) for value in speeds) / median if median else None,
            "sample_count": len(samples),
            "repeat_output_consistent": len({s["output_sha256"] for s in samples}) == 1}


def choose_winner(results: list[dict], *, max_ttft: float | None = None) -> str | None:
    candidates = [r for r in results if r.get("status") == "ok" and
                  (max_ttft is None or r["metrics"]["median_ttft_seconds"] <= max_ttft)]
    if not candidates:
        return None
    # Throughput is the declared objective. TTFT breaks exact ties.
    return max(candidates, key=lambda r: (r["metrics"]["median_decode_tokens_per_second"],
                                         -r["metrics"]["median_ttft_seconds"]))["profile"]["name"]


def tune(plan: dict, model_path: str, server_path: str, output: Path, *,
         prompt: str = DEFAULT_PROMPT, tokens: int = 128, repeats: int = 3, warmups: int = 1,
         threads: int = 4, timeout: float = 120, startup_timeout: float = 300,
         max_ttft: float | None = None, names: list[str] | None = None,
         notify=print, checksums: dict | None = None) -> dict:
    if repeats < 1 or warmups < 0 or tokens < 1 or threads < 1:
        raise ValueError("Invalid experiment settings")
    if tokens >= plan["context"]:
        raise ValueError("The generation budget must be smaller than the fixed context")
    info = runtime_info(server_path)
    settings = {"context": plan["context"], "tokens": tokens, "repeats": repeats, "warmups": warmups,
                "threads": threads, "seed": 42, "temperature": 0, "cache_prompt": False,
                "ignore_eos": True, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_characters": len(prompt), "max_median_ttft_seconds": max_ttft,
                "objective": "highest median decode tokens/s among successful candidates satisfying the TTFT constraint",
                "completion_timeout_seconds": timeout, "startup_timeout_seconds": startup_timeout}
    report = {"schema_version": 1, "mistralfit_version": __version__, "status": "running",
              "created_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
              "plan": plan, "runtime": {k: v for k, v in info.items() if k != "help"},
              "full_model_sha256": checksums, "results": [], "winner": None,
              "limits": ["No language-quality benchmark is performed.",
                         "Output hashes expose differences; matching hashes are not proof of model quality.",
                         "Memory numbers are estimates, not measured peak RSS or VRAM.",
                         "One process start per candidate; repeated requests are not independent thermal runs.",
                         "No model weights, prompts, generated text, API keys, or absolute model paths are included."]}
    candidates = [Profile(**p) for p in plan["profiles"]]
    if names:
        unknown = set(names) - {p.name for p in candidates}
        if unknown:
            raise ValueError("Unknown profiles: " + ", ".join(sorted(unknown)))
        candidates = [p for p in candidates if p.name in names]
    # Record a deterministic shuffled order to limit always favoring the same placement.
    random.Random(42).shuffle(candidates)
    report["trial_order"] = [p.name for p in candidates]
    atomic_json(output, report)
    try:
        for profile in candidates:
            entry = {"profile": profile.to_dict(), "status": "pending", "samples": []}
            report["results"].append(entry)
            missing = unsupported(profile, info["help"])
            if not profile.eligible or missing:
                entry["status"] = "skipped"
                entry["reason"] = profile.reason if not profile.eligible else "Runtime lacks: " + ", ".join(missing)
                notify(f"[{profile.name}] skipped: {entry['reason']}")
                atomic_json(output, report)
                continue
            notify(f"[{profile.name}] loading model; {warmups} warmup(s), {repeats} measured request(s)")

            def builder(port, key):
                return command(server_path, model_path, profile, context=plan["context"], threads=threads,
                               port=port, gpu_index=plan["gpu_index"], help_text=info["help"], api_key=key)

            try:
                with Server(builder, startup_timeout=startup_timeout) as local:
                    props = json_request(local.base, "/props", api_key=local.api_key)
                    effective = effective_context(props)
                    entry["effective_context"] = effective
                    if effective != plan["context"]:
                        raise RuntimeError_(f"Backend context is {effective!r}; expected {plan['context']}. Trial excluded.")
                    tokenized = json_request(local.base, "/tokenize", {"content": prompt, "add_special": True}, api_key=local.api_key)
                    actual_tokens = tokenized.get("tokens")
                    if not isinstance(actual_tokens, list) or len(actual_tokens) + tokens > plan["context"]:
                        raise RuntimeError_("Prompt plus generation budget exceeds the fixed context")
                    entry["tokenized_prompt_length"] = len(actual_tokens)
                    for _ in range(warmups):
                        stream_completion(local.base, prompt, tokens=tokens, timeout=timeout, api_key=local.api_key)
                    for index in range(repeats):
                        sample = stream_completion(local.base, prompt, tokens=tokens, timeout=timeout, api_key=local.api_key)
                        entry["samples"].append(sample)
                        notify(f"[{profile.name}] {index + 1}/{repeats}: {sample['decode_tokens_per_second']:.2f} tok/s, TTFT {sample['ttft_seconds']:.3f}s")
                        atomic_json(output, report)
                    entry["metrics"] = aggregate(entry["samples"])
                    entry["status"] = "ok"
            except (OSError, ValueError, RuntimeError_, TimeoutError) as exc:
                entry["status"] = "failed"
                # No backend log tails in portable reports: these may contain local paths.
                entry["reason"] = str(exc).split("\n")[0].replace(model_path, "<model>")[:500]
                notify(f"[{profile.name}] failed: {entry['reason']}")
            atomic_json(output, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        atomic_json(output, report)
        raise
    report["winner"] = choose_winner(report["results"], max_ttft=max_ttft)
    baseline = next((r for r in report["results"] if r["profile"]["name"] == "upstream-auto" and r["status"] == "ok"), None)
    winner = next((r for r in report["results"] if r["profile"]["name"] == report["winner"]), None)
    if baseline and winner:
        baseline_speed = baseline["metrics"]["median_decode_tokens_per_second"]
        report["speedup_vs_upstream_auto"] = winner["metrics"]["median_decode_tokens_per_second"] / baseline_speed
        report["winner_matches_baseline_output"] = (
            {s["output_sha256"] for s in winner["samples"]} == {s["output_sha256"] for s in baseline["samples"]})
    report["status"] = "complete" if winner else "no_successful_candidate"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output, report)
    return report
