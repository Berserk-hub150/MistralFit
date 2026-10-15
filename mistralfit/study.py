"""Controlled holdout evaluation of three selected launch configurations."""

from __future__ import annotations

import hashlib
import html
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .gguf import full_sha256, inspect_model
from .hardware import detect
from .planner import Profile, plan
from .runtime import (RuntimeError_, Server, command, completion_response, json_request,
                      runtime_info, stream_completion, unsupported)
from .telemetry import ResourceSampler, counter_delta, environment_details, process_counters
from .tuning import DEFAULT_PROMPT, aggregate, atomic_json, effective_context, tune

REFERENCE_REPOSITORY = "https://github.com/Najafu/llama-autotune"
REFERENCE_COMMIT = "b03fe68e476181b4acf9f3dec1e80274426e3877"

WORKLOADS = [
    {"id": "python-lru", "prompt": "Implement an LRU cache in Python using an ordered dictionary. Explain get, put, eviction, zero capacity and testing. Give runnable examples.\n"},
    {"id": "incident-analysis", "prompt": "A web service has 120 ms median latency but 8 s p99 latency. CPU averages 35%, the database connection pool is full, and requests retry twice. Give a step-by-step diagnosis and mitigations, distinguishing observations from hypotheses.\n"},
    {"id": "document-reasoning", "prompt": "Policy: backups run daily at 02:00 UTC, retained for 14 days; a weekly backup is retained for 12 weeks; failed jobs page the on-call engineer. Explain a restore after a deletion discovered 20 days later, the missing information, and a safe verification procedure.\n"},
]
QUALITY = [
    {"id": "sum-json", "prompt": 'Return only a JSON object with key "answer" containing the integer 19 + 23.', "expected": {"answer": 42}},
    {"id": "extract-json", "prompt": 'From "Marta has 3 apples; Paolo has 5 pears", return only JSON with keys "name" and "pears" for the person with pears.', "expected": {"name": "Paolo", "pears": 5}},
    {"id": "sort-json", "prompt": 'Return only JSON with key "values" containing [8, 2, 8, -1] sorted ascending, keeping duplicates.', "expected": {"values": [-1, 2, 8, 8]}},
]


def import_autotune(path: Path, context: int, gpu: int) -> dict:
    """Read the reference export, never execute commands embedded in a file."""
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("External profile exceeds 1 MiB")
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON object containing an external launch profile")
    args = document.get("args")
    if not isinstance(args, list) or len(args) > 128 or not all(isinstance(x, str) for x in args):
        raise ValueError("Expected a llama-autotune profile containing an argv list")
    aliases = {"-t": "--threads", "-tb": "--threads-batch", "-b": "--batch-size", "-ub": "--ubatch-size",
               "-ngl": "--n-gpu-layers", "-fa": "--flash-attn", "-c": "--ctx-size", "-mg": "--main-gpu",
               "-ctk": "--cache-type-k", "-ctv": "--cache-type-v", "-mmp": "mmap", "-sm": "--split-mode"}
    integers = {"--threads", "--threads-batch", "--batch-size", "--ubatch-size", "--ctx-size", "--parallel"}
    values, extras, index = {}, [], 0
    while index < len(args):
        flag = aliases.get(args[index], args[index])
        index += 1
        if flag in ("--mlock", "--no-mmap"):
            if flag in extras:
                raise ValueError("Duplicate external flag: " + flag)
            extras.append(flag)
            continue
        if flag not in integers | {"--n-gpu-layers", "--flash-attn", "--main-gpu", "--cache-type-k", "--cache-type-v", "mmap", "--split-mode", "--numa"}:
            raise ValueError("Unsupported external flag: " + flag)
        if flag in values or index >= len(args):
            raise ValueError("Duplicate or valueless external flag: " + flag)
        value = args[index]
        index += 1
        if flag in integers:
            number = int(value)
            if not 1 <= number <= 1048576:
                raise ValueError("Invalid positive external argument: " + flag)
            value = str(number)
        if flag == "--main-gpu" and int(value) != gpu:
            raise ValueError("External profile selects a different GPU")
        if flag in ("--cache-type-k", "--cache-type-v") and value != "f16":
            raise ValueError("This controlled study requires identical f16 KV precision")
        if flag == "--parallel" and value != "1":
            raise ValueError("External profile must use a single server slot")
        if flag == "--split-mode" and value != "none":
            raise ValueError("This study supports one GPU; external split-mode must be none")
        if flag == "--flash-attn":
            value = {"1": "on", "0": "off"}.get(value, value)
            if value not in ("on", "off", "auto"):
                raise ValueError("Invalid flash-attention mode")
        if flag == "mmap":
            if value not in ("0", "1"):
                raise ValueError("Invalid mmap mode")
            if value == "0":
                extras.append("--no-mmap")
        if flag == "--numa" and value not in ("distribute", "isolate", "numactl"):
            raise ValueError("Invalid NUMA mode")
        values[flag] = value
    layers = values.pop("--n-gpu-layers", "auto")
    if layers == "all":
        layers = 10000
    elif layers == "auto":
        layers = None
    else:
        layers = int(layers)
        if not 0 <= layers <= 10000:
            raise ValueError("Invalid GPU layer count")
    original_context = values.pop("--ctx-size", None)
    values.pop("mmap", None)
    return {"name": "llama-autotune-selected", "profile": Profile("llama-autotune-selected", layers).to_dict(),
            "overrides": values, "extra_flags": extras,
            "source": {"repository": REFERENCE_REPOSITORY, "profile_sha256": hashlib.sha256(raw).hexdigest(),
                       "revision": "not recorded by this export; supply training provenance separately",
                       "exported_context": original_context, "evaluation_context": context,
                       "context_normalized": original_context is not None and int(original_context) != context}}


def candidate_argv(candidate, server, model, *, context, threads, gpu, help_text, port, key, repack):
    argv = command(server, model, Profile(**candidate["profile"]), context=context, threads=threads,
                   gpu_index=gpu, port=port, api_key=key, help_text=help_text, repack=repack)
    for flag, value in candidate.get("overrides", {}).items():
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv.extend([flag, value])
    for flag in candidate.get("extra_flags", []):
        if flag not in argv:
            argv.append(flag)
    return argv


def functional_checks(local, timeout):
    results = []
    for task in QUALITY:
        body = {"prompt": task["prompt"], "n_predict": 128, "stream": False, "temperature": 0,
                "seed": 42, "cache_prompt": False, "ignore_eos": False}
        with completion_response(local.base, body, timeout=timeout, api_key=local.api_key) as response:
            answer = json.load(response)
        if not isinstance(answer, dict) or not isinstance(answer.get("content"), str):
            raise RuntimeError_("Backend did not return completion text for the functional check")
        text = answer.get("content", "")
        try:
            actual = json.loads(text)
            passed = (json.dumps(actual, sort_keys=True) == json.dumps(task["expected"], sort_keys=True)
                      and not answer.get("truncated", False))
        except (ValueError, TypeError):
            passed = False
        results.append({"id": task["id"], "passed": passed,
                        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "scope": "Small raw-completion JSON functional check; not a quality benchmark"})
    return results


def balanced_order(names: list[str], rounds: int) -> list[list[str]]:
    order = list(names)
    random.Random(42).shuffle(order)
    return [order[n % len(order):] + order[:n % len(order)] for n in range(rounds)]


def diagnose(run):
    def median(field):
        values = [s[field] / 1000 for s in run["samples"] if isinstance(s.get(field), (float, int))]
        return statistics.median(values) if values else None
    def total(field):
        values = [s["process_counter_delta"][field] for s in run["samples"] if s["process_counter_delta"].get(field) is not None]
        return sum(values) if values else None
    prompt, decode = median("prompt_ms"), median("decode_ms")
    return {"median_native_prompt_seconds": prompt, "median_native_decode_seconds": decode,
            "larger_native_phase": ("prompt" if prompt > decode else "decode") if prompt is not None and decode is not None else None,
            "post_warmup_major_faults": total("major_faults"), "post_warmup_read_bytes": total("read_bytes"),
            "pcie_transfer_seconds": None,
            "scope": "Timed requests only. Locates observed cost; does not identify a kernel or isolate expert transfers. Missing counters stay null."}


def summarize(report):
    summaries = []
    for candidate in report.get("candidates", []):
        runs = [r for r in report["runs"] if r["candidate"] == candidate["name"]]
        expected_count = report["settings"]["rounds"] * report["settings"]["repeats"]
        complete = len(runs) == report["settings"]["rounds"] and all(r["status"] == "ok" for r in runs)
        complete = complete and all(sum(s["case_id"] == c["id"] for r in runs for s in r.get("samples", [])) == expected_count for c in report["cases"])
        rows = []
        for case in report["cases"]:
            samples = [s for r in runs for s in r.get("samples", []) if s["case_id"] == case["id"]]
            if samples:
                row = {"case_id": case["id"], **aggregate(samples)}
                baseline = [s for r in report["runs"] if r["candidate"] == "upstream-auto" and r["status"] == "ok"
                            for s in r["samples"] if s["case_id"] == case["id"]]
                row["decode_ratio_vs_upstream_auto"] = (row["median_decode_tokens_per_second"] / aggregate(baseline)["median_decode_tokens_per_second"]
                                                           if complete and len(baseline) == expected_count and len(samples) == expected_count else None)
                row["matches_baseline_output_hashes"] = ({s["output_sha256"] for s in samples} == {s["output_sha256"] for s in baseline}
                                                         if complete and len(baseline) == expected_count and len(samples) == expected_count else None)
                rows.append(row)
        checks = [c for r in runs for c in r.get("functional_checks", [])]
        summaries.append({"candidate": candidate["name"], "complete": complete,
                          "functional_checks_passed": len(checks) == len(QUALITY) * report["settings"]["rounds"] and all(c["passed"] for c in checks),
                          "cases": rows})
    return summaries


def render_study(report):
    esc = lambda x: html.escape(str(x), quote=True)
    rows = []
    resources = []
    for candidate in report.get("summary", []):
        for case in candidate["cases"]:
            ratio = case.get("decode_ratio_vs_upstream_auto")
            ratio_text = "—" if ratio is None else f"{ratio:.3f}"
            rows.append(f"<tr><td>{esc(candidate['candidate'])}</td><td>{esc(case['case_id'])}</td>"
                        f"<td>{case['median_decode_tokens_per_second']:.3f}</td><td>{case['median_ttft_seconds']:.3f}</td>"
                        f"<td>{ratio_text}</td><td>{esc(candidate['complete'])}</td><td>{esc(candidate['functional_checks_passed'])}</td></tr>")
    for run in report.get("runs", []):
        memory = run.get("resources", {})
        def mib(field):
            value = memory.get(field)
            return f"{value / 1024 ** 2:.1f}" if isinstance(value, (int, float)) else "unavailable"
        startup = run.get("startup_to_ready_seconds")
        startup_text = f"{startup:.3f}" if isinstance(startup, (int, float)) else "—"
        resources.append(f"<tr><td>{esc(run['candidate'])}</td><td>{esc(run['round'])}</td><td>{startup_text}</td>"
                         f"<td>{mib('observed_peak_rss_bytes')}</td><td>{mib('observed_peak_gpu_bytes')}</td><td>{esc(run['status'])}</td></tr>")
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>MistralFit controlled study</title><style>body{font:16px system-ui;background:#11151a;color:#e5ebef;max-width:1150px;margin:40px auto;padding:20px}'
            'table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #46505a}pre{white-space:pre-wrap;overflow-wrap:anywhere}.scroll{overflow-x:auto}</style>'
            f'<h1>MistralFit controlled study</h1><p>Status: {esc(report["status"])}</p>'
            '<p>Observed configuration results, not a ranking of tuning algorithms. No general speed or quality claim is established. Read the JSON for individual runs, exclusions and memory samples.</p>'
            '<div class="scroll"><table><tr><th>Configuration</th><th>Workload / tokens</th><th>Decode tok/s</th><th>TTFT s</th><th>Ratio vs auto</th><th>Complete</th><th>Functional checks</th></tr>'
            + ''.join(rows) + '</table></div><h2>Process observations</h2><p>Requested sampling interval; observed peaks only. Unavailable counters are not zero allocations. RSS includes resident mapped pages; GPU query support varies.</p>'
            '<div class="scroll"><table><tr><th>Configuration</th><th>Round</th><th>Startup to ready s</th><th>Observed RSS MiB</th><th>Observed VRAM MiB</th><th>Status</th></tr>'
            + ''.join(resources) + '</table></div><h2>Controls and limitations</h2><pre>'
            + esc(json.dumps({"settings": report["settings"], "limits": report["limits"], "blockers": report.get("blockers", [])}, indent=2)) + '</pre></html>')


def study(model_path, server_path, output: Path, *, competitor_profile: Path | None, competitor_provenance: Path | None = None, context=4096,
          token_budgets=(128, 256), rounds=3, repeats=3, warmups=1, threads=4,
          timeout=600, startup_timeout=600, sample_interval=1.0, gpu=0,
          ram_gib=None, gpu_gib=None, repack=True, preflight=False, allow_cpu=False, notify=print):
    if rounds < 1 or repeats < 1 or warmups < 1 or any(n < 128 or n >= context for n in token_budgets) or not token_budgets:
        raise ValueError("Study requires warmups, positive rounds/repeats and token budgets >=128 below the context")
    model, hardware = inspect_model(model_path), detect()
    planning = plan(model, hardware, context=context, ram_gib=ram_gib, gpu_gib=gpu_gib, gpu_index=gpu)
    cases = [{"id": f"{w['id']}-{n}", "prompt_id": w["id"], "tokens": n,
              "prompt_sha256": hashlib.sha256(w["prompt"].encode()).hexdigest()} for w in WORKLOADS for n in token_budgets]
    report = {"schema_version": "study-1", "mistralfit_version": __version__, "status": "preflight",
              "created_at": datetime.now(timezone.utc).isoformat(), "hardware": hardware.to_dict(), "environment": environment_details(), "plan": planning,
              "settings": {"context": context, "token_budgets": list(token_budgets), "rounds": rounds, "repeats": repeats,
                           "warmups_per_case": warmups, "default_threads": threads, "sample_interval_seconds": sample_interval,
                           "temperature": 0, "seed": 42, "cache_prompt": False, "ignore_eos": True,
                           "cache_type_k": "f16", "cache_type_v": "f16", "weight_repacking": repack,
                           "completion_timeout_seconds": timeout, "startup_timeout_seconds": startup_timeout},
              "cases": cases, "candidates": [], "runs": [], "summary": [], "blockers": [],
              "limits": ["Selected configurations are evaluated on shared holdout prompts; search budgets/workloads differ, so this is not an algorithm-superiority experiment.",
                         "The external export's context is explicitly normalized to the shared evaluation context. Other accepted settings are preserved; non-f16 KV or multi-GPU profiles are rejected.",
                         "Model checksum is computed once before search/evaluation; the external training history is separate provenance.",
                         "Memory preflight is an estimate, not proof of residency. Post-warmup Linux page-fault/I/O deltas expose paging when counters are available.",
                         "Three JSON functional checks do not establish general language quality or quality parity.",
                         "No PCIe-transfer profiler runs. Process counters cannot isolate expert transfer time.",
                         "No prompts, generated text, API credentials or absolute model paths are stored in this report."]}
    if not hardware.gpus and not allow_cpu:
        report["blockers"].append("No NVIDIA GPU detected. This protocol targets a single NVIDIA GPU workstation.")
    if hardware.gpus and (len(hardware.gpus) != 1 or hardware.gpus[0].index != gpu):
        report["blockers"].append("This first controlled protocol requires one detected NVIDIA GPU with the requested index.")
    ram = planning["ram_budget_bytes"]
    vram = planning["gpu_budget_bytes"]
    combined = ram if hardware.unified_memory else ram + vram
    if model.weight_bytes > combined:
        report["blockers"].append("Weights exceed the combined available RAM/VRAM budget; paging-heavy studies are excluded.")
    if not any(p["eligible"] for p in planning["profiles"]):
        report["blockers"].append("No planned placement meets the estimated memory budget including runtime headroom.")
    if competitor_profile is None and not preflight:
        report["blockers"].append("Supply an exported llama-autotune profile for the third configuration.")
    external = import_autotune(competitor_profile, context, gpu) if competitor_profile else None
    if competitor_provenance:
        if not external or competitor_provenance.stat().st_size > 1024 * 1024:
            raise ValueError("Training provenance requires an external profile and a file <=1 MiB")
        training = json.loads(competitor_provenance.read_text(encoding="utf-8"))
        if not isinstance(training, dict):
            raise ValueError("Training provenance must be a JSON object")
        if training.get("profile_sha256") != external["source"]["profile_sha256"]:
            raise ValueError("Training provenance does not match the supplied profile")
        external["source"]["training"] = training
    if report["blockers"] or preflight:
        report["status"] = "blocked" if report["blockers"] else "ready"
        atomic_json(output, report)
        output.with_suffix(".html").write_text(render_study(report), encoding="utf-8")
        return report
    info = runtime_info(server_path)
    if not repack and "--no-repack" not in info["help"]:
        raise ValueError("Runtime lacks --no-repack")
    report["runtime"] = {k: v for k, v in info.items() if k != "help"}
    report["runtime"]["executable_sha256"] = full_sha256([server_path])
    report["full_model_sha256"] = full_sha256(model.paths)
    report["status"] = "calibrating"
    atomic_json(output, report)
    started = time.monotonic()
    try:
        calibration = tune(planning, model.paths[0], server_path, output.parent / (output.stem + "-calibration.json"),
                           prompt=DEFAULT_PROMPT, tokens=128, repeats=repeats, warmups=warmups, threads=threads,
                           timeout=timeout, startup_timeout=startup_timeout, notify=notify,
                           checksums=report["full_model_sha256"], repack=repack)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        atomic_json(output, report)
        output.with_suffix(".html").write_text(render_study(report), encoding="utf-8")
        raise
    report["calibration_elapsed_seconds"] = time.monotonic() - started
    if not calibration["winner"]:
        report["status"] = "calibration_failed"
        atomic_json(output, report)
        output.with_suffix(".html").write_text(render_study(report), encoding="utf-8")
        return report
    selected = next(e["profile"] for e in calibration["results"] if e["profile"]["name"] == calibration["winner"])
    report["calibration_winner"] = calibration["winner"]
    report["candidates"] = [{"name": "upstream-auto", "profile": Profile("upstream-auto", None).to_dict()},
                            {"name": "mistralfit-selected", "profile": selected}, external]
    report["round_order"] = balanced_order([c["name"] for c in report["candidates"]], rounds)
    report["status"] = "running"
    atomic_json(output, report)
    workloads = {w["id"]: w["prompt"] for w in WORKLOADS}
    try:
        for round_index, order in enumerate(report["round_order"]):
            case_order = list(cases)
            random.Random(1337 + round_index).shuffle(case_order)
            for name in order:
                candidate = next(c for c in report["candidates"] if c["name"] == name)
                run = {"round": round_index + 1, "candidate": name, "status": "pending", "samples": [],
                       "case_order": [c["id"] for c in case_order]}
                report["runs"].append(run)
                atomic_json(output, report)
                sampler = ResourceSampler(sample_interval)
                def builder(port, key):
                    return candidate_argv(candidate, server_path, model.paths[0], context=context, threads=threads,
                                          gpu=gpu, help_text=info["help"], port=port, key=key, repack=repack)
                try:
                    missing = unsupported(Profile(**candidate["profile"]), info["help"])
                    if missing:
                        raise ValueError("Runtime lacks: " + ", ".join(missing))
                    notify(f"[study {round_index + 1}/{rounds}] {name}: loading")
                    with Server(builder, startup_timeout=startup_timeout, on_spawn=sampler.start) as local:
                        run["startup_to_ready_seconds"] = local.startup_seconds
                        if effective_context(json_request(local.base, "/props", api_key=local.api_key)) != context:
                            raise ValueError("Backend changed the controlled context")
                        for case in case_order:
                            prompt = workloads[case["prompt_id"]]
                            tokenized = json_request(local.base, "/tokenize", {"content": prompt, "add_special": True}, api_key=local.api_key)
                            if not isinstance(tokenized.get("tokens"), list) or len(tokenized["tokens"]) + case["tokens"] > context:
                                raise ValueError("Holdout workload exceeds the controlled context")
                            for _ in range(warmups):
                                stream_completion(local.base, prompt, tokens=case["tokens"], timeout=timeout, api_key=local.api_key)
                            for repeat in range(repeats):
                                before = process_counters(local.process.pid)
                                sample = stream_completion(local.base, prompt, tokens=case["tokens"], timeout=timeout, api_key=local.api_key)
                                sample.update(case_id=case["id"], repetition=repeat + 1,
                                              process_counter_delta=counter_delta(before, process_counters(local.process.pid)))
                                run["samples"].append(sample)
                                atomic_json(output, report)
                            notify(f"[{name}] {case['id']}: {repeats} measured request(s)")
                        run["functional_checks"] = functional_checks(local, timeout)
                        sampler.stop()
                        run["status"] = "ok"
                except (OSError, ValueError, RuntimeError_) as exc:
                    run["status"] = "failed"
                    run["reason"] = str(exc).split("\n")[0].replace(model.paths[0], "<model>")[:500]
                finally:
                    sampler.stop()
                    run["resources"] = sampler.summary()
                    run["diagnostics"] = diagnose(run)
                    atomic_json(output, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["summary"] = summarize(report)
        atomic_json(output, report)
        output.with_suffix(".html").write_text(render_study(report), encoding="utf-8")
        raise
    report["summary"] = summarize(report)
    report["status"] = "complete" if all(c["complete"] for c in report["summary"]) else "incomplete"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output, report)
    output.with_suffix(".html").write_text(render_study(report), encoding="utf-8")
    return report
