from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__
from .gguf import GGUFError, Model, Tensor, full_sha256, inspect_model
from .hardware import GIB, Hardware, detect
from .planner import Profile, plan
from .report import render
from .runtime import RuntimeError_, clean_environment, command, json_request, resolve_server, runtime_info, unsupported
from .tuning import DEFAULT_PROMPT, atomic_json, tune


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def positive_float(value):
    import math
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def hardware_text(hw):
    print(f"{hw.system} {hw.machine} | {hw.cpu_count} logical CPU(s)")
    print(f"RAM total: {hw.ram_total / GIB:.2f} GiB | available: {hw.ram_available / GIB:.2f} GiB")
    for gpu in hw.gpus:
        print(f"GPU {gpu.index}: {gpu.name} | {gpu.free_bytes / GIB:.2f}/{gpu.total_bytes / GIB:.2f} GiB free")
    for note in hw.notes:
        print("Note:", note)


def print_plan(result):
    model = result["model"]
    print(f"{model['name']} | {model['architecture']} | {model['weight_bytes'] / GIB:.2f} GiB tensor storage")
    print(f"{model['blocks']} blocks | {model['routed_expert_bytes'] / GIB:.2f} GiB routed-expert storage")
    print(f"Fixed context: {result['context']} tokens. Memory below is ESTIMATED, not measured.")
    print(f"{'Profile':<22}{'GPU GiB':>12}{'RAM GiB':>12}  Candidate")
    for p in result["profiles"]:
        gpu = "auto" if p["estimated_gpu_bytes"] is None else f"{p['estimated_gpu_bytes'] / GIB:.2f}"
        ram = "unknown" if p["estimated_ram_bytes"] is None else f"{p['estimated_ram_bytes'] / GIB:.2f}"
        print(f"{p['name']:<22}{gpu:>12}{ram:>12}  {'try' if p['eligible'] else 'skip'} {p['reason']}")
    for note in result["notes"]:
        print("Note:", note)


def add_plan_options(parser):
    parser.add_argument("model", help="GGUF file, or first shard of a split model")
    parser.add_argument("--ctx", type=positive, default=4096, help="fixed context for every candidate (default 4096)")
    parser.add_argument("--ram-gib", type=positive_float, help="override AVAILABLE host/shared RAM budget")
    parser.add_argument("--gpu-gib", type=positive_float, help="override AVAILABLE single-GPU memory budget")
    parser.add_argument("--gpu", type=nonnegative, default=0, help="llama.cpp main GPU index / detected NVIDIA index")
    parser.add_argument("--allow-paging", action="store_true", help="try candidates whose estimated RAM exceeds budget")


def make_plan(args):
    model = inspect_model(args.model)
    result = plan(model, detect(), context=args.ctx, ram_gib=args.ram_gib, gpu_gib=args.gpu_gib,
                  gpu_index=args.gpu, allow_paging=args.allow_paging)
    return model, result


def parser():
    root = argparse.ArgumentParser(description="MistralFit — measured local Mistral configuration search using llama.cpp")
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="action", required=True)
    doctor = sub.add_parser("doctor", help="inspect RAM, NVIDIA GPUs and optional llama-server capabilities")
    doctor.add_argument("--server")
    doctor.add_argument("--json", action="store_true")
    inspect = sub.add_parser("inspect", help="read GGUF tensor layout without loading weights")
    inspect.add_argument("model")
    inspect.add_argument("--sha256", action="store_true", help="also hash all weight bytes; can take minutes")
    planning = sub.add_parser("plan", help="estimate placement candidates without loading a model")
    add_plan_options(planning)
    planning.add_argument("--out", type=Path)
    planning.add_argument("--json", action="store_true")
    tuning = sub.add_parser("tune", help="start llama-server once per candidate and measure real token streams")
    add_plan_options(tuning)
    tuning.add_argument("--server")
    tuning.add_argument("--out", type=Path, default=Path("results/experiment.json"))
    tuning.add_argument("--tokens", type=positive, default=128)
    tuning.add_argument("--repeats", type=positive, default=3)
    tuning.add_argument("--warmups", type=nonnegative, default=1)
    tuning.add_argument("--threads", type=positive, default=max(1, (os.cpu_count() or 2) // 2))
    tuning.add_argument("--timeout", type=positive_float, default=120)
    tuning.add_argument("--startup-timeout", type=positive_float, default=300)
    tuning.add_argument("--max-ttft", type=positive_float, help="only choose a winner within this median TTFT in seconds")
    tuning.add_argument("--profiles", help="comma-separated subset of generated profile names")
    tuning.add_argument("--prompt-file", type=Path, help="local workload; only its hash is saved in the report")
    tuning.add_argument("--sha256", action="store_true", help="hash full weights for stronger reproducibility")
    reporting = sub.add_parser("report", help="render an experiment JSON as a standalone offline HTML report")
    reporting.add_argument("experiment", type=Path)
    reporting.add_argument("--out", type=Path, default=Path("results/report.html"))
    serving = sub.add_parser("serve", help="launch the measured winner, after checking model/runtime identity")
    serving.add_argument("model")
    serving.add_argument("--report", type=Path, required=True)
    serving.add_argument("--server")
    serving.add_argument("--port", type=positive, default=8000)
    serving.add_argument("--dry-run", action="store_true")
    smoke = sub.add_parser("check-tools", help="one local API tool-call smoke check; executes no tools")
    smoke.add_argument("--url", default="http://127.0.0.1:8000")
    smoke.add_argument("--timeout", type=positive_float, default=120)
    sub.add_parser("demo", help="illustrative memory plan only: no model, inference or benchmark")
    return root


def _serve(args):
    if args.port > 65535:
        raise ValueError("Port must be <= 65535")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported report schema")
    if not report.get("winner"):
        raise ValueError("This report has no successful winner")
    model = inspect_model(args.model)
    if model.header_fingerprint != report["plan"]["model"]["header_fingerprint"]:
        raise ValueError("GGUF structural fingerprint differs from the measured model")
    if report.get("full_model_sha256") and full_sha256(model.paths) != report["full_model_sha256"]:
        raise ValueError("Weight checksums differ from the measured model")
    server = resolve_server(args.server)
    info = runtime_info(server)
    if info["version"] != report["runtime"]["version"]:
        raise ValueError("Runtime version differs from the experiment; rerun tuning with this build")
    entry = next(r for r in report["results"] if r["profile"]["name"] == report["winner"] and r["status"] == "ok")
    profile = Profile(**entry["profile"])
    missing = unsupported(profile, info["help"])
    if missing:
        raise ValueError("Runtime lacks: " + ", ".join(missing))
    argv = command(server, model.paths[0], profile, context=report["settings"]["context"],
                   threads=report["settings"]["threads"], port=args.port,
                   gpu_index=report["plan"]["gpu_index"], help_text=info["help"])
    if args.dry_run:
        # JSON is an exact argv representation, not a shell command to re-evaluate.
        print(json.dumps({"argv": argv, "bind": "127.0.0.1"}, indent=2))
        return 0
    print(f"Starting {profile.name}. API: http://127.0.0.1:{args.port}/v1 ; Ctrl+C stops the server.", flush=True)
    process = subprocess.Popen(argv, env=clean_environment())
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=8)
        return 130


def _check_tools(args):
    from urllib.parse import urlparse
    url = urlparse(args.url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.path not in ("", "/") or url.query or url.fragment or url.username:
        raise ValueError("Use a local HTTP base URL such as http://127.0.0.1:8000")
    payload = {"messages": [{"role": "user", "content": "Call add with a=19 and b=23. Use the tool."}],
               "tools": [{"type": "function", "function": {"name": "add", "description": "Add two integers",
                          "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                         "required": ["a", "b"], "additionalProperties": False}}}],
               "tool_choice": "required", "temperature": 0, "max_tokens": 512}
    models = json_request(args.url.rstrip("/"), "/v1/models", timeout=args.timeout)
    payload["model"] = models["data"][0]["id"]
    response = json_request(args.url.rstrip("/"), "/v1/chat/completions", payload, timeout=args.timeout)
    calls = response["choices"][0]["message"].get("tool_calls", [])
    valid = len(calls) == 1 and calls[0].get("function", {}).get("name") == "add"
    if valid:
        arguments = calls[0]["function"]["arguments"]
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        valid = arguments == {"a": 19, "b": 23}
    print(json.dumps({"passed": valid, "scope": "Single API contract smoke check; no tool executed, no quality benchmark."}, indent=2))
    return 0 if valid else 1


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "doctor":
            hw = detect()
            details = {"hardware": hw.to_dict(), "python": sys.version.split()[0]}
            if args.server:
                info = runtime_info(resolve_server(args.server))
                details["runtime"] = {k: v for k, v in info.items() if k != "help"}
            if args.json:
                print(json.dumps(details, indent=2))
            else:
                hardware_text(hw)
                if "runtime" in details:
                    print(json.dumps(details["runtime"], indent=2))
        elif args.action == "inspect":
            model = inspect_model(args.model)
            result = model.summary()
            if args.sha256:
                result["full_sha256"] = full_sha256(model.paths)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.action in ("plan", "tune"):
            model, result = make_plan(args)
            if args.action == "plan":
                if args.out:
                    atomic_json(args.out, result)
                if args.json:
                    print(json.dumps(result, indent=2))
                else:
                    print_plan(result)
            else:
                print_plan(result)
                prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
                if not prompt.strip():
                    raise ValueError("Prompt is empty")
                if len(prompt.encode()) > 8 * 1024 * 1024:
                    raise ValueError("Prompt file exceeds the 8 MiB limit")
                experiment = tune(result, model.paths[0], resolve_server(args.server), args.out, prompt=prompt,
                                  tokens=args.tokens, repeats=args.repeats, warmups=args.warmups, threads=args.threads,
                                  timeout=args.timeout, startup_timeout=args.startup_timeout, max_ttft=args.max_ttft,
                                  names=args.profiles.split(",") if args.profiles else None,
                                  checksums=full_sha256(model.paths) if args.sha256 else None)
                html_path = args.out.with_suffix(".html")
                html_path.write_text(render(experiment), encoding="utf-8")
                print("Winner:", experiment["winner"] or "none")
                print("JSON:", args.out, "| HTML:", html_path)
                return 0 if experiment["winner"] else 2
        elif args.action == "report":
            result = json.loads(args.experiment.read_text(encoding="utf-8"))
            if result.get("schema_version") != 1:
                raise ValueError("Unsupported report schema")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(render(result), encoding="utf-8")
            print(args.out)
        elif args.action == "serve":
            return _serve(args)
        elif args.action == "check-tools":
            return _check_tools(args)
        elif args.action == "demo":
            print("ILLUSTRATIVE PLAN — SYNTHETIC TENSOR SIZES, NOT A REAL MODEL OR BENCHMARK\n")
            tensors = [Tensor("token_embd.weight", [4096, 32000], 12, 0, 256 * 1024 ** 2)]
            for i in range(36):
                tensors.extend([Tensor(f"blk.{i}.attn_q.weight", [4096, 4096], 12, 0, 64 * 1024 ** 2),
                                Tensor(f"blk.{i}.ffn_up_exps.weight", [4096, 2048, 128], 12, 0, 1536 * 1024 ** 2)])
            demo = Model([], 0, sum(t.stored_bytes for t in tensors),
                         {"general.architecture": "mistral4", "general.name": "Illustrative MoE layout", "mistral4.block_count": 36},
                         tensors, "synthetic")
            hw = Hardware("Example", "x86_64", 16, 64 * GIB, 60 * GIB)
            print_plan(plan(demo, hw, gpu_gib=12))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Any started benchmark server has been stopped.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError_) as exc:
        print("MistralFit:", exc, file=sys.stderr)
        return 2
