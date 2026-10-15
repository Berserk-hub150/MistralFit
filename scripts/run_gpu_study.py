"""Run preflight, a pinned external search and MistralFit's controlled study.

Requires local weights, llama.cpp binaries and a separately installed llama-autotune.
Nothing is downloaded, installed or rented by this script.
"""

from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mistralfit.runtime import clean_environment, resolve_server
from mistralfit.gguf import full_sha256
from mistralfit.study import REFERENCE_COMMIT, REFERENCE_REPOSITORY
from mistralfit.tuning import atomic_json


def run(argv, *, env=None, log=None):
    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT if log else None,
                               stdin=subprocess.DEVNULL)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=8)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--server", required=True)
    parser.add_argument("--autotune-root", type=Path, required=True, help="checkout pinned to the documented reference commit")
    parser.add_argument("--autotune-python", default=sys.executable, help="Python environment with the reference tool's dependencies")
    parser.add_argument("--out-dir", type=Path, default=Path("results/gpu-study"))
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials-b", type=int, default=15)
    parser.add_argument("--trials-c", type=int, default=40)
    parser.add_argument("--no-repack", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.rounds, args.repeats) < 1 or min(args.trials_b, args.trials_c) < 0:
        parser.error("Counts must be positive; search trial counts may be zero")
    model, directory = args.model.expanduser().resolve(), args.out_dir.expanduser().resolve()
    server = resolve_server(args.server)
    directory.mkdir(parents=True, exist_ok=True)
    runtime_directory = Path(server).parent
    bench = runtime_directory / ("llama-bench.exe" if os.name == "nt" else "llama-bench")
    if not bench.is_file():
        raise ValueError("The external tuner also needs llama-bench next to llama-server")
    common = [sys.executable, "-m", "mistralfit", "study", str(model), "--server", server,
              "--threads", str(args.threads), "--ctx", str(args.ctx), "--rounds", str(args.rounds), "--repeats", str(args.repeats)]
    if args.no_repack:
        common.append("--no-repack")
    preflight_path = directory / "preflight.json"
    status = run(common + ["--preflight", "--out", str(preflight_path)])
    if status:
        print("Preflight blocked. No external search or model generation was started.")
        return status
    reference = args.autotune_root.expanduser().resolve()
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(reference), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != REFERENCE_COMMIT or dirty:
        raise ValueError("Use a clean llama-autotune checkout at " + REFERENCE_COMMIT)
    environment = clean_environment()
    environment["LLAMA_CPP_DIR"] = str(runtime_directory)
    environment["PYTHONPATH"] = str(reference / "src")
    dependencies = json.loads(subprocess.check_output(
        [args.autotune_python, "-c", "import importlib.metadata,json,platform; print(json.dumps({'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['typer','pydantic','rich','sqlalchemy','optuna','psutil','structlog']}}))"],
        env=environment, text=True))
    profile = directory / "autotune-profile.json"
    if profile.exists():
        raise ValueError("Output directory already contains an external profile; choose a new directory to retain the previous run")
    metadata = {"schema_version": 1, "repository": REFERENCE_REPOSITORY, "revision": revision,
                "tracked_checkout_clean": True, "started_at": datetime.now(timezone.utc).isoformat(),
                "environment": dependencies, "llama_bench_sha256": full_sha256([str(bench)]),
                "stage_b_trials": args.trials_b, "stage_c_trials": args.trials_c,
                "scope": "External native search workload/budget differs from MistralFit calibration. Holdout measurements are separate."}
    started = time.monotonic()
    print("Running pinned llama-autotune search; see autotune-search.log for progress.", flush=True)
    with (directory / "autotune-search.log").open("w", encoding="utf-8") as log:
        code = run([args.autotune_python, "-m", "llama_autotune", "search", str(model),
                    "--objective", "max_generation_tps", "--trials-b", str(args.trials_b),
                    "--trials-c", str(args.trials_c), "--profile", str(profile)], env=environment, log=log)
    metadata.update(elapsed_seconds=time.monotonic() - started, return_code=code,
                    finished_at=datetime.now(timezone.utc).isoformat())
    if profile.exists():
        metadata["profile_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
    atomic_json(directory / "autotune-training.json", metadata)
    if code or not profile.exists():
        print("External search failed; its log/provenance were retained. No substitute result was invented.")
        return code or 2
    return run(common + ["--competitor-profile", str(profile), "--competitor-provenance", str(directory / "autotune-training.json"),
                         "--out", str(directory / "study.json")], env=clean_environment())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; the active child process was stopped.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("Study:", exc, file=sys.stderr)
        raise SystemExit(2)
