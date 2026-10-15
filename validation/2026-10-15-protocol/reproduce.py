"""Compare HTTP client behavior only. No model, GPU or inference is involved."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mistralfit.runtime import stream_completion
from mistralfit.tuning import atomic_json
from tests.test_runtime import fixture

BASELINE = "52554f55b6e87755a6f91f5385b0393584a1d61e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/protocol-regression.json")
    args = parser.parse_args()
    source = subprocess.check_output(["git", "show", BASELINE + ":mistralfit/runtime.py"], cwd=ROOT)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "runtime_before.py"
        path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("mistralfit._baseline_runtime", path)
        before = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(before)
        result = {"scope": "HTTP protocol regression fixture only. No inference, model speed, GPU speed or quality measurement.",
                  "python": platform.python_version(), "baseline_commit": BASELINE,
                  "baseline_runtime_sha256": hashlib.sha256(source).hexdigest(),
                  "fixed_runtime_sha256": hashlib.sha256((ROOT / "mistralfit/runtime.py").read_bytes()).hexdigest(),
                  "fixture_sha256": hashlib.sha256((ROOT / "tests/fake_server.py").read_bytes()).hexdigest(),
                  "repetitions_per_case": 3, "results": []}
        for label, complete in (("before", before.stream_completion), ("after", stream_completion)):
            for mode in ("slow-headers", "slow-line", "linger-after-final"):
                timeout = 2 if mode == "linger-after-final" else 0.08
                samples = []
                for _ in range(3):
                    with fixture(mode) as local:
                        started = time.monotonic()
                        try:
                            response = complete(local.base, "HTTP fixture only", tokens=4, timeout=timeout, api_key=local.api_key)
                            sample = {"outcome": "completed", "generated_fixture_tokens": response["generated_tokens"],
                                      "fixture_output_sha256": response["output_sha256"]}
                        except Exception as exc:
                            sample = {"outcome": type(exc).__name__, "message": str(exc)}
                        sample["elapsed_seconds"] = time.monotonic() - started
                        samples.append(sample)
                result["results"].append({"client": label, "fixture": mode, "configured_timeout_seconds": timeout,
                                           "median_elapsed_seconds": statistics.median(s["elapsed_seconds"] for s in samples),
                                           "samples": samples})
        old = next(r for r in result["results"] if r["client"] == "before" and r["fixture"] == "linger-after-final")
        new = next(r for r in result["results"] if r["client"] == "after" and r["fixture"] == "linger-after-final")
        assert {s["fixture_output_sha256"] for s in old["samples"]} == {s["fixture_output_sha256"] for s in new["samples"]}
        assert all(s["generated_fixture_tokens"] == 4 for s in old["samples"] + new["samples"])
        atomic_json(args.out, result)
        for entry in result["results"]:
            print(entry["client"], entry["fixture"], f"{entry['median_elapsed_seconds']:.4f} seconds")
        print("Saved protocol evidence only:", args.out)


if __name__ == "__main__":
    main()
