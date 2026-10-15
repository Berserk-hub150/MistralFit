"""Rebuild derived reports from the unchanged, native-timing collector files.

Run from any directory; no model, server, network or third-party package is needed.
"""

from pathlib import Path
import copy
import hashlib
import json
import math
import sys

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1]))

from mistralfit.report import render
from mistralfit.tuning import aggregate, atomic_json, choose_winner


def rebuild(source: Path, provenance: dict) -> dict:
    report = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    if report["status"] != "complete":
        raise ValueError("Only completed collector runs may be published")
    for entry in report["results"]:
        for sample in entry.get("samples", []):
            native = sample["backend_timings"]
            rate = native["predicted_per_second"]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid original native decode rate")
            if native["predicted_n"] != report["settings"]["tokens"] or native.get("cache_n", 0) != 0:
                raise ValueError("Token budget or prompt-cache contract differs")
            sample["collector_derived_rate_before_fix"] = sample["decode_tokens_per_second"]
            sample["decode_tokens_per_second"] = float(rate)
        if entry.get("status") == "ok":
            entry["metrics"] = aggregate(entry["samples"])
    report["settings"]["decode_rate_source"] = "backend timings.predicted_per_second"
    report["winner"] = choose_winner(
        report["results"], max_ttft=report["settings"].get("max_median_ttft_seconds")
    )
    baseline = next((x for x in report["results"] if x["profile"]["name"] == "upstream-auto" and x["status"] == "ok"), None)
    winner = next((x for x in report["results"] if x["profile"]["name"] == report["winner"]), None)
    if baseline and winner:
        report["speedup_vs_upstream_auto"] = (
            winner["metrics"]["median_decode_tokens_per_second"] / baseline["metrics"]["median_decode_tokens_per_second"]
        )
        report["winner_matches_baseline_output"] = (
            {s["output_sha256"] for s in winner["samples"]} == {s["output_sha256"] for s in baseline["samples"]}
        )
    report["provenance"] = provenance
    report["metric_reporting_correction"] = {
        "reason": "Collector used N/predicted_ms; this backend reports N-1 decode intervals. Native counters are unchanged.",
        "source": "Every displayed decode rate is now the original backend_timings.predicted_per_second value.",
        "original_collector_file": source.name,
        "original_collector_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reporting_code_commit": provenance["metric_reporting_source_commit"],
    }
    for note in (
        "Short CPU microbenchmark under an 8 GiB container cap with weights larger than RAM; not representative GPU or resident-model performance.",
        "CPU-only and upstream-auto both execute on CPU here. This run does not establish an expert-offload or GPU optimization advantage.",
    ):
        if note not in report["limits"]:
            report["limits"].append(note)
    return report


def main():
    provenance = json.loads((DIRECTORY / "provenance.json").read_text(encoding="utf-8"))
    for name in ("probe", "comparison"):
        report = rebuild(DIRECTORY / f"{name}.collector.json", provenance)
        prompt = (DIRECTORY / "prompt.txt").read_bytes()
        if hashlib.sha256(prompt).hexdigest() != report["settings"]["prompt_sha256"]:
            raise ValueError("Public prompt does not match the recorded prompt hash")
        atomic_json(DIRECTORY / f"{name}.json", report)
        (DIRECTORY / f"{name}.html").write_text(render(report), encoding="utf-8")
        print(f"Rebuilt {name}.json and {name}.html from native timing counters")


if __name__ == "__main__":
    main()
