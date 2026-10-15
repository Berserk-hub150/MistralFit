import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mistralfit.hardware import GIB, Hardware
from mistralfit.planner import plan
from mistralfit.report import render
from mistralfit.runtime import Server
from mistralfit.tuning import aggregate, choose_winner, effective_context, tune
from tests.test_planner import example

FIXTURE = str(Path(__file__).with_name("fake_server.py"))
INFO = {"version": "TEST PROTOCOL FIXTURE — NOT INFERENCE", "help": "--model --fit --n-cpu-moe --moe-cache-mib --perf --jinja --api-key",
        "capabilities": {}}


class TuningTests(unittest.TestCase):
    def test_winner_is_filtered_by_ttft_and_success(self):
        results = [{"profile": {"name": "fast"}, "status": "ok", "metrics": {"median_decode_tokens_per_second": 50, "median_ttft_seconds": 8}},
                   {"profile": {"name": "responsive"}, "status": "ok", "metrics": {"median_decode_tokens_per_second": 40, "median_ttft_seconds": 1}},
                   {"profile": {"name": "failed"}, "status": "failed"}]
        self.assertEqual(choose_winner(results), "fast")
        self.assertEqual(choose_winner(results, max_ttft=2), "responsive")
        self.assertIsNone(choose_winner(results, max_ttft=0.5))

    def test_context_contract(self):
        self.assertEqual(effective_context({"default_generation_settings": {"n_ctx": 4096}}), 4096)
        self.assertIsNone(effective_context({}))

    def test_medians_and_variation(self):
        samples = [{"decode_tokens_per_second": n, "ttft_seconds": 1, "output_sha256": "same"} for n in (2, 10, 11)]
        result = aggregate(samples)
        self.assertEqual(result["median_decode_tokens_per_second"], 10)
        self.assertAlmostEqual(result["relative_mad"], 0.1)
        self.assertTrue(result["repeat_output_consistent"])

    def run_experiment(self, mode="ok"):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        output = Path(temp.name) / "nested" / "result.json"
        planning = plan(example(), Hardware("Fixture", "test", 4, 64 * GIB, 60 * GIB))

        def make_server(builder, **kwargs):
            def proxy(port, key):
                argv = builder(port, key)
                return [sys.executable, FIXTURE, *argv[1:], "--fixture-mode", mode]
            return Server(proxy, **kwargs)

        with patch("mistralfit.tuning.runtime_info", return_value=INFO), patch("mistralfit.tuning.Server", side_effect=make_server):
            report = tune(planning, "private-model.gguf", "fixture-executable", output, prompt="PRIVATE WORKLOAD TEXT",
                          tokens=4, repeats=2, warmups=1, startup_timeout=5, notify=lambda _: None)
        return report, output

    def test_pipeline_with_real_subprocess_and_http_fixture(self):
        report, path = self.run_experiment()
        self.assertEqual(report["status"], "complete")
        self.assertIn(report["winner"], ("upstream-auto", "cpu-only"))
        self.assertEqual(report["speedup_vs_upstream_auto"], 1)
        self.assertEqual(len(report["results"][0]["samples"]), 2)
        self.assertEqual(json.loads(path.read_text())["status"], "complete")
        self.assertNotIn("PRIVATE WORKLOAD TEXT", path.read_text())
        self.assertNotIn("private-model.gguf", path.read_text())

    def test_context_change_excludes_trial(self):
        report, _ = self.run_experiment("wrong-context")
        self.assertEqual(report["status"], "no_successful_candidate")
        self.assertIsNone(report["winner"])
        self.assertTrue(all(r["status"] == "failed" for r in report["results"]))

    def test_html_escapes_untrusted_fields(self):
        report, _ = self.run_experiment()
        report["plan"]["model"]["name"] = '<script>alert("x")</script>'
        report["results"][0]["reason"] = "<img src=x onerror=alert(1)>"
        output = render(report)
        self.assertNotIn("<script>", output)
        self.assertNotIn("<img", output)
        self.assertIn("&lt;script&gt;", output)


if __name__ == "__main__":
    unittest.main()
