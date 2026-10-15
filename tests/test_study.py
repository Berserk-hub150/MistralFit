import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mistralfit.cli import main
from mistralfit.hardware import GIB, Hardware
from mistralfit.runtime import Server
from mistralfit.study import balanced_order, import_autotune, render_study, study, summarize
from tests.helpers import write_gguf
from tests.test_tuning import FIXTURE, INFO


class StudyTests(unittest.TestCase):
    def profile(self, folder, args=None):
        path = folder / "external.json"
        path.write_text(json.dumps({"name": "test export", "args": args or ["-ngl", "0", "-t", "2", "-c", "2048"]}))
        return path

    def test_import_keeps_selected_settings_and_records_context_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            p = self.profile(Path(temp), ["-ngl", "5", "-fa", "1", "-mmp", "0", "-c", "8192", "-b", "1024"])
            result = import_autotune(p, 4096, 0)
        self.assertEqual(result["profile"]["gpu_layers"], 5)
        self.assertEqual(result["overrides"]["--flash-attn"], "on")
        self.assertEqual(result["overrides"]["--batch-size"], "1024")
        self.assertIn("--no-mmap", result["extra_flags"])
        self.assertTrue(result["source"]["context_normalized"])
        self.assertNotIn("--ctx-size", result["overrides"])

    def test_import_rejects_uncontrolled_precision_devices_and_commands(self):
        cases = [(["--model", "other.gguf"], "Unsupported"), (["-ctk", "q8_0"], "f16"),
                 (["--host", "0.0.0.0"], "Unsupported"), (["-mg", "1"], "different GPU"),
                 (["-t", "2", "--threads", "8"], "Duplicate"), (["--parallel", "2"], "single"),
                 (["-sm", "layer"], "one GPU")]
        with tempfile.TemporaryDirectory() as temp:
            for args, message in cases:
                with self.subTest(args=args):
                    with self.assertRaisesRegex(ValueError, message):
                        import_autotune(self.profile(Path(temp), args), 4096, 0)

    def test_three_rounds_balance_every_position(self):
        names = ["upstream", "ours", "external"]
        order = balanced_order(names, 3)
        for position in range(3):
            self.assertEqual({round_[position] for round_ in order}, set(names))

    def test_preflight_blocks_without_a_gpu_and_runs_no_server(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = write_gguf(folder / "fixture.gguf")
            output = folder / "blocked.json"
            with patch("mistralfit.study.detect", return_value=Hardware("Fixture", "test", 4, 8 * GIB, 6 * GIB)), patch("mistralfit.study.Server") as server:
                result = study(str(model), None, output, competitor_profile=None, preflight=True)
            server.assert_not_called()
            self.assertEqual(result["status"], "blocked")
            self.assertIn("No NVIDIA", result["blockers"][0])
            self.assertEqual(result["runs"], [])
            self.assertTrue(output.with_suffix(".html").exists())

    def test_complete_protocol_pipeline_and_separate_functional_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = write_gguf(folder / "fixture.gguf")
            profile = self.profile(folder)
            output = folder / "study.json"
            def proxy(builder, **kwargs):
                def argv(port, key):
                    original = builder(port, key)
                    return [sys.executable, FIXTURE, *original[1:]]
                return Server(argv, **kwargs)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("mistralfit.study.detect", return_value=Hardware("Fixture", "test", 4, 64 * GIB, 60 * GIB)))
                for module in ("mistralfit.study", "mistralfit.tuning"):
                    stack.enter_context(patch(module + ".runtime_info", return_value=INFO))
                    stack.enter_context(patch(module + ".Server", side_effect=proxy))
                result = study(str(model), sys.executable, output, competitor_profile=profile,
                               token_budgets=(128,), rounds=3, repeats=1, warmups=1, threads=2,
                               allow_cpu=True, sample_interval=0.01, notify=lambda _: None)
            self.assertEqual(result["status"], "complete", [(r["candidate"], r.get("reason")) for r in result["runs"]])
            self.assertEqual(len(result["runs"]), 9)
            self.assertTrue(all(len(r["samples"]) == 3 for r in result["runs"]))
            self.assertTrue(all(c["complete"] and c["functional_checks_passed"] for c in result["summary"]))
            self.assertTrue(all(case["sample_count"] == 3 for c in result["summary"] for case in c["cases"]))
            self.assertTrue(all(r["startup_to_ready_seconds"] > 0 for r in result["runs"]))
            self.assertNotIn(str(folder), output.read_text())
            self.assertNotIn("Bearer ", output.read_text())
            self.assertNotIn("--api-key", output.read_text())
            result["limits"].append("<script>unsafe()</script>")
            self.assertNotIn("<script>", render_study(result))

    def test_cli_preflight_does_not_require_a_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = write_gguf(folder / "fixture.gguf")
            with contextlib.redirect_stdout(io.StringIO()), patch("mistralfit.study.detect", return_value=Hardware("Fixture", "test", 4, 8 * GIB, 6 * GIB)):
                code = main(["study", str(model), "--preflight", "--allow-cpu", "--out", str(folder / "report.json")])
            self.assertEqual(code, 0)

    def test_incomplete_rounds_do_not_receive_a_baseline_ratio(self):
        sample = {"case_id": "case", "decode_tokens_per_second": 10, "ttft_seconds": 1, "output_sha256": "same"}
        report = {"settings": {"rounds": 3, "repeats": 1}, "cases": [{"id": "case"}],
                  "candidates": [{"name": "upstream-auto"}, {"name": "other"}],
                  "runs": [{"candidate": name, "status": "ok", "samples": [sample]} for name in ["upstream-auto"] * 3 + ["other"] * 2]}
        result = summarize(report)
        self.assertTrue(result[0]["complete"])
        self.assertFalse(result[1]["complete"])
        self.assertIsNone(result[1]["cases"][0]["decode_ratio_vs_upstream_auto"])

    def test_interrupted_calibration_updates_parent_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            model = write_gguf(folder / "fixture.gguf")
            profile = self.profile(folder)
            output = folder / "study.json"
            with patch("mistralfit.study.detect", return_value=Hardware("Fixture", "test", 4, 64 * GIB, 60 * GIB)), \
                 patch("mistralfit.study.runtime_info", return_value=INFO), \
                 patch("mistralfit.study.tune", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    study(str(model), sys.executable, output, competitor_profile=profile, allow_cpu=True)
            self.assertEqual(json.loads(output.read_text())["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
