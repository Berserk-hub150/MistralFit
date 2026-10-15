import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from mistralfit.planner import Profile
from mistralfit.runtime import RuntimeError_, Server, clean_environment, command, json_request, stream_completion, unsupported

FIXTURE = str(Path(__file__).with_name("fake_server.py"))


def fixture(mode="ok", timeout=10):
    return Server(lambda port, key: [sys.executable, FIXTURE, "--port", str(port), "--api-key", key,
                                    "--fixture-mode", mode], startup_timeout=timeout)


class RuntimeTests(unittest.TestCase):
    def test_command_has_fixed_single_slot_and_no_shell(self):
        argv = command("/path with spaces/server", "/model with spaces/m.gguf", Profile("cpu", 0),
                       context=4096, threads=4, port=8000, help_text="--fit --perf --jinja --cache-ram --ctx-checkpoints")
        self.assertIn("/model with spaces/m.gguf", argv)
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "4096")
        self.assertEqual(argv[argv.index("--parallel") + 1], "1")
        self.assertEqual(argv[argv.index("--fit") + 1], "off")
        self.assertEqual(argv[argv.index("--device") + 1], "none")
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "0")

    def test_cache_profile_requires_capability(self):
        profile = Profile("cache", 99, 36, 4096)
        self.assertEqual(unsupported(profile, "--fit --n-cpu-moe"), ["--moe-cache-mib"])

    def test_explicit_no_repack_is_passed_to_backend(self):
        argv = command("server", "model.gguf", Profile("cpu", 0), context=1024, threads=4,
                       port=8000, help_text="--fit --no-repack", repack=False)
        self.assertIn("--no-repack", argv)

    def test_missing_no_repack_capability_is_reported(self):
        with self.assertRaisesRegex(RuntimeError_, "lacks --no-repack"):
            command("server", "model.gguf", Profile("cpu", 0), context=1024, threads=4,
                    port=8000, help_text="--fit", repack=False)

    def test_environment_does_not_override_experiment(self):
        with patch.dict(os.environ, {"LLAMA_ARG_CTX_SIZE": "8", "MISTRALFIT_TEST": "keep"}):
            env = clean_environment()
        self.assertNotIn("LLAMA_ARG_CTX_SIZE", env)
        self.assertEqual(env["MISTRALFIT_TEST"], "keep")

    def test_real_http_stream_and_cleanup(self):
        with fixture() as local:
            result = stream_completion(local.base, "test", tokens=4, api_key=local.api_key)
            process = local.process
            # The backend reports 3 decode intervals for 4 emitted tokens.
            self.assertAlmostEqual(result["decode_tokens_per_second"], 150)
            self.assertGreater(result["ttft_seconds"], 0)
            self.assertEqual(result["generated_tokens"], 4)
            self.assertNotIn("fixture", json.dumps(result))
        self.assertIsNotNone(process.poll())

    def test_deadline_covers_trickling_headers_and_partial_lines(self):
        for mode in ("slow-headers", "slow-line"):
            with self.subTest(mode=mode), fixture(mode) as local:
                started = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, "wall-clock deadline"):
                    stream_completion(local.base, "test", tokens=4, timeout=0.08, api_key=local.api_key)
                self.assertLess(time.monotonic() - started, 0.45)

    def test_final_event_does_not_wait_for_connection_close(self):
        with fixture("linger-after-final") as local:
            started = time.monotonic()
            result = stream_completion(local.base, "test", tokens=4, timeout=2, api_key=local.api_key)
            self.assertEqual(result["generated_tokens"], 4)
            self.assertLess(time.monotonic() - started, 0.4)

    def test_incomplete_or_incomparable_trials_are_rejected(self):
        for mode, message in [("missing-final", "before"), ("wrong-count", "expected"),
                              ("cached", "reused"), ("missing-timing", "expected"),
                              ("missing-rate", "valid decode rate"), ("truncated", "truncated")]:
            with self.subTest(mode=mode), fixture(mode) as local:
                with self.assertRaisesRegex(RuntimeError_, message):
                    stream_completion(local.base, "test", tokens=4, api_key=local.api_key)

    def test_authentication_is_required_for_props(self):
        from urllib.error import HTTPError
        with fixture() as local:
            with self.assertRaises(HTTPError):
                json_request(local.base, "/props")

    def test_random_credential_with_leading_dash_remains_an_argument(self):
        with patch("mistralfit.runtime.secrets.token_urlsafe", return_value="-leading-dash"):
            with fixture() as local:
                response = json_request(local.base, "/props", api_key=local.api_key)
                self.assertEqual(response["default_generation_settings"]["n_ctx"], 4096)

    def test_protocol_fixture_does_not_require_reverse_dns(self):
        probe = """import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError('Reverse DNS is forbidden in this regression test')
socket.getfqdn = forbidden
runpy.run_path(sys.argv[1], run_name='__main__')
"""
        local = Server(lambda port, key: [sys.executable, "-c", probe, FIXTURE,
                                         "--port", str(port), "--api-key", key], startup_timeout=10)
        with local:
            result = stream_completion(local.base, "test", tokens=4, api_key=local.api_key)
            self.assertEqual(result["generated_tokens"], 4)

    def test_startup_crash_is_reaped(self):
        local = Server(lambda *_: [sys.executable, "-c", "raise SystemExit(7)"], startup_timeout=5)
        with self.assertRaisesRegex(RuntimeError_, "exited"):
            with local:
                pass
        self.assertEqual(local.process.returncode, 7)

    def test_startup_timeout_terminates_child(self):
        local = Server(lambda *_: [sys.executable, "-c", "import time; time.sleep(20)"], startup_timeout=0.05)
        with self.assertRaisesRegex(RuntimeError_, "timed out.*last health check"):
            with local:
                pass
        self.assertIsNotNone(local.process.poll())


if __name__ == "__main__":
    unittest.main()
