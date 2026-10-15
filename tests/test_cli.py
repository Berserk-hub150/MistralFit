import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from mistralfit.cli import main
from tests.helpers import write_gguf


class CLITests(unittest.TestCase):
    def test_inspection_and_planning(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_gguf(Path(temp) / "model.gguf")
            out = Path(temp) / "plan.json"
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(["inspect", str(model)]), 0)
                self.assertEqual(main(["plan", str(model), "--ram-gib", "8", "--out", str(out)]), 0)
            self.assertIn("ESTIMATED", stdout.getvalue())
            self.assertTrue(out.exists())

    def test_missing_file_is_actionable_error(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(main(["inspect", "no-such-model.gguf"]), 2)
        self.assertIn("Missing model files", stderr.getvalue())

    def test_demo_is_unambiguously_synthetic(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(["demo"]), 0)
        self.assertIn("SYNTHETIC", stdout.getvalue())
        self.assertIn("NOT A REAL MODEL OR BENCHMARK", stdout.getvalue())

    def test_doctor_works_without_dependencies(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(["doctor", "--json"]), 0)
        self.assertIn("ram_available", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
