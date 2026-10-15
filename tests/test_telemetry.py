import math
import os
import time
import unittest
from unittest.mock import patch

from mistralfit.telemetry import ResourceSampler, counter_delta, process_counters, proc_directory


class TelemetryTests(unittest.TestCase):
    def test_missing_counters_remain_null_and_deltas_are_nonnegative(self):
        result = counter_delta({"cpu_seconds": None, "read_bytes": 100}, {"cpu_seconds": 3, "read_bytes": 180})
        self.assertIsNone(result["cpu_seconds"])
        self.assertIsNone(result["major_faults"])
        self.assertEqual(result["read_bytes"], 80)

    def test_sampling_an_actual_process_and_stopping_thread(self):
        with patch("mistralfit.telemetry.shutil.which", return_value=None):
            sampler = ResourceSampler(0.01)
        sampler.start(os.getpid())
        time.sleep(0.04)
        sampler.stop()
        result = sampler.summary()
        self.assertFalse(sampler._thread.is_alive())
        self.assertIsNone(result["observed_peak_gpu_bytes"])
        if os.name == "nt" or os.path.exists("/proc/self/status"):
            self.assertGreater(result["observed_peak_rss_bytes"], 0)
            self.assertGreater(result["rss_sample_count"], 0)

    def test_invalid_intervals_are_rejected(self):
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                ResourceSampler(value)

    def test_host_procfs_pid_mismatch_does_not_sample_an_unrelated_pid(self):
        if os.name == "nt":
            self.skipTest("Linux procfs mapping regression")
        with patch("mistralfit.telemetry.Path.read_text", return_value=f"Pid:\t{os.getpid() + 10000}\n"):
            self.assertIsNone(proc_directory(os.getpid() + 1))
            self.assertTrue(all(v is None for v in process_counters(os.getpid() + 1).values()))


if __name__ == "__main__":
    unittest.main()
