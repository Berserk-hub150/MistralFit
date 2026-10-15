import unittest

from mistralfit.hardware import GIB, cgroup_available


class ContainerMemoryTests(unittest.TestCase):
    def test_a_downloaded_model_cache_is_not_treated_as_unreclaimable(self):
        self.assertEqual(cgroup_available(8 * GIB, 7 * GIB, {"inactive_file": 5 * GIB}), 6 * GIB)

    def test_dirty_file_cache_is_not_added_as_immediate_headroom(self):
        self.assertEqual(cgroup_available(8 * GIB, 7 * GIB,
                                         {"inactive_file": 5 * GIB, "file_dirty": GIB, "file_writeback": GIB}), 4 * GIB)

    def test_anonymous_usage_and_missing_cache_counters_remain_constrained(self):
        self.assertEqual(cgroup_available(8 * GIB, 7 * GIB, {}), GIB)
        self.assertEqual(cgroup_available(8 * GIB, 9 * GIB, {}), 0)


if __name__ == "__main__":
    unittest.main()
