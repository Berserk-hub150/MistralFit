import unittest

from mistralfit.gguf import Model, Tensor
from mistralfit.hardware import GIB, GPU, Hardware
from mistralfit.planner import kv_estimate, placement, plan


def example():
    tensors = [Tensor("token_embd.weight", [4, 8], 0, 0, GIB)]
    for i in range(4):
        tensors.extend([Tensor(f"blk.{i}.attn_q.weight", [4, 4], 0, 0, GIB),
                        Tensor(f"blk.{i}.ffn_up_exps.weight", [4, 4, 8], 0, 0, 4 * GIB)])
    tensors.append(Tensor("output.weight", [4, 8], 0, 0, GIB))
    return Model(["model.gguf"], 22 * GIB, 22 * GIB,
                 {"general.architecture": "mistral4", "mistral4.block_count": 4}, tensors, "fixture")


class PlannerTests(unittest.TestCase):
    def test_last_blocks_are_offloaded(self):
        self.assertEqual(placement(example(), 2, 0), 10 * GIB)
        self.assertEqual(placement(example(), 5, 4), 5 * GIB)
        self.assertEqual(placement(example(), 2, 3), 6 * GIB)

    def test_no_gpu(self):
        hw = Hardware("Linux", "x86_64", 8, 64 * GIB, 60 * GIB)
        profiles = plan(example(), hw)["profiles"]
        self.assertEqual([p["name"] for p in profiles], ["upstream-auto", "cpu-only"])

    def test_cpu_experts_can_fit_small_gpu(self):
        hw = Hardware("Linux", "x86_64", 8, 64 * GIB, 60 * GIB, [GPU(0, "Fixture", 12 * GIB, 11 * GIB)])
        result = plan(example(), hw)
        profiles = {p["name"]: p for p in result["profiles"]}
        self.assertFalse(profiles["gpu-all"]["eligible"])
        self.assertTrue(profiles["cpu-experts"]["eligible"])
        self.assertIn("cached-experts", profiles)

    def test_insufficient_host_ram(self):
        hw = Hardware("Linux", "x86_64", 8, 8 * GIB, 6 * GIB)
        self.assertTrue(all(not p["eligible"] for p in plan(example(), hw)["profiles"]))
        self.assertTrue(all(p["eligible"] for p in plan(example(), hw, allow_paging=True)["profiles"]))

    def test_unified_memory_is_not_added_twice(self):
        hw = Hardware("Darwin", "arm64", 8, 16 * GIB, 14 * GIB, unified_memory=True)
        result = plan(example(), hw)
        self.assertTrue(all(not p["eligible"] for p in result["profiles"]))
        self.assertNotIn("cached-experts", [p["name"] for p in result["profiles"]])

    def test_unknown_ram_is_not_treated_as_infinite(self):
        hw = Hardware("Unknown", "Unknown", 8, 0, 0)
        self.assertTrue(all(not p["eligible"] for p in plan(example(), hw)["profiles"]))
        self.assertTrue(plan(example(), hw, ram_gib=40)["profiles"][0]["eligible"])

    def test_kv_metadata_formula(self):
        model = example()
        model.metadata.update({"mistral4.attention.head_count": 8, "mistral4.attention.head_count_kv": 2,
                               "mistral4.attention.key_length": 128, "mistral4.attention.value_length": 128})
        self.assertEqual(kv_estimate(model, 4096)[0], 4096 * 4 * 2 * 256 * 2)


if __name__ == "__main__":
    unittest.main()
