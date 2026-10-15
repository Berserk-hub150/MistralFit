import struct
import tempfile
import unittest
from pathlib import Path

from mistralfit.gguf import GGUFError, full_sha256, inspect_model
from tests.helpers import write_gguf


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_layout_experts_and_header_identity(self):
        path = write_gguf(self.root / "small.gguf")
        model = inspect_model(path)
        self.assertEqual(model.blocks, 2)
        self.assertEqual(model.weight_bytes, 256)
        self.assertEqual(model.expert_bytes, 128)
        before = model.header_fingerprint
        checksum_before = full_sha256(model.paths)
        data = path.read_bytes()
        path.write_bytes(data[:-1] + b"X")
        self.assertEqual(inspect_model(path).header_fingerprint, before)
        self.assertNotEqual(checksum_before, full_sha256(model.paths))

    def test_tokenizer_strings_are_skipped(self):
        path = write_gguf(self.root / "m.gguf", metadata={"tokenizer.ggml.tokens": ["hello", "world", "α"]})
        self.assertNotIn("tokenizer.ggml.tokens", inspect_model(path).metadata)

    def test_bad_magic_and_truncated_header(self):
        path = self.root / "bad.gguf"
        for data in (b"nope", b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 1, 1)):
            path.write_bytes(data)
            with self.assertRaises(GGUFError):
                inspect_model(path)

    def test_unsupported_version(self):
        path = self.root / "v.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 99, 1, 0))
        with self.assertRaisesRegex(GGUFError, "version"):
            inspect_model(path)

    def test_truncated_unquantized_payload(self):
        path = write_gguf(self.root / "short.gguf")
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(GGUFError, "payload"):
            inspect_model(path)

    def test_invalid_alignment(self):
        path = write_gguf(self.root / "m.gguf", metadata={"general.alignment": 3})
        with self.assertRaisesRegex(GGUFError, "alignment"):
            inspect_model(path)

    def test_complete_shard_set(self):
        first = write_gguf(self.root / "m-00001-of-00002.gguf", metadata={"split.count": 2, "split.no": 0},
                           tensor_specs=[("blk.0.attn_q.weight", [4, 4], 64)])
        write_gguf(self.root / "m-00002-of-00002.gguf", metadata={"split.count": 2, "split.no": 1},
                   tensor_specs=[("blk.1.attn_q.weight", [4, 4], 64)])
        model = inspect_model(first)
        self.assertEqual(len(model.paths), 2)
        self.assertEqual(model.weight_bytes, 128)

    def test_missing_shard(self):
        first = write_gguf(self.root / "m-00001-of-00002.gguf", metadata={"split.count": 2})
        with self.assertRaisesRegex(GGUFError, "Missing"):
            inspect_model(first)

    def test_reject_nonfirst_shard(self):
        with self.assertRaisesRegex(GGUFError, "first shard"):
            inspect_model(self.root / "m-00002-of-00002.gguf")

    def test_reject_mixed_shards(self):
        first = write_gguf(self.root / "m-00001-of-00002.gguf", metadata={"split.count": 2, "split.no": 0},
                           tensor_specs=[("blk.0.attn_q.weight", [4, 4], 64)])
        write_gguf(self.root / "m-00002-of-00002.gguf", architecture="llama", metadata={"split.count": 2, "split.no": 1},
                   tensor_specs=[("blk.1.attn_q.weight", [4, 4], 64)])
        with self.assertRaisesRegex(GGUFError, "architecture"):
            inspect_model(first)

    def test_reject_duplicate_tensors(self):
        path = write_gguf(self.root / "dup.gguf", tensor_specs=[("blk.0.x", [4], 64), ("blk.0.x", [4], 64)])
        with self.assertRaisesRegex(GGUFError, "Duplicate"):
            inspect_model(path)


if __name__ == "__main__":
    unittest.main()
