import struct
from pathlib import Path


def string(value):
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, *, architecture="mistral4", metadata=None, tensor_specs=None):
    values = {"general.architecture": architecture, "general.name": "Test fixture only",
              architecture + ".block_count": 2, **(metadata or {})}
    specs = tensor_specs or [("token_embd.weight", [4, 4], 64),
                             ("blk.0.ffn_up_exps.weight", [4, 4, 2], 128),
                             ("blk.1.attn_q.weight", [4, 4], 64)]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(specs), len(values))
    for key, value in values.items():
        if isinstance(value, str):
            header += string(key) + struct.pack("<I", 8) + string(value)
        elif isinstance(value, list):
            header += string(key) + struct.pack("<IIQ", 9, 8, len(value)) + b"".join(string(x) for x in value)
        else:
            header += string(key) + struct.pack("<IQ", 10, value)
    offset = 0
    for name, dims, size in specs:
        assert size % 32 == 0
        header += string(name) + struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", 0, offset)
        offset += size
    header += b"\0" * ((-len(header)) % 32)
    path.write_bytes(header + b"\0" * offset)
    return path
