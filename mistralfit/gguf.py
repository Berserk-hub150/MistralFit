"""Bounded GGUF v2/v3 inspection without importing or reading model weights."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
MAX_STRING = 64 * 1024 * 1024
MAX_ITEMS = 2_000_000
SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
BLOCK = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    pass


@dataclass
class Tensor:
    name: str
    dimensions: list[int]
    type_id: int
    offset: int
    stored_bytes: int = 0

    @property
    def block(self) -> int | None:
        match = BLOCK.match(self.name)
        return int(match.group(1)) if match else None

    @property
    def routed_expert(self) -> bool:
        return len(self.dimensions) >= 3 and ("_exps" in self.name or ".experts." in self.name)


@dataclass
class Model:
    paths: list[str]
    file_bytes: int
    weight_bytes: int
    metadata: dict
    tensors: list[Tensor]
    header_fingerprint: str

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def blocks(self) -> int:
        inferred = max((t.block for t in self.tensors if t.block is not None), default=-1) + 1
        return int(self.metadata.get(self.architecture + ".block_count", inferred))

    @property
    def expert_bytes(self) -> int:
        return sum(t.stored_bytes for t in self.tensors if t.routed_expert)

    def summary(self) -> dict:
        return {"files": [Path(p).name for p in self.paths], "file_bytes": self.file_bytes,
                "weight_bytes": self.weight_bytes, "architecture": self.architecture,
                "name": self.metadata.get("general.name", "unnamed"), "blocks": self.blocks,
                "routed_expert_bytes": self.expert_bytes, "tensor_count": len(self.tensors),
                "header_fingerprint": self.header_fingerprint,
                "identity_note": "SHA-256 of structural headers and file sizes; NOT a full weights checksum."}


class Reader:
    def __init__(self, handle: BinaryIO):
        self.f = handle
        self.size = handle.seek(0, 2)
        handle.seek(0)

    def read(self, n: int) -> bytes:
        if n < 0 or n > self.size - self.f.tell():
            raise GGUFError("Truncated GGUF header or invalid length")
        data = self.f.read(n)
        if len(data) != n:
            raise GGUFError("Truncated GGUF")
        return data

    def skip(self, n: int) -> None:
        if n < 0 or n > self.size - self.f.tell():
            raise GGUFError("GGUF field extends beyond the file")
        self.f.seek(n, 1)

    def number(self, fmt: str):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, keep: bool = True):
        length = self.number("Q")
        if length > MAX_STRING:
            raise GGUFError("Unreasonably large GGUF string")
        if not keep:
            self.skip(length)
            return None
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("Invalid UTF-8 in GGUF header") from exc

    def value(self, kind: int, keep: bool = True, depth: int = 0):
        if depth > 4:
            raise GGUFError("Nested metadata is too deep")
        if kind in SCALARS:
            fmt = SCALARS[kind]
            if keep:
                return self.number(fmt)
            self.skip(struct.calcsize("<" + fmt))
        elif kind == 8:
            return self.string(keep)
        elif kind == 9:
            element, count = self.number("I"), self.number("Q")
            if count > MAX_ITEMS:
                raise GGUFError("Too many GGUF array entries")
            if element in SCALARS:
                self.skip(count * struct.calcsize("<" + SCALARS[element]))
            else:
                for _ in range(count):
                    self.value(element, False, depth + 1)
        else:
            raise GGUFError(f"Unknown GGUF metadata value type {kind}")
        return None


def _inspect_file(path: Path):
    with path.open("rb") as handle:
        r = Reader(handle)
        if r.read(4) != b"GGUF":
            raise GGUFError("Expected little-endian GGUF magic")
        version = r.number("I")
        if version not in (2, 3):
            raise GGUFError(f"Unsupported GGUF version {version}")
        nt, nkv = r.number("Q"), r.number("Q")
        if not 0 < nt <= MAX_ITEMS or nkv > MAX_ITEMS:
            raise GGUFError("Invalid GGUF entry counts")
        metadata = {}
        for _ in range(nkv):
            key, kind = r.string(), r.number("I")
            keep = key.startswith(("general.", "split.")) or any(
                term in key for term in ("block_count", "embedding_length", "context_length", "attention.", "expert_count"))
            value = r.value(kind, keep)
            if keep and value is not None:
                metadata[key] = value
        tensors = []
        for _ in range(nt):
            name, ndim = r.string(), r.number("I")
            if not 1 <= ndim <= 4:
                raise GGUFError("Invalid tensor rank")
            dims = [r.number("Q") for _ in range(ndim)]
            if any(d <= 0 for d in dims):
                raise GGUFError("Invalid tensor dimension")
            tensors.append(Tensor(name, dims, r.number("I"), r.number("Q")))
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment > 65536 or alignment & (alignment - 1):
            raise GGUFError("Invalid GGUF alignment")
        header_end = handle.tell()
        data_start = (header_end + alignment - 1) // alignment * alignment
        data_size = r.size - data_start
        ordered = sorted(tensors, key=lambda t: t.offset)
        for i, tensor in enumerate(ordered):
            end = ordered[i + 1].offset if i + 1 < len(ordered) else data_size
            if tensor.offset < 0 or tensor.offset % alignment or end <= tensor.offset or end > data_size:
                raise GGUFError("Invalid or overlapping tensor offsets")
            tensor.stored_bytes = end - tensor.offset
            # Validate the unquantized encodings we can size without GGML.
            # Quantized/unknown types retain a structural span check only.
            if tensor.type_id in (0, 1):
                elements = 1
                for dimension in tensor.dimensions:
                    elements *= dimension
                required = elements * (4 if tensor.type_id == 0 else 2)
                if required > tensor.stored_bytes:
                    raise GGUFError("Tensor payload is shorter than its declared shape")
        handle.seek(0)
        digest = hashlib.sha256()
        remaining = header_end
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            digest.update(chunk)
            remaining -= len(chunk)
        digest.update(str(r.size).encode("ascii"))
        return metadata, tensors, r.size, digest.hexdigest()


def inspect_model(filename: str | Path) -> Model:
    path = Path(filename).expanduser().resolve()
    match = SHARD.match(path.name)
    if match:
        prefix, index, count = match.group(1), int(match.group(2)), int(match.group(3))
        if index != 1:
            raise GGUFError("Pass the first shard (-00001-of-xxxxx.gguf)")
        if not 1 <= count <= 10000:
            raise GGUFError("Invalid shard count")
        paths = [path.with_name(f"{prefix}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    else:
        paths = [path]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise GGUFError("Missing model files: " + ", ".join(missing[:8]))
    all_tensors, metadata, sizes, digests = [], {}, 0, []
    for index, part in enumerate(paths):
        md, tensors, size, digest = _inspect_file(part)
        if index == 0:
            metadata = md
        if md.get("general.architecture") != metadata.get("general.architecture"):
            raise GGUFError("Shards disagree on architecture")
        if int(md.get("split.count", len(paths))) != len(paths):
            raise GGUFError("GGUF split.count does not match the supplied shard set")
        if "split.no" in md and int(md["split.no"]) != index:
            raise GGUFError("GGUF shard index mismatch")
        all_tensors.extend(tensors)
        sizes += size
        digests.append(digest)
    names = [t.name for t in all_tensors]
    if len(names) != len(set(names)):
        raise GGUFError("Duplicate tensor names across model shards")
    fingerprint = hashlib.sha256("\n".join(digests).encode()).hexdigest()
    return Model([str(p) for p in paths], sizes, sum(t.stored_bytes for t in all_tensors), metadata, all_tensors, fingerprint)


def full_sha256(paths: list[str]) -> dict[str, str]:
    result = {}
    for filename in paths:
        digest = hashlib.sha256()
        with open(filename, "rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        result[Path(filename).name] = digest.hexdigest()
    return result
