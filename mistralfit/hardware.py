from __future__ import annotations

import csv
import ctypes
import io
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 1024 ** 3


@dataclass
class GPU:
    index: int
    name: str
    total_bytes: int
    free_bytes: int


@dataclass
class Hardware:
    system: str
    machine: str
    cpu_count: int
    ram_total: int
    ram_available: int
    gpus: list[GPU] = field(default_factory=list)
    unified_memory: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _run(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=8).strip()


def detect() -> Hardware:
    system, machine = platform.system(), platform.machine()
    notes, total, available = [], 0, 0
    try:
        if system == "Linux":
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                values[key] = int(value.split()[0]) * 1024
            total, available = values["MemTotal"], values.get("MemAvailable", values.get("MemFree", 0))
            # Respect container memory limits when this is run inside one.
            limit, usage = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
            if limit.exists() and limit.read_text().strip().isdigit():
                cap = int(limit.read_text())
                if 0 < cap < total:
                    total = cap
                    available = min(available, max(0, cap - int(usage.read_text())))
                    notes.append("RAM constrained by the container cgroup.")
        elif system == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ("total", "available", "pagefile", "available_pagefile", "virtual", "available_virtual", "extended")]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            total, available = status.total, status.available
        elif system == "Darwin":
            total = int(_run(["sysctl", "-n", "hw.memsize"]))
            # Available memory on macOS is pressure-dependent. Do not equate total with available.
            notes.append("macOS available RAM is not estimated; use --ram-gib for a measured budget.")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        notes.append("Memory discovery incomplete: " + str(exc))
    gpus = []
    try:
        text = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"])
        for row in csv.reader(io.StringIO(text), skipinitialspace=True):
            if len(row) == 4:
                gpus.append(GPU(int(row[0]), row[1], int(float(row[2]) * 1024 ** 2), int(float(row[3]) * 1024 ** 2)))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    unified = system == "Darwin" and machine.lower() in ("arm64", "aarch64")
    if not gpus and not unified:
        notes.append("No NVIDIA GPU detected; AMD/Intel GPU budgets can be supplied manually.")
    return Hardware(system, machine, os.cpu_count() or 1, total, available, gpus, unified, notes)
