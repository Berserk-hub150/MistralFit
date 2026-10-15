"""Sample one server process. Observed peaks are not guaranteed maxima."""

from __future__ import annotations

import csv
import ctypes
import io
import math
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path


def environment_details() -> dict:
    details = {"platform": platform.platform(), "python": platform.python_version(), "cpu_model": None,
               "affinity_cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               "cgroup_cpu_quota_cores": None, "nvidia_driver_versions": None}
    try:
        if os.name == "nt":
            details["cpu_model"] = os.environ.get("PROCESSOR_IDENTIFIER")
        else:
            details["cpu_model"] = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None)
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            details["cgroup_cpu_quota_cores"] = int(quota) / int(period)
    except (OSError, ValueError, ZeroDivisionError):
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            result = subprocess.run([smi, "--query-gpu=index,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=2, check=True)
            details["nvidia_driver_versions"] = [{"index": int(row[0]), "driver": row[1].strip()}
                                                   for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 2]
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return details


def proc_directory(pid: int) -> Path | None:
    if pid == os.getpid():
        return Path("/proc/self")
    try:
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
        if int(status["Pid"]) != os.getpid():
            # A host-mounted procfs can use different IDs from Popen. Never sample an unrelated host PID.
            return None
    except (OSError, ValueError, KeyError):
        return None
    return Path("/proc") / str(pid)


def process_counters(pid: int) -> dict:
    values = dict.fromkeys(("rss_bytes", "cpu_seconds", "major_faults", "read_bytes", "write_bytes"))
    if os.name == "nt":
        from ctypes import wintypes

        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_rss", "rss", "peak_pool_paged", "pool_paged", "peak_pool_nonpaged",
                    "pool_nonpaged", "pagefile", "peak_pagefile")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if handle:
            try:
                memory = Memory()
                memory.cb = ctypes.sizeof(memory)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb):
                    values["rss_bytes"] = memory.rss
            finally:
                kernel.CloseHandle(handle)
        return values
    process = proc_directory(pid)
    if process is None:
        return values
    try:
        for line in (process / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                values["rss_bytes"] = int(line.split()[1]) * 1024
        # comm can contain spaces and parentheses; fields start after its final ')'.
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        values["major_faults"] = int(fields[9])
        values["cpu_seconds"] = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        pass
    try:
        counters = dict(line.split(":", 1) for line in (process / "io").read_text().splitlines())
        values["read_bytes"] = int(counters["read_bytes"])
        values["write_bytes"] = int(counters["write_bytes"])
    except (OSError, ValueError, KeyError):
        pass
    return values


def counter_delta(before: dict, after: dict) -> dict:
    return {key: max(0, after[key] - before[key]) if before.get(key) is not None and after.get(key) is not None else None
            for key in ("cpu_seconds", "major_faults", "read_bytes", "write_bytes")}


class ResourceSampler:
    def __init__(self, interval: float = 1.0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Sampling interval must be positive")
        self.interval = interval
        self.pid = None
        self.peak_rss = self.peak_gpu = None
        self.rss_samples = self.gpu_samples = 0
        self.gpu_query_failures = 0
        self.smi = shutil.which("nvidia-smi")
        self._stop = threading.Event()
        self._thread = None
        self.started = None
        self.pid_namespace_supported = True

    def _sample(self):
        rss = process_counters(self.pid)["rss_bytes"]
        if rss is not None:
            self.peak_rss = max(self.peak_rss or 0, rss)
            self.rss_samples += 1
        if self.smi and self.pid_namespace_supported:
            try:
                result = subprocess.run(
                    [self.smi, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2, check=True,
                )
                rows = [row for row in csv.reader(io.StringIO(result.stdout)) if len(row) == 2 and row[0].strip() == str(self.pid)]
                if rows:
                    used = sum(int(float(row[1].strip()) * 1024 ** 2) for row in rows)
                    self.peak_gpu = max(self.peak_gpu or 0, used)
                    self.gpu_samples += 1
            except (OSError, ValueError, subprocess.SubprocessError):
                self.gpu_query_failures += 1

    def start(self, pid: int):
        if self._thread is not None:
            raise ValueError("A sampler can only monitor one process")
        self.pid, self.started = pid, time.monotonic()
        self.pid_namespace_supported = os.name == "nt" or proc_directory(pid) is not None

        def run():
            self._sample()
            while not self._stop.wait(self.interval):
                self._sample()

        self._thread = threading.Thread(target=run, name="mistralfit-resources", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self) -> dict:
        return {"scope": "One server PID; excludes client and other processes. RSS includes resident mapped pages.",
                "requested_interval_seconds": self.interval,
                "observed_peak_rss_bytes": self.peak_rss, "rss_sample_count": self.rss_samples,
                "observed_peak_gpu_bytes": self.peak_gpu, "gpu_sample_count": self.gpu_samples,
                "gpu_query_failures": self.gpu_query_failures,
                "pid_namespace_supported": self.pid_namespace_supported,
                "limits": ["Observed peaks are sampling lower bounds, not guaranteed maxima.",
                           "GPU sampling uses nvidia-smi compute-apps; unsupported counters remain null.",
                           "GPU queries can extend the interval and add overhead; no PCIe-transfer timing is measured.",
                           "A procfs PID-namespace mismatch disables server counters rather than sampling an unrelated PID.",
                           "CPU/I/O/fault counters require Linux procfs; RSS also supports Windows. Missing values stay null."]}
