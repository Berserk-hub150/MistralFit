"""llama-server subprocess lifecycle and streaming timing client."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))

from .planner import Profile


class RuntimeError_(RuntimeError):
    pass


def resolve_server(value: str | None) -> str:
    candidate = value or os.environ.get("MISTRALFIT_SERVER") or shutil.which("llama-server")
    if not candidate:
        raise RuntimeError_("llama-server not found. Install an official llama.cpp build and pass --server PATH.")
    resolved = shutil.which(candidate) or str(Path(candidate).expanduser().resolve())
    if not Path(resolved).is_file():
        raise RuntimeError_(f"Server executable does not exist: {resolved}")
    return resolved


def runtime_info(server: str) -> dict:
    try:
        help_run = subprocess.run([server, "--help"], capture_output=True, text=True, timeout=20, errors="replace")
        version_run = subprocess.run([server, "--version"], capture_output=True, text=True, timeout=20, errors="replace")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError_(f"Cannot execute llama-server: {exc}") from exc
    help_text = help_run.stdout + help_run.stderr
    if help_run.returncode != 0 or "--model" not in help_text:
        raise RuntimeError_("Executable did not provide llama-server's expected --help contract.")
    return {"version": (version_run.stdout + version_run.stderr).strip()[:2000], "help": help_text,
            "capabilities": {flag: flag in help_text for flag in
                             ("--fit", "--n-cpu-moe", "--cpu-moe", "--moe-cache-mib", "--perf", "--jinja", "--no-repack")}}


def unsupported(profile: Profile, help_text: str) -> list[str]:
    needed = ["--fit"]
    if profile.cpu_moe_layers:
        needed.append("--n-cpu-moe")
    if profile.expert_cache_mib:
        needed.append("--moe-cache-mib")
    return [flag for flag in needed if flag not in help_text]


def command(server: str, model: str, profile: Profile, *, context: int, threads: int,
            port: int, gpu_index: int = 0, help_text: str = "", api_key: str | None = None,
            repack: bool = True) -> list[str]:
    args = [server, "--model", model, "--ctx-size", str(context), "--threads", str(threads),
            "--parallel", "1", "--host", "127.0.0.1", "--port", str(port),
            "--batch-size", "512", "--ubatch-size", "128", "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--flash-attn", "auto", "--split-mode", "none", "--main-gpu", str(gpu_index)]
    if profile.gpu_layers is None:
        args.extend(["--fit", "on", "--n-gpu-layers", "auto"])
    else:
        args.extend(["--fit", "off", "--n-gpu-layers", str(profile.gpu_layers)])
    if profile.gpu_layers == 0:
        args.extend(["--device", "none"])
    if profile.cpu_moe_layers:
        args.extend(["--n-cpu-moe", str(profile.cpu_moe_layers)])
    if profile.expert_cache_mib:
        args.extend(["--moe-cache-mib", str(profile.expert_cache_mib)])
    if not repack:
        if "--no-repack" not in help_text:
            raise RuntimeError_("Runtime lacks --no-repack requested by this experiment")
        args.append("--no-repack")
    for flag in ("--perf", "--jinja"):
        if flag in help_text:
            args.append(flag)
    if "--cache-ram" in help_text:
        args.extend(["--cache-ram", "0"])
    if "--ctx-checkpoints" in help_text:
        args.extend(["--ctx-checkpoints", "0"])
    if api_key:
        args.extend(["--api-key", api_key])
    return args


def clean_environment() -> dict:
    # Host LLAMA_ARG_* presets could silently change the experiment (including model/host/ctx).
    return {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}


def json_request(base: str, route: str, payload=None, *, timeout: float = 30, api_key: str | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base + route, data=data, headers=headers)
    with LOCAL_HTTP.open(req, timeout=timeout) as response:
        return json.load(response)


class Server:
    def __init__(self, args_builder, *, startup_timeout: float = 300):
        self.args_builder = args_builder
        self.startup_timeout = startup_timeout
        self.process = None
        self.log = None
        # A random URL-safe token can begin with '-'; some command-line
        # parsers would then interpret the credential as another option.
        self.api_key = "mistralfit-" + secrets.token_urlsafe(24)
        self.base = ""
        self.argv = []
        self.last_health_error = "No health response received"

    def __enter__(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        self.argv = self.args_builder(port, self.api_key)
        self.log = tempfile.TemporaryFile(mode="w+b")
        try:
            self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT,
                                            env=clean_environment(), stdin=subprocess.DEVNULL)
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError_(f"llama-server exited during loading (code {self.process.returncode}). {self.log_tail()}")
                try:
                    health = json_request(self.base, "/health", timeout=1, api_key=self.api_key)
                    if health.get("status") == "ok":
                        # Authenticated probe prevents adopting an unrelated server on a raced port.
                        json_request(self.base, "/props", timeout=2, api_key=self.api_key)
                        return self
                    self.last_health_error = "Backend health status was " + repr(health.get("status"))
                except (OSError, ValueError, urllib.error.URLError) as exc:
                    self.last_health_error = type(exc).__name__ + ": " + str(exc)
                time.sleep(0.1)
            raise RuntimeError_(f"Model loading timed out after {self.startup_timeout:g}s; last health check: {self.last_health_error}\n{self.log_tail()}")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def log_tail(self):
        if self.log is None:
            return ""
        self.log.flush()
        self.log.seek(0, 2)
        self.log.seek(max(0, self.log.tell() - 4000))
        text = self.log.read().decode("utf-8", errors="replace").replace(self.api_key, "<redacted>")
        return re.sub(r"\x1b\[[0-9;]*m", "", text)

    def __exit__(self, *_):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=8)
        if self.log:
            self.log.close()


def stream_completion(base: str, prompt: str, *, tokens: int, seed: int = 42,
                      timeout: float = 120, api_key: str | None = None) -> dict:
    body = {"prompt": prompt, "n_predict": tokens, "stream": True, "temperature": 0,
            "seed": seed, "cache_prompt": False, "ignore_eos": True, "return_tokens": True}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base + "/completion", data=json.dumps(body).encode(), headers=headers)
    start, first, final = time.monotonic(), None, None
    output, token_ids = [], []
    # urlopen's timeout also bounds stalled reads; the monotonic deadline bounds active streams.
    with LOCAL_HTTP.open(req, timeout=timeout) as response:
        event_lines = []
        for line in response:
            if time.monotonic() - start > timeout:
                raise RuntimeError_("Completion exceeded the wall-clock deadline")
            decoded = line.decode("utf-8").rstrip("\r\n")
            if decoded.startswith("data:"):
                event_lines.append(decoded[5:].lstrip())
            elif decoded == "" and event_lines:
                raw = "\n".join(event_lines)
                event_lines = []
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except ValueError as exc:
                    raise RuntimeError_("Invalid JSON in llama-server's event stream") from exc
                if "error" in event:
                    raise RuntimeError_("Backend completion error: " + str(event["error"]))
                piece, ids = event.get("content", ""), event.get("tokens", [])
                if (piece or ids) and first is None:
                    first = time.monotonic() - start
                output.append(piece)
                token_ids.extend(ids)
                if event.get("stop"):
                    final = event
        if event_lines:
            raise RuntimeError_("Unterminated SSE event")
    if final is None:
        raise RuntimeError_("Stream ended before the backend reported completion")
    if final.get("truncated"):
        raise RuntimeError_("Prompt or completion was truncated; use a larger fixed context")
    timing = final.get("timings", {})
    count, duration = timing.get("predicted_n"), timing.get("predicted_ms")
    if not isinstance(count, int) or count != tokens:
        raise RuntimeError_(f"Backend generated {count!r} tokens, expected the fixed budget {tokens}")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError_("Backend did not expose valid decode timings")
    if first is None:
        raise RuntimeError_("No generated token was observed in the stream")
    prompt_n, cached_n = timing.get("prompt_n", 0), timing.get("cache_n", 0)
    if cached_n:
        raise RuntimeError_("Backend reused prompt tokens despite cache_prompt=false; trial excluded")
    text = "".join(output)
    return {"ttft_seconds": first, "wall_seconds": time.monotonic() - start,
            "decode_tokens_per_second": count / (duration / 1000),
            "generated_tokens": count, "prompt_tokens": prompt_n,
            "prompt_ms": timing.get("prompt_ms"), "decode_ms": duration,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest() if token_ids else None,
            "backend_timings": timing}
