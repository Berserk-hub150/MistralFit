"""Protocol fixture only. This is NOT inference and MUST NOT be used for benchmarks."""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import TCPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--ctx-size", type=int, default=4096)
parser.add_argument("--api-key", default="")
parser.add_argument("--fixture-mode", default="ok")
args, _ = parser.parse_known_args()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def auth(self):
        return not args.api_key or self.headers.get("Authorization") == "Bearer " + args.api_key

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/health" and not self.auth():
            return self.send_json({"error": "unauthorized"}, 401)
        if self.path == "/health":
            return self.send_json({"status": "ok"})
        if self.path == "/props":
            ctx = args.ctx_size // 2 if args.fixture_mode == "wrong-context" else args.ctx_size
            return self.send_json({"default_generation_settings": {"n_ctx": ctx}})
        if self.path == "/v1/models":
            return self.send_json({"data": [{"id": "fixture-only"}]})
        self.send_json({}, 404)

    def do_POST(self):
        if not self.auth():
            return self.send_json({"error": "unauthorized"}, 401)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self.send_json({"tokens": [1, 2, 3]})
        if self.path == "/v1/chat/completions":
            return self.send_json({"choices": [{"message": {"tool_calls": [{"function": {"name": "add", "arguments": '{"a":19,"b":23}'}}]}}]})
        if self.path != "/completion":
            return self.send_json({}, 404)
        if body.get("stream") is False:
            # Fixed protocol responses only; this fixture does not reason or run a model.
            prompt = body.get("prompt", "")
            answer = {"answer": 42} if "19 + 23" in prompt else {"name": "Paolo", "pears": 5} if "Paolo" in prompt else {"values": [-1, 2, 8, 8]}
            return self.send_json({"content": json.dumps(answer), "truncated": False})
        if args.fixture_mode == "slow-headers":
            try:
                for byte in b"HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\n":
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
        count = body["n_predict"]
        try:
            if args.fixture_mode == "slow-line":
                for byte in b': an unfinished event line keeps delivering bytes without a newline\n':
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.01)
            time.sleep(0.005)
            self.wfile.write(b'data: {"content":"fixture", "tokens":[123], "stop":false}\n\n')
            self.wfile.flush()
            time.sleep(0.005)
            if args.fixture_mode == "missing-final":
                return
            timings = {"predicted_n": count, "predicted_ms": 20.0, "predicted_per_second": (count - 1) / 0.02,
                       "prompt_n": 3, "prompt_ms": 2, "cache_n": 0}
            if args.fixture_mode == "wrong-count":
                timings["predicted_n"] = count - 1
            if args.fixture_mode == "cached":
                timings["cache_n"] = 2
            if args.fixture_mode == "missing-timing":
                timings = {}
            if args.fixture_mode == "missing-rate":
                timings.pop("predicted_per_second", None)
            data = {"content": "", "tokens": [], "stop": True, "timings": timings,
                    "truncated": args.fixture_mode == "truncated"}
            self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
            self.wfile.flush()
            if args.fixture_mode == "linger-after-final":
                time.sleep(0.6)
        except (BrokenPipeError, ConnectionResetError):
            pass


class FixtureHTTPServer(ThreadingHTTPServer):
    def server_bind(self):
        # HTTPServer normally reverse-resolves its hostname here. A local
        # protocol fixture must not wait for CI DNS or macOS mDNS services.
        TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


server = FixtureHTTPServer(("127.0.0.1", args.port), Handler)
print("Protocol fixture listening on loopback; no model inference", flush=True)
server.serve_forever()
