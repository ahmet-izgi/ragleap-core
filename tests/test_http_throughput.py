"""benchmarks/http_throughput.py against a small local server, plus the WEB_CONCURRENCY behaviour THROUGHPUT.md relies on."""
import importlib.util
import json
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("http_throughput", ROOT / "benchmarks" / "http_throughput.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

SEEN = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        SEEN.append(self.headers.get("x-api-key"))
        if self.path == "/slow":
            time.sleep(0.03)
        status = 500 if self.path == "/boom" else 200
        if self.path == "/auth" and self.headers.get("x-api-key") != "bench-test-value":
            status = 401
        body = b"{}"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    del SEEN[:]
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def test_percentiles_use_the_nearest_rank():
    data = [float(n) for n in range(1, 101)]
    assert bench.percentile(data, 50) == 50.0
    assert bench.percentile(data, 95) == 95.0
    assert bench.percentile(data, 99) == 99.0
    assert bench.percentile([7.0], 99) == 7.0
    assert bench.percentile([], 50) == 0.0


def test_it_counts_requests_and_latency_per_path(server):
    out = bench.run(server, ["/ok", "/slow"], concurrency=3, seconds=0.6)
    assert out["total"]["requests"] > 10 and out["total"]["errors"] == 0
    assert out["paths"]["/ok"]["requests"] > 0 and out["paths"]["/slow"]["requests"] > 0
    assert out["paths"]["/slow"]["p50_ms"] >= 25                       # the handler sleeps 30 ms
    assert out["paths"]["/ok"]["p50_ms"] < out["paths"]["/slow"]["p50_ms"]
    assert out["total"]["req_per_s"] > 0 and out["concurrency"] == 3 and out["cpu_count"]


def test_error_responses_and_a_missing_key_are_counted_as_errors(server):
    out = bench.run(server, ["/boom", "/auth"], concurrency=2, seconds=0.4)
    assert out["paths"]["/boom"]["errors"] == out["paths"]["/boom"]["requests"] > 0
    assert out["paths"]["/auth"]["errors"] == out["paths"]["/auth"]["requests"] > 0


def test_the_key_is_sent_as_the_api_key_header(server):
    out = bench.run(server, ["/auth"], concurrency=2, seconds=0.4, key="bench-test-value")
    assert out["paths"]["/auth"]["errors"] == 0 and set(SEEN) == {"bench-test-value"}


def test_a_dead_server_gives_errors_not_a_crash():
    out = bench.run("http://127.0.0.1:9", ["/health"], concurrency=2, seconds=0.3, timeout=0.5)
    assert out["total"]["requests"] > 0 and out["total"]["errors"] == out["total"]["requests"]


def test_unreasonable_settings_are_refused():
    for kwargs in ({"concurrency": 0}, {"concurrency": bench.MAX_CONCURRENCY + 1}, {"seconds": 0}, {"seconds": bench.MAX_SECONDS + 1}):
        with pytest.raises(ValueError):
            bench.run("http://127.0.0.1:1", ["/x"], **kwargs)
    for url in ("ftp://example.com", "not a url", ""):
        with pytest.raises(ValueError):
            bench.run(url, ["/x"], seconds=0.1)
    with pytest.raises(ValueError):
        bench.run("http://127.0.0.1:1", [], seconds=0.1)


def test_the_command_line_reads_the_key_from_the_environment_only(server, monkeypatch, capsys):
    monkeypatch.delenv("BENCH_TEST_ENV", raising=False)
    assert bench.main(["--url", server, "--paths", "/auth", "--key-env", "BENCH_TEST_ENV", "--seconds", "0.2"]) == 2
    monkeypatch.setenv("BENCH_TEST_ENV", "bench-test-value")
    assert bench.main(["--url", server, "--paths", "/auth", "--key-env", "BENCH_TEST_ENV", "--seconds", "0.3", "--json"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[printed.index("{"):])["total"]["errors"] == 0
    assert "bench-test-value" not in printed                         # the key is never echoed


def test_web_concurrency_in_the_environment_sets_the_number_of_web_processes(monkeypatch):
    import uvicorn
    monkeypatch.setenv("WEB_CONCURRENCY", "3")
    assert uvicorn.Config("core.api:app").workers == 3
    monkeypatch.delenv("WEB_CONCURRENCY")
    assert uvicorn.Config("core.api:app").workers == 1


def test_the_dockerfile_does_not_hard_code_the_number_of_workers():
    assert "--workers" not in (ROOT / "Dockerfile").read_text()


def test_the_throughput_notes_exist_and_name_the_per_process_state():
    text = (ROOT / "benchmarks" / "THROUGHPUT.md").read_text()
    for needle in ("WEB_CONCURRENCY", "--scale worker", "/metrics", "Failed-login throttle", "REDIS_URL"):
        assert needle in text
