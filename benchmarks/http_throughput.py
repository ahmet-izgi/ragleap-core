#!/usr/bin/env python3
"""Small HTTP throughput and latency check for a running RagLeap app. Standard library only.

It sends GET requests to the paths you name from several threads for a fixed time, then prints requests per second,
errors and latency percentiles per path. It is a measurement tool, not a claim: run it on the hardware you care about
and quote the numbers together with the machine, the model settings and WEB_CONCURRENCY. See THROUGHPUT.md.

Example (the key is read from an environment variable, never from the command line):
    read -rs RAGLEAP_PERSONAL_KEY; export RAGLEAP_PERSONAL_KEY
    python3 benchmarks/http_throughput.py --url http://127.0.0.1:8000 --paths /health,/me --key-env RAGLEAP_PERSONAL_KEY
"""
import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse

MAX_CONCURRENCY = 64
MAX_SECONDS = 300


def percentile(sorted_ms, p):
    """Nearest-rank percentile of an already sorted list; 0.0 for an empty list."""
    if not sorted_ms:
        return 0.0
    rank = max(1, -(-len(sorted_ms) * p // 100))        # ceil(n * p / 100)
    return sorted_ms[int(rank) - 1]


def _summary(samples, elapsed):
    ms = sorted(s[1] for s in samples)
    errors = sum(1 for s in samples if not s[2])
    return {"requests": len(samples), "errors": errors,
            "req_per_s": round(len(samples) / elapsed, 1) if elapsed > 0 else 0.0,
            "p50_ms": round(percentile(ms, 50), 2), "p95_ms": round(percentile(ms, 95), 2),
            "p99_ms": round(percentile(ms, 99), 2), "max_ms": round(ms[-1], 2) if ms else 0.0}


def run(base_url, paths, concurrency=8, seconds=10.0, key=None, timeout=10.0):
    """Hit the paths round-robin from `concurrency` threads for `seconds`. Returns a dict of results."""
    if not paths:
        raise ValueError("give at least one path")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError("concurrency must be 1 to %d" % MAX_CONCURRENCY)
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError("seconds must be above 0 and at most %d" % MAX_SECONDS)
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("url must start with http:// or https://")
    headers = {"x-api-key": key} if key else {}
    results = [[] for _ in range(concurrency)]
    started = time.monotonic()
    deadline = started + seconds

    def worker(slot):
        factory = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        conn, i = None, slot
        while time.monotonic() < deadline:
            path = paths[i % len(paths)]
            i += 1
            t0 = time.perf_counter()
            ok = False
            try:
                if conn is None:
                    conn = factory(parts.hostname, parts.port, timeout=timeout)
                conn.request("GET", path, headers=headers)
                resp = conn.getresponse()
                resp.read()
                ok = resp.status < 400
            except Exception:
                if conn is not None:
                    conn.close()
                conn = None
            results[slot].append((path, (time.perf_counter() - t0) * 1000.0, ok))
        if conn is not None:
            conn.close()

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - started
    every = [s for r in results for s in r]
    return {"url": base_url, "concurrency": concurrency, "seconds": round(elapsed, 2), "cpu_count": os.cpu_count(),
            "web_concurrency_env": os.environ.get("WEB_CONCURRENCY", ""),
            "paths": {p: _summary([s for s in every if s[0] == p], elapsed) for p in paths},
            "total": _summary(every, elapsed)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="HTTP throughput check for a running RagLeap app.")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--paths", default="/health", help="comma separated, for example /health,/me")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--key-env", default="", help="name of the environment variable that holds the API key")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)
    key = None
    if args.key_env:
        key = os.environ.get(args.key_env, "")
        if not key:
            print("The environment variable %s is empty or not set." % args.key_env, file=sys.stderr)
            return 2
    try:
        out = run(args.url, [p.strip() for p in args.paths.split(",") if p.strip()], args.concurrency, args.seconds, key)
    except ValueError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print("url %s | %d threads | %.1f s | %s CPUs | WEB_CONCURRENCY=%s" % (
        out["url"], out["concurrency"], out["seconds"], out["cpu_count"], out["web_concurrency_env"] or "(not set here)"))
    print("%-28s %9s %7s %9s %8s %8s %8s %8s" % ("path", "requests", "errors", "req/s", "p50 ms", "p95 ms", "p99 ms", "max ms"))
    for name, s in list(out["paths"].items()) + [("TOTAL", out["total"])]:
        print("%-28s %9d %7d %9.1f %8.2f %8.2f %8.2f %8.2f" % (
            name[:28], s["requests"], s["errors"], s["req_per_s"], s["p50_ms"], s["p95_ms"], s["p99_ms"], s["max_ms"]))
    return 1 if out["total"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
