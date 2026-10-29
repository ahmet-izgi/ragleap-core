"""Request metrics in the Prometheus text format, with no new dependency.

Counts live in memory of this process only (several workers each report their own). `/metrics` is off unless
METRICS_ENABLED is set, and needs an admin key when it is on. Labels are the route template (for example
/users/{user_id}), never the raw path, and the number of label combinations is capped, so ids and scanners cannot
grow memory. Also makes the X-Request-ID that every response carries."""
import os
import re
import threading
import time
import uuid

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
MAX_SERIES = 500
_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
_ID_OK = re.compile(r"[A-Za-z0-9._-]{8,64}")

_lock = threading.Lock()
_counts = {}        # (method, route, status) -> requests
_hist = {}          # (method, route) -> {"b": [per-bucket counts], "sum": seconds, "n": requests}
_state = {"inflight": 0, "dropped": 0}


def enabled() -> bool:
    return os.environ.get("METRICS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def reset() -> None:
    with _lock:
        _counts.clear()
        _hist.clear()
        _state.update(inflight=0, dropped=0)


def start() -> float:
    return time.perf_counter()


def since(t0: float) -> float:
    return time.perf_counter() - t0


def request_id(incoming=None) -> str:
    """Use the caller's id only when it is short and made of safe characters; otherwise make a new one."""
    if isinstance(incoming, str) and _ID_OK.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


def inflight(delta: int) -> None:
    with _lock:
        _state["inflight"] = max(0, _state["inflight"] + delta)


def record(method: str, route: str, status: int, seconds: float) -> None:
    method = method.upper() if isinstance(method, str) and method.upper() in _METHODS else "OTHER"
    route = (route or "unmatched")[:120]
    with _lock:
        key = (method, route, str(int(status)))
        if key not in _counts and len(_counts) >= MAX_SERIES:
            _state["dropped"] += 1
            return
        _counts[key] = _counts.get(key, 0) + 1
        h = _hist.get((method, route))
        if h is None:
            if len(_hist) >= MAX_SERIES:
                return
            h = _hist[(method, route)] = {"b": [0] * len(BUCKETS), "sum": 0.0, "n": 0}
        for i, upper in enumerate(BUCKETS):
            if seconds <= upper:
                h["b"][i] += 1
        h["sum"] += seconds
        h["n"] += 1


def _esc(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render(version: str = "") -> str:
    out = ["# HELP ragleap_build_info The running version (always 1).",
           "# TYPE ragleap_build_info gauge",
           'ragleap_build_info{version="%s"} 1' % _esc(version),
           "# HELP ragleap_http_requests_in_flight Requests being handled right now.",
           "# TYPE ragleap_http_requests_in_flight gauge"]
    with _lock:
        out.append("ragleap_http_requests_in_flight %d" % _state["inflight"])
        out += ["# HELP ragleap_metrics_dropped_total Requests not counted because the label limit was reached.",
                "# TYPE ragleap_metrics_dropped_total counter",
                "ragleap_metrics_dropped_total %d" % _state["dropped"],
                "# HELP ragleap_http_requests_total Requests by method, route template and status.",
                "# TYPE ragleap_http_requests_total counter"]
        for (m, r, s), n in sorted(_counts.items()):
            out.append('ragleap_http_requests_total{method="%s",route="%s",status="%s"} %d' % (m, _esc(r), s, n))
        out += ["# HELP ragleap_http_request_duration_seconds Time until the response starts (streams: until the first byte).",
                "# TYPE ragleap_http_request_duration_seconds histogram"]
        for (m, r), h in sorted(_hist.items()):
            lab = 'method="%s",route="%s"' % (m, _esc(r))
            for upper, c in zip(BUCKETS, h["b"]):
                out.append('ragleap_http_request_duration_seconds_bucket{%s,le="%s"} %d' % (lab, upper, c))
            out.append('ragleap_http_request_duration_seconds_bucket{%s,le="+Inf"} %d' % (lab, h["n"]))
            out.append("ragleap_http_request_duration_seconds_sum{%s} %.6f" % (lab, h["sum"]))
            out.append("ragleap_http_request_duration_seconds_count{%s} %d" % (lab, h["n"]))
    return "\n".join(out) + "\n"
