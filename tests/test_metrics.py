"""Request metrics and X-Request-ID. No database needed: the shared test key is the owner."""
import re

import pytest
from fastapi.testclient import TestClient

from core import metrics, permissions

BOOT = {"x-api-key": "boot-key"}


@pytest.fixture(autouse=True)
def fresh():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def client(monkeypatch):
    from core import api
    monkeypatch.setattr(api, "RAGLEAP_API_KEY", "boot-key")
    return TestClient(api.app)


def _on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "true")


def test_the_text_has_counters_and_a_cumulative_histogram():
    metrics.record("GET", "/health", 200, 0.003)
    metrics.record("GET", "/health", 200, 0.3)
    text = metrics.render("9.9.9")
    assert 'ragleap_build_info{version="9.9.9"} 1' in text
    assert 'ragleap_http_requests_total{method="GET",route="/health",status="200"} 2' in text
    assert 'ragleap_http_request_duration_seconds_bucket{method="GET",route="/health",le="0.005"} 1' in text
    assert 'ragleap_http_request_duration_seconds_bucket{method="GET",route="/health",le="0.25"} 1' in text
    assert 'ragleap_http_request_duration_seconds_bucket{method="GET",route="/health",le="0.5"} 2' in text
    assert 'ragleap_http_request_duration_seconds_bucket{method="GET",route="/health",le="+Inf"} 2' in text
    assert 'ragleap_http_request_duration_seconds_count{method="GET",route="/health"} 2' in text
    assert text.endswith("\n")


def test_the_number_of_label_combinations_is_capped():
    for i in range(metrics.MAX_SERIES + 50):
        metrics.record("GET", "/r%d" % i, 200, 0.01)
    text = metrics.render()
    assert text.count("ragleap_http_requests_total{") == metrics.MAX_SERIES
    assert "ragleap_metrics_dropped_total 50" in text
    metrics.record("GET", "/r0", 200, 0.01)                       # an existing series still counts
    assert 'route="/r0",status="200"} 2' in metrics.render()


def test_odd_methods_and_quotes_cannot_break_the_format():
    metrics.record("BREW", 'a"b\nc', 200, 0.01)
    text = metrics.render()
    assert 'method="OTHER"' in text and 'route="a\\"b\\nc"' in text
    assert all(not l.startswith("c") for l in text.split("\n"))


def test_request_ids_are_kept_only_when_safe():
    assert metrics.request_id("abc-123_DEF.456") == "abc-123_DEF.456"
    for bad in (None, "", "short", "has space inside", "x" * 65, "line\nbreak12", "semi;colon;1", 12345678):
        got = metrics.request_id(bad)
        assert re.fullmatch(r"[0-9a-f]{32}", got), bad


def test_metrics_is_off_by_default_even_for_the_owner(client):
    assert client.get("/metrics", headers=BOOT).status_code == 404


def test_metrics_needs_a_key_and_an_admin_role(client, monkeypatch):
    _on(monkeypatch)
    assert client.get("/metrics").status_code == 401
    assert permissions.required_role("GET", "/metrics") == "admin"
    r = client.get("/metrics", headers=BOOT)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "ragleap_build_info{version=" in r.text


def test_requests_are_counted_by_route_template_not_raw_path(client, monkeypatch):
    _on(monkeypatch)
    client.get("/health")
    client.get("/users/not-a-real-id/channels")                  # 401 before routing: the template is still found
    client.get("/no/such/page/abc123")
    text = client.get("/metrics", headers=BOOT).text
    assert 'route="/health",status="200"} 1' in text
    assert 'route="/users/{user_id}/channels",status="401"} 1' in text
    assert 'route="unmatched",status="401"} 1' in text or 'route="unmatched",status="404"} 1' in text
    assert "not-a-real-id" not in text and "abc123" not in text


def test_nothing_is_counted_while_metrics_are_off(client):
    client.get("/health")
    assert "ragleap_http_requests_total{" not in metrics.render()


def test_every_response_carries_a_request_id(client):
    a = client.get("/health")
    assert re.fullmatch(r"[0-9a-f]{32}", a.headers["x-request-id"])
    assert client.get("/health", headers={"x-request-id": "trace-abc-12345"}).headers["x-request-id"] == "trace-abc-12345"
    assert re.fullmatch(r"[0-9a-f]{32}", client.get("/health", headers={"x-request-id": "bad id!"}).headers["x-request-id"])
    assert client.get("/documents").headers["x-request-id"]       # also on a refused request
