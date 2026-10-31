import json
import threading
import urllib.error
import urllib.request

import pytest

from ragleap_physical import Capability, ConfigError, PhysicalTools
from ragleap_physical.mcp_server import MAX_BODY, handle_message, make_server

from helpers import make
from test_tools import args


def rpc(method, params=None, id=1):
    m = {"jsonrpc": "2.0", "id": id, "method": method}
    if params is not None:
        m["params"] = params
    return m


def setup(level=Capability.CONTROL):
    b, adapter, _ = make(level=level)
    return PhysicalTools(b), b, adapter


# ---- message level
def test_initialize_and_version_choice():
    t, _, _ = setup()
    r = handle_message(t, rpc("initialize", {"protocolVersion": "2025-03-26"}))["result"]
    assert r["protocolVersion"] == "2025-03-26" and r["capabilities"] == {"tools": {"listChanged": False}}
    assert handle_message(t, rpc("initialize", {"protocolVersion": "1999-01-01"}))["result"]["protocolVersion"] == "2025-06-18"


def test_notifications_and_replies_get_no_answer():
    t, _, _ = setup()
    assert handle_message(t, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert handle_message(t, {"jsonrpc": "2.0", "id": 5, "result": {}}) is None


def test_bad_messages():
    t, _, _ = setup()
    for bad in (None, 5, "x", [], {"id": 1, "method": "ping"}, {"jsonrpc": "2.0", "id": True, "method": "ping"},
                {"jsonrpc": "2.0", "id": 1, "method": 7}):
        assert handle_message(t, bad)["error"]["code"] == -32600
    assert handle_message(t, rpc("resources/list"))["error"]["code"] == -32601


def test_tools_list_and_call_roundtrip():
    t, _, adapter = setup()
    names = [x["name"] for x in handle_message(t, rpc("tools/list"))["result"]["tools"]]
    assert names[-1] == "physical_propose_action"
    r = handle_message(t, rpc("tools/call", {"name": "physical_propose_action", "arguments": args()}))["result"]
    body = json.loads(r["content"][0]["text"])
    assert r["isError"] is False and body["trust"] == "untrusted" and body["data"]["status"] == "validated"
    assert adapter.applied == []


def test_tool_errors_are_marked_is_error():
    t, _, _ = setup()
    for params in ({"name": "physical_apply", "arguments": {}}, {"name": 5}, {}, {"name": "physical_get_observation", "arguments": 3}):
        assert handle_message(t, rpc("tools/call", params))["result"]["isError"] is True


# ---- HTTP level
class Running:
    def __init__(self, tools, **kw):
        self.server = make_server(tools, **kw)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


def post(url, body, headers=None, raw=None, method="POST"):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_happy_path_and_human_in_the_loop():
    t, b, adapter = setup()
    with Running(t) as s:
        st, body = post(s.url, rpc("initialize", {"protocolVersion": "2025-06-18"}))
        assert st == 200 and json.loads(body)["result"]["serverInfo"]["name"] == "ragleap-physical"
        assert post(s.url, {"jsonrpc": "2.0", "method": "notifications/initialized"})[0] == 202
        st, body = post(s.url, rpc("tools/call", {"name": "physical_propose_action", "arguments": args()}, id=2))
        pid = json.loads(json.loads(body)["result"]["content"][0]["text"])["data"]["proposal"]["proposal_id"]
        st, body = post(s.url, rpc("tools/call", {"name": "physical_apply", "arguments": {"proposal_id": pid}}, id=3))
        assert json.loads(body)["result"]["isError"] is True and adapter.applied == []
    p = b.get_proposal(pid)
    b.approve(p, "person:alice")
    assert b.apply(p).state == "verified"


def test_http_rejections():
    t, _, _ = setup()
    with Running(t) as s:
        base = s.url
        assert post(base.replace("/mcp", "/other"), rpc("ping"))[0] == 404
        for m in ("GET", "PUT", "DELETE"):
            assert post(base, None, method=m)[0] == 405
        assert post(base, None, raw=b"{not json")[0] == 400
        assert post(base, [rpc("ping")])[0] == 400
        assert post(base, None, raw=b"x" * (MAX_BODY + 1))[0] == 413
        assert post(base, rpc("ping"), headers={"Content-Type": "text/plain"})[0] == 415
        assert post(base, rpc("ping"), headers={"Origin": "https://evil.example"})[0] == 403
        assert post(base, rpc("ping"), headers={"Host": "evil.example"})[0] == 403
        assert post(base, rpc("ping"))[0] == 200


def test_bearer_token():
    t, _, _ = setup()
    tok = "t" * 20
    with Running(t, auth_token=tok) as s:
        assert post(s.url, rpc("ping"))[0] == 401
        assert post(s.url, rpc("ping"), headers={"Authorization": "Bearer wrong"})[0] == 401
        assert post(s.url, rpc("ping"), headers={"Authorization": f"Bearer {tok}"})[0] == 200


def test_unsafe_bind_and_weak_token_refused():
    t, _, _ = setup()
    with pytest.raises(ConfigError):
        make_server(t, host="0.0.0.0")
    with pytest.raises(ConfigError):
        make_server(t, auth_token="short")
    s = make_server(t, host="0.0.0.0", auth_token="x" * 20)
    s.server_close()


def test_internal_errors_do_not_leak():
    t, _, _ = setup()
    t.call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret /etc/shadow"))
    with Running(t) as s:
        st, body = post(s.url, rpc("tools/call", {"name": "physical_get_observation", "arguments": {}}))
        assert st == 500 and b"shadow" not in body
