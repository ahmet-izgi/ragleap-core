import json

from ragleap_physical import ApprovalError, Capability, PhysicalTools

import pytest

from helpers import make

OUTCOME = {"entity": "joint_1", "quantity": "pos", "lo": 0.45, "hi": 0.55, "unit": "rad", "by_sim_time_s": 5.0}


def args(**changes):
    a = {"target": "joint_1", "command": "set_position_target", "value": 0.5, "unit": "rad",
         "expected_outcome": dict(OUTCOME), "reason": "move a little"}
    a.update(changes)
    return a


def tools(level=Capability.CONTROL):
    b, adapter, clock = make(level=level)
    return PhysicalTools(b), b, adapter


def test_exactly_four_tools_at_recommend_and_three_at_observe():
    t, _, _ = tools(Capability.RECOMMEND)
    assert [x["name"] for x in t.list_tools()] == ["physical_list_sources", "physical_get_observation",
                                                    "physical_get_envelope", "physical_propose_action"]
    t, _, _ = tools(Capability.OBSERVE)
    assert "physical_propose_action" not in [x["name"] for x in t.list_tools()]


def test_no_tool_can_apply_approve_stop_or_change_limits():
    t, _, adapter = tools()
    for name in ("physical_apply", "physical_approve", "physical_stop", "physical_set_envelope", "apply",
                 "approve", "clear_stop", "_sources", "__init__", "", None, 5):
        r = t.call(name, {})
        assert r["ok"] is False and r["error"]["code"] == "unknown_tool"
    assert adapter.applied == [] and not adapter.stopped


def test_observe_level_cannot_call_propose():
    t, _, _ = tools(Capability.OBSERVE)
    assert t.call("physical_propose_action", args())["error"]["code"] == "unknown_tool"


def test_propose_returns_proposal_and_moves_nothing():
    t, b, adapter = tools()
    r = t.call("physical_propose_action", args())
    assert r["ok"] and r["trust"] == "untrusted" and r["data"]["status"] == "validated"
    assert r["data"]["digest"] == b.get_proposal(r["data"]["proposal"]["proposal_id"]).digest()
    assert adapter.applied == [] and adapter.pos == 0.0


def test_a_person_still_has_to_approve_before_apply():
    t, b, adapter = tools()
    pid = t.call("physical_propose_action", args())["data"]["proposal"]["proposal_id"]
    p = b.get_proposal(pid)
    with pytest.raises(ApprovalError):
        b.apply(p)
    b.approve(p, "person:alice")
    assert b.apply(p).state == "verified"


def test_requested_by_cannot_be_chosen_by_the_caller():
    t, _, _ = tools()
    assert t.call("physical_propose_action", args(requested_by="person:alice"))["error"]["code"] == "bad_arguments"
    p = t.call("physical_propose_action", args())["data"]["proposal"]
    assert p["requested_by"] == "employee:agent"


def test_strict_arguments():
    t, _, _ = tools()
    bad = [args(value="0.5"), args(value=True), args(target=5), args(reason="x" * 501), args(reason="x" * 300 + "y" * 201),
           args(extra=1), args(expected_outcome=dict(OUTCOME, extra=1)), args(expected_outcome="none"),
           {k: v for k, v in args().items() if k != "unit"}, ["not", "an", "object"]]
    for a in bad:
        assert t.call("physical_propose_action", a)["error"]["code"] == "bad_arguments"
    assert t.call("physical_get_observation", {"x": 1})["error"]["code"] == "bad_arguments"


def test_envelope_errors_come_back_as_data_not_exceptions():
    t, _, adapter = tools()
    r = t.call("physical_propose_action", args(value=5.0, expected_outcome=dict(OUTCOME, lo=4.0, hi=6.0)))
    assert r["ok"] is False and r["error"]["code"] == "range"
    r = t.call("physical_propose_action", args(value=float("nan")))
    assert r["ok"] is False and r["error"]["code"] == "invalid_value"
    assert adapter.applied == []


def test_envelope_view_tool_cannot_change_limits():
    t, b, _ = tools()
    t.call("physical_get_envelope", {})["data"]["targets"]["joint_1"]["pos_max"] = 99.0
    assert t.call("physical_get_envelope", {})["data"]["targets"]["joint_1"]["pos_max"] == 1.0


def test_results_are_json_and_marked_untrusted():
    t, _, _ = tools()
    for name in ("physical_list_sources", "physical_get_observation", "physical_get_envelope"):
        r = t.call(name, None)
        json.dumps(r, allow_nan=False)
        assert r["trust"] == "untrusted" and r["ok"]


def test_internal_failures_do_not_leak_details():
    t, b, _ = tools()
    b.observe = lambda: (_ for _ in ()).throw(RuntimeError("secret path /etc/shadow"))
    r = t.call("physical_get_observation", {})
    assert r["error"] == {"code": "internal_error", "message": "the tool failed; see the operator log"}
