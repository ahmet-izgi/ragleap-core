"""Linked chats: a person's Telegram, WhatsApp or Discord chat is linked by the owner, and a YES or NO reply from it is
recorded under that person's name (operator or higher only). Real database for the links; approvals are mocked."""
import random
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core import autonomy, document_store, users

BOOT = {"x-api-key": "boot-key"}


def _sql(sql, args=()):
    conn = document_store._connect()
    try:
        conn.cursor().execute(sql, args)
        conn.commit()
    finally:
        conn.close()


def _chat():
    return str(random.randint(10 ** 9, 10 ** 10))


def _person(role="operator"):
    return users.create_user("pytest-" + uuid.uuid4().hex[:8], role)


def _owner_cfg():
    return patch.object(autonomy, "get_autonomy_settings", return_value={
        "mode": "semi", "channels": [], "actions": [], "approval_channel": "telegram", "approval_target": "424242"})


@pytest.fixture(autouse=True)
def clean():
    yield
    _sql("DELETE FROM users WHERE name LIKE %s", ("pytest-%",))     # links go with the person (cascade)
    users._clear_cache()


@pytest.fixture
def client(monkeypatch):
    from core import api
    monkeypatch.setattr(api, "RAGLEAP_API_KEY", "boot-key")
    return TestClient(api.app)


def test_link_list_resolve_and_unlink_round_trip():
    p, chat = _person(), _chat()
    link = users.link_channel(p["id"], "telegram", chat)
    assert link["channel"] == "telegram" and link["sender"] == chat
    assert [c["id"] for c in users.list_channels(p["id"])] == [link["id"]]
    got = users.resolve_channel("telegram", chat)
    assert got == {"user_id": p["id"], "name": p["name"], "role": "operator"}
    users.unlink_channel(p["id"], link["id"])
    assert users.resolve_channel("telegram", chat) is None and users.list_channels(p["id"]) == []


def test_the_same_chat_number_on_another_channel_is_a_different_chat():
    p, chat = _person(), _chat()
    users.link_channel(p["id"], "telegram", chat)
    assert users.resolve_channel("discord", chat) is None and users.resolve_channel("whatsapp", chat) is None


def test_whatsapp_numbers_are_matched_in_every_usual_format():
    p = _person()
    users.link_channel(p["id"], "whatsapp", "+91 98765-43210")
    for sender in ("+919876543210", "919876543210", "whatsapp:+919876543210"):
        assert users.resolve_channel("whatsapp", sender)["name"] == p["name"], sender
    assert users.resolve_channel("whatsapp", "+919876543211") is None


def test_one_chat_belongs_to_one_person():
    a, b, chat = _person(), _person(), _chat()
    users.link_channel(a["id"], "telegram", chat)
    with pytest.raises(users.UserError) as e:
        users.link_channel(b["id"], "telegram", chat)
    assert e.value.status == 409
    assert users.resolve_channel("telegram", chat)["name"] == a["name"]


def test_bad_input_and_unknown_or_disabled_people_are_refused():
    p = _person()
    for channel, sender, status in (("sms", "1", 400), ("telegram", "", 400), ("telegram", "   ", 400),
                                    ("telegram", "a\nb", 400), ("telegram", "x" * 121, 400), ("telegram", True, 400),
                                    ("whatsapp", "+", 400), ("telegram", None, 400)):
        with pytest.raises(users.UserError) as e:
            users.link_channel(p["id"], channel, sender)
        assert e.value.status == status, (channel, sender)
    with pytest.raises(users.UserError) as e:
        users.link_channel(str(uuid.uuid4()), "telegram", _chat())
    assert e.value.status == 404
    users.update_user(p["id"], status="disabled")
    with pytest.raises(users.UserError) as e:
        users.link_channel(p["id"], "telegram", _chat())
    assert e.value.status == 409


def test_a_disabled_person_is_no_longer_found_by_their_chat():
    p, chat = _person(), _chat()
    users.link_channel(p["id"], "telegram", chat)
    users.update_user(p["id"], status="disabled")
    assert users.resolve_channel("telegram", chat) is None


def test_removing_a_person_removes_their_links():
    p, chat = _person(), _chat()
    users.link_channel(p["id"], "telegram", chat)
    _sql("DELETE FROM users WHERE id = %s", (p["id"],))
    assert users.resolve_channel("telegram", chat) is None


def test_the_lookup_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(users, "_connect", boom)
    assert users.resolve_channel("telegram", "123") is None


def test_a_linked_operator_can_approve_and_is_recorded_by_name():
    p, chat = _person("operator"), _chat()
    users.link_channel(p["id"], "telegram", chat)
    with _owner_cfg(), patch.object(autonomy, "process_approval_response", return_value="done") as real:
        assert autonomy.process_approval_from("telegram", int(chat), "YES ABCD1234") == "done"
    real.assert_called_once_with("YES ABCD1234", decided_by={
        "name": p["name"], "role": "operator", "kind": "channel", "user_id": p["id"]})


def test_roles_below_operator_and_unlinked_chats_cannot_approve():
    v, chat = _person("viewer"), _chat()
    users.link_channel(v["id"], "telegram", chat)
    with _owner_cfg(), patch.object(autonomy, "process_approval_response") as real:
        assert autonomy.process_approval_from("telegram", int(chat), "YES ABCD1234") is None
        assert autonomy.process_approval_from("telegram", 987654321, "YES ABCD1234") is None
    real.assert_not_called()


def test_a_disabled_person_cannot_approve_and_a_lookup_error_means_nobody():
    p, chat = _person("admin"), _chat()
    users.link_channel(p["id"], "telegram", chat)
    users.update_user(p["id"], status="disabled")
    with _owner_cfg(), patch.object(autonomy, "process_approval_response") as real:
        assert autonomy.process_approval_from("telegram", int(chat), "NO ABCD1234") is None
    real.assert_not_called()
    with _owner_cfg(), patch.object(users, "resolve_channel", side_effect=Exception("db down")), \
            patch.object(autonomy, "process_approval_response") as real:
        assert autonomy.process_approval_from("telegram", int(chat), "NO ABCD1234") is None
    real.assert_not_called()


def test_the_configured_owner_chat_still_works_as_before():
    with _owner_cfg(), patch.object(autonomy, "process_approval_response", return_value="ok") as real:
        assert autonomy.process_approval_from("telegram", 424242, "YES ABCD1234") == "ok"
    real.assert_called_once_with("YES ABCD1234", decided_by={
        "name": "the owner's telegram chat", "role": "owner", "kind": "channel", "user_id": None})


def test_only_the_owner_can_manage_links_through_the_api(client):
    p = _person("operator")
    op_key = users.create_key(p["id"], "t")["key"]
    chat = _chat()
    assert client.post("/users/%s/channels" % p["id"], headers={"x-api-key": op_key},
                       json={"channel": "telegram", "sender": chat}).status_code == 403
    assert client.get("/users/%s/channels" % p["id"], headers={"x-api-key": op_key}).status_code == 403
    made = client.post("/users/%s/channels" % p["id"], headers=BOOT, json={"channel": "telegram", "sender": int(chat)})
    assert made.status_code == 200 and made.json()["sender"] == chat
    listed = client.get("/users/%s/channels" % p["id"], headers=BOOT).json()["channels"]
    assert [c["id"] for c in listed] == [made.json()["id"]]
    assert client.post("/users/%s/channels" % p["id"], headers=BOOT,
                       json={"channel": "telegram", "sender": chat}).status_code == 409
    assert client.delete("/users/%s/channels/%s" % (p["id"], made.json()["id"]), headers={"x-api-key": op_key}).status_code == 403
    assert client.delete("/users/%s/channels/%s" % (p["id"], made.json()["id"]), headers=BOOT).status_code == 200
    assert client.delete("/users/%s/channels/%s" % (p["id"], made.json()["id"]), headers=BOOT).status_code == 404
