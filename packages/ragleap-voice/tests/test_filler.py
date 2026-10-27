import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragleap_voice.filler import PHRASES, check, normalize  # noqa: E402
from test_transcribe import FakeStt, go, make_env  # noqa: E402,F401


def test_normalize():
    assert normalize("  Thank   YOU. ") == "thank you"
    assert normalize("Thanks for watching!") == "thanks for watching"
    assert normalize("...") == ""


def test_check_rules():
    assert check("Thank you.") and check("Thanks for watching!") and check("You")
    assert check("Thank you very much for the report") == []
    assert check("hello world") == []
    assert len(check(".")) == 1 and len(check("")) == 1
    assert check("", ["the engine produced no text."]) == []
    assert all(PHRASES.values())


def test_through_transcribe_text_is_kept(tmp_path):
    env = make_env(tmp_path)
    t = go(env, FakeStt(text="Thank you."), language="en", engine="fake")
    assert t.text == "Thank you." and any("check the audio" in n for n in t.notes)
    t = go(env, FakeStt(text="Thank you for coming today"), language="en", engine="fake")
    assert not any("check the audio" in n for n in t.notes)
    t = go(env, FakeStt(text=""), language="en", engine="fake")
    assert t.text == "" and any("no text" in n for n in t.notes)
