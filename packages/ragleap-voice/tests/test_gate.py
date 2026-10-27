import array
import io
import math
import pathlib
import random
import struct
import sys
import wave

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ragleap_voice import GateDecision, UnsupportedFeatureError  # noqa: E402
from ragleap_voice.gate import describe, evaluate, validate  # noqa: E402
from test_transcribe import FakeStt, go, make_env, wav_bytes  # noqa: E402,F401


def wav_from(values, rate=16000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", v) * channels for v in values))
    return buf.getvalue()


def tone_wav(n=3200, amp=6000, rate=16000, channels=1):
    return wav_from([int(amp * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)], rate, channels)


def test_silence_tone_and_quiet_tone():
    assert evaluate(wav_bytes()).passed is False
    d = evaluate(tone_wav())
    assert isinstance(d, GateDecision) and d.passed and d.speech_frames == d.total_frames == 10
    quiet = evaluate(tone_wav(amp=100))
    assert not quiet.passed and "below the floor" in quiet.reason
    assert evaluate(tone_wav(amp=100), {"floor_dbfs": -70}).passed


def test_short_clip_and_speech_with_pauses():
    d = evaluate(wav_bytes(frames=100))
    assert not d.passed and d.total_frames == 0 and "shorter" in d.reason
    vals = [0] * 3200 + [int(6000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(3200)] + [0] * 3200
    d = evaluate(wav_from(vals))
    assert d.passed and d.speech_frames == 10 and d.total_frames == 30


def test_loud_constant_noise_passes_documented_limit():
    rnd = random.Random(1)
    assert evaluate(wav_from([int(rnd.gauss(0, 3000)) for _ in range(3200)])).passed


def test_input_and_option_checks():
    with pytest.raises(UnsupportedFeatureError):
        evaluate(tone_wav(channels=2))
    for bad in ({"nope": 1}, {"floor_dbfs": "x"}, {"relative_db": -1}, {"min_speech_frames": 0}, {"frame_ms": True}):
        with pytest.raises(UnsupportedFeatureError):
            validate("energy", bad)
    with pytest.raises(ValueError):
        validate("auto", None)
    assert "unmeasured" in describe(evaluate(tone_wav()))


def test_off_leaves_behaviour_unchanged(tmp_path):
    env = make_env(tmp_path)
    eng = FakeStt()
    t = go(env, eng, language="en", engine="fake")
    assert t.text == "hello" and eng.loaded is not None and not any("energy gate" in n for n in t.notes)


def test_energy_gate_drops_silence_without_running_the_engine(tmp_path):
    env = make_env(tmp_path)
    eng = FakeStt()
    t = go(env, eng, language="en", engine="fake", gate="energy")
    assert t.text == "" and t.segments == () and t.model_status == "verified" and t.engine == "fake"
    assert eng.loaded is None and eng.request is None
    assert any("engine was not run" in n for n in t.notes)


def test_energy_gate_passes_speech_like_audio(tmp_path):
    env = make_env(tmp_path)
    eng = FakeStt()
    t = go(env, eng, audio=tone_wav(), language="en", engine="fake", gate="energy")
    assert t.text == "hello" and eng.request is not None
    assert any("energy gate: passed" in n for n in t.notes) and "engine note" in t.notes


def test_gate_options_and_errors_through_transcribe(tmp_path):
    env = make_env(tmp_path)
    quiet = tone_wav(amp=100)
    assert go(env, FakeStt(), audio=quiet, language="en", engine="fake", gate="energy").text == ""
    t = go(env, FakeStt(), audio=quiet, language="en", engine="fake", gate="energy", gate_options={"floor_dbfs": -70})
    assert t.text == "hello"
    with pytest.raises(UnsupportedFeatureError):
        go(env, FakeStt(), audio=tone_wav(channels=2), language="en", engine="fake", gate="energy")
    with pytest.raises(ValueError):
        go(env, FakeStt(), language="en", engine="fake", gate="maybe")
