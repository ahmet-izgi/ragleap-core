"""Energy gate for transcribe() (docs/design/vad-gate.md). Standard library only.

Status: pre-alpha. The thresholds are starting points from existing evidence, NOT measured results: the floor is the older
voice channel's RMS of 150 (about -46.8 dBFS, 8 kHz Twilio audio); relative_db and min_speech_frames are the author's choices.
The gate only measures level, so loud constant noise passes it. It never cuts audio inside a clip and never fails silently:
a decision is always described in the result notes.
"""
import array
import io
import math
import sys
import wave
from dataclasses import dataclass

from .audio import frames, peak_dbfs, rms_dbfs
from .errors import AudioDecodeError, UnsupportedFeatureError

MODES = ("off", "energy")
DEFAULTS = {"floor_dbfs": -46.8, "relative_db": 30.0, "min_speech_frames": 5, "frame_ms": 20}


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    speech_frames: int
    total_frames: int
    peak_dbfs: float
    rms_dbfs: float
    threshold_dbfs: float


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise UnsupportedFeatureError("gate option %s must be a number." % label)
    return float(value)


def _int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UnsupportedFeatureError("gate option %s must be a positive integer." % label)
    return value


def validate(mode, options):
    """Return the merged options, or raise. Called before any model work."""
    if mode not in MODES:
        raise ValueError("gate must be one of: %s" % ", ".join(MODES))
    options = dict(options or {})
    unknown = sorted(set(options) - set(DEFAULTS))
    if unknown:
        raise UnsupportedFeatureError("unknown gate option(s): %s." % ", ".join(unknown),
                                      hint="supported: %s." % ", ".join(sorted(DEFAULTS)))
    merged = dict(DEFAULTS)
    merged.update(options)
    _number(merged["floor_dbfs"], "floor_dbfs")
    if _number(merged["relative_db"], "relative_db") < 0:
        raise UnsupportedFeatureError("gate option relative_db must not be negative.")
    _int(merged["min_speech_frames"], "min_speech_frames")
    _int(merged["frame_ms"], "frame_ms")
    return merged


def evaluate(wav_bytes, options=None):
    """Judge a mono 16-bit PCM WAV clip. Returns a GateDecision."""
    opt = validate("energy", options)
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            channels, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            if (channels, width) != (1, 2):
                raise UnsupportedFeatureError(
                    "the energy gate needs mono 16-bit PCM WAV, got %d channel(s), %d-bit." % (channels, width * 8),
                    hint='convert first, or pass gate="off".')
            raw = w.readframes(n)
    except (wave.Error, EOFError) as exc:
        raise AudioDecodeError("not a readable PCM WAV file (%s)." % exc)
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()
    size = max(1, rate * opt["frame_ms"] // 1000)
    levels = [rms_dbfs(f) for f in frames(samples, size, "drop")]
    clip_rms, clip_peak = rms_dbfs(samples), peak_dbfs(samples)
    floor = float(opt["floor_dbfs"])
    if not levels:
        return GateDecision(False, "the clip is shorter than one frame of %d ms" % opt["frame_ms"], 0, 0,
                            clip_peak, clip_rms, floor)
    threshold = max(floor, max(levels) - float(opt["relative_db"]))
    speech = sum(1 for lv in levels if lv >= threshold)
    passed = speech >= opt["min_speech_frames"]
    if passed:
        reason = "enough speech-like frames"
    elif max(levels) < floor:
        reason = "the whole clip is below the floor"
    else:
        reason = "fewer speech-like frames than min_speech_frames (%d)" % opt["min_speech_frames"]
    return GateDecision(passed, reason, speech, len(levels), clip_peak, clip_rms, threshold)


def describe(d):
    """The note text for a decision."""
    head = "energy gate: %s (%s)" % ("passed" if d.passed else "no speech-like audio, the engine was not run", d.reason)
    return ("%s; %d of %d frames at or above %.1f dBFS; clip RMS %.1f dBFS, peak %.1f dBFS. "
            "The thresholds are unmeasured starting points, and a level gate cannot tell loud noise from speech."
            % (head, d.speech_frames, d.total_frames, d.threshold_dbfs, d.rms_dbfs, d.peak_dbfs))
