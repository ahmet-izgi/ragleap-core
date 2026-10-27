"""Filler-phrase and empty-output notes for transcribe() (docs/design/api.md section 13, item 6). Standard library only.

A note only: the text is never changed or removed. The list is short, English only and NOT a detector of invented text in
general. Sources: benchmarks/voice/README.md (Whisper large-v3 wrote these for silence and white noise), and the filler list
of the earlier noise-engine script ("thank you", "thanks for watching", "you", ".").
"""
import unicodedata

PHRASES = {
    "thank you": "gate benchmark 2026-10-27: Whisper large-v3 on silence and white noise",
    "thanks for watching": "gate benchmark 2026-10-27: Whisper large-v3 on white noise at -20 dBFS",
    "you": "filler list of the noise-engine script (not seen in the gate benchmark)",
}


def normalize(text):
    """Lowercase, drop punctuation, collapse whitespace."""
    kept = "".join(c for c in text.lower() if not unicodedata.category(c).startswith("P"))
    return " ".join(kept.split())


def check(text, existing_notes=()):
    """Return a list of note strings for an engine's output text."""
    n = normalize(text)
    if not n:
        if any("no text" in x for x in existing_notes):
            return []
        return ["the engine returned no text (empty, or only punctuation)."]
    if n in PHRASES:
        return ["the output equals a phrase this kind of model often writes for non-speech (%r); check the audio. "
                "This is a note only, the text was not changed. The phrase list is short, English only and cannot "
                "detect other kinds of invented text." % n]
    return []
