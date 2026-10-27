"""Energy gate benchmark: false drops on real speech (A) and invented text on silence and noise (B). Colab T4 runtime, no token.

Installs ragleap-voice from the repository main (the gate is not in a PyPI release when this was run) and faster-whisper 1.2.1.
A: first 30 FLEURS test clips (CC-BY-4.0) of en_us, es_419, ru_ru, as recorded (no level fix), against four floor rules.
B: Whisper large-v3 (language=en, device cuda) on 10 s of digital silence (5 clips) and white noise at -70, -60, -50, -40, -20 dBFS
   (30 clips each, seed 7). Wilson 95% intervals. Public data and synthetic signals only.
Needs RAM for 90 speech clips plus the model; on a CPU-only runtime the CUDA load fails ("driver version is insufficient").
"""
import subprocess, sys, os, glob, ctypes, site, io, wave, json, csv, math, array, collections
import numpy as np
def sh(*a):
    r = subprocess.run(list(a), capture_output=True, text=True); return r.returncode, r.stdout + r.stderr
assert "HF_TOKEN" not in os.environ, "a token is present: stop"
print(sh(sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "--no-cache-dir",
         "git+https://github.com/ragleap/ragleap-core.git#subdirectory=packages/ragleap-voice")[0])
print(sh(sys.executable, "-m", "pip", "install", "-q", "faster-whisper==1.2.1", "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
         "datasets==3.6.0", "soundfile")[0])
for r in site.getsitepackages():
    for p in sorted(glob.glob(r + "/nvidia/cublas/lib/libcublas*.so.12"), reverse=True) + sorted(glob.glob(r + "/nvidia/cudnn/lib/libcudnn*.so.9")):
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError as e: print("not loaded:", p, e)
print(sh(sys.executable, "-m", "ragleap_voice", "models", "pull", "whisper-large-v3", "--allow-unverified")[1][-200:])

from datasets import load_dataset
from ragleap_voice import audio as A
from ragleap_voice.gate import evaluate
from ragleap_voice.whisper_engine import WhisperEngine
from ragleap_voice.engine import Request
from ragleap_voice.cache import model_dir
from ragleap_voice.manifest import load_manifest

def wav_bytes(x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
    return b.getvalue()

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
def pct(k, n):
    lo, hi = wilson(k, n); return "%3d/%-3d %5.1f%% [%4.1f-%5.1f]" % (k, n, 100 * k / max(n, 1), 100 * lo, 100 * hi)

RULES = {"default(-46.8)": {}, "floor-60": {"floor_dbfs": -60}, "floor-70": {"floor_dbfs": -70}, "floor-80": {"floor_dbfs": -80}}
os.makedirs("/content/results", exist_ok=True)

# ---- A: false drops on real speech, as recorded ----
speech = {}
for code, cfg in (("en", "en_us"), ("es", "es_419"), ("ru", "ru_ru")):
    ds = load_dataset("google/fleurs", cfg, split="test", streaming=True, trust_remote_code=True)
    speech[code] = [np.asarray(ex["audio"]["array"], dtype="float32") for i, ex in zip(range(30), ds)]
rowsA = []
print("\n== A: share of real speech clips that PASS the gate (as recorded, no level fix) ==")
for code, clips in speech.items():
    levels = []
    for x in clips:
        s = array.array("h", (np.clip(x, -1, 1) * 32767).astype("<i2").tolist()); levels.append(A.rms_dbfs(s))
    print("%s: clip RMS dBFS  min %.1f  median %.1f  max %.1f  | at or below -55.9: %d of %d" % (
        code, min(levels), float(np.median(levels)), max(levels), sum(1 for v in levels if v <= -55.9), len(levels)))
    for name, opt in RULES.items():
        k = sum(1 for x in clips if evaluate(wav_bytes(x), opt).passed)
        print("   %-15s passed %s" % (name, pct(k, len(clips)))); rowsA.append([code, name, k, len(clips)])

# ---- B: invented text on silence and noise ----
rng = np.random.default_rng(7)
N16 = 16000 * 10
def white(db): return (rng.standard_normal(N16) * 10 ** (db / 20)).astype("float32")
cases = [("zeros", [np.zeros(N16, dtype="float32")] * 5)] + [("white%+d" % db, [white(db) for _ in range(30)]) for db in (-70, -60, -50, -40, -20)]
entry = {e.name: e for e in load_manifest()[0]}["whisper-large-v3"]
eng = WhisperEngine(); eng.load({"whisper-large-v3": model_dir(entry.name, entry.revision)})
rowsB, texts = [], collections.defaultdict(collections.Counter)
print("\n== B: Whisper on silence/noise: how many clips produced text, and how many the gate would stop ==")
for name, clips in cases:
    produced = 0; stopped = {r: 0 for r in RULES}; stopped_text = {r: 0 for r in RULES}
    for x in clips:
        b = wav_bytes(x)
        t = eng.run(Request(kind="stt", language="en", audio=b, options={"device": "cuda"})).text.strip()
        if t: produced += 1; texts[name][t[:60]] += 1
        for r, opt in RULES.items():
            if not evaluate(b, opt).passed:
                stopped[r] += 1
                if t: stopped_text[r] += 1
    print("%-9s text produced: %s" % (name, pct(produced, len(clips))))
    for r in RULES:
        print("   %-15s gate drops %s | of the clips with text, stopped %d of %d" % (r, pct(stopped[r], len(clips)), stopped_text[r], produced))
        rowsB.append([name, r, len(clips), produced, stopped[r], stopped_text[r]])
print("\nTexts produced (top 3 per case):")
for name, c in texts.items(): print(" ", name, c.most_common(3))
with open("/content/results/gate_A.csv", "w", newline="") as f: csv.writer(f).writerows([["lang", "rule", "passed", "n"]] + rowsA)
with open("/content/results/gate_B.csv", "w", newline="") as f: csv.writer(f).writerows([["case", "rule", "n", "text_produced", "gate_drops", "dropped_with_text"]] + rowsB)
eng.close()
