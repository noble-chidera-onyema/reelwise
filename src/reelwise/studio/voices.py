"""Narration: one audio file per line, with word-level timing.

Engines:
  edge         Microsoft Edge neural voices through edge-tts. Free, and the
               same voices Luvvoice uses, so Ren/Bex/Ozzy sound as they do now.
               Gives exact word boundaries.
  elevenlabs   A voice cloned from the sample you upload. Paid. Gives
               character-level timing, grouped here into words.
  files        Lines you recorded or generated elsewhere, uploaded one file per
               line. Word timing is spread across the speech by character count.
  mock         A synthetic tone at a fixed speaking rate. Tests and dry runs only.

Every engine returns the same thing: a 48 kHz mono WAV with leading and
trailing silence trimmed, its duration, and a list of words with start/end
times relative to the file.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 48000


@dataclass
class Spoken:
    path: str
    duration_s: float
    words: list = field(default_factory=list)   # [{"w": str, "start": s, "end": s}]


# --------------------------------------------------------------------------- audio helpers

def _ffmpeg_to_wav(src: str | Path, dst: str | Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR),
                    "-c:a", "pcm_s16le", str(dst)], check=True)


def read_wav(path: str | Path) -> np.ndarray:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, np.int16).astype(np.float32) / 32768.0


def write_wav(path: str | Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1",
                    "-i", "-", "-c:a", "pcm_s16le", str(path)], input=pcm.tobytes(), check=True)


def speech_bounds(x: np.ndarray, thresh_db: float = -45.0) -> tuple[int, int]:
    """First and last sample above the threshold (20 ms blocks)."""
    blk = int(SR * 0.02)
    n = len(x) // blk
    if n == 0:
        return 0, len(x)
    rms = np.sqrt((x[: n * blk].reshape(n, blk) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    on = np.flatnonzero(db > thresh_db)
    if on.size == 0:
        return 0, len(x)
    return max(0, on[0] * blk - blk), min(len(x), (on[-1] + 2) * blk)


def _trim(path: Path, words: list[dict]) -> Spoken:
    x = read_wav(path)
    a, b = speech_bounds(x)
    off = a / SR
    write_wav(path, x[a:b])
    dur = (b - a) / SR
    w2 = [{"w": w["w"], "start": round(max(0.0, w["start"] - off), 3),
           "end": round(min(dur, max(0.0, w["end"] - off)), 3)} for w in words]
    return Spoken(str(path), round(dur, 3), w2)


def spread_words(text: str, duration_s: float) -> list[dict]:
    """Word timing estimated from character counts. Used when an engine gives none."""
    toks = text.split()
    if not toks:
        return []
    weights = np.array([len(t) + 1 for t in toks], float)
    edges = np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * duration_s
    return [{"w": t, "start": round(edges[i], 3), "end": round(edges[i + 1], 3)} for i, t in enumerate(toks)]


# --------------------------------------------------------------------------- engines

class Engine:
    name = "base"

    def config(self) -> dict:
        return {}

    def key(self, text: str) -> str:
        raw = json.dumps({"engine": self.name, "cfg": self.config(), "text": text}, sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def speak(self, text: str, out: Path) -> Spoken:
        raise NotImplementedError


class EdgeEngine(Engine):
    name = "edge"

    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%"):
        self.voice, self.rate, self.pitch, self.volume = voice, rate, pitch, volume

    def config(self):
        return {"voice": self.voice, "rate": self.rate, "pitch": self.pitch, "volume": self.volume}

    def speak(self, text: str, out: Path) -> Spoken:
        import edge_tts

        mp3 = out.with_suffix(".mp3")
        words: list[dict] = []

        async def run():
            com = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch,
                                       volume=self.volume, boundary="WordBoundary")
            with open(mp3, "wb") as f:
                async for ch in com.stream():
                    if ch["type"] == "audio":
                        f.write(ch["data"])
                    elif ch["type"] == "WordBoundary":
                        st = ch["offset"] / 1e7
                        words.append({"w": ch["text"], "start": st, "end": st + ch["duration"] / 1e7})

        asyncio.run(run())
        _ffmpeg_to_wav(mp3, out)
        mp3.unlink(missing_ok=True)
        return _trim(out, words)


class ElevenLabsEngine(Engine):
    name = "elevenlabs"
    API = "https://api.elevenlabs.io/v1"

    def __init__(self, voice_id: str, model_id: str = "eleven_multilingual_v2", speed: float = 1.0,
                 stability: float = 0.5, similarity: float = 0.8, api_key: str | None = None, session=None):
        self.voice_id, self.model_id, self.speed = voice_id, model_id, speed
        self.stability, self.similarity = stability, similarity
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.session = session

    def config(self):
        return {"voice_id": self.voice_id, "model": self.model_id, "speed": self.speed,
                "stability": self.stability, "similarity": self.similarity}

    def _s(self):
        if self.session is None:
            import requests
            self.session = requests.Session()
        return self.session

    def speak(self, text: str, out: Path) -> Spoken:
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        r = self._s().post(
            f"{self.API}/text-to-speech/{self.voice_id}/with-timestamps",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": self.model_id,
                  "voice_settings": {"stability": self.stability, "similarity_boost": self.similarity,
                                     "speed": self.speed}},
            timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        mp3 = out.with_suffix(".mp3")
        mp3.write_bytes(base64.b64decode(data["audio_base64"]))
        _ffmpeg_to_wav(mp3, out)
        mp3.unlink(missing_ok=True)
        return _trim(out, chars_to_words(data.get("alignment") or {}))


def chars_to_words(al: dict) -> list[dict]:
    chars = al.get("characters") or []
    st = al.get("character_start_times_seconds") or []
    en = al.get("character_end_times_seconds") or []
    words, cur, s0, e0 = [], "", None, None
    for c, a, b in zip(chars, st, en):
        if c.isspace():
            if cur:
                words.append({"w": cur, "start": s0, "end": e0})
            cur, s0 = "", None
            continue
        if s0 is None:
            s0 = a
        cur += c
        e0 = b
    if cur:
        words.append({"w": cur, "start": s0, "end": e0})
    return words


def clone_elevenlabs_voice(name: str, sample_paths: list[str], api_key: str | None = None,
                           session=None) -> str:
    """Create an instant voice clone from your own samples. Returns the voice_id."""
    import requests
    key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    s = session or requests.Session()
    files = [("files", (Path(p).name, open(p, "rb"))) for p in sample_paths]
    try:
        r = s.post(f"{ElevenLabsEngine.API}/voices/add", headers={"xi-api-key": key},
                   data={"name": name, "remove_background_noise": "true",
                         "description": f"reelwise host voice: {name}"},
                   files=files, timeout=300)
    finally:
        for _, (_, fh) in files:
            fh.close()
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs clone HTTP {r.status_code}: {r.text[:300]}")
    return r.json()["voice_id"]


class FileEngine(Engine):
    """Pre-made line files: {line_id: path}. Lines without a file raise."""
    name = "files"

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.current_line = None

    def config(self):
        return {"file": self.files.get(self.current_line, "")}

    def speak(self, text: str, out: Path) -> Spoken:
        src = self.files.get(self.current_line)
        if not src:
            raise RuntimeError(f"no audio file uploaded for line {self.current_line}")
        _ffmpeg_to_wav(src, out)
        x = read_wav(out)
        a, b = speech_bounds(x)
        write_wav(out, x[a:b])
        dur = (b - a) / SR
        return Spoken(str(out), round(dur, 3), spread_words(text, dur))


class MockEngine(Engine):
    """Speech-shaped tone at a fixed rate. For tests and timing dry runs."""
    name = "mock"

    def __init__(self, words_per_second: float = 3.0, pitch_hz: float = 160.0):
        self.wps, self.f0 = words_per_second, pitch_hz

    def config(self):
        return {"wps": self.wps, "f0": self.f0}

    def speak(self, text: str, out: Path) -> Spoken:
        words = spread_words(text, len(text.split()) / self.wps)
        dur = words[-1]["end"] if words else 0.2
        t = np.arange(int(SR * dur)) / SR
        env = np.zeros_like(t)
        for w in words:
            a, b = int(w["start"] * SR), int(w["end"] * SR)
            span = max(b - a, 1)
            env[a:b] = np.sin(np.linspace(0, np.pi, span)) ** 0.5
        x = 0.25 * env * (np.sin(2 * np.pi * self.f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * self.f0 * t))
        write_wav(out, x.astype(np.float32))
        return Spoken(str(out), round(dur, 3), words)



# --------------------------------------------------------------------------- free local models

CACHE = Path(os.environ.get("REELWISE_CACHE", Path.home() / ".cache" / "reelwise"))
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}
# British and American male voices that suit the hosts. All 54 Kokoro voices work; these are the shortlist.
KOKORO_SHORTLIST = ["bm_george", "bm_lewis", "bm_daniel", "bm_fable", "am_michael", "am_puck",
                    "am_eric", "am_liam", "am_onyx", "am_adam", "am_echo", "am_fenrir"]


def fetch(url: str, dest: Path, progress=None) -> Path:
    """Download once into the cache. Free public files only."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total)
    tmp.replace(dest)
    return dest


def align_by_pauses(x: np.ndarray, text: str, sr: int = SR) -> list[dict]:
    """Word timing without a timing-aware engine.

    The audio's own pauses are found (quiet runs of at least 40 ms) and each
    pause is matched to the word gap whose expected position, by character
    count, is closest. Words between matched pauses share the speech span in
    proportion to their length. Accurate to roughly a word, which is what
    three-word captions need.
    """
    toks = text.split()
    if not toks:
        return []
    blk = int(0.01 * sr)
    nb = len(x) // blk
    if nb == 0:
        return spread_words(text, len(x) / sr)
    rms = np.sqrt((x[: nb * blk].reshape(nb, blk) ** 2).mean(1))
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    quiet = db < max(np.percentile(db, 90) - 28, -55)
    pauses = []
    k = 0
    while k < nb:
        if quiet[k]:
            j = k
            while j < nb and quiet[j]:
                j += 1
            if j - k >= 4 and k > 0 and j < nb:
                pauses.append(((k + j) / 2) * 0.01)
            k = j
        else:
            k += 1
    dur = len(x) / sr
    w = np.array([len(t) + 1 for t in toks], float)
    expected = np.cumsum(w)[:-1] / w.sum() * dur          # gap after word i
    # Speakers pause at punctuation, so a gap after a comma or full stop is preferred
    # over a slightly closer gap mid-phrase.
    punct = np.array([t[-1:] in ",.;:?!" for t in toks[:-1]], bool)
    anchors = {}
    for p in pauses:
        cost = np.abs(expected - p) - 0.35 * punct
        i = int(np.argmin(cost))
        if abs(expected[i] - p) < 0.7 and i not in anchors:
            anchors[i] = p
    edges = [0.0]
    last_i, last_t = -1, 0.0
    for i in sorted(anchors):
        span = w[last_i + 1:i + 1]
        cum = np.cumsum(span) / span.sum() * (anchors[i] - last_t) + last_t
        edges += list(cum)
        last_i, last_t = i, anchors[i]
    span = w[last_i + 1:]
    edges += list(np.cumsum(span) / span.sum() * (dur - last_t) + last_t)
    return [{"w": t, "start": round(edges[i], 3), "end": round(edges[i + 1], 3)} for i, t in enumerate(toks)]


class KokoroEngine(Engine):
    """Kokoro-82M (Apache 2.0). Free, runs on an ordinary CPU, natural delivery.
    Model files (about 350 MB) download once on first use."""
    name = "kokoro"
    _model = None

    def __init__(self, voice: str = "bm_george", speed: float = 1.0, lang: str = "en-gb"):
        self.voice, self.speed, self.lang = voice, float(speed), lang

    def config(self):
        return {"voice": self.voice, "speed": self.speed, "lang": self.lang}

    @classmethod
    def model(cls, progress=None):
        if cls._model is None:
            from kokoro_onnx import Kokoro
            paths = {n: fetch(u, CACHE / "kokoro" / n, progress) for n, u in KOKORO_FILES.items()}
            cls._model = Kokoro(str(paths["kokoro-v1.0.onnx"]), str(paths["voices-v1.0.bin"]))
        return cls._model

    def speak(self, text: str, out: Path) -> Spoken:
        samples, sr = self.model().create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        tmp = out.with_suffix(".raw.wav")
        import soundfile as sf
        sf.write(str(tmp), samples, sr)
        _ffmpeg_to_wav(tmp, out)
        tmp.unlink(missing_ok=True)
        x = read_wav(out)
        a, b = speech_bounds(x)
        x = x[a:b]
        write_wav(out, x)
        return Spoken(str(out), round(len(x) / SR, 3), align_by_pauses(x, text))


class ChatterboxEngine(Engine):
    """Chatterbox (MIT). Free voice cloning from a 10-30 s sample of your own host.
    Heavier than Kokoro: needs PyTorch and runs about real time on a laptop CPU."""
    name = "chatterbox"
    _model = None

    def __init__(self, sample: str, exaggeration: float = 0.5, cfg_weight: float = 0.5):
        self.sample, self.exaggeration, self.cfg_weight = sample, float(exaggeration), float(cfg_weight)

    def config(self):
        return {"sample": Path(self.sample).name, "exaggeration": self.exaggeration, "cfg": self.cfg_weight}

    @classmethod
    def model(cls):
        if cls._model is None:
            from chatterbox.tts import ChatterboxTTS
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            cls._model = ChatterboxTTS.from_pretrained(device=dev)
        return cls._model

    def speak(self, text: str, out: Path) -> Spoken:
        import torchaudio
        m = self.model()
        wav = m.generate(text, audio_prompt_path=self.sample, exaggeration=self.exaggeration,
                         cfg_weight=self.cfg_weight)
        tmp = out.with_suffix(".raw.wav")
        torchaudio.save(str(tmp), wav, m.sr)
        _ffmpeg_to_wav(tmp, out)
        tmp.unlink(missing_ok=True)
        x = read_wav(out)
        a, b = speech_bounds(x)
        x = x[a:b]
        write_wav(out, x)
        return Spoken(str(out), round(len(x) / SR, 3), align_by_pauses(x, text))


def engine_from_config(cfg: dict) -> Engine:
    kind = cfg.get("engine", "kokoro")
    if kind == "kokoro":
        return KokoroEngine(cfg.get("voice", "bm_george"), float(cfg.get("speed", 1.0)), cfg.get("lang", "en-gb"))
    if kind == "chatterbox":
        return ChatterboxEngine(cfg["sample"], float(cfg.get("exaggeration", 0.5)), float(cfg.get("cfg_weight", 0.5)))
    if kind == "edge":
        return EdgeEngine(cfg["voice"], cfg.get("rate", "+0%"), cfg.get("pitch", "+0Hz"), cfg.get("volume", "+0%"))
    if kind == "elevenlabs":
        return ElevenLabsEngine(cfg["voice_id"], cfg.get("model_id", "eleven_multilingual_v2"),
                                float(cfg.get("speed", 1.0)), float(cfg.get("stability", 0.5)),
                                float(cfg.get("similarity", 0.8)))
    if kind == "files":
        return FileEngine(cfg.get("files", {}))
    if kind == "mock":
        return MockEngine(float(cfg.get("words_per_second", 3.0)), float(cfg.get("pitch_hz", 160)))
    raise ValueError(f"unknown voice engine {kind!r}")


# --------------------------------------------------------------------------- timing the plan

def time_plan(plan: dict, engines: dict[str, Engine], audio_dir: str | Path,
              zoom_s: float = 1.0, min_story_s: float = 1.5) -> dict:
    """Voice every line, then set each segment's length from its narration.

    Layout inside a segment: lead-in, line, gap, line, ..., tail. The first
    line of the whole episode starts at the profile's first_line.start_s.
    The intro is stretched if needed so the push into the TV (zoom_s) runs
    under its last words, never in silence.
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    prof = plan["profile"]
    g_lo, g_hi = prof.get("gap_between_lines_s", [0.14, 0.26])
    gap = round((g_lo + g_hi) / 2, 3)
    first_start = prof.get("first_line", {}).get("start_s", 0.3)
    lead, tail = 0.15, 0.25

    first = True
    for seg in plan["segments"]:
        t = first_start if first else lead
        for ln in seg["lines"]:
            eng = engines.get(ln["speaker"])
            if eng is None:
                raise RuntimeError(f"no voice set for {ln['speaker']}")
            if isinstance(eng, FileEngine):
                eng.current_line = ln["id"]
            out = audio_dir / f"{ln['id']}_{eng.key(ln['say'])}.wav"
            meta = out.with_suffix(".json")
            if out.exists() and meta.exists():
                sp = Spoken(**json.loads(meta.read_text()))
            else:
                sp = eng.speak(ln["say"], out)
                # Captions show the real spelling; map spoken words back when counts match.
                cap = ln["text"].split()
                if len(cap) == len(sp.words):
                    for w, c in zip(sp.words, cap):
                        w["w"] = c
                else:
                    sp.words = align_by_pauses(read_wav(sp.path), ln["text"]) if sp.duration_s else []
                meta.write_text(json.dumps(sp.__dict__))
            ln["audio"] = sp.path
            ln["start_s"] = round(t, 3)
            ln["duration_s"] = sp.duration_s
            ln["words"] = sp.words
            t += sp.duration_s + gap
            first = False
        end = (t - gap + tail) if seg["lines"] else 0.0
        if seg["kind"] == "intro":
            end = max(end, 2 * zoom_s + 0.8)   # pull out of the TV, hold on the hosts, push back in
        if seg["kind"] == "outro":
            end = max(end, zoom_s + 0.6)
        if seg["kind"] in ("story", "cold_open"):
            end = max(end, min_story_s if seg["kind"] == "story" else 1.2)
        seg["duration_s"] = round(end, 3)
    return plan


# --------------------------------------------------------------------------- voice match

def median_f0(x: np.ndarray, sr: int = SR) -> float | None:
    """Median pitch of voiced 40 ms frames by autocorrelation (70-400 Hz)."""
    frame = int(0.04 * sr)
    hop = frame // 2
    lo, hi = int(sr / 400), int(sr / 70)
    f0s = []
    for i in range(0, len(x) - frame, hop):
        f = x[i:i + frame]
        if np.sqrt((f ** 2).mean()) < 0.02:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        k = int(np.argmax(seg)) + lo
        if ac[k] / ac[0] > 0.45:
            f0s.append(sr / k)
    return float(np.median(f0s)) if len(f0s) >= 5 else None


def voice_match(sample_path: str, rendered_path: str) -> dict:
    """Compare pitch and brightness of a rendered line with the uploaded sample."""
    a, b = read_wav(sample_path), read_wav(rendered_path)
    fa, fb = median_f0(a), median_f0(b)

    def centroid(x):
        spec = np.abs(np.fft.rfft(x[: SR * 20] if len(x) > SR * 20 else x))
        freqs = np.fft.rfftfreq(min(len(x), SR * 20), 1 / SR)
        return float((spec * freqs).sum() / max(spec.sum(), 1e-9))

    semis = 12 * np.log2(fb / fa) if fa and fb else None
    ca, cb = centroid(a), centroid(b)
    return {"sample_f0_hz": round(fa, 1) if fa else None, "render_f0_hz": round(fb, 1) if fb else None,
            "pitch_diff_semitones": round(float(semis), 2) if semis is not None else None,
            "brightness_ratio": round(cb / ca, 2) if ca else None,
            "match": bool(semis is not None and abs(semis) <= 1.5 and ca and 0.75 <= cb / ca <= 1.33)}


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg"))


# --------------------------------------------------------------------------- samples from an old episode

def split_utterances(path: str, min_gap_s: float = 0.18, min_len_s: float = 0.4) -> list[tuple[float, float]]:
    """Speech runs in a video or audio file, split at pauses of at least min_gap_s."""
    x = read_wav(path)
    blk = int(0.02 * SR)
    nb = len(x) // blk
    if nb == 0:
        return []
    db = 20 * np.log10(np.maximum(np.sqrt((x[: nb * blk].reshape(nb, blk) ** 2).mean(1)), 1e-7))
    act = db > np.percentile(db, 30) + 8
    gap = max(1, int(round(min_gap_s / 0.02)))
    out, k = [], 0
    while k < nb:
        if not act[k]:
            k += 1
            continue
        j, quiet = k, 0
        while j < nb and quiet < gap:
            quiet = 0 if act[j] else quiet + 1
            j += 1
        a, b = k * 0.02, (j - quiet) * 0.02
        if b - a >= min_len_s:
            out.append((round(a, 2), round(b, 2)))
        k = j
    return out


def build_samples(path: str, labelled: list[tuple[float, float, str]], out_dir: str | Path,
                  max_s: float = 30.0) -> dict[str, str]:
    """Join each host's labelled lines (with short pauses) into one sample WAV per host."""
    x = read_wav(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by: dict[str, list] = {}
    for a, b, who in labelled:
        if not who:
            continue
        seg = x[max(0, int((a - 0.05) * SR)): int((b + 0.08) * SR)]
        parts = by.setdefault(who, [])
        if sum(len(p) for p in parts) / SR < max_s:
            parts += [seg, np.zeros(int(0.3 * SR), np.float32)]
    files = {}
    for who, parts in by.items():
        p = out_dir / f"{who.lower()}_sample.wav"
        write_wav(p, np.concatenate(parts))
        files[who] = str(p)
    return files
