"""Voiceprints: is this line really in this host's voice?

A voice clone can drift on a line and come out sounding like someone else. Each
locked host voice keeps a reference read (profiles/<profile>/voices/<host>/reference.wav);
every newly voiced line is compared with it and voiced again if it does not match.

Model: WeSpeaker ResNet34 trained on VoxCeleb (English speakers), run with sherpa-onnx,
26 MB, downloaded once. Calibrated on the Worth Telling cast: lines in the right voice
score 0.73 to 0.94 against their own reference; a line that drifted scored 0.55, and
other people's voices reading the same line scored at most 0.71. MIN_MATCH sits
between the two.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import subprocess

from .voices import CACHE, fetch

MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
             "wespeaker_en_voxceleb_resnet34_LM.onnx")
MIN_MATCH = 0.70
_extractor = None
_ref_cache: dict[str, np.ndarray] = {}


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
        return True
    except ImportError:
        return False


def _ex():
    global _extractor
    if _extractor is None:
        import sherpa_onnx as so
        model = fetch(MODEL_URL, CACHE / "voiceprint" / "wespeaker_en_voxceleb_resnet34_LM.onnx")
        _extractor = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(model=str(model), num_threads=2))
    return _extractor


def embed(path: str | Path) -> np.ndarray:
    """Unit-length voiceprint of a speech file (any rate the app's read_wav accepts)."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True, stdin=subprocess.DEVNULL)
    y = np.frombuffer(out.stdout, np.int16).astype(np.float32) / 32768.0
    ex = _ex()
    s = ex.create_stream()
    s.accept_waveform(16000, y)
    s.input_finished()
    e = np.array(ex.compute(s), np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def match(path: str | Path, reference: str | Path) -> float:
    """Cosine similarity between a line and the host's reference read, -1 to 1."""
    key = str(reference)
    if key not in _ref_cache:
        _ref_cache[key] = embed(reference)
    return float(embed(path) @ _ref_cache[key])
