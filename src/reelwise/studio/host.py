"""The host scene: Ren, Bex and Ozzy on the couch, in front of the TV.

Two ways to supply it:

  Rig (recommended). A background image or video of the set with the TV,
  plus each host's transparent cut-outs: mouth closed, mouth open and
  (optionally) eyes closed. Mouths open only while that host's own audio is
  above a speech threshold, so lips follow the voice frame by frame, and each
  host blinks on a staggered schedule. This automates the CapCut recipe and
  uses the locked character art, so the hosts never drift.

  Plate. A single image or video with the hosts already in it. Nothing is
  animated on the hosts, so keep narration over a plate short or off camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .voices import SR, read_wav


def load_rgba(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        a = np.full(img.shape[:2], 255, np.uint8)
        img = np.dstack([img, a])
    return img


class Plate:
    """Background still or video. frame(i) loops a video."""

    def __init__(self, path: str, size: tuple[int, int] | None = None):
        self.path = str(path)
        img = cv2.imread(self.path)
        self.frames: list[np.ndarray] = []
        if img is not None:
            self.frames = [img]
        else:
            cap = cv2.VideoCapture(self.path)
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                self.frames.append(f)
            cap.release()
        if not self.frames:
            raise ValueError(f"cannot read host plate {path}")
        if size:
            w, h = size
            self.frames = [_cover(f, w, h) for f in self.frames]
        self.h, self.w = self.frames[0].shape[:2]
        self.is_video = len(self.frames) > 1

    def frame(self, i: int) -> np.ndarray:
        if not self.is_video:
            return self.frames[0]
        n = len(self.frames)
        k = i % (2 * n - 2) if n > 1 else 0   # ping-pong loop: no jump at the loop point
        return self.frames[k if k < n else 2 * n - 2 - k]


def _cover(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (int(np.ceil(iw * s)), int(np.ceil(ih * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


@dataclass
class HostLayer:
    name: str
    closed: np.ndarray
    open: np.ndarray | None
    blink: np.ndarray | None
    x: float            # centre, fraction of canvas width
    y: float            # bottom edge, fraction of canvas height
    height: float       # cut-out height, fraction of canvas height
    mouth: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))   # per output frame
    blinks: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


class HostRig:
    def __init__(self, cfg: dict, canvas: tuple[int, int]):
        self.w, self.h = canvas
        self.plate = Plate(cfg["background"], canvas)
        self.layers: list[HostLayer] = []
        for name, c in cfg.get("hosts", {}).items():
            self.layers.append(HostLayer(
                name=name,
                closed=load_rgba(c["closed"]),
                open=load_rgba(c["open"]) if c.get("open") else None,
                blink=load_rgba(c["blink"]) if c.get("blink") else None,
                x=float(c.get("x", 0.5)), y=float(c.get("y", 0.9)), height=float(c.get("height", 0.3)),
            ))
        self._scaled: dict = {}

    def _sprite(self, layer: HostLayer, which: str) -> np.ndarray:
        key = (layer.name, which)
        if key not in self._scaled:
            src = getattr(layer, which) if getattr(layer, which) is not None else layer.closed
            th = max(2, int(layer.height * self.h))
            tw = max(2, int(src.shape[1] * th / src.shape[0]))
            self._scaled[key] = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA)
        return self._scaled[key]

    def drive(self, lines: list[dict], seg_start_s: float, n_frames: int, fps: int, seed: int = 7) -> None:
        """Set mouth and blink states for one segment from its narration lines."""
        rng = np.random.default_rng(seed)
        for li, layer in enumerate(self.layers):
            layer.mouth = np.zeros(n_frames, bool)
            layer.blinks = np.zeros(n_frames, bool)
            t = rng.uniform(0.6, 2.0) + li * 0.9
            while t < n_frames / fps:
                a = int(t * fps)
                layer.blinks[a:a + 3] = True
                t += rng.uniform(2.8, 4.6)
        by_name = {l.name.lower(): l for l in self.layers}
        for ln in lines:
            layer = by_name.get(ln["speaker"].lower())
            if layer is None or not ln.get("audio"):
                continue
            x = read_wav(ln["audio"])
            open_frames = mouth_track(x, fps)
            start = int(round((ln["start_s"]) * fps))
            for i, o in enumerate(open_frames):
                if 0 <= start + i < n_frames and o:
                    layer.mouth[start + i] = True

    def overlay(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Hosts for frame i as (straight BGR, alpha 0..1), canvas-sized."""
        pre = np.zeros((self.h, self.w, 3), np.float32)     # premultiplied colour
        alpha = np.zeros((self.h, self.w), np.float32)
        for layer in self.layers:
            which = "closed"
            if i < len(layer.mouth) and layer.mouth[i] and layer.open is not None:
                which = "open"
            elif i < len(layer.blinks) and layer.blinks[i] and layer.blink is not None:
                which = "blink"
            spr = self._sprite(layer, which)
            sh, sw = spr.shape[:2]
            x0 = int(round(layer.x * self.w - sw / 2))
            y0 = int(round(layer.y * self.h - sh))
            xa, ya = max(0, x0), max(0, y0)
            xb, yb = min(self.w, x0 + sw), min(self.h, y0 + sh)
            if xb <= xa or yb <= ya:
                continue
            s = spr[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32)
            a = s[..., 3] / 255.0
            pre[ya:yb, xa:xb] = s[..., :3] * a[..., None] + pre[ya:yb, xa:xb] * (1 - a[..., None])
            alpha[ya:yb, xa:xb] = a + alpha[ya:yb, xa:xb] * (1 - a)
        straight = pre / np.maximum(alpha, 1e-6)[..., None]
        return np.clip(straight, 0, 255).astype(np.uint8), alpha


def mouth_track(x: np.ndarray, fps: int, open_db: float = -32.0, hold: int = 2) -> np.ndarray:
    """One bool per video frame: mouth open when speech energy is above open_db.
    Short flickers are suppressed by holding each state at least `hold` frames."""
    spf = SR / fps
    n = int(np.ceil(len(x) / spf))
    out = np.zeros(n, bool)
    for i in range(n):
        a, b = int(i * spf), int(min(len(x), (i + 1) * spf))
        if b > a:
            rms = np.sqrt((x[a:b] ** 2).mean())
            out[i] = 20 * np.log10(max(rms, 1e-7)) > open_db
    # hold
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < hold and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


def default_positions(names: list[str]) -> dict:
    """Three hosts evenly spaced on a couch in the lower third."""
    xs = np.linspace(0.22, 0.78, max(len(names), 1)) if len(names) > 1 else [0.5]
    return {n: {"x": round(float(x), 3), "y": 0.93, "height": 0.34} for n, x in zip(names, xs)}


def write_rig_config(path: Path, cfg: dict) -> None:
    import json
    path.write_text(json.dumps(cfg, indent=2))
