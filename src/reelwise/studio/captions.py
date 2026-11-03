"""Captions, the cold-open stamp and the end card, drawn straight onto frames.

Drawn in Python with a bundled font (DejaVu Sans Bold, free licence, see
assets/fonts), so the output is identical on Windows, macOS, Linux and
Streamlit Cloud, and nothing depends on the system's font setup. On Windows,
Arial Black is used when installed, matching the CapCut style.

Layout rules:
  captions   3 words at most, speaker colour, thick black outline, a short pop
             when each chunk appears; inside the platform safe area
  stamp      big yellow, tilted, wrapped and shrunk until it fits
  end card   white, centred, wrapped to fit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
CANDIDATES = [Path("C:/Windows/Fonts/ariblk.ttf"), FONT_DIR / "DejaVuSans-Bold.ttf"]


def font_path() -> str:
    for p in CANDIDATES:
        if p.exists():
            return str(p)
    raise FileNotFoundError("no caption font found; expected assets/fonts/DejaVuSans-Bold.ttf")


_fonts: dict = {}


def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(font_path(), size)
    return _fonts[size]


def text_width(text: str, size: int, stroke: int = 0) -> int:
    l, t, r, b = font(size).getbbox(text, stroke_width=stroke)
    return r - l


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@dataclass
class Event:
    t0: float
    t1: float
    lines: list[str]
    size: int
    colour: tuple[int, int, int]
    x: float
    y: float
    stroke: int
    angle: float = 0.0
    pop: float = 1.12          # starting scale of the pop-in
    pop_s: float = 0.09
    fade_s: float = 0.0
    kind: str = "caption"


def wrap_to_fit(text: str, size: int, max_px: float, max_lines: int, stroke: int) -> tuple[list[str], int]:
    """Greedy word wrap, shrinking the font until every line fits within max_px."""
    words = text.split()
    while size > 20:
        lines, cur = [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if cur and text_width(trial, size, stroke) > max_px:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        if len(lines) <= max_lines and all(text_width(l, size, stroke) <= max_px for l in lines):
            return lines, size
        size -= 4
    return [text], size


class CaptionRenderer:
    def __init__(self, timeline: dict, profile: dict):
        from .assemble import chunk_words
        W, H = timeline["W"], timeline["H"]
        self.W, self.H = W, H
        fs = int(round(H * profile.get("caption_size", 0.048)))
        stroke = max(4, fs // 9)
        safe_l, safe_r = 0.08 * W, 0.82 * W
        self.max_px = safe_r - safe_l
        cx = (safe_l + safe_r) / 2
        y_story = H * profile.get("caption_y_story", 0.64)
        y_host = H * profile.get("caption_y_host", 0.17)
        colours = {c["name"]: hex_rgb(c.get("caption_colour", "#FFFFFF")) for c in profile.get("characters", [])}
        self.events: list[Event] = []
        self.report: list[dict] = []
        for seg in timeline["segments"]:
            y = y_host if seg["kind"] in ("intro", "outro") else H * seg.get("caption_y", y_story / H)
            for ln in seg["lines"]:
                words = ln["words"] or [{"w": w, "t0": ln["t0"], "t1": ln["t1"]} for w in ln["text"].split()]
                # Chunk by measured width, not character count.
                chunks, cur = [], []
                for w in words:
                    trial = " ".join(x["w"] for x in cur + [w])
                    if cur and (len(cur) >= 3 or text_width(trial, fs, stroke) > self.max_px
                                or cur[-1]["w"][-1:] in ".?!,"):
                        chunks.append(cur)
                        cur = []
                    cur.append(w)
                if cur:
                    chunks.append(cur)
                for i, ch in enumerate(chunks):
                    t0 = ch[0]["t0"]
                    t1 = chunks[i + 1][0]["t0"] if i + 1 < len(chunks) else max(ch[-1]["t1"], t0 + 0.25)
                    text = " ".join(w["w"] for w in ch)
                    lines, size = wrap_to_fit(text, fs, self.max_px, 2, stroke)
                    wpx = max(text_width(l, size, stroke) for l in lines)
                    self.report.append({"t0": round(t0, 2), "text": text, "est_px": wpx, "fits": wpx <= self.max_px,
                                        "kind": "caption"})
                    self.events.append(Event(t0, t1, lines, size, colours.get(ln["speaker"], (255, 255, 255)),
                                             cx, y, stroke))
            if seg["kind"] == "cold_open" and seg.get("stamp"):
                ss = int(round(H * 0.085))
                sst = max(6, ss // 9)
                lines, size = wrap_to_fit(seg["stamp"].upper(), ss, self.max_px * 0.92, 3, sst)
                wpx = max(text_width(l, size, sst) for l in lines)
                self.report.append({"t0": seg["t0"], "text": " / ".join(lines), "est_px": wpx,
                                    "fits": wpx <= self.max_px, "kind": "stamp"})
                self.events.append(Event(seg["t0"] + 0.05, seg["t1"], lines, size, (255, 212, 0), W / 2,
                                         H * (0.2 if seg.get("caption_y") else 0.36),
                                         sst, angle=6.0, pop=1.5, pop_s=0.16, kind="stamp"))
            if seg["kind"] == "end_card":
                card = profile.get("end_card_text", "FOLLOW for more")
                body, _, handle = card.partition(" @")
                cs = int(fs * 1.25)
                cst = max(4, cs // 10)
                lines, size = wrap_to_fit(body, cs, self.max_px, 2, cst)
                if handle:
                    hl, hs = wrap_to_fit("@" + handle, size, self.max_px, 1, cst)
                    size = min(size, hs)
                    lines += hl
                wpx = max(text_width(l, size, cst) for l in lines)
                self.report.append({"t0": seg["t0"], "text": " / ".join(lines), "est_px": wpx,
                                    "fits": wpx <= self.max_px, "kind": "end_card"})
                self.events.append(Event(seg["t0"], seg["t1"], lines, size, (255, 255, 255), cx, H * 0.45, cst,
                                         pop=1.0, fade_s=0.15, kind="end_card"))
        self._cache: dict[int, Image.Image] = {}

    def _sprite(self, k: int) -> Image.Image:
        if k not in self._cache:
            e = self.events[k]
            f = font(e.size)
            line_h = int(e.size * 1.18)
            widths = [text_width(l, e.size, e.stroke) for l in e.lines]
            w = max(widths) + 2 * e.stroke + 8
            h = line_h * len(e.lines) + 2 * e.stroke + 8
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            for i, (l, lw) in enumerate(zip(e.lines, widths)):
                d.text(((w - lw) / 2, e.stroke + 4 + i * line_h), l, font=f, fill=e.colour + (255,),
                       stroke_width=e.stroke, stroke_fill=(0, 0, 0, 255))
            if e.angle:
                img = img.rotate(e.angle, resample=Image.BICUBIC, expand=True)
            self._cache[k] = img
        return self._cache[k]

    def draw(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Draw every event active at time t onto a BGR frame (in place) and return it."""
        for k, e in enumerate(self.events):
            if not (e.t0 <= t < e.t1):
                continue
            spr = self._sprite(k)
            age = t - e.t0
            scale = 1.0
            if e.pop != 1.0 and age < e.pop_s:
                u = age / e.pop_s
                scale = e.pop + (1.0 - e.pop) * (u * u * (3 - 2 * u))
            alpha_mul = min(1.0, age / e.fade_s) if e.fade_s else 1.0
            if scale != 1.0:
                spr = spr.resize((max(1, int(spr.width * scale)), max(1, int(spr.height * scale))), Image.BICUBIC)
            a = np.asarray(spr, dtype=np.float32)
            x0 = int(round(e.x - spr.width / 2))
            y0 = int(round(e.y - spr.height / 2))
            xa, ya = max(0, x0), max(0, y0)
            xb, yb = min(self.W, x0 + spr.width), min(self.H, y0 + spr.height)
            if xb <= xa or yb <= ya:
                continue
            sub = a[ya - y0:yb - y0, xa - x0:xb - x0]
            al = sub[..., 3:4] / 255.0 * alpha_mul
            rgb = sub[..., 2::-1]      # RGBA -> BGR
            region = frame[ya:yb, xa:xb].astype(np.float32)
            frame[ya:yb, xa:xb] = (region * (1 - al) + rgb * al).astype(np.uint8)
        return frame
