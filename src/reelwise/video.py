"""Read a video once and keep compact per-frame measurements.

Everything downstream (scene cuts, motion, colour, hook metrics) works from
the arrays built here, so the file is decoded a single time. Frames are
shrunk before measuring; shot boundaries and colour statistics survive
downscaling, and it keeps a ten-minute video inside a few hundred MB of RAM.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Small working size. Histogram and pixel-difference measures are stable at
# this resolution and it is roughly 100x cheaper than full HD.
WORK_W, WORK_H = 128, 72
HIST_BINS = (8, 4, 4)  # H, S, V


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float
    has_audio: bool
    codec: str = ""

    @property
    def orientation(self) -> str:
        if self.height > self.width * 1.1:
            return "vertical"
        if self.width > self.height * 1.1:
            return "horizontal"
        return "square"


@dataclass
class FrameStats:
    """Per-frame measurements, all arrays of length n_frames."""

    fps: float
    hist: np.ndarray            # (n, 128) normalised HSV histograms
    brightness: np.ndarray      # mean V channel, 0..255
    saturation: np.ndarray      # mean S channel, 0..255
    colourfulness: np.ndarray   # Hasler and Suesstrunk (2003) metric
    motion: np.ndarray          # mean abs grey difference to previous frame, 0..255
    thumbs: list = field(default_factory=list)  # (index, small BGR frame) for keyframe picking

    @property
    def n(self) -> int:
        return len(self.brightness)

    def t(self, i: int) -> float:
        return i / self.fps


def probe(path: str | Path) -> VideoInfo:
    """Container metadata via ffprobe, falling back to OpenCV if ffprobe is missing."""
    path = str(path)
    if shutil.which("ffprobe"):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", path],
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0:
            meta = json.loads(out.stdout)
            streams = meta.get("streams", [])
            v = next((s for s in streams if s.get("codec_type") == "video"), None)
            if v is not None:
                num, _, den = v.get("avg_frame_rate", "0/1").partition("/")
                fps = float(num) / float(den or 1) if float(den or 1) else 0.0
                if fps <= 0:
                    num, _, den = v.get("r_frame_rate", "30/1").partition("/")
                    fps = float(num) / float(den or 1)
                dur = float(v.get("duration") or meta.get("format", {}).get("duration") or 0)
                n = int(v.get("nb_frames") or round(dur * fps))
                w, h = int(v["width"]), int(v["height"])
                # Phone footage often stores rotation as side data, not in the pixels.
                rot = 0
                for sd in v.get("side_data_list", []) or []:
                    if "rotation" in sd:
                        rot = abs(int(sd["rotation"]))
                if rot in (90, 270):
                    w, h = h, w
                return VideoInfo(
                    path=path, width=w, height=h, fps=fps, n_frames=n,
                    duration_s=dur,
                    has_audio=any(s.get("codec_type") == "audio" for s in streams),
                    codec=v.get("codec_name", ""),
                )
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps, n_frames=n, duration_s=n / fps if fps else 0.0,
        has_audio=False,
    )
    cap.release()
    return info


def _colourfulness(bgr: np.ndarray) -> float:
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def read_frames(path: str | Path, thumb_every_s: float = 0.5, thumb_width: int = 512) -> FrameStats:
    """Decode every frame once and measure it.

    A larger copy of a frame is kept every `thumb_every_s` seconds so keyframes
    can be sent to a vision model later without decoding the file again.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    thumb_step = max(1, int(round(thumb_every_s * fps)))

    hists, bright, sat, colour, motion, thumbs = [], [], [], [], [], []
    prev_grey = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256]).ravel()
        h /= max(h.sum(), 1.0)
        hists.append(h.astype(np.float32))
        bright.append(float(hsv[..., 2].mean()))
        sat.append(float(hsv[..., 1].mean()))
        colour.append(_colourfulness(small))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        motion.append(0.0 if prev_grey is None else float(np.abs(grey - prev_grey).mean()))
        prev_grey = grey
        if i % thumb_step == 0:
            scale = thumb_width / frame.shape[1]
            thumbs.append((i, cv2.resize(frame, (thumb_width, int(frame.shape[0] * scale)),
                                         interpolation=cv2.INTER_AREA)))
        i += 1
    cap.release()
    if i == 0:
        raise ValueError(f"No frames decoded from {path}")
    return FrameStats(
        fps=fps,
        hist=np.vstack(hists),
        brightness=np.array(bright),
        saturation=np.array(sat),
        colourfulness=np.array(colour),
        motion=np.array(motion),
        thumbs=thumbs,
    )


def grab_frame(path: str | Path, index: int, width: int = 512) -> np.ndarray | None:
    """Fetch one frame at full quality (used when a thumb is not close enough)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
