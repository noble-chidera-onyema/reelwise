"""Free motion engine: real 3D camera moves through a still image.

Each story image gets a depth map from MiDaS v2.1 small (MIT licence, 66 MB,
downloaded once, about 0.1 s per image on a CPU). With depth known, the camera
can move through the picture the way a real camera would: things near the
lens slide and grow faster than the background, so a push-in has depth and a
sideways track shows parallax. On top of the camera move, a slow breathing
motion on the foreground and a faint handheld drift keep the frame alive.

What it does not do: make characters act. Limbs and faces move only as much
as the camera reveals. For a character doing something (a drift, a punch, a
walk), use free AI motion on that shot (see providers.HFSpaceProvider) or
write the action into the next image.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .voices import CACHE, fetch

MIDAS_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
_net = None


def depth_map(img: np.ndarray) -> np.ndarray:
    """Relative inverse depth (disparity), 0 = far, 1 = near, same size as img, edge-aware smoothed."""
    global _net
    if _net is None:
        _net = cv2.dnn.readNet(str(fetch(MIDAS_URL, CACHE / "midas" / "model-small.onnx")))
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    _net.setInput(blob)
    d = _net.forward()[0].astype(np.float32)
    d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    # Edge-aware smoothing, then grow the foreground slightly so its edge never tears.
    d = cv2.bilateralFilter(d, 9, 0.1, 9)
    d = cv2.dilate(d, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(d, (0, 0), 2)


def _cover(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (int(np.ceil(iw * s)), int(np.ceil(ih * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def subject_point(d: np.ndarray) -> tuple[float, float]:
    """Centre of the nearest large region (usually the character), as fractions."""
    m = (d > np.percentile(d, 80)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return 0.5, 0.45
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cent[k]
    # bias toward the top of the region: faces sit high in a figure
    top = stats[k, cv2.CC_STAT_TOP]
    hgt = stats[k, cv2.CC_STAT_HEIGHT]
    return float(cx / d.shape[1]), float(min(cy, top + 0.5 * hgt) / d.shape[0])


# Moves: each returns per-progress (zoom, focus_x, focus_y, shift_x, shift_y, dolly).
# zoom frames the crop; shift moves the camera sideways (scaled by depth = parallax);
# dolly pushes the camera forward (near things grow faster than far things).
MOVES = {
    "dolly_in":   lambda e, sp: (1.05, 0.5, 0.48, 0.0, 0.0, 0.10 * e),
    "pull_out":   lambda e, sp: (1.12, 0.5, 0.5, 0.0, 0.0, 0.10 * (1 - e)),
    "truck_left": lambda e, sp: (1.18, 0.5, 0.5, 0.045 * (e - 0.5), 0.0, 0.02),
    "truck_right": lambda e, sp: (1.18, 0.5, 0.5, -0.045 * (e - 0.5), 0.0, 0.02),
    "crane_up":   lambda e, sp: (1.18, 0.5, 0.5, 0.0, 0.04 * (e - 0.5), 0.03 * e),
    "close_push": lambda e, sp: (1.9, sp[0], sp[1], 0.0, 0.0, 0.08 * e),
    "close_drift": lambda e, sp: (2.1, sp[0], sp[1], 0.018 * (e - 0.5), 0.0, 0.03),
    "orbit":      lambda e, sp: (1.25, 0.5, 0.5, 0.035 * np.sin(np.pi * (e - 0.5)), 0.0, 0.05 * e),
}


def move_for(prompt: str, k: int) -> str:
    p = prompt.lower()
    if any(w in p for w in ("close-up", "close up", "insert", "detail", "eyes", "hands", "face")):
        return "close_push" if k % 2 == 0 else "close_drift"
    if any(w in p for w in ("top-down", "drone", "high", "aerial", "crane", "rise")):
        return "crane_up"
    if any(w in p for w in ("pull", "reveal", "wide establishing")):
        return "pull_out"
    if any(w in p for w in ("track", "pan", "sideways", "alongside", "walks", "drives")):
        return "truck_left" if k % 2 == 0 else "truck_right"
    if "orbit" in p or "around" in p:
        return "orbit"
    if "push" in p or "dramatic" in p:
        return "dolly_in"
    return ["dolly_in", "truck_left", "close_push", "orbit", "truck_right", "crane_up"][k % 6]


class ParallaxRenderer:
    def __init__(self, image_path: str, out_w: int, out_h: int, oversample: float = 1.0):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cannot read {image_path}")
        self.full_w, self.full_h = out_w, out_h
        self.fit = None
        if (img.shape[1] / img.shape[0]) / (out_w / out_h) > 1.25:
            # Landscape panel in a vertical frame: show the whole picture across the middle
            # with a soft blurred copy above and below, so nothing is cropped away.
            inner_h = int(out_w * img.shape[0] / img.shape[1]) // 2 * 2
            bg = _cover(img, out_w, out_h)
            bg = cv2.GaussianBlur(bg, (0, 0), 28)
            self.fit = ((bg.astype(np.float32) * 0.5).astype(np.uint8), (out_h - inner_h) // 2)
            out_h = inner_h
        self.ow, self.oh = out_w, out_h
        self.m = 1.12 if self.fit is not None else 1.3   # canvas margin for camera moves
        # How far we can zoom before upscaled blur shows: total magnification of the source
        # stays at or under 2.4x, and anything above 1.5x gets a light unsharp mask.
        self.cover = max(out_w * self.m / img.shape[1], out_h * self.m / img.shape[0])
        self.zmax = max(1.05, 2.4 / self.cover)
        # Work canvas at output aspect, 30% larger so moves never run off the edge.
        self.W, self.H = int(out_w * self.m * oversample), int(out_h * self.m * oversample)
        self.img = _cover(img, self.W, self.H)
        self.d = depth_map(self.img)
        self.sp = subject_point(self.d)
        gy, gx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        self.gx, self.gy = gx / out_w, gy / out_h      # output grid in 0..1

    def frame(self, move: str, u: float, t: float, life: float = 1.0) -> np.ndarray:
        e = u * u * (3 - 2 * u)
        z, fx, fy, sx, sy, dolly = MOVES[move](e, self.sp)
        if z > self.zmax:            # keep the framing sharp on low-resolution images
            z = self.zmax
        # handheld drift: two slow incommensurate sines, a fraction of a percent
        hx = life * 0.0016 * (np.sin(t * 0.9) + 0.6 * np.sin(t * 2.3 + 1.1))
        hy = life * 0.0012 * (np.sin(t * 0.7 + 0.4) + 0.5 * np.sin(t * 1.9))
        cw, ch = 1.0 / (self.m * z), 1.0 / (self.m * z)          # crop size in canvas fractions
        cx = np.clip(fx, cw / 2, 1 - cw / 2)
        cy = np.clip(fy, ch / 2, 1 - ch / 2)
        # source position before parallax (canvas fractions)
        X = cx + (self.gx - 0.5) * cw + hx
        Y = cy + (self.gy - 0.5) * ch + hy
        for _ in range(2):   # fixed-point: look up depth where this pixel comes from
            dd = cv2.remap(self.d, (X * self.W).astype(np.float32), (Y * self.H).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            breathe = life * 0.004 * np.sin(t * 1.6) * np.clip((dd - 0.6) / 0.4, 0, 1)
            scale = 1.0 + dolly * dd + breathe
            Xs = cx + (self.gx - 0.5) * cw / scale + hx - sx * dd
            Ys = cy + (self.gy - 0.5) * ch / scale + hy - sy * dd
            X, Y = Xs, Ys
        out = cv2.remap(self.img, (X * self.W).astype(np.float32), (Y * self.H).astype(np.float32),
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if self.cover * z > 1.5:
            blur = cv2.GaussianBlur(out, (0, 0), 1.2)
            out = cv2.addWeighted(out, 1.45, blur, -0.45, 0)
        if self.fit is not None:
            bg, y0 = self.fit
            frame = bg.copy()
            frame[y0:y0 + out.shape[0]] = out
            return frame
        return out


def render_job(image_path: str, shots: list[dict], lengths: list[float], total_s: float, out: Path,
               w: int, h: int, fps: int, seed_key: str = "") -> None:
    r = ParallaxRenderer(image_path, w, h)
    lengths = (np.array(lengths, float) / sum(lengths) * total_s).tolist()
    off = int(hashlib.sha1(seed_key.encode()).hexdigest(), 16) % 6
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
           "-pix_fmt", "yuv420p", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t = 0.0
    prev = None
    for k, (sh, L) in enumerate(zip(shots, lengths)):
        mv = move_for(sh.get("prompt", ""), k + off)
        if mv == prev:
            mv = move_for("", k + off + 1)
        prev = mv
        n = max(1, int(round(L * fps)))
        for i in range(n):
            f = r.frame(mv, i / max(n - 1, 1), t)
            proc.stdin.write(f.tobytes())
            t += 1 / fps
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed writing motion clip")
