"""Synthetic videos with known shot boundaries, for measuring the scene detector.

Real short-form videos with frame-accurate labels are not available to this
project, so the ground truth is manufactured. Each video is a sequence of
procedurally drawn scenes (textured background, moving shapes, camera pan)
joined by hard cuts, dissolves or fades through black. Hard negatives are
mixed in on purpose: fast pans, white flashes, captions popping on, and cuts
between two scenes that share a colour palette. The file is then re-encoded
with H.264 at a low quality setting so compression noise is present, as it
would be in a downloaded upload.

Usage:
    python eval/synth.py --out data/synth_dev --seeds 0-19
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 360, 640, 30


def _texture(rng: np.random.Generator, palette: np.ndarray, size=(W * 2, H * 2)) -> np.ndarray:
    """Low-frequency noise mapped onto a 3-colour palette."""
    small = rng.random((12, 7)).astype(np.float32)
    field = cv2.resize(small, size, interpolation=cv2.INTER_CUBIC)
    field = (field - field.min()) / (np.ptp(field) + 1e-6)
    idx = np.clip((field * 2.999).astype(int), 0, 2)
    frac = (field * 2.999) - idx
    nxt = np.clip(idx + 1, 0, 2)
    img = palette[idx] * (1 - frac[..., None]) + palette[nxt] * frac[..., None]
    grain = rng.normal(0, 6, img.shape)
    return np.clip(img + grain, 0, 255).astype(np.uint8)


def _palette(rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(0, 180)
    hues = (base + np.array([0, rng.uniform(15, 50), rng.uniform(60, 120)])) % 180
    hsv = np.stack([hues, rng.uniform(80, 230, 3), rng.uniform(60, 240, 3)], axis=1)
    hsv = hsv.astype(np.uint8)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0].astype(np.float32)


class SceneSpec:
    def __init__(self, rng, n_frames, palette=None, kind="normal"):
        self.rng = rng
        self.n = n_frames
        self.kind = kind
        self.palette = _palette(rng) if palette is None else palette
        self.bg = _texture(rng, self.palette)
        speed = rng.uniform(12, 22) if kind == "fastpan" else rng.uniform(0, 3)
        ang = rng.uniform(0, 2 * np.pi)
        self.v = np.array([np.cos(ang), np.sin(ang)]) * speed
        self.origin = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        self.shapes = []
        for _ in range(rng.integers(1, 4)):
            self.shapes.append(dict(
                p=np.array([rng.uniform(0, W), rng.uniform(0, H)]),
                v=rng.uniform(-6, 6, 2),
                r=int(rng.uniform(20, 70)),
                c=tuple(int(x) for x in rng.uniform(0, 255, 3)),
                rect=bool(rng.integers(0, 2)),
            ))
        self.flash_at = int(rng.integers(5, max(6, n_frames - 5))) if kind == "flash" else None
        self.caption_at = int(rng.integers(3, max(4, n_frames - 3))) if kind == "caption" else None

    def frame(self, i: int) -> np.ndarray:
        # Triangle-wave pan: the camera bounces at the texture edge instead of
        # wrapping, so the only discontinuities in a scene are the ones we label.
        o = self.origin + self.v * i
        x = int(W - abs((o[0] % (2 * W)) - W))
        y = int(H - abs((o[1] % (2 * H)) - H))
        img = self.bg[y:y + H, x:x + W].copy()
        for s in self.shapes:
            q = s["p"] + s["v"] * i
            p = np.array([W - abs((q[0] % (2 * W)) - W), H - abs((q[1] % (2 * H)) - H)]).astype(int)
            if s["rect"]:
                cv2.rectangle(img, (p[0] - s["r"], p[1] - s["r"]), (p[0] + s["r"], p[1] + s["r"]), s["c"], -1)
            else:
                cv2.circle(img, (int(p[0]), int(p[1])), s["r"], s["c"], -1)
        if self.flash_at is not None and self.flash_at <= i < self.flash_at + 2:
            img[:] = 250
        if self.caption_at is not None and i >= self.caption_at:
            cv2.rectangle(img, (20, H - 170), (W - 20, H - 100), (20, 20, 20), -1)
            cv2.putText(img, "WAIT FOR IT", (40, H - 122), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 230, 255), 2)
        return img


def make_video(seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    total_s = rng.uniform(20, 40)
    scenes: list[SceneSpec] = []
    transitions: list[str] = []
    frames_used = 0
    while frames_used < total_s * FPS:
        n = int(rng.uniform(0.6, 4.0) * FPS)
        kind = rng.choice(["normal", "normal", "normal", "fastpan", "flash", "caption"])
        similar = scenes and rng.random() < 0.2
        palette = scenes[-1].palette if similar else None
        scenes.append(SceneSpec(rng, n, palette=palette, kind=str(kind)))
        if len(scenes) > 1:
            r = rng.random()
            transitions.append("cut" if r < 0.75 else ("dissolve" if r < 0.87 else "fade"))
        frames_used += n

    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    truth = []
    t_frame = 0
    for si, sc in enumerate(scenes):
        trans_in = transitions[si - 1] if si > 0 else None
        trans_out = transitions[si] if si < len(transitions) else None
        tlen = 15  # 0.5 s transitions
        body_end = sc.n - (tlen // 2 if trans_out in ("dissolve", "fade") else 0)
        body_start = tlen - tlen // 2 if trans_in in ("dissolve", "fade") else 0
        if trans_in == "cut":
            truth.append({"frame": t_frame, "time_s": t_frame / FPS, "kind": "cut",
                          "similar_palette": bool(scenes[si - 1].palette is sc.palette)})
        for i in range(body_start, body_end):
            vw.write(sc.frame(i))
            t_frame += 1
        if trans_out in ("dissolve", "fade"):
            nxt = scenes[si + 1]
            mid = t_frame + tlen // 2
            truth.append({"frame": mid, "time_s": mid / FPS, "kind": "gradual",
                          "similar_palette": bool(nxt.palette is sc.palette)})
            for k in range(tlen):
                a = sc.frame(body_end + k)
                b = nxt.frame(k)
                alpha = (k + 1) / (tlen + 1)
                if trans_out == "dissolve":
                    img = cv2.addWeighted(a, 1 - alpha, b, alpha, 0)
                else:
                    img = (a * (1 - 2 * alpha)).astype(np.uint8) if alpha < 0.5 else (b * (2 * alpha - 1)).astype(np.uint8)
                vw.write(img)
                t_frame += 1
    vw.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"synth_{seed:03d}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p", str(out)], check=True)
    meta = {
        "seed": seed, "file": out.name, "fps": FPS, "n_frames": t_frame,
        "boundaries": truth,
        "hard_negatives": {
            "fastpan": sum(s.kind == "fastpan" for s in scenes),
            "flash": sum(s.kind == "flash" for s in scenes),
            "caption": sum(s.kind == "caption" for s in scenes),
        },
    }
    (out_dir / f"synth_{seed:03d}.json").write_text(json.dumps(meta, indent=1))
    return meta


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0-19")
    args = ap.parse_args()
    for s in parse_seeds(args.seeds):
        m = make_video(s, Path(args.out))
        print(m["file"], len(m["boundaries"]), "boundaries")
