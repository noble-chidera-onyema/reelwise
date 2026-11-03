"""Shot boundary detection.

Two detectors run over the per-frame measurements from video.py:

Hard cuts. For each frame, the HSV histogram distance and the grey-level
pixel difference to the previous frame are compared with the same measures
in the surrounding half second. A cut needs to be large in absolute terms and
large relative to its neighbourhood, so a fast camera move (high difference
on every frame) does not fire while a cut inside a busy sequence still does.
A one-to-four frame flash that returns to the earlier picture is discarded.

Gradual transitions (dissolves, fades through black). The histogram of frame
t-w is compared with frame t+w. A transition shows as a peak in that wide
comparison while no single step between neighbouring frames is large.

Thresholds were tuned on a development set of synthetic videos and then
measured once on a separate held-out set. See eval/ and docs/EVAL_SCENES.md.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .video import FrameStats


@dataclass
class SceneParams:
    window_s: float = 0.5          # neighbourhood used for the adaptive baseline
    hist_min: float = 0.24         # absolute floor for histogram distance at a cut
    pixel_min: float = 14.0        # absolute floor for grey difference at a cut (0..255)
    ratio: float = 2.0             # how far a cut must stand above its neighbourhood
    flash_max_frames: int = 4      # a jump that comes back within this many frames is a flash
    gradual_w_s: float = 0.4       # half-width of the wide comparison for dissolves and fades
    gradual_min: float = 0.45      # wide-comparison distance needed for a gradual transition
    min_scene_s: float = 0.3       # two boundaries closer than this merge into one


@dataclass
class Boundary:
    frame: int
    time_s: float
    kind: str      # "cut" | "gradual"
    score: float


@dataclass
class Scene:
    index: int
    start_frame: int
    end_frame: int          # exclusive
    start_s: float
    end_s: float
    duration_s: float
    transition_in: str      # "start" | "cut" | "gradual"
    brightness: float
    saturation: float
    colourfulness: float
    motion: float
    keyframe: int

    def to_dict(self) -> dict:
        return asdict(self)


def _bhattacharyya(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise Bhattacharyya distance between normalised histograms, 0..1."""
    bc = np.sqrt(p * q).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))


def _local_baseline(x: np.ndarray, w: int) -> np.ndarray:
    """Mean of x over [t-w, t+w], leaving out t-1..t+1 so a cut does not raise its own baseline."""
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    base = np.empty(n)
    for t in range(n):
        lo, hi = max(0, t - w), min(n, t + w + 1)
        ex_lo, ex_hi = max(lo, t - 1), min(hi, t + 2)
        total = (c[hi] - c[lo]) - (c[ex_hi] - c[ex_lo])
        count = (hi - lo) - (ex_hi - ex_lo)
        base[t] = total / count if count > 0 else 0.0
    return base


def hard_cut_scores(fs: FrameStats, p: SceneParams) -> tuple[np.ndarray, np.ndarray]:
    d = np.zeros(fs.n)
    d[1:] = _bhattacharyya(fs.hist[1:], fs.hist[:-1])
    return d, fs.motion.copy()


def detect_boundaries(fs: FrameStats, p: SceneParams | None = None) -> list[Boundary]:
    p = p or SceneParams()
    n, fps = fs.n, fs.fps
    if n < 3:
        return []
    w = max(2, int(round(p.window_s * fps)))
    d, m = hard_cut_scores(fs, p)
    base_d = _local_baseline(d, w)
    base_m = _local_baseline(m, w)

    cand: list[Boundary] = []
    t = 1
    while t < n:
        hist_hit = d[t] >= p.hist_min and d[t] >= p.ratio * (base_d[t] + 0.02)
        pix_hit = m[t] >= p.pixel_min and m[t] >= p.ratio * (base_m[t] + 1.0) and d[t] >= 0.5 * p.hist_min
        if hist_hit or pix_hit:
            # Flash check: does the picture return to frame t-1 within a few frames?
            flash_end = None
            for k in range(1, p.flash_max_frames + 1):
                if t + k >= n:
                    break
                back = _bhattacharyya(fs.hist[t - 1], fs.hist[t + k])
                if back < 0.35 * d[t]:
                    flash_end = t + k
                    break
            if flash_end is not None:
                t = flash_end + 1
                continue
            score = max(d[t] / max(p.hist_min, 1e-6), m[t] / max(p.pixel_min, 1e-6))
            cand.append(Boundary(t, t / fps, "cut", float(score)))
        t += 1

    # Gradual transitions.
    gw = max(2, int(round(p.gradual_w_s * fps)))
    if n > 2 * gw + 1:
        wide = np.zeros(n)
        wide[gw:n - gw] = _bhattacharyya(fs.hist[: n - 2 * gw], fs.hist[2 * gw:])
        hard_frames = np.array([b.frame for b in cand], dtype=int)
        for t in range(gw + 1, n - gw - 1):
            if wide[t] < p.gradual_min:
                continue
            if not (wide[t] >= wide[t - 1] and wide[t] >= wide[t + 1]):
                continue
            # Ignore the wide peak that a hard cut produces around itself.
            if hard_frames.size and np.min(np.abs(hard_frames - t)) <= gw:
                continue
            # A real transition has no single big step inside it.
            if d[t - gw: t + gw + 1].max() >= p.hist_min * 1.5:
                continue
            # The middle frame should sit between the two ends (a mix) or be dark (fade via black).
            a = _bhattacharyya(fs.hist[t - gw], fs.hist[t])
            b = _bhattacharyya(fs.hist[t], fs.hist[t + gw])
            mixed = a < wide[t] and b < wide[t]
            dark = fs.brightness[t] < 0.5 * min(fs.brightness[t - gw], fs.brightness[t + gw])
            if not (mixed or dark):
                continue
            cand.append(Boundary(t, t / fps, "gradual", float(wide[t] / p.gradual_min)))

    # Keep the strongest boundary when several fall inside min_scene_s.
    cand.sort(key=lambda b: b.frame)
    min_gap = max(1, int(round(p.min_scene_s * fps)))
    kept: list[Boundary] = []
    for b in cand:
        if kept and b.frame - kept[-1].frame < min_gap:
            if b.score > kept[-1].score:
                kept[-1] = b
            continue
        kept.append(b)
    return kept


def build_scenes(fs: FrameStats, boundaries: list[Boundary]) -> list[Scene]:
    starts = [0] + [b.frame for b in boundaries]
    kinds = ["start"] + [b.kind for b in boundaries]
    ends = starts[1:] + [fs.n]
    scenes = []
    for i, (s, e, k) in enumerate(zip(starts, ends, kinds)):
        if e <= s:
            continue
        sl = slice(s, e)
        # Motion on the first frame of a scene is the cut itself, not movement.
        mot = fs.motion[s + 1:e] if e - s > 1 else fs.motion[sl]
        scenes.append(Scene(
            index=len(scenes),
            start_frame=s, end_frame=e,
            start_s=round(s / fs.fps, 3), end_s=round(e / fs.fps, 3),
            duration_s=round((e - s) / fs.fps, 3),
            transition_in=k,
            brightness=round(float(fs.brightness[sl].mean()), 1),
            saturation=round(float(fs.saturation[sl].mean()), 1),
            colourfulness=round(float(fs.colourfulness[sl].mean()), 1),
            motion=round(float(mot.mean()), 2),
            keyframe=(s + e) // 2,
        ))
    return scenes
