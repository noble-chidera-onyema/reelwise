"""Cut, composite, caption, mix and export the episode.

Timeline, in order of the plan's segments. The default ("end" reveal):
  COLD OPEN  the most dramatic story shot, full frame, with the yellow stamp
  INTRO      the story keeps playing full frame; the hosts are heard, not seen
  SCENES     story shots, each cut to the narration over it
  OUTRO      still full frame, until the sign-off: on "And that's one worth
             telling" the camera pulls back out of the TV to the hosts
  END CARD   2 s follow card over the last frame, blurred

With the "classic" reveal the camera pulls out to the hosts for the whole INTRO
and pushes back through the screen into scene 1, and the OUTRO pulls out at once.

In the TV layout (16:9) the vertical rooftop set sits in the middle of the
frame with a soft blurred copy either side, so the hosts are never cropped.

Every segment is a whole number of frames and every shot is cut to the
frame, so picture and narration cannot drift apart: the narration for a
scene is placed at that scene's first frame.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .host import HostRig
from .tv import TVCompositor, detect_screen
from .voices import SR, read_wav, write_wav

REVEAL_LEAD_S = 0.35    # the pull-out starts this long before the sign-off's first word
HEAD_TRIM_S = 0.2       # generated clips often start with a few near-still frames
MAX_STRETCH = 1.2       # slowest we slow footage down before asking for a longer clip


class AssemblyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- timing

def build_timeline(plan: dict, zoom_s: float, end_card_s: float) -> dict:
    fps = plan["format"]["fps"]
    reveal = plan["format"].get("host_reveal", "classic")
    t = 0.0
    segs = []
    for s in plan["segments"]:
        f0 = int(round(t * fps))
        t1 = t + s["duration_s"]
        f1 = int(round(t1 * fps))
        lines = []
        for ln in s["lines"]:
            g0 = t + ln["start_s"]
            lines.append({"speaker": ln["speaker"], "text": ln["text"], "audio": ln["audio"],
                          "t0": round(g0, 3), "t1": round(g0 + ln["duration_s"], 3),
                          "words": [{"w": w["w"], "t0": round(g0 + w["start"], 3), "t1": round(g0 + w["end"], 3)}
                                    for w in ln.get("words", [])]})
        seg = {"id": s["id"], "kind": s["kind"], "label": s["label"], "t0": round(f0 / fps, 3),
               "t1": round(f1 / fps, 3), "f0": f0, "f1": f1, "lines": lines, "stamp": s.get("stamp", "")}
        # Where the hosts can be seen: the frame within an OUTRO where the pull-out starts.
        if s["kind"] == "outro":
            if reveal == "end" and lines:
                start = max(seg["t0"], lines[-1]["t0"] - REVEAL_LEAD_S)
                seg["reveal_f"] = min(f1 - f0, max(0, int(round((start - seg["t0"]) * fps))))
            else:
                seg["reveal_f"] = 0
        for ln in lines:
            if s["kind"] == "intro":
                ln["view"] = "host" if reveal == "classic" else "story"
            elif s["kind"] == "outro":
                ln["view"] = "host" if ln["t0"] >= seg["t0"] + seg["reveal_f"] / fps - 1e-6 else "story"
            else:
                ln["view"] = "story"
        segs.append(seg)
        t = t1
    end = None
    if end_card_s > 0:
        f0 = int(round(t * fps))
        f1 = int(round((t + end_card_s) * fps))
        end = {"id": "end", "kind": "end_card", "label": "End card", "t0": f0 / fps, "t1": f1 / fps,
               "f0": f0, "f1": f1, "lines": [], "stamp": ""}
        segs.append(end)
    kinds = [s["kind"] for s in segs]
    story_idx = [i for i, k in enumerate(kinds) if k == "story"]
    return {"fps": fps, "W": plan["format"]["width"], "H": plan["format"]["height"], "zoom_s": zoom_s,
            "reveal": reveal,
            "segments": segs, "frames": segs[-1]["f1"] if segs else 0,
            "has_intro": "intro" in kinds, "has_outro": "outro" in kinds,
            "first_story": story_idx[0] if story_idx else None,
            "last_story": story_idx[-1] if story_idx else None}


def roll_frames(timeline: dict) -> dict[str, list[int]]:
    """Extra story frames each segment's footage must carry for the TV, as [pre, post].

    The episode opens inside the TV. If there is a cold open, its footage keeps
    playing on the TV while the camera pulls out to the hosts (post). While the
    hosts talk, the TV is already playing scene 1 (pre), and the push back in
    lands on that same footage, so nothing on the TV ever freezes. During the
    outro the TV keeps playing the last scene to the end (post).
    """
    z = int(round(timeline["zoom_s"] * timeline["fps"]))
    segs = timeline["segments"]
    out = {s["id"]: [0, 0] for s in segs}
    intro = next((s for s in segs if s["kind"] == "intro"), None)
    outro = next((s for s in segs if s["kind"] == "outro"), None)
    cold = next((s for s in segs if s["kind"] == "cold_open"), None)
    if timeline["first_story"] is None:
        return out
    first = segs[timeline["first_story"]]
    last = segs[timeline["last_story"]]
    if intro:
        n = intro["f1"] - intro["f0"]
        if cold:
            out[cold["id"]][1] = z
            out[first["id"]][0] = n - z
        else:
            out[first["id"]][0] = n
    if outro:
        out[last["id"]][1] = outro["f1"] - outro["f0"]
    return out


# --------------------------------------------------------------------------- shots

def _cover_filter(W: int, H: int) -> str:
    return (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={W}:{H},setsar=1")


def shot_sources(job: dict, clip_meta: dict) -> list[tuple[float, float]]:
    """Source (start, end) inside the clip for each shot of a job.

    If the model cut between shots where asked, its own cut points are used
    (the ones nearest the planned split points). Otherwise the clip is divided
    in proportion to the planned shot lengths. The first few frames of a
    generated clip are often nearly still, so they are skipped when there is
    footage to spare.
    """
    dur = clip_meta["duration_s"]
    n = len(job["shots"])
    w = np.array(job.get("lengths") or [s["duration_s"] for s in job["shots"]], float)
    planned = list(np.cumsum(w / w.sum() * dur)[:-1])
    cuts = [c for c in clip_meta.get("cuts_s", []) if 0.3 < c < dur - 0.3]
    edges = None
    if n > 1 and len(cuts) >= n - 1:
        chosen = sorted({min(cuts, key=lambda c: abs(c - p)) for p in planned})
        if len(chosen) == n - 1:
            edges = [0.0] + chosen + [dur]
    if edges is None:
        edges = [0.0] + planned + [dur]
    out = []
    for i in range(n):
        start, end = edges[i], edges[i + 1]
        if i == 0 and end - start - HEAD_TRIM_S >= w[0] * 0.98:
            start += HEAD_TRIM_S
        out.append((round(start, 3), round(end, 3)))
    return out


def render_shot(clip: str, src: tuple[float, float], frames: int, fps: int, W: int, H: int, out: Path) -> dict:
    need = frames / fps
    have = max(0.05, src[1] - src[0])
    stretch = need / have
    take = have if stretch > 1 else need
    factor = max(1.0, stretch)
    vf = (f"setpts={factor:.5f}*(PTS-STARTPTS),fps={fps},{_cover_filter(W, H)},"
          f"tpad=stop_mode=clone:stop_duration=1")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{src[0]:.3f}", "-i", clip, "-t", f"{take:.3f}",
                    "-vf", vf, "-frames:v", str(frames), "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "14", "-pix_fmt", "yuv420p", str(out)], check=True)
    return {"file": str(out), "frames": frames, "stretch": round(stretch, 3), "src": [round(src[0], 3), round(src[1], 3)]}


class FrameReader:
    """Sequential reader over a list of shot files, as one stream."""

    def __init__(self, files: list[str]):
        self.files = list(files)
        self.k = 0
        self.cap = cv2.VideoCapture(self.files[0]) if self.files else None
        self.last = None

    def read(self) -> np.ndarray:
        while self.cap is not None:
            ok, f = self.cap.read()
            if ok:
                self.last = f
                return f
            self.cap.release()
            self.k += 1
            self.cap = cv2.VideoCapture(self.files[self.k]) if self.k < len(self.files) else None
        if self.last is None:
            raise AssemblyError("no frames to read")
        return self.last


# --------------------------------------------------------------------------- captions

def ass_colour(hex_rgb: str, alpha: int = 0) -> str:
    h = hex_rgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def chunk_words(words: list[dict], max_words: int = 3, max_chars: int = 16) -> list[list[dict]]:
    out, cur = [], []
    for w in words:
        text = " ".join(x["w"] for x in cur + [w])
        if cur and (len(cur) >= max_words or len(text) > max_chars or cur[-1]["w"][-1:] in ".?!,"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def fit_stamp(text: str, size: int, max_px: float, max_lines: int = 3) -> tuple[list[str], int]:
    """Break the stamp into balanced lines and shrink the font until every line fits."""
    words = text.split()
    for size_ in range(size, 24, -4):
        per_line = max(1, int(max_px / (size_ * 0.66)))
        lines_, cur = [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > per_line:
                lines_.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines_.append(cur)
        if len(lines_) <= max_lines and all(len(l) <= per_line for l in lines_):
            return lines_, size_
    return [text], 24


def _ts(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_ass(timeline: dict, profile: dict, out: Path) -> list[dict]:
    W, H = timeline["W"], timeline["H"]
    font = profile.get("caption_font", "Arial Black")
    fs = int(round(H * profile.get("caption_size", 0.048)))
    stamp_fs = int(round(H * 0.085))
    colours = {c["name"]: c.get("caption_colour", "#FFFFFF") for c in profile.get("characters", [])}
    safe_l, safe_r = 0.08 * W, 0.82 * W
    max_px = safe_r - safe_l
    cx = (safe_l + safe_r) / 2
    y_story = H * profile.get("caption_y_story", 0.64)
    y_host = H * profile.get("caption_y_host", 0.17)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{max(5, fs // 8)},{max(2, fs // 30)},5,0,0,0,1
Style: Stamp,{font},{stamp_fs},{ass_colour('#FFD400')},&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,2,0,1,{max(6, stamp_fs // 9)},0,5,0,0,0,1
Style: Card,{font},{int(fs * 0.95)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{max(4, fs // 12)},0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events, report = [], []
    for seg in timeline["segments"]:
        y = y_host if seg["kind"] in ("intro", "outro") else y_story
        for ln in seg["lines"]:
            col = ass_colour(colours.get(ln["speaker"], "#FFFFFF"))
            chunks = chunk_words(ln["words"] or [{"w": w, "t0": ln["t0"], "t1": ln["t1"]} for w in ln["text"].split()],
                                 max_chars=max(8, int(max_px / (fs * 0.62))))
            for i, ch in enumerate(chunks):
                t0 = ch[0]["t0"]
                t1 = chunks[i + 1][0]["t0"] if i + 1 < len(chunks) else max(ch[-1]["t1"], t0 + 0.25)
                text = " ".join(w["w"] for w in ch)
                width = len(text) * fs * 0.62
                report.append({"t0": t0, "text": text, "est_px": int(width), "fits": width <= max_px})
                pop = r"{\fscx112\fscy112\t(0,90,\fscx100\fscy100)}"
                events.append(f"Dialogue: 1,{_ts(t0)},{_ts(t1)},Cap,,0,0,0,,{{\\pos({cx:.0f},{y:.0f})\\c{col}}}{pop}{_esc(text)}")
        if seg["kind"] == "cold_open" and seg.get("stamp"):
            lines_, size = fit_stamp(seg["stamp"].upper(), stamp_fs, max_px)
            width = max(len(l) for l in lines_) * size * 0.66
            report.append({"t0": seg["t0"], "text": " / ".join(lines_), "est_px": int(width),
                           "fits": width <= max_px, "kind": "stamp"})
            st = r"{\pos(%d,%d)\fs%d\frz6\fscx150\fscy150\t(0,160,\fscx100\fscy100)}" % (W // 2, int(H * 0.36), size)
            events.append(f"Dialogue: 2,{_ts(seg['t0'] + 0.05)},{_ts(seg['t1'])},Stamp,,0,0,0,,{st}"
                          + r"\N".join(_esc(l) for l in lines_))
        if seg["kind"] == "end_card":
            card = profile.get("end_card_text", "FOLLOW for more")
            body, handle = (card.split(" @", 1) + [""])[:2]
            cfs = int(fs * 1.25)
            lines_, size = fit_stamp(body, cfs, max_px, max_lines=2)
            if handle:
                hl, hs = fit_stamp("@" + handle, size, max_px * 0.95, max_lines=1)
                size = min(size, hs)
                lines_ += hl
            width = max(len(l) for l in lines_) * size * 0.66
            report.append({"t0": seg["t0"], "text": " / ".join(lines_), "est_px": int(width),
                           "fits": width <= max_px, "kind": "end_card"})
            events.append(f"Dialogue: 2,{_ts(seg['t0'])},{_ts(seg['t1'])},Card,,0,0,0,,"
                          f"{{\\pos({cx:.0f},{int(H * 0.45)})\\fs{size}\\fad(150,0)}}"
                          + r"\N".join(_esc(l) for l in lines_))
    out.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return report


def write_srt(timeline: dict, out: Path) -> None:
    def ts(t):
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    rows, k = [], 1
    for seg in timeline["segments"]:
        for ln in seg["lines"]:
            rows.append(f"{k}\n{ts(ln['t0'])} --> {ts(ln['t1'])}\n{ln['text']}\n")
            k += 1
    out.write_text("\n".join(rows), encoding="utf-8")


# --------------------------------------------------------------------------- audio

def whoosh(dur: float, rising: bool = True, seed: int = 3) -> np.ndarray:
    """A soft filtered-noise swell, generated here so it carries no rights issues."""
    n = int(SR * dur)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, n).astype(np.float32)
    t = np.linspace(0, 1, n)
    env = np.sin(np.pi * t) ** 2 * (t if rising else 1 - t) * 1.6
    # one-pole low-pass whose cutoff sweeps with the envelope
    out = np.zeros(n, np.float32)
    y = 0.0
    for i in range(n):
        a = 0.02 + 0.25 * (t[i] if rising else 1 - t[i])
        y += a * (noise[i] - y)
        out[i] = y
    out *= env / max(np.abs(out).max(), 1e-6)
    return out * 0.18


def mix_audio(timeline: dict, music_path: str | None, out: Path, music_db: float = -20.0,
              duck_db: float = -9.0, sfx: bool = True) -> dict:
    total = timeline["frames"] / timeline["fps"]
    n = int(math.ceil(total * SR)) + SR // 10
    voice = np.zeros(n, np.float32)
    for seg in timeline["segments"]:
        for ln in seg["lines"]:
            x = read_wav(ln["audio"])
            a = int(round(ln["t0"] * SR))
            b = min(n, a + len(x))
            voice[a:b] += x[: b - a]
    fx = np.zeros(n, np.float32)
    if sfx:
        z = timeline["zoom_s"]
        for seg in timeline["segments"]:
            if seg["kind"] == "intro" and timeline.get("reveal", "classic") == "classic":
                a = int((seg["t1"] - z - 0.1) * SR)
                w = whoosh(z + 0.25, True)
                fx[max(0, a):max(0, a) + len(w)] += w[: n - max(0, a)]
            if seg["kind"] == "outro":
                a = int(max(0.0, seg["t0"] + seg.get("reveal_f", 0) / timeline["fps"] - 0.15) * SR)
                w = whoosh(z + 0.25, False, seed=5)
                fx[a:a + len(w)] += w[: n - a]
    music = np.zeros(n, np.float32)
    if music_path:
        m = read_wav(music_path)
        if len(m):
            reps = int(math.ceil(n / len(m)))
            m = np.tile(m, reps)[:n]
            m = m / max(np.sqrt((m ** 2).mean()), 1e-6) * 0.1   # roughly -20 dBFS RMS before trim
            # Duck under the voice: envelope from voice energy, 80 ms attack, 300 ms release.
            blk = int(0.02 * SR)
            nb = n // blk
            e = np.sqrt((voice[: nb * blk].reshape(nb, blk) ** 2).mean(1)) > 0.01
            g = np.ones(nb, np.float32)
            duck = 10 ** (duck_db / 20)
            cur = 1.0
            for i in range(nb):
                target = duck if e[i] else 1.0
                rate = 0.25 if target < cur else 0.067
                cur += (target - cur) * rate
                g[i] = cur
            gain = np.repeat(g, blk)
            gain = np.concatenate([gain, np.full(n - len(gain), gain[-1] if len(gain) else 1.0)])
            fade = np.ones(n, np.float32)
            fi, fo = int(0.5 * SR), int(1.5 * SR)
            fade[:fi] = np.linspace(0, 1, fi)
            fade[-fo:] = np.linspace(1, 0, fo)
            music = m * gain * fade * 10 ** ((music_db + 20) / 20)
    mix = voice + music + fx
    peak = float(np.abs(mix).max()) or 1.0
    if peak > 0.98:
        mix = mix / peak * 0.98
    write_wav(out, mix)
    return {"voice_rms_db": round(20 * math.log10(max(np.sqrt((voice ** 2).mean()), 1e-7)), 1),
            "music": bool(music_path)}


def _measure(wav: Path, pre: str = "") -> dict:
    """ffmpeg loudnorm measurement of a file, optionally after a filter chain."""
    af = (pre + "," if pre else "") + "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-nostdin", "-i", str(wav), "-af", af, "-f", "null", "-"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    return json.loads(m.group(0)) if m else {}


def loudnorm_args(wav: Path, target: float = -14.0, tp: float = -1.5, lra: float = 11.0) -> str:
    """Filter chain that brings the mix to `target` LUFS with true peak at or under `tp`.

    When plain gain would push the peaks over `tp` (cloned voices are often quiet with sharp
    peaks), loudnorm would quietly fall back and land short of the target (-15.2 LUFS on the
    first Chatterbox render). Then the mix gets gain plus a peak limiter instead, and the gain
    is corrected from a fresh measurement until the result is within 0.3 LU of the target."""
    d = _measure(wav)
    if not d:
        return f"loudnorm=I={target}:TP={tp}:LRA={lra}"
    i, peak = float(d["input_i"]), float(d["input_tp"])
    if peak + (target - i) <= tp - 0.2:
        return (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={d['input_i']}:measured_TP={d['input_tp']}:"
                f"measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}:"
                f"linear=true")
    gain, limit = target - i, 10 ** ((tp - 1.2) / 20)   # headroom for AAC overshoot
    af = ""
    for _ in range(4):
        af = f"volume={gain:.2f}dB,alimiter=limit={limit:.3f}:attack=1:release=50:level=false"
        m = _measure(wav, af)
        if not m:
            break
        deficit = target - float(m["input_i"])
        if abs(deficit) < 0.3:
            break
        gain += deficit
    return af


# --------------------------------------------------------------------------- picture

def render_picture(plan: dict, timeline: dict, shot_files: dict[str, list[str]], host_cfg: dict | None,
                   out: Path, progress=None) -> dict:
    W, H, fps = timeline["W"], timeline["H"], timeline["fps"]
    z_frames = int(round(timeline["zoom_s"] * fps))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "12", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    from .captions import CaptionRenderer
    caps = CaptionRenderer(timeline, plan["profile"])
    counter = {"i": 0}

    def write(frame: np.ndarray) -> None:
        # Draw on a copy: the same story frames can be shown again on the TV.
        out = caps.draw(frame.copy(), counter["i"] / fps)
        enc.stdin.write(out.tobytes())
        counter["i"] += 1

    end_reveal = timeline.get("reveal", "classic") == "end"
    need_host = timeline["has_outro"] or (timeline["has_intro"] and not end_reveal)
    rig = tvc = plate = None
    info = {}
    # The set is drawn on a vertical canvas. In the TV layout it is placed in the middle of
    # the 16:9 frame with a blurred, dimmed copy either side (pad > 0).
    vw = W if H >= W else int(round(H * 9 / 16)) // 2 * 2
    pad = (W - vw) // 2
    if need_host:
        if not host_cfg:
            raise AssemblyError("INTRO/OUTRO need a host scene. Upload one, or remove those sections.")
        vcanvas = (vw, H)
        if host_cfg.get("hosts"):
            rig = HostRig(host_cfg, vcanvas)
            plate_frame = rig.plate.frame(0)
        else:
            from .host import Plate
            plate = Plate(host_cfg["background"], vcanvas)
            plate_frame = plate.frame(0)
        screen = detect_screen(plate_frame, host_cfg.get("screen_corners"))
        if screen is None:
            raise AssemblyError("No TV screen found in the host scene. Make the screen flat green (#00FF00) "
                                "or enter its four corners.")
        if pad:
            from .tv import Screen
            matte = None
            if screen.matte is not None:
                matte = np.zeros((H, W), np.uint8)
                matte[:, pad:pad + vw] = screen.matte
            screen = Screen(screen.corners + np.array([pad, 0], np.float32), screen.method, matte)
        tvc = TVCompositor((W, H), screen, (W, H))
        info["screen"] = screen.to_dict()

    side_cache: dict = {}

    def widen(frame: np.ndarray, still: bool) -> np.ndarray:
        if not pad:
            return frame
        if still and "bg" in side_cache:
            bg = side_cache["bg"]
        else:
            # Cover the 16:9 frame with the set (scaled to the full width), blur and dim it.
            sh = int(round(H * W / vw))
            big = cv2.resize(frame, (W // 4, sh // 4), interpolation=cv2.INTER_AREA)
            y0 = (big.shape[0] - H // 4) // 2
            bg = cv2.resize(big[y0:y0 + H // 4], (W, H), interpolation=cv2.INTER_LINEAR)
            bg = (cv2.GaussianBlur(bg, (0, 0), 10).astype(np.float32) * 0.5).astype(np.uint8)
            if still:
                side_cache["bg"] = bg
        out = bg.copy()
        out[:, pad:pad + vw] = frame
        return out

    def host_bg(i):
        p = rig.plate if rig else plate
        return widen(p.frame(i), not p.is_video)

    def host_overlay(i):
        if not rig:
            return None
        bgr, a = rig.overlay(i)
        if not pad:
            return bgr, a
        bw = np.zeros((H, W, 3), bgr.dtype)
        aw = np.zeros((H, W), a.dtype)
        bw[:, pad:pad + vw] = bgr
        aw[:, pad:pad + vw] = a
        return bw, aw

    segs = timeline["segments"]
    rf = roll_frames(timeline)
    cold_tail: list[np.ndarray] = []
    written = 0
    story_tail_frames: list[np.ndarray] = []
    last_frame = None
    k_global = 0
    for si, seg in enumerate(segs):
        n = seg["f1"] - seg["f0"]
        if seg["kind"] in ("story", "cold_open"):
            reader = FrameReader(shot_files[seg["id"]])
            pre_n, post_n = rf[seg["id"]]
            for _ in range(pre_n):           # already shown on the TV during the intro
                reader.read()
            for i in range(n):
                f = reader.read()
                write(f)
                last_frame = f
            post = [reader.read() for _ in range(post_n)]
            if seg["kind"] == "cold_open":
                cold_tail = post
            if si == timeline["last_story"]:
                story_tail_frames = post
            written += n
        elif seg["kind"] == "intro":
            # Starts inside the TV (full frame), pulls back to the hosts, then pushes back in.
            first_id = segs[timeline["first_story"]]["id"]
            pre_n = rf[first_id][0]
            pre = FrameReader(shot_files[first_id])
            pre_frames = [pre.read() for _ in range(pre_n)]
            tv_feed = (cold_tail + pre_frames) if cold_tail else pre_frames
            if end_reveal:
                # The hosts are heard, not seen: the story keeps playing full frame.
                for i in range(n):
                    f = tv_feed[min(i, len(tv_feed) - 1)]
                    write(f)
                    last_frame = f
                written += n
                k_global += n
                if progress:
                    progress(si + 1, len(segs), seg["label"])
                continue
            if rig:
                rig.drive(plan_lines(plan, seg["id"]), 0.0, n, fps, seed=11)
            for i in range(n):
                if i < z_frames:
                    u = 1.0 - 0.94 * ((i + 1) / z_frames)
                elif i >= n - z_frames:
                    j = i - (n - z_frames)
                    u = 0.09 + 0.91 * ((j + 1) / z_frames)
                else:
                    u = 0.06 + 0.03 * ((i - z_frames) / max(n - 2 * z_frames - 1, 1))   # slow drift toward the TV
                story = tv_feed[min(i, len(tv_feed) - 1)]
                ov = host_overlay(i)
                f = tvc.render(host_bg(k_global + i), story, min(max(u, 0.0), 1.0), overlay=ov)
                write(f)
                last_frame = f
            written += n
        elif seg["kind"] == "outro":
            tail = story_tail_frames or [last_frame]
            if rig:
                rig.drive(plan_lines(plan, seg["id"]), 0.0, n, fps, seed=13)
            r0 = seg.get("reveal_f", 0)          # full frame until the sign-off, then pull out
            for i in range(n):
                story = tail[min(i, len(tail) - 1)]
                if i < r0:
                    write(story)
                    last_frame = story
                    continue
                j = i - r0
                if j < z_frames:
                    u = 1.0 - 0.94 * ((j + 1) / z_frames)
                else:
                    u = 0.06 * (1 - (j - z_frames) / max(n - r0 - z_frames - 1, 1))
                f = tvc.render(host_bg(k_global + i), story, max(u, 0.0), overlay=host_overlay(i))
                write(f)
                last_frame = f
            written += n
        elif seg["kind"] == "end_card":
            base = last_frame if last_frame is not None else np.zeros((H, W, 3), np.uint8)
            card = cv2.GaussianBlur(base, (0, 0), 18)
            card = (card.astype(np.float32) * 0.45).astype(np.uint8)
            for i in range(n):
                a = min(1.0, (i + 1) / 6)
                f = (base.astype(np.float32) * (1 - a) + card.astype(np.float32) * a).astype(np.uint8)
                write(f)
            written += n
        k_global += n
        if progress:
            progress(si + 1, len(segs), seg["label"])
    enc.stdin.close()
    if enc.wait() != 0:
        raise AssemblyError("ffmpeg failed while writing the picture")
    info["frames"] = written
    info["captions"] = caps.report
    return info


def plan_lines(plan: dict, seg_id: str) -> list[dict]:
    for s in plan["segments"]:
        if s["id"] == seg_id:
            return s["lines"]
    return []


def _ffmpeg(args: list[str], timeout: float) -> None:
    """Run ffmpeg with no console input and a hard time limit, so a stuck encode fails loudly."""
    try:
        r = subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", *args],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssemblyError(f"ffmpeg did not finish within {int(timeout)}s: {' '.join(args[:6])}")
    if r.returncode != 0:
        raise AssemblyError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")


def encode_final(picture: Path, mix: Path, out: Path, preset: str = "medium", crf: int = 17) -> None:
    """Final H.264 + AAC file, in three separate steps.

    Audio and video are never filtered in the same ffmpeg command: some recent
    Windows builds loop forever when two inputs, -shortest and the loudness
    filter meet (seen on ffmpeg N-128293, 3 November 2026). Captions are already
    drawn into the picture."""
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                "default=nw=1:nk=1", str(picture)], capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=60).stdout.strip() or 0)
    audio = out.with_name(out.stem + "_audio.m4a")
    video = out.with_name(out.stem + "_video.mp4")
    af = loudnorm_args(mix)
    _ffmpeg(["-i", str(mix), "-t", f"{dur:.3f}", "-af", f"{af},aresample=48000", "-c:a", "aac", "-b:a", "192k",
             str(audio)], timeout=300)
    _ffmpeg(["-i", str(picture), "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-profile:v", "high", str(video)], timeout=max(600, dur * 60))
    _ffmpeg(["-i", str(video), "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c", "copy", "-t", f"{dur:.3f}",
             "-movflags", "+faststart", str(out)], timeout=300)
    audio.unlink(missing_ok=True)
    video.unlink(missing_ok=True)
