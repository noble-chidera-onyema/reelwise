"""Cut, composite, caption, mix and export the episode.

Timeline, in order of the plan's segments:
  COLD OPEN  the most dramatic story shot, full frame, with the yellow stamp
  INTRO      the hosts on the couch; the camera eases toward the TV and, under
             the last words, pushes through the screen into scene 1
  SCENES     story shots, each cut to the narration over it
  OUTRO      the camera pulls back out of the TV to the hosts for the sign-off
  END CARD   2 s follow card over the last frame, blurred

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

HEAD_TRIM_S = 0.2       # generated clips often start with a few near-still frames
MAX_STRETCH = 1.2       # slowest we slow footage down before asking for a longer clip


class AssemblyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- timing

def build_timeline(plan: dict, zoom_s: float, end_card_s: float) -> dict:
    fps = plan["format"]["fps"]
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
        segs.append({"id": s["id"], "kind": s["kind"], "label": s["label"], "t0": round(f0 / fps, 3),
                     "t1": round(f1 / fps, 3), "f0": f0, "f1": f1, "lines": lines, "stamp": s.get("stamp", "")})
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
            if seg["kind"] == "intro":
                a = int((seg["t1"] - z - 0.1) * SR)
                w = whoosh(z + 0.25, True)
                fx[max(0, a):max(0, a) + len(w)] += w[: n - max(0, a)]
            if seg["kind"] == "outro":
                a = int(max(0.0, seg["t0"] - 0.15) * SR)
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


def loudnorm_args(wav: Path, target: float = -14.0, tp: float = -1.5, lra: float = 11.0) -> str:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af",
                        f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        return f"loudnorm=I={target}:TP={tp}:LRA={lra}"
    d = json.loads(m.group(0))
    return (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={d['input_i']}:measured_TP={d['input_tp']}:"
            f"measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}:"
            f"linear=true")


# --------------------------------------------------------------------------- picture

def render_picture(plan: dict, timeline: dict, shot_files: dict[str, list[str]], host_cfg: dict | None,
                   out: Path, progress=None) -> dict:
    W, H, fps = timeline["W"], timeline["H"], timeline["fps"]
    z_frames = int(round(timeline["zoom_s"] * fps))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "12", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    need_host = timeline["has_intro"] or timeline["has_outro"]
    rig = tvc = None
    info = {}
    if need_host:
        if not host_cfg:
            raise AssemblyError("INTRO/OUTRO need a host scene. Upload one, or remove those sections.")
        canvas = (W, H)
        if host_cfg.get("hosts"):
            rig = HostRig(host_cfg, canvas)
            plate_frame = rig.plate.frame(0)
        else:
            from .host import Plate
            rig = None
            plate = Plate(host_cfg["background"], canvas)
            plate_frame = plate.frame(0)
        screen = detect_screen(plate_frame, host_cfg.get("screen_corners"))
        if screen is None:
            raise AssemblyError("No TV screen found in the host scene. Make the screen flat green (#00FF00) "
                                "or enter its four corners.")
        tvc = TVCompositor(canvas, screen, (W, H))
        info["screen"] = screen.to_dict()

    def host_bg(i):
        return rig.plate.frame(i) if rig else plate.frame(i)

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
                enc.stdin.write(f.tobytes())
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
                ov = rig.overlay(i) if rig else None
                f = tvc.render(host_bg(k_global + i), story, min(max(u, 0.0), 1.0), overlay=ov)
                enc.stdin.write(f.tobytes())
                last_frame = f
            written += n
        elif seg["kind"] == "outro":
            tail = story_tail_frames or [last_frame]
            if rig:
                rig.drive(plan_lines(plan, seg["id"]), 0.0, n, fps, seed=13)
            for i in range(n):
                if i < z_frames:
                    u = 1.0 - 0.94 * ((i + 1) / z_frames)
                else:
                    u = 0.06 * (1 - (i - z_frames) / max(n - z_frames - 1, 1))
                story = tail[min(i, len(tail) - 1)]
                ov = rig.overlay(i) if rig else None
                f = tvc.render(host_bg(k_global + i), story, max(u, 0.0), overlay=ov)
                enc.stdin.write(f.tobytes())
                last_frame = f
            written += n
        elif seg["kind"] == "end_card":
            base = last_frame if last_frame is not None else np.zeros((H, W, 3), np.uint8)
            card = cv2.GaussianBlur(base, (0, 0), 18)
            card = (card.astype(np.float32) * 0.45).astype(np.uint8)
            for i in range(n):
                a = min(1.0, (i + 1) / 6)
                f = (base.astype(np.float32) * (1 - a) + card.astype(np.float32) * a).astype(np.uint8)
                enc.stdin.write(f.tobytes())
            written += n
        k_global += n
        if progress:
            progress(si + 1, len(segs), seg["label"])
    enc.stdin.close()
    if enc.wait() != 0:
        raise AssemblyError("ffmpeg failed while writing the picture")
    info["frames"] = written
    return info


def plan_lines(plan: dict, seg_id: str) -> list[dict]:
    for s in plan["segments"]:
        if s["id"] == seg_id:
            return s["lines"]
    return []


def encode_final(picture: Path, mix: Path, ass: Path, out: Path, fonts_dir: Path | None,
                 preset: str = "medium", crf: int = 17) -> None:
    ass_arg = str(ass).replace("\\", "/").replace(":", "\\:")
    vf = f"ass='{ass_arg}'"
    if fonts_dir and fonts_dir.exists():
        fd = str(fonts_dir).replace("\\", "/").replace(":", "\\:")
        vf = f"ass='{ass_arg}':fontsdir='{fd}'"
    af = loudnorm_args(mix)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(picture), "-i", str(mix),
                    "-map", "0:v", "-map", "1:a", "-vf", vf, "-af", f"{af},aresample=48000",
                    "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-profile:v", "high", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", str(out)], check=True)
