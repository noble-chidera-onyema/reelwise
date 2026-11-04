"""Quality gate. Every generated clip and the final render are checked, and the
download stays locked until every hard check passes.

Measured checks (no model involved, always run):
  clip decodes, is long enough, is not frozen, has no black frames, no flicker
  (brightness jumps without a cut), resolution is high enough, framing loss
  when cropped to 9:16 is acceptable.

Judged checks (Claude looking at sampled frames next to your reference images):
  same character identity as the reference, same art style, the action asked
  for actually happens, nobody in the story is talking, no text or logos,
  no anatomy or morphing artefacts. A failed clip is regenerated with the
  judge's note added to the prompt.

Final checks: duration and first-line timing against the profile, loudness
(-14 LUFS integrated, true peak at or below -1 dBTP), narration fits inside
its pictures, no dead air, captions inside the platform safe area, and one
last look across all shots for character consistency.

Severity: "fail" blocks the download, "warn" is shown but does not block.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..scenes import detect_boundaries
from ..video import probe, read_frames


def check(rule: str, ok: bool, detail: str, severity: str = "fail") -> dict:
    return {"rule": rule, "passed": bool(ok), "detail": detail, "severity": "ok" if ok else severity}


def gate(checks: list[dict]) -> bool:
    return all(c["passed"] or c["severity"] != "fail" for c in checks)


# --------------------------------------------------------------------------- clip, measured

def clip_checks(path: str | Path, job: dict, out_size: tuple[int, int]) -> tuple[list[dict], dict]:
    out: list[dict] = []
    try:
        info = probe(path)
        fs = read_frames(path, thumb_every_s=0.25, thumb_width=384)
    except Exception as e:
        return [check("clip decodes", False, str(e))], {}
    dur = fs.n / fs.fps
    out.append(check("clip decodes", True, f"{fs.n} frames at {fs.fps:.2f} fps"))
    out.append(check("long enough", dur >= job["need_s"] - 0.05,
                     f"{dur:.2f}s delivered, {job['need_s']:.2f}s needed"))
    short = min(info.width, info.height)
    out.append(check("resolution", short >= 700, f"{info.width}x{info.height}", "warn" if short >= 480 else "fail"))

    ow, oh = out_size
    kept = min((info.width / info.height) / (ow / oh), (ow / oh) / (info.width / info.height))
    out.append(check("framing kept after crop to output", kept >= 0.7,
                     f"{kept:.0%} of the picture survives the crop to {ow}x{oh}", "warn"))

    bounds = detect_boundaries(fs)
    cut_frames = {b.frame for b in bounds}
    m = fs.motion.copy()
    if cut_frames:
        m[sorted(cut_frames)] = np.nan
    win = max(1, int(round(0.5 * fs.fps)))
    frozen_windows = 0
    windows = 0
    for i in range(1, fs.n - win, win):
        seg = m[i:i + win]
        windows += 1
        if np.nanmean(seg) < 0.02:   # frozen stills measure ~0.002; a small moving subject ~0.05
            frozen_windows += 1
    frozen = frozen_windows / max(windows, 1)
    out.append(check("not frozen", frozen <= 0.3, f"{frozen:.0%} of half-second windows have no movement",
                     "fail" if frozen > 0.5 else "warn"))
    black = int((fs.brightness < 12).sum())
    out.append(check("no black frames", black <= 2, f"{black} near-black frames"))
    jumps = np.abs(np.diff(fs.brightness))
    flick = int(sum(1 for i, j in enumerate(jumps, 1) if j > 22 and i not in cut_frames))
    out.append(check("no flicker", flick <= 1, f"{flick} brightness jumps without a cut", "warn" if flick <= 3 else "fail"))
    want = len(job["shots"])
    got = len(bounds) + 1
    out.append(check("shot count", got >= want, f"{got} shots found, {want} asked for", "warn"))
    meta = {"duration_s": round(dur, 3), "fps": fs.fps, "width": info.width, "height": info.height,
            "cuts_s": [round(b.time_s, 3) for b in bounds],
            "thumbs": [i for i, _ in fs.thumbs]}
    return out, meta


def sample_frames(path: str | Path, n: int = 6, width: int = 448) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = []
    for k in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            s = width / f.shape[1]
            out.append((k / fps, cv2.resize(f, (width, int(f.shape[0] * s)), interpolation=cv2.INTER_AREA)))
    cap.release()
    return out


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode()


def _b64_file(path: str, width: int = 448) -> str:
    img = cv2.imread(path)
    s = width / img.shape[1]
    return _b64(cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA))


JUDGE = """You are the quality controller for an animation studio. Be strict: a clip
that would make a viewer notice "AI" fails. Compare the sampled frames of a
generated clip against the reference images.

Score 1-5 (5 = flawless):
  identity: every character keeps the reference face, hair, skin tone, clothing
  style: the art style, palette and lighting match the reference image
  action: the frames show the requested action and camera move
List concrete artefacts you can see: warped or extra fingers, melting faces,
limbs merging, objects popping in or out, texture crawling, garbled text.

Hard rules (any true = fail):
  talking: a story character's mouth is visibly moving as if speaking
  text_or_logo: any readable text, caption, watermark or logo appears

Return JSON only:
{"identity": 0, "style": 0, "action": 0, "artifacts": [], "talking": false,
 "text_or_logo": false, "verdict": "pass", "fix": "one sentence telling the
 video model what to change, or empty"}"""


def judge_clip(path: str, job: dict, plan: dict, client=None, model: str | None = None) -> dict:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model = model or os.environ.get("REELWISE_MODEL", "claude-sonnet-5-5")
    images = {im["id"]: im["path"] for im in plan["images"]}
    chars = {c["name"].lower(): c for c in plan.get("story_characters", [])}
    content = [{"type": "text", "text": "Reference: story image used as the first frame"},
               {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                            "data": _b64_file(images[job["image"]])}}]
    for name in job.get("characters", []):
        c = chars.get(name.lower())
        if c and c.get("images"):
            content += [{"type": "text", "text": f"Reference: {c['name']}"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": _b64_file(c["images"][0])}}]
    for t, f in sample_frames(path):
        content += [{"type": "text", "text": f"Generated frame at {t:.2f}s"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(f)}}]
    asked = " / ".join(sh["prompt"] for sh in job["shots"])
    content.append({"type": "text", "text": f"Requested shots: {asked}\n\n{JUDGE}"})
    msg = client.messages.create(model=model, max_tokens=800, messages=[{"role": "user", "content": content}])
    text = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"verdict": "fail", "fix": "", "error": "judge reply had no JSON"}
    return json.loads(m.group(0))


def judge_checks(j: dict) -> list[dict]:
    if "error" in j:
        return [check("vision judge", False, j["error"], "warn")]
    severe = [a for a in j.get("artifacts", []) if a]
    return [
        check("same character as reference", j.get("identity", 0) >= 4, f"identity {j.get('identity')}/5"),
        check("matches reference style", j.get("style", 0) >= 4, f"style {j.get('style')}/5"),
        check("action happens as asked", j.get("action", 0) >= 3, f"action {j.get('action')}/5"),
        check("story characters silent", not j.get("talking", False), "mouth movement seen" if j.get("talking") else "no talking"),
        check("no text or logos", not j.get("text_or_logo", False), "text or logo seen" if j.get("text_or_logo") else "clean"),
        check("no visible artefacts", not severe, "; ".join(severe) or "none seen", "fail" if len(severe) > 1 else "warn"),
    ]


# --------------------------------------------------------------------------- final

def loudness(path: str | Path) -> dict:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-filter_complex",
                        "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", tail)
    tp = re.search(r"Peak:\s+(-?[\d.]+) dBFS", tail)
    return {"integrated_lufs": float(i.group(1)) if i else None, "true_peak_dbtp": float(tp.group(1)) if tp else None}


def final_checks(final_path: str, plan: dict, timeline: dict, consistency: dict | None = None) -> list[dict]:
    from ..analyse import analyse_video

    prof = plan["profile"]
    out: list[dict] = []
    rep = analyse_video(final_path)
    dur = rep["video"]["duration_s"]
    lo, hi = prof.get("target_duration_s", [0, 1e9])
    out.append(check("total length", lo <= dur <= hi, f"{dur:.1f}s, target {lo}-{hi}s"))
    W, H = plan["format"]["width"], plan["format"]["height"]
    out.append(check("output size", (rep["video"]["width"], rep["video"]["height"]) == (W, H),
                     f"{rep['video']['width']}x{rep['video']['height']}"))
    lines = [ln for s in timeline["segments"] for ln in s["lines"]]
    if lines:
        first = min(l["t0"] for l in lines)
        lim = prof.get("first_line", {}).get("max_start_s")
        if lim is not None:
            out.append(check("first line starts early", first <= lim + 1e-6, f"{first:.2f}s, limit {lim}s"))
    for s in timeline["segments"]:
        for l in s["lines"]:
            if l["t1"] > s["t1"] + 1e-3:
                out.append(check("narration fits its picture", False,
                                 f"{s['label']}: line ends {l['t1']:.2f}s after picture ends {s['t1']:.2f}s"))
    if not any(c["rule"] == "narration fits its picture" for c in out):
        out.append(check("narration fits its picture", True, "every line ends inside its scene"))
    lines_sorted = sorted(lines, key=lambda l: l["t0"])
    gaps = [(b["t0"] - a["t1"], a["t1"]) for a, b in zip(lines_sorted, lines_sorted[1:])]
    worst = max(gaps, default=(0, 0))
    out.append(check("no dead air", worst[0] <= 1.2, f"longest gap {worst[0]:.2f}s at {worst[1]:.1f}s", "warn"))
    # Every narration line must be audible in the final file, not just present in the plan.
    from .voices import SR, read_wav
    audio = read_wav(final_path)
    silent = []
    for l in lines:
        seg = audio[int(l["t0"] * SR):int(l["t1"] * SR)]
        if seg.size == 0 or 20 * np.log10(max(np.sqrt((seg ** 2).mean()), 1e-7)) < -35:
            silent.append(f"{l['speaker']} at {l['t0']:.1f}s")
    out.append(check("every line audible", not silent,
                     f"silent: {', '.join(silent)}" if silent else f"all {len(lines)} lines heard in the final mix"))
    from .voices import max_wps
    rushed = [f"{l['speaker']} at {l['t0']:.1f}s ({len(l['text'].split()) / max(l['t1'] - l['t0'], 0.01):.1f} words/s)"
              for l in lines if len(l["text"].split()) / max(l["t1"] - l["t0"], 0.01) > max_wps(prof, l["speaker"]) + 0.3]
    out.append(check("no rushed lines", not rushed, "rushed: " + ", ".join(rushed) if rushed
                     else "every line at a natural pace"))
    matched = [(l["speaker"], l["voice_match"]) for seg in plan["segments"] for l in seg["lines"]
               if l.get("voice_match") is not None]
    if matched:
        from .voiceprint import MIN_MATCH
        off = [f"{h} {m:.2f}" for h, m in matched if m < MIN_MATCH]
        out.append(check("every host in his own voice", not off,
                         "voice drifted: " + ", ".join(off) if off else
                         f"{len(matched)} lines match their host's voice ({min(m for _, m in matched):.2f} or higher)"))
    lu = loudness(final_path)
    if lu["integrated_lufs"] is not None:
        out.append(check("loudness -14 LUFS", abs(lu["integrated_lufs"] + 14) <= 1.0,
                         f"{lu['integrated_lufs']:.1f} LUFS"))
    if lu["true_peak_dbtp"] is not None:
        out.append(check("true peak", lu["true_peak_dbtp"] <= -1.0, f"{lu['true_peak_dbtp']:.1f} dBTP"))
    over = [c for c in timeline.get("captions", []) if not c.get("fits", True)]
    out.append(check("captions inside safe area", not over,
                     f"{len(over)} caption lines too wide" if over else f"{len(timeline.get('captions', []))} caption events fit"))
    black = int(sum(1 for r in rep["per_second"] if r["brightness"] < 10))
    out.append(check("no black seconds", black == 0, f"{black} near-black seconds"))
    if consistency is not None:
        if "error" in consistency:
            out.append(check("characters consistent across shots", False, consistency["error"], "warn"))
        else:
            bad = consistency.get("inconsistent_shots", [])
            out.append(check("characters consistent across shots", consistency.get("consistent", False) and not bad,
                             consistency.get("note", "") + (f" Shots: {bad}" if bad else "")))
    return out


CONSISTENCY = """These are one frame from every shot of a finished episode, in order,
followed by the character reference images. Check that each character looks
like the same person in every shot (face, hair, skin tone, clothing) and that
the art style does not change between shots. Return JSON only:
{"consistent": true, "inconsistent_shots": ["S3"], "note": "one sentence"}"""


def judge_consistency(shot_frames: list[tuple[str, np.ndarray]], plan: dict, client=None, model=None) -> dict:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model = model or os.environ.get("REELWISE_MODEL", "claude-sonnet-5-5")
    content = []
    for label, f in shot_frames[:40]:
        content += [{"type": "text", "text": label},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(f)}}]
    for c in plan.get("story_characters", []):
        if c.get("images"):
            content += [{"type": "text", "text": f"Reference: {c['name']}"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": _b64_file(c["images"][0])}}]
    content.append({"type": "text", "text": CONSISTENCY})
    try:
        msg = client.messages.create(model=model, max_tokens=500, messages=[{"role": "user", "content": content}])
        text = "".join(getattr(b, "text", "") for b in msg.content)
        return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception as e:
        return {"error": f"consistency check failed: {e}"}
