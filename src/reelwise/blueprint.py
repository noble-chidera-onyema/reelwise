"""From a breakdown to a reusable format, and from a format to your own episode.

Three steps, kept separate so each can be checked:

1. build_format(report)  deterministic. Turns one analysed video into a
   timing template: where the hook ends, how long each beat runs as a share
   of the video, how fast it cuts, how fast people talk. No model involved.

2. write_episode(format, profile, story)  uses Claude. Writes a script and
   shot list for YOUR characters and YOUR story, fitted to that template.
   The model is told to use only the facts in the story text you give it.

3. check_script(script, profile)  deterministic again. Estimates the timing
   of every line from each character's speaking rate and checks the result
   against the rules in your profile (length, first line, who may speak how
   often, spoken digits). A script that fails is reported, not hidden.

The template copies structure and timing. It does not copy anyone's words,
footage, characters or music, and nothing here should be used to.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

DEFAULT_MODEL = os.environ.get("REELWISE_MODEL", "claude-sonnet-5-5")


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _energy_labels(values: list[float]) -> list[str]:
    if not values:
        return []
    lo, hi = np.percentile(values, [33, 67])
    return ["high" if v > hi else "low" if v < lo else "medium" for v in values]


def build_format(report: dict) -> dict:
    dur = report["video"]["duration_s"]
    scenes = report["scenes"]
    energy = _energy_labels([s["motion"] for s in scenes])
    vis = {s.get("scene"): s for s in (report.get("vision") or {}).get("scenes", []) if isinstance(s, dict)}
    lines = report.get("transcript") or []

    beats = []
    for s, e in zip(scenes, energy):
        spoken = [l["text"] for l in lines if s["start_s"] <= l["start"] < s["end_s"]]
        v = vis.get(s["index"], {})
        role = "hook" if s["index"] == 0 else ("close" if s["index"] == len(scenes) - 1 else "body")
        beats.append({
            "beat": s["index"],
            "role": role,
            "start_pct": round(100 * s["start_s"] / dur, 1),
            "end_pct": round(100 * s["end_s"] / dur, 1),
            "duration_s": s["duration_s"],
            "transition_in": s["transition_in"],
            "energy": e,
            "shot_type": v.get("shot_type"),
            "on_screen_text_style": v.get("caption_style") or None,
            "lines_in_reference": len(spoken),
            "words_in_reference": sum(len(t.split()) for t in spoken),
        })

    speech = report.get("speech") or {}
    hook = report["hook"]
    return {
        "derived_from": report["source"]["file"],
        "orientation": report["video"]["orientation"],
        "reference_duration_s": dur,
        "target_duration_range_s": [round(dur * 0.85, 1), round(dur * 1.15, 1)],
        "pacing": {k: report["pacing"][k] for k in
                   ("cuts_per_minute", "median_shot_s", "shot_length_cv", "gradual_transitions")},
        "hook": {
            "first_cut_s": hook["first_cut_s"],
            "first_word_s": hook["first_word_s"],
            "opens_with_question": hook["first_line_is_question"],
            "motion_hook_vs_rest": hook["motion_hook_vs_rest"],
            "opening_frame": (report.get("vision") or {}).get("opening_frame"),
        },
        "speech": {k: speech.get(k) for k in ("words_per_second", "questions", "n_lines", "n_words")},
        "format_summary": (report.get("vision") or {}).get("format_summary"),
        "beats": beats,
    }


SYSTEM = """You write scripts and shot lists for short vertical videos with recurring characters.
You follow the creator's profile rules exactly. You never invent facts: every
factual claim must come from the story text supplied. If the story text does
not support a beat, write a line that stays true (a reaction, a question)
instead of adding detail. Output JSON only."""


def _episode_prompt(fmt: dict, profile: dict, story: str) -> str:
    return f"""FORMAT TEMPLATE (timing to follow, derived from a reference video):
{json.dumps(fmt, indent=1)}

CREATOR PROFILE (characters and rules; these override the template where they conflict):
{json.dumps(profile, indent=1)}

STORY (the only source of facts):
{story}

Write one episode. Return JSON in exactly this shape:
{{
  "title": "",
  "lines": [{{"speaker": "", "text": "", "beat": 0, "delivery": ""}}],
  "shots": [{{"beat": 0, "shot": "", "who": "", "action": "", "on_screen_text": ""}}],
  "cover_image_prompt": "",
  "facts_used": ["each factual claim, copied from the story"]
}}"""


def write_episode(fmt: dict, profile: dict, story: str, model: str | None = None, client=None) -> dict:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": _episode_prompt(fmt, profile, story)}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("Model reply contained no JSON object")
    return json.loads(m.group(0))


_DIGIT = re.compile(r"\d")


def estimate_timeline(script: dict, profile: dict) -> list[dict]:
    chars = {c["name"]: c for c in profile.get("characters", [])}
    gap_lo, gap_hi = profile.get("gap_between_lines_s", [0.15, 0.25])
    gap = (gap_lo + gap_hi) / 2
    t = profile.get("first_line", {}).get("start_s", 0.3)
    rows = []
    for line in script.get("lines", []):
        wps = chars.get(line.get("speaker"), {}).get("words_per_second", 3.0)
        n = len(line.get("text", "").split())
        d = n / wps if wps else 0
        rows.append({"speaker": line.get("speaker"), "start_s": round(t, 2), "end_s": round(t + d, 2),
                     "words": n, "text": line.get("text", "")})
        t += d + gap
    return rows


def check_script(script: dict, profile: dict) -> dict:
    timeline = estimate_timeline(script, profile)
    lines = script.get("lines", [])
    checks = []

    def add(rule, ok, detail):
        checks.append({"rule": rule, "passed": bool(ok), "detail": detail})

    end_card = profile.get("end_card_s", 0)
    total = (timeline[-1]["end_s"] if timeline else 0) + end_card
    lo, hi = profile.get("target_duration_s", [0, 1e9])
    add("total duration", lo <= total <= hi, f"estimated {total:.1f}s including {end_card}s end card; target {lo}-{hi}s")

    fl = profile.get("first_line", {})
    if lines:
        first = lines[0]
        if fl.get("speaker"):
            add("first speaker", first.get("speaker") == fl["speaker"],
                f"first line is {first.get('speaker')}, profile wants {fl['speaker']}")
        if fl.get("must_be_question"):
            add("first line is a question", first.get("text", "").strip().endswith("?"), first.get("text", ""))

    counts = {}
    for l in lines:
        counts[l.get("speaker")] = counts.get(l.get("speaker"), 0) + 1
    for c in profile.get("characters", []):
        if "max_lines" in c:
            n = counts.get(c["name"], 0)
            add(f"{c['name']} line limit", n <= c["max_lines"], f"{n} line{'' if n == 1 else 's'}, limit {c['max_lines']}")
        if "max_words_per_line" in c:
            over = [l["text"] for l in lines if l.get("speaker") == c["name"]
                    and len(l["text"].split()) > c["max_words_per_line"]]
            add(f"{c['name']} line length", not over,
                f"{len(over)} lines over {c['max_words_per_line']} words")

    lead = profile.get("lead_narrator")
    if lead and lines:
        n_words = sum(len(l.get("text", "").split()) for l in lines)
        mine = sum(len(l.get("text", "").split()) for l in lines if l.get("speaker") == lead)
        share = mine / n_words if n_words else 0
        need = profile.get("lead_share_min", 0.7)
        add("lead narrator carries the story", share >= need,
            f"{lead} speaks {share:.0%} of the words; target at least {need:.0%}")

    so = profile.get("sign_off")
    if so and lines:
        last = lines[-1]
        norm = lambda t: "".join(ch for ch in t.lower().replace("\u2019", "'") if ch.isalnum() or ch == "'")
        ok = last.get("speaker") == so["speaker"] and norm(last.get("text", "")).endswith(norm(so["text"]))
        add("sign-off line", ok, f'last line is {last.get("speaker")}: "{last.get("text", "")}"; '
                                 f'profile wants {so["speaker"]}: "{so["text"]}"')

    known = {c["name"] for c in profile.get("characters", [])}
    unknown = sorted({l.get("speaker") for l in lines} - known)
    add("only profile characters speak", not unknown, ", ".join(unknown) or "all speakers are in the profile")

    min_words = profile.get("min_words_per_line", 0)
    if min_words:
        terse = {c["name"] for c in profile.get("characters", []) if c.get("max_words_per_line", 99) <= 8}
        short = [l["text"] for l in lines if len(l["text"].split()) < min_words and l.get("speaker") not in terse]
        add("no choppy lines", not short, f"{len(short)} lines under {min_words} words: {short[:3]}")

    if profile.get("spell_out_numbers"):
        digits = [l["text"] for l in lines if _DIGIT.search(l["text"])]
        add("numbers written as words for the voice", not digits, f"{len(digits)} lines contain digits: {digits[:3]}")

    return {"passed": all(c["passed"] for c in checks), "estimated_duration_s": round(total, 1),
            "checks": checks, "timeline": timeline}


def check_video(report: dict, profile: dict) -> dict:
    """Check a finished render against the profile using its measured breakdown."""
    checks = []
    dur = report["video"]["duration_s"]
    lo, hi = profile.get("target_duration_s", [0, 1e9])
    checks.append({"rule": "duration", "passed": lo <= dur <= hi, "detail": f"{dur:.1f}s; target {lo}-{hi}s"})
    max_start = profile.get("first_line", {}).get("max_start_s")
    if max_start is not None:
        t = report["hook"].get("first_word_s")
        src = "captions"
        if t is None:
            t = report["hook"].get("first_sound_s")
            src = "first audible sound"
        if t is not None:
            checks.append({"rule": "first line starts early", "passed": t <= max_start,
                           "detail": f"{t:.2f}s by {src}; limit {max_start}s"})
    if report["video"]["orientation"] != profile.get("orientation", report["video"]["orientation"]):
        checks.append({"rule": "orientation", "passed": False,
                       "detail": f"{report['video']['orientation']}, profile wants {profile['orientation']}"})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
