"""The episode plan: what is on screen, who says what, and for how long.

The plan is a plain dict saved as plan.json in the project folder, so it can
be read, edited in the Studio page, and re-run without paying twice.

Scene text format (what you type into the Studio page):

    COLD OPEN
    Image: 3
    Stamp: 738 DAYS
    Bex: Would you live up a tree for seven hundred and thirty eight days?

    INTRO
    Ren: This is a true story from California.

    SCENE 1
    Image: 1
    Characters: Julia
    Action: A young woman climbs a giant redwood at night, rope over her shoulder.
    Shots:
    - extreme close-up of her hands gripping the rope
    - wide shot from below, the tree vanishing into fog
    Ren: In nineteen ninety seven, Julia Hill climbed a redwood called Luna.
    Bex: And she did not come back down.

    OUTRO
    Ren: That is one worth telling.

Rules:
  - Lines are "Name: text", where Name is one of your hosts. Hosts narrate;
    story characters never speak.
  - "Image:" picks an uploaded image by number. Leave it out and the planner
    picks one.
  - "Shots:" is optional. Leave it out and the planner writes the shots in the
    reference video's style.
  - "Say:" under a line (or "Name: text | spoken text") gives the voice a
    phonetic spelling while the caption keeps the real one.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path

SPECIAL = {"COLD OPEN": "cold_open", "INTRO": "intro", "OUTRO": "outro"}
_HEADER = re.compile(r"^\s*(COLD OPEN|INTRO|OUTRO|SCENE\s*(\d+)?)\s*:?\s*$", re.I)
_FIELD = re.compile(r"^\s*(Image|Images|Characters|Character|Action|Stamp|Shots|Camera)\s*:\s*(.*)$", re.I)
_LINE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,30}?)\s*:\s*(.+)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.+)$")

# Shot grammar measured from the reference studied for this project (37 shots,
# median 0.83 s). Used when no Claude key is set, and as the default menu the
# planner chooses from.
SHOT_GRAMMAR = [
    "extreme close-up detail, shallow depth of field",
    "medium shot, slow push-in",
    "low-angle wide shot, subject walks through frame",
    "close-up on the face, eyes moving, mouth closed",
    "tracking shot alongside the subject",
    "high top-down wide shot",
    "insert shot of hands doing the action",
    "wide establishing shot, camera drifts sideways",
]


COLD_OPEN_SHOT = "slow dramatic push-in on the whole scene, the most striking moment, full figure in frame"


class PlanError(ValueError):
    pass


def parse_scene_text(text: str, host_names: list[str]) -> list[dict]:
    """Turn the scene text box into a list of segments. Raises PlanError with the line number."""
    hosts = {h.lower(): h for h in host_names}
    segments: list[dict] = []
    cur: dict | None = None
    field = None
    scene_no = 0

    def new(kind: str, label: str) -> dict:
        return {"kind": kind, "label": label, "image": None, "characters": [], "action": "",
                "stamp": "", "shots": [], "lines": []}

    for n, raw in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _HEADER.match(line)
        if m:
            head = m.group(1).upper()
            if head.startswith("SCENE"):
                scene_no += 1
                cur = new("story", f"Scene {m.group(2) or scene_no}")
            else:
                cur = new(SPECIAL[re.sub(r"\s+", " ", head)], head.title())
            segments.append(cur)
            field = None
            continue
        if cur is None:
            raise PlanError(f"line {n}: text before the first SCENE / INTRO / COLD OPEN header: {line.strip()!r}")
        b = _BULLET.match(line)
        if b and field == "shots":
            cur["shots"].append({"prompt": b.group(1).strip()})
            continue
        m = _FIELD.match(line)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key in ("image", "images"):
                nums = re.findall(r"\d+", val)
                if not nums:
                    raise PlanError(f"line {n}: Image: needs a number, got {val!r}")
                cur["image"] = int(nums[0])
                field = None
            elif key in ("characters", "character"):
                cur["characters"] = [c.strip() for c in re.split(r"[,&]| and ", val) if c.strip()]
                field = None
            elif key == "stamp":
                cur["stamp"] = val
                field = None
            elif key in ("shots", "camera"):
                field = "shots"
                if val:
                    cur["shots"].append({"prompt": val})
            else:  # action, may continue on following lines
                cur["action"] = val
                field = "action"
            continue
        m = _LINE.match(line)
        if m and m.group(1).strip().lower() in hosts:
            caption, _, spoken = m.group(2).partition("|")
            cur["lines"].append({"speaker": hosts[m.group(1).strip().lower()],
                                 "text": caption.strip(), "say": spoken.strip() or caption.strip()})
            field = None
            continue
        if m and m.group(1).strip().lower() == "say" and cur["lines"]:
            cur["lines"][-1]["say"] = m.group(2).strip()
            continue
        if m and m.group(1).strip().lower() not in hosts and len(m.group(1).split()) <= 2 \
                and m.group(1).strip()[:1].isupper():
            raise PlanError(f"line {n}: {m.group(1).strip()!r} is not one of your hosts "
                            f"({', '.join(host_names)}). Story characters do not speak.")
        if field == "action":
            cur["action"] = (cur["action"] + " " + line.strip()).strip()
            continue
        raise PlanError(f"line {n}: could not read {line.strip()!r}")

    story = [s for s in segments if s["kind"] == "story"]
    if not story:
        raise PlanError("no SCENE sections found; the story needs at least one")
    for s in story:
        if not s["lines"]:
            raise PlanError(f"{s['label']} has no narration lines")
        if not s["action"] and not s["shots"]:
            raise PlanError(f"{s['label']} needs an Action: line or a Shots: list")
    kinds = [s["kind"] for s in segments]
    for k in ("cold_open", "intro", "outro"):
        if kinds.count(k) > 1:
            raise PlanError(f"only one {k.replace('_', ' ').upper()} section is allowed")
    return segments


def new_plan(segments: list[dict], profile: dict, images: list[dict], story_characters: list[dict],
             reference: dict | None = None, title: str = "") -> dict:
    """Assemble a plan from parsed segments. Images are [{'id': 1, 'path': ...}]."""
    ids = {im["id"] for im in images}
    for s in segments:
        if s["image"] is not None and s["image"] not in ids:
            raise PlanError(f"{s['label']}: Image {s['image']} was not uploaded (have {sorted(ids)})")
    known = {c["name"].lower() for c in story_characters}
    for s in segments:
        missing = [c for c in s["characters"] if c.lower() not in known]
        if missing:
            raise PlanError(f"{s['label']}: no reference images for {', '.join(missing)}. "
                            "Add them under Story characters, or remove them from Characters:.")
    w, h = (1080, 1920) if profile.get("orientation", "vertical") == "vertical" else (1920, 1080)
    plan = {
        "title": title,
        "format": {"width": w, "height": h, "fps": 30},
        "profile": profile,
        "images": images,
        "story_characters": story_characters,
        "reference": _reference_summary(reference) if reference else None,
        "segments": deepcopy(segments),
    }
    for i, s in enumerate(plan["segments"]):
        s["id"] = f"seg{i:02d}"
        for j, ln in enumerate(s["lines"]):
            ln["id"] = f"{s['id']}_l{j}"
    return plan


def _reference_summary(r: dict) -> dict:
    p = r.get("pacing", {})
    return {
        "file": r.get("source", {}).get("file"),
        "duration_s": r.get("video", {}).get("duration_s"),
        "median_shot_s": p.get("median_shot_s"),
        "cuts_per_minute": p.get("cuts_per_minute"),
        "shot_lengths": [s["duration_s"] for s in r.get("scenes", [])],
        "shot_types": [s.get("shot_type") for s in (r.get("vision") or {}).get("scenes", [])],
        "format_summary": (r.get("vision") or {}).get("format_summary"),
    }


# --------------------------------------------------------------------------- shots

def _target_shot_s(plan: dict) -> float:
    ref = plan.get("reference") or {}
    med = ref.get("median_shot_s") or 1.2
    # Generated clips need a little longer than a pure insert to read; floor at 0.9 s.
    return max(0.9, min(float(med) * 1.6, 2.5))


def fill_shots(plan: dict) -> dict:
    """Give every story segment a shot list, deterministically.

    Used directly when there is no Claude key, and as the starting point the
    Claude planner refines. Needs segment durations (set by voices.time_plan).
    """
    target = _target_shot_s(plan)
    k = 0
    for s in plan["segments"]:
        if s["kind"] not in ("story", "cold_open"):
            continue
        dur = s.get("duration_s") or 3.0
        if not s["shots"]:
            n = max(1, min(6, round(dur / target)))
            if s["kind"] == "cold_open":
                n = 1
            if s["kind"] == "cold_open":
                s["shots"] = [{"prompt": f"{COLD_OPEN_SHOT}. {s['action']}".strip()}]
            else:
                s["shots"] = [{"prompt": f"{SHOT_GRAMMAR[(k + i) % len(SHOT_GRAMMAR)]}. {s['action']}".strip()}
                              for i in range(n)]
            k += n
        _spread(s, dur)
    return plan


def _spread(seg: dict, dur: float) -> None:
    n = len(seg["shots"])
    weights = [float(sh.get("weight", 1.0)) for sh in seg["shots"]]
    total = sum(weights)
    t = 0.0
    for sh, wgt in zip(seg["shots"], weights):
        d = dur * wgt / total
        sh["start_s"] = round(t, 3)
        sh["duration_s"] = round(d, 3)
        t += d
    if n:
        seg["shots"][-1]["duration_s"] = round(dur - seg["shots"][-1]["start_s"], 3)


PLANNER_SYSTEM = """You are the director of short vertical true-story videos.
Your job is to turn each scene into a shot list for an image-to-video model.
Match the reference edit: fast, varied angles (extreme close-ups, inserts of
hands and objects, low wides, top-downs, tracking shots), never two similar
angles in a row. Each prompt describes ONE continuous camera move and the
physical action only. Characters never talk: mouths stay closed or still.
No on-screen text, no logos, no captions in the image. Keep every character's
clothing and features exactly as in the image. Output JSON only."""


def claude_shots(plan: dict, image_b64: dict[int, str], client=None, model: str | None = None) -> dict:
    """Ask Claude to write the shots for scenes that have none, using the images."""
    import os
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    model = model or os.environ.get("REELWISE_MODEL", "claude-sonnet-5-5")
    target = _target_shot_s(plan)
    todo = [s for s in plan["segments"] if s["kind"] in ("story", "cold_open") and not s.get("shots_locked")]
    content: list[dict] = []
    for im in plan["images"]:
        if im["id"] in image_b64:
            content.append({"type": "text", "text": f"Image {im['id']}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                         "data": image_b64[im["id"]]}})
    brief = [{"segment": s["id"], "label": s["label"], "image": s["image"], "characters": s["characters"],
              "action": s["action"], "duration_s": s.get("duration_s"),
              "narration": [f"{l['speaker']}: {l['text']}" for l in s["lines"]],
              "user_shots": [sh["prompt"] for sh in s["shots"]]} for s in todo]
    content.append({"type": "text", "text": f"""REFERENCE EDIT: {json.dumps(plan.get('reference'))}
Aim for shots of about {target:.1f} s; the cold open is one shot.

SCENES:
{json.dumps(brief, indent=1)}

For each scene: if "image" is null pick the best uploaded image number. If the
user gave shots, keep their intent and only sharpen the wording. Give each shot
a weight (relative length; the hero shot of a scene can be 2).
Return: {{"segments": [{{"segment": "seg01", "image": 1, "shots": [{{"prompt": "", "weight": 1}}]}}]}}"""})
    msg = client.messages.create(model=model, max_tokens=4000, system=PLANNER_SYSTEM,
                                 messages=[{"role": "user", "content": content}])
    text = "".join(getattr(b, "text", "") for b in msg.content)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise PlanError("planner reply had no JSON")
    out = {d["segment"]: d for d in json.loads(m.group(0)).get("segments", [])}
    ids = {im["id"] for im in plan["images"]}
    for s in todo:
        d = out.get(s["id"])
        if not d:
            continue
        if s["image"] is None and d.get("image") in ids:
            s["image"] = d["image"]
        shots = [{"prompt": str(x.get("prompt", "")).strip(), "weight": float(x.get("weight", 1) or 1)}
                 for x in d.get("shots", []) if str(x.get("prompt", "")).strip()]
        if shots:
            s["shots"] = shots
    return fill_shots(plan)


def assign_missing_images(plan: dict) -> dict:
    """Without a planner, scenes with no Image: take uploaded images in order."""
    ids = [im["id"] for im in plan["images"]]
    used = [s["image"] for s in plan["segments"] if s["image"] is not None]
    pool = [i for i in ids if i not in used] or ids
    k = 0
    for s in plan["segments"]:
        if s["kind"] in ("story", "cold_open") and s["image"] is None:
            if not pool:
                raise PlanError(f"{s['label']} has no image and none were uploaded")
            s["image"] = pool[k % len(pool)]
            k += 1
    return plan


# --------------------------------------------------------------------------- generation jobs

def generation_jobs(plan: dict, allowed_durations: list[int], handle_s: float = 0.6) -> list[dict]:
    """One image-to-video request per story segment, split when longer than the model allows.

    Each request asks for a little more footage than the edit needs (handle_s)
    so slow first frames can be trimmed. The first story scene also carries the
    footage that plays on the TV during the push-in (pre_roll_s), and the last
    carries the footage for the pull-out (post_roll_s).
    """
    jobs = []
    lo, hi = min(allowed_durations), max(allowed_durations)
    for s in plan["segments"]:
        if s["kind"] not in ("story", "cold_open"):
            continue
        pre, post = s.get("pre_roll_s", 0.0), s.get("post_roll_s", 0.0)
        chunks, cur, lens, cur_d = [], [], [], 0.0
        for k, sh in enumerate(s["shots"]):
            d = sh["duration_s"] + (pre if k == 0 else 0) + (post if k == len(s["shots"]) - 1 else 0)
            if cur and cur_d + d + handle_s > hi:
                chunks.append((cur, lens, cur_d))
                cur, lens, cur_d = [], [], 0.0
            cur.append(sh)
            lens.append(round(d, 3))
            cur_d += d
        if cur:
            chunks.append((cur, lens, cur_d))
        for ci, (chunk, lengths, need) in enumerate(chunks):
            want = need + handle_s
            dur = next((d for d in sorted(allowed_durations) if d >= want), hi)
            jobs.append({
                "job_id": f"{s['id']}_c{ci}",
                "segment": s["id"],
                "image": s["image"],
                "characters": s["characters"],
                "shots": chunk,
                "lengths": lengths,          # on-screen time per shot, including TV roll
                "duration": max(lo, dur),
                "need_s": round(need, 3),
            })
    return jobs


def estimate_cost(jobs: list[dict], price_per_s: float, retry_factor: float = 1.4) -> dict:
    secs = sum(j["duration"] for j in jobs)
    return {"clips": len(jobs), "seconds": secs, "price_per_s": price_per_s,
            "first_pass_usd": round(secs * price_per_s, 2),
            "with_retries_usd": round(secs * price_per_s * retry_factor, 2)}


def save(plan: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(plan, indent=2, default=str), encoding="utf-8")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def total_duration(plan: dict) -> float:
    return round(sum(s.get("duration_s") or 0 for s in plan["segments"]), 3)


def ceil_to(x: float, step: float) -> float:
    return math.ceil(x / step) * step
