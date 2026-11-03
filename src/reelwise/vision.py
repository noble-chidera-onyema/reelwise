"""Describe what is on screen, scene by scene, with a vision model.

The numeric pipeline can tell you a cut happened at 2.4 s and the next shot
is darker and moves more. It cannot tell you the shot is a close-up of a man
holding a cheque with yellow caption text. This module sends one keyframe per
scene (plus the very first frame, which is the thumbnail on most feeds) to
Claude and asks for a fixed JSON description of each.

Optional: needs ANTHROPIC_API_KEY. Cost is roughly one image per scene, and
frames are capped at MAX_FRAMES per video.
"""

from __future__ import annotations

import base64
import json
import os
import re

import cv2
import numpy as np

from .scenes import Scene
from .video import FrameStats

MAX_FRAMES = 20
DEFAULT_MODEL = os.environ.get("REELWISE_MODEL", "claude-sonnet-5-5")

PROMPT = """You are logging the shots of a short video for a production analyst.
Each image below is one frame, labelled with its scene number and timestamp.
Frame "S0-open" is the very first frame of the video.

Describe only what is visible. Do not guess at audio, intent or audience
reaction. If something is unclear, say "unclear". Keep each field short.

Return JSON only, no prose, in exactly this shape:
{
  "opening_frame": {"what_is_shown": "", "on_screen_text": "", "why_it_might_stop_a_scroll": ""},
  "scenes": [
    {"scene": 0, "shot_type": "extreme close-up|close-up|medium|wide|screen recording|graphic|other",
     "subject": "", "action": "", "setting": "", "on_screen_text": "",
     "caption_style": "", "people_count": 0, "facial_expression": "", "notable": ""}
  ],
  "format_summary": "one or two sentences describing the format as a repeatable template"
}"""


def _encode(img: np.ndarray, quality: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _nearest_thumb(fs: FrameStats, frame: int) -> np.ndarray | None:
    if not fs.thumbs:
        return None
    idx = min(range(len(fs.thumbs)), key=lambda i: abs(fs.thumbs[i][0] - frame))
    return fs.thumbs[idx][1]


def pick_keyframes(fs: FrameStats, scenes: list[Scene]) -> list[tuple[str, float, np.ndarray]]:
    chosen = scenes
    if len(scenes) > MAX_FRAMES - 1:
        idx = np.linspace(0, len(scenes) - 1, MAX_FRAMES - 1).round().astype(int)
        chosen = [scenes[i] for i in sorted(set(idx))]
    out = []
    first = _nearest_thumb(fs, 0)
    if first is not None:
        out.append(("S0-open", 0.0, first))
    for sc in chosen:
        img = _nearest_thumb(fs, sc.keyframe)
        if img is not None:
            out.append((f"S{sc.index}", sc.keyframe / fs.fps, img))
    return out


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("Model reply contained no JSON object")
    return json.loads(m.group(0))


def describe_scenes(fs: FrameStats, scenes: list[Scene], model: str | None = None,
                    client=None) -> dict:
    """Return the model's JSON description. `client` can be injected for tests."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    frames = pick_keyframes(fs, scenes)
    content: list[dict] = []
    for label, t, img in frames:
        content.append({"type": "text", "text": f"{label} at {t:.2f}s"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": _encode(img)}})
    content.append({"type": "text", "text": PROMPT})
    msg = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content)
    result = _extract_json(text)
    result["_frames_sent"] = [label for label, _, _ in frames]
    result["_model"] = model or DEFAULT_MODEL
    return result
