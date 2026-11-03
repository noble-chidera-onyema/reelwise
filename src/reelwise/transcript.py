"""Transcripts: load them, then measure how the words are delivered.

Sources, in order of preference:
  1. A caption file you supply (.srt or .vtt), for example the one your
     editor or TikTok/CapCut exported. This works for any platform.
  2. The public captions of a YouTube video, fetched with
     youtube-transcript-api. Only works where the uploader or YouTube's
     auto-captioning has produced captions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_TAG = re.compile(r"<[^>]+>")


@dataclass
class Line:
    start: float
    end: float
    text: str


@dataclass
class SpeechStats:
    n_lines: int
    n_words: int
    first_word_s: float | None
    first_line: str
    first_line_is_question: bool
    words_per_second: float | None       # over time when someone is speaking
    words_per_second_overall: float | None
    questions: int
    spoken_numbers: int
    longest_line_words: int
    one_word_lines: int

    def to_dict(self) -> dict:
        return asdict(self)


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(f"Bad timestamp: {s!r}")
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def parse_captions(text: str) -> list[Line]:
    """Parse SRT or WebVTT text into lines. Cue numbers and styling are dropped."""
    lines: list[Line] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    for block in blocks:
        rows = [r for r in block.split("\n") if r.strip()]
        arrow = next((i for i, r in enumerate(rows) if "-->" in r), None)
        if arrow is None:
            continue
        a, _, b = rows[arrow].partition("-->")
        body = " ".join(_TAG.sub("", r).strip() for r in rows[arrow + 1:]).strip()
        if body:
            lines.append(Line(_ts(a), _ts(b.split()[0]), body))
    return lines


def load_caption_file(path: str | Path) -> list[Line]:
    return parse_captions(Path(path).read_text(encoding="utf-8", errors="replace"))


def youtube_video_id(url_or_id: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url_or_id)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    raise ValueError(f"Not a YouTube video URL or id: {url_or_id}")


def fetch_youtube_transcript(url_or_id: str, languages=("en", "en-GB", "en-US")) -> list[Line]:
    from youtube_transcript_api import YouTubeTranscriptApi

    vid = youtube_video_id(url_or_id)
    fetched = YouTubeTranscriptApi().fetch(vid, languages=list(languages))
    return [Line(s.start, s.start + s.duration, s.text.replace("\n", " ")) for s in fetched]


def speech_stats(lines: list[Line], video_duration_s: float | None = None) -> SpeechStats:
    if not lines:
        return SpeechStats(0, 0, None, "", False, None, None, 0, 0, 0, 0)
    words = [len(l.text.split()) for l in lines]
    spoken_time = sum(max(l.end - l.start, 0.01) for l in lines)
    total = video_duration_s or lines[-1].end
    first = lines[0].text.strip()
    return SpeechStats(
        n_lines=len(lines),
        n_words=sum(words),
        first_word_s=round(lines[0].start, 2),
        first_line=first,
        first_line_is_question=first.rstrip().endswith("?"),
        words_per_second=round(sum(words) / spoken_time, 2) if spoken_time else None,
        words_per_second_overall=round(sum(words) / total, 2) if total else None,
        questions=sum(l.text.strip().endswith("?") for l in lines),
        spoken_numbers=sum(len(_NUMBER.findall(l.text)) for l in lines),
        longest_line_words=max(words),
        one_word_lines=sum(w == 1 for w in words),
    )
