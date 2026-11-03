"""Run the full breakdown of one video and return a plain dict.

The dict is the single output format: the CLI writes it to JSON, the
dashboard reads it, and blueprint.py turns it into a format template.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import __version__
from .audio import analyse_audio
from .scenes import SceneParams, build_scenes, detect_boundaries
from .transcript import Line, speech_stats
from .video import probe, read_frames

HOOK_S = 3.0


def _pacing(scenes, duration_s: float) -> dict:
    lengths = np.array([s.duration_s for s in scenes])
    cuts = len(scenes) - 1
    return {
        "scenes": len(scenes),
        "cuts": cuts,
        "cuts_per_minute": round(cuts / duration_s * 60, 1) if duration_s else None,
        "mean_shot_s": round(float(lengths.mean()), 2),
        "median_shot_s": round(float(np.median(lengths)), 2),
        "shortest_shot_s": round(float(lengths.min()), 2),
        "longest_shot_s": round(float(lengths.max()), 2),
        # Coefficient of variation: 0 means every shot is the same length.
        "shot_length_cv": round(float(lengths.std() / lengths.mean()), 2) if lengths.mean() else None,
        "gradual_transitions": sum(s.transition_in == "gradual" for s in scenes),
    }


def _hook(fs, scenes, audio, speech) -> dict:
    n_hook = min(fs.n, int(round(HOOK_S * fs.fps)))
    rest = fs.motion[n_hook:]
    first_cut = next((s.start_s for s in scenes if s.index > 0), None)

    def ratio(a, b):
        return round(float(a) / float(b), 2) if b else None

    return {
        "window_s": HOOK_S,
        "first_cut_s": first_cut,
        "cuts_in_window": sum(1 for s in scenes if 0 < s.start_s < HOOK_S),
        "motion_hook_vs_rest": ratio(fs.motion[1:n_hook].mean(), rest.mean()) if rest.size else None,
        "brightness_first_frame": round(float(fs.brightness[0]), 1),
        "brightness_vs_video_mean": ratio(fs.brightness[0], fs.brightness.mean()),
        "colourfulness_hook_vs_rest": ratio(fs.colourfulness[:n_hook].mean(),
                                            fs.colourfulness[n_hook:].mean()) if fs.n > n_hook else None,
        "first_sound_s": audio.first_sound_s if audio else None,
        "first_word_s": speech.first_word_s if speech else None,
        "first_line": speech.first_line if speech else None,
        "first_line_is_question": speech.first_line_is_question if speech else None,
    }


def _per_second(fs, audio) -> list[dict]:
    """One row per second: how much is moving and how loud it is."""
    secs = int(np.ceil(fs.n / fs.fps))
    loud = {}
    if audio:
        for t, db in audio.loudness_curve:
            loud.setdefault(int(t), []).append(db)
    rows = []
    for s in range(secs):
        a, b = int(s * fs.fps), int(min(fs.n, (s + 1) * fs.fps))
        if b <= a:
            break
        rows.append({
            "t": s,
            "motion": round(float(fs.motion[a:b].mean()), 2),
            "brightness": round(float(fs.brightness[a:b].mean()), 1),
            "loudness_dbfs": round(float(np.mean(loud[s])), 1) if s in loud else None,
        })
    return rows


def analyse_video(path: str | Path, transcript: list[Line] | None = None, vision: bool = False,
                  params: SceneParams | None = None, vision_client=None) -> dict:
    info = probe(path)
    fs = read_frames(path)
    boundaries = detect_boundaries(fs, params)
    scenes = build_scenes(fs, boundaries)
    duration = fs.n / fs.fps
    audio = analyse_audio(path) if info.has_audio else None
    speech = speech_stats(transcript, duration) if transcript else None

    report = {
        "reelwise_version": __version__,
        "source": {"file": Path(path).name},
        "video": {
            "width": info.width, "height": info.height, "orientation": info.orientation,
            "fps": round(fs.fps, 3), "frames": fs.n, "duration_s": round(duration, 2),
            "has_audio": info.has_audio, "codec": info.codec,
        },
        "pacing": _pacing(scenes, duration),
        "hook": _hook(fs, scenes, audio, speech),
        "scenes": [s.to_dict() for s in scenes],
        "audio": {k: v for k, v in audio.to_dict().items() if k != "loudness_curve"} if audio else None,
        "speech": speech.to_dict() if speech else None,
        "transcript": [asdict(l) for l in transcript] if transcript else None,
        "per_second": _per_second(fs, audio),
        "visual": {
            "mean_brightness": round(float(fs.brightness.mean()), 1),
            "mean_saturation": round(float(fs.saturation.mean()), 1),
            "mean_colourfulness": round(float(fs.colourfulness.mean()), 1),
            "mean_motion": round(float(fs.motion[1:].mean()), 2) if fs.n > 1 else 0.0,
        },
        "vision": None,
    }
    if vision:
        from .vision import describe_scenes
        report["vision"] = describe_scenes(fs, scenes, client=vision_client)
    return report
