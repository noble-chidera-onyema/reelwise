"""Audio measurements: loudness over time, when sound starts, pauses.

The soundtrack is pulled out with ffmpeg as 16 kHz mono and measured in 50 ms
blocks. Loudness is RMS in dBFS, which is simpler than LUFS but tracks it
closely for speech; it is labelled dBFS everywhere so nobody mistakes it for
a broadcast loudness figure.

"Active" means louder than the quiet floor of this particular file (its 10th
percentile plus 12 dB, capped at the median minus 3 dB), with an absolute
floor of -50 dBFS. Using the file's own floor matters: music beds and room
tone sit well above digital silence.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

SR = 16000
BLOCK_S = 0.05


@dataclass
class AudioStats:
    duration_s: float
    mean_active_dbfs: float | None
    loudness_range_db: float | None      # 95th minus 10th percentile of block loudness
    active_ratio: float                  # share of blocks above the quiet floor
    first_sound_s: float | None          # first active block
    longest_pause_s: float               # longest run of quiet blocks after first sound
    pauses_over_300ms: int
    loudness_curve: list                 # [(t, dBFS)] at 0.25 s steps, for charts

    def to_dict(self) -> dict:
        return asdict(self)


def extract_pcm(path: str | Path) -> np.ndarray | None:
    if not shutil.which("ffmpeg"):
        return None
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
         "-f", "s16le", "-"],
        capture_output=True, check=False,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def measure(samples: np.ndarray) -> AudioStats:
    block = int(SR * BLOCK_S)
    n_blocks = len(samples) // block
    if n_blocks == 0:
        return AudioStats(0.0, None, None, 0.0, None, 0.0, 0, [])
    x = samples[: n_blocks * block].reshape(n_blocks, block)
    rms = np.sqrt((x ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-6))

    # Quiet floor: 12 dB above the file's quietest tenth, but never above the
    # median minus 3 dB (otherwise a steady music bed would count as silent).
    floor = max(min(np.percentile(db, 10) + 12.0, np.median(db) - 3.0), -50.0)
    active = db > floor
    first = int(np.argmax(active)) if active.any() else None

    longest, pauses, run = 0, 0, 0
    if first is not None:
        for a in active[first:]:
            if not a:
                run += 1
            else:
                if run * BLOCK_S >= 0.3:
                    pauses += 1
                longest = max(longest, run)
                run = 0
        # trailing quiet after the last sound is an ending, not a pause

    step = int(round(0.25 / BLOCK_S))
    curve = [(round(i * BLOCK_S, 2), round(float(db[i: i + step].mean()), 1)) for i in range(0, n_blocks, step)]
    act_db = db[active]
    return AudioStats(
        duration_s=round(n_blocks * BLOCK_S, 2),
        mean_active_dbfs=round(float(act_db.mean()), 1) if act_db.size else None,
        loudness_range_db=round(float(np.percentile(db, 95) - np.percentile(db, 10)), 1),
        active_ratio=round(float(active.mean()), 3),
        first_sound_s=round(first * BLOCK_S, 2) if first is not None else None,
        longest_pause_s=round(longest * BLOCK_S, 2),
        pauses_over_300ms=pauses,
        loudness_curve=curve,
    )


def analyse_audio(path: str | Path) -> AudioStats | None:
    pcm = extract_pcm(path)
    if pcm is None or pcm.size == 0:
        return None
    return measure(pcm)
