"""Voice each host's typical lines with the profile voices, plus Ren's alternatives.

    python tools/voice_check.py            # writes voice_check/*.wav and prints a summary

Each file is trimmed speech at 48 kHz. The summary gives length, words per second
and median pitch, so the three hosts can be compared before an episode is made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reelwise.studio.voices import engine_from_config, median_f0, read_wav  # noqa: E402

LINES = {
    "Ren": "In June of nineteen ninety seven, a crew arrived to cut down forty old trees on Broadway. "
           "And that's one worth telling.",
    "Bex": "Wait, he did what? Okay, but how did he get down?",
    "Ozzy": "Worst treehouse party ever.",
}


def main() -> int:
    prof = json.loads((ROOT / "profiles" / "worth_telling.json").read_text())
    out = ROOT / "voice_check"
    out.mkdir(exist_ok=True)
    rows = []
    for c in prof["characters"]:
        name = c["name"]
        cfgs = [c["voice"]]
        for v in prof.get("voice_candidates", {}).get(name, []):
            if v != c["voice"].get("voice"):
                cfgs.append({**c["voice"], "voice": v})
        for cfg in cfgs:
            path = out / f"{name}_{cfg['voice']}.wav"
            try:
                sp = engine_from_config(cfg).speak(LINES[name], path)
                f0 = median_f0(read_wav(sp.path))
                rows.append((name, cfg["voice"], sp.duration_s, len(LINES[name].split()) / sp.duration_s, f0,
                             len(sp.words)))
            except Exception as e:
                rows.append((name, cfg["voice"], 0, 0, None, f"error: {e}"))
    for r in rows:
        f0 = f"{r[4]:.0f} Hz" if r[4] else "n/a"
        print(f"{r[0]:5s} {r[1]:32s} {r[2]:5.2f}s  {r[3]:4.2f} words/s  pitch {f0:7s} word timings: {r[5]}")
    print(f"files in {out}")
    return 0 if all(r[2] > 0 for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
