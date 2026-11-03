"""Make an episode from the command line, exactly as the Studio page does.

    python tools/make_episode.py --script story.txt --images DIR [--profile worth_telling]
                                 [--motion motion|motion+hf|draft] [--music track.mp3] [--out projects/NAME]

Images in DIR are numbered in name order (Image: 1 is the first file). Writes
final.mp4, final.srt and report.json into --out and prints every check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reelwise.blueprint import check_script  # noqa: E402
from reelwise.env import load_env  # noqa: E402
from reelwise.studio import plan as P  # noqa: E402
from reelwise.studio.pipeline import Studio  # noqa: E402


def rig_for(profile_name: str) -> dict | None:
    d = ROOT / "profiles" / profile_name
    f = d / "rig.json"
    if not f.exists():
        return None
    cfg = json.loads(f.read_text())
    cfg["background"] = str(d / cfg["background"])
    for h in cfg.get("hosts", {}).values():
        for k in ("closed", "open", "blink"):
            if h.get(k):
                h[k] = str(d / h[k])
    return cfg


def main() -> int:
    load_env(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--profile", default="worth_telling")
    ap.add_argument("--motion", default="motion")
    ap.add_argument("--music", default=None)
    ap.add_argument("--out", default=str(ROOT / "projects" / "episode"))
    a = ap.parse_args()

    profile = json.loads((ROOT / "profiles" / f"{a.profile}.json").read_text())
    hosts = [c["name"] for c in profile["characters"]]
    segs = P.parse_scene_text(Path(a.script).read_text(encoding="utf-8"), hosts)
    lines = [{"speaker": l["speaker"], "text": l["text"]} for s in segs for l in s["lines"]]
    print("Script checks:")
    for c in check_script({"lines": lines}, profile)["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'WARN'}] {c['rule']}: {c['detail']}")
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
    images = [{"id": i, "path": str(p)} for i, p in enumerate(files, 1)]
    plan = P.new_plan(segs, profile, images, [], title=Path(a.out).name)
    st = Studio(a.out)
    voices = {c["name"]: c["voice"] for c in profile["characters"]}
    st.prepare(plan, voices, zoom_s=1.0)
    clips = st.generate(plan, a.motion)
    rep = st.assemble(plan, clips, rig_for(a.profile), music=a.music)
    print("Episode checks:")
    for c in rep["final"]:
        print(f"  [{'PASS' if c['passed'] else c['severity'].upper()}] {c['rule']}: {c['detail']}")
    print("READY" if rep["passed"] else "NOT READY", rep["file"])
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
