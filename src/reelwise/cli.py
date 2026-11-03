"""Command line entry point.

    reelwise analyse VIDEO [--captions FILE | --youtube URL] [--vision] [--out DIR]
    reelwise channel @handle [--limit 200] [--out DIR]
    reelwise trending [--region GB] [--category 24] [--out DIR]
    reelwise blueprint BREAKDOWN.json --profile PROFILE.json --story STORY.txt [--out DIR]
    reelwise check-script SCRIPT.json --profile PROFILE.json
    reelwise check-video BREAKDOWN.json --profile PROFILE.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .env import load_env


def _write(out_dir: Path, name: str, data, md: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    if md:
        (out_dir / f"{name}.md").write_text(md, encoding="utf-8")
    print(f"wrote {out_dir / name}.json" + (" and .md" if md else ""))


def cmd_analyse(a) -> int:
    from .analyse import analyse_video
    from .report import breakdown_md
    from .transcript import fetch_youtube_transcript, load_caption_file

    lines = None
    if a.captions:
        lines = load_caption_file(a.captions)
    elif a.youtube:
        lines = fetch_youtube_transcript(a.youtube)
    r = analyse_video(a.video, transcript=lines, vision=a.vision)
    _write(Path(a.out), Path(a.video).stem + ".breakdown", r, breakdown_md(r))
    return 0


def cmd_channel(a) -> int:
    from .outliers import compare, score, to_frame
    from .report import outliers_md
    from .youtube import YouTubeClient

    yt = YouTubeClient()
    ch, vids = yt.channel_videos(a.channel, limit=a.limit)
    df = score(to_frame(vids))
    comp = compare(df)
    out = Path(a.out)
    name = (ch.get("handle") or ch["id"]).lstrip("@") + ".outliers"
    _write(out, name, {"channel": ch, "comparison": comp,
                       "videos": df.drop(columns=["description"]).to_dict(orient="records")},
           outliers_md(ch, df, comp))
    df.drop(columns=["description"]).to_csv(out / f"{name}.csv", index=False)
    print(f"YouTube API quota used: {yt.quota_used} units")
    return 0


def cmd_trending(a) -> int:
    from .outliers import summarise_trending
    from .youtube import YouTubeClient

    yt = YouTubeClient()
    vids = yt.trending(region=a.region, category_id=a.category)
    _write(Path(a.out), f"trending_{a.region}", {"summary": summarise_trending(vids), "videos": vids})
    return 0


def cmd_blueprint(a) -> int:
    from .blueprint import build_format, check_script, load_profile, write_episode

    report = json.loads(Path(a.breakdown).read_text(encoding="utf-8"))
    profile = load_profile(a.profile)
    fmt = build_format(report)
    story = Path(a.story).read_text(encoding="utf-8")
    script = write_episode(fmt, profile, story)
    qa = check_script(script, profile)
    _write(Path(a.out), Path(a.story).stem + ".episode", {"format": fmt, "script": script, "check": qa})
    _print_checks(qa)
    return 0 if qa["passed"] else 1


def cmd_check_script(a) -> int:
    from .blueprint import check_script, load_profile

    script = json.loads(Path(a.script).read_text(encoding="utf-8"))
    script = script.get("script", script)
    qa = check_script(script, load_profile(a.profile))
    _print_checks(qa)
    return 0 if qa["passed"] else 1


def cmd_check_video(a) -> int:
    from .blueprint import check_video, load_profile

    qa = check_video(json.loads(Path(a.breakdown).read_text(encoding="utf-8")), load_profile(a.profile))
    _print_checks(qa)
    return 0 if qa["passed"] else 1


def _print_checks(qa: dict) -> None:
    for c in qa["checks"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['rule']}: {c['detail']}")


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(prog="reelwise", description="Measure short-form video, scene by scene.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyse", help="break down a video file")
    p.add_argument("video")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--captions", help=".srt or .vtt file for this video")
    g.add_argument("--youtube", help="YouTube URL of the same video, to fetch its captions")
    p.add_argument("--vision", action="store_true", help="describe each shot with Claude (needs ANTHROPIC_API_KEY)")
    p.add_argument("--out", default="outputs")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("channel", help="find a channel's overperforming uploads (needs YOUTUBE_API_KEY)")
    p.add_argument("channel", help="@handle, channel URL or UC... id")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out", default="outputs")
    p.set_defaults(func=cmd_channel)

    p = sub.add_parser("trending", help="summarise YouTube's most popular chart for a region")
    p.add_argument("--region", default="GB")
    p.add_argument("--category", default=None, help="video category id, e.g. 24 Entertainment, 22 People & Blogs")
    p.add_argument("--out", default="outputs")
    p.set_defaults(func=cmd_trending)

    p = sub.add_parser("blueprint", help="write an episode for your characters on a reference format")
    p.add_argument("breakdown")
    p.add_argument("--profile", required=True)
    p.add_argument("--story", required=True, help="text file with the true story's facts")
    p.add_argument("--out", default="outputs")
    p.set_defaults(func=cmd_blueprint)

    p = sub.add_parser("check-script", help="check a script against your profile rules")
    p.add_argument("script")
    p.add_argument("--profile", required=True)
    p.set_defaults(func=cmd_check_script)

    p = sub.add_parser("check-video", help="check a finished render against your profile rules")
    p.add_argument("breakdown")
    p.add_argument("--profile", required=True)
    p.set_defaults(func=cmd_check_video)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except Exception as e:  # one readable line instead of a traceback for expected failures
        if os.environ.get("REELWISE_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
