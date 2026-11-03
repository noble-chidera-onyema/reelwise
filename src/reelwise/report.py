"""Plain markdown versions of the JSON outputs, for reading or pasting into notes."""

from __future__ import annotations


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _n(count, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def breakdown_md(r: dict) -> str:
    v, p, h = r["video"], r["pacing"], r["hook"]
    out = [f"# Breakdown: {r['source']['file']}", ""]
    out.append(f"{v['width']}x{v['height']} ({v['orientation']}), {v['duration_s']}s at {v['fps']} fps, "
               f"audio: {'yes' if v['has_audio'] else 'no'}.")
    out += ["", "## Pacing", "",
            f"{p['scenes']} shots, {p['cuts_per_minute']} cuts per minute. Median shot {p['median_shot_s']}s "
            f"(shortest {p['shortest_shot_s']}s, longest {p['longest_shot_s']}s). "
            f"Shot length variation (CV) {_fmt(p['shot_length_cv'])}; {p['gradual_transitions']} dissolves or fades.",
            "", "## First three seconds", ""]
    out.append(f"- First cut at {_fmt(h['first_cut_s'])}s; {_n(h['cuts_in_window'], 'cut')} inside the window.")
    out.append(f"- Motion in the opening versus the rest of the video: {_fmt(h['motion_hook_vs_rest'])}x.")
    out.append(f"- First frame brightness {h['brightness_first_frame']} (video average x{_fmt(h['brightness_vs_video_mean'])}).")
    out.append(f"- First sound at {_fmt(h['first_sound_s'])}s; first word at {_fmt(h['first_word_s'])}s.")
    if h.get("first_line"):
        q = " (a question)" if h["first_line_is_question"] else ""
        out.append(f"- Opening line{q}: \"{h['first_line']}\"")
    if r.get("speech"):
        s = r["speech"]
        out += ["", "## Speech", "",
                f"{s['n_words']} words in {s['n_lines']} caption lines. {_fmt(s['words_per_second'])} words per second "
                f"while speaking. {_n(s['questions'], 'question')}, {_n(s['spoken_numbers'], 'number')}, "
                f"{_n(s['one_word_lines'], 'one-word line')}."]
    if r.get("audio"):
        a = r["audio"]
        out += ["", "## Audio", "",
                f"Sound is present {round(100 * a['active_ratio'])}% of the time at about {_fmt(a['mean_active_dbfs'])} dBFS. "
                f"Longest pause {a['longest_pause_s']}s; {_n(a['pauses_over_300ms'], 'pause')} over 0.3s."]
    out += ["", "## Shots", "", "| # | start | length | in | motion | brightness |", "|---|---|---|---|---|---|"]
    for s in r["scenes"]:
        out.append(f"| {s['index']} | {s['start_s']}s | {s['duration_s']}s | {s['transition_in']} | {s['motion']} | {s['brightness']} |")
    vis = r.get("vision")
    if vis:
        out += ["", "## What is on screen", ""]
        of = vis.get("opening_frame") or {}
        if of:
            out.append(f"Opening frame: {of.get('what_is_shown', '')}. Text: {of.get('on_screen_text', '') or 'none'}.")
            out.append("")
        for s in vis.get("scenes", []):
            out.append(f"- S{s.get('scene')}: {s.get('shot_type', '')}, {s.get('subject', '')}; "
                       f"{s.get('action', '')}. Text: {s.get('on_screen_text', '') or 'none'}.")
        if vis.get("format_summary"):
            out += ["", f"Format: {vis['format_summary']}"]
    return "\n".join(out) + "\n"


def outliers_md(channel: dict, df, comp: dict, top: int = 10) -> str:
    out = [f"# Outlier report: {channel['title']}", "",
           f"{channel['subscribers']:,} subscribers. {len(df)} uploads analysed, "
           f"{comp['n_overperformers']} scored at 2x the channel's normal or better.", "",
           f"Read this as: {comp['evidence']}.", "", "## Biggest overperformers", "",
           "| score | views | length | title |", "|---|---|---|---|"]
    best = df.dropna(subset=["outlier_score"]).sort_values("outlier_score", ascending=False).head(top)
    for _, row in best.iterrows():
        out.append(f"| {row['outlier_score']}x | {int(row['views']):,} | {int(row['duration_s'])}s | {row['title']} |")
    out += ["", "## What the overperformers had in common", "",
            "Effect size runs from -1 to +1 (Cliff's delta). Around 0.15 is small, 0.33 medium, 0.47 large.", "",
            "| feature | overperformers | rest | measure | effect |", "|---|---|---|---|---|"]
    for f in comp["features"]:
        out.append(f"| {f['feature']} | {f['overperformers']} | {f['rest']} | {f['summary']} | {f['effect_size']} |")
    if comp["title_words"]:
        out += ["", "## Title words more common in overperformers", ""]
        out += [f"- {w['word']}: {round(100 * w['over_share'])}% of overperformer titles vs {round(100 * w['rest_share'])}% of the rest"
                for w in comp["title_words"]]
    return "\n".join(out) + "\n"
