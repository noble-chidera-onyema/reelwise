"""reelwise dashboard.

    streamlit run app/streamlit_app.py

Keys are read from .env or the environment: ANTHROPIC_API_KEY for shot
descriptions and episode writing, YOUTUBE_API_KEY for channel and trending
data. Every tab works up to the point where its key is needed and says so.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reelwise.analyse import analyse_video  # noqa: E402
from reelwise.blueprint import build_format, check_script, check_video, write_episode  # noqa: E402
from reelwise.env import load_env  # noqa: E402
from reelwise.outliers import compare, score, summarise_trending, to_frame  # noqa: E402
from reelwise.report import breakdown_md  # noqa: E402
from reelwise.transcript import fetch_youtube_transcript, parse_captions  # noqa: E402
from reelwise.video import grab_frame  # noqa: E402

load_env(ROOT / ".env")
st.set_page_config(page_title="reelwise", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .rw-note {color: #6b7280; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

HAS_CLAUDE = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_YT = bool(os.environ.get("YOUTUBE_API_KEY"))


def _profiles() -> dict[str, Path]:
    found = {}
    for folder in (ROOT / "profiles", ROOT / "examples"):
        for p in sorted(folder.glob("*.json")):
            found[p.stem] = p
    return found


st.title("reelwise")
st.markdown('<p class="rw-note">Measure a short video shot by shot, find what overperforms on a channel, '
            'and fit a proven format to your own characters.</p>', unsafe_allow_html=True)

tab_video, tab_channel, tab_trend, tab_write, tab_method = st.tabs(
    ["Break down a video", "Channel outliers", "Trending", "Write an episode", "Method and limits"])

# --------------------------------------------------------------------------- video
with tab_video:
    left, right = st.columns([2, 1])
    with left:
        up = st.file_uploader("Video file (mp4, mov, webm, mkv)", type=["mp4", "mov", "webm", "mkv", "m4v"])
    with right:
        cap_up = st.file_uploader("Captions, optional (.srt or .vtt)", type=["srt", "vtt"])
        yt_url = st.text_input("Or the YouTube URL of the same video, for its captions")
    use_vision = st.checkbox("Describe each shot with Claude", value=False, disabled=not HAS_CLAUDE,
                             help=None if HAS_CLAUDE else "Add ANTHROPIC_API_KEY to .env to turn this on.")

    if up and st.button("Analyse", type="primary"):
        suffix = Path(up.name).suffix
        tmp = Path(tempfile.mkdtemp()) / ("upload" + suffix)
        tmp.write_bytes(up.getbuffer())
        lines = None
        try:
            if cap_up:
                lines = parse_captions(cap_up.getvalue().decode("utf-8", errors="replace"))
            elif yt_url.strip():
                lines = fetch_youtube_transcript(yt_url.strip())
        except Exception as e:
            st.warning(f"Captions not loaded: {e}")
        with st.spinner("Reading every frame"):
            report = analyse_video(tmp, transcript=lines, vision=use_vision)
        report["source"]["file"] = up.name
        st.session_state["report"] = report
        st.session_state["video_path"] = str(tmp)

    report = st.session_state.get("report")
    if report:
        v, p, h = report["video"], report["pacing"], report["hook"]
        c = st.columns(5)
        c[0].metric("Length", f"{v['duration_s']}s")
        c[1].metric("Shots", p["scenes"])
        c[2].metric("Cuts per minute", p["cuts_per_minute"])
        c[3].metric("Median shot", f"{p['median_shot_s']}s")
        c[4].metric("First cut", f"{h['first_cut_s']}s" if h["first_cut_s"] is not None else "none")

        ps = pd.DataFrame(report["per_second"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ps["t"], y=ps["motion"], name="Motion", line=dict(color="#2563eb")))
        if ps["loudness_dbfs"].notna().any():
            fig.add_trace(go.Scatter(x=ps["t"], y=ps["loudness_dbfs"], name="Loudness (dBFS)",
                                     yaxis="y2", line=dict(color="#d97706")))
        for s in report["scenes"][1:]:
            fig.add_vline(x=s["start_s"], line=dict(color="#9ca3af", width=1, dash="dot"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                          title="Motion and loudness per second (dotted lines are cuts)",
                          yaxis=dict(title="Motion"), yaxis2=dict(title="dBFS", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Shots")
        vp = st.session_state.get("video_path")
        vis = {s.get("scene"): s for s in (report.get("vision") or {}).get("scenes", [])}
        cols = st.columns(6)
        for i, s in enumerate(report["scenes"][:24]):
            with cols[i % 6]:
                img = grab_frame(vp, s["keyframe"], width=240) if vp and Path(vp).exists() else None
                if img is not None:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
                cap = f"S{s['index']}  {s['start_s']}s, {s['duration_s']}s"
                if s["index"] in vis:
                    cap += f"  {vis[s['index']].get('shot_type', '')}"
                st.caption(cap)
        if len(report["scenes"]) > 24:
            st.caption(f"Showing 24 of {len(report['scenes'])} shots. The full list is in the download.")

        st.markdown(breakdown_md(report).split("## Shots")[0])

        prof = _profiles()
        if prof:
            name = st.selectbox("Check this video against a profile", list(prof))
            qa = check_video(report, json.loads(prof[name].read_text()))
            for chk in qa["checks"]:
                (st.success if chk["passed"] else st.error)(f"{chk['rule']}: {chk['detail']}")

        d1, d2 = st.columns(2)
        stem = Path(report["source"]["file"]).stem
        d1.download_button("Download breakdown (JSON)", json.dumps(report, indent=2, default=str),
                           file_name=f"{stem}.breakdown.json")
        d2.download_button("Download breakdown (Markdown)", breakdown_md(report), file_name=f"{stem}.breakdown.md")

# --------------------------------------------------------------------------- channel
with tab_channel:
    if not HAS_YT:
        st.info("Add YOUTUBE_API_KEY to .env to pull channel data. The README explains how to get a free key.")
    ref = st.text_input("Channel (@handle, URL or channel id)")
    limit = st.slider("Uploads to analyse", 20, 500, 200, step=20)
    if ref and HAS_YT and st.button("Find overperformers", type="primary"):
        from reelwise.youtube import YouTubeClient
        yt = YouTubeClient()
        with st.spinner("Fetching uploads"):
            ch, vids = yt.channel_videos(ref, limit=limit)
        df = score(to_frame(vids))
        st.session_state["channel"] = (ch, df, compare(df), yt.quota_used)

    if "channel" in st.session_state:
        ch, df, comp, quota = st.session_state["channel"]
        st.markdown(f"**{ch['title']}**, {ch['subscribers']:,} subscribers. {len(df)} uploads, "
                    f"{comp['n_overperformers']} at 2x the channel's normal or better. API quota used: {quota}.")
        st.caption(f"Read the comparison as {comp['evidence']}.")
        scored = df.dropna(subset=["outlier_score"])
        colours = {"overperformer": "#16a34a", "typical": "#9ca3af", "underperformer": "#dc2626"}
        fig = go.Figure()
        for g, col in colours.items():
            part = scored[scored["group"] == g]
            fig.add_trace(go.Scatter(x=part["published_at"], y=part["views"], mode="markers", name=g,
                                     marker=dict(color=col, size=8), text=part["title"],
                                     hovertemplate="%{text}<br>%{y:,} views<extra></extra>"))
        fig.update_layout(height=360, yaxis_type="log", yaxis_title="Views (log scale)",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(scored.sort_values("outlier_score", ascending=False)
                     [["outlier_score", "views", "duration_s", "title", "published_at", "url"]].head(25),
                     width="stretch", hide_index=True)
        st.subheader("What the overperformers had in common")
        st.dataframe(pd.DataFrame(comp["features"]), width="stretch", hide_index=True)
        if comp["title_words"]:
            st.dataframe(pd.DataFrame(comp["title_words"]), width="stretch", hide_index=True)
        st.download_button("Download CSV", df.drop(columns=["description"]).to_csv(index=False),
                           file_name=f"{(ch.get('handle') or ch['id']).lstrip('@')}_outliers.csv")

# --------------------------------------------------------------------------- trending
with tab_trend:
    if not HAS_YT:
        st.info("Add YOUTUBE_API_KEY to .env to read YouTube's most popular chart.")
    c1, c2 = st.columns(2)
    region = c1.selectbox("Region", ["GB", "US", "NG", "IE", "CA", "AU", "IN"])
    category = c2.selectbox("Category", ["All", "24 Entertainment", "22 People & Blogs", "25 News & Politics",
                                         "27 Education", "23 Comedy"])
    if HAS_YT and st.button("Load chart", type="primary"):
        from reelwise.youtube import YouTubeClient
        cat = None if category == "All" else category.split()[0]
        vids = YouTubeClient().trending(region=region, category_id=cat)
        st.session_state["trending"] = (vids, summarise_trending(vids))
    if "trending" in st.session_state:
        vids, summ = st.session_state["trending"]
        c = st.columns(4)
        c[0].metric("Videos", summ["n"])
        c[1].metric("Median length", f"{summ['median_duration_s']:.0f}s")
        c[2].metric("Question in title", f"{round(100 * summ['share_title_question'])}%")
        c[3].metric("Number in title", f"{round(100 * summ['share_title_number'])}%")
        st.dataframe(pd.DataFrame(vids)[["title", "channel", "views", "duration_s", "published_at", "url"]],
                     width="stretch", hide_index=True)
        st.caption("Most common title words: " + ", ".join(f"{w} ({n})" for w, n in summ["top_title_words"]))

# --------------------------------------------------------------------------- write
with tab_write:
    report = st.session_state.get("report")
    if not report:
        st.info("Break down a reference video first. Its timing becomes the template for the episode.")
    else:
        fmt = build_format(report)
        st.caption(f"Template from {fmt['derived_from']}: {len(fmt['beats'])} beats, "
                   f"{fmt['pacing']['cuts_per_minute']} cuts per minute, target "
                   f"{fmt['target_duration_range_s'][0]}-{fmt['target_duration_range_s'][1]}s.")
        prof = _profiles()
        pname = st.selectbox("Profile", list(prof), key="write_profile")
        profile = json.loads(prof[pname].read_text())
        story = st.text_area("The true story (facts only, with sources if you have them)", height=200)
        if not HAS_CLAUDE:
            st.info("Add ANTHROPIC_API_KEY to .env to write episodes.")
        elif story.strip() and st.button("Write episode", type="primary"):
            with st.spinner("Writing"):
                st.session_state["episode"] = write_episode(fmt, profile, story)
        ep = st.session_state.get("episode")
        if ep:
            qa = check_script(ep, profile)
            st.subheader(ep.get("title", "Episode"))
            st.dataframe(pd.DataFrame(qa["timeline"]), width="stretch", hide_index=True)
            for chk in qa["checks"]:
                (st.success if chk["passed"] else st.error)(f"{chk['rule']}: {chk['detail']}")
            if ep.get("shots"):
                st.dataframe(pd.DataFrame(ep["shots"]), width="stretch", hide_index=True)
            if ep.get("facts_used"):
                st.markdown("Facts the script relies on, to verify before you publish:")
                for f in ep["facts_used"]:
                    st.markdown(f"- {f}")
            st.download_button("Download episode (JSON)", json.dumps({"script": ep, "check": qa}, indent=2),
                               file_name="episode.json")

# --------------------------------------------------------------------------- method
with tab_method:
    st.markdown((ROOT / "docs" / "how_it_works.md").read_text(encoding="utf-8"))
