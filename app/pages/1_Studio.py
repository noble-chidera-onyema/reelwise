"""Studio: reference video + story images + scene script + host voices -> finished episode."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reelwise.analyse import analyse_video  # noqa: E402
from reelwise.blueprint import check_script  # noqa: E402
from reelwise.env import load_env  # noqa: E402
from reelwise.studio import plan as P  # noqa: E402
from reelwise.studio.pipeline import Studio  # noqa: E402
from reelwise.studio.providers import MODELS  # noqa: E402
from reelwise.studio.tv import detect_screen  # noqa: E402
from reelwise.studio.voices import build_samples, split_utterances  # noqa: E402

load_env(ROOT / ".env")
st.set_page_config(page_title="reelwise studio", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.6rem;max-width:1250px}
.rw-note{color:#6b7280;font-size:.9rem}</style>""", unsafe_allow_html=True)

HAS_CLAUDE = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_FAL = bool(os.environ.get("FAL_KEY"))
HAS_11 = bool(os.environ.get("ELEVENLABS_API_KEY"))

TEMPLATE = """COLD OPEN
Image: 3
Stamp: 738 days
Bex: Wait, he lived up a tree for seven hundred and thirty eight days?

INTRO
Ren: This is a true story from Eugene, Oregon, and it starts with forty old trees.

SCENE 1
Image: 1
Characters: Jim
Action: A young man climbs a sweet gum tree at night, rope over his shoulder.
Ren: In June of nineteen ninety seven, a crew arrived to cut down forty old trees on Broadway.
Bex: Okay, but who climbs a tree to stop a chainsaw?

SCENE 2
Image: 2
Characters: Jim
Action: Police below spray up into the branches; he shields his face.
Ren: Protesters climbed into the branches, and the police answered with a dozen cans of pepper spray, all aimed at one man.
Ozzy: Worst treehouse party ever.

OUTRO
Bex: Send this to the friend who would never come down.
Ren: And that's one worth telling.
"""


def _profiles() -> dict[str, Path]:
    return {p.stem: p for folder in (ROOT / "profiles", ROOT / "examples") for p in sorted(folder.glob("*.json"))}


def _save(upload, folder: Path, name: str | None = None) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (name or upload.name)
    path.write_bytes(upload.getbuffer())
    return str(path)


def _resolve_rig(profile_name: str) -> dict | None:
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


st.title("Studio")
st.markdown('<p class="rw-note">Reference video, story images and a scene script in; a checked, captioned, '
            'mixed episode out. Work top to bottom.</p>', unsafe_allow_html=True)

profiles = _profiles()
c1, c2 = st.columns([1, 1])
pname = c1.selectbox("Channel profile", list(profiles), index=list(profiles).index("worth_telling")
                     if "worth_telling" in profiles else 0)
profile = json.loads(profiles[pname].read_text())
hosts = [c["name"] for c in profile["characters"]]
project = c2.text_input("Project name", value=st.session_state.get("project", "episode_01"))
st.session_state["project"] = project
proj = ROOT / "projects" / re.sub(r"[^\w-]+", "_", project)
inputs = proj / "inputs"

# --------------------------------------------------------------------------- 1 reference
with st.expander("1. Reference video", expanded=True):
    ref_up = st.file_uploader("The finished video whose pace and shot style to match", type=["mp4", "mov", "webm", "mkv"])
    if ref_up:
        rp = _save(ref_up, inputs, "reference" + Path(ref_up.name).suffix)
        if st.session_state.get("ref_path") != rp:
            with st.spinner("Breaking down every shot"):
                st.session_state["ref_report"] = analyse_video(rp)
            st.session_state["ref_path"] = rp
    ref = st.session_state.get("ref_report")
    if ref:
        p = ref["pacing"]
        m = st.columns(4)
        m[0].metric("Length", f"{ref['video']['duration_s']}s")
        m[1].metric("Shots", p["scenes"])
        m[2].metric("Median shot", f"{p['median_shot_s']}s")
        m[3].metric("Cuts per minute", p["cuts_per_minute"])
        st.caption("Shot lengths and pacing from this video steer how the planner splits each scene.")

# --------------------------------------------------------------------------- 2 images and characters
with st.expander("2. Story images and story characters", expanded=True):
    imgs_up = st.file_uploader("Story images, numbered in upload order (vertical 9:16 works best)",
                               type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    images = []
    if imgs_up:
        cols = st.columns(min(6, len(imgs_up)))
        for i, f in enumerate(imgs_up, 1):
            path = _save(f, inputs / "images", f"{i:02d}_{f.name}")
            images.append({"id": i, "path": path})
            im = cv2.imread(path)
            h, w = im.shape[:2]
            with cols[(i - 1) % len(cols)]:
                st.image(cv2.cvtColor(cv2.resize(im, (int(200 * w / h), 200)), cv2.COLOR_BGR2RGB), caption=f"Image {i}")
                if abs((w / h) - 9 / 16) > 0.08:
                    st.caption(f"{w}x{h}: will be cropped to 9:16")
    st.markdown("Story characters keep the same face and clothes in every shot. Give each one up to four "
                "reference images (front view first). Story characters never speak.")
    n_chars = st.number_input("How many story characters", 0, 6, value=1)
    story_chars = []
    for k in range(int(n_chars)):
        a, b = st.columns([1, 3])
        name = a.text_input(f"Name {k + 1}", key=f"cn{k}")
        refs = b.file_uploader(f"Reference images for {name or 'character'}", type=["png", "jpg", "jpeg", "webp"],
                               accept_multiple_files=True, key=f"cr{k}")
        if name and refs:
            story_chars.append({"name": name, "images": [_save(r, inputs / "characters" / name, r.name) for r in refs[:4]]})

# --------------------------------------------------------------------------- 3 script
with st.expander("3. Scene script", expanded=True):
    st.markdown(f"Hosts who can speak: **{', '.join(hosts)}**. "
                f"{profile.get('style_rules', [''])[0]}")
    text = st.text_area("Scenes", value=st.session_state.get("script", TEMPLATE), height=360)
    st.session_state["script"] = text
    if st.button("Check script"):
        try:
            segs = P.parse_scene_text(text, hosts)
            lines = [{"speaker": l["speaker"], "text": l["text"]} for s in segs for l in s["lines"]]
            qa = check_script({"lines": lines}, profile)
            for c in qa["checks"]:
                (st.success if c["passed"] else st.warning)(f"{c['rule']}: {c['detail']}")
        except P.PlanError as e:
            st.error(str(e))

# --------------------------------------------------------------------------- 4 hosts
with st.expander("4. Host scene and TV", expanded=True):
    rig = _resolve_rig(pname)
    mode = st.radio("Host scene", ["Profile rig" if rig else "Upload", "Upload"], horizontal=True)
    if mode == "Upload":
        bg = st.file_uploader("Set with the TV (image or short looping video)", type=["png", "jpg", "jpeg", "mp4", "mov"])
        rig = None
        if bg:
            rig = {"background": _save(bg, inputs / "host", bg.name), "hosts": {}}
            st.caption("Optional: each host's transparent cut-outs. Without them the hosts are part of the set and do not move.")
            cols = st.columns(len(hosts))
            for h, col in zip(hosts, cols):
                with col:
                    files = {}
                    for state in ("closed", "open", "blink"):
                        u = st.file_uploader(f"{h} {state}", type=["png"], key=f"{h}{state}")
                        if u:
                            files[state] = _save(u, inputs / "host", f"{h.lower()}_{state}.png")
                    if "closed" in files:
                        x = st.slider(f"{h} position", 0.0, 1.0, 0.3 + 0.2 * hosts.index(h), key=f"{h}x")
                        rig["hosts"][h] = {**files, "x": x, "y": 0.93, "height": 0.275}
    if rig:
        bgimg = cv2.imread(rig["background"])
        if bgimg is None:
            cap = cv2.VideoCapture(rig["background"])
            _, bgimg = cap.read()
        manual = st.text_input("TV screen corners (leave empty to detect): x1,y1,x2,y2,x3,y3,x4,y4", "")
        corners = None
        if manual.strip():
            v = [float(t) for t in re.findall(r"-?\d+\.?\d*", manual)]
            if len(v) == 8:
                corners = [v[i:i + 2] for i in range(0, 8, 2)]
                rig["screen_corners"] = corners
        scr = detect_screen(bgimg, corners)
        if scr is None:
            st.error("No TV screen found. Make the screen flat green or flat cream, or type its four corners.")
        else:
            prev = bgimg.copy()
            cv2.polylines(prev, [scr.corners.astype(np.int32)], True, (0, 0, 255), 6)
            h, w = prev.shape[:2]
            st.image(cv2.cvtColor(cv2.resize(prev, (int(420 * w / h), 420)), cv2.COLOR_BGR2RGB),
                     caption=f"Screen found by {scr.method}: red outline")
        st.session_state["rig"] = rig

# --------------------------------------------------------------------------- 5 voices
VOICE_HELP = {
    "edge": "Free. Microsoft neural voices: the voices your episodes already use. Needs internet.",
    "kokoro": "Free. Kokoro-82M, runs offline on any computer. Downloads 350 MB once.",
    "chatterbox": "Free. Clones each host from your own 10-30 s sample. Needs: pip install chatterbox-tts",
    "edge": "Free. Microsoft Edge neural voices (the Luvvoice voices). Needs internet.",
    "elevenlabs": "Paid service. Optional.",
}
with st.expander("5. Host voices", expanded=True):
    from reelwise.studio.voices import KOKORO_SHORTLIST, engine_from_config
    voices = {}
    first_lines = {}
    try:
        for s_ in P.parse_scene_text(st.session_state.get("script", TEMPLATE), hosts):
            for l in s_["lines"]:
                first_lines.setdefault(l["speaker"], l["say"])
    except P.PlanError:
        pass
    for c in profile["characters"]:
        h = c["name"]
        default = c.get("voice", {"engine": "edge", "voice": "en-GB-RyanNeural"})
        engines = ["edge", "chatterbox", "kokoro", "elevenlabs"]
        a, b, d, e = st.columns([1, 1.3, 1, 1])
        eng = a.selectbox(f"{h} voice engine", engines, index=engines.index(default.get("engine", "kokoro"))
                          if default.get("engine") in engines else 0, key=f"eng{h}",
                          help=" ".join(VOICE_HELP[k] for k in ("edge", "chatterbox", "kokoro")))
        if eng == "kokoro":
            opts = KOKORO_SHORTLIST
            v = c.get("voice_kokoro", {}).get("voice", "bm_george")
            voice = b.selectbox(f"{h} voice", opts, index=opts.index(v) if v in opts else 0, key=f"kv{h}")
            speed = d.slider(f"{h} speed", 0.8, 1.25, float(c.get("voice_kokoro", {}).get("speed", 1.0)), 0.01, key=f"ks{h}")
            lang = "en-gb" if voice.startswith("b") else "en-us"
            voices[h] = {"engine": "kokoro", "voice": voice, "speed": speed, "lang": lang}
        elif eng == "chatterbox":
            sample = b.file_uploader(f"{h} voice sample", type=["wav", "mp3", "m4a"], key=f"cbs{h}")
            known = inputs / "voices" / f"{h.lower()}_sample.wav"
            path = _save(sample, inputs / "voices", f"{h.lower()}_sample{Path(sample.name).suffix}") if sample else \
                (str(known) if known.exists() else "")
            ex = d.slider(f"{h} emotion", 0.25, 1.0, 0.5, 0.05, key=f"cbe{h}")
            voices[h] = {"engine": "chatterbox", "sample": path, "exaggeration": ex}
            if not path:
                b.caption("Upload a sample, or cut one from an old episode below.")
        elif eng == "edge":
            ev = default if default.get("engine") == "edge" else {"voice": "en-GB-RyanNeural", "rate": "+0%", "pitch": "+0Hz"}
            cands = list(dict.fromkeys([ev["voice"]] + profile.get("voice_candidates", {}).get(h, [])))
            voice = b.selectbox(f"{h} voice", cands + ["other"], key=f"v{h}")
            if voice == "other":
                voice = b.text_input(f"{h} Edge voice name", "en-GB-RyanNeural", key=f"vo{h}")
            rate = d.text_input(f"{h} rate", ev.get("rate", "+0%"), key=f"r{h}")
            voices[h] = {"engine": "edge", "voice": voice, "rate": rate, "pitch": ev.get("pitch", "+0Hz")}
        else:
            vid = b.text_input(f"{h} ElevenLabs voice id", "", key=f"vidin{h}")
            voices[h] = {"engine": "elevenlabs", "voice_id": vid, "speed": 1.0}
        if e.button(f"Hear {h}", key=f"aud{h}"):
            try:
                line = first_lines.get(h, f"Hi, I am {h}, and this one is worth telling.")
                with st.spinner(f"Voicing {h} (first use downloads the model)"):
                    out = proj / "audio" / "audition"
                    out.mkdir(parents=True, exist_ok=True)
                    eng_ = engine_from_config(voices[h])
                    sp = eng_.speak(line, out / f"{h}_{eng_.key(line)}.wav")
                e.audio(sp.path)
            except Exception as ex_:
                e.error(str(ex_)[:200])
    with st.expander("Cut voice samples from an old episode"):
        old = st.file_uploader("Old episode", type=["mp4", "mov", "wav", "mp3"], key="old_ep")
        if old:
            op = _save(old, inputs / "voices", "old_episode" + Path(old.name).suffix)
            utt = split_utterances(op)
            st.caption(f"{len(utt)} lines found. Tick who is speaking in each; leave the rest as skip.")
            cap = cv2.VideoCapture(op)
            labels = []
            for i, (a_, b_) in enumerate(utt[:60]):
                c_a, c_b, c_c = st.columns([1, 2, 1])
                cap.set(cv2.CAP_PROP_POS_MSEC, (a_ + b_) / 2 * 1000)
                ok, fr = cap.read()
                if ok:
                    c_a.image(cv2.cvtColor(cv2.resize(fr, (90, 160)), cv2.COLOR_BGR2RGB))
                c_b.audio(op, start_time=int(a_), end_time=int(b_) + 1)
                who = c_c.selectbox(f"{a_:.1f}-{b_:.1f}s", ["skip"] + hosts, key=f"u{i}")
                labels.append((a_, b_, "" if who == "skip" else who))
            if st.button("Save samples"):
                files = build_samples(op, labels, inputs / "voices")
                st.success("Saved " + ", ".join(f"{k}: {Path(v).name}" for k, v in files.items())
                           + ". Choose chatterbox for that host to use it.")
    music = st.file_uploader("Music bed (optional; YouTube Audio Library tracks are free and claim-safe)",
                             type=["mp3", "wav", "m4a"])

# --------------------------------------------------------------------------- 6 settings and plan
from reelwise.studio.providers import FREE_MODELS, PAID_MODELS  # noqa: E402

MODEL_HELP = {
    "motion": "Free, runs on your computer. Real 3D camera moves through each image using depth.",
    "motion+hf": "Free. Same as motion, plus AI-animated hero shots from a free Hugging Face Space "
                 "(free account, about 5 GPU minutes a day). Falls back to motion when the quota is used.",
    "draft": "Free and fastest. Flat camera moves, for checking timing.",
}
with st.expander("6. Plan", expanded=True):
    a, b = st.columns([1, 2])
    model = a.selectbox("Motion", FREE_MODELS, format_func=lambda m: {"motion": "Free 3D motion",
                        "motion+hf": "Free 3D motion + free AI hero shots", "draft": "Draft"}[m])
    b.caption(MODEL_HELP[model])
    if model == "motion+hf" and not os.environ.get("HF_TOKEN"):
        b.info("Add a free Hugging Face token as HF_TOKEN in .env (huggingface.co, Settings, Access Tokens) "
               "so the daily free quota is yours.")
    LAYOUT_LABELS = {"vertical": "Phone, vertical 9:16 (fills the whole screen)",
                     "tv": "TV, 16:9 (plays in the middle of the feed, like a TV screen)"}
    layout = st.radio("How should this episode be shaped? Choose before anything is made.",
                      list(LAYOUT_LABELS), index=None, format_func=LAYOUT_LABELS.get, key="layout")
    if layout == "tv":
        st.caption("16:9 at 1920x1080. Story panels are shown whole, and at the sign-off the rooftop set "
                   "sits in the middle of the frame with a soft blurred copy either side.")
    elif layout == "vertical":
        st.caption("9:16 at 1080x1920, filling a phone screen.")
    REVEALS = {"end": "Only at the sign-off: the story fills the screen, and the camera pulls out to the "
                      "hosts on \"And that's one worth telling\" (last 2 to 3 seconds)",
               "classic": "At the start and the end: pull out to the hosts for the intro, push back in "
                          "for the story, pull out again for the sign-off"}
    reveal = st.radio("When are the hosts seen?", list(REVEALS), format_func=REVEALS.get,
                      index=list(REVEALS).index(profile.get("host_reveal", "end")))
    zoom = st.slider("Pull out of the TV (seconds)", 0.6, 1.6, 1.0, 0.1)
    with st.expander("Optional paid extras (off by default)"):
        st.caption("Nothing here is needed. These use paid API keys if you ever add them.")
        paid_model = st.selectbox("Paid video model", ["none"] + PAID_MODELS, disabled=not HAS_FAL)
        if paid_model != "none":
            model = paid_model
        judge = st.checkbox("Claude checks every clip", value=False, disabled=not HAS_CLAUDE)
        use_claude = st.checkbox("Claude plans the shots", value=False, disabled=not HAS_CLAUDE)
    attempts = 2
    if layout is None:
        st.info("Choose the shape (phone or TV) to build the plan.")
    if st.button("Build plan", type="primary", disabled=layout is None):
        try:
            if not images:
                raise P.PlanError("upload at least one story image")
            segs = P.parse_scene_text(st.session_state["script"], hosts)
            plan = P.new_plan(segs, {**profile, "host_reveal": reveal}, images, story_chars,
                              st.session_state.get("ref_report"), title=project, layout=layout)
            studio = Studio(proj)
            with st.spinner("Narrating every line and planning shots (first run downloads the voice model)"):
                studio.prepare(plan, voices, zoom_s=zoom, use_claude=use_claude)
            st.session_state["plan"] = plan
            st.session_state["model"] = model
        except Exception as e:
            st.error(str(e))
    plan = st.session_state.get("plan")
    if plan:
        rows = []
        for s in plan["segments"]:
            for k, sh in enumerate(s.get("shots", [])):
                rows.append({"segment": s["id"], "scene": s["label"], "image": s["image"], "shot": k + 1,
                             "seconds": round(sh["duration_s"], 2), "prompt": sh["prompt"]})
        st.caption(f"Total {P.total_duration(plan):.1f}s plus end card. Shot words steer the camera "
                   "(close-up, track, top-down, push, reveal). Edit any shot before making the episode.")
        edited = st.data_editor(pd.DataFrame(rows), disabled=["segment", "scene", "image", "shot", "seconds"],
                                width="stretch", hide_index=True, key="shots_editor")
        for _, r in edited.iterrows():
            seg = next(s for s in plan["segments"] if s["id"] == r["segment"])
            seg["shots"][int(r["shot"]) - 1]["prompt"] = r["prompt"]
            seg["shots_locked"] = True
        narr = [(l["speaker"], l["duration_s"]) for s in plan["segments"] for l in s["lines"]]
        st.markdown("**Narration:** " + ", ".join(f"{h} {sum(d for sp, d in narr if sp == h):.1f}s" for h in hosts))
        cols = st.columns(min(4, max(1, len(narr))))
        k = 0
        for s in plan["segments"]:
            for l in s["lines"]:
                with cols[k % len(cols)]:
                    st.caption(f"{s['label']}: {l['speaker']}")
                    st.audio(l["audio"])
                k += 1
        if model in PAID_MODELS:
            jobs = P.generation_jobs(plan, MODELS[model]["durations"])
            cost = P.estimate_cost(jobs, MODELS[model]["price_per_s"])
            st.warning(f"Paid model selected: about ${cost['first_pass_usd']:.2f}.")
        else:
            st.success("Cost: $0. Everything runs free.")

# --------------------------------------------------------------------------- 7 make it
with st.expander("7. Generate, check and export", expanded=True):
    plan = st.session_state.get("plan")
    if not plan:
        st.info("Build the plan first.")
    elif st.button("Make the episode", type="primary"):
        studio = Studio(proj)
        log = st.empty()
        bar = st.progress(0.0)
        events: list[str] = []
        studio.log = lambda m: (events.append(f"{time.strftime('%H:%M:%S')}  {m}"), log.code("\n".join(events[-12:])))
        try:
            mpath = _save(music, inputs, "music" + Path(music.name).suffix) if music else None
            clips = studio.generate(plan, st.session_state.get("model", "motion"), max_attempts=int(attempts), judge=judge,
                                    progress=lambda i, n, _: bar.progress(i / n * 0.7))
            rep = studio.assemble(plan, clips, st.session_state.get("rig"), music=mpath, judge=judge,
                                  progress=lambda i, n, _: bar.progress(0.7 + 0.3 * i / n))
            st.session_state["report"] = rep
        except Exception as e:
            st.error(f"Stopped: {e}")
        bar.progress(1.0)
    rep = st.session_state.get("report")
    if rep:
        df = pd.DataFrame(rep["final"])[["rule", "passed", "severity", "detail"]]
        st.dataframe(df, width="stretch", hide_index=True)
        final = Path(rep["file"])
        if final.exists():
            st.video(str(final))
            unlocked = rep["passed"]
            if not unlocked:
                st.error("Not ready: fix the failed checks above, or regenerate the clips they name.")
                unlocked = st.checkbox("I have watched it and accept the failed checks")
            if unlocked:
                st.download_button("Download episode (MP4)", final.read_bytes(), file_name=f"{project}.mp4",
                                   mime="video/mp4", type="primary")
                srt = final.with_suffix(".srt")
                if srt.exists():
                    st.download_button("Download subtitles (SRT)", srt.read_text(), file_name=f"{project}.srt")
