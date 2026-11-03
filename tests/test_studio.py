import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reelwise.studio import plan as P
from reelwise.studio.assemble import build_ass, build_timeline, chunk_words, fit_stamp, shot_sources
from reelwise.studio.host import mouth_track
from reelwise.studio.providers import FalProvider, build_prompt, integer_split
from reelwise.studio.tv import TVCompositor, detect_screen, order_corners
from reelwise.studio.voices import SR, MockEngine, chars_to_words, time_plan

ROOT = Path(__file__).resolve().parents[1]
PROFILE = json.loads((ROOT / "profiles" / "worth_telling.json").read_text())
HOSTS = ["Ren", "Bex", "Ozzy"]

SCRIPT = """COLD OPEN
Image: 2
Stamp: 738 days
Bex: Would you live up a tree for two whole years?

INTRO
Ren: This is a true story from California.

SCENE 1
Image: 1
Characters: Julia
Action: A woman climbs a giant redwood at night,
rope over her shoulder.
Shots:
- extreme close-up of her hands on the rope
- wide shot from below
Ren: In nineteen ninety seven Julia Hill climbed a redwood. | In nineteen ninety seven Julia Hill climbed a redwood.
Bex: And she did not come back down for a very long time.

SCENE 2
Action: Loggers wait at the base of the tree.
Ozzy: She won.

OUTRO
Ren: That is one worth telling.
"""


def test_parse_scene_text():
    segs = P.parse_scene_text(SCRIPT, HOSTS)
    assert [s["kind"] for s in segs] == ["cold_open", "intro", "story", "story", "outro"]
    s1 = segs[2]
    assert s1["image"] == 1 and s1["characters"] == ["Julia"]
    assert s1["action"].endswith("rope over her shoulder.")
    assert [sh["prompt"] for sh in s1["shots"]][0].startswith("extreme close-up")
    assert s1["lines"][0]["speaker"] == "Ren"
    assert segs[0]["stamp"] == "738 days"


@pytest.mark.parametrize("bad, msg", [
    ("SCENE 1\nAction: x\nJulia: hello there", "not one of your hosts"),
    ("Ren: hi", "before the first"),
    ("INTRO\nRen: hi", "no SCENE"),
    ("SCENE 1\nAction: x", "no narration"),
    ("SCENE 1\nImage: one\nAction: x\nRen: hi", "needs a number"),
])
def test_parse_errors_are_readable(bad, msg):
    with pytest.raises(P.PlanError, match=msg):
        P.parse_scene_text(bad, HOSTS)


@pytest.fixture
def images(tmp_path):
    out = []
    for k in (1, 2):
        img = np.zeros((1280, 720, 3), np.uint8)
        img[:] = (40 * k, 90, 160)
        cv2.circle(img, (360, 600), 150, (230, 230, 230), -1)
        p = tmp_path / f"img{k}.png"
        cv2.imwrite(str(p), img)
        out.append({"id": k, "path": str(p)})
    return out


def _plan(images, tmp_path):
    chars = [{"name": "Julia", "images": [images[0]["path"]]}]
    plan = P.new_plan(P.parse_scene_text(SCRIPT, HOSTS), PROFILE, images, chars)
    eng = {h: MockEngine(3.0) for h in HOSTS}
    time_plan(plan, eng, tmp_path / "audio", zoom_s=1.0)
    P.assign_missing_images(plan)
    P.fill_shots(plan)
    return plan


def test_new_plan_rejects_unknown_image_and_character(images):
    segs = P.parse_scene_text(SCRIPT.replace("Image: 1", "Image: 9"), HOSTS)
    with pytest.raises(P.PlanError, match="Image 9"):
        P.new_plan(segs, PROFILE, images, [{"name": "Julia", "images": ["x"]}])
    with pytest.raises(P.PlanError, match="Julia"):
        P.new_plan(P.parse_scene_text(SCRIPT, HOSTS), PROFILE, images, [])


def test_timing_follows_narration(images, tmp_path):
    plan = _plan(images, tmp_path)
    first = plan["segments"][0]["lines"][0]
    assert first["start_s"] == PROFILE["first_line"]["start_s"]
    for s in plan["segments"]:
        for ln in s["lines"]:
            assert ln["start_s"] + ln["duration_s"] <= s["duration_s"] + 1e-6
        if s["kind"] == "story":
            assert abs(sum(sh["duration_s"] for sh in s["shots"]) - s["duration_s"]) < 0.01
    # captions keep the real spelling even when the voice gets a phonetic version
    assert plan["segments"][2]["lines"][0]["words"][0]["w"] == "In"
    assert plan["segments"][3]["image"] in (1, 2)  # auto-assigned


def test_generation_jobs_cover_need_and_rolls(images, tmp_path):
    plan = _plan(images, tmp_path)
    plan["segments"][2]["pre_roll_s"] = 1.0
    jobs = P.generation_jobs(plan, list(range(3, 16)))
    j = next(j for j in jobs if j["segment"] == plan["segments"][2]["id"])
    assert j["lengths"][0] == pytest.approx(plan["segments"][2]["shots"][0]["duration_s"] + 1.0, abs=1e-3)
    assert j["duration"] >= j["need_s"]
    cost = P.estimate_cost(jobs, 0.112)
    assert cost["first_pass_usd"] == pytest.approx(sum(j["duration"] for j in jobs) * 0.112, abs=0.01)


def test_long_scene_is_split_across_requests():
    plan = {"segments": [{"kind": "story", "id": "s", "image": 1, "characters": [],
                          "shots": [{"prompt": "a", "duration_s": 6.0}, {"prompt": "b", "duration_s": 6.0},
                                    {"prompt": "c", "duration_s": 6.0}]}]}
    jobs = P.generation_jobs(plan, list(range(3, 16)))
    assert len(jobs) == 2
    assert all(j["duration"] <= 15 for j in jobs)


def test_integer_split():
    assert integer_split([1.2, 2.5, 0.8], 5) == [1, 3, 1] or sum(integer_split([1.2, 2.5, 0.8], 5)) == 5
    assert all(x >= 1 for x in integer_split([0.1, 0.1, 5], 6))
    assert sum(integer_split([0.1, 0.1, 5], 6)) == 6


def test_prompt_tags_elements_and_bans_talking():
    p = build_prompt([{"prompt": "Julia climbs", "duration_s": 2}], ["Julia"])
    assert "@Element1 (Julia)" in p and "do not speak" in p


class FakeFal:
    def __init__(self):
        self.calls = []

    def upload_file(self, path):
        return f"https://fal.test/{Path(path).name}"

    def subscribe(self, endpoint, arguments, with_logs=False):
        self.calls.append((endpoint, arguments))
        return {"video": {"url": "https://fal.test/out.mp4"}}


def test_fal_arguments_and_call(images, tmp_path):
    plan = _plan(images, tmp_path)
    plan["segments"][2]["shots"] = [{"prompt": "Julia grips the rope", "duration_s": 2.0},
                                    {"prompt": "wide shot of Julia", "duration_s": 3.0}]
    job = {"job_id": "j", "segment": "x", "image": 1, "characters": ["Julia"],
           "shots": plan["segments"][2]["shots"], "lengths": [2.0, 3.0], "duration": 6, "need_s": 5.6}
    fake = FakeFal()
    got = {}
    prov = FalProvider("kling-v3-pro", client=fake, downloader=lambda u, o: got.setdefault("url", u),
                       cache_path=tmp_path / "up.json")
    meta = prov.generate(job, plan, tmp_path / "c.mp4")
    endpoint, args = fake.calls[0]
    assert endpoint == "fal-ai/kling-video/v3/pro/image-to-video"
    assert args["start_image_url"].endswith("img1.png")
    assert args["elements"][0]["frontal_image_url"].endswith("img1.png")
    assert [m["duration"] for m in args["multi_prompt"]] == ["2", "4"] or \
        sum(int(m["duration"]) for m in args["multi_prompt"]) == 6
    assert "@Element1" in args["multi_prompt"][0]["prompt"]
    assert args["generate_audio"] is False and args["duration"] == "6"
    assert got["url"] == "https://fal.test/out.mp4" and meta["model"] == "kling-v3-pro"


def test_elevenlabs_alignment_to_words():
    al = {"characters": list("Hi you"), "character_start_times_seconds": [0, .1, .2, .3, .4, .5],
          "character_end_times_seconds": [.1, .2, .3, .4, .5, .6]}
    assert chars_to_words(al) == [{"w": "Hi", "start": 0, "end": .2}, {"w": "you", "start": .3, "end": .6}]


def test_mouth_track_follows_audio():
    t = np.arange(SR) / SR
    x = np.where((t > 0.3) & (t < 0.6), 0.3 * np.sin(2 * np.pi * 200 * t), 0).astype(np.float32)
    m = mouth_track(x, 30)
    assert not m[:8].any() and m[10:17].all() and not m[20:].any()


def _host_with_green(angle=0):
    host = np.full((640, 360, 3), (70, 60, 50), np.uint8)
    pts = np.array([[60, 160], [300, 160 + angle], [300, 300], [60, 300]])
    cv2.fillConvexPoly(host, pts, (0, 255, 0))
    return host


def test_green_screen_detected_exactly():
    s = detect_screen(_host_with_green())
    assert s.method == "green"
    assert np.allclose(s.corners, [[60, 160], [300, 160], [300, 300], [60, 300]], atol=2)


def test_manual_corners_are_ordered():
    c = order_corners([[300, 300], [60, 160], [60, 300], [300, 160]])
    assert c.tolist() == [[60, 160], [300, 160], [300, 300], [60, 300]]


def test_push_through_ends_on_the_story_and_leaves_no_green():
    host = _host_with_green()
    story = np.zeros((640, 360, 3), np.uint8)
    story[:, :, 2] = np.linspace(0, 255, 360)[None, :]
    tv = TVCompositor((360, 640), detect_screen(host), (360, 640))
    start = tv.render(host, story, 0.0)
    end = tv.render(host, story, 1.0)
    assert np.abs(end.astype(int) - story.astype(int)).mean() < 2
    g = start.astype(int)
    assert ((g[..., 1] > g[..., 0] + 60) & (g[..., 1] > g[..., 2] + 60)).sum() == 0
    mid = tv.render(host, story, 0.5)
    assert mid.shape == (640, 360, 3)


def test_shot_sources_use_model_cuts_or_proportions():
    job = {"shots": [{"duration_s": 1.0}, {"duration_s": 2.0}], "lengths": [1.0, 2.0]}
    by_cut = shot_sources(job, {"duration_s": 3.6, "cuts_s": [1.25]})
    assert by_cut[0][1] == 1.25 and by_cut[1] == (1.25, 3.6)
    by_prop = shot_sources(job, {"duration_s": 3.6, "cuts_s": []})
    assert by_prop[1][0] == pytest.approx(1.2)


def test_captions_and_stamp_fit(images, tmp_path):
    plan = _plan(images, tmp_path)
    plan["segments"][0]["stamp"] = "A VERY LONG STAMP THAT WOULD NEVER FIT ON ONE LINE"
    tl = build_timeline(plan, 1.0, 2.0)
    rep = build_ass(tl, PROFILE, tmp_path / "c.ass")
    assert rep and all(r["fits"] for r in rep)
    text = (tmp_path / "c.ass").read_text()
    assert "\\N" in text and "@watchworthtelling" in text
    lines, size = fit_stamp("738 DAYS UP A TREE", 160, 780)
    assert all(len(l) * size * 0.66 <= 780 for l in lines)


def test_chunk_words_breaks_on_punctuation():
    ws = [{"w": w, "t0": i, "t1": i + 1} for i, w in enumerate("Wait, what happened next?".split())]
    ch = chunk_words(ws, max_words=3, max_chars=20)
    assert [" ".join(x["w"] for x in c) for c in ch][0] == "Wait,"


class FakeClaude:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)

        class B:
            text = self.reply

        class M:
            content = [B()]
        return M()


def test_claude_planner_fills_images_and_shots(images, tmp_path):
    plan = _plan(images, tmp_path)
    sid = plan["segments"][3]["id"]
    plan["segments"][3]["image"] = None
    reply = json.dumps({"segments": [{"segment": sid, "image": 2, "shots": [
        {"prompt": "insert of boots on the ladder", "weight": 1}, {"prompt": "top-down wide", "weight": 2}]}]})
    fake = FakeClaude(reply)
    P.claude_shots(plan, {1: "", 2: ""}, client=fake)
    seg = plan["segments"][3]
    assert seg["image"] == 2 and len(seg["shots"]) == 2
    assert seg["shots"][1]["duration_s"] == pytest.approx(2 * seg["shots"][0]["duration_s"], rel=0.01)


def test_judge_failures_block_the_gate(tmp_path, images):
    from reelwise.studio import qc
    clip = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (720, 1280))
    for i in range(60):
        f = np.full((1280, 720, 3), 90, np.uint8)
        cv2.circle(f, (100 + i * 2, 300), 40, (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    plan = {"images": images, "story_characters": []}
    job = {"image": 1, "characters": [], "shots": [{"prompt": "a ball rolls", "duration_s": 2}], "need_s": 1.8}
    bad = {"identity": 2, "style": 5, "action": 4, "artifacts": ["melting face"], "talking": True,
           "text_or_logo": False, "verdict": "fail", "fix": "keep the face stable"}
    v = qc.judge_clip(str(clip), job, plan, client=FakeClaude(json.dumps(bad)))
    checks = qc.judge_checks(v)
    assert not qc.gate(checks)
    assert {c["rule"] for c in checks if not c["passed"]} >= {"same character as reference", "story characters silent"}
    measured, meta = qc.clip_checks(clip, job, (1080, 1920))
    assert qc.gate(measured) and meta["duration_s"] == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("layout,size", [("vertical", (1080, 1920)), ("tv", (1920, 1080))])
def test_layout_sets_the_frame(images, layout, size):
    segs = P.parse_scene_text(SCRIPT, HOSTS)
    plan = P.new_plan(segs, PROFILE, images, [{"name": "Julia", "images": []}], layout=layout)
    assert (plan["format"]["width"], plan["format"]["height"]) == size
    assert plan["format"]["layout"] == layout
    with pytest.raises(P.PlanError):
        P.new_plan(segs, PROFILE, images, [{"name": "Julia", "images": []}], layout="square")


def test_hosts_seen_only_at_the_sign_off(images, tmp_path):
    """Default reveal: the intro stays on the story, the pull-out starts just before the sign-off."""
    plan = _plan(images, tmp_path)
    assert plan["format"]["host_reveal"] == "end"
    tl = build_timeline(plan, 1.0, 2.0)
    assert tl["reveal"] == "end"
    intro = next(s for s in tl["segments"] if s["kind"] == "intro")
    outro = next(s for s in tl["segments"] if s["kind"] == "outro")
    assert all(ln["view"] == "story" for ln in intro["lines"])
    sign_off = outro["lines"][-1]
    assert sign_off["view"] == "host"
    start = outro["t0"] + outro["reveal_f"] / tl["fps"]
    assert sign_off["t0"] - 0.4 <= start <= sign_off["t0"] + 1e-6
    # The hosts are on screen only for the last few seconds, end card included.
    assert tl["frames"] / tl["fps"] - start <= 6.0


def test_classic_reveal_keeps_the_couch_intro(images, tmp_path):
    segs = P.parse_scene_text(SCRIPT, HOSTS)
    plan = P.new_plan(segs, {**PROFILE, "host_reveal": "classic"}, images, [{"name": "Julia", "images": []}])
    time_plan(plan, {h: MockEngine() for h in HOSTS}, tmp_path)
    tl = build_timeline(plan, 1.0, 2.0)
    intro = next(s for s in tl["segments"] if s["kind"] == "intro")
    outro = next(s for s in tl["segments"] if s["kind"] == "outro")
    assert intro["lines"][0]["view"] == "host" and outro["reveal_f"] == 0
