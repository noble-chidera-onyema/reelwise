import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reelwise.studio.voices import CACHE, SR, align_by_pauses, read_wav

ROOT = Path(__file__).resolve().parents[1]
HAVE_KOKORO = (CACHE / "kokoro" / "kokoro-v1.0.onnx").exists()
HAVE_MIDAS = (CACHE / "midas" / "model-small.onnx").exists()


def test_align_by_pauses_finds_the_gap():
    t = np.arange(int(SR * 2.0)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 180 * t)
    x = tone.copy()
    x[int(0.8 * SR):int(1.0 * SR)] = 0          # a pause after the first two words
    words = align_by_pauses(x.astype(np.float32), "hello there, friend")
    assert [w["w"] for w in words] == ["hello", "there,", "friend"]
    assert words[1]["end"] == pytest.approx(0.9, abs=0.06)
    assert words[-1]["end"] == pytest.approx(2.0, abs=0.01)


@pytest.mark.skipif(not HAVE_KOKORO, reason="Kokoro model not downloaded")
def test_kokoro_speaks_real_audio(tmp_path):
    from reelwise.studio.voices import KokoroEngine
    sp = KokoroEngine("bm_george", 1.0).speak("This is one worth telling.", tmp_path / "a.wav")
    x = read_wav(sp.path)
    assert 1.0 < sp.duration_s < 4.0
    assert 20 * np.log10(np.sqrt((x ** 2).mean())) > -30     # clearly audible speech
    assert [w["w"] for w in sp.words] == "This is one worth telling.".split()


@pytest.mark.skipif(not HAVE_MIDAS, reason="depth model not downloaded")
def test_motion_engine_moves_with_parallax(tmp_path):
    from reelwise.studio.motion import ParallaxRenderer
    img = np.full((1280, 720, 3), (60, 90, 140), np.uint8)
    for y in range(0, 1280, 40):
        cv2.line(img, (0, y), (720, y), (90, 120, 170), 2)
    cv2.circle(img, (360, 700), 220, (40, 40, 200), -1)
    p = tmp_path / "i.png"
    cv2.imwrite(str(p), img)
    r = ParallaxRenderer(str(p), 270, 480)
    a, b = r.frame("truck_left", 0.0, 0.0), r.frame("truck_left", 1.0, 1.0)
    assert a.shape == (480, 270, 3)
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0     # the camera really moved


class FakeSpace:
    def __init__(self, video):
        self.video = video
        self.calls = []

    def view_api(self, print_info=False, return_format="dict"):
        return {"named_endpoints": {"/generate_video": {"parameters": [
            {"parameter_name": "input_image", "component": "Image"},
            {"parameter_name": "prompt", "component": "Textbox"},
            {"parameter_name": "negative_prompt", "component": "Textbox"},
            {"parameter_name": "duration_seconds", "component": "Slider"},
            {"parameter_name": "seed", "component": "Slider"},
            {"parameter_name": "randomize_seed", "component": "Checkbox"},
            {"parameter_name": "steps", "component": "Slider"}]}}}

    def predict(self, api_name, **kw):
        self.calls.append((api_name, kw))
        return ({"video": self.video}, 42)


def _art(colour, seed=1, w=270, h=480):
    """A textured picture with a figure, closer to real story art than a flat fill."""
    rng = np.random.default_rng(seed)
    base = cv2.resize(rng.integers(0, 255, (12, 7, 3), dtype=np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.addWeighted(base, 0.35, np.full((h, w, 3), colour, np.uint8), 0.65, 0)
    cv2.ellipse(img, (w // 2, h // 2), (60, 120), 0, 0, 360, (230, 230, 230), -1)
    cv2.circle(img, (w // 2, h // 2 - 150), 40, (60, 80, 120), -1)
    return img


def _clip(path, colour, n=60, w=270, h=480, fps=30, seed=1):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    art = _art(colour, seed, w, h)
    for i in range(n):
        M = np.float32([[1.02, 0, -i * 0.2], [0, 1.02, -4]])
        vw.write(cv2.warpAffine(art, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    vw.release()


def _hf_setup(tmp_path, ai_colour):
    from reelwise.studio.providers import HFSpaceProvider, Provider

    img = _art((40, 90, 160))
    ip = tmp_path / "img.png"
    cv2.imwrite(str(ip), img)
    ai = tmp_path / "ai.mp4"
    _clip(ai, ai_colour, n=90, seed=1 if ai_colour == (40, 90, 160) else 9)

    class StubMotion(Provider):
        name = "motion"
        w, h, fps = 270, 480, 30

        def __init__(self):
            pass

        def generate(self, job, plan, out_path, attempt=0, feedback=""):
            _clip(out_path, (40, 90, 160), n=int(job["duration"] * 30))
            return {"provider": "motion"}

    plan = {"images": [{"id": 1, "path": str(ip)}]}
    job = {"job_id": "j", "image": 1, "shots": [{"prompt": "wide", "duration_s": 1.0},
                                                 {"prompt": "hero", "duration_s": 2.0}],
           "lengths": [1.0, 2.0], "duration": 3}
    fake = FakeSpace(str(ai))
    return HFSpaceProvider(StubMotion(), client=fake), fake, plan, job


def test_hf_hero_shot_spliced_with_same_length(tmp_path, monkeypatch):
    import gradio_client
    monkeypatch.setattr(gradio_client, "handle_file", lambda p: {"path": p})
    prov, fake, plan, job = _hf_setup(tmp_path, (40, 90, 160))
    out = tmp_path / "c.mp4"
    meta = prov.generate(job, plan, out)
    name, kw = fake.calls[0]
    assert name == "/generate_video"
    assert kw["input_image"] == {"path": plan["images"][0]["path"]}
    assert "do not speak" in kw["prompt"] and "talking" in kw["negative_prompt"]
    assert kw["randomize_seed"] is False and 2.0 <= kw["duration_seconds"] <= 5.0
    assert "steps" not in kw                       # unknown fields keep the Space's defaults
    assert meta["provider"] == "motion+hf" and meta["hero_shot"] == 2
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 90


def test_hf_shot_rejected_when_it_drifts_from_the_image(tmp_path, monkeypatch):
    import gradio_client
    monkeypatch.setattr(gradio_client, "handle_file", lambda p: {"path": p})
    prov, fake, plan, job = _hf_setup(tmp_path, (200, 30, 220))    # different colours entirely
    meta = prov.generate(job, plan, tmp_path / "c.mp4")
    assert meta["provider"] == "motion" and "drifted" in meta["hf_note"]


def test_hf_falls_back_when_space_fails(tmp_path, monkeypatch):
    import gradio_client
    monkeypatch.setattr(gradio_client, "handle_file", lambda p: {"path": p})
    prov, fake, plan, job = _hf_setup(tmp_path, (40, 90, 160))

    def boom(**kw):
        raise RuntimeError("You have exceeded your GPU quota")
    fake.predict = lambda api_name, **kw: boom(**kw)
    meta = prov.generate(job, plan, tmp_path / "c.mp4")
    assert meta["provider"] == "motion" and "quota" in meta["hf_note"]
    assert (tmp_path / "c.mp4").exists()


def test_rushed_line_is_voiced_again_slower(tmp_path):
    from reelwise.studio import plan as P
    from reelwise.studio.voices import MockEngine, time_plan

    class Rate(MockEngine):
        """Speaks at 6 words a second unless slowed."""
        def __init__(self, wps=6.0):
            super().__init__(wps)

        def slowed(self, factor):
            return Rate(self.wps / factor)

    prof = json.loads((ROOT / "profiles" / "worth_telling.json").read_text())
    segs = P.parse_scene_text("SCENE 1\nAction: x\nRen: And that's one worth telling, my friends.", ["Ren", "Bex", "Ozzy"])
    plan = P.new_plan(segs, prof, [{"id": 1, "path": "x"}], [])
    time_plan(plan, {"Ren": Rate()}, tmp_path)
    ln = plan["segments"][0]["lines"][0]
    assert ln.get("slowed") is True
    assert len(ln["text"].split()) / ln["duration_s"] <= prof.get("max_words_per_second", 3.8) + 1e-6


def test_captions_drawn_inside_safe_area(tmp_path):
    from reelwise.studio.captions import CaptionRenderer
    prof = json.loads((ROOT / "profiles" / "worth_telling.json").read_text())
    tl = {"W": 1080, "H": 1920, "segments": [
        {"kind": "cold_open", "t0": 0.0, "t1": 2.0, "stamp": "A dozen cans of pepper spray on one man",
         "lines": [{"speaker": "Bex", "text": "Wait, they did what?", "t0": 0.3, "t1": 1.5,
                    "words": [{"w": "Wait,", "t0": 0.3, "t1": 0.6}, {"w": "they", "t0": 0.6, "t1": 0.8},
                              {"w": "did", "t0": 0.8, "t1": 1.0}, {"w": "what?", "t0": 1.0, "t1": 1.5}]}]},
        {"kind": "end_card", "t0": 2.0, "t1": 4.0, "stamp": "", "lines": []}]}
    cr = CaptionRenderer(tl, prof)
    assert cr.report and all(r["fits"] for r in cr.report)
    assert {r["kind"] for r in cr.report} == {"caption", "stamp", "end_card"}
    frame = np.zeros((1920, 1080, 3), np.uint8)
    out = cr.draw(frame.copy(), 0.5)
    yellow = (out[..., 2] > 200) & (out[..., 1] > 170) & (out[..., 0] < 60)      # stamp
    teal = (out[..., 1] > 130) & (out[..., 0] > 130) & (out[..., 2] < 60)        # Bex's caption
    assert yellow.sum() > 2000 and teal.sum() > 500
    cols = np.flatnonzero(out.any(axis=(0, 2)))
    assert cols.min() >= 0.08 * 1080 - 40 and cols.max() <= 0.82 * 1080 + 40


def test_landscape_panel_stays_whole_in_every_move(tmp_path):
    """A label 5% in from the panel edge must stay in frame whatever the camera does."""
    pytest.importorskip("cv2")
    if not (CACHE / "midas" / "model-small.onnx").exists():
        pytest.skip("depth model not downloaded")
    from reelwise.studio.motion import MOVES, ParallaxRenderer
    img = np.full((600, 1000, 3), 200, np.uint8)
    img[30:70, 50:170] = (0, 0, 255)          # red label, top left, 5% in
    img[530:570, 830:950] = (255, 0, 0)       # blue figure, bottom right, 5% in
    f = tmp_path / "panel.png"
    cv2.imwrite(str(f), img)
    r = ParallaxRenderer(str(f), 540, 960)
    assert r.fit is not None
    for mv in MOVES:
        for u in (0.0, 0.5, 1.0):
            fr = r.frame(mv, u, u * 3)
            red = (fr[..., 2] > 180) & (fr[..., 0] < 80) & (fr[..., 1] < 80)
            blue = (fr[..., 0] > 180) & (fr[..., 2] < 80) & (fr[..., 1] < 80)
            # each marker is 120x40 px in a 1000 px wide source shown at 540 px: about 64x21
            assert red.sum() > 0.8 * 64 * 21, (mv, u, "label cut")
            assert blue.sum() > 0.8 * 64 * 21, (mv, u, "figure cut")


def test_panel_in_tv_frame_is_shown_whole(tmp_path):
    """A 4:3 panel in a 16:9 frame becomes a column down the middle, never cropped top or bottom."""
    if not (CACHE / "midas" / "model-small.onnx").exists():
        pytest.skip("depth model not downloaded")
    from reelwise.studio.motion import MOVES, ParallaxRenderer
    img = np.full((600, 800, 3), 200, np.uint8)
    img[30:70, 40:160] = (0, 0, 255)          # label, top left, 5% in
    img[530:570, 640:760] = (255, 0, 0)       # figure, bottom right, 5% in
    f = tmp_path / "panel43.png"
    cv2.imwrite(str(f), img)
    r = ParallaxRenderer(str(f), 960, 540)
    assert r.fit is not None and r.fit[1] > 0 and r.fit[2] == 0
    for mv in MOVES:
        for u in (0.0, 1.0):
            fr = r.frame(mv, u, u * 3)
            assert fr.shape == (540, 960, 3)
            red = (fr[..., 2] > 180) & (fr[..., 0] < 80) & (fr[..., 1] < 80)
            blue = (fr[..., 0] > 180) & (fr[..., 2] < 80) & (fr[..., 1] < 80)
            assert red.sum() > 0.8 * 81 * 27 and blue.sum() > 0.8 * 81 * 27, mv


def test_chatterbox_worker_replies_survive_library_chatter(tmp_path):
    """Libraries print to stdout (e.g. 'loaded PerthNet ...'); the app must read past it."""
    import subprocess
    import sys
    from reelwise.studio.voices import ChatterboxEngine
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import sys, json\n"
        "print('loaded PerthNet (Implicit) at step 250,000', flush=True)\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    print('Sampling: 10%', flush=True)\n"
        "    print(json.dumps({'ok': True, 'out': json.loads(line)['out']}), flush=True)\n")
    p = subprocess.Popen([sys.executable, str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    try:
        assert ChatterboxEngine._reply(p) == {"ready": True}
        p.stdin.write(json.dumps({"out": "x.wav"}) + "\n")
        p.stdin.flush()
        assert ChatterboxEngine._reply(p) == {"ok": True, "out": "x.wav"}
    finally:
        p.kill()


def test_drifted_line_is_voiced_again_in_the_right_voice(tmp_path, monkeypatch):
    """A take that does not match the host's reference read is voiced again, steadier."""
    from reelwise.studio import voiceprint
    from reelwise.studio.voices import ChatterboxEngine, write_wav
    ref = tmp_path / "ref.wav"
    write_wav(ref, np.zeros(SR, np.float32))
    eng = ChatterboxEngine(str(ref), reference=str(ref), temperature=0.7)
    temps = []

    def fake_take(text, out, t):
        temps.append(t)
        x = (0.1 * np.sin(np.arange(SR) / 20)).astype(np.float32)
        write_wav(out, x)
        return x

    scores = iter([0.55, 0.81])
    monkeypatch.setattr(eng, "_take", fake_take)
    monkeypatch.setattr(voiceprint, "available", lambda: True)
    monkeypatch.setattr(voiceprint, "match", lambda path, reference: next(scores))
    sp = eng.speak("Worst car wash in Oregon.", tmp_path / "line.wav")
    assert temps == [0.7, 0.5]          # first take drifted, second (steadier) matched
    assert eng.last_match == 0.81 and sp.duration_s > 0


def test_quiet_peaky_mix_still_reaches_target_loudness(tmp_path):
    """Cloned voices are quiet with sharp peaks; plain loudnorm fell short (-15.2 LUFS)."""
    import re
    import subprocess
    from reelwise.studio.assemble import loudnorm_args
    from reelwise.studio.voices import write_wav
    rng = np.random.default_rng(1)
    t = np.arange(SR * 12) / SR
    x = 0.03 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t))
    x[rng.integers(0, len(x), 40)] = 0.6                     # sharp peaks
    wav = tmp_path / "mix.wav"
    write_wav(wav, x.astype(np.float32))
    af = loudnorm_args(wav)
    out = tmp_path / "out.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", af + ",aresample=48000",
                    "-c:a", "aac", "-b:a", "192k", str(out)], check=True)
    r = subprocess.run(["ffmpeg", "-nostats", "-i", str(out), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    i = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r)[-1])
    peak = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r)[-1])
    assert abs(i + 14) <= 1.0 and peak <= -1.0, (i, peak, af)


def test_new_voice_sample_means_new_takes(tmp_path):
    from reelwise.studio.voices import ChatterboxEngine, write_wav
    a, b = tmp_path / "a" / "sample.wav", tmp_path / "b" / "sample.wav"
    a.parent.mkdir()
    b.parent.mkdir()
    write_wav(a, np.zeros(SR, np.float32))
    write_wav(b, np.full(SR, 0.1, np.float32))
    assert ChatterboxEngine(str(a)).key("Toys?") != ChatterboxEngine(str(b)).key("Toys?")
