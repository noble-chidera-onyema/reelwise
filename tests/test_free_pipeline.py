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
