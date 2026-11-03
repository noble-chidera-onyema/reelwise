import cv2
import numpy as np
import pytest


def _solid_scene(colour, n, w=160, h=284, moving=True, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    base = np.full((h, w, 3), colour, np.uint8)
    noise = rng.integers(0, 25, (h, w, 3), dtype=np.uint8)
    base = cv2.add(base, noise)
    for i in range(n):
        f = base.copy()
        if moving:
            cv2.circle(f, (20 + (i * 3) % (w - 40), h // 2), 18, (255, 255, 255), -1)
        frames.append(f)
    return frames


@pytest.fixture(scope="session")
def cut_video(tmp_path_factory):
    """4 s clip at 25 fps: red 1.0 s | blue 1.2 s | green 1.8 s. Cuts at frames 25 and 55."""
    path = tmp_path_factory.mktemp("vid") / "cuts.mp4"
    frames = (_solid_scene((30, 30, 200), 25, seed=1)
              + _solid_scene((200, 60, 30), 30, seed=2)
              + _solid_scene((40, 180, 40), 45, seed=3))
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path, [25, 55]


@pytest.fixture(scope="session")
def flash_video(tmp_path_factory):
    """One continuous shot with a 2-frame white flash in the middle. No real cut."""
    path = tmp_path_factory.mktemp("vid") / "flash.mp4"
    frames = _solid_scene((90, 120, 60), 60, seed=4)
    frames[30] = np.full_like(frames[30], 250)
    frames[31] = np.full_like(frames[31], 250)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return path


class FakeBlock:
    def __init__(self, text):
        self.text = text


class FakeMessage:
    def __init__(self, text):
        self.content = [FakeBlock(text)]


class FakeClient:
    """Stands in for anthropic.Anthropic; records the request and returns canned text."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return FakeMessage(self.reply)


@pytest.fixture
def fake_client():
    return FakeClient
