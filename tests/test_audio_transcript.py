import numpy as np
import pytest

from reelwise.audio import SR, measure
from reelwise.transcript import parse_captions, speech_stats, youtube_video_id


def _tone(s, amp=0.3):
    t = np.arange(int(SR * s)) / SR
    return amp * np.sin(2 * np.pi * 220 * t)


def test_audio_onset_and_pause():
    x = np.concatenate([np.zeros(SR), _tone(2), np.zeros(SR // 2), _tone(1)]).astype(np.float32)
    a = measure(x)
    assert a.first_sound_s == pytest.approx(1.0, abs=0.06)
    assert a.longest_pause_s == pytest.approx(0.5, abs=0.06)
    assert a.pauses_over_300ms == 1


def test_steady_music_bed_counts_as_sound():
    a = measure(_tone(3).astype(np.float32))
    assert a.active_ratio > 0.9
    assert a.first_sound_s == 0.0


SRT = """1
00:00:00,350 --> 00:00:02,000
Would you live in a tree for two years?

2
00:00:02,100 --> 00:00:04,500
<i>She did.</i>
"""

VTT = """WEBVTT

00:00.400 --> 00:01.900 align:start
Is this the strangest job in Britain?

00:02.000 --> 00:03.000
Yes.
"""


def test_parse_srt_strips_tags():
    lines = parse_captions(SRT)
    assert [l.text for l in lines] == ["Would you live in a tree for two years?", "She did."]
    assert lines[0].start == pytest.approx(0.35)
    assert lines[1].end == pytest.approx(4.5)


def test_parse_vtt_short_timestamps_and_settings():
    lines = parse_captions(VTT)
    assert len(lines) == 2
    assert lines[0].start == pytest.approx(0.4)
    assert lines[1].text == "Yes."


def test_speech_stats():
    s = speech_stats(parse_captions(VTT), video_duration_s=10)
    assert s.first_line_is_question
    assert s.questions == 1
    assert s.one_word_lines == 1
    assert s.n_words == 8
    assert s.words_per_second_overall == 0.8


@pytest.mark.parametrize("ref", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ?t=3",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "dQw4w9WgXcQ",
])
def test_youtube_ids(ref):
    assert youtube_video_id(ref) == "dQw4w9WgXcQ"


def test_bad_youtube_ref():
    with pytest.raises(ValueError):
        youtube_video_id("https://example.com/video")
