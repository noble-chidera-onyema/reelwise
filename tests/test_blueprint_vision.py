import json
from pathlib import Path

import pytest

from reelwise.analyse import analyse_video
from reelwise.blueprint import build_format, check_script, check_video, load_profile, write_episode

PROFILE = load_profile(Path(__file__).resolve().parents[1] / "examples" / "profile_example.json")


def _script(lines):
    return {"lines": [{"speaker": s, "text": t} for s, t in lines]}


def test_good_script_passes():
    s = _script([
        ("Host", "Would you live up a tree for two whole years to save it?"),
        ("Sidekick", "One woman did exactly that, and the loggers had to wait her out."),
        ("Host", "She came down only when the company signed a deal to protect the forest."),
        ("Sidekick", "The tree she lived in is still standing today, and people visit it."),
        ("Host", "Two years, one tree, and a promise that she made stick in the end."),
        ("Sidekick", "Follow us if you want more true stories like this one every week."),
        ("Host", "We have plenty more of these waiting for you, so stick around for them."),
        ("Sidekick", "That is the end of this one, see you on the next story from us."),
    ])
    qa = check_script(s, PROFILE)
    failed = [c for c in qa["checks"] if not c["passed"]]
    assert qa["passed"], failed
    assert 30 <= qa["estimated_duration_s"] <= 45
    detail = next(c for c in qa["checks"] if c["rule"] == "total duration")["detail"]
    assert f"estimated {qa['estimated_duration_s']:.1f}s" in detail


def test_bad_script_fails_the_right_rules():
    s = _script([
        ("Sidekick", "This is a story about a tree."),
        ("Host", "Wow."),
        ("Stranger", "It lasted 738 days in total."),
    ])
    qa = check_script(s, PROFILE)
    failed = {c["rule"] for c in qa["checks"] if not c["passed"]}
    assert {"total duration", "first speaker", "first line is a question", "no choppy lines",
            "numbers written as words for the voice", "only profile characters speak"} <= failed


def test_timeline_uses_speaker_rates():
    qa = check_script(_script([("Host", "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen?")]), PROFILE)
    row = qa["timeline"][0]
    assert row["start_s"] == 0.3
    assert row["end_s"] == pytest.approx(0.3 + 16 / 3.2, abs=0.01)


def test_format_and_episode_with_fake_model(cut_video, fake_client):
    path, _ = cut_video
    reply = {"opening_frame": {"what_is_shown": "red frame", "on_screen_text": "", "why_it_might_stop_a_scroll": "colour"},
             "scenes": [{"scene": 0, "shot_type": "graphic", "subject": "red", "action": "ball moves",
                         "setting": "", "on_screen_text": "", "caption_style": "", "people_count": 0,
                         "facial_expression": "", "notable": ""}],
             "format_summary": "three colour cards"}
    vclient = fake_client("Here you go:\n" + json.dumps(reply))
    report = analyse_video(path, vision=True, vision_client=vclient)
    sent = vclient.calls[0]["messages"][0]["content"]
    assert sum(1 for c in sent if c["type"] == "image") == 4  # opening frame + 3 scenes
    assert report["vision"]["format_summary"] == "three colour cards"

    fmt = build_format(report)
    assert [b["role"] for b in fmt["beats"]] == ["hook", "body", "close"]
    assert fmt["beats"][0]["shot_type"] == "graphic"
    assert fmt["beats"][-1]["end_pct"] == 100.0

    episode = {"title": "T", "lines": [{"speaker": "Host", "text": "Is this real?"}],
               "shots": [], "cover_image_prompt": "", "facts_used": []}
    eclient = fake_client(json.dumps(episode))
    out = write_episode(fmt, PROFILE, "A true story.", client=eclient)
    assert out["title"] == "T"
    prompt = eclient.calls[0]["messages"][0]["content"]
    assert "A true story." in prompt and "Host" in prompt


def test_check_video(cut_video):
    path, _ = cut_video
    qa = check_video(analyse_video(path), PROFILE)
    assert not qa["passed"]  # 4 s clip is outside the 30-45 s target
    assert qa["checks"][0]["rule"] == "duration"
