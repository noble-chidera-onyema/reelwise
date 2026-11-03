from reelwise.analyse import analyse_video
from reelwise.scenes import build_scenes, detect_boundaries
from reelwise.video import probe, read_frames


def test_probe_reads_geometry(cut_video):
    path, _ = cut_video
    info = probe(path)
    assert (info.width, info.height) == (160, 284)
    assert info.orientation == "vertical"
    assert abs(info.fps - 25) < 0.01


def test_detects_both_cuts_on_the_right_frame(cut_video):
    path, truth = cut_video
    fs = read_frames(path)
    found = [b.frame for b in detect_boundaries(fs)]
    assert found == truth


def test_scenes_cover_whole_video(cut_video):
    path, _ = cut_video
    fs = read_frames(path)
    scenes = build_scenes(fs, detect_boundaries(fs))
    assert [s.start_frame for s in scenes] == [0, 25, 55]
    assert scenes[-1].end_frame == fs.n
    assert abs(sum(s.duration_s for s in scenes) - fs.n / fs.fps) < 1e-6


def test_flash_is_not_a_cut(flash_video):
    fs = read_frames(flash_video)
    assert detect_boundaries(fs) == []


def test_analyse_without_audio_or_captions(cut_video):
    path, _ = cut_video
    r = analyse_video(path)
    assert r["pacing"]["scenes"] == 3
    assert r["hook"]["first_cut_s"] == 1.0
    assert r["audio"] is None and r["speech"] is None
    assert len(r["per_second"]) == 4
