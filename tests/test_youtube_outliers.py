from datetime import datetime, timedelta, timezone

import pytest

from reelwise.outliers import cliffs_delta, compare, score, summarise_trending, to_frame
from reelwise.youtube import YouTubeClient, YouTubeError, parse_duration


class Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._d


class FakeSession:
    """Replays responses shaped like the documented YouTube Data API v3 payloads."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url.rsplit("/", 1)[-1], dict(params)))
        ep = url.rsplit("/", 1)[-1]
        if ep == "channels":
            if params.get("forHandle") == "@missing":
                return Resp({"items": []})
            return Resp({"items": [{
                "id": "UC" + "x" * 22,
                "snippet": {"title": "Test Channel", "customUrl": "@test"},
                "statistics": {"subscriberCount": "1200", "videoCount": "3"},
                "contentDetails": {"relatedPlaylists": {"uploads": "UUabc"}},
            }]})
        if ep == "playlistItems":
            if "pageToken" not in params:
                return Resp({"items": [{"contentDetails": {"videoId": "aaaaaaaaaaa"}},
                                       {"contentDetails": {"videoId": "bbbbbbbbbbb"}}],
                             "nextPageToken": "P2"})
            return Resp({"items": [{"contentDetails": {"videoId": "ccccccccccc"}}]})
        if ep == "videos":
            ids = params["id"].split(",")
            return Resp({"items": [{
                "id": i,
                "snippet": {"title": f"Video {i[0]}?", "publishedAt": "2026-09-01T18:00:00Z",
                            "channelTitle": "Test Channel", "categoryId": "24"},
                "statistics": {"viewCount": "1000", "likeCount": "50"},
                "contentDetails": {"duration": "PT45S"},
            } for i in ids]})
        return Resp({"error": {"message": "nope"}}, 404)


def test_parse_duration():
    assert parse_duration("PT45S") == 45
    assert parse_duration("PT1H2M3S") == 3723
    assert parse_duration("P1DT1S") == 86401
    assert parse_duration("") == 0


def test_channel_videos_follows_pages_and_counts_quota():
    s = FakeSession()
    yt = YouTubeClient(api_key="k", session=s)
    ch, vids = yt.channel_videos("https://www.youtube.com/@test")
    assert ch["uploads_playlist"] == "UUabc"
    assert [v["video_id"] for v in vids] == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    assert vids[0]["duration_s"] == 45
    assert vids[0]["comments"] is None  # hidden counts stay None
    assert yt.quota_used == 4
    assert s.calls[0][1]["forHandle"] == "@test"


def test_missing_channel_and_missing_key(monkeypatch):
    yt = YouTubeClient(api_key="k", session=FakeSession())
    with pytest.raises(YouTubeError):
        yt.resolve_channel("@missing")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    with pytest.raises(YouTubeError):
        YouTubeClient()


def _channel(now):
    vids = []
    for i in range(20):
        views = 1000 + 10 * i
        title = f"Ordinary upload number {i}"
        if i in (4, 9, 14, 17, 19):
            views = 5000
            title = f"Would you EVER do this? Part {i}"
        vids.append({"video_id": str(i), "title": title, "description": "", "tags": [],
                     "published_at": (now - timedelta(days=100 - 4 * i)).isoformat(),
                     "duration_s": 40, "views": views, "likes": 10, "comments": 1,
                     "category_id": "24"})
    vids.append({"video_id": "new", "title": "Fresh", "description": "", "tags": [],
                 "published_at": (now - timedelta(days=1)).isoformat(), "duration_s": 40,
                 "views": 99999, "likes": 0, "comments": 0, "category_id": "24"})
    return vids


def test_outlier_scoring_and_comparison():
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    df = score(to_frame(_channel(now), now=now))
    assert df.loc[df.video_id == "new", "group"].item() == "unscored"
    over = set(df.loc[df.group == "overperformer", "video_id"])
    assert over == {"4", "9", "14", "17", "19"}
    comp = compare(df)
    assert comp["n_overperformers"] == 5
    feats = {f["feature"]: f for f in comp["features"]}
    assert feats["title_question"]["overperformers"] == 1.0
    assert feats["title_question"]["rest"] == 0.0
    assert feats["title_question"]["effect_size"] == 1.0
    assert comp["title_words"][0]["word"] in {"would", "ever", "part"}


def test_cliffs_delta():
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [3, 4]) == -1.0
    assert cliffs_delta([1, 2], [1, 2]) == 0.0


def test_trending_summary():
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    s = summarise_trending(_channel(now))
    assert s["n"] == 21
    assert s["share_short_form"] == 1.0
