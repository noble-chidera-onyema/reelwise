"""Public YouTube data through the official YouTube Data API v3.

Needs a free API key (Google Cloud console, enable "YouTube Data API v3").
Set it as YOUTUBE_API_KEY. Every call here is a read of public data and costs
1 quota unit; the default daily allowance is 10,000 units, so pulling a
channel's last 200 uploads costs about 9 units.

What the API gives: titles, descriptions, tags, publish times, durations,
view/like/comment counts. What it does not give for other people's videos:
watch time, retention curves, impressions or click-through rate. Those exist
only in YouTube Studio for the channel owner.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import requests

API = "https://www.googleapis.com/youtube/v3"
_DUR = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


class YouTubeError(RuntimeError):
    pass


def parse_duration(iso: str) -> int:
    """ISO 8601 duration (PT1M5S) to seconds."""
    m = _DUR.fullmatch(iso or "")
    if not m:
        return 0
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s


class YouTubeClient:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.key = api_key or os.environ.get("YOUTUBE_API_KEY")
        if not self.key:
            raise YouTubeError("No YouTube API key. Set YOUTUBE_API_KEY or pass api_key.")
        self.s = session or requests.Session()
        self.quota_used = 0

    def _get(self, endpoint: str, **params) -> dict:
        params["key"] = self.key
        r = self.s.get(f"{API}/{endpoint}", params=params, timeout=30)
        self.quota_used += 1
        if r.status_code != 200:
            try:
                msg = r.json()["error"]["message"]
            except Exception:
                msg = r.text[:300]
            raise YouTubeError(f"{endpoint}: HTTP {r.status_code}: {msg}")
        return r.json()

    def resolve_channel(self, ref: str) -> dict:
        """Accepts @handle, a channel URL, or a UC... channel id."""
        ref = ref.strip()
        m = re.search(r"youtube\.com/(@[\w.\-]+)", ref)
        if m:
            ref = m.group(1)
        m = re.search(r"youtube\.com/channel/(UC[\w-]{22})", ref)
        if m:
            ref = m.group(1)
        if re.fullmatch(r"UC[\w-]{22}", ref):
            data = self._get("channels", part="snippet,contentDetails,statistics", id=ref)
        else:
            handle = ref if ref.startswith("@") else "@" + ref
            data = self._get("channels", part="snippet,contentDetails,statistics", forHandle=handle)
        items = data.get("items") or []
        if not items:
            raise YouTubeError(f"Channel not found: {ref}")
        c = items[0]
        return {
            "id": c["id"],
            "title": c["snippet"]["title"],
            "handle": c["snippet"].get("customUrl"),
            "subscribers": int(c["statistics"].get("subscriberCount", 0)),
            "videos": int(c["statistics"].get("videoCount", 0)),
            "uploads_playlist": c["contentDetails"]["relatedPlaylists"]["uploads"],
        }

    def upload_ids(self, playlist_id: str, limit: int = 200) -> list[str]:
        ids, token = [], None
        while len(ids) < limit:
            params = dict(part="contentDetails", playlistId=playlist_id, maxResults=50)
            if token:
                params["pageToken"] = token
            data = self._get("playlistItems", **params)
            ids += [it["contentDetails"]["videoId"] for it in data.get("items", [])]
            token = data.get("nextPageToken")
            if not token:
                break
        return ids[:limit]

    def videos(self, ids: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(ids), 50):
            data = self._get("videos", part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50]))
            out += [_flatten(v) for v in data.get("items", [])]
        return out

    def channel_videos(self, ref: str, limit: int = 200) -> tuple[dict, list[dict]]:
        ch = self.resolve_channel(ref)
        return ch, self.videos(self.upload_ids(ch["uploads_playlist"], limit))

    def trending(self, region: str = "GB", category_id: str | None = None, limit: int = 50) -> list[dict]:
        params = dict(part="snippet,statistics,contentDetails", chart="mostPopular",
                      regionCode=region, maxResults=min(limit, 50))
        if category_id:
            params["videoCategoryId"] = category_id
        data = self._get("videos", **params)
        return [_flatten(v) for v in data.get("items", [])]


def _flatten(v: dict) -> dict:
    sn, st, cd = v.get("snippet", {}), v.get("statistics", {}), v.get("contentDetails", {})
    published = sn.get("publishedAt")
    return {
        "video_id": v["id"],
        "url": f"https://www.youtube.com/watch?v={v['id']}",
        "title": sn.get("title", ""),
        "description": sn.get("description", ""),
        "tags": sn.get("tags", []),
        "channel": sn.get("channelTitle", ""),
        "category_id": sn.get("categoryId"),
        "published_at": published,
        "duration_s": parse_duration(cd.get("duration", "")),
        # Counts can be hidden by the uploader; keep None, never a pretend 0.
        "views": int(st["viewCount"]) if "viewCount" in st else None,
        "likes": int(st["likeCount"]) if "likeCount" in st else None,
        "comments": int(st["commentCount"]) if "commentCount" in st else None,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
