"""Which videos on a channel beat that channel's own normal, and what differs about them.

Raw view counts mislead: a video on a 2M-subscriber channel with 400k views
may be a flop, and 40k views can be a breakout for a channel of 3k. So every
video is scored against its own channel's recent baseline:

    outlier_score = views / median views of the N videos published closest in time

A score of 3.0 means three times the channel's normal at that point in its
history. Videos younger than `settle_days` are left unscored because their
counts are still climbing.

The comparison between overperformers and the rest is descriptive. It shows
what the breakout videos had in common on measurable features (title shape,
length, timing). It does not show that any feature caused the views; with a
few dozen videos per channel that claim cannot be supported, and the output
says so.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

STOPWORDS = set("""a an and are as at be but by for from has have he her his how i if in into is it its
me my no not of on or our she so than that the their them they this to up us was we what when who why
will with you your just new vs""".split())

OVER = 2.0
UNDER = 0.5


def to_frame(videos: list[dict], now: datetime | None = None) -> pd.DataFrame:
    df = pd.DataFrame(videos)
    if df.empty:
        return df
    now = now or datetime.now(timezone.utc)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["age_days"] = (pd.Timestamp(now) - df["published_at"]).dt.total_seconds() / 86400
    t = df["title"].fillna("")
    df["title_chars"] = t.str.len()
    df["title_words"] = t.str.split().str.len()
    df["title_question"] = t.str.contains(r"\?")
    df["title_number"] = t.str.contains(r"\d")
    df["title_caps_word"] = t.apply(lambda s: bool(re.search(r"\b[A-Z]{3,}\b", s)))
    df["title_you"] = t.str.contains(r"\byour?\b", case=False)
    df["title_first_person"] = t.str.contains(r"\b(?:I|my|we|our)\b")
    df["short_form"] = df["duration_s"] <= 180
    df["publish_hour_utc"] = df["published_at"].dt.hour
    df["publish_weekday"] = df["published_at"].dt.day_name()
    df["tags_count"] = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["description_chars"] = df["description"].fillna("").str.len()
    views = df["views"].astype(float)
    df["engagement_rate"] = ((df["likes"].fillna(0) + df["comments"].fillna(0)) / views).where(views > 0)
    return df


def score(df: pd.DataFrame, neighbours: int = 10, settle_days: float = 7.0) -> pd.DataFrame:
    df = df.sort_values("published_at").reset_index(drop=True)
    settled = (df["age_days"] >= settle_days) & df["views"].notna()
    scores = np.full(len(df), np.nan)
    idx = np.flatnonzero(settled.to_numpy())
    times = df["published_at"].astype("int64").to_numpy()
    views = df["views"].astype(float).to_numpy()
    for i in idx:
        others = idx[idx != i]
        if len(others) < 3:
            continue
        nearest = others[np.argsort(np.abs(times[others] - times[i]))[:neighbours]]
        base = np.median(views[nearest])
        if base > 0:
            scores[i] = views[i] / base
    df["outlier_score"] = np.round(scores, 2)
    df["group"] = np.where(df["outlier_score"] >= OVER, "overperformer",
                           np.where(df["outlier_score"] <= UNDER, "underperformer",
                                    np.where(df["outlier_score"].notna(), "typical", "unscored")))
    return df


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size for two samples: +1 means every a is larger than every b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


FEATURES_NUM = ["duration_s", "title_chars", "title_words", "tags_count", "description_chars",
                "publish_hour_utc", "engagement_rate"]
FEATURES_BOOL = ["title_question", "title_number", "title_caps_word", "title_you",
                 "title_first_person", "short_form"]


def compare(df: pd.DataFrame) -> dict:
    over = df[df["group"] == "overperformer"]
    rest = df[df["group"].isin(["typical", "underperformer"])]
    rows = []
    for f in FEATURES_NUM + FEATURES_BOOL:
        a, b = over[f].dropna(), rest[f].dropna()
        if a.empty or b.empty:
            continue
        is_bool = f in FEATURES_BOOL
        a_num, b_num = a.astype(float), b.astype(float)
        rows.append({
            "feature": f,
            "overperformers": round(float(a_num.mean() if is_bool else a_num.median()), 3),
            "rest": round(float(b_num.mean() if is_bool else b_num.median()), 3),
            "summary": "share" if is_bool else "median",
            "effect_size": round(cliffs_delta(a_num.to_numpy(), b_num.to_numpy()), 2),
        })
    rows.sort(key=lambda r: -abs(r["effect_size"]))
    n_over = int(len(over))
    return {
        "n_overperformers": n_over,
        "n_rest": int(len(rest)),
        "n_unscored": int((df["group"] == "unscored").sum()),
        "evidence": ("too few overperformers to say anything (need at least 5)" if n_over < 5
                     else "descriptive only; differences are associations, not causes"),
        "features": rows,
        "title_words": title_word_lift(over["title"].tolist(), rest["title"].tolist()),
        "weekday_share_over": over["publish_weekday"].value_counts(normalize=True).round(2).to_dict(),
    }


def _words(title: str) -> list[str]:
    return [w for w in re.findall(r"[a-z']+", title.lower()) if w not in STOPWORDS and len(w) > 2]


def title_word_lift(over_titles: list[str], rest_titles: list[str], min_count: int = 2) -> list[dict]:
    """Words that appear in a larger share of overperformer titles than of the rest."""
    co = Counter(w for t in over_titles for w in set(_words(t)))
    cr = Counter(w for t in rest_titles for w in set(_words(t)))
    no, nr = max(len(over_titles), 1), max(len(rest_titles), 1)
    out = []
    for w, c in co.items():
        if c < min_count:
            continue
        share_o = c / no
        share_r = cr.get(w, 0) / nr
        out.append({"word": w, "over_share": round(share_o, 2), "rest_share": round(share_r, 2),
                    "lift": round((share_o + 0.01) / (share_r + 0.01), 1)})
    return sorted(out, key=lambda r: -r["lift"])[:15]


def summarise_trending(videos: list[dict]) -> dict:
    df = to_frame(videos)
    if df.empty:
        return {"n": 0}
    words = Counter(w for t in df["title"] for w in set(_words(t)))
    return {
        "n": int(len(df)),
        "median_duration_s": float(df["duration_s"].median()),
        "share_short_form": round(float(df["short_form"].mean()), 2),
        "share_title_question": round(float(df["title_question"].mean()), 2),
        "share_title_number": round(float(df["title_number"].mean()), 2),
        "median_title_chars": float(df["title_chars"].median()),
        "top_title_words": words.most_common(20),
        "categories": df["category_id"].value_counts().to_dict(),
    }
