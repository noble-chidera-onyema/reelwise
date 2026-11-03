"""Measure shot-boundary detection against the synthetic ground truth.

Matching rules:
  hard cut   counts as found if a detected boundary lies within 2 frames
  gradual    counts as found if a detected boundary lies within 0.5 s of its midpoint
Each detection can match one true boundary only. Unmatched detections are
false positives.

Compares reelwise with PySceneDetect's ContentDetector and AdaptiveDetector
at their default settings, so the numbers have a public reference point.

Usage:
    python eval/benchmark_scenes.py --data data/synth_dev
    python eval/benchmark_scenes.py --data data/synth_holdout --json docs/eval_holdout.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelwise.scenes import SceneParams, detect_boundaries  # noqa: E402
from reelwise.video import read_frames  # noqa: E402

CUT_TOL_FRAMES = 2
GRADUAL_TOL_S = 0.5


def match(truth: list[dict], detected_frames: list[int], fps: float) -> dict:
    used = set()
    res = {"cut_tp": 0, "cut_n": 0, "grad_tp": 0, "grad_n": 0,
           "similar_tp": 0, "similar_n": 0, "fp": 0, "timing_err_frames": []}
    for tb in sorted(truth, key=lambda b: b["frame"]):
        tol = CUT_TOL_FRAMES if tb["kind"] == "cut" else int(round(GRADUAL_TOL_S * fps))
        best, best_d = None, None
        for j, f in enumerate(detected_frames):
            if j in used:
                continue
            dd = abs(f - tb["frame"])
            if dd <= tol and (best_d is None or dd < best_d):
                best, best_d = j, dd
        key = "cut" if tb["kind"] == "cut" else "grad"
        res[f"{key}_n"] += 1
        if tb.get("similar_palette"):
            res["similar_n"] += 1
        if best is not None:
            used.add(best)
            res[f"{key}_tp"] += 1
            if tb.get("similar_palette"):
                res["similar_tp"] += 1
            if tb["kind"] == "cut":
                res["timing_err_frames"].append(best_d)
    res["fp"] = len(detected_frames) - len(used)
    res["n_detected"] = len(detected_frames)
    return res


def run_reelwise(video: Path, cache_dir: Path, params: SceneParams) -> tuple[list[int], float]:
    cache = cache_dir / (video.stem + ".pkl")
    if cache.exists():
        fs = pickle.loads(cache.read_bytes())
    else:
        fs = read_frames(video)
        fs.thumbs = []
        cache.write_bytes(pickle.dumps(fs))
    return [b.frame for b in detect_boundaries(fs, params)], fs.fps


def run_pyscenedetect(video: Path, which: str) -> list[int]:
    from scenedetect import AdaptiveDetector, ContentDetector, detect
    det = ContentDetector() if which == "content" else AdaptiveDetector()
    scenes = detect(str(video), det, show_progress=False)
    return [s[0].get_frames() for s in scenes[1:]]


def summarise(rows: list[dict]) -> dict:
    agg = {k: sum(r[k] for r in rows) for k in
           ["cut_tp", "cut_n", "grad_tp", "grad_n", "similar_tp", "similar_n", "fp", "n_detected"]}
    tp = agg["cut_tp"] + agg["grad_tp"]
    n_true = agg["cut_n"] + agg["grad_n"]
    prec = tp / agg["n_detected"] if agg["n_detected"] else 0.0
    rec = tp / n_true if n_true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    errs = [e for r in rows for e in r["timing_err_frames"]]
    return {
        "videos": len(rows),
        "true_boundaries": n_true,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "cut_recall": round(agg["cut_tp"] / agg["cut_n"], 3) if agg["cut_n"] else None,
        "gradual_recall": round(agg["grad_tp"] / agg["grad_n"], 3) if agg["grad_n"] else None,
        "similar_palette_recall": round(agg["similar_tp"] / agg["similar_n"], 3) if agg["similar_n"] else None,
        "false_positives": agg["fp"],
        "cuts": agg["cut_n"], "graduals": agg["grad_n"], "similar_palette": agg["similar_n"],
        "mean_cut_timing_error_frames": round(sum(errs) / len(errs), 2) if errs else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    cache_dir = data / ".cache"
    cache_dir.mkdir(exist_ok=True)

    params = SceneParams()
    methods = {"reelwise": []}
    if not args.skip_baselines:
        methods["pyscenedetect_content"] = []
        methods["pyscenedetect_adaptive"] = []

    for meta_path in sorted(data.glob("synth_*.json")):
        meta = json.loads(meta_path.read_text())
        video = data / meta["file"]
        frames, fps = run_reelwise(video, cache_dir, params)
        methods["reelwise"].append(match(meta["boundaries"], frames, fps))
        if not args.skip_baselines:
            methods["pyscenedetect_content"].append(match(meta["boundaries"], run_pyscenedetect(video, "content"), fps))
            methods["pyscenedetect_adaptive"].append(match(meta["boundaries"], run_pyscenedetect(video, "adaptive"), fps))

    out = {name: summarise(rows) for name, rows in methods.items()}
    print(json.dumps(out, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
