"""Image to video.

FalProvider calls hosted models on fal.ai with one key. The model table below
holds what each endpoint expects; when fal changes an endpoint, edit the
table, not the code. Endpoint ids and fields were checked against fal's API
reference on 3 November 2026.

DraftProvider costs nothing and needs no network. It moves a virtual camera
over your image (push-ins, pans, detail close-ups, top-down drift) with hard
cuts between shots. Use it to check timing, captions and the TV transition
before paying for real generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

STYLE_SUFFIX = ("Keep every character's face, hair, skin tone and clothing exactly as in the reference. "
                "Characters do not speak: mouths closed or still. Smooth, physically plausible motion. "
                "No on-screen text, no captions, no logos.")
NEGATIVE = ("talking, speaking, lip movement, mouth moving, subtitles, text, watermark, logo, "
            "morphing face, identity change, extra fingers, deformed hands, extra limbs, flicker, "
            "blur, low quality, jitter")

MODELS = {
    "kling-v3-pro": {
        "endpoint": "fal-ai/kling-video/v3/pro/image-to-video",
        "image_field": "start_image_url",
        "durations": list(range(3, 16)),
        "price_per_s": 0.112,
        "elements": True,
        "multi_prompt": True,
        "extra": {"generate_audio": False, "negative_prompt": NEGATIVE, "cfg_scale": 0.5},
    },
    "kling-o3-pro": {
        "endpoint": "fal-ai/kling-video/o3/pro/image-to-video",
        "image_field": "image_url",
        "durations": list(range(3, 16)),
        "price_per_s": 0.112,
        "elements": False,
        "multi_prompt": True,
        "extra": {"generate_audio": False},
    },
    "seedance-2": {
        "endpoint": "bytedance/seedance-2.0/image-to-video",
        "image_field": "image_url",
        "durations": list(range(4, 16)),
        "price_per_s": 0.3034,
        "elements": False,
        "multi_prompt": False,
        "extra": {"generate_audio": False, "resolution": "720p"},
    },
    "veo-3.1-fast": {
        "endpoint": "fal-ai/veo3.1/fast/image-to-video",
        "image_field": "image_url",
        "durations": [4, 6, 8],
        "price_per_s": 0.10,
        "elements": False,
        "multi_prompt": False,
        "duration_format": "{}s",
        "extra": {"generate_audio": False, "resolution": "1080p"},
    },
    "draft": {"endpoint": None, "durations": list(range(1, 31)), "price_per_s": 0.0},
    "motion": {"endpoint": None, "durations": list(range(1, 31)), "price_per_s": 0.0, "free": True},
    "motion+hf": {"endpoint": None, "durations": list(range(1, 31)), "price_per_s": 0.0, "free": True},
}
FREE_MODELS = ["motion", "motion+hf", "draft"]
PAID_MODELS = ["kling-v3-pro", "kling-o3-pro", "seedance-2", "veo-3.1-fast"]


def file_hash(path: str | Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _images(plan: dict) -> dict[int, str]:
    return {im["id"]: im["path"] for im in plan["images"]}


def _chars(plan: dict) -> dict[str, dict]:
    return {c["name"].lower(): c for c in plan.get("story_characters", [])}


def build_prompt(shots: list[dict], element_names: list[str]) -> str:
    def tag(p: str) -> str:
        for i, name in enumerate(element_names, 1):
            p = re.sub(rf"\b{re.escape(name)}\b", f"@Element{i} ({name})", p, flags=re.I)
        return p

    if len(shots) == 1:
        body = tag(shots[0]["prompt"])
    else:
        body = " ".join(f"Shot {i + 1} ({sh['duration_s']:.1f}s): {tag(sh['prompt'])}." for i, sh in enumerate(shots))
        body += " Hard cut between shots."
    return f"{body} {STYLE_SUFFIX}"


def integer_split(durations: list[float], total: int) -> list[int]:
    """Whole seconds per shot that sum to total, each at least 1."""
    raw = np.array(durations, float)
    raw = raw / raw.sum() * total
    out = np.maximum(1, np.floor(raw)).astype(int)
    while out.sum() < total:
        out[int(np.argmax(raw - out))] += 1
    while out.sum() > total:
        i = int(np.argmax(out))
        out[i] -= 1
    return out.tolist()


class Provider:
    name = "base"

    def __init__(self, model: str):
        self.model = model
        self.spec = MODELS[model]

    def generate(self, job: dict, plan: dict, out_path: Path, attempt: int = 0, feedback: str = "") -> dict:
        raise NotImplementedError


class FalProvider(Provider):
    name = "fal"

    def __init__(self, model: str = "kling-v3-pro", client=None, downloader=None, cache_path: Path | None = None):
        super().__init__(model)
        if not self.spec.get("endpoint"):
            raise ValueError(f"{model} is not a fal model")
        if client is None:
            import fal_client
            if not os.environ.get("FAL_KEY"):
                raise RuntimeError("FAL_KEY is not set. Create a key at fal.ai and add it to .env")
            client = fal_client
        self.client = client
        self.download = downloader or _download
        self.cache_path = cache_path
        self.uploads = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}

    def _url(self, path: str) -> str:
        h = file_hash(path)
        if h not in self.uploads:
            self.uploads[h] = self.client.upload_file(path)
            if self.cache_path:
                self.cache_path.write_text(json.dumps(self.uploads, indent=1))
        return self.uploads[h]

    def arguments(self, job: dict, plan: dict, feedback: str = "") -> dict:
        spec = self.spec
        img = _images(plan)[job["image"]]
        chars = _chars(plan)
        args = {spec["image_field"]: self._url(img)}
        names = []
        if spec.get("elements"):
            els = []
            for name in job.get("characters", []):
                c = chars.get(name.lower())
                if not c or not c.get("images"):
                    continue
                el = {"frontal_image_url": self._url(c["images"][0])}
                if len(c["images"]) > 1:
                    el["reference_image_urls"] = [self._url(p) for p in c["images"][1:4]]
                els.append(el)
                names.append(c["name"])
            if els:
                args["elements"] = els
        fmt = spec.get("duration_format", "{}")
        args["duration"] = fmt.format(job["duration"])
        shots = job["shots"]
        if spec.get("multi_prompt") and len(shots) > 1:
            secs = integer_split(job.get("lengths") or [s["duration_s"] for s in shots], job["duration"])
            args["multi_prompt"] = [{"prompt": f"{build_prompt([s], names)}", "duration": str(d)}
                                    for s, d in zip(shots, secs)]
            args["shot_type"] = "customize"
        else:
            args["prompt"] = build_prompt(shots, names)
        if feedback:
            note = f" Fix from the last attempt: {feedback}"
            if "prompt" in args:
                args["prompt"] += note
            else:
                args["multi_prompt"][0]["prompt"] += note
        args.update(spec.get("extra", {}))
        return args

    def generate(self, job: dict, plan: dict, out_path: Path, attempt: int = 0, feedback: str = "") -> dict:
        args = self.arguments(job, plan, feedback)
        try:
            res = self.client.subscribe(self.spec["endpoint"], arguments=args, with_logs=False)
        except Exception as e:
            # Some endpoints reject multi_prompt for some inputs; fall back to one prompt.
            if "multi_prompt" in args and ("422" in str(e) or "multi_prompt" in str(e)):
                names = [n for n in job.get("characters", [])]
                args.pop("multi_prompt")
                args.pop("shot_type", None)
                args["prompt"] = build_prompt(job["shots"], names if self.spec.get("elements") else [])
                res = self.client.subscribe(self.spec["endpoint"], arguments=args, with_logs=False)
            else:
                raise
        url = res["video"]["url"]
        self.download(url, out_path)
        return {"provider": "fal", "model": self.model, "endpoint": self.spec["endpoint"],
                "arguments": {k: v for k, v in args.items() if "url" not in k and k != "elements"},
                "attempt": attempt, "source_url": url}


def _download(url: str, out: Path) -> None:
    import requests
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


# --------------------------------------------------------------------------- draft

def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    r = cv2.resize(img, (int(np.ceil(iw * s)), int(np.ceil(ih * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y0:y0 + h, x0:x0 + w]


def _detail_point(img: np.ndarray) -> tuple[float, float]:
    """Centre of the most detailed region, as fractions of width and height."""
    g = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    lap = cv2.GaussianBlur(lap, (0, 0), 6)
    y, x = np.unravel_index(int(np.argmax(lap[8:56, 8:56])), (48, 48))
    return (x + 8) / 64, (y + 8) / 64


MOVES = ["push_in", "detail", "pan_left", "pull_out", "pan_right", "tilt_up", "detail2", "drift"]


def _move_for(prompt: str, k: int) -> str:
    p = prompt.lower()
    if "close-up" in p or "insert" in p or "detail" in p:
        return "detail" if k % 2 == 0 else "detail2"
    if "top-down" in p or "drone" in p or "high" in p:
        return "drift"
    if "push" in p:
        return "push_in"
    if "pull" in p or "reveal" in p:
        return "pull_out"
    if "track" in p or "pan" in p or "sideways" in p:
        return "pan_left" if k % 2 == 0 else "pan_right"
    if "low-angle" in p or "tilt" in p:
        return "tilt_up"
    return MOVES[k % len(MOVES)]


def _camera(move: str, u: float, dp: tuple[float, float]) -> tuple[float, float, float]:
    """(zoom, centre_x, centre_y) at progress u in [0, 1]. Centre in image fractions."""
    e = u * u * (3 - 2 * u)  # smoothstep
    if move == "push_in":
        return 1.0 + 0.18 * e, 0.5, 0.48
    if move == "pull_out":
        return 1.22 - 0.2 * e, 0.5, 0.5
    if move == "pan_left":
        return 1.25, 0.6 - 0.2 * e, 0.5
    if move == "pan_right":
        return 1.25, 0.4 + 0.2 * e, 0.5
    if move == "tilt_up":
        return 1.25, 0.5, 0.6 - 0.2 * e
    if move in ("detail", "detail2"):
        z = 2.2 + (0.25 * e if move == "detail" else -0.2 * e)
        return z, dp[0], dp[1]
    return 1.15 + 0.05 * e, 0.47 + 0.06 * e, 0.47 + 0.04 * e  # drift


class DraftProvider(Provider):
    name = "draft"

    def __init__(self, width: int = 1080, height: int = 1920, fps: int = 30):
        super().__init__("draft")
        self.w, self.h, self.fps = width, height, fps

    def generate(self, job: dict, plan: dict, out_path: Path, attempt: int = 0, feedback: str = "") -> dict:
        img = cv2.imread(_images(plan)[job["image"]])
        if img is None:
            raise ValueError(f"cannot read image {job['image']}")
        # Work on a cover-fitted canvas 2.5x the output so close-ups stay sharp.
        scale = 2.5
        base = _cover(img, int(self.w * scale), int(self.h * scale))
        bh, bw = base.shape[:2]
        dp = _detail_point(base)
        total = job["duration"]
        lengths = job.get("lengths") or [sh["duration_s"] for sh in job["shots"]]
        lengths = (np.array(lengths) / sum(lengths) * total).tolist()
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.w}x{self.h}", "-r", str(self.fps), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k, (sh, L) in enumerate(zip(job["shots"], lengths)):
            move = _move_for(sh["prompt"], k + int(hashlib.sha1(job["job_id"].encode()).hexdigest(), 16) % 3)
            n = max(1, int(round(L * self.fps)))
            for i in range(n):
                z, cx, cy = _camera(move, i / max(n - 1, 1), dp)
                cw, ch = bw / z, bh / z  # crop size in base pixels; z=1 shows the whole canvas
                x0 = np.clip(cx * bw - cw / 2, 0, bw - cw)
                y0 = np.clip(cy * bh - ch / 2, 0, bh - ch)
                M = np.array([[self.w / cw, 0, -x0 * self.w / cw], [0, self.h / ch, -y0 * self.h / ch]], np.float32)
                frame = cv2.warpAffine(base, M, (self.w, self.h), flags=cv2.INTER_AREA if z < 1.6 else cv2.INTER_LINEAR)
                proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed writing draft clip")
        return {"provider": "draft", "model": "draft", "attempt": attempt}


class MotionProvider(Provider):
    """Free. Depth-based 3D camera moves through each image (see motion.py)."""
    name = "motion"

    def __init__(self, width: int = 1080, height: int = 1920, fps: int = 30):
        super().__init__("motion")
        self.w, self.h, self.fps = width, height, fps

    def generate(self, job: dict, plan: dict, out_path: Path, attempt: int = 0, feedback: str = "") -> dict:
        from .motion import render_job
        lengths = job.get("lengths") or [sh["duration_s"] for sh in job["shots"]]
        render_job(_images(plan)[job["image"]], job["shots"], lengths, job["duration"], out_path,
                   self.w, self.h, self.fps, seed_key=f"{job['job_id']}:{attempt}")
        return {"provider": "motion", "model": "motion", "attempt": attempt}


class HFSpaceProvider(Provider):
    """Free AI image-to-video on a public Hugging Face Space (free account, about 5 GPU
    minutes a day). Used for one hero shot per scene; every other shot, and any shot the
    Space cannot serve (quota used, Space asleep or changed), falls back to MotionProvider.

    The Space's API is read at run time and its inputs are filled by name (image, prompt,
    duration, seed), so small changes to the Space do not break this."""
    name = "motion+hf"
    DEFAULT_SPACE = "mcp-tools/wan2-2-fp8da-aoti-faster"

    def __init__(self, fallback: "MotionProvider", space: str | None = None, token: str | None = None,
                 client=None, budget_shots: int = 3):
        super().__init__("motion+hf")
        self.fallback = fallback
        self.space = space or os.environ.get("REELWISE_HF_SPACE", self.DEFAULT_SPACE)
        self.token = token or os.environ.get("HF_TOKEN")
        self._client = client
        self.budget = budget_shots
        self.used = 0
        self.min_identity = 0.40   # measured: same image 0.43-0.86, different images -0.03-0.39
        self.notes: list[str] = []

    def client(self):
        if self._client is None:
            from gradio_client import Client
            self._client = Client(self.space, hf_token=self.token) if self.token else Client(self.space)
        return self._client

    @staticmethod
    def pick_endpoint(api: dict) -> tuple[str, list[dict]]:
        for name, ep in (api.get("named_endpoints") or {}).items():
            params = ep.get("parameters", [])
            if any("image" in (p.get("parameter_name") or "").lower() or
                   "image" in str(p.get("component", "")).lower() for p in params):
                return name, params
        raise RuntimeError("no image-to-video endpoint found on the Space")

    @staticmethod
    def fill(params: list[dict], image, prompt: str, seconds: float, seed: int) -> dict:
        args = {}
        for p in params:
            n = p.get("parameter_name") or ""
            ln = n.lower()
            comp = str(p.get("component", "")).lower()
            if "image" in ln or comp == "image":
                if "end" in ln or "last" in ln:
                    continue
                args[n] = image
            elif "negative" in ln:
                args[n] = NEGATIVE
            elif "prompt" in ln or ln in ("text", "caption"):
                args[n] = prompt
            elif "duration" in ln or "seconds" in ln or "length" in ln:
                args[n] = round(float(seconds), 1)
            elif ln == "seed":
                args[n] = int(seed)
            elif "random" in ln and "seed" in ln:
                args[n] = False
        return args

    def _hero(self, job: dict) -> int:
        lengths = job.get("lengths") or [s["duration_s"] for s in job["shots"]]
        return int(np.argmax(lengths))

    def generate(self, job: dict, plan: dict, out_path: Path, attempt: int = 0, feedback: str = "") -> dict:
        meta = self.fallback.generate(job, plan, out_path, attempt, feedback)
        if self.used >= self.budget:
            return meta
        k = self._hero(job)
        try:
            from gradio_client import handle_file
            c = self.client()
            name, params = self.pick_endpoint(c.view_api(print_info=False, return_format="dict"))
            shot = job["shots"][k]
            secs = min(5.0, max(2.0, (job.get("lengths") or [shot["duration_s"]] * len(job["shots"]))[k] + 0.5))
            args = self.fill(params, handle_file(_images(plan)[job["image"]]),
                             build_prompt([shot], []), secs, 1000 + attempt)
            res = c.predict(api_name=name, **args)
            vid = _find_video(res)
            if not vid:
                raise RuntimeError("Space returned no video")
            self.used += 1
            score = identity_score(_images(plan)[job["image"]], vid)
            if score < self.min_identity:
                raise RuntimeError(f"AI clip drifted from the image (similarity {score:.2f} < {self.min_identity})")
            splice_hero(out_path, vid, job, k, self.fallback.w, self.fallback.h, self.fallback.fps)
            meta["identity_score"] = round(score, 3)
            meta.update({"provider": "motion+hf", "hero_shot": k + 1, "space": self.space})
        except Exception as e:  # quota, sleep, or API change: keep the free motion clip
            self.notes.append(f"{job['job_id']}: AI shot skipped ({str(e)[:120]})")
            meta["hf_note"] = self.notes[-1]
        return meta


def identity_score(image_path: str, video_path: str) -> float:
    """Free check that an AI clip still shows the same picture: colour (HSV histogram
    correlation) and layout (correlation of small greyscale thumbnails) between the source
    image and the clip's middle and last frames. 1.0 = identical. Measured on real story
    art: clips of the same image score 0.43-0.86, different images -0.03-0.39."""
    img = cv2.imread(image_path)
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for k in (n // 2, max(0, n - 2)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if img is None or not frames:
        return 0.0

    def hist(x):
        h = cv2.calcHist([cv2.cvtColor(x, cv2.COLOR_BGR2HSV)], [0, 1], None, [30, 16], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    def thumb(x):
        g = cv2.cvtColor(cv2.resize(x, (32, 56), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (g - g.mean()) / (g.std() + 1e-6)

    hi, ti = hist(img), thumb(img)
    scores = []
    for f in frames:
        c = cv2.compareHist(hi, hist(f), cv2.HISTCMP_CORREL)
        l = float((ti * thumb(f)).mean())
        scores.append(0.6 * c + 0.4 * l)
    return float(min(scores))


def _find_video(res) -> str | None:
    if isinstance(res, str) and res.lower().endswith((".mp4", ".webm", ".mov")):
        return res
    if isinstance(res, dict):
        for v in res.values():
            f = _find_video(v)
            if f:
                return f
    if isinstance(res, (list, tuple)):
        for v in res:
            f = _find_video(v)
            if f:
                return f
    return None


def splice_hero(clip: Path, hero: str, job: dict, k: int, w: int, h: int, fps: int) -> None:
    """Replace shot k of the motion clip with the AI clip, keeping every frame count the same."""
    lengths = job.get("lengths") or [s["duration_s"] for s in job["shots"]]
    total = job["duration"]
    secs = (np.array(lengths, float) / sum(lengths) * total)
    frames = [max(1, int(round(x * fps))) for x in secs]
    a = sum(frames[:k])
    n = frames[k]
    tmp = clip.with_suffix(".hero.mp4")
    vf = (f"[1:v]fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,"
          f"tpad=stop_mode=clone:stop_duration=5,trim=end_frame={n},setpts=PTS-STARTPTS[h];"
          f"[0:v]trim=end_frame={a},setpts=PTS-STARTPTS[a];"
          f"[0:v]trim=start_frame={a + n},setpts=PTS-STARTPTS[b];"
          f"[a][h][b]concat=n=3:v=1:a=0[v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-i", hero, "-filter_complex", vf,
                    "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                    str(tmp)], check=True)
    tmp.replace(clip)


def make_provider(name: str, plan: dict, project_dir: Path) -> Provider:
    f = plan["format"]
    if name == "motion":
        return MotionProvider(f["width"], f["height"], f["fps"])
    if name == "motion+hf":
        return HFSpaceProvider(MotionProvider(f["width"], f["height"], f["fps"]))
    if name == "draft":
        return DraftProvider(f["width"], f["height"], f["fps"])
    return FalProvider(name, cache_path=project_dir / "uploads.json")
