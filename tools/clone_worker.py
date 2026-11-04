"""Chatterbox worker, run with the .venv-clone Python (see tools/setup_voice_clone.bat).

Keeps PyTorch out of the main app. Two ways to use it:

    python tools/clone_worker.py --jobs jobs.json [--model turbo]   # a list of jobs, then exit
    python tools/clone_worker.py --serve [--model turbo]            # one JSON job per stdin line, one reply per line

--model turbo (default) is Chatterbox Turbo: smaller and much faster on a CPU, and it reads
tags such as [chuckle] or [laugh]. --model standard is the original, with the exaggeration
and cfg controls.

A job is {"text": ..., "sample": "voice.wav", "out": "line.wav", "exaggeration": 0.5, "cfg": 0.5}.
The model loads once per run, so a whole episode costs one load.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time


def load(kind: str = "turbo"):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        m = ChatterboxTurboTTS.from_pretrained(device=dev)
    else:
        from chatterbox.tts import ChatterboxTTS
        m = ChatterboxTTS.from_pretrained(device=dev)
    m.kind = kind
    return m


def run(model, job: dict) -> dict:
    import soundfile as sf
    t0 = time.time()
    if getattr(model, "kind", "standard") == "turbo":
        wav = model.generate(job["text"], audio_prompt_path=job.get("sample") or None,
                             temperature=float(job.get("temperature", 0.8)))
    else:
        wav = model.generate(job["text"], audio_prompt_path=job.get("sample") or None,
                             exaggeration=float(job.get("exaggeration", 0.5)),
                             cfg_weight=float(job.get("cfg", 0.5)))
    sf.write(job["out"], wav.squeeze(0).cpu().numpy(), model.sr)
    return {"ok": True, "out": job["out"], "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--model", choices=["turbo", "standard"], default="turbo")
    a = ap.parse_args()
    # Libraries print progress to stdout; stdout carries only this worker's JSON replies.
    with contextlib.redirect_stdout(sys.stderr):
        model = load(a.model)
    if a.jobs:
        for job in json.load(open(a.jobs, encoding="utf-8")):
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    reply = run(model, job)
                print(json.dumps(reply), flush=True)
            except Exception as e:  # keep going: one bad line should not stop an audition
                print(json.dumps({"ok": False, "out": job.get("out"), "error": str(e)}), flush=True)
        print("=== done", flush=True)
        return 0
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        job = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                reply = run(model, job)
        except Exception as e:
            reply = {"ok": False, "error": str(e)}
        print(json.dumps(reply), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
