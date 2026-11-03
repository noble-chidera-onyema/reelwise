"""Run the whole episode: plan, voice, generate, check, assemble, check again.

Everything lands in one project folder, and every expensive step is cached by
a hash of its inputs. Re-running after a fix only redoes what changed: edit
one scene's shots and only that scene's clip is generated again.

    projects/<name>/
        inputs/        your uploads
        audio/         one WAV per narration line, plus timing
        clips/         generated clips, one JSON report beside each
        shots/         clips cut to length at output size
        plan.json      the plan as last run
        report.json    every check, pass or fail
        final.mp4      the episode (written only when it is assembled)
        final.srt      subtitles for upload
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from . import assemble as asm
from . import plan as P
from . import qc
from .motion import MOTION_VERSION
from .providers import MODELS, Provider, file_hash, make_provider
from .voices import engine_from_config, time_plan


class Studio:
    def __init__(self, project_dir: str | Path, log=print):
        self.dir = Path(project_dir)
        for sub in ("inputs", "audio", "clips", "shots"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.log = log
        self.report: dict = {"clips": {}, "final": [], "events": []}

    def note(self, msg: str) -> None:
        self.report["events"].append({"t": time.strftime("%H:%M:%S"), "msg": msg})
        self.log(msg)

    # ----------------------------------------------------------------- plan and voice

    def prepare(self, plan: dict, voices: dict[str, dict], zoom_s: float = 1.0,
                planner_client=None, use_claude: bool = False) -> dict:
        engines = {name: engine_from_config(cfg) for name, cfg in voices.items()}
        self.note("Narrating every line")
        time_plan(plan, engines, self.dir / "audio", zoom_s=zoom_s)
        if use_claude:
            self.note("Planning shots with Claude")
            import base64
            import cv2
            b64 = {}
            for im in plan["images"]:
                img = cv2.imread(im["path"])
                s = 512 / img.shape[1]
                ok, buf = cv2.imencode(".jpg", cv2.resize(img, (512, int(img.shape[0] * s))))
                b64[im["id"]] = base64.b64encode(buf.tobytes()).decode()
            P.claude_shots(plan, b64, client=planner_client)
            P.assign_missing_images(plan)
        else:
            P.assign_missing_images(plan)
            P.fill_shots(plan)
        # Footage the TV needs beyond each scene's own running time (see assemble.roll_frames).
        for s in plan["segments"]:
            s["pre_roll_s"] = s["post_roll_s"] = 0.0
        story = [s for s in plan["segments"] if s["kind"] == "story"]
        by_kind = {s["kind"]: s for s in plan["segments"]}
        if story:
            intro, outro, cold = by_kind.get("intro"), by_kind.get("outro"), by_kind.get("cold_open")
            if intro:
                if cold:
                    cold["post_roll_s"] = zoom_s
                    story[0]["pre_roll_s"] = intro["duration_s"] - zoom_s
                else:
                    story[0]["pre_roll_s"] = intro["duration_s"]
            if outro:
                story[-1]["post_roll_s"] = outro["duration_s"]
        plan["zoom_s"] = zoom_s
        P.save(plan, self.dir / "plan.json")
        return plan

    # ----------------------------------------------------------------- generation

    def _job_key(self, job: dict, plan: dict, provider: Provider) -> str:
        imgs = {im["id"]: im["path"] for im in plan["images"]}
        chars = {c["name"].lower(): c for c in plan.get("story_characters", [])}
        payload = {
            "model": provider.model, "engine": MOTION_VERSION, "image": file_hash(imgs[job["image"]]),
            "chars": [[file_hash(p) for p in chars[c.lower()]["images"]] for c in job.get("characters", [])
                      if c.lower() in chars],
            "shots": [[s["prompt"], round(s["duration_s"], 2)] for s in job["shots"]],
            "duration": job["duration"], "need": job["need_s"],
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def generate(self, plan: dict, provider_name: str = "motion", max_attempts: int = 3,
                 judge: bool = False, judge_client=None, progress=None, provider: Provider | None = None) -> dict:
        provider = provider or make_provider(provider_name, plan, self.dir)
        durations = MODELS[provider.model]["durations"]
        jobs = P.generation_jobs(plan, durations)
        plan["jobs"] = jobs
        out_size = (plan["format"]["width"], plan["format"]["height"])
        results = {}
        for ji, job in enumerate(jobs):
            key = self._job_key(job, plan, provider)
            rec_path = self.dir / "clips" / f"{job['job_id']}_{key}.json"
            if rec_path.exists():
                rec = json.loads(rec_path.read_text())
                if rec.get("accepted") and Path(rec["path"]).exists():
                    results[job["job_id"]] = rec
                    self.note(f"{job['job_id']}: reused accepted clip")
                    if progress:
                        progress(ji + 1, len(jobs), job["job_id"])
                    continue
            feedback = ""
            attempts = []
            accepted = None
            for a in range(max_attempts):
                clip = self.dir / "clips" / f"{job['job_id']}_{key}_a{a}.mp4"
                self.note(f"{job['job_id']}: generating ({job['duration']}s, attempt {a + 1})")
                try:
                    meta = provider.generate(job, plan, clip, attempt=a, feedback=feedback)
                except Exception as e:
                    attempts.append({"attempt": a, "error": str(e)})
                    self.note(f"{job['job_id']}: provider error: {e}")
                    continue
                checks, cm = qc.clip_checks(clip, job, out_size)
                verdict = None
                if judge:
                    verdict = qc.judge_clip(str(clip), job, plan, client=judge_client)
                    checks += qc.judge_checks(verdict)
                ok = qc.gate(checks)
                attempts.append({"attempt": a, "path": str(clip), "meta": cm, "provider": meta,
                                 "checks": checks, "judge": verdict, "passed": ok})
                if ok:
                    accepted = attempts[-1]
                    break
                failed = [c["rule"] for c in checks if not c["passed"] and c["severity"] == "fail"]
                feedback = (verdict or {}).get("fix") or ("; ".join(failed))
                if any(c["rule"] == "long enough" and not c["passed"] for c in checks):
                    bigger = [d for d in durations if d > job["duration"]]
                    if bigger:
                        job["duration"] = bigger[0]
                self.note(f"{job['job_id']}: failed {', '.join(failed)}; retrying")
            if accepted is None:
                # Keep the best attempt so the edit can be previewed, but the episode gate will fail.
                usable = [x for x in attempts if "path" in x]
                best = max(usable, key=lambda x: sum(c["passed"] for c in x["checks"]), default=None)
                rec = {"job": job, "accepted": False, "attempts": attempts,
                       "path": best["path"] if best else None, "meta": best["meta"] if best else None,
                       "checks": best["checks"] if best else [qc.check("clip generated", False, "every attempt failed")]}
            else:
                rec = {"job": job, "accepted": True, "attempts": attempts, "path": accepted["path"],
                       "meta": accepted["meta"], "checks": accepted["checks"]}
            rec_path.write_text(json.dumps(rec, indent=1, default=str))
            results[job["job_id"]] = rec
            if progress:
                progress(ji + 1, len(jobs), job["job_id"])
        self.report["clips"] = {k: {"accepted": v["accepted"], "checks": v["checks"],
                                    "attempts": len(v["attempts"]) if "attempts" in v else 0}
                                for k, v in results.items()}
        P.save(plan, self.dir / "plan.json")
        return results

    # ----------------------------------------------------------------- assembly

    def cut_shots(self, plan: dict, timeline: dict, clips: dict) -> tuple[dict[str, list[str]], list[dict]]:
        fps, W, H = timeline["fps"], timeline["W"], timeline["H"]
        rf = asm.roll_frames(timeline)
        segs = {s["id"]: s for s in timeline["segments"]}
        files: dict[str, list[str]] = {}
        checks = []
        for pseg in plan["segments"]:
            if pseg["kind"] not in ("story", "cold_open"):
                continue
            tseg = segs[pseg["id"]]
            n_seg = tseg["f1"] - tseg["f0"]
            durs = [sh["duration_s"] for sh in pseg["shots"]]
            cum = [0.0]
            for d in durs:
                cum.append(cum[-1] + d)
            edges = [int(round(c / cum[-1] * n_seg)) for c in cum]
            frames = [edges[i + 1] - edges[i] for i in range(len(durs))]
            frames[0] += rf[pseg["id"]][0]
            frames[-1] += rf[pseg["id"]][1]
            jobs = [j for j in plan["jobs"] if j["segment"] == pseg["id"]]
            k = 0
            files[pseg["id"]] = []
            for job in jobs:
                rec = clips.get(job["job_id"])
                if not rec or not rec.get("path"):
                    raise asm.AssemblyError(f"{job['job_id']} has no usable clip")
                srcs = asm.shot_sources(job, rec["meta"])
                for si, src in enumerate(srcs):
                    out = self.dir / "shots" / f"{job['job_id']}_s{si}.mp4"
                    r = asm.render_shot(rec["path"], src, frames[k], fps, W, H, out)
                    files[pseg["id"]].append(r["file"])
                    if r["stretch"] > asm.MAX_STRETCH:
                        checks.append(qc.check(f"{job['job_id']} shot {si + 1} footage length", False,
                                               f"slowed to {1 / r['stretch']:.0%} speed; regenerate a longer clip"))
                    k += 1
        if not checks:
            checks.append(qc.check("footage length", True, "every shot has enough footage at normal speed"))
        return files, checks

    def assemble(self, plan: dict, clips: dict, host_cfg: dict | None, music: str | None = None,
                 judge: bool = False, judge_client=None, final_preset: str = "medium",
                 progress=None) -> dict:
        prof = plan["profile"]
        timeline = asm.build_timeline(plan, plan.get("zoom_s", 1.0), prof.get("end_card_s", 2.0))
        # Landscape panels are shown whole across the middle; their captions go in the band below.
        from .motion import is_panel
        imgs = {im["id"]: im["path"] for im in plan["images"]}
        by_id = {s["id"]: s for s in plan["segments"]}
        for ts in timeline["segments"]:
            ps = by_id.get(ts["id"])
            if ps and ps["kind"] in ("story", "cold_open") and ps.get("image") in imgs and \
                    is_panel(imgs[ps["image"]], timeline["W"], timeline["H"]):
                ts["caption_y"] = 0.745
        self.note("Cutting shots to the narration")
        shot_files, shot_checks = self.cut_shots(plan, timeline, clips)
        self.note("Rendering picture")
        pic = self.dir / "picture.mp4"
        info = asm.render_picture(plan, timeline, shot_files, host_cfg, pic, progress=progress)
        self.note("Mixing audio")
        mix = self.dir / "mix.wav"
        asm.mix_audio(timeline, music, mix)
        timeline["captions"] = info.get("captions", [])
        asm.build_ass(timeline, prof, self.dir / "captions.ass")      # editable copy for other editors
        asm.write_srt(timeline, self.dir / "final.srt")
        self.note("Encoding final")
        final = self.dir / "final.mp4"
        asm.encode_final(pic, mix, final, preset=final_preset)
        consistency = None
        if judge:
            frames = []
            import cv2
            for seg_id, fl in shot_files.items():
                for i, f in enumerate(fl):
                    cap = cv2.VideoCapture(f)
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
                    ok, img = cap.read()
                    cap.release()
                    if ok:
                        frames.append((f"{seg_id} S{i + 1}", cv2.resize(img, (288, 512))))
            consistency = qc.judge_consistency(frames, plan, client=judge_client)
        self.note("Final checks")
        clip_gate = [qc.check(f"clip {k} accepted", v["accepted"], "passed its checks" if v["accepted"]
                              else "failed after every attempt") for k, v in clips.items()]
        final_checks = clip_gate + shot_checks + qc.final_checks(str(final), plan, timeline, consistency)
        passed = qc.gate(final_checks)
        self.report.update({"final": final_checks, "passed": passed, "timeline": _slim(timeline),
                            "tv": info.get("screen"), "file": str(final)})
        (self.dir / "report.json").write_text(json.dumps(self.report, indent=1, default=str))
        self.note("PASSED: ready to download" if passed else "Not ready: see failed checks")
        return self.report


def _slim(tl: dict) -> dict:
    return {"fps": tl["fps"], "frames": tl["frames"], "zoom_s": tl["zoom_s"],
            "segments": [{k: s[k] for k in ("id", "kind", "label", "t0", "t1")} |
                         {"lines": [{k: l[k] for k in ("speaker", "text", "t0", "t1")} for l in s["lines"]]}
                         for s in tl["segments"]]}
