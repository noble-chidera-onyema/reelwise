# reelwise

Shot-by-shot measurement of short-form video, channel outlier analysis from the official YouTube API, and a format-to-script pipeline for creators who run their own recurring characters.

Built for the Worth Telling channel (true stories told by three original characters) and written so any creator with a video file can use it.

## What it does

You give reelwise a video. It reads every frame, finds each cut, and reports the numbers an editor would want: how many shots, how long each one lasts, when the first cut lands, how much the opening moves compared with the rest, when the first sound and first word arrive, how fast people talk, and where the pauses are. With an Anthropic key it also logs what is on screen in each shot (shot type, subject, caption text, expression).

You give it a YouTube channel. Through the YouTube Data API it pulls the channel's uploads, scores every video against that channel's own normal, and shows what the breakout videos have in common: title shape, length, timing, words that recur.

You give it your character profile and a true story. It turns a breakdown into a timing template, has Claude write a script and shot list for your characters on that template, then checks the script against your own rules (length, who opens, which character may speak how often, numbers written out for the voice) before you record anything.

## Studio: make the episode, free

The Studio page takes a reference video, your story images, a scene script, your host scene and the host voices, and returns a finished 9:16 episode. It asks first whether the episode is shaped for a phone (9:16) or as a TV-style 16:9 video in the middle of the feed. The story plays full screen from the hook to the last line, and only on "And that's one worth telling" does the camera pull back out of the TV to the hosts on the couch. Every step is free: the hosts' own Microsoft Edge neural voices (Kokoro offline voices and Chatterbox clones are there too), a depth-based 3D camera that moves through each image, optional free AI hero shots from a Hugging Face Space, lip flaps driven by each host's audio, checks on every clip and on the final file, and a download that unlocks only when the episode passes.

Full guide: [docs/studio.md](docs/studio.md).

## What it does not do

It does not promise views. No tool can, and the outlier report states plainly that it shows associations, not causes.

It does not download or scrape other people's videos. Frame analysis runs on files you have, and YouTube data comes through the official API. YouTube's and Meta's terms forbid scraping, and a creator account is worth more than any shortcut.

It does not copy anyone's content. The template carries timing and structure only. Your characters, your story and your words go into it.

## How it works

Every frame is shrunk to 128x72 and measured once: an HSV colour histogram, brightness, saturation, a colourfulness score (Hasler and Suesstrunk, 2003) and the grey-level difference to the previous frame.

A hard cut is a frame where the histogram distance or the pixel difference jumps well above the surrounding half second. Testing against the neighbourhood is what lets a cut inside a fast, busy sequence register while a whip pan does not. A jump that returns to the earlier picture within four frames is a flash, not a cut. Dissolves and fades are found by comparing frames 0.4 s apart: a transition peaks there while no single step between neighbouring frames is large.

Audio is pulled with ffmpeg and measured in 50 ms blocks. Captions come from an .srt or .vtt file you supply, or from a YouTube video's public captions.

Channel scoring divides each video's views by the median views of the ten uploads published closest to it. A score of 3.0 means three times the channel's normal at that point in its history. Videos under seven days old are left unscored because their counts are still climbing.

Full method, every threshold and the reasoning behind it: [docs/how_it_works.md](docs/how_it_works.md).

## Results

The scene detector was measured on synthetic videos with frame-exact ground truth, tuned on a development set of 20 videos, then run once on 20 held-out videos it had never seen. PySceneDetect, the standard open-source detector, ran on the same files at its default settings as a reference point.

| Held-out set (20 videos, 256 boundaries) | reelwise | PySceneDetect content | PySceneDetect adaptive |
|---|---|---|---|
| Precision | 0.951 | 0.848 | 0.833 |
| Recall | 0.977 | 0.871 | 0.836 |
| F1 | 0.963 | 0.859 | 0.834 |
| Hard cut recall (203) | 0.995 | 0.951 | 0.956 |
| Dissolve and fade recall (53) | 0.906 | 0.566 | 0.377 |
| Cuts between same-palette shots (45) | 0.889 | 0.844 | 0.778 |
| False positives | 13 | 40 | 43 |

Development F1 was 0.966 and held-out F1 0.963, so the thresholds were not fitted to the development videos. Every matched hard cut landed on the exact frame.

Speed: a 38-second vertical video takes 1.3 s to analyse at 360x640 and 7.0 s at 1080x1920 on a two-core machine, without the optional vision step.

63 unit tests cover cut detection, flash rejection, audio onset and pauses, caption parsing, YouTube API parsing and paging, outlier scoring, effect sizes, the script checker, the Studio's scene parser, timing, TV detection and push-through, shot cutting, captions, Kokoro speech, pause-based word timing, the 3D motion engine (including panels kept whole in every move), the free AI hero-shot path (fallback, identity guard, splice), the fal.ai request it builds, and the Claude planner and judge (with stand-in clients).

## Known limitations

The detector's numbers come from synthetic footage. It has hard negatives built in (fast pans, white flashes, captions appearing, same-palette cuts) and is compressed with H.264, but it is not real phone footage. Treat 0.96 as an upper bound until a hand-labelled set of real Shorts has been measured. The benchmark script is ready for one.

Cuts between two shots with near-identical colour and layout are the weakest case: 89% recall on the held-out set.

The YouTube API gives views, likes, comments, titles, durations and timing for other people's videos. It does not give retention, impressions or click-through rate; those exist only in YouTube Studio for the channel owner. Instagram and Facebook expose no comparable public data for accounts you do not manage, so channel analysis is YouTube only. Video breakdowns work for a file from any platform.

The outlier comparison needs at least five overperformers before the report draws any comparison, and even then it is descriptive.

The live API calls (YouTube, Anthropic, caption fetching) are covered by tests using recorded response shapes. Run them once with your own keys before relying on them.

## Run it locally

On Windows, double-click `tools\setup_windows.bat` once. It fetches ffmpeg if you do not have it, creates `.venv` and installs everything. Then double-click `tools\run_checks_windows.bat` to run the tests, voice each host into `voice_check\`, and make the Eugene demo in both shapes, `projects\eugene_vertical\final.mp4` and `projects\eugene_tv\final.mp4`. To start the app: `.venv\Scripts\streamlit run app\streamlit_app.py`.

By hand:

```powershell
cd reelwise
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
copy .env.example .env
```

Put your keys in `.env`. ffmpeg must be on PATH for audio (`winget install Gyan.FFmpeg`).

Dashboard:

```powershell
streamlit run app/streamlit_app.py
```

Command line:

```powershell
reelwise analyse my_episode.mp4 --captions my_episode.srt
reelwise analyse reference.mp4 --youtube https://youtube.com/shorts/VIDEO_ID --vision
reelwise check-video outputs/my_episode.breakdown.json --profile profiles/worth_telling.json
reelwise channel @somechannel
reelwise trending --region GB --category 24
reelwise blueprint outputs/reference.breakdown.json --profile profiles/worth_telling.json --story story.txt
reelwise check-script outputs/story.episode.json --profile profiles/worth_telling.json
```

Re-run the benchmark:

```powershell
pip install -r requirements-dev.txt
python eval/synth.py --out data/synth_holdout --seeds 100-119
python eval/benchmark_scenes.py --data data/synth_holdout
python -m pytest
```

## Keys

ANTHROPIC_API_KEY, from console.anthropic.com, is used for shot descriptions and episode writing. Each video breakdown with vision sends up to 20 frames.

YOUTUBE_API_KEY: in console.cloud.google.com create a project, enable "YouTube Data API v3", then create an API key under Credentials. The free allowance is 10,000 units a day; a 200-video channel report costs about 9.

## Profiles

A profile is a JSON file describing your channel's rules: target length, who speaks first and how fast, how often each character may speak, the gap between lines, the end card. `examples/profile_example.json` shows every field. Your own profiles go in `profiles/`, which git ignores.

## Author

Built by Noble Chidera Onyema, MSc Applied Artificial Intelligence and User Experience, Abertay University, with a background in Mechatronics Engineering.

Email: onyemanoble1628@gmail.com LinkedIn: https://www.linkedin.com/in/noble-chidera-onyema-1a88b53ab/

## Status and licence

Working prototype, version 0.2.0.

Copyright (c) 2026 Noble Chidera Onyema. All rights reserved. See LICENSE.
