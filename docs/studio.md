# Studio

The Studio turns five inputs into a finished vertical episode: a reference video, your story images, a scene script, your host scene with the TV, and the host voices. It plans every shot, moves the camera through your images in 3D, voices every line, cuts the pictures to the words, composites the TV, captions and mixes the result, and checks all of it before the download unlocks.

Everything runs free. No account, card or paid key is needed for any step.

Open it with `streamlit run app/streamlit_app.py` and pick **Studio** in the sidebar.

## Shape: phone or TV

Before anything is made, the Studio asks how the episode should be shaped:

- **Phone, vertical 9:16** (1080x1920). Fills the whole screen.
- **TV, 16:9** (1920x1080). Plays in the middle of the feed, like a TV screen, the way many story channels post. Story panels are shown whole (a soft blurred copy fills the sides, nothing is cropped), captions sit low and larger so they read at feed size, and at the sign-off the vertical rooftop set sits in the middle of the frame with a blurred copy either side.

The **Build plan** button stays off until one is chosen. From the command line it is `--layout vertical` or `--layout tv`.

## The episode shape

The default ("hosts seen only at the sign-off"):

1. **Cold open, inside the TV.** The most dramatic story shot fills the screen with a yellow stamp, and Bex asks the hook question within the first 0.35 s.
2. **The story, full screen, start to finish.** Ren tells it; Bex and Ozzy are heard but not seen. Each scene's image is filmed with real camera moves, cut to the narration over it. Story characters never speak.
3. **The reveal.** Just before "And that's one worth telling" (0.35 s ahead of the first word) the camera pulls back out of the TV to Ren, Bex and Ozzy on the rooftop couch, with the story still playing on the screen. The hosts are on screen only for the last 2 to 3 seconds.
4. **The follow card.**

The older shape is still there as **At the start and the end**: the camera pulls out to the hosts for the intro, pushes back through the screen into the story, and pulls out again for the sign-off. In a profile it is `"host_reveal": "classic"`; from the command line, `--reveal classic`.

## Writing the script

```
COLD OPEN
Image: 3
Stamp: 738 days
Bex: Would you live up a tree for seven hundred and thirty eight days?

INTRO
Ren: This is a true story from Eugene, Oregon.

SCENE 1
Image: 1
Characters: Jim
Action: A young man climbs a sweet gum tree at night, rope over his shoulder.
Shots:
- extreme close-up of his face, eyes up at the branches
- low wide shot, the tree vanishing into the dark
Ren: In nineteen ninety seven, a crew came to cut down forty old trees on Broadway.

OUTRO
Ren: And that's one worth telling.
```

- `Name: text` lines are narration, and only your hosts can have them. A line from anyone else is refused, because story characters stay silent.
- `Image:` picks an uploaded image by its number. Leave it out and the planner chooses.
- Shot words steer the camera: "close-up", "face", "hands" or "insert" frame tight on the subject; "track", "pan", "walks" or "drives" move sideways; "top-down", "drone" or "crane" rise; "reveal" or "pull" pull back; "push" or "dramatic" dolly in.
- For a hard name, write `Ren: Barsheem walked in. | Bar-sheem walked in.` The voice reads the part after the bar and the caption shows the real spelling.

**Check script** runs your profile rules before anything is made: who opens, whether the first line is a question, Ren's share of the words (60% minimum), Bex's and Ozzy's line limits, digits the voice would stumble on, and an estimate of the running time.

## Voices (free)

### Who says what

Ren (orange) is the storyteller. He has a calm, steady voice and carries the facts. He tells the story from start to finish, gives the dates, names and twists, and always closes with "And that's one worth telling."

Bex (teal) is the viewer's voice. He sounds quicker and more surprised. He opens with the hook question ("Wait, he did what?"), asks what the audience is thinking ("Okay, but how did he get off the ship?"), and gives the share line ("Send this to the friend who..."). At most three lines.

Ozzy (yellow) is the joker: at most one short funny line per video, without slowing the story.

**Check script** enforces this: Bex opens with a question, Ren speaks at least 60% of the words, Bex has at most three lines, Ozzy at most one, and the last line is Ren's sign-off.

### Engines

| Engine | What it is | Needs |
|---|---|---|
| **edge** (default) | Microsoft neural voices, the same voices your episodes use | internet; no account |
| **chatterbox** | clones a host from a 10-30 s sample, with an emotion slider | `pip install -r requirements-voice-clone.txt` |
| **kokoro** | offline voices that run on any computer | nothing; 350 MB model downloads once |

### The cast's voices

Each voice is chosen by measurement: every candidate reads the same lines, and each read is scored on clarity (the words transcribed back exactly), expression (how far the pitch moves on the key line), the "a" test (the word "a" said as "uh", not "ay"), and distance from the other hosts (at least 4 semitones apart, or a clearly different accent). The old episodes failed the last two: all three hosts shared one voice, and it often said "a" like the letter.

| Host | Status | Voice |
|---|---|---|
| Ren | target saved, shortlist next | deep and warm, about 115 to 125 Hz; calm long-form pace 3.2 words a second, quicker short-form pace 3.9 |
| Bex | target saved, shortlist next | brighter, about 169 Hz (at least 160), not Irish; 3.4 words a second |
| Ozzy | **locked** | en-IE-ConnorNeural, rate +10%, pitch +6Hz: chosen from 36 candidates; clearest, the most expressive punchline (19.4 semitones), a natural 0.77 s beat before the punch, 4 of 4 on the "a" test |

The targets are in `profiles/worth_telling/voice_targets/` (measurements only: no audio from other creators is stored or cloned). A locked voice has its own folder, such as `profiles/worth_telling/voices/ozzy/`, with its settings, its measurements and reference clips of its test lines, and the profile points at it.

To clone your hosts from real recordings, open **Cut voice samples from an old episode**, tick who speaks on each line, save, and set that host's engine to chatterbox.

On Dessa: Block announced it on 23 October 2026 as a research preview. It has no released weights, no API and no text input, so it cannot read a script yet.

## Motion (free)

**Free 3D motion** (default). Each image gets a depth map from MiDaS v2.1 small (MIT, 66 MB, downloaded once, 0.1 s per image). The camera then moves through the picture in 3D: near objects slide and grow faster than the background, so push-ins have depth and side moves show parallax. A slow breathing motion on the foreground and a faint handheld drift keep every frame alive. Close-ups are limited to 2.4x the image's own resolution and lightly sharpened, so they never go soft. Characters and faces are your own image, so they cannot change between shots.

**Free 3D motion + free AI hero shots.** The longest shot in each scene is also sent to a public Hugging Face Space running Wan 2.2 image-to-video. A free Hugging Face account gives about 5 GPU minutes a day, which is roughly 3 AI shots. The AI clip is kept only if it still looks like your image (see checks). If the quota is used up or the Space is busy, that shot keeps its 3D motion and the episode carries on. Add a free token as `HF_TOKEN` in `.env`.

**Draft.** Flat camera moves, fastest, for checking timing.

A 34 s episode renders in about 4 minutes on a 2-core machine, and faster on a typical laptop.

## Captions, pace and panels

Captions, the stamp and the end card are drawn in Python with a bundled font (DejaVu Sans Bold; Arial Black when Windows has it), so they look the same on every computer and on Streamlit Cloud.

A line spoken faster than 3.8 words a second sounds rushed. When that happens the line is voiced again, slower, before anything is cut, and the **no rushed lines** check blocks the download if one still slips through. On the Eugene demo, Ren's sign-off came back from the voice at 5.4 words a second and was fixed this way.

Story panels whose shape differs from the frame are shown whole, with a soft blurred copy filling the rest: a landscape panel in the phone shape becomes a band across the middle, and a 4:3 panel in the TV shape a column down the middle. The camera still pushes, pans and shows depth on them, but never crops more than about 2% from any edge, so labels and numbers at the edges stay in frame (a test checks this for every camera move). Their captions move into the band below the picture so they never cover the panel's own text.

## The host scene

The Worth Telling profile ships with its rig: the rooftop set and each host's closed, open and blink cut-outs, placed in the couch reference's order (Ren, Bex, Ozzy). Mouths open only while that host's own audio is above a speech threshold, measured frame by frame, so lips follow the voice. Blinks are staggered.

The TV screen is found automatically:

| Method | When it is used | Matte |
|---|---|---|
| green | screen is flat bright green | keyed every frame, so anything in front of the TV stays in front |
| panel | screen is a flat cream or white panel (the Worth Telling set) | the screen's exact rounded outline |
| quad | any other screen with a clear outline | the four corners |
| manual | you type the four corners | the four corners |

On the Worth Telling set the panel method finds the screen at (294, 417) to (785, 785), its exact edges.

## What is checked (all free)

Every clip: it decodes, it is long enough, it is not frozen, no black frames, no flicker without a cut, enough resolution, and acceptable framing after the crop to 9:16.

Every free AI hero shot: an identity score compares the clip's colours and layout with your image. Measured on real story art, clips of the same image score 0.43 to 0.86 and clips of different images score -0.03 to 0.39. A clip below 0.40 is thrown away and the 3D motion shot is kept.

The finished episode: running time against the profile; first line by 0.35 s; every line audible in the final mix (each line's window measured, not assumed); every line inside its own pictures; no dead air over 1.2 s; -14 LUFS loudness with true peak at or below -1 dBTP; captions, stamp and end card inside the platform safe area; no black seconds; and every shot playing at normal speed.

The download unlocks when every hard check passes. If one fails, the report names it. You can still unlock the file, but only by ticking that you watched it and accept the failure.

## Measured on this build

Earlier full run on the Worth Telling set with your cut-outs and Kokoro voices: 34.4 s, first line at 0.30 s, all 8 lines audible at -12.7 to -14.1 dBFS, -14.2 LUFS, true peak -1.4 dBTP, 46 caption events inside the safe area, every check passed. Ren speaks 6 of the 8 lines.

Faults found by watching the frames, not by the checks, and fixed:

- the stamp and the end card ran off the edges; both now wrap to fit and are part of the safe-area check
- close-ups magnified low-resolution images until they were soft; magnification is now capped
- the final encode hung on some Windows ffmpeg builds when loudness, mixing and the picture ran in one command; it now runs as three short steps, each with a time limit
- the earlier draft passed every check with tone placeholders and no speech; the "every line audible" check now measures the final file

## Deploying free

The repository includes `packages.txt` (ffmpeg) and `.streamlit/config.toml`, so it deploys on Streamlit Community Cloud as is. Kokoro and the depth model download on first use. Chatterbox needs PyTorch and more memory than the free cloud tier gives, so voice cloning is for running on your own computer.

## Optional paid extras

Kling, Seedance and Veo through fal.ai, ElevenLabs voices, and Claude as shot planner and clip judge are still in the code under **Optional paid extras**, off by default. Nothing in the free pipeline uses them.

## Limits

- The free motion engine moves the camera, not the characters. A character walks or drifts only in a free AI hero shot, or when you supply an image of that moment. Write scenes as a sequence of moments, one image each, the way the Eugene episode panels work.
- The free AI quota is Hugging Face's: about 5 GPU minutes a day per free account.
- Kokoro gives no word timings. Caption timing comes from the audio's own pauses, which is accurate to roughly a word.
- Host mouths have two shapes, open and closed, like the CapCut recipe.
