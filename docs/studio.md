# Studio

The Studio turns five inputs into a finished vertical episode: a reference video, your story images, a scene script, your host scene with the TV, and the host voices. It plans every shot, moves the camera through your images in 3D, voices every line, cuts the pictures to the words, composites the TV, captions and mixes the result, and checks all of it before the download unlocks.

Everything runs free. No account, card or paid key is needed for any step.

Open it with `streamlit run app/streamlit_app.py` and pick **Studio** in the sidebar.

## The episode shape

1. **Cold open, inside the TV.** The most dramatic story shot fills the screen with a yellow stamp, and Bex asks the hook question within the first 0.35 s.
2. **Pull back.** The camera pulls out through the TV screen to Ren, Bex and Ozzy on the couch. The TV keeps playing the story; it never freezes on a still.
3. **Push in.** Under the last words of the intro, the camera pushes back through the screen and the story fills the frame again.
4. **The story.** Ren tells it. Each scene's image is filmed with real camera moves, cut to the narration over it. Story characters never speak.
5. **Pull out for the sign-off**, then the follow card.

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

Worth Telling defaults:

| Host | Voice | Rate | Pitch |
|---|---|---|---|
| Ren | en-US-AndrewMultilingualNeural (new: calm, warm, steady) | -8% | +0Hz |
| Bex | en-GB-ThomasNeural (as before, a little quicker) | +15% | -3Hz |
| Ozzy | en-IE-ConnorNeural (as before) | +15% | +5Hz |

Ren has four other storyteller voices in his list to audition: Brian Multilingual, Christopher, Ryan (his old voice) and William (Australian). Press **Hear Ren** to play his first line in the selected voice.

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

Landscape story panels are shown whole across the middle of the frame with a soft blurred copy above and below, instead of being cropped to vertical. The camera still pushes, pans and shows depth on them, but never crops more than about 2% from any edge, so labels and numbers at the edges stay in frame (a test checks this for every camera move). Their captions move into the band below the picture so they never cover the panel's own text.

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
