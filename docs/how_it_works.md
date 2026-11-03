# How reelwise works

This page covers each measurement, the thresholds, how they were chosen, and where the numbers stop being trustworthy.

## 1. Reading the video

OpenCV decodes the file once. Each frame is shrunk to 128x72 with area averaging and measured:

- an HSV histogram with 8 hue, 4 saturation and 4 value bins (128 bins, normalised to sum to 1)
- mean brightness (the V channel, 0 to 255) and mean saturation
- colourfulness, from Hasler and Suesstrunk (2003): the spread and mean of the red-green and yellow-blue opponent channels
- motion: the mean absolute grey-level difference to the previous frame, 0 to 255

A 512-pixel-wide copy of one frame every half second is kept in memory, so keyframes can be sent to a vision model without a second decode.

Shrinking first is a deliberate trade. Shot boundaries and colour statistics barely change at 128x72, and the cost drops by about two orders of magnitude against full HD. Fine detail such as small caption text is not measured here; the vision step handles that.

## 2. Finding cuts

For frame t the detector computes two step sizes: the Bhattacharyya distance between the histograms of t-1 and t, and the motion value at t. Each is compared with its own average over the surrounding half second, leaving out t-1 to t+1 so a cut does not raise its own baseline.

Frame t is a cut when either test passes:

- histogram distance at least 0.24 and at least twice the local average (plus 0.02)
- pixel difference at least 14 and at least twice the local average (plus 1), with the histogram distance at least half its floor

The pixel test catches cuts between two shots that share a palette, where the histogram hardly moves but everything in the picture changes position. The neighbourhood ratio is what keeps a whip pan from firing: during a fast pan every frame differs a lot from the last, so no single frame stands out.

Flashes. If the picture returns to frame t-1 (histogram distance under 35% of the jump) within four frames, the jump is discarded as a flash.

Dissolves and fades. Frame t is compared with frame t+2w, where w is 0.4 s. A gradual transition shows as a peak in that wide comparison above 0.45, with no single neighbour-to-neighbour step above 1.5 times the hard-cut floor. The middle frame must also look like a mix of the two ends, or be at least twice as dark as both (a fade through black). Peaks within w of a hard cut are ignored, since every hard cut produces one.

Two boundaries closer than 0.3 s are merged, keeping the stronger.

## 3. Measuring the detector

Real short-form footage with frame-exact labels was not available, so `eval/synth.py` manufactures it. Each test video is 20 to 40 seconds of vertical 360x640 footage at 30 fps, built from scenes of 0.6 to 4 seconds. A scene is a textured background in a random three-colour palette, one to three moving shapes, and a slow camera pan that bounces at the edges. Transitions are 75% hard cuts, 12% dissolves and 13% fades through black, each 0.5 s long.

Hard negatives appear on purpose. One scene in six pans at 12 to 22 pixels per frame. One in six contains a two-frame white flash. One in six has a caption box appear mid-shot. One transition in five joins two scenes with the same palette. Every file is re-encoded with H.264 at CRF 30 so compression artefacts are present.

Matching: a detected boundary within 2 frames of a true hard cut, or within 0.5 s of a dissolve's midpoint, counts as a hit. Each detection can match one true boundary. Anything unmatched is a false positive.

Thresholds were chosen by grid search on 20 development videos (seeds 0 to 19). F1 stayed between 0.95 and 0.97 for every neighbouring setting tried, so the result does not hinge on one exact value. The held-out set (seeds 100 to 119) was generated after the thresholds were fixed and run once.

| | Dev reelwise | Held-out reelwise | Held-out PySceneDetect content | Held-out PySceneDetect adaptive |
|---|---|---|---|---|
| Boundaries | 274 | 256 | 256 | 256 |
| Precision | 0.941 | 0.951 | 0.848 | 0.833 |
| Recall | 0.993 | 0.977 | 0.871 | 0.836 |
| F1 | 0.966 | 0.963 | 0.859 | 0.834 |
| Hard cut recall | 0.995 | 0.995 | 0.951 | 0.956 |
| Dissolve and fade recall | 0.985 | 0.906 | 0.566 | 0.377 |
| Same-palette cut recall | 0.969 | 0.889 | 0.844 | 0.778 |
| False positives | 17 | 13 | 40 | 43 |

Two caveats on the comparison. PySceneDetect ran at its defaults and was not tuned for this footage, while reelwise was tuned on footage from the same generator. And dissolve recall dropped from 0.985 to 0.906 between development and held-out, the largest gap in the table, so gradual transitions are the part most likely to do worse on real video.

One bug in the generator was found and fixed during development: the first version wrapped the camera pan around the texture edge, which put an unlabelled jump into some scenes. The development set was regenerated after the fix and every number above comes from the corrected generator.

The raw results are in `docs/eval_scenes_dev.json` and `docs/eval_scenes_holdout.json`.

## 4. Audio

ffmpeg extracts 16 kHz mono PCM. Loudness is RMS per 50 ms block in dBFS. It is close to LUFS for speech but not the same thing, and the reports always say dBFS.

A block counts as sound when it is above the file's own quiet floor: 12 dB above its quietest tenth, capped at 3 dB below its median, never below -50 dBFS. The cap exists because a steady music bed would otherwise have its quietest tenth equal to its median and count as silence. From this come the first-sound time, the share of the video with sound, the longest pause after the first sound, and the count of pauses over 0.3 s. Silence after the last sound is the ending and is not counted as a pause.

## 5. Speech

Captions give words and timing. From them: when the first word starts, whether the first line is a question, words per second while someone is talking, how many lines are questions, how many contain digits, and how many are a single word.

Words per second while talking is taken over caption durations. Caption files pad lines unevenly, so the figure is approximate. It is still the right number to compare against a voice setting: Ren at 1.08 speed measures about 3.0 words per second, Bex at 1.05 about 3.4.

## 6. Shot descriptions

With `--vision`, the opening frame and one keyframe per shot (the middle frame, capped at 20 per video and spread evenly when there are more shots) go to Claude in one request. The prompt asks for visible facts only, in fixed JSON fields, and to answer "unclear" where it cannot tell. Model output is descriptive and can be wrong. Nothing in the numeric pipeline depends on it.

## 7. Channel outliers

The YouTube Data API returns each upload's publish time, duration and public counts. Each settled video (seven days or older) is divided by the median views of the ten settled uploads published closest to it in time.

Comparing against neighbours in time handles channel growth: a channel that went from 500 to 50,000 views per video is not treated as if its early uploads all failed. Scores of 2.0 and above are "overperformer", 0.5 and below "underperformer".

Overperformers are compared with the rest on title length, words, question marks, digits, all-caps words, "you" and first-person words, video length, short form (180 s or under), publish hour in UTC, tags, description length and engagement rate. Each comparison reports the two medians (or shares) and Cliff's delta, an effect size from -1 to +1 that needs no assumption about the distribution. Title words are ranked by how much more often they appear in overperformer titles.

With fewer than five overperformers the report says it cannot draw a comparison. With more, it is still a description of one channel's history. Topic, timing in the news cycle, the thumbnail and luck are not in the data.

## 8. From breakdown to episode

`build_format` keeps the structure of a reference video and nothing else: each beat's position as a percentage of the running time, its length, how it begins, its energy relative to the other beats, the shot type if the vision step ran, and how many words were spoken in it. The reference's words are not passed on.

`write_episode` sends that template, your profile and your story text to Claude, with an instruction to use only facts present in the story and to list every fact it relied on. Check those facts before publishing. A true-story channel lives or dies on them.

`check_script` estimates each line's start and end from its word count and the speaker's words-per-second in the profile, adds the gaps and the end card, and tests the result against the profile: total length, first speaker, first line a question, per-character line limits and line lengths, minimum words per line, unknown speakers, and digits that the voice would read awkwardly.

`check_video` applies the same profile to a finished render: running time, how early the first line starts (from captions, or the first audible sound when there are none), and orientation.

## 9. What is not measured

Retention, impressions and click-through rate for videos you do not own. Thumbnails as shown in the feed. Music identification. On-screen text without the vision step. Anything on Instagram or Facebook beyond a video file you have.
