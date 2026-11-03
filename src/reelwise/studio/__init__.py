"""Studio: turn a reference video, story images, a scene script and host voices
into a finished episode.

Pipeline, in order:
    plan.py       parse the scene text, plan every shot, estimate cost
    voices.py     narrate each line and keep word-level timing
    providers.py  turn each story image into motion (fal.ai, or a free offline draft)
    tv.py         find the TV screen in the host scene, push through it into the story
    qc.py         check every clip and the final render; nothing ships unchecked
    assemble.py   cut, caption, mix and export
    pipeline.py   runs the steps above with caching, retries and a written report
"""
