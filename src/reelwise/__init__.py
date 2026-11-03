"""reelwise: measure what a short-form video does, scene by scene, and make episodes."""

import os as _os
from pathlib import Path as _Path

__version__ = "0.2.0"

# A local ffmpeg (fetched by tools/setup_windows.bat) is used when none is on PATH.
for _d in (_Path(_os.environ.get("REELWISE_CACHE", _Path.home() / ".cache" / "reelwise")) / "ffmpeg" / "bin",):
    if _d.exists() and str(_d) not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = str(_d) + _os.pathsep + _os.environ.get("PATH", "")
