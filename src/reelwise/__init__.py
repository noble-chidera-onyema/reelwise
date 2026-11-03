"""reelwise: measure what a short-form video does, scene by scene, and make episodes."""

import os as _os
from pathlib import Path as _Path

__version__ = "0.2.0"

# A local ffmpeg (fetched by tools/get_ffmpeg.ps1) is put first on PATH when present.
# ffmpeg-8.1 is the release build; plain "ffmpeg" is where older setups put a development build.
_cache = _Path(_os.environ.get("REELWISE_CACHE", _Path.home() / ".cache" / "reelwise"))
for _name in ("ffmpeg-8.1", "ffmpeg"):
    _d = _cache / _name / "bin"
    if (_d / "ffmpeg.exe").exists() or (_d / "ffmpeg").exists():
        if str(_d) not in _os.environ.get("PATH", ""):
            _os.environ["PATH"] = str(_d) + _os.pathsep + _os.environ.get("PATH", "")
        break
