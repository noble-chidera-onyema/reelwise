@echo off
rem Makes the Eugene demo in both shapes (tests and voice checks are in run_checks_windows.bat).
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
.venv\Scripts\python.exe tools\make_episode.py --layout vertical --script examples\eugene\story.txt --images examples\eugene\images --out projects\eugene_vertical > .local\episode_vertical.log 2>&1
echo === done >> .local\episode_vertical.log
.venv\Scripts\python.exe tools\make_episode.py --layout tv --script examples\eugene\story.txt --images examples\eugene\images --out projects\eugene_tv > .local\episode_tv.log 2>&1
echo === done >> .local\episode_tv.log
