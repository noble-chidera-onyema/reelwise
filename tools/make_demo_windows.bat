@echo off
rem Makes the Eugene demo episode only (tests and voice checks are in run_checks_windows.bat).
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
.venv\Scripts\python.exe tools\make_episode.py --script examples\eugene\story.txt --images examples\eugene\images --out projects\eugene_demo_v2 > .local\episode_v3.log 2>&1
echo === done >> .local\episode_v3.log
