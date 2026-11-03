@echo off
rem Runs the tests, voices each host, and makes the Eugene demo in both shapes.
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
set PY=.venv\Scripts\python.exe
echo Running tests...
%PY% -m pytest -q > .local\tests_v3.log 2>&1
echo Voicing the hosts...
%PY% tools\voice_check.py > .local\voices_v3.log 2>&1
echo Making the Eugene demo, phone shape (a few minutes)...
%PY% tools\make_episode.py --layout vertical --script examples\eugene\story.txt --images examples\eugene\images --out projects\eugene_vertical > .local\episode_vertical.log 2>&1
echo === done >> .local\episode_vertical.log
echo Making the Eugene demo, TV shape (a few minutes)...
%PY% tools\make_episode.py --layout tv --script examples\eugene\story.txt --images examples\eugene\images --out projects\eugene_tv > .local\episode_tv.log 2>&1
echo === done >> .local\episode_tv.log
echo Finished. Logs are in .local
