@echo off
rem Runs the tests, voices each host, and makes the Eugene demo episode.
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
set PY=.venv\Scripts\python.exe
echo Running tests...
%PY% -m pytest -q > .local\tests_v2.log 2>&1
echo Voicing the hosts...
%PY% tools\voice_check.py > .local\voices_v2.log 2>&1
echo Making the Eugene demo episode (a few minutes)...
%PY% tools\make_episode.py --script examples\eugene\story.txt --images examples\eugene\images --out projects\eugene_demo_v2 > .local\episode_v2.log 2>&1
echo === done >> .local\episode_v2.log
echo Finished. Logs are in .local
