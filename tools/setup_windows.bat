@echo off
rem One-click setup for reelwise on Windows. Free: no accounts or keys.
rem Fetches ffmpeg if it is missing, creates .venv and installs everything.
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
set LOG=.local\setup.log
echo reelwise setup %date% %time% > %LOG%
echo Checking ffmpeg...
where ffmpeg >nul 2>&1
if errorlevel 1 powershell -NoProfile -ExecutionPolicy Bypass -File tools\get_ffmpeg.ps1 >> %LOG% 2>&1
echo Creating Python environment...
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv >> %LOG% 2>&1
echo Installing packages (a few minutes the first time)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >> %LOG% 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt >> %LOG% 2>&1
".venv\Scripts\python.exe" -m pip install -e . >> %LOG% 2>&1
echo === done >> %LOG%
echo Setup finished. See .local\setup.log
