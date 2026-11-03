@echo off
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
powershell -NoProfile -ExecutionPolicy Bypass -File tools\get_ffmpeg.ps1 > .local\ffmpeg.log 2>&1
echo === done >> .local\ffmpeg.log
