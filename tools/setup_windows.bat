@echo off
rem One-click setup for reelwise on Windows. Free: no accounts or keys.
rem Creates .venv, installs everything, and fetches ffmpeg if it is missing.
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
set LOG=.local\setup.log
echo reelwise setup %date% %time% > %LOG%

where ffmpeg >nul 2>&1
if errorlevel 1 (
  if not exist "%USERPROFILE%\.cache\reelwise\ffmpeg\bin\ffmpeg.exe" (
    echo Fetching ffmpeg ^(free, about 90 MB^)...
    echo === ffmpeg download >> %LOG%
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='Stop'; $d=Join-Path $env:USERPROFILE '.cache\reelwise'; New-Item -ItemType Directory -Force -Path $d | Out-Null; $z=Join-Path $d 'ffmpeg.zip'; Invoke-WebRequest -UseBasicParsing 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile $z; $t=Join-Path $d 'ffmpeg_tmp'; if (Test-Path $t) {Remove-Item -Recurse -Force $t}; Expand-Archive $z $t; $src=Get-ChildItem $t -Directory | Select-Object -First 1; $f=Join-Path $d 'ffmpeg'; if (Test-Path $f) {Remove-Item -Recurse -Force $f}; Move-Item $src.FullName $f; Remove-Item $z; Remove-Item -Recurse -Force $t" >> %LOG% 2>&1
  )
)
"%USERPROFILE%\.cache\reelwise\ffmpeg\bin\ffmpeg.exe" -version 2>&1 | findstr /b "ffmpeg version" >> %LOG%

echo Creating Python environment...
echo === venv >> %LOG%
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv >> %LOG% 2>&1
echo Installing packages (a few minutes the first time)...
echo === pip >> %LOG%
".venv\Scripts\python.exe" -m pip install --upgrade pip >> %LOG% 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt >> %LOG% 2>&1
".venv\Scripts\python.exe" -m pip install -e . >> %LOG% 2>&1
echo === done >> %LOG%
echo Setup finished. See .local\setup.log
