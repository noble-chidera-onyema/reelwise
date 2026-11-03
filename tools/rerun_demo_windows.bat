@echo off
rem Swaps in the ffmpeg 8.1 release build, then makes the demo again.
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and $_.CommandLine -like '*make_episode*') -or ($_.Name -eq 'ffmpeg.exe' -and $_.CommandLine -like '*eugene_demo*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get_ffmpeg.ps1" > .local\ffmpeg_update.log 2>&1
del /q projects\eugene_demo_v2\final*.m4a projects\eugene_demo_v2\final*.mp4 2>nul
call "%~dp0make_demo_windows.bat"
