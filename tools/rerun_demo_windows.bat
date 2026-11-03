@echo off
rem Swaps in the ffmpeg 8.1 release build, then makes the demo again.
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -and $_.CommandLine -like '*make_episode*') -or ($_.Name -eq 'ffmpeg.exe' -and $_.CommandLine -like '*projects*eugene*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0get_ffmpeg.ps1" > .local\ffmpeg_update.log 2>&1
del /q projects\eugene_vertical\final_*.m4a projects\eugene_tv\final_*.m4a 2>nul
call "%~dp0make_demo_windows.bat"
