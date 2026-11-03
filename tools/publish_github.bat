@echo off
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
echo publish %date% %time% > .local\publish.log
powershell -NoProfile -ExecutionPolicy Bypass -File tools\publish_github.ps1 >> .local\publish.log 2>&1
echo Finished. See .local\publish.log
