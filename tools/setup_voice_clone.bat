@echo off
rem Installs Chatterbox (free voice cloning, MIT) in its own environment, .venv-clone,
rem so its PyTorch install never changes the main app. Large download, one time.
cd /d "%~dp0\.."
if not exist ".local" mkdir ".local"
set LOG=.local\setup_clone.log
echo setup %date% %time% > %LOG%
if not exist ".venv-clone\Scripts\python.exe" py -3.11 -m venv .venv-clone >> %LOG% 2>&1
".venv-clone\Scripts\python.exe" -m pip install --upgrade pip >> %LOG% 2>&1
".venv-clone\Scripts\python.exe" -m pip install chatterbox-tts soundfile >> %LOG% 2>&1
".venv-clone\Scripts\python.exe" -c "import chatterbox, torch; print('chatterbox ok, torch', torch.__version__)" >> %LOG% 2>&1
echo === done >> %LOG%
