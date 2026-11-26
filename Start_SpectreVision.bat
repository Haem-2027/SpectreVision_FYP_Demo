@echo off
rem Double-click to start the SpectreVision demo site, then use http://127.0.0.1:5000
cd /d "%~dp0"
set PY=python
if exist "D:\Work\Computer Vision\Allied Cables Project\venv311\Scripts\python.exe" set PY="D:\Work\Computer Vision\Allied Cables Project\venv311\Scripts\python.exe"
%PY% -c "import flask" 2>nul || (echo Installing Flask, one time only... & %PY% -m pip install flask)
start "" http://127.0.0.1:5000
%PY% app.py
pause
