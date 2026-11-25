@echo off
REM Launch Onmyoji Tactical Helper (auto-elevates via UAC when private .env is present;
REM enters zero-config Public Sandbox Mode when .env is absent). Pass --demo for sandbox.
cd /d "%~dp0"
start "" pythonw run_app.py %*
exit
