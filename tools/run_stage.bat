@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" call setup.bat
if not exist ".venv\Scripts\python.exe" (
  echo Python environment setup failed.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u tools\launch.py %*
set EXIT_CODE=%ERRORLEVEL%
pause
exit /b %EXIT_CODE%

