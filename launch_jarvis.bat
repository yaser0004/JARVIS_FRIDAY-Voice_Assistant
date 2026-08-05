@echo off
setlocal

rem Always run from this script's own folder, regardless of where it's launched from.
cd /d "%~dp0"

set "VENV_PYTHON=%~dp0.venv_run\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [JARVIS launcher] Could not find "%VENV_PYTHON%".
    echo Set up the virtual environment first, e.g.:
    echo     python -m venv .venv_run
    echo     .venv_run\Scripts\python.exe -m pip install -r requirements.txt
    echo     .venv_run\Scripts\python.exe setup_models.py
    pause
    exit /b 1
)

echo [JARVIS launcher] Starting JARVIS...
"%VENV_PYTHON%" main.py

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [JARVIS launcher] JARVIS exited with code %EXIT_CODE%.
    pause
)

endlocal
