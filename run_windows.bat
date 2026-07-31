@echo off

title BPM Analyzer

cd /d "%~dp0"

REM Create virtual environment only once
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"

REM Install packages only once
if not exist ".venv\installed.flag" (
    echo First run: installing packages...
    python -m pip install -r requirements.txt

    REM Create a marker file
    type nul > ".venv\installed.flag"
)

REM Start server
start "" /min python app.py
start "" http://127.0.0.1:5000
REM Wait for Flask
:wait
powershell -Command ^
"try { Invoke-WebRequest http://127.0.0.1:5000 -UseBasicParsing | Out-Null; exit 0 } catch { exit 1 }"


if errorlevel 1 (
    timeout /t 1 >nul
    goto wait
)



