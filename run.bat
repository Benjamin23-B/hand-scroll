@echo off
cd /d "%~dp0"
echo ========================================
echo        Starting Hand Scroll...
echo ========================================
.\venv\Scripts\python.exe main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Hand Scroll encountered an error.
    pause
)
