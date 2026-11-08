@echo off
rem ---------------------------------------------------------------------------
rem Launcher for the Document Card Extractor.
rem Runs main.py with the Python interpreter that is on the PATH; all packages
rem are expected in that interpreter's normal site-packages directory.
rem ---------------------------------------------------------------------------
setlocal

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

python main.py %*
if errorlevel 1 (
    echo.
    echo The application exited with an error.
    pause
)

endlocal

