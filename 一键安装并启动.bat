@echo off
setlocal
cd /d "%~dp0"
title BoomBeachSonarAuto Setup

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
set "BBMA_EXIT_CODE=%ERRORLEVEL%"

if not "%BBMA_EXIT_CODE%"=="0" (
    echo.
    echo Setup or startup failed. Keep the error message shown above.
    echo See the one-click setup section in README.md for help.
    pause
)

exit /b %BBMA_EXIT_CODE%
