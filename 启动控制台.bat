@echo off
setlocal
cd /d "%~dp0"
title BBMA Control Panel

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Python environment is not installed.
    echo Please run the one-click setup script first.
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_control_panel.ps1"
set "PANEL_EXIT_CODE=%ERRORLEVEL%"

if not "%PANEL_EXIT_CODE%"=="0" (
    echo.
    echo Control panel startup failed.
    pause
)

exit /b %PANEL_EXIT_CODE%
