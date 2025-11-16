@echo off
REM Batch file wrapper for starting the local server
REM This file calls the PowerShell script

echo Starting longhl104.github.io local server...
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"
