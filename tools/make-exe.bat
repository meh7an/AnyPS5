@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0make-exe.ps1" %*
pause
