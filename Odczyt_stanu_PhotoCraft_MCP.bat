@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   Odczyt stanu PhotoCraft MCP (bez zmian)
echo ==========================================
echo Folder: %CD%
if exist "venv\Scripts\python.exe" (
  set "PY=venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
echo Python: %PY%
"%PY%" "%~dp0odczyt_stanu_photocraft.py"
echo.
pause
