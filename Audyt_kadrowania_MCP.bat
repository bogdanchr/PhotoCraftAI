@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" audyt_kadrowania_photocraft.py
if errorlevel 1 (
  echo.
  echo Nie udalo sie polaczyc z MCP. Mozesz uruchomic odczyt zapisanej listy:
  echo "%PYTHON_EXE%" audyt_kadrowania_photocraft.py --offline
)
echo.
pause
