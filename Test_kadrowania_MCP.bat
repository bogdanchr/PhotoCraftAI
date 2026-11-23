@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_EXE=venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" test_kadrowania_mcp.py --apply
echo.
pause
