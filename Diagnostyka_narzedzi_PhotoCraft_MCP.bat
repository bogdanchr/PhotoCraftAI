@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   Diagnostyka narzedzi PhotoCraft MCP
echo ==========================================
echo Folder projektu: %CD%
set "PYTHON_EXE="
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

if not defined PYTHON_EXE (
    echo BLAD: Nie znaleziono Pythona ze srodowiskiem MCP.
    echo Szukano w folderze venv obok skryptu.
    pause
    exit /b 1
)
echo Python: %PYTHON_EXE%
"%PYTHON_EXE%" "%~dp0diag_narzedzia_mcp.py"
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
