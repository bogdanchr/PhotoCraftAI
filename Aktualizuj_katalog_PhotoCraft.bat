@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   Aktualizacja katalogu PhotoCraft MCP
echo ==========================================
echo Folder projektu: %CD%
set "PYTHON_EXE="
if exist "%~dp0venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"

if not defined PYTHON_EXE (
    echo BLAD: Nie znaleziono srodowiska Python z MCP.
    echo Szukano w folderze venv obok skryptu.
    pause
    exit /b 1
)
echo Python: %PYTHON_EXE%
"%PYTHON_EXE%" "%~dp0aktualizuj_katalog_photocraft.py"
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
