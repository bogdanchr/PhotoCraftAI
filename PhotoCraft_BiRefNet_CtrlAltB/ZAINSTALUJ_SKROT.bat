@echo off
setlocal DisableDelayedExpansion
title Instalacja skrotu BiRefNet Ctrl+Alt+B
set "BASE=%~dp0.."
set "SRC=%~dp0Uruchom_BiRefNet.ps1"
set "DEST=%BASE%\shortcuts"
set "TARGET=%DEST%\Uruchom_BiRefNet.ps1"
set "PY=%BASE%\venv\Scripts\python.exe"
set "BRIDGE=%BASE%\comfy_bridge\photocraft_comfy_birefnet_v1_4.py"
echo.
echo === PhotoCraft AI - skrot Ctrl+Alt+B ===
if not exist "%SRC%" goto missing_source
if not exist "%PY%" goto missing_python
if not exist "%BRIDGE%" goto missing_bridge
if not exist "%DEST%" mkdir "%DEST%"
copy /Y "%SRC%" "%TARGET%" >nul
if errorlevel 1 goto copy_fail
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$desktop=[Environment]::GetFolderPath('Desktop'); $link=Join-Path $desktop 'PhotoCraft - Wytnij tlo BiRefNet.lnk'; $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut($link); $s.TargetPath=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'; $s.Arguments='-NoProfile -ExecutionPolicy Bypass -File '+[char]34+'%TARGET%'+[char]34; $s.WorkingDirectory='%BASE%'; $s.Hotkey='CTRL+ALT+B'; $s.WindowStyle=7; $s.Description='BiRefNet - wycinanie tla z aktywnego PhotoCraft'; $s.Save(); Write-Host ('Utworzono: '+$link)"
if errorlevel 1 goto shortcut_fail
echo.
echo [OK] Zainstalowano skrot na pulpicie.
echo [OK] Ctrl+Alt+B - wycinanie tla BiRefNet.
echo Programy PhotoCraft, MCP i ComfyUI musza juz dzialac (launcher v3).
echo Jesli skrot nie zadziala z klawiatury, sprawdz dwuklikiem ikone na pulpicie.
echo.
pause
exit /b 0
:missing_source
echo [BLAD] Brak Uruchom_BiRefNet.ps1 obok tego BAT.
goto failed
:missing_python
echo [BLAD] Brak %PY%
goto failed
:missing_bridge
echo [BLAD] Brak %BRIDGE%
goto failed
:copy_fail
echo [BLAD] Nie mozna skopiowac pliku pomocniczego.
goto failed
:shortcut_fail
echo [BLAD] Nie mozna utworzyc skrotu na pulpicie.
:failed
pause
exit /b 1
