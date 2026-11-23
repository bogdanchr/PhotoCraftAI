@echo off
setlocal DisableDelayedExpansion
title PhotoCraft AI - Start wszystkiego v3

rem Jeden launcher v3: PhotoCraft 8765, MCP 8766, ComfyUI 8188, Gradio 7860.
rem Programy juz uruchomione sa pomijane.
set "BASE=%~dp0"

rem Domyslne ustawienia
set "PC=C:\Program Files\PhotoCraft"
set "EXPORT_DIR=%BASE%exports"
set "IMPORT_DIR=%BASE%import"
set "COMFY_ROOT="

rem Wczytaj lokalne ustawienia uzytkownika
if exist "%BASE%local_config.bat" call "%BASE%local_config.bat"

rem Sciezki i zmienne wyliczane po wczytaniu konfiguracji
set "TOKEN_FILE=%BASE%photocraft-token.txt"
set "PHOTOCRAFT_EXPORT_DIR=%EXPORT_DIR%"
set "PHOTOCRAFT_IMPORT_DIR=%IMPORT_DIR%"
set "COMFY_PY=%COMFY_ROOT%\python_embeded\python.exe"
set "PHOTOCRAFT_COMFY_PYTHON=%COMFY_PY%"
set "COMFY_MAIN=%COMFY_ROOT%\ComfyUI\main.py"
set "LOG=%TEMP%\photocraft_ai_start_status.txt"

>"%LOG%" echo PhotoCraft AI - raport uruchamiania %DATE% %TIME%
echo.
echo === PhotoCraft AI: start zestawu ===
echo.

call :start_photo
call :start_mcp
call :start_comfy
call :start_gradio

echo.
echo === Oczekiwanie na gotowosc ComfyUI i Gradio (maks. 90 s) ===
call :wait_services 90

echo.
echo === Status portow ===
call :status 8765 PhotoCraft
call :status 8766 MCP
call :status 8188 ComfyUI
call :status 7860 Gradio

echo.
echo Raport: %LOG%
echo Status OK oznacza, ze program nasluchuje na swoim porcie.
echo Ponowne uruchomienie tego BAT nie powinno otworzyc drugich kopii.
echo Nie zamykaj okien MCP, ComfyUI i Gradio podczas pracy.
echo.
pause
exit /b 0

:start_photo
echo [1/4] PhotoCraft
call :port_open 8765
if "%PORT_OPEN%"=="1" goto photo_ready
rem Nie otwieraj drugiej kopii, jesli okno GUI juz dziala bez portu kontrolnego.
tasklist /FI "IMAGENAME eq photocraft.exe" /NH 2>nul | findstr /I /C:"photocraft.exe" >nul
if not errorlevel 1 goto photo_running_without_port
if not exist "%PC%\photocraft.exe" goto photo_missing
if not exist "%TOKEN_FILE%" goto token_missing
if not exist "%EXPORT_DIR%" mkdir "%EXPORT_DIR%"
if not exist "%IMPORT_DIR%" mkdir "%IMPORT_DIR%"
call :load_token
if not defined PHOTOCRAFT_CONTROL_TOKEN goto token_empty
echo   [START] PhotoCraft GUI
>>"%LOG%" echo PhotoCraft: start
start "PhotoCraft GUI" "%PC%\photocraft.exe" --control 8765 --control-token "%PHOTOCRAFT_CONTROL_TOKEN%" --automation-write-root "%EXPORT_DIR%" --automation-read-root "%IMPORT_DIR%"
call :wait_port 8765 15
exit /b 0
:photo_ready
echo   [OK] Port 8765 juz dziala
>>"%LOG%" echo PhotoCraft: already running
exit /b 0
:photo_running_without_port
echo   [UWAGA] PhotoCraft jest otwarty, ale port 8765 jest nieaktywny.
echo           Zamknij tamta kopie i ponownie uruchom launcher.
>>"%LOG%" echo PhotoCraft: process exists, port 8765 unavailable
exit /b 0
:photo_missing
echo   [BLAD] Brak "%PC%\photocraft.exe"
>>"%LOG%" echo PhotoCraft: executable missing
exit /b 0
:token_missing
echo   [BLAD] Nie znaleziono pliku tokenu: "%TOKEN_FILE%"
>>"%LOG%" echo Token: file missing
exit /b 0
:token_empty
echo   [BLAD] Plik tokenu jest pusty
>>"%LOG%" echo Token: empty
exit /b 0

:start_mcp
echo [2/4] Serwer MCP
call :port_open 8766
if "%PORT_OPEN%"=="1" goto mcp_ready
call :port_open 8765
if not "%PORT_OPEN%"=="1" goto mcp_no_photo
if not exist "%PC%\photocraft-cli.exe" goto mcp_no_cli
if not exist "%BASE%venv\Scripts\mcp-proxy.exe" goto mcp_no_venv
if not exist "%TOKEN_FILE%" goto token_missing
call :load_token
if not defined PHOTOCRAFT_CONTROL_TOKEN goto token_empty
echo   [START] MCP proxy
>>"%LOG%" echo MCP: start
rem Nie podawaj --automation-root do cli mcp --bridge (odrzucane przez PhotoCraft).
start "PhotoCraft MCP" cmd /k ""%BASE%venv\Scripts\mcp-proxy.exe" --host 127.0.0.1 --port 8766 --pass-environment -- "%PC%\photocraft-cli.exe" mcp --bridge 127.0.0.1:8765"
call :wait_port 8766 15
exit /b 0
:mcp_ready
echo   [OK] Port 8766 juz dziala
>>"%LOG%" echo MCP: already running
exit /b 0
:mcp_no_photo
echo   [UWAGA] Nie mozna wystartowac MCP bez PhotoCraft na porcie 8765
>>"%LOG%" echo MCP: skipped - PhotoCraft control unavailable
exit /b 0
:mcp_no_cli
echo   [BLAD] Brak photocraft-cli.exe
>>"%LOG%" echo MCP: CLI missing
exit /b 0
:mcp_no_venv
echo   [BLAD] Brak MCP Proxy w "%BASE%venv\Scripts"
>>"%LOG%" echo MCP: venv missing
exit /b 0

:start_comfy
echo [3/4] ComfyUI
call :port_open 8188
if "%PORT_OPEN%"=="1" goto comfy_ready
rem Nie odpalaj ponownie podczas ladowania: szukaj procesu po sciezce do main.py.
rem PowerShell odpalany poza blokiem IF, aby blad skladni nie przerwal calego BAT.
if not defined COMFY_ROOT goto comfy_no_config
set "COMFY_ROOT_CHECK=%COMFY_ROOT%"
powershell.exe -NoProfile -NonInteractive -Command "$p=Get-CimInstance Win32_Process -Filter 'Name = ''python.exe'' OR Name = ''pythonw.exe''' -ErrorAction SilentlyContinue; if (@($p | Where-Object { $_.CommandLine -match 'ComfyUI[\\/]main\.py' -and $_.CommandLine -like ('*' + $env:COMFY_ROOT_CHECK + '*') }).Count -gt 0) { exit 0 } else { exit 1 }" >nul 2>nul
if not errorlevel 1 goto comfy_loading
if not exist "%COMFY_PY%" goto comfy_no_python
if not exist "%COMFY_MAIN%" goto comfy_no_main
echo   [START] ComfyUI
>>"%LOG%" echo ComfyUI: start
start "ComfyUI" /D "%COMFY_ROOT%" "%COMFY_PY%" -s "ComfyUI\main.py" --windows-standalone-build
exit /b 0
:comfy_ready
echo   [OK] Port 8188 juz dziala
>>"%LOG%" echo ComfyUI: already running
exit /b 0
:comfy_loading
echo   [OK] Proces ComfyUI juz dziala - moze jeszcze ladowac modele
>>"%LOG%" echo ComfyUI: process already running, waiting for port
exit /b 0
:comfy_no_config
echo   [BLAD] Ustaw COMFY_ROOT w local_config.bat
>>"%LOG%" echo ComfyUI: configuration missing
exit /b 0
:comfy_no_python
echo   [BLAD] Brak: "%COMFY_PY%"
>>"%LOG%" echo ComfyUI: python missing
exit /b 0
:comfy_no_main
echo   [BLAD] Brak: "%COMFY_MAIN%"
>>"%LOG%" echo ComfyUI: main.py missing
exit /b 0

:start_gradio
echo [4/4] PhotoCraft AI - Gradio
call :port_open 7860
if "%PORT_OPEN%"=="1" goto gradio_ready
if not exist "%BASE%\venv\Scripts\python.exe" goto gradio_no_python
if not exist "%BASE%\app.py" goto gradio_no_app
echo   [START] Gradio
>>"%LOG%" echo Gradio: start
start "PhotoCraft AI - Gradio" /D "%BASE%" "%BASE%\venv\Scripts\python.exe" "%BASE%\app.py"
exit /b 0
:gradio_ready
echo   [OK] Port 7860 juz dziala
>>"%LOG%" echo Gradio: already running
exit /b 0
:gradio_no_python
echo   [BLAD] Brak Pythona w %BASE%\venv
>>"%LOG%" echo Gradio: python missing
exit /b 0
:gradio_no_app
echo   [BLAD] Brak: "%BASE%\app.py"
>>"%LOG%" echo Gradio: app.py missing
exit /b 0

:load_token
if defined PHOTOCRAFT_CONTROL_TOKEN exit /b 0
set /p PHOTOCRAFT_CONTROL_TOKEN=<"%TOKEN_FILE%"
exit /b 0

:port_open
set "PORT_OPEN=0"
netstat -ano -p TCP | findstr /R /C:":%~1 .*LISTENING" >nul 2>nul
if not errorlevel 1 set "PORT_OPEN=1"
exit /b 0

:wait_port
set /a TRY=0
:wait_again
call :port_open %~1
if "%PORT_OPEN%"=="1" goto wait_ok
set /a TRY+=1
if %TRY% GEQ %~2 goto wait_timeout
timeout /t 2 /nobreak >nul
goto wait_again
:wait_ok
echo   [OK] Port %~1 uruchomiony
>>"%LOG%" echo Port %~1: ready
exit /b 0
:wait_timeout
echo   [UWAGA] Port %~1 nie jest jeszcze aktywny
>>"%LOG%" echo Port %~1: timeout
exit /b 0

:status
call :port_open %~1
if "%PORT_OPEN%"=="1" goto status_ok
echo   [CZEKA/BLAD] %~2 - %~1
>>"%LOG%" echo %~2 %~1: not listening
exit /b 0
:status_ok
echo   [OK] %~2 - %~1
>>"%LOG%" echo %~2 %~1: listening
exit /b 0

:wait_services
rem Oczekuj na oba serwisy jednoczesnie, bez niepotrzebnej zwloki.
set /a WAIT_ELAPSED=0
:wait_services_loop
call :port_open 8188
set "COMFY_READY=%PORT_OPEN%"
call :port_open 7860
set "GRADIO_READY=%PORT_OPEN%"
if "%COMFY_READY%"=="1" if "%GRADIO_READY%"=="1" goto wait_services_done
if %WAIT_ELAPSED% GEQ %~1 goto wait_services_timeout
if %WAIT_ELAPSED%==0 echo   [CZEKA] Sprawdzam porty 8188 i 7860...
if %WAIT_ELAPSED%==10 echo   [CZEKA] 10 s - programy nadal sie uruchamiaja...
if %WAIT_ELAPSED%==30 echo   [CZEKA] 30 s - programy nadal sie uruchamiaja...
if %WAIT_ELAPSED%==60 echo   [CZEKA] 60 s - programy nadal sie uruchamiaja...
timeout /t 2 /nobreak >nul
set /a WAIT_ELAPSED+=2
goto wait_services_loop
:wait_services_done
echo   [OK] ComfyUI i Gradio gotowe po %WAIT_ELAPSED% s.
>>"%LOG%" echo ComfyUI and Gradio: ready after %WAIT_ELAPSED% s
exit /b 0
:wait_services_timeout
echo   [UWAGA] Po %WAIT_ELAPSED% s co najmniej jeden serwis jeszcze nie nasluchuje.
echo           Zajrzyj do okien ComfyUI i Gradio po ewentualny blad.
>>"%LOG%" echo Waiting timed out after %WAIT_ELAPSED% s; ComfyUI=%COMFY_READY%; Gradio=%GRADIO_READY%
exit /b 0
