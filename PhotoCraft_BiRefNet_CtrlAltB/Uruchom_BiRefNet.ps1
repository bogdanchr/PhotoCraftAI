# PhotoCraft AI - BiRefNet hotkey runner
# Uruchamiany przez skrot Windows Ctrl+Alt+B. Nie zmienia launchera v3.
$ErrorActionPreference = 'Stop'
$base = Split-Path -Parent $PSScriptRoot
# Katalogi domyslne
$env:PHOTOCRAFT_EXPORT_DIR = Join-Path $base 'exports'
$env:PHOTOCRAFT_IMPORT_DIR = Join-Path $base 'import'

# Lokalna konfiguracja uzytkownika
$config = Join-Path $base 'local_config.bat'

if (Test-Path -LiteralPath $config) {
    $settings = & cmd.exe /d /c "`"call `"`"$config`"`" >nul && set`""

    foreach ($line in $settings) {
        if ($line -match '^(IMPORT_DIR|EXPORT_DIR)=(.*)$') {
            if ($matches[1] -eq 'IMPORT_DIR') {
                $env:PHOTOCRAFT_IMPORT_DIR = $matches[2]
            }
            else {
                $env:PHOTOCRAFT_EXPORT_DIR = $matches[2]
            }
        }
    }
}
$python = Join-Path $base 'venv\Scripts\python.exe'
$script = Join-Path $base 'comfy_bridge\photocraft_comfy_birefnet_v1_4.py'
$logs = Join-Path $base 'comfy_bridge\logs'
$mutex = $null
$locked = $false
try {
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Brak Pythona: $python" }
    if (-not (Test-Path -LiteralPath $script -PathType Leaf)) { throw "Brak skryptu: $script" }
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $mutex = New-Object System.Threading.Mutex($false, 'Local\PhotoCraft_BiRefNet_Running')
    $locked = $mutex.WaitOne(0)
    if (-not $locked) { exit 0 }  # Ignoruj ponowne Ctrl+Alt+B podczas wycinania.
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $out = Join-Path $logs "birefnet_$stamp.log"
    $err = Join-Path $logs "birefnet_$stamp.err.log"
    # Uruchomienie synchroniczne: mutex chroni przez caly czas dzialania Pythona.
    $proc = Start-Process -FilePath $python -ArgumentList ('"' + $script + '"') -WorkingDirectory $base -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden
    if ($proc.ExitCode -ne 0) {
        Add-Content -LiteralPath $err -Value "`nKod wyjscia: $($proc.ExitCode)" -Encoding UTF8
    }
} catch {
    try {
        New-Item -ItemType Directory -Path $logs -Force | Out-Null
        Add-Content -Path (Join-Path $logs 'birefnet_launcher_errors.log') -Value "$(Get-Date -Format s) : $($_.Exception.Message)" -Encoding UTF8
    } catch {}
} finally {
    if ($locked -and $null -ne $mutex) { $mutex.ReleaseMutex() }
    if ($null -ne $mutex) { $mutex.Dispose() }
}
