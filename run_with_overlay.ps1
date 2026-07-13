$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$main = Join-Path $root "main.py"
$overlay = Join-Path $root "tools\log_overlay.py"
$stdout = Join-Path $root "run_stdout.log"
$stderr = Join-Path $root "run_stderr.log"

Remove-Item -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue

Start-Process `
    -FilePath $python `
    -ArgumentList @($overlay) `
    -WorkingDirectory $root `
    -WindowStyle Hidden

Start-Sleep -Milliseconds 500

Start-Process `
    -FilePath $python `
    -ArgumentList @($main) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru |
    Select-Object Id, ProcessName
