$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$main = Join-Path $root "main.py"
$overlay = Join-Path $root "tools\log_overlay.py"
$stdout = Join-Path $root "run_stdout.log"
$stderr = Join-Path $root "run_stderr.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Remove-Item -LiteralPath $stdout, $stderr -ErrorAction SilentlyContinue

$mainProcess = Start-Process `
    -FilePath $python `
    -ArgumentList @($main) `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Milliseconds 750
if ($mainProcess.HasExited) {
    throw "main.py exited during startup (exit code=$($mainProcess.ExitCode)); check run_stderr.log"
}

Start-Process `
    -FilePath $python `
    -ArgumentList @($overlay) `
    -WorkingDirectory $root `
    -WindowStyle Hidden

$mainProcess | Select-Object Id, ProcessName
