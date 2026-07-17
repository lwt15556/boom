$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$panel = Join-Path $root "tools\control_panel.py"

if (-not (Test-Path -LiteralPath $python)) {
    $setup = Join-Path $root "setup.ps1"
    if (-not (Test-Path -LiteralPath $setup)) {
        throw "Python virtual environment not found: $python"
    }
    & $setup -SkipLaunch
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Start-Process `
    -FilePath $python `
    -ArgumentList @($panel) `
    -WorkingDirectory $root `
    -WindowStyle Hidden
