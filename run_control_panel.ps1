$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$panel = Join-Path $root "tools\control_panel.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Start-Process `
    -FilePath $python `
    -ArgumentList @($panel) `
    -WorkingDirectory $root `
    -WindowStyle Hidden
