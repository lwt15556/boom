$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$overlay = Join-Path $root "tools\log_overlay.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$stopScript = @'
from tools.control_panel import stop_program

print(stop_program())
'@

& $python -c $stopScript
if ($LASTEXITCODE -ne 0) {
    throw "安全停止主程序失败；网络状态未确认，请查看上方错误"
}

$stoppedOverlays = Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -eq $python -and
    $_.CommandLine -like "*$overlay*" -and
    $_.ProcessId -ne $PID
}

foreach ($process in $stoppedOverlays) {
    Stop-Process -Id $process.ProcessId -Force
}

$stoppedOverlays | Select-Object ProcessId, CommandLine
