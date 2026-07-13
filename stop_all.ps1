$ErrorActionPreference = "Continue"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

$procs = Get-CimInstance Win32_Process | Where-Object {
    ($_.CommandLine -like "*BoomBeachSonarAuto-main*main.py*" -or
     $_.CommandLine -like "*BoomBeachSonarAuto-main*log_overlay.py*") -and
    $_.ProcessId -ne $PID
}

foreach ($proc in $procs) {
    Stop-Process -Id $proc.ProcessId -Force
}

& $python -c "from utils.adb_control import AdbController; from config import GAME_PACKAGE_NAME; adb=AdbController(); adb.ensure_root_shell(); adb.disable_reject_network(GAME_PACKAGE_NAME); adb.disable_weak_network(GAME_PACKAGE_NAME); print('network restored')"

$procs | Select-Object ProcessId, CommandLine
