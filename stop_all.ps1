$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$pidFile = Join-Path $root "_debug\runtime\main.pid"
$statusFile = Join-Path $root "_debug\runtime\status.json"
$stopped = @()

$blockScript = @'
from config import GAME_PACKAGE_NAME
from utils.adb_control import AdbController

adb = AdbController()
adb.ensure_root_shell()
adb.enable_weak_network(GAME_PACKAGE_NAME)
adb.enable_reject_network(GAME_PACKAGE_NAME)
adb.delay(0.2)
'@

& $python -c $blockScript
if ($LASTEXITCODE -ne 0) {
    throw "无法在停止程序前锁定游戏网络，已中止停止操作"
}

if (Test-Path -LiteralPath $pidFile) {
    $mainPidText = (Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    $mainPid = 0
    if ([int]::TryParse($mainPidText, [ref]$mainPid)) {
        $proc = Get-Process -Id $mainPid -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $mainPid -Force
            $stopped += [pscustomobject]@{ ProcessId = $mainPid; CommandLine = "main.py pid file" }
        }
    }
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}

$procs = Get-CimInstance Win32_Process | Where-Object {
    ($_.CommandLine -like "*BoomBeachSonarAuto-main*main.py*" -or
     $_.CommandLine -like "*BoomBeachSonarAuto-main*log_overlay.py*") -and
    $_.ProcessId -ne $PID
}

foreach ($proc in $procs) {
    Stop-Process -Id $proc.ProcessId -Force
    $stopped += $proc
}

$finishScript = @'
from config import GAME_PACKAGE_NAME
from utils.adb_control import AdbController
from utils.pending_probe import clear_pending_probe

adb = AdbController()
adb.ensure_root_shell()
adb.close_app(GAME_PACKAGE_NAME)
if not adb.wait_until_app_stopped(GAME_PACKAGE_NAME, timeout=5.0, poll_interval=0.1):
    raise RuntimeError("游戏进程未完全退出，保留 DROP/REJECT 断网")
adb.delay(0.5)
clear_pending_probe()
adb.disable_weak_network(GAME_PACKAGE_NAME)
adb.disable_reject_network(GAME_PACKAGE_NAME)
print("network restored after game process stopped")
'@

& $python -c $finishScript
if ($LASTEXITCODE -ne 0) {
    throw "游戏尚未完全停止，网络保持断开；请不要手动恢复网络"
}

Remove-Item -LiteralPath $statusFile -Force -ErrorAction SilentlyContinue

$stopped | Select-Object ProcessId, CommandLine
