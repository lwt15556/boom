param(
    [switch]$SkipLaunch
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $root ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $root "requirements.txt"
$requirementsMarker = Join-Path $venvDir ".requirements.sha256"
$adb = Join-Path $root "tools\platform-tools\adb.exe"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit code: $LASTEXITCODE)"
    }
}

function Test-PythonCandidate {
    param(
        [string]$FilePath,
        [string[]]$PrefixArguments = @()
    )

    if ([string]::IsNullOrWhiteSpace($FilePath)) {
        return $null
    }

    try {
        $arguments = @($PrefixArguments) + @(
            "-c",
            "import sys; print('BBMA_OK' if sys.version_info >= (3, 10) else 'BBMA_OLD')"
        )
        $result = & $FilePath @arguments 2>$null
        if ($LASTEXITCODE -eq 0 -and ($result | Select-Object -Last 1) -eq "BBMA_OK") {
            return [pscustomobject]@{
                FilePath = $FilePath
                PrefixArguments = @($PrefixArguments)
            }
        }
    }
    catch {
        return $null
    }

    return $null
}

function Find-CompatiblePython {
    $launcher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        $candidate = Test-PythonCandidate -FilePath $launcher.Source -PrefixArguments @("-3.11")
        if ($null -ne $candidate) {
            return $candidate
        }
        $candidate = Test-PythonCandidate -FilePath $launcher.Source -PrefixArguments @("-3")
        if ($null -ne $candidate) {
            return $candidate
        }
    }

    $knownPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    if (Test-Path -LiteralPath $knownPython) {
        $candidate = Test-PythonCandidate -FilePath $knownPython
        if ($null -ne $candidate) {
            return $candidate
        }
    }

    foreach ($commandName in @("python.exe", "python3.exe")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue
        if ($null -eq $command) {
            continue
        }
        $candidate = Test-PythonCandidate -FilePath $command.Source
        if ($null -ne $candidate) {
            return $candidate
        }
    }

    return $null
}

function Install-Python311 {
    $winget = Get-Command "winget.exe" -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        throw "未找到 Python 3.10+，并且系统没有 winget。请先从 https://www.python.org/downloads/windows/ 安装 Python 3.11，然后重新双击一键启动文件。"
    }

    Write-Step "未找到可用的 Python，正在通过 winget 安装 Python 3.11"
    Invoke-Checked `
        -FilePath $winget.Source `
        -Arguments @(
            "install",
            "--id", "Python.Python.3.11",
            "--exact",
            "--scope", "user",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--silent"
        ) `
        -FailureMessage "Python 3.11 自动安装失败"
}

Write-Host "BoomBeachSonarAuto 一键环境配置" -ForegroundColor Green
Write-Host "项目目录：$root"

if (-not (Test-Path -LiteralPath $requirements)) {
    throw "缺少 requirements.txt，请重新下载完整项目。"
}
if (-not (Test-Path -LiteralPath $adb)) {
    throw "缺少内置 ADB：$adb。请重新下载完整项目，不要单独下载源码文件。"
}

$venvCandidate = $null
if (Test-Path -LiteralPath $venvPython) {
    $venvCandidate = Test-PythonCandidate -FilePath $venvPython
}

if ($null -eq $venvCandidate) {
    if (Test-Path -LiteralPath $venvDir) {
        $backupName = ".venv.invalid.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Write-Host "检测到不完整的 .venv，已改名为 $backupName 并准备重建。" -ForegroundColor Yellow
        Rename-Item -LiteralPath $venvDir -NewName $backupName
    }

    Write-Step "检查 Python 3.10 或更高版本"
    $python = Find-CompatiblePython
    if ($null -eq $python) {
        Install-Python311
        $python = Find-CompatiblePython
    }
    if ($null -eq $python) {
        throw "Python 已安装，但当前窗口仍无法找到它。请关闭此窗口后再次双击一键启动文件。"
    }

    Write-Step "创建项目专用 Python 环境"
    $venvArguments = @($python.PrefixArguments) + @("-m", "venv", $venvDir)
    Invoke-Checked `
        -FilePath $python.FilePath `
        -Arguments $venvArguments `
        -FailureMessage "创建 .venv 失败"
}

Write-Step "检查并安装运行依赖"
$requirementsHash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
$installedHash = ""
if (Test-Path -LiteralPath $requirementsMarker) {
    $installedHash = (Get-Content -LiteralPath $requirementsMarker -Raw).Trim()
}

if ($installedHash -ne $requirementsHash) {
    Invoke-Checked `
        -FilePath $venvPython `
        -Arguments @("-m", "pip", "install", "--upgrade", "pip") `
        -FailureMessage "pip 更新失败，请检查网络"
    Invoke-Checked `
        -FilePath $venvPython `
        -Arguments @("-m", "pip", "install", "-r", $requirements) `
        -FailureMessage "项目依赖安装失败，请检查网络"
    Set-Content -LiteralPath $requirementsMarker -Value $requirementsHash -Encoding ASCII
}
else {
    Write-Host "依赖没有变化，跳过重复安装。" -ForegroundColor DarkGray
}

Write-Step "验证 Python、OpenCV、PyQt6、NumPy 和内置 ADB"
Invoke-Checked `
    -FilePath $venvPython `
    -Arguments @(
        "-c",
        "import cv2, numpy, PyQt6; print('Python 环境验证通过')"
    ) `
    -FailureMessage "Python 依赖验证失败"
Invoke-Checked `
    -FilePath $adb `
    -Arguments @("version") `
    -FailureMessage "内置 ADB 无法运行"

Write-Host ""
Write-Host "环境配置完成。" -ForegroundColor Green
Write-Host "使用前请在模拟器中确认：已开启 Root 和 ADB、本地地址为 127.0.0.1:5555、分辨率为 1280x720。" -ForegroundColor Yellow

if (-not $SkipLaunch) {
    Write-Step "启动控制台"
    & (Join-Path $root "run_control_panel.ps1")
    Write-Host "控制台已启动，可以关闭此窗口。" -ForegroundColor Green
}
