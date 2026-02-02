# windows_run_paper_live.ps1 - Start paper-live trading engine on Windows
# Works from Task Scheduler and manual PowerShell execution

# === Resolve repo root safely (handles Task Scheduler where $PSScriptRoot may be empty) ===
$scriptPath = $MyInvocation.MyCommand.Path
$scriptDir = $null
if ($scriptPath) {
    $scriptDir = Split-Path -Parent $scriptPath
} elseif ($PSScriptRoot) {
    $scriptDir = $PSScriptRoot
}

$repoRoot = $null
if ($scriptDir) {
    $resolved = Resolve-Path (Join-Path $scriptDir "..\..") -ErrorAction SilentlyContinue
    if ($resolved) { $repoRoot = $resolved.Path }
}
if (-not $repoRoot) {
    $repoRoot = "C:\work\algo_trading_lab"
}

Write-Host "Repo root: $repoRoot"
Set-Location -LiteralPath $repoRoot

# === Setup paths ===
$logsDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$stdoutLogPath = Join-Path $logsDir "paper_live_longrun.out.log"
$stderrLogPath = Join-Path $logsDir "paper_live_longrun.err.log"
$pidPath = Join-Path $logsDir "paper_live.pid"
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scriptPy = Join-Path $repoRoot "run_unified_trading.py"
$argsStr = "run --mode paper_live_data --interval 60 --capital 10000"

# === Validate venv python ===
if (-not (Test-Path $pythonExe)) {
    Write-Host "ERROR: Missing venv python: $pythonExe"
    Write-Host "Run: python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

# === Check if already running (by PID file) ===
if (Test-Path $pidPath) {
    $existingPid = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($existingPid -match '^\d+$') {
        $existingProc = Get-CimInstance Win32_Process -Filter "ProcessId=$existingPid" -ErrorAction SilentlyContinue
        if ($existingProc -and $existingProc.CommandLine -match "run_unified_trading\.py") {
            Write-Host "Paper-live already running. PID: $existingPid"
            Write-Host ""
            Write-Host "Log files:"
            Write-Host "  stdout: $stdoutLogPath"
            Write-Host "  stderr: $stderrLogPath"
            Write-Host ""
            Write-Host "Helper commands:"
            Write-Host "  Get-Content -Tail 50 $stdoutLogPath"
            Write-Host "  Get-Content -Tail 50 $stderrLogPath"
            Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\ops\windows_stop_paper_live.ps1"
            exit 0
        }
    }
    # Stale PID file - remove it
    Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
}

# === Double-check no orphan process running ===
$orphans = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_unified_trading\.py" }
if ($orphans) {
    Write-Host "WARNING: Found orphan run_unified_trading.py process(es). Stopping them first..."
    $orphans | ForEach-Object {
        Write-Host "  Stopping PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

# === Start the engine ===
Write-Host "Starting paper-live engine..."

$process = Start-Process -FilePath $pythonExe -ArgumentList "$scriptPy $argsStr" -WorkingDirectory $repoRoot -NoNewWindow -PassThru -RedirectStandardOutput $stdoutLogPath -RedirectStandardError $stderrLogPath

# Wait briefly for process to stabilize
Start-Sleep -Seconds 3

# === Find the actual engine process ===
# The started process may spawn a child (due to venv re-exec). Find the one actually running the engine.
$enginePid = $null

# Check if the direct process is still alive and is the engine
if (-not $process.HasExited) {
    $directProc = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)" -ErrorAction SilentlyContinue
    if ($directProc -and $directProc.CommandLine -match "run_unified_trading\.py") {
        $enginePid = $process.Id
    }
}

# If direct process died or isn't the engine, look for child process
if (-not $enginePid) {
    $childProcs = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $process.Id -and $_.CommandLine -match "run_unified_trading\.py"
    }
    if ($childProcs) {
        $enginePid = ($childProcs | Select-Object -First 1).ProcessId
    }
}

# Last resort: find any process running the script
if (-not $enginePid) {
    $anyEngine = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match "run_unified_trading\.py"
    } | Select-Object -First 1
    if ($anyEngine) {
        $enginePid = $anyEngine.ProcessId
    }
}

# === Validate engine started ===
if (-not $enginePid) {
    Write-Host "ERROR: Engine process not found after start."
    Write-Host "Check error log: Get-Content $stderrLogPath"
    exit 1
}

# Verify the process is still alive
$engineProc = Get-Process -Id $enginePid -ErrorAction SilentlyContinue
if (-not $engineProc) {
    Write-Host "ERROR: Engine process $enginePid died immediately after start."
    Write-Host "Check error log: Get-Content $stderrLogPath"
    exit 1
}

# === Write PID file ===
$enginePid | Out-File -FilePath $pidPath -Encoding ascii -NoNewline

Write-Host ""
Write-Host "=============================================="
Write-Host "Paper-live engine started successfully"
Write-Host "=============================================="
Write-Host "PID: $enginePid"
Write-Host "PID file: $pidPath"
Write-Host ""
Write-Host "Log files:"
Write-Host "  stdout: $stdoutLogPath"
Write-Host "  stderr: $stderrLogPath"
Write-Host ""
Write-Host "Verification commands:"
Write-Host '  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_unified_trading.py" } | Select ProcessId,ParentProcessId,CommandLine | Format-List'
Write-Host '  $pidfile = Get-Content logs\paper_live.pid; $hb = Get-Content data\rl\paper_live_heartbeat.json | ConvertFrom-Json; "pidfile=$pidfile heartbeatPid=$($hb.pid)"'
Write-Host ""
Write-Host "Stop command:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\ops\windows_stop_paper_live.ps1"
