# windows_stop_paper_live.ps1 - Stop paper-live trading engine on Windows
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
$pidPath = Join-Path $repoRoot "logs\paper_live.pid"

# === Helper function to stop process gracefully ===
function Stop-EngineProcess {
    param([int]$ProcessId, [int]$TimeoutSeconds = 10)

    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $proc) {
        return $true
    }

    # Try graceful stop first (SIGTERM equivalent)
    Write-Host "Stopping process $ProcessId gracefully..."
    Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue

    # Wait for graceful shutdown
    $waited = 0
    while ($waited -lt $TimeoutSeconds) {
        Start-Sleep -Seconds 1
        $waited++
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if (-not $proc) {
            Write-Host "Process $ProcessId stopped gracefully."
            return $true
        }
    }

    # Force kill if still running
    Write-Host "Process $ProcessId did not stop gracefully. Force killing..."
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1

    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    return (-not $proc)
}

# === Try to stop by PID file ===
$stoppedByPidFile = $false

if (Test-Path $pidPath) {
    $pidContent = (Get-Content $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($pidContent) {
        $procId = $pidContent.Trim()
        if ($procId -match '^\d+$') {
            Write-Host "Found PID file with PID: $procId"

            # Verify it's actually the engine
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
            if ($proc -and $proc.CommandLine -match "run_unified_trading\.py") {
                $stopped = Stop-EngineProcess -ProcessId ([int]$procId)
                if ($stopped) {
                    $stoppedByPidFile = $true
                    Write-Host "Paper-live process stopped (PID: $procId)"
                } else {
                    Write-Host "WARNING: Failed to stop process $procId"
                }
            } else {
                Write-Host "PID $procId is not running run_unified_trading.py (stale PID file)"
            }
        }
    }
    # Remove PID file
    Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
    Write-Host "PID file removed."
} else {
    Write-Host "PID file not found: $pidPath"
}

# === Cleanup any orphan processes ===
$orphans = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_unified_trading\.py" }
if ($orphans) {
    Write-Host ""
    Write-Host "Found orphan run_unified_trading.py process(es):"
    $orphans | ForEach-Object {
        Write-Host "  PID: $($_.ProcessId) | Parent: $($_.ParentProcessId)"
        $stopped = Stop-EngineProcess -ProcessId $_.ProcessId
        if ($stopped) {
            Write-Host "  -> Stopped"
        } else {
            Write-Host "  -> WARNING: Failed to stop"
        }
    }
}

# === Final verification ===
Start-Sleep -Seconds 1
$remaining = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_unified_trading\.py" }
if ($remaining) {
    Write-Host ""
    Write-Host "WARNING: Some processes may still be running:"
    $remaining | ForEach-Object {
        Write-Host "  PID: $($_.ProcessId) | $($_.CommandLine.Substring(0, [Math]::Min(80, $_.CommandLine.Length)))..."
    }
    exit 1
} else {
    Write-Host ""
    Write-Host "=============================================="
    Write-Host "Paper-live engine stopped successfully"
    Write-Host "=============================================="
    Write-Host ""
    Write-Host "Verification command:"
    Write-Host '  Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "run_unified_trading.py" } | Select ProcessId'
    exit 0
}
