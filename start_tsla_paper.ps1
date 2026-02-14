# TSLA Paper Trading Startup Script
# Usage: .\start_tsla_paper.ps1 [start|stop|status]

param(
    [Parameter(Position=0)]
    [ValidateSet("start", "stop", "status")]
    [string]$Action = "start"
)

$ProjectDir = "C:\Users\Ogulcan\Desktop\Projects\algo_trading_lab"
$VenvPython = "$ProjectDir\.venv\Scripts\python.exe"
$Script = "$ProjectDir\run_tsla_paper.py"
$PidFile = "$ProjectDir\data\tsla_paper\tsla_paper.pid"

function Get-RunningPid {
    if (Test-Path $PidFile) {
        $pid = Get-Content $PidFile -Raw
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -eq "python") {
            return [int]$pid
        }
    }
    return $null
}

switch ($Action) {
    "start" {
        $existingPid = Get-RunningPid
        if ($existingPid) {
            Write-Host "TSLA paper trading already running (PID: $existingPid)"
            exit 0
        }

        Write-Host "Starting TSLA Paper Trading..."
        Write-Host "Model: v6_improved (55% walk-forward accuracy)"
        Write-Host "Capital: $10,000"
        Write-Host ""

        # Start in background
        $process = Start-Process -FilePath $VenvPython -ArgumentList $Script -WorkingDirectory $ProjectDir -PassThru -WindowStyle Hidden
        
        # Save PID
        New-Item -ItemType Directory -Path "$ProjectDir\data\tsla_paper" -Force | Out-Null
        $process.Id | Out-File -FilePath $PidFile -NoNewline

        Write-Host "TSLA paper trading started (PID: $($process.Id))"
        Write-Host ""
        Write-Host "Monitor with:"
        Write-Host "  Get-Content $ProjectDir\data\tsla_paper\logs\tsla_paper_$(Get-Date -Format 'yyyyMMdd').log -Wait"
        Write-Host ""
        Write-Host "Check status:"
        Write-Host "  .\start_tsla_paper.ps1 status"
    }

    "stop" {
        $existingPid = Get-RunningPid
        if (-not $existingPid) {
            Write-Host "TSLA paper trading not running"
            exit 0
        }

        Write-Host "Stopping TSLA paper trading (PID: $existingPid)..."
        Stop-Process -Id $existingPid -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped"
    }

    "status" {
        $existingPid = Get-RunningPid
        if ($existingPid) {
            Write-Host "TSLA paper trading is RUNNING (PID: $existingPid)"
        } else {
            Write-Host "TSLA paper trading is STOPPED"
        }
        Write-Host ""
        
        # Show trading status
        & $VenvPython $Script --status
    }
}
