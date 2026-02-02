# Linux/DGX Deployment Runbook

## Overview

This runbook covers deploying and operating the algo_trading_lab paper-live engine on a Linux/DGX host using systemd for process management.

## Prerequisites

- SSH access to the target host (configured in `~/.ssh/config`)
- Python 3.9+ on the target host
- systemd with user service support

## Quick Deployment

```bash
# From your local machine
./scripts/ops/linux_deploy.sh spark
```

This will:
1. Sync the repo to `~/work/algo_trading_lab` on the spark host
2. Create and configure a Python virtual environment
3. Install all dependencies
4. Set up a systemd user service with auto-restart and boot persistence

## Manual Setup (Alternative)

### 1. Clone/Copy Repository

```bash
# SSH to target host
ssh spark

# Create directory
mkdir -p ~/work
cd ~/work

# Clone repo (if git available)
git clone https://github.com/weezboo/algo_trading_lab.git

# Or copy via rsync from local machine
# rsync -avz /path/to/algo_trading_lab/ spark:~/work/algo_trading_lab/
```

### 2. Setup Virtual Environment

```bash
cd ~/work/algo_trading_lab
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `.env` file for API keys:
```bash
cat > .env << 'EOF'
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
OANDA_ACCOUNT_ID=your_account
OANDA_ACCESS_TOKEN=your_token
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
EOF
chmod 600 .env
```

### 4. Install Systemd Service

```bash
# Copy service file
mkdir -p ~/.config/systemd/user
cp scripts/ops/algo_trading_paper_live.service ~/.config/systemd/user/

# Edit paths in the service file
nano ~/.config/systemd/user/algo_trading_paper_live.service

# Reload systemd
systemctl --user daemon-reload

# Enable service (auto-start on boot)
systemctl --user enable algo_trading_paper_live.service

# Enable linger (run services after logout)
loginctl enable-linger $(whoami)
```

## Operations Commands

### Start/Stop/Restart

```bash
# Start
systemctl --user start algo_trading_paper_live

# Stop
systemctl --user stop algo_trading_paper_live

# Restart
systemctl --user restart algo_trading_paper_live

# Status
systemctl --user status algo_trading_paper_live
```

### View Logs

```bash
# Systemd journal
journalctl --user -u algo_trading_paper_live -f

# Application logs
tail -f ~/work/algo_trading_lab/logs/paper_live_longrun.out.log
tail -f ~/work/algo_trading_lab/logs/paper_live_longrun.err.log
```

### Process Verification

```bash
# Check single process
pgrep -af 'run_unified_trading.py'

# Should show only ONE process like:
# 12345 /home/weezboo/work/algo_trading_lab/.venv/bin/python run_unified_trading.py run --mode paper_live_data ...
```

### PID Parity Verification

```bash
# Get PID from pidfile
cat ~/work/algo_trading_lab/logs/paper_live.pid

# Get PID from heartbeat
cat ~/work/algo_trading_lab/data/rl/paper_live_heartbeat.json | python3 -c 'import sys,json; hb=json.load(sys.stdin); print(f"pid={hb[\"pid\"]} timestamp={hb[\"timestamp\"]}")'

# Compare - should match!
```

### Heartbeat Monitoring

```bash
# Watch heartbeat updates
watch -n5 'cat ~/work/algo_trading_lab/data/rl/paper_live_heartbeat.json | python3 -m json.tool'
```

## Verification Checklist

After deployment, verify:

| Gate | Command | Expected |
|------|---------|----------|
| Service running | `systemctl --user status algo_trading_paper_live` | Active (running) |
| Single process | `pgrep -af 'run_unified_trading.py' \| wc -l` | 1 |
| PID file exists | `cat logs/paper_live.pid` | PID number |
| Heartbeat exists | `cat data/rl/paper_live_heartbeat.json` | JSON with pid |
| PID parity | Compare pidfile and heartbeat | PIDs match |
| Turnover overrides | `grep 'turnover overrides' logs/paper_live_longrun.out.log` | min_ratio, max_daily logged |
| Auto-restart | `systemctl --user restart algo_trading_paper_live` | Service restarts, PID changes, parity holds |

## Reboot Persistence Test

```bash
# Verify service is enabled
systemctl --user is-enabled algo_trading_paper_live
# Should output: enabled

# Verify linger is enabled
loginctl show-user $(whoami) | grep Linger
# Should output: Linger=yes

# After system reboot, service should auto-start
# Check with: systemctl --user status algo_trading_paper_live
```

## Paper-Live Turnover Overrides

Environment variables for paper-live mode only:

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPER_LIVE_TURNOVER_MIN_RATIO` | 2.0 | Minimum EV/cost ratio |
| `PAPER_LIVE_TURNOVER_MAX_DAILY` | 10 | Max decisions per symbol per day |

To change overrides, edit the service file:
```bash
nano ~/.config/systemd/user/algo_trading_paper_live.service
# Change Environment lines
systemctl --user daemon-reload
systemctl --user restart algo_trading_paper_live
```

## Troubleshooting

### Service won't start
```bash
# Check journal for errors
journalctl --user -u algo_trading_paper_live --no-pager -n 50

# Check Python can import modules
cd ~/work/algo_trading_lab
.venv/bin/python -c "import ccxt, fastapi; print('OK')"
```

### Duplicate processes
```bash
# Kill all and restart clean
pkill -f 'run_unified_trading.py'
rm -f logs/paper_live.pid data/rl/paper_live_heartbeat.json
systemctl --user start algo_trading_paper_live
```

### PID parity mismatch
This should not happen with the updated code. If it does:
```bash
# The Python code now writes both pidfile and heartbeat with os.getpid()
# Check logs for startup messages
grep 'PID file written' logs/paper_live_longrun.out.log
```

### Service doesn't survive logout
```bash
# Ensure linger is enabled
loginctl enable-linger $(whoami)
# May require sudo on some systems
```

## Manual Start (Without Systemd)

If systemd user services aren't available:

```bash
cd ~/work/algo_trading_lab

# Set turnover overrides
export PAPER_LIVE_TURNOVER_MIN_RATIO=2.0
export PAPER_LIVE_TURNOVER_MAX_DAILY=10

# Start with nohup
nohup .venv/bin/python run_unified_trading.py run --mode paper_live_data --interval 60 --capital 10000 \
    >> logs/paper_live_longrun.out.log 2>> logs/paper_live_longrun.err.log &

# Or use the helper script
./scripts/ops/linux_run_paper_live.sh
```

## File Locations

| File | Purpose |
|------|---------|
| `logs/paper_live.pid` | Process ID file |
| `data/rl/paper_live_heartbeat.json` | Heartbeat with PID and stats |
| `logs/paper_live_longrun.out.log` | Application stdout |
| `logs/paper_live_longrun.err.log` | Application stderr |
| `data/unified_trading/state.json` | Trading state |
| `.env` | API credentials |
