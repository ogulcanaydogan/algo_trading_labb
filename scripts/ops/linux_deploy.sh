#!/usr/bin/env bash
# linux_deploy.sh - Deploy algo_trading_lab to a Linux/DGX host via SSH
#
# Usage:
#   ./scripts/ops/linux_deploy.sh [ssh-host] [repo-path]
#
# Examples:
#   ./scripts/ops/linux_deploy.sh spark                    # Deploy to spark host at ~/work/algo_trading_lab
#   ./scripts/ops/linux_deploy.sh spark /opt/algo_trading_lab  # Custom path
#   SSH_HOST=spark REPO_PATH=~/work/algo ./scripts/ops/linux_deploy.sh

set -euo pipefail

# === Configuration ===
SSH_HOST="${1:-${SSH_HOST:-spark}}"
REPO_PATH="${2:-${REPO_PATH:-\$HOME/work/algo_trading_lab}}"
LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=============================================="
echo "Deploying algo_trading_lab to ${SSH_HOST}"
echo "=============================================="
echo "Local repo: ${LOCAL_REPO_ROOT}"
echo "Remote path: ${REPO_PATH}"
echo ""

# === Test SSH connection ===
echo "Testing SSH connection to ${SSH_HOST}..."
if ! ssh -o ConnectTimeout=10 "${SSH_HOST}" "echo 'SSH connection successful'"; then
    echo "ERROR: Cannot connect to ${SSH_HOST}"
    echo "Check your ~/.ssh/config and ensure the host is reachable."
    exit 1
fi

# === Create remote directory and sync repo ===
echo ""
echo "Syncing repository to ${SSH_HOST}:${REPO_PATH}..."

ssh "${SSH_HOST}" "mkdir -p ${REPO_PATH}"

# Use rsync to sync the repo (excluding .venv, __pycache__, etc.)
rsync -avz --progress \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='data/models/*.pkl' \
    --exclude='logs/*.log' \
    "${LOCAL_REPO_ROOT}/" "${SSH_HOST}:${REPO_PATH}/"

echo "Repository synced."

# === Setup venv and install dependencies ===
echo ""
echo "Setting up Python virtual environment..."

ssh "${SSH_HOST}" bash -s "${REPO_PATH}" << 'REMOTE_SETUP'
REPO_PATH="$1"
cd "${REPO_PATH}"

# Check Python version
PYTHON_BIN=""
for py in python3.11 python3.10 python3.9 python3; do
    if command -v "$py" &> /dev/null; then
        PYTHON_BIN="$py"
        break
    fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: No suitable Python found (need 3.9+)"
    exit 1
fi

echo "Using Python: ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

# Create venv if not exists
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    ${PYTHON_BIN} -m venv .venv
fi

# Upgrade pip and install requirements
echo "Installing dependencies..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
.venv/bin/python -c "import ccxt, fastapi, numpy, pandas, sklearn; print('All dependencies installed successfully')"

echo "Virtual environment setup complete."
REMOTE_SETUP

# === Setup systemd service ===
echo ""
echo "Setting up systemd user service..."

# Expand REPO_PATH on remote and update service file
ssh "${SSH_HOST}" bash -s "${REPO_PATH}" << 'REMOTE_SYSTEMD'
REPO_PATH="$1"
EXPANDED_PATH="$(eval echo ${REPO_PATH})"

# Create user systemd directory
mkdir -p ~/.config/systemd/user

# Copy and customize service file
SERVICE_FILE=~/.config/systemd/user/algo_trading_paper_live.service

cat > "${SERVICE_FILE}" << EOF
[Unit]
Description=Algo Trading Paper-Live Engine
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${EXPANDED_PATH}
ExecStart=${EXPANDED_PATH}/.venv/bin/python run_unified_trading.py run --mode paper_live_data --interval 60 --capital 10000

# Paper-live turnover overrides
Environment="PAPER_LIVE_TURNOVER_MIN_RATIO=2.0"
Environment="PAPER_LIVE_TURNOVER_MAX_DAILY=10"

# Load .env file if present
EnvironmentFile=-${EXPANDED_PATH}/.env

# Restart policy
Restart=always
RestartSec=30
StartLimitIntervalSec=600
StartLimitBurst=5

# Graceful shutdown
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM

# Logging
StandardOutput=append:${EXPANDED_PATH}/logs/paper_live_longrun.out.log
StandardError=append:${EXPANDED_PATH}/logs/paper_live_longrun.err.log

[Install]
WantedBy=default.target
EOF

echo "Service file created: ${SERVICE_FILE}"

# Create logs directory
mkdir -p "${EXPANDED_PATH}/logs"

# Enable linger for user services to run after logout
loginctl enable-linger $(whoami) 2>/dev/null || echo "Note: loginctl enable-linger may require sudo"

# Reload systemd
systemctl --user daemon-reload

# Enable service (auto-start on boot)
systemctl --user enable algo_trading_paper_live.service

echo "Systemd service configured and enabled."
REMOTE_SYSTEMD

echo ""
echo "=============================================="
echo "Deployment complete!"
echo "=============================================="
echo ""
echo "Next steps on ${SSH_HOST}:"
echo ""
echo "1. Start the service:"
echo "   systemctl --user start algo_trading_paper_live"
echo ""
echo "2. Check status:"
echo "   systemctl --user status algo_trading_paper_live"
echo ""
echo "3. View logs:"
echo "   journalctl --user -u algo_trading_paper_live -f"
echo "   tail -f ${REPO_PATH}/logs/paper_live_longrun.out.log"
echo ""
echo "4. Verify PID parity:"
echo "   cat ${REPO_PATH}/logs/paper_live.pid"
echo "   cat ${REPO_PATH}/data/rl/paper_live_heartbeat.json | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"pid\"])'"
echo ""
echo "5. Restart service:"
echo "   systemctl --user restart algo_trading_paper_live"
echo ""
echo "6. Stop service:"
echo "   systemctl --user stop algo_trading_paper_live"
