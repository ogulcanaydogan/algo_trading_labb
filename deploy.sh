#!/bin/bash
# Deploy local changes to Spark
# Usage: ./deploy.sh [--restart]

set -e

REMOTE="spark"
REMOTE_PATH="~/Ogulcan/algo_trading_lab"

echo "Syncing to Spark..."

# Sync files (excluding unnecessary stuff)
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='.DS_Store' \
  --exclude='data/cache/*' \
  --exclude='data/models/*' \
  --exclude='data/logs/*' \
  --exclude='.env' \
  ./ ${REMOTE}:${REMOTE_PATH}/

echo ""
echo "Files synced!"

# Restart containers if --restart flag is passed
if [[ "$1" == "--restart" ]]; then
  echo ""
  echo "Restarting containers..."
  ssh ${REMOTE} "cd ${REMOTE_PATH} && docker compose down && docker compose up -d"
  echo "Containers restarted!"
fi

echo ""
echo "Dashboard: http://100.80.116.20:8000"
echo ""
echo "To restart containers manually:"
echo "  ssh spark 'cd ~/Ogulcan/algo_trading_lab && docker compose restart'"
