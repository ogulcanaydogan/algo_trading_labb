#!/bin/bash
# Setup automatic model retraining cron jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_PATH="$PROJECT_DIR/.venv/bin/python"

# Daily crypto retraining at 4 AM UTC
CRYPTO_CRON="0 4 * * * cd $PROJECT_DIR && $PYTHON_PATH scripts/ml/auto_retrain_cron.py --asset-types crypto >> data/logs/cron_crypto.log 2>&1"

# Daily forex/indices retraining at 5 AM UTC (after market close)
FOREX_CRON="0 5 * * * cd $PROJECT_DIR && $PYTHON_PATH scripts/ml/auto_retrain_cron.py --asset-types forex indices >> data/logs/cron_forex.log 2>&1"

# Weekly full retraining on Sundays at 6 AM UTC
FULL_CRON="0 6 * * 0 cd $PROJECT_DIR && $PYTHON_PATH scripts/ml/auto_retrain_cron.py --force >> data/logs/cron_full.log 2>&1"

echo "Adding cron jobs for automatic model retraining..."
echo ""

# Show current crontab
echo "Current crontab:"
crontab -l 2>/dev/null || echo "(empty)"
echo ""

# Add new jobs
(crontab -l 2>/dev/null | grep -v "auto_retrain_cron"; echo "$CRYPTO_CRON"; echo "$FOREX_CRON"; echo "$FULL_CRON") | crontab -

echo "New crontab:"
crontab -l
echo ""

echo "✓ Cron jobs installed:"
echo "  - Daily crypto retraining at 4 AM UTC"
echo "  - Daily forex/indices retraining at 5 AM UTC"
echo "  - Weekly full retraining on Sundays at 6 AM UTC"
echo ""
echo "Logs will be written to:"
echo "  - data/logs/cron_crypto.log"
echo "  - data/logs/cron_forex.log"
echo "  - data/logs/cron_full.log"
