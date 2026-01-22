#!/bin/bash
# Auto-retraining cron script
# Add to crontab with: crontab -e
# Then add: 0 2 * * 0 /path/to/algo_trading_lab/scripts/ml/retrain_cron.sh
# Or run directly: ./scripts/ml/retrain_cron.sh

set -e

# Navigate to project directory
cd /Users/ogulcanaydogan/Desktop/YaPAY/algo_trading_lab

# Activate virtual environment
source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create log directory
mkdir -p data/logs

LOG_FILE="data/logs/retrain_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Starting model retraining at $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Extended symbol list
SYMBOLS="BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,DOGE/USDT,XRP/USDT"

# 1. Retrain traditional ML models
echo "Step 1: Training traditional ML models..." | tee -a "$LOG_FILE"
python scripts/ml/auto_retrain.py \
    --symbols "$SYMBOLS" \
    --days 90 \
    >> "$LOG_FILE" 2>&1 || echo "ML training had some errors" | tee -a "$LOG_FILE"

# 2. Train regularized deep learning models
echo "Step 2: Training regularized DL models..." | tee -a "$LOG_FILE"
python scripts/ml/train_dl_regularized.py >> "$LOG_FILE" 2>&1 || echo "DL training had some errors" | tee -a "$LOG_FILE"

# 3. Validate models
echo "Step 3: Validating model performance..." | tee -a "$LOG_FILE"
python -c "
from bot.ml.ensemble_predictor import create_ensemble_predictor
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
for symbol in symbols:
    p = create_ensemble_predictor(symbol)
    if p:
        print(f'{symbol}: {len(p.models)} ML + {len(p.dl_models)} DL models loaded')
    else:
        print(f'{symbol}: Failed to load models')
" >> "$LOG_FILE" 2>&1

echo "========================================" | tee -a "$LOG_FILE"
echo "Retraining completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
