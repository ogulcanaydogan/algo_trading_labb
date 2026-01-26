#!/bin/bash
# Start paper trading engine in background with logging

cd /Users/ogulcanaydogan/Desktop/YaPAY/algo_trading_lab

# Kill any existing instances
pkill -f "run_unified_trading.py" 2>/dev/null

# Wait for cleanup
sleep 2

# Set environment
export PYTHONPATH=/Users/ogulcanaydogan/Desktop/YaPAY/algo_trading_lab

# Start in background with nohup
nohup /Users/ogulcanaydogan/Desktop/YaPAY/algo_trading_lab/.venv-1/bin/python \
    run_unified_trading.py run \
    --mode paper_live_data \
    --capital 50000 \
    > logs/trading_stdout.log 2>&1 &

PID=$!
echo "✓ Paper trading engine started (PID: $PID)"
echo "Monitor with: tail -f data/unified_trading/logs/paper_live_data_*.log"
echo "Check status: python run_unified_trading.py status"
