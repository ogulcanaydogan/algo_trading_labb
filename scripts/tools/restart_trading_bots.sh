#!/bin/bash
# Restart all trading bots cleanly

set -e

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "🛑 Stopping all trading bots..."

# Kill all running trading processes
pkill -f "run_multi_market.py" 2>/dev/null || true
pkill -f "run_regime_trading.py" 2>/dev/null || true
pkill -f "run_live_paper_trading.py" 2>/dev/null || true
pkill -f "run_commodity_trading.py" 2>/dev/null || true
pkill -f "run_stock_trading.py" 2>/dev/null || true

sleep 2

echo "✓ All bots stopped"
echo ""

# Determine Python path (use venv if available, otherwise system python)
if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
    echo "Using virtual environment Python"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
    echo "Using activated virtual environment"
else
    PYTHON_CMD="python3"
    echo "⚠️  No virtual environment found, using system python3"
    echo "   Install dependencies: pip3 install -r requirements.txt"
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "🚀 Starting trading bots..."
echo ""

# Start crypto (live paper trading)
echo "Starting Crypto bot..."
nohup $PYTHON_CMD "$PROJECT_ROOT/scripts/trading/run_live_paper_trading.py" > /dev/null 2>&1 &
CRYPTO_PID=$!
sleep 2

# Start commodity bot
echo "Starting Commodity bot..."
nohup $PYTHON_CMD "$PROJECT_ROOT/scripts/trading/run_commodity_trading.py" > /dev/null 2>&1 &
COMMODITY_PID=$!
sleep 2

# Start stock bot
echo "Starting Stock bot..."
nohup $PYTHON_CMD "$PROJECT_ROOT/scripts/trading/run_stock_trading.py" > /dev/null 2>&1 &
STOCK_PID=$!
sleep 2

echo ""
echo "✓ All bots started"
echo ""
echo "PIDs:"
echo "  Crypto:    $CRYPTO_PID"
echo "  Commodity: $COMMODITY_PID"
echo "  Stock:     $STOCK_PID"
echo ""
echo "Check status with:"
echo "  ps aux | grep 'python.*trading' | grep -v grep"
echo ""
echo "View logs:"
echo "  tail -f data/logs/commodity_trading.log"
echo "  tail -f data/logs/stock_trading.log"
