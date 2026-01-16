#!/bin/bash
# Quick Start Script for Paper Trading

echo "============================================================"
echo "ALGO TRADING LAB - STARTING PAPER TRADING"
echo "============================================================"
echo ""
echo "Mode: Paper Trading with Live Data"
echo "Capital: $10,000"
echo "Symbols: BTC/USDT, ETH/USDT, SOL/USDT, etc."
echo ""
echo "Dashboard: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================================"
echo ""

# Activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start paper trading
python3 run_unified_trading.py run --mode paper_live_data
