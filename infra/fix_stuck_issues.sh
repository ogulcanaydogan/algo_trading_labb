#!/bin/bash
# Fix hanging/stuck issues in the trading system

echo "============================================================"
echo "TRADING SYSTEM DIAGNOSTICS & FIXES"
echo "============================================================"
echo ""

# 1. Check processes
echo "📊 Checking running processes..."
TRADING_PROCS=$(pgrep -f "run_unified_trading|run_.*_trading" | wc -l)
API_PROCS=$(pgrep -f "uvicorn api.api" | wc -l)
echo "   Trading processes: $TRADING_PROCS"
echo "   API processes: $API_PROCS"
echo ""

# 2. Kill stuck processes
if [ $TRADING_PROCS -gt 3 ]; then
    echo "⚠️  Too many trading processes (possible stuck)!"
    echo "   Killing all trading processes..."
    pkill -f "run_unified_trading"
    pkill -f "run_.*_trading"
    sleep 2
    echo "   ✅ Processes killed"
    echo ""
fi

# 3. Check and clear locks
echo "🔒 Checking database locks..."
find data/ -name "*.db-wal" -o -name "*.db-shm" | while read f; do
    echo "   Removing lock: $f"
    rm -f "$f"
done
echo "   ✅ Locks cleared"
echo ""

# 4. Check port usage
echo "🔌 Checking port usage..."
PORT_8000=$(lsof -i :8000 2>/dev/null | grep LISTEN | wc -l)
if [ $PORT_8000 -gt 1 ]; then
    echo "   ⚠️  Multiple processes on port 8000"
    lsof -i :8000 | grep LISTEN | awk '{print $2}' | tail -n +2 | xargs -r kill -9
    echo "   ✅ Cleared duplicate processes"
fi
echo ""

# 5. Increase file descriptor limits
echo "📈 Adjusting system limits..."
ulimit -n 16384 2>/dev/null
echo "   ✅ File descriptors: $(ulimit -n)"
echo ""

# 6. Clear cache
echo "🧹 Clearing old cache files..."
find data/ -name "*.cache" -mtime +7 -delete
find data/ -name "*.tmp" -mtime +1 -delete
echo "   ✅ Cache cleared"
echo ""

# 7. Check disk space
echo "💾 Checking disk space..."
DISK_USE=$(df . | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USE" -gt 80 ]; then
    echo "   ⚠️  WARNING: Disk usage at ${DISK_USE}%"
    echo "   Remove old logs: rm -rf data/*/logs/*.log"
fi
echo ""

echo "✅ Diagnostics complete!"
echo ""
echo "RECOMMENDATIONS:"
echo "1. If still hanging, restart with: ./start_paper_trading.sh"
echo "2. Monitor logs: tail -f data/unified_trading/logs/*.log"
echo "3. Check API health: curl http://localhost:8000/health"
echo "4. Emergency stop: python3 run_unified_trading.py emergency-stop"
echo ""
