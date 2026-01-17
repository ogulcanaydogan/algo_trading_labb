#!/bin/bash
# Deploy algo_trading_lab to Spark compute server

SPARK_HOST="100.80.116.20"
SPARK_PORT="8000"
SPARK_USER="root"  # Update if different
REMOTE_PATH="/root/algo_trading_lab"

echo "============================================================"
echo "DEPLOYING TO SPARK: $SPARK_HOST:$SPARK_PORT"
echo "============================================================"
echo ""

# Check if SSH is available
if ! command -v ssh &> /dev/null; then
    echo "❌ SSH not found. Please install openssh-client"
    exit 1
fi

echo "📦 Pulling latest code from GitHub..."
git pull origin main

echo "📤 Deploying to Spark server..."

# Check if directory exists on remote
ssh "$SPARK_USER@$SPARK_HOST" "ls -la $REMOTE_PATH > /dev/null 2>&1"
if [ $? -ne 0 ]; then
    echo "📁 Creating remote directory..."
    ssh "$SPARK_USER@$SPARK_HOST" "mkdir -p $REMOTE_PATH"
fi

# Sync code (excluding large directories and git)
echo "📤 Syncing code files..."
rsync -avz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='data/models' \
    --exclude='data/backtest_results' \
    --exclude='.venv' \
    --exclude='node_modules' \
    . "$SPARK_USER@$SPARK_HOST:$REMOTE_PATH/"

echo ""
echo "✅ Code deployed successfully!"
echo ""
echo "📋 NEXT STEPS on Spark server:"
echo ""
echo "1. SSH into Spark:"
echo "   ssh $SPARK_USER@$SPARK_HOST"
echo ""
echo "2. Navigate to project:"
echo "   cd $REMOTE_PATH"
echo ""
echo "3. Setup environment:"
echo "   python3 -m venv .venv"
echo "   source .venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Copy .env file:"
echo "   # Copy your .env file to Spark"
echo "   scp .env $SPARK_USER@$SPARK_HOST:$REMOTE_PATH/.env"
echo ""
echo "5. Start paper trading:"
echo "   ./start_paper_trading.sh"
echo ""
echo "6. In another terminal, start API:"
echo "   python3 -m uvicorn api.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "7. Access dashboard:"
echo "   http://$SPARK_HOST:8000"
echo ""
echo "============================================================"
