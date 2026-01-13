#!/bin/bash
# Setup script for AI-enhanced trading on NVIDIA Spark
# Run this on the Spark: bash setup_spark_ai.sh

set -e

echo "=========================================="
echo "  AI Trading Setup for NVIDIA Spark"
echo "=========================================="

# 1. Install Ollama
echo ""
echo "[1/5] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed successfully"
else
    echo "Ollama already installed: $(ollama --version)"
fi

# 2. Start Ollama service
echo ""
echo "[2/5] Starting Ollama service..."
sudo systemctl enable ollama 2>/dev/null || true
sudo systemctl start ollama 2>/dev/null || ollama serve &>/dev/null &
sleep 5

# 3. Pull AI models (choose based on your needs)
echo ""
echo "[3/5] Pulling AI models..."
echo "This may take a while depending on your internet speed..."

# Fast model for quick decisions (4GB)
echo "Pulling qwen2.5:7b (fast decisions)..."
ollama pull qwen2.5:7b

# Medium model for analysis (19GB)
echo "Pulling qwen2.5:32b (deep analysis)..."
ollama pull qwen2.5:32b

# Optional: Large reasoning model (43GB) - uncomment if you want the best
# echo "Pulling llama3.1:70b (best reasoning)..."
# ollama pull llama3.1:70b

# Multi-expert model (26GB)
echo "Pulling mixtral:8x7b (multi-expert)..."
ollama pull mixtral:8x7b

# 4. Test Ollama
echo ""
echo "[4/5] Testing Ollama..."
ollama list
echo ""
echo "Testing inference..."
echo '{"role": "user", "content": "Say OK if working"}' | ollama run qwen2.5:7b

# 5. Update trading bot configuration
echo ""
echo "[5/5] Configuring trading bot..."
cd ~/Ogulcan/algo_trading_lab

# Add Ollama host to .env if not present
if ! grep -q "OLLAMA_HOST" .env 2>/dev/null; then
    echo "" >> .env
    echo "# AI Configuration" >> .env
    echo "OLLAMA_HOST=http://localhost:11434" >> .env
    echo "AI_PRIMARY_MODEL=qwen2.5:32b" >> .env
    echo "AI_FAST_MODEL=qwen2.5:7b" >> .env
    echo "AI_ENABLED=true" >> .env
fi

# Restart trading containers
echo ""
echo "Restarting trading bot with AI enabled..."
docker compose down
docker compose up -d

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Available AI models:"
ollama list
echo ""
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "To test AI manually:"
echo "  ollama run qwen2.5:32b 'Analyze BTC price action'"
echo ""
