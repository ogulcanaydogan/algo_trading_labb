#!/bin/bash
# Install Trading Bot as systemd service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Trading Bot Service Installer"
echo "=========================================="

# Check if running as root for systemd installation
if [[ $EUID -ne 0 ]]; then
   echo "Note: Run with sudo to install systemd service"
   echo ""
fi

# Create directories
echo "[1/6] Creating directories..."
mkdir -p "$PROJECT_DIR/data/logs"
mkdir -p "$PROJECT_DIR/data/production"

# Setup environment file
echo "[2/6] Setting up environment..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.template" "$PROJECT_DIR/.env"
    echo "Created .env file from template"
    echo "IMPORTANT: Edit $PROJECT_DIR/.env with your credentials"
else
    echo ".env file already exists"
fi

# Install Python dependencies
echo "[3/6] Installing Python dependencies..."
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    pip install -q aiohttp requests pyyaml
else
    echo "Warning: venv not found at $PROJECT_DIR/venv"
    echo "Create it with: python3 -m venv $PROJECT_DIR/venv"
fi

# Test import
echo "[4/6] Testing imports..."
cd "$PROJECT_DIR"
if [ -d "venv" ]; then
    source venv/bin/activate
    python -c "from bot.production_engine import ProductionEngine; print('OK')" 2>/dev/null && echo "Imports OK" || echo "Import failed - check dependencies"
fi

# Install systemd service (requires root)
echo "[5/6] Installing systemd service..."
if [[ $EUID -eq 0 ]]; then
    cp "$SCRIPT_DIR/trading-bot.service" /etc/systemd/system/
    systemctl daemon-reload
    echo "Service installed"
    echo ""
    echo "Commands:"
    echo "  sudo systemctl enable trading-bot    # Auto-start on boot"
    echo "  sudo systemctl start trading-bot     # Start now"
    echo "  sudo systemctl status trading-bot    # Check status"
    echo "  sudo journalctl -u trading-bot -f    # View logs"
else
    echo "Skipped (run with sudo to install)"
fi

echo ""
echo "[6/6] Setup complete!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "=========================================="
echo "1. Edit $PROJECT_DIR/.env with your API keys"
echo "2. Test manually: python scripts/trading/run_production.py"
echo "3. Enable service: sudo systemctl enable trading-bot"
echo "4. Start service: sudo systemctl start trading-bot"
echo ""
