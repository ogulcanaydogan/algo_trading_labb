#!/bin/bash
#
# Algo Trading Bot - Deployment Setup Script
#
# Usage: sudo ./setup.sh [install|update|start|stop|status|logs]
#

set -e

# Configuration
INSTALL_DIR="/opt/algo-trading"
SERVICE_USER="trading"
REPO_URL="https://github.com/your-org/algo-trading-lab.git"
PYTHON_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root (sudo)"
        exit 1
    fi
}

install_dependencies() {
    log_info "Installing system dependencies..."

    if command -v apt-get &> /dev/null; then
        apt-get update
        apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
            git curl wget build-essential libssl-dev libffi-dev
    elif command -v yum &> /dev/null; then
        yum install -y python3 python3-venv python3-devel git curl wget \
            gcc openssl-devel libffi-devel
    elif command -v brew &> /dev/null; then
        brew install python@${PYTHON_VERSION} git
    else
        log_error "Unsupported package manager"
        exit 1
    fi
}

create_user() {
    log_info "Creating service user..."

    if ! id "$SERVICE_USER" &>/dev/null; then
        useradd -r -s /bin/false -d "$INSTALL_DIR" "$SERVICE_USER"
        log_info "Created user: $SERVICE_USER"
    else
        log_info "User $SERVICE_USER already exists"
    fi
}

setup_directories() {
    log_info "Setting up directories..."

    mkdir -p "$INSTALL_DIR"/{data,logs,models}
    mkdir -p "$INSTALL_DIR/data"/{unified_trading,models,reports}

    chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
    chmod 750 "$INSTALL_DIR"
}

install_code() {
    log_info "Installing application code..."

    if [ -d "$INSTALL_DIR/.git" ]; then
        log_info "Updating existing installation..."
        cd "$INSTALL_DIR"
        sudo -u "$SERVICE_USER" git pull
    else
        log_info "Cloning repository..."
        # Copy from local if available, otherwise clone
        if [ -d "$(dirname "$0")/.." ]; then
            cp -r "$(dirname "$0")/.." "$INSTALL_DIR/"
            chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
        else
            sudo -u "$SERVICE_USER" git clone "$REPO_URL" "$INSTALL_DIR"
        fi
    fi
}

setup_venv() {
    log_info "Setting up Python virtual environment..."

    cd "$INSTALL_DIR"

    if [ ! -d "venv" ]; then
        sudo -u "$SERVICE_USER" python${PYTHON_VERSION} -m venv venv
    fi

    sudo -u "$SERVICE_USER" ./venv/bin/pip install --upgrade pip wheel
    sudo -u "$SERVICE_USER" ./venv/bin/pip install -r requirements.txt
}

setup_env() {
    log_info "Setting up environment file..."

    ENV_FILE="$INSTALL_DIR/.env"

    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << 'EOF'
# Trading Mode: paper or live
TRADING_MODE=paper

# API Keys (uncomment and fill for live trading)
# BINANCE_API_KEY=your_key_here
# BINANCE_API_SECRET=your_secret_here

# Notification Settings (optional)
# TELEGRAM_BOT_TOKEN=your_token
# TELEGRAM_CHAT_ID=your_chat_id
# DISCORD_WEBHOOK_URL=your_webhook_url

# Logging
LOG_LEVEL=INFO

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
EOF
        chown "$SERVICE_USER:$SERVICE_USER" "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        log_warn "Created .env file - please configure before starting"
    fi
}

install_services() {
    log_info "Installing systemd services..."

    cp "$INSTALL_DIR/deploy/algo-trading.service" /etc/systemd/system/
    cp "$INSTALL_DIR/deploy/algo-api.service" /etc/systemd/system/

    systemctl daemon-reload
    systemctl enable algo-trading algo-api

    log_info "Services installed and enabled"
}

start_services() {
    log_info "Starting services..."
    systemctl start algo-trading algo-api
    sleep 2
    systemctl status algo-trading algo-api --no-pager
}

stop_services() {
    log_info "Stopping services..."
    systemctl stop algo-trading algo-api || true
}

show_status() {
    echo ""
    echo "=== Trading Bot Status ==="
    systemctl status algo-trading --no-pager -l || true
    echo ""
    echo "=== API Server Status ==="
    systemctl status algo-api --no-pager -l || true
}

show_logs() {
    journalctl -u algo-trading -u algo-api -f
}

do_install() {
    check_root
    log_info "Starting full installation..."

    install_dependencies
    create_user
    setup_directories
    install_code
    setup_venv
    setup_env
    install_services

    log_info "Installation complete!"
    log_warn "Please edit $INSTALL_DIR/.env before starting"
    echo ""
    echo "Commands:"
    echo "  Start:  sudo systemctl start algo-trading algo-api"
    echo "  Stop:   sudo systemctl stop algo-trading algo-api"
    echo "  Status: sudo systemctl status algo-trading algo-api"
    echo "  Logs:   sudo journalctl -u algo-trading -u algo-api -f"
}

do_update() {
    check_root
    log_info "Updating installation..."

    stop_services
    install_code
    setup_venv
    start_services

    log_info "Update complete!"
}

# Main
case "${1:-install}" in
    install)
        do_install
        ;;
    update)
        do_update
        ;;
    start)
        check_root
        start_services
        ;;
    stop)
        check_root
        stop_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {install|update|start|stop|status|logs}"
        exit 1
        ;;
esac
