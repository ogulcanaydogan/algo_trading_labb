# Development & Deployment Workflow

## Environment Overview

| Environment | Location | Purpose |
|-------------|----------|---------|
| **Local (Dev)** | Your Mac | Development, testing, debugging |
| **Spark (Prod)** | NVIDIA Spark (100.80.116.20) | 24/7 production trading |

## Quick Commands

### Development (Local)
```bash
# Run locally for testing
docker compose up -d

# View logs
docker compose logs -f orchestrator

# Stop
docker compose down

# Run tests
pytest tests/
```

### Production (Spark)
```bash
# Deploy changes to Spark
./deploy.sh

# Deploy and restart containers
./deploy.sh --restart

# SSH to Spark
ssh spark

# View production logs
ssh spark "cd ~/Ogulcan/algo_trading_lab && docker compose logs -f orchestrator"
```

## Workflow

### 1. Make Changes Locally
```bash
# Edit code locally using your preferred editor
# Test changes with local Docker
docker compose up -d
docker compose logs -f

# Run tests
pytest tests/
```

### 2. Deploy to Production
```bash
# Option A: Just sync files (bot will pick up changes on next iteration)
./deploy.sh

# Option B: Sync and restart (for immediate effect)
./deploy.sh --restart
```

### 3. Monitor Production
```bash
# Check dashboard
open http://100.80.116.20:8000

# Check logs
ssh spark "cd ~/Ogulcan/algo_trading_lab && docker compose logs --tail=50 orchestrator"

# Check container status
ssh spark "cd ~/Ogulcan/algo_trading_lab && docker compose ps"
```

## Configuration

### config.yaml
Main configuration file. Edit locally, then deploy:
- `trading.loop_interval`: Time between trading iterations (default: 180s)
- `trading.symbol_fetch_delay`: Delay between API calls (default: 5s)
- `crypto.symbols`: List of crypto pairs to trade
- `commodities.symbols`: List of commodity symbols
- `stocks.symbols`: List of stock symbols

### .env (Secrets)
Keep secrets in `.env` file (not synced to Spark for security):
- `TELEGRAM_BOT_TOKEN`: Telegram notifications
- `TELEGRAM_CHAT_ID`: Telegram chat ID
- `ANTHROPIC_API_KEY`: For AI features

**Note:** You need to manually copy `.env` to Spark if you update secrets:
```bash
scp .env spark:~/Ogulcan/algo_trading_lab/.env
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NVIDIA Spark (Production)                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Orchestrator   │  │      API        │  │  Dashboard  │ │
│  │ (Multi-Market)  │──│   (FastAPI)     │──│   (HTML)    │ │
│  └────────┬────────┘  └─────────────────┘  └─────────────┘ │
│           │                                                 │
│  ┌────────┴────────────────────────────────────────┐       │
│  │                 Trading Bots                     │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │       │
│  │  │  Crypto  │  │Commodity │  │  Stocks  │      │       │
│  │  │   Bot    │  │   Bot    │  │   Bot    │      │       │
│  │  └──────────┘  └──────────┘  └──────────┘      │       │
│  └──────────────────────────────────────────────────┘       │
│                                                             │
│  Dashboard: http://100.80.116.20:8000                      │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Rate Limiting Errors
If you see "Too Many Requests" errors:
1. Increase `loop_interval` in config.yaml
2. Increase `symbol_fetch_delay` in config.yaml
3. Deploy and restart

### Charts Not Working
1. Check API responses: `curl http://100.80.116.20:8000/api/bot/state/all`
2. Ensure orchestrator is running: `docker compose ps`
3. Check orchestrator logs for errors

### Bot Not Trading
1. Check if market is open (stocks/commodities)
2. Check confidence thresholds in config
3. Check logs for signal generation

## File Locations

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration |
| `docker-compose.yml` | Docker services |
| `deploy.sh` | Deployment script |
| `data/live_paper_trading/state.json` | Crypto bot state |
| `data/commodity_trading/state.json` | Commodity bot state |
| `data/stock_trading/state.json` | Stock bot state |
