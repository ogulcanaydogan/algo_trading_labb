# Docker Setup Guide

This document provides instructions for running Algo Trading Lab using Docker containers.

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose v2.0+
- At least 4GB RAM available for containers

### Basic Usage

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

2. **Start all services (development):**
   ```bash
   docker compose up -d
   ```

3. **Access the dashboard:**
   ```
   http://localhost:8000/dashboard
   ```

## Services Overview

| Service | Description | Port | Container Name |
|---------|-------------|------|----------------|
| `api` | FastAPI dashboard and REST API | 8000 | algo-trading-api |
| `crypto-bot` | Cryptocurrency trading (BTC, ETH, SOL, AVAX) | - | algo-trading-crypto |
| `commodity-bot` | Commodity trading (Gold, Silver, Oil) | - | algo-trading-commodity |
| `stock-bot` | Stock trading (AAPL, MSFT, GOOGL, etc.) | - | algo-trading-stock |
| `multi-market-bot` | All markets in single process (optional) | - | algo-trading-multi-market |

## Docker Commands

### Development

```bash
# Start all services
docker compose up -d

# Start specific services
docker compose up -d api crypto-bot

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f crypto-bot

# Stop all services
docker compose down

# Rebuild containers
docker compose build --no-cache

# Restart a service
docker compose restart crypto-bot
```

### Production

```bash
# Start with production configuration
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services (if needed)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale crypto-bot=2
```

### Multi-Market Mode

The multi-market bot runs all trading bots in a single process:

```bash
# Start with multi-market profile
docker compose --profile multi-market up -d

# This will start: api + multi-market-bot (instead of separate bots)
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Trading Configuration
LOOP_INTERVAL=60                    # Seconds between trading iterations
INITIAL_CAPITAL=10000               # Starting capital for crypto bot
COMMODITY_INITIAL_CAPITAL=10000     # Starting capital for commodity bot
STOCK_INITIAL_CAPITAL=10000         # Starting capital for stock bot
TOTAL_INITIAL_CAPITAL=30000         # For multi-market mode

# Deep Learning
USE_DEEP_LEARNING=true
DL_MODEL_SELECTION=regime_based

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Exchange API Keys (for live trading)
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=
EXCHANGE_SANDBOX=true

# API Security
API_KEY=your_api_key_here

# AI Analysis (optional)
ANTHROPIC_API_KEY=your_anthropic_key
CLAUDE_DAILY_BUDGET=5.0
```

### Resource Limits

Default resource allocations in `docker-compose.yml`:

| Service | CPU Limit | Memory Limit |
|---------|-----------|--------------|
| api | 0.5 | 512MB |
| crypto-bot | 1.0 | 1GB |
| commodity-bot | 1.0 | 1GB |
| stock-bot | 1.0 | 1GB |
| multi-market-bot | 2.0 | 2GB |

Adjust these in `docker-compose.yml` based on your system resources.

## Data Persistence

Trading data is stored in a Docker volume:

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect algo_trading_data

# Backup data
docker run --rm -v algo_trading_data:/data -v $(pwd)/backup:/backup \
    alpine tar cvf /backup/trading_data.tar /data

# Restore data
docker run --rm -v algo_trading_data:/data -v $(pwd)/backup:/backup \
    alpine tar xvf /backup/trading_data.tar -C /
```

## Production Deployment

### SSL/TLS Setup

1. **Generate self-signed certificates (development):**
   ```bash
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
       -keyout nginx/ssl/privkey.pem \
       -out nginx/ssl/fullchain.pem \
       -subj "/CN=localhost"
   ```

2. **For Let's Encrypt (production):**
   ```bash
   # Install certbot
   # Run: certbot certonly --standalone -d yourdomain.com
   # Copy certificates to nginx/ssl/
   ```

3. **Start with production config:**
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

### Nginx Configuration

The production setup includes:
- Reverse proxy to API service
- SSL/TLS termination
- Rate limiting (10 requests/second per IP)
- Gzip compression
- Security headers
- Log rotation

Edit `nginx/conf.d/default.conf` to customize:
- Domain name
- SSL certificate paths
- Rate limiting rules
- Additional security headers

## Health Checks

### API Health Endpoint

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00Z",
  "uptime_seconds": 3600.5,
  "bot_last_update": "2024-01-15T11:59:30Z",
  "bot_stale": false,
  "components": {
    "bot": "healthy",
    "state_store": "healthy"
  }
}
```

### Docker Health Status

```bash
docker compose ps

# Example output:
# NAME                    STATUS                   PORTS
# algo-trading-api        Up 2 hours (healthy)     0.0.0.0:8000->8000/tcp
# algo-trading-crypto     Up 2 hours
# algo-trading-commodity  Up 2 hours
# algo-trading-stock      Up 2 hours
```

## Troubleshooting

### Common Issues

1. **Container won't start:**
   ```bash
   # Check logs
   docker compose logs api

   # Check container status
   docker compose ps -a
   ```

2. **Health check failing:**
   ```bash
   # Check if API is responding
   docker compose exec api curl -f http://localhost:8000/health

   # Check dependencies
   docker compose exec crypto-bot python -c "import ccxt; print('OK')"
   ```

3. **Data not persisting:**
   ```bash
   # Verify volume mount
   docker compose exec api ls -la /app/data

   # Check volume
   docker volume inspect algo_trading_data
   ```

4. **Out of memory:**
   ```bash
   # Check container stats
   docker stats

   # Increase memory limits in docker-compose.yml
   ```

### Logs

```bash
# All logs
docker compose logs

# Follow logs
docker compose logs -f

# Last 100 lines
docker compose logs --tail=100

# Specific service
docker compose logs -f crypto-bot

# Log files inside container
docker compose exec api cat /app/data/logs/paper_trading.log
```

### Container Shell Access

```bash
# API container
docker compose exec api /bin/bash

# Bot container
docker compose exec crypto-bot /bin/bash

# As root (for debugging)
docker compose exec -u root api /bin/bash
```

## Monitoring

### Prometheus Metrics (Optional)

Add to `docker-compose.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

### Grafana Dashboard (Optional)

```yaml
  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Updating

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker compose build --no-cache
docker compose up -d

# Or for zero-downtime updates
docker compose up -d --build --force-recreate --no-deps api
docker compose up -d --build --force-recreate --no-deps crypto-bot
```

## Security Best Practices

1. **Never commit `.env` file** - Use `.env.example` as template
2. **Use secrets management** - Consider Docker secrets for sensitive data
3. **Restrict network access** - Use firewall rules in production
4. **Regular updates** - Keep base images and dependencies updated
5. **Non-root user** - Containers run as `appuser` (UID 1000)
6. **Read-only filesystem** - Consider `read_only: true` in production

## Support

For issues specific to Docker deployment:
1. Check the troubleshooting section above
2. Review container logs
3. Verify environment configuration
4. Check resource availability
