# API Contracts

> **Response shapes must NEVER change unless versioned.**
> This document defines the contract between API and consumers.

---

## Base URL

```
http://localhost:8000
```

## Authentication

All protected endpoints require API key in header:
```
X-API-Key: your_api_key
```

---

## Core Endpoints

### Health Check

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

**Contract:**
- `status`: string, one of: "healthy", "degraded", "unhealthy"
- `timestamp`: ISO 8601 string
- `version`: semantic version string

---

### Bot Status

```http
GET /status
```

**Response (200 OK):**
```json
{
  "mode": "paper_live_data",
  "status": "running",
  "current_balance": 10250.50,
  "initial_balance": 10000.00,
  "pnl_pct": 2.505,
  "positions": [
    {
      "symbol": "BTC/USDT",
      "side": "long",
      "quantity": 0.01,
      "entry_price": 42000.00,
      "current_price": 42500.00,
      "unrealized_pnl": 5.00,
      "unrealized_pnl_pct": 1.19
    }
  ],
  "last_update": "2026-01-15T10:30:00Z"
}
```

**Contract:**
- `mode`: string, one of: "paper_live_data", "paper_historical", "live"
- `status`: string, one of: "running", "paused", "stopped"
- `current_balance`: float, current portfolio value
- `initial_balance`: float, starting balance
- `pnl_pct`: float, percentage P&L
- `positions`: array of position objects
- `last_update`: ISO 8601 string

---

## Trading Control Endpoints

### Get Risk Settings

```http
GET /api/trading/risk-settings
```

**Response (200 OK):**
```json
{
  "shorting": false,
  "leverage": false,
  "aggressive": false
}
```

**Contract:**
- All fields are boolean
- Default values are `false`

---

### Update Risk Settings

```http
POST /api/trading/risk-settings
Content-Type: application/json

{
  "shorting": true,
  "leverage": false,
  "aggressive": false
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "settings": {
    "shorting": true,
    "leverage": false,
    "aggressive": false
  }
}
```

**Contract:**
- `success`: boolean
- `settings`: object with updated values

---

### Pause Trading

```http
POST /api/trading/pause-all
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "All trading paused",
  "affected_markets": ["crypto", "stocks"]
}
```

---

### Resume Trading

```http
POST /api/trading/resume-all
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "All trading resumed",
  "affected_markets": ["crypto", "stocks"]
}
```

---

## AI Brain Endpoints

### Get AI Brain Status

```http
GET /api/ai-brain/status
```

**Response (200 OK):**
```json
{
  "active_strategy": "Balanced Growth",
  "daily_pnl_pct": 0.75,
  "daily_target_pct": 1.0,
  "max_daily_loss_pct": 2.0,
  "trades_today": 3,
  "target_achieved": false,
  "can_still_trade": true,
  "market_condition": "bull",
  "confidence": 0.72
}
```

**Contract:**
- `active_strategy`: string or null
- `daily_pnl_pct`: float, current day P&L
- `daily_target_pct`: float, target percentage
- `max_daily_loss_pct`: float, loss limit
- `trades_today`: integer
- `target_achieved`: boolean
- `can_still_trade`: boolean
- `market_condition`: string, one of: "bull", "bear", "sideways", "volatile", "unknown"
- `confidence`: float, 0-1

---

### Get Daily Target Progress

```http
GET /api/ai-brain/daily-target
```

**Response (200 OK):**
```json
{
  "target_pct": 1.0,
  "current_pct": 0.75,
  "progress": 0.75,
  "target_achieved": false,
  "trades_today": 3,
  "winning_trades": 2,
  "losing_trades": 1,
  "can_still_trade": true,
  "status": "on_track"
}
```

**Contract:**
- `progress`: float, 0-1+ (current/target ratio)
- `status`: string, one of: "on_track", "target_achieved", "loss_limit_hit", "paused"

---

### Get Available Strategies

```http
GET /api/ai-brain/strategies
```

**Response (200 OK):**
```json
{
  "available": [
    {
      "name": "Balanced Growth",
      "description": "Moderate risk, steady gains",
      "direction": "long",
      "leverage_ok": false,
      "shorting_ok": false,
      "confidence": 0.75,
      "is_active": true
    },
    {
      "name": "Aggressive Bull",
      "description": "High risk, high reward in bull markets",
      "direction": "long",
      "leverage_ok": true,
      "shorting_ok": false,
      "confidence": 0.65,
      "is_active": false
    }
  ],
  "active_strategy": "Balanced Growth"
}
```

---

### Activate Strategy

```http
POST /api/ai-brain/strategies/{name}/activate
```

**Response (200 OK):**
```json
{
  "success": true,
  "strategy": "Balanced Growth",
  "message": "Strategy activated successfully"
}
```

**Error Response (400):**
```json
{
  "success": false,
  "error": "Backtest failed: negative expected return"
}
```

---

## ML Performance Endpoints

### Get Model Performance

```http
GET /api/ml/model-performance?model_type=xgboost&days=30
```

**Response (200 OK):**
```json
{
  "model_type": "xgboost",
  "market_condition": "all",
  "total_predictions": 150,
  "correct_predictions": 98,
  "accuracy": 65.33,
  "avg_confidence": 0.72,
  "avg_return": 0.45,
  "profit_factor": 1.85,
  "days": 30
}
```

**Contract:**
- `accuracy`: float, percentage (0-100)
- `avg_confidence`: float, 0-1
- `avg_return`: float, percentage
- `profit_factor`: float, gross profit / gross loss

---

### Get Model Ranking

```http
GET /api/ml/model-ranking?days=30
```

**Response (200 OK):**
```json
{
  "period_days": 30,
  "total_models": 4,
  "ranking": [
    {
      "model_type": "ensemble",
      "total_predictions": 200,
      "accuracy": 68.5,
      "avg_confidence": 0.75,
      "avg_return": 0.52,
      "profit_factor": 2.1
    },
    {
      "model_type": "xgboost",
      "total_predictions": 180,
      "accuracy": 65.0,
      "avg_confidence": 0.71,
      "avg_return": 0.38,
      "profit_factor": 1.75
    }
  ]
}
```

---

### Get Best Model for Condition

```http
GET /api/ml/best-model?market_condition=bull&min_predictions=10
```

**Response (200 OK):**
```json
{
  "market_condition": "bull",
  "recommended_model": "xgboost",
  "accuracy": 72.5,
  "total_predictions": 45,
  "avg_return": 0.65
}
```

**No Data Response (200 OK):**
```json
{
  "market_condition": "bull",
  "recommended_model": null,
  "message": "No models with at least 10 predictions for this condition"
}
```

---

### Get Model Recommendation

```http
GET /api/ml/recommendation?market_condition=sideways
```

**Response (200 OK):**
```json
{
  "recommended_model": "technical_analysis",
  "confidence": 0.68,
  "accuracy": 61.2,
  "total_samples": 35,
  "avg_return": 0.22,
  "reason": "Best performer for sideways with 61.2% accuracy"
}
```

---

### Record Prediction Outcome

```http
POST /api/ml/record-outcome
Content-Type: application/json

{
  "prediction_id": "123",
  "actual_return": 1.5
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "prediction_id": "123",
  "actual_return": 1.5
}
```

---

## Trade History Endpoints

### Get Recent Trades

```http
GET /api/trade-history?limit=50
```

**Response (200 OK):**
```json
{
  "trades": [
    {
      "id": "trade_123",
      "symbol": "BTC/USDT",
      "side": "long",
      "quantity": 0.01,
      "entry_price": 42000.00,
      "exit_price": 42500.00,
      "pnl": 5.00,
      "pnl_pct": 1.19,
      "entry_time": "2026-01-15T08:00:00Z",
      "exit_time": "2026-01-15T10:00:00Z",
      "exit_reason": "take_profit",
      "mode": "paper_live_data"
    }
  ],
  "total": 1,
  "limit": 50
}
```

**Contract:**
- `exit_reason`: string, one of: "take_profit", "stop_loss", "trailing_stop", "signal_sell", "manual"
- `mode`: string, one of: "paper_live_data", "paper_historical", "live"

---

### Get Equity History

```http
GET /equity?limit=100
```

**Response (200 OK):**
```json
[
  {
    "timestamp": "2026-01-15T10:00:00Z",
    "balance": 10250.50,
    "pnl_pct": 2.505
  },
  {
    "timestamp": "2026-01-15T09:00:00Z",
    "balance": 10200.00,
    "pnl_pct": 2.0
  }
]
```

---

## Error Response Format

All errors follow this format:

```json
{
  "error": "Error message describing what went wrong",
  "code": "ERROR_CODE",
  "details": {}
}
```

**Common Error Codes:**
- `AUTH_FAILED`: Invalid or missing API key
- `RATE_LIMITED`: Too many requests
- `NOT_FOUND`: Resource not found
- `VALIDATION_ERROR`: Invalid request parameters
- `INTERNAL_ERROR`: Server error

---

## WebSocket Endpoints

### Real-time Updates

```
WS /ws/updates
```

**Message Format:**
```json
{
  "type": "update_type",
  "data": {},
  "timestamp": "2026-01-15T10:30:00Z"
}
```

**Update Types:**
- `price_update`: New price data
- `trade_executed`: Trade was executed
- `position_update`: Position changed
- `status_change`: Bot status changed
- `alert`: Important notification

---

## Rate Limits

| Endpoint Type | Limit |
|--------------|-------|
| Read endpoints | 100/minute |
| Write endpoints | 20/minute |
| WebSocket | 1 connection per client |

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312200
```

---

## Versioning

Current version: **v1** (implicit, no prefix needed)

Future versions will use prefix:
```
/v2/api/...
```

Breaking changes will bump major version.

---

*Last Updated: 2026-01-15*
