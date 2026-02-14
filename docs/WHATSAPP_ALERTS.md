# WhatsApp Trade Alerts Integration

This document explains how WhatsApp trade alerts work with Clawdbot.

## Overview

The trading bot writes alerts to a JSON file (`data/trade_alerts.json`) that Clawdbot can poll and deliver via WhatsApp. This decoupled design allows:

- Trading bot runs independently
- Clawdbot handles message delivery
- No direct WhatsApp API integration needed
- Works with Clawdbot's existing WhatsApp channel

## Alert Flow

```
Trading Bot → trade_alerts.json → Clawdbot (poll) → WhatsApp
```

1. **Trade occurs** in unified engine
2. **Alert written** to `data/trade_alerts.json`
3. **Clawdbot polls** via heartbeat or cron
4. **Message delivered** via WhatsApp
5. **Alert marked** as delivered

## Alert Types

| Type | Priority | When |
|------|----------|------|
| `trade_entry` | HIGH | Position opened |
| `trade_exit` | NORMAL/HIGH | Position closed (HIGH if loss) |
| `signal` | NORMAL | Signal generated (optional) |
| `daily_summary` | NORMAL | End of day summary |
| `risk_alert` | NORMAL-URGENT | Risk threshold hit |
| `system_status` | NORMAL/HIGH | System start/stop/error |

## Clawdbot Integration

### Polling for Alerts

Use the polling script or import directly:

```bash
# Get pending alerts as JSON
python scripts/tools/poll_trade_alerts.py --pending

# Mark an alert as delivered
python scripts/tools/poll_trade_alerts.py --mark-delivered <ALERT_ID>

# Show system status
python scripts/tools/poll_trade_alerts.py --status
```

### Python Integration

```python
from bot.whatsapp_alerts import get_pending_alerts, mark_alert_delivered

# Get undelivered alerts
alerts = get_pending_alerts()

for alert in alerts:
    message = alert['message']
    alert_id = alert['id']
    
    # Send via WhatsApp...
    # success = send_whatsapp_message(message)
    
    # Mark as delivered
    mark_alert_delivered(alert_id)
```

## Alert Format

Alerts use WhatsApp-compatible formatting:

- **Bold**: `*text*`
- **Italic**: `_text_`
- Emojis for visual indicators
- Confidence bars using Unicode blocks

Example trade entry alert:

```
🔔 *TRADE ALERT*

🟢 *LONG* TSLA

*Entry:* $248.50
*Size:* 10.00
*Value:* $2,485.00
*Stop Loss:* $245.00 (1.4%)
*Take Profit:* $260.00 (4.6%)
*Confidence:* [███████░░░] 72%
*Reason:* ML momentum signal

_2026-02-14 13:46:51_
```

## Configuration

### Enable/Disable

Set environment variable:

```bash
# Enable (default)
export WHATSAPP_ALERTS_ENABLED=true

# Disable
export WHATSAPP_ALERTS_ENABLED=false
```

### Custom Alert File

```python
from bot.whatsapp_alerts import WhatsAppAlertManager
from pathlib import Path

manager = WhatsAppAlertManager(
    alert_file=Path("custom/path/alerts.json")
)
```

## Heartbeat Integration

Add to Clawdbot's `HEARTBEAT.md`:

```markdown
## Trade Alerts
- Check `C:\Users\Ogulcan\Desktop\Projects\algo_trading_lab\data\trade_alerts.json`
- Deliver any pending alerts via WhatsApp
- Mark delivered alerts
```

Or use a cron job:

```bash
# Poll every 5 minutes
*/5 * * * * cd /path/to/algo_trading_lab && python scripts/tools/poll_trade_alerts.py --pending
```

## Testing

Generate test alerts:

```bash
python -c "from bot.whatsapp_alerts import test_alerts; test_alerts()"
```

Or use the script:

```bash
python scripts/tools/poll_trade_alerts.py --test
```

## Maintenance

Clear delivered alerts periodically:

```bash
python scripts/tools/poll_trade_alerts.py --clear-delivered
```

The system automatically keeps only the last 100 alerts to prevent file growth.

## Troubleshooting

### Alerts not appearing

1. Check `WHATSAPP_ALERTS_ENABLED=true`
2. Verify `data/trade_alerts.json` exists
3. Run `--status` to see system state

### Duplicate alerts

Each alert has a unique ID. Clawdbot should:
1. Check if alert was already delivered (`delivered: true`)
2. Only send alerts where `delivered: false`
3. Mark as delivered after successful send

### Alert file too large

Run `--clear-delivered` to remove old alerts.
