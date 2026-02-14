#!/usr/bin/env python3
"""
Poll Trade Alerts for Clawdbot Integration.

This script is designed to be run by Clawdbot to check for pending trade alerts
and return them for WhatsApp delivery.

Usage:
    python scripts/tools/poll_trade_alerts.py [--mark-delivered ALERT_ID]
    python scripts/tools/poll_trade_alerts.py --pending
    python scripts/tools/poll_trade_alerts.py --test

Output:
    JSON formatted alerts for Clawdbot to parse and deliver.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.whatsapp_alerts import (
    get_whatsapp_alert_manager,
    get_pending_alerts,
    mark_alert_delivered,
    test_alerts,
)


def main():
    parser = argparse.ArgumentParser(description="Poll trade alerts for WhatsApp delivery")
    parser.add_argument("--pending", action="store_true", help="Get pending alerts as JSON")
    parser.add_argument("--mark-delivered", type=str, help="Mark an alert as delivered by ID")
    parser.add_argument("--clear-delivered", action="store_true", help="Clear all delivered alerts")
    parser.add_argument("--test", action="store_true", help="Generate test alerts")
    parser.add_argument("--status", action="store_true", help="Show alert system status")
    
    args = parser.parse_args()
    
    manager = get_whatsapp_alert_manager()
    
    if args.test:
        # Generate test alerts
        alerts = test_alerts()
        print(f"\n✅ Generated {len(alerts)} test alerts")
        return
    
    if args.mark_delivered:
        # Mark specific alert as delivered
        success = mark_alert_delivered(args.mark_delivered)
        if success:
            print(json.dumps({"success": True, "alert_id": args.mark_delivered}))
        else:
            print(json.dumps({"success": False, "error": "Alert not found"}))
        return
    
    if args.clear_delivered:
        # Clear delivered alerts
        count = manager.clear_delivered()
        print(json.dumps({"success": True, "cleared": count}))
        return
    
    if args.status:
        # Show status
        pending = get_pending_alerts()
        all_alerts = manager._load_alerts()
        delivered = len(all_alerts) - len(pending)
        
        print(f"""
📊 WhatsApp Alert System Status
================================
Enabled: {manager.enabled}
Alert file: {manager.alert_file}
Total alerts: {len(all_alerts)}
Pending: {len(pending)}
Delivered: {delivered}
""")
        return
    
    # Default: get pending alerts
    pending = get_pending_alerts()
    
    if not pending:
        print(json.dumps({"alerts": [], "count": 0}))
        return
    
    # Output alerts for Clawdbot
    output = {
        "alerts": pending,
        "count": len(pending),
    }
    
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
