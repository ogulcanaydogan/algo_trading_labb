#!/usr/bin/env python3
"""
Generate Test Shadow Decisions.

Phase 2B Operational Script.
Generates simulated shadow decisions to verify the data pipeline is working.

This script creates realistic-looking shadow decision entries for testing
the daily health check and weekly report scripts.

NOTE: This generates TEST data only for pipeline verification.
Production data will come from actual paper trading with shadow mode enabled.
"""

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_decision(
    decision_num: int,
    base_time: datetime,
    symbols: List[str],
    regimes: List[str],
) -> Dict[str, Any]:
    """Generate a single simulated shadow decision."""
    symbol = random.choice(symbols)
    regime = random.choice(regimes)

    # Simulate realistic values
    price = {
        "BTC/USDT": random.uniform(40000, 45000),
        "ETH/USDT": random.uniform(2500, 3000),
        "SOL/USDT": random.uniform(80, 120),
    }.get(symbol, random.uniform(100, 1000))

    # RL recommendation
    rl_actions = ["buy", "hold", "sell", "short"]
    rl_action = random.choice(rl_actions)
    rl_confidence = random.uniform(0.4, 0.95)

    # Actual action (may or may not follow RL)
    follow_rl = random.random() < 0.6  # 60% follow rate
    actual_action = rl_action if follow_rl else random.choice(rl_actions)

    # Gate decision
    gate_approved = random.random() > 0.25  # 75% approval rate
    gate_rejection_reasons = [
        "low_confidence",
        "high_risk",
        "position_limit",
        "cooldown_active",
    ]

    # Costs (realistic values)
    position_value = random.uniform(500, 2000)
    slippage_bps = random.uniform(2, 15)
    fee_bps = 10  # Fixed 0.1%
    spread_bps = random.uniform(3, 10)

    slippage_cost = position_value * (slippage_bps / 10000)
    fee_cost = position_value * (fee_bps / 10000)
    spread_cost = position_value * (spread_bps / 10000)

    # Outcome (if executed)
    executed = gate_approved and actual_action != "hold"
    pnl = 0.0
    pnl_pct = 0.0
    if executed:
        # Simulate P&L with slight positive edge
        pnl_pct = random.gauss(0.2, 1.5)  # Mean +0.2%, std 1.5%
        pnl = position_value * (pnl_pct / 100)

    # Timestamp (spread over the day)
    timestamp = base_time + timedelta(
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

    return {
        "timestamp": timestamp.isoformat(),
        "decision_id": f"DEC_{timestamp.strftime('%Y%m%d%H%M%S')}_{decision_num:04d}",
        "data_mode": "TEST",  # Marks this as test data, not production
        "symbol": symbol,
        "market_context": {
            "price": round(price, 2),
            "volatility": random.uniform(0.01, 0.05),
            "spread_bps": round(spread_bps, 2),
            "daily_volume": random.randint(1000000, 10000000000),
        },
        "regime_context": {
            "regime": regime,
            "regime_confidence": random.uniform(0.6, 0.95),
        },
        "sentiment_context": {
            "news_sentiment": random.uniform(-0.5, 0.5),
            "fear_greed_index": random.uniform(30, 70),
        },
        "rl_recommendation": {
            "enabled": True,
            "action": rl_action,
            "confidence": round(rl_confidence, 4),
            "primary_agent": random.choice([
                "TrendFollower",
                "MeanReversion",
                "MomentumTrader",
                "ShortSpecialist",
                "Scalper",
            ]),
            "strategy_preferences": {
                "TrendFollower": random.uniform(0.1, 0.3),
                "MeanReversion": random.uniform(0.1, 0.3),
                "MomentumTrader": random.uniform(0.1, 0.3),
                "ShortSpecialist": random.uniform(0.1, 0.3),
                "Scalper": random.uniform(0.1, 0.3),
            },
            "directional_bias": random.choice(["bullish", "bearish", "neutral"]),
        },
        "gate_decision": {
            "approved": gate_approved,
            "score": random.uniform(0.3, 0.9) if gate_approved else random.uniform(0.1, 0.4),
            "rejection_reason": "" if gate_approved else random.choice(gate_rejection_reasons),
        },
        "preservation_state": {
            "level": "normal",
            "restrictions": {},
        },
        "actual_decision": {
            "action": actual_action,
            "confidence": random.uniform(0.5, 0.9),
            "strategy_used": random.choice([
                "TrendFollower",
                "MeanReversion",
                "MomentumTrader",
                "Scalper",
            ]),
        },
        "execution": {
            "executed": executed,
            "entry_price": round(price, 2) if executed else 0.0,
            "exit_price": round(price * (1 + pnl_pct / 100), 2) if executed else 0.0,
            "position_size": round(position_value / price, 6) if executed else 0.0,
            "leverage": random.choice([1.0, 1.5, 2.0]) if executed else 1.0,
        },
        "costs": {
            "slippage": round(slippage_cost, 4),
            "fees": round(fee_cost, 4),
            "spread": round(spread_cost, 4),
            "total": round(slippage_cost + fee_cost + spread_cost, 4),
        },
        "outcome": {
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "mae": round(random.uniform(0, abs(pnl_pct) * 2), 4) if executed else 0.0,
            "mfe": round(random.uniform(0, abs(pnl_pct) * 2), 4) if executed else 0.0,
            "holding_time_minutes": random.randint(5, 480) if executed else 0.0,
        },
    }


def main():
    """Generate test shadow decisions."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate test shadow decisions")
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of decisions to generate",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to spread decisions over",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rl/shadow_decisions.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file instead of overwriting",
    )
    args = parser.parse_args()

    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    regimes = ["bull", "bear", "sideways", "volatile", "recovery"]

    # Generate decisions spread over the past N days
    decisions = []
    now = datetime.now()

    for i in range(args.count):
        day_offset = random.randint(0, args.days - 1)
        base_time = now - timedelta(days=day_offset)
        decision = generate_decision(i + 1, base_time, symbols, regimes)
        decisions.append(decision)

    # Sort by timestamp
    decisions.sort(key=lambda x: x["timestamp"])

    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    with open(output_path, mode) as f:
        for decision in decisions:
            f.write(json.dumps(decision) + "\n")

    print(f"Generated {args.count} shadow decisions")
    print(f"Output: {output_path}")
    print(f"Mode: {'append' if args.append else 'overwrite'}")

    # Stats
    executed = sum(1 for d in decisions if d["execution"]["executed"])
    total_pnl = sum(d["outcome"]["pnl"] for d in decisions)
    by_symbol = {}
    for d in decisions:
        sym = d["symbol"]
        by_symbol[sym] = by_symbol.get(sym, 0) + 1

    print(f"\nStatistics:")
    print(f"  Executed trades: {executed}/{args.count}")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  By symbol: {by_symbol}")


if __name__ == "__main__":
    main()
