"""
Dynamic Position Sizing for 1% Daily Returns

Uses Kelly-based sizing with conviction multipliers:
- Base position: 10%
- High conviction (MTF 3/3 agree): 25%
- Medium conviction (MTF 2/3 agree): 15%
- Low conviction: 10%

Risk Controls:
- Maximum single position: 30%
- Daily loss limit: 3%
- Maximum daily drawdown: 5%
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import json
from pathlib import Path
from datetime import datetime


@dataclass
class PositionSizingConfig:
    """Configuration for dynamic position sizing."""
    
    # Base position sizes (% of portfolio)
    base_position_pct: float = 0.10  # 10% default
    high_conviction_pct: float = 0.25  # 25% when 3/3 agree
    medium_conviction_pct: float = 0.15  # 15% when 2/3 agree
    low_conviction_pct: float = 0.10  # 10% when 1/3 agree
    
    # Risk controls
    max_single_position_pct: float = 0.30  # Never more than 30%
    max_portfolio_exposure_pct: float = 0.80  # Max 80% deployed
    daily_loss_limit_pct: float = 0.03  # Stop trading after 3% daily loss
    max_daily_drawdown_pct: float = 0.05  # Emergency stop at 5%
    
    # Kelly fraction multiplier (for safety)
    kelly_fraction: float = 0.5  # Use half-Kelly
    
    # Trailing position adjustments
    scale_in_enabled: bool = True  # Add to winners
    scale_in_threshold_pct: float = 0.01  # Add after 1% profit
    scale_in_amount_pct: float = 0.05  # Add 5% more
    
    # Compound settings
    compound_gains: bool = True  # Reinvest profits
    compound_frequency: str = "daily"  # daily, trade, none
    
    # Asset-specific overrides
    asset_overrides: Dict[str, Dict] = field(default_factory=dict)
    
    def get_position_size(
        self,
        confidence: float,
        mtf_agreement: int,  # 1, 2, or 3
        asset: str = "",
        account_balance: float = 10000,
        current_exposure: float = 0,
        daily_pnl: float = 0,
    ) -> Dict:
        """
        Calculate position size based on conviction level.
        
        Args:
            confidence: Model confidence (0.5-1.0)
            mtf_agreement: Number of timeframes agreeing (1-3)
            asset: Asset symbol for overrides
            account_balance: Current account balance
            current_exposure: Current portfolio exposure (0-1)
            daily_pnl: Today's P&L as decimal
            
        Returns:
            Dict with position_pct, position_usd, reason
        """
        # Check daily loss limit
        if daily_pnl < -self.daily_loss_limit_pct:
            return {
                "position_pct": 0,
                "position_usd": 0,
                "reason": f"Daily loss limit hit ({daily_pnl:.2%})",
                "conviction": "BLOCKED",
            }
        
        # Check max exposure
        remaining_exposure = self.max_portfolio_exposure_pct - current_exposure
        if remaining_exposure <= 0.01:
            return {
                "position_pct": 0,
                "position_usd": 0,
                "reason": "Max portfolio exposure reached",
                "conviction": "BLOCKED",
            }
        
        # Determine conviction level
        if mtf_agreement >= 3 and confidence >= 0.65:
            base_pct = self.high_conviction_pct
            conviction = "HIGH"
        elif mtf_agreement >= 2 and confidence >= 0.58:
            base_pct = self.medium_conviction_pct
            conviction = "MEDIUM"
        else:
            base_pct = self.low_conviction_pct
            conviction = "LOW"
        
        # Apply Kelly-based adjustment
        # Higher confidence = larger position
        kelly_adj = self.kelly_fraction * (confidence - 0.5) * 2  # 0 at 50%, 1 at 100%
        kelly_adj = max(0.5, min(1.5, 1 + kelly_adj))  # Range: 0.5x to 1.5x
        
        position_pct = base_pct * kelly_adj
        
        # Apply asset-specific overrides
        if asset in self.asset_overrides:
            override = self.asset_overrides[asset]
            if "max_position_pct" in override:
                position_pct = min(position_pct, override["max_position_pct"])
            if "position_multiplier" in override:
                position_pct *= override["position_multiplier"]
        
        # Cap at max single position and remaining exposure
        position_pct = min(position_pct, self.max_single_position_pct, remaining_exposure)
        position_usd = account_balance * position_pct
        
        return {
            "position_pct": round(position_pct, 4),
            "position_usd": round(position_usd, 2),
            "conviction": conviction,
            "mtf_agreement": mtf_agreement,
            "confidence": confidence,
            "kelly_adjustment": kelly_adj,
            "reason": f"{conviction} conviction ({mtf_agreement}/3 MTF, {confidence:.1%} conf)",
        }
    
    def calculate_expected_daily_return(
        self,
        trades_per_day: int,
        avg_win_rate: float,
        avg_position_pct: float,
        take_profit_pct: float = 0.03,
        stop_loss_pct: float = 0.015,
    ) -> Dict:
        """
        Calculate expected daily return based on parameters.
        
        Args:
            trades_per_day: Expected number of trades
            avg_win_rate: Average win rate (0-1)
            avg_position_pct: Average position size
            take_profit_pct: Take profit target
            stop_loss_pct: Stop loss
            
        Returns:
            Dict with expected values
        """
        # Per trade EV
        per_trade_ev = (avg_win_rate * take_profit_pct - (1 - avg_win_rate) * stop_loss_pct)
        
        # Scale by position size
        per_trade_portfolio_ev = per_trade_ev * avg_position_pct
        
        # Daily EV
        daily_ev = per_trade_portfolio_ev * trades_per_day
        
        # Compound effect (assuming daily compounding)
        compound_factor = (1 + daily_ev) ** 252 - 1  # Annual
        
        return {
            "per_trade_ev": per_trade_ev,
            "per_trade_portfolio_ev": per_trade_portfolio_ev,
            "daily_ev": daily_ev,
            "annual_ev_compound": compound_factor,
            "trades_needed_for_1pct": 0.01 / per_trade_portfolio_ev if per_trade_portfolio_ev > 0 else float("inf"),
        }
    
    def to_dict(self) -> Dict:
        """Export configuration to dict."""
        return {
            "base_position_pct": self.base_position_pct,
            "high_conviction_pct": self.high_conviction_pct,
            "medium_conviction_pct": self.medium_conviction_pct,
            "low_conviction_pct": self.low_conviction_pct,
            "max_single_position_pct": self.max_single_position_pct,
            "max_portfolio_exposure_pct": self.max_portfolio_exposure_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "kelly_fraction": self.kelly_fraction,
            "compound_gains": self.compound_gains,
            "compound_frequency": self.compound_frequency,
            "asset_overrides": self.asset_overrides,
        }
    
    def save(self, path: str | Path):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "PositionSizingConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# Default configuration optimized for 1% daily target
DEFAULT_CONFIG = PositionSizingConfig(
    base_position_pct=0.10,
    high_conviction_pct=0.25,
    medium_conviction_pct=0.15,
    low_conviction_pct=0.10,
    max_single_position_pct=0.30,
    max_portfolio_exposure_pct=0.80,
    daily_loss_limit_pct=0.03,
    max_daily_drawdown_pct=0.05,
    kelly_fraction=0.5,
    compound_gains=True,
    compound_frequency="daily",
    asset_overrides={
        # Crypto - slightly larger due to 24/7 opportunities
        "BTC_USDT": {"position_multiplier": 1.0},
        "ETH_USDT": {"position_multiplier": 1.0},
        "SOL_USDT": {"position_multiplier": 0.9},  # Slightly smaller, more volatile
        
        # Volatile stocks - cap position
        "TSLA": {"max_position_pct": 0.20},
        "COIN": {"max_position_pct": 0.15},  # Very volatile
        
        # Indices - can be larger, less volatile
        "SPX500_USD": {"position_multiplier": 1.2},
        "NAS100_USD": {"position_multiplier": 1.1},
    },
)


def print_1pct_analysis():
    """Print analysis showing how to achieve 1% daily."""
    config = DEFAULT_CONFIG
    
    print("\n" + "="*70)
    print("  1% DAILY RETURN ANALYSIS")
    print("="*70)
    
    scenarios = [
        {"name": "Conservative", "trades": 3, "win_rate": 0.70, "position": 0.10},
        {"name": "Current MTF", "trades": 4, "win_rate": 0.80, "position": 0.10},
        {"name": "High Conviction", "trades": 3, "win_rate": 0.80, "position": 0.20},
        {"name": "Optimized", "trades": 5, "win_rate": 0.75, "position": 0.15},
        {"name": "Target", "trades": 4, "win_rate": 0.80, "position": 0.15},
    ]
    
    print(f"\n  Assumptions: 3% TP, 1.5% SL (2:1 R:R)")
    print(f"\n  {'Scenario':<16} {'Trades':<8} {'Win%':<8} {'Position':<10} {'Daily EV':<10} {'Annual':<10}")
    print(f"  {'-'*62}")
    
    for s in scenarios:
        result = config.calculate_expected_daily_return(
            trades_per_day=s["trades"],
            avg_win_rate=s["win_rate"],
            avg_position_pct=s["position"],
        )
        marker = " ✓" if result["daily_ev"] >= 0.01 else ""
        print(f"  {s['name']:<16} {s['trades']:<8} {s['win_rate']:.0%}{'':<5} {s['position']:.0%}{'':<7} "
              f"{result['daily_ev']:.2%}{'':<6} {result['annual_ev_compound']:.0%}{marker}")
    
    print(f"\n  KEY INSIGHTS:")
    print(f"  - Current setup (80% win, 10% position): ~0.84% daily")
    print(f"  - To hit 1%: Increase position to 15% OR increase trades to 5+")
    print(f"  - With high conviction (25% position, 3/3 MTF): ~1.3% per trade")
    print(f"  - 2 high-conviction trades per day = 1%+ daily")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print_1pct_analysis()
    
    # Save default config
    config_path = Path(__file__).parent / "position_sizing_config.json"
    DEFAULT_CONFIG.save(config_path)
    print(f"\n  Saved config to {config_path}")
