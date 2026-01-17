"""
Multi-Asset Portfolio Configuration and Management.
Defines asset classes, allocation strategies, and portfolio composition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path

class AssetType(str, Enum):
    """Supported asset types."""
    CRYPTO = "crypto"
    EQUITY = "equity"
    COMMODITY = "commodity"
    FOREX = "forex"
    ETF = "etf"
    BOND = "bond"

class RebalanceStrategy(str, Enum):
    """Portfolio rebalancing strategies."""
    THRESHOLD = "threshold"  # Rebalance when drift > threshold
    CALENDAR = "calendar"     # Rebalance on schedule (weekly/monthly)
    MOMENTUM = "momentum"      # Rebalance based on performance
    ADAPTIVE = "adaptive"      # Dynamic based on volatility

@dataclass
class Asset:
    """Single asset in the portfolio."""
    symbol: str
    asset_type: AssetType
    data_source: str  # "binance", "yfinance", "kraken", etc.
    allocation_pct: float  # Target allocation percentage
    risk_limit_pct: float = 2.0  # Max loss per asset
    min_position_usd: float = 50.0
    max_position_usd: Optional[float] = None
    correlation_threshold: float = 0.7  # Alert if correlation > threshold
    active: bool = True
    metadata: Dict = field(default_factory=dict)

@dataclass
class Portfolio:
    """Portfolio configuration and state."""
    name: str
    total_capital: float
    assets: List[Asset]
    rebalance_strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD
    rebalance_threshold_pct: float = 5.0  # Rebalance if drift > 5%
    rebalance_frequency_days: int = 7  # For calendar strategy
    max_correlation: float = 0.8  # Max allowed asset correlation
    diversification_target: float = 0.85  # Target Herfindahl index
    leverage_allowed: bool = False
    max_drawdown_pct: float = 10.0  # Portfolio-level max drawdown
    
    def get_asset(self, symbol: str) -> Optional[Asset]:
        """Get asset by symbol."""
        return next((a for a in self.assets if a.symbol == symbol), None)
    
    def get_total_allocation(self) -> float:
        """Get sum of all allocations."""
        return sum(a.allocation_pct for a in self.assets if a.active)
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate portfolio configuration."""
        errors = []
        
        total_alloc = self.get_total_allocation()
        if not (99 <= total_alloc <= 101):
            errors.append(f"Allocations sum to {total_alloc}%, expected ~100%")
        
        if self.total_capital <= 0:
            errors.append("Total capital must be positive")
        
        for asset in self.assets:
            if asset.allocation_pct < 0:
                errors.append(f"{asset.symbol}: allocation cannot be negative")
            
            if asset.min_position_usd > self.total_capital * asset.allocation_pct / 100:
                errors.append(f"{asset.symbol}: min position exceeds allocation")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_capital": self.total_capital,
            "assets": [
                {
                    "symbol": a.symbol,
                    "asset_type": a.asset_type.value,
                    "data_source": a.data_source,
                    "allocation_pct": a.allocation_pct,
                    "risk_limit_pct": a.risk_limit_pct,
                    "metadata": a.metadata,
                }
                for a in self.assets
            ],
            "rebalance_strategy": self.rebalance_strategy.value,
            "rebalance_threshold_pct": self.rebalance_threshold_pct,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Portfolio':
        """Load from dictionary."""
        assets = [
            Asset(
                symbol=a["symbol"],
                asset_type=AssetType(a["asset_type"]),
                data_source=a["data_source"],
                allocation_pct=a["allocation_pct"],
                risk_limit_pct=a.get("risk_limit_pct", 2.0),
                metadata=a.get("metadata", {}),
            )
            for a in data["assets"]
        ]
        
        return cls(
            name=data["name"],
            total_capital=data["total_capital"],
            assets=assets,
            rebalance_strategy=RebalanceStrategy(data.get("rebalance_strategy", "threshold")),
            rebalance_threshold_pct=data.get("rebalance_threshold_pct", 5.0),
        )

class PortfolioLoader:
    """Load/save portfolio configurations."""
    
    @staticmethod
    def load(path: str) -> Portfolio:
        """Load portfolio from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return Portfolio.from_dict(data)
    
    @staticmethod
    def save(portfolio: Portfolio, path: str):
        """Save portfolio to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2)

# Example portfolio configurations
EXAMPLE_CRYPTO_PORTFOLIO = Portfolio(
    name="Crypto Alpha",
    total_capital=10000,
    assets=[
        Asset(symbol="BTC/USDT", asset_type=AssetType.CRYPTO, data_source="binance", allocation_pct=50),
        Asset(symbol="ETH/USDT", asset_type=AssetType.CRYPTO, data_source="binance", allocation_pct=30),
        Asset(symbol="SOL/USDT", asset_type=AssetType.CRYPTO, data_source="binance", allocation_pct=20),
    ],
)

EXAMPLE_MULTI_ASSET_PORTFOLIO = Portfolio(
    name="Balanced Multi-Asset",
    total_capital=50000,
    assets=[
        Asset(symbol="BTC/USDT", asset_type=AssetType.CRYPTO, data_source="binance", allocation_pct=30),
        Asset(symbol="AAPL", asset_type=AssetType.EQUITY, data_source="yfinance", allocation_pct=25),
        Asset(symbol="GC=F", asset_type=AssetType.COMMODITY, data_source="yfinance", allocation_pct=20),
        Asset(symbol="SPY", asset_type=AssetType.ETF, data_source="yfinance", allocation_pct=25),
    ],
)
