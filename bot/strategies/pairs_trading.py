"""
Pairs Trading Strategy - Statistical arbitrage on correlated assets.

Identifies cointegrated pairs and trades mean reversion of the spread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint, adfuller
    from statsmodels.regression.linear_model import OLS
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logger = logging.getLogger(__name__)


@dataclass
class PairStats:
    """Statistics for a trading pair."""
    asset1: str
    asset2: str
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    half_life: float  # Mean reversion speed
    current_zscore: float
    is_cointegrated: bool
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "asset1": self.asset1,
            "asset2": self.asset2,
            "correlation": round(self.correlation, 4),
            "cointegration_pvalue": round(self.cointegration_pvalue, 4),
            "hedge_ratio": round(self.hedge_ratio, 4),
            "spread_mean": round(self.spread_mean, 6),
            "spread_std": round(self.spread_std, 6),
            "half_life": round(self.half_life, 2),
            "current_zscore": round(self.current_zscore, 4),
            "is_cointegrated": self.is_cointegrated,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class PairsSignal:
    """Trading signal for a pair."""
    pair_name: str
    asset1: str
    asset2: str
    action: Literal["LONG_SPREAD", "SHORT_SPREAD", "CLOSE", "FLAT"]
    zscore: float
    entry_zscore: Optional[float]
    confidence: float
    position_ratio: float  # Units of asset1 per unit of asset2
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "pair_name": self.pair_name,
            "asset1": self.asset1,
            "asset2": self.asset2,
            "action": self.action,
            "zscore": round(self.zscore, 4),
            "entry_zscore": round(self.entry_zscore, 4) if self.entry_zscore else None,
            "confidence": round(self.confidence, 4),
            "position_ratio": round(self.position_ratio, 6),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PairsConfig:
    """Configuration for pairs trading."""
    # Cointegration settings
    cointegration_pvalue: float = 0.05
    min_correlation: float = 0.5
    lookback_days: int = 90

    # Trading signals
    entry_zscore: float = 2.0        # Enter when z-score exceeds this
    exit_zscore: float = 0.5         # Exit when z-score drops below this
    stop_loss_zscore: float = 3.5    # Stop loss if z-score exceeds this

    # Position sizing
    max_position_value: float = 1000  # Max value per leg

    # Pair selection
    min_half_life: int = 5           # Min mean reversion half-life (days)
    max_half_life: int = 30          # Max half-life


class PairsTradingStrategy:
    """
    Statistical arbitrage using cointegrated pairs.

    Process:
    1. Test pairs for cointegration
    2. Calculate spread and z-score
    3. Enter when spread deviates significantly
    4. Exit when spread reverts to mean
    """

    def __init__(self, config: Optional[PairsConfig] = None):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels required. Install: pip install statsmodels")

        self.config = config or PairsConfig()
        self.pair_stats: Dict[str, PairStats] = {}
        self.active_positions: Dict[str, Dict] = {}  # pair_name -> position info

    def test_cointegration(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        asset1: str,
        asset2: str,
    ) -> PairStats:
        """
        Test if two price series are cointegrated.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            asset1: Name of asset 1
            asset2: Name of asset 2

        Returns:
            PairStats with cointegration test results
        """
        # Align series
        prices1, prices2 = prices1.align(prices2, join="inner")
        prices1 = prices1.dropna()
        prices2 = prices2.dropna()

        if len(prices1) < 30:
            return PairStats(
                asset1=asset1,
                asset2=asset2,
                correlation=0,
                cointegration_pvalue=1.0,
                hedge_ratio=1.0,
                spread_mean=0,
                spread_std=1,
                half_life=float("inf"),
                current_zscore=0,
                is_cointegrated=False,
            )

        # Calculate correlation
        correlation = prices1.corr(prices2)

        # Cointegration test
        score, pvalue, _ = coint(prices1, prices2)

        # Calculate hedge ratio using OLS
        model = OLS(prices1, prices2).fit()
        hedge_ratio = model.params[0]

        # Calculate spread
        spread = prices1 - hedge_ratio * prices2
        spread_mean = spread.mean()
        spread_std = spread.std()

        # Calculate half-life of mean reversion
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        spread_lag = spread_lag.dropna()
        spread_diff = spread_diff.dropna()

        if len(spread_lag) > 0 and len(spread_diff) > 0:
            model_hl = OLS(spread_diff, spread_lag.values[:len(spread_diff)]).fit()
            if model_hl.params[0] < 0:
                half_life = -np.log(2) / model_hl.params[0]
            else:
                half_life = float("inf")
        else:
            half_life = float("inf")

        # Current z-score
        current_spread = spread.iloc[-1]
        current_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        # Check if cointegrated
        is_cointegrated = (
            pvalue < self.config.cointegration_pvalue and
            abs(correlation) > self.config.min_correlation and
            self.config.min_half_life < half_life < self.config.max_half_life
        )

        stats = PairStats(
            asset1=asset1,
            asset2=asset2,
            correlation=correlation,
            cointegration_pvalue=pvalue,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            current_zscore=current_zscore,
            is_cointegrated=is_cointegrated,
        )

        # Cache stats
        pair_name = f"{asset1}_{asset2}"
        self.pair_stats[pair_name] = stats

        return stats

    def find_pairs(
        self,
        price_data: Dict[str, pd.DataFrame],
        min_correlation: Optional[float] = None,
    ) -> List[PairStats]:
        """
        Find cointegrated pairs from price data.

        Args:
            price_data: Dict mapping symbol to DataFrame with 'close' column
            min_correlation: Minimum correlation threshold

        Returns:
            List of cointegrated PairStats sorted by quality
        """
        min_corr = min_correlation or self.config.min_correlation
        symbols = list(price_data.keys())
        pairs = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                prices1 = price_data[sym1]["close"]
                prices2 = price_data[sym2]["close"]

                # Quick correlation check first
                corr = prices1.corr(prices2)
                if abs(corr) < min_corr:
                    continue

                # Full cointegration test
                stats = self.test_cointegration(prices1, prices2, sym1, sym2)
                if stats.is_cointegrated:
                    pairs.append(stats)

        # Sort by quality (lower p-value and half-life closer to ideal)
        pairs.sort(key=lambda p: (p.cointegration_pvalue, abs(p.half_life - 15)))

        logger.info(f"Found {len(pairs)} cointegrated pairs from {len(symbols)} assets")
        return pairs

    def generate_signal(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        pair_stats: PairStats,
    ) -> PairsSignal:
        """
        Generate trading signal for a pair.

        Args:
            prices1: Current price series for asset 1
            prices2: Current price series for asset 2
            pair_stats: Pre-calculated pair statistics

        Returns:
            PairsSignal with trading decision
        """
        pair_name = f"{pair_stats.asset1}_{pair_stats.asset2}"

        # Calculate current spread and z-score
        current_spread = prices1.iloc[-1] - pair_stats.hedge_ratio * prices2.iloc[-1]
        zscore = (current_spread - pair_stats.spread_mean) / pair_stats.spread_std if pair_stats.spread_std > 0 else 0

        # Update stats with current z-score
        pair_stats.current_zscore = zscore

        # Check if we have an active position
        active_position = self.active_positions.get(pair_name)

        # Determine action
        action = "FLAT"
        reasoning = ""
        confidence = 0
        entry_zscore = None

        if active_position:
            entry_zscore = active_position.get("entry_zscore")
            position_type = active_position.get("type")

            # Check for exit conditions
            if abs(zscore) < self.config.exit_zscore:
                action = "CLOSE"
                reasoning = f"Z-score reverted to {zscore:.2f}, closing position"
                confidence = 0.8
            elif abs(zscore) > self.config.stop_loss_zscore:
                action = "CLOSE"
                reasoning = f"Stop loss triggered: z-score {zscore:.2f} exceeded {self.config.stop_loss_zscore}"
                confidence = 0.9
            else:
                action = "FLAT"  # Hold position
                reasoning = f"Holding {position_type} position, z-score: {zscore:.2f}"
        else:
            # Check for entry conditions
            if zscore > self.config.entry_zscore:
                action = "SHORT_SPREAD"  # Spread too high, short asset1, long asset2
                reasoning = f"Spread elevated (z={zscore:.2f}), shorting spread"
                confidence = min(0.9, abs(zscore) / 4)
            elif zscore < -self.config.entry_zscore:
                action = "LONG_SPREAD"  # Spread too low, long asset1, short asset2
                reasoning = f"Spread depressed (z={zscore:.2f}), going long spread"
                confidence = min(0.9, abs(zscore) / 4)
            else:
                action = "FLAT"
                reasoning = f"Z-score {zscore:.2f} within normal range"

        return PairsSignal(
            pair_name=pair_name,
            asset1=pair_stats.asset1,
            asset2=pair_stats.asset2,
            action=action,
            zscore=zscore,
            entry_zscore=entry_zscore,
            confidence=confidence,
            position_ratio=pair_stats.hedge_ratio,
            reasoning=reasoning,
        )

    def open_position(
        self,
        pair_name: str,
        position_type: Literal["LONG_SPREAD", "SHORT_SPREAD"],
        entry_zscore: float,
    ):
        """Record opening a pairs position."""
        self.active_positions[pair_name] = {
            "type": position_type,
            "entry_zscore": entry_zscore,
            "opened_at": datetime.now().isoformat(),
        }
        logger.info(f"Opened {position_type} on {pair_name} at z={entry_zscore:.2f}")

    def close_position(self, pair_name: str):
        """Record closing a pairs position."""
        if pair_name in self.active_positions:
            del self.active_positions[pair_name]
            logger.info(f"Closed position on {pair_name}")

    def calculate_position_sizes(
        self,
        pair_stats: PairStats,
        capital: float,
        price1: float,
        price2: float,
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for each leg of the pair.

        Returns:
            Tuple of (quantity1, quantity2) to trade
        """
        # Use max position value per leg
        max_value = min(self.config.max_position_value, capital * 0.1)

        # Asset 2 is the base, calculate quantity
        qty2 = max_value / price2

        # Asset 1 quantity based on hedge ratio
        qty1 = qty2 * pair_stats.hedge_ratio

        # Verify total value doesn't exceed limits
        total_value = qty1 * price1 + qty2 * price2
        if total_value > capital * 0.2:
            scale = (capital * 0.2) / total_value
            qty1 *= scale
            qty2 *= scale

        return qty1, qty2

    def get_active_pairs(self) -> List[str]:
        """Get list of pairs with active positions."""
        return list(self.active_positions.keys())

    def get_pair_stats(self, pair_name: str) -> Optional[PairStats]:
        """Get stats for a specific pair."""
        return self.pair_stats.get(pair_name)

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all analyzed pairs."""
        return {name: stats.to_dict() for name, stats in self.pair_stats.items()}


def create_pairs_strategy(config: Optional[PairsConfig] = None) -> PairsTradingStrategy:
    """Factory function to create pairs trading strategy."""
    return PairsTradingStrategy(config=config)
