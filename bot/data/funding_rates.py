"""
Funding Rate Data - Perpetual futures funding rate analysis.

Tracks funding rates across exchanges for sentiment signals
and arbitrage opportunities.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingRate:
    """Single funding rate data point."""

    symbol: str
    exchange: str
    rate: float  # Funding rate (e.g., 0.0001 = 0.01%)
    next_funding_time: datetime
    mark_price: float
    index_price: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def rate_pct(self) -> float:
        """Rate as percentage."""
        return self.rate * 100

    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (assuming 8h intervals)."""
        return self.rate * 3 * 365  # 3 times per day * 365 days

    @property
    def premium(self) -> float:
        """Mark price premium/discount vs index."""
        if self.index_price > 0:
            return (self.mark_price - self.index_price) / self.index_price
        return 0

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "rate": self.rate,
            "rate_pct": round(self.rate_pct, 4),
            "annualized_rate": round(self.annualized_rate * 100, 2),
            "next_funding_time": self.next_funding_time.isoformat(),
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "premium_pct": round(self.premium * 100, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FundingSignal:
    """Trading signal from funding rate analysis."""

    symbol: str
    signal: Literal["LONG", "SHORT", "NEUTRAL"]
    strength: float  # 0 to 1
    avg_rate: float
    rate_trend: Literal["rising", "falling", "stable"]
    exchange_divergence: float  # How much exchanges disagree
    is_extreme: bool  # Extreme funding (potential reversal)
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "strength": round(self.strength, 4),
            "avg_rate_pct": round(self.avg_rate * 100, 4),
            "rate_trend": self.rate_trend,
            "exchange_divergence": round(self.exchange_divergence, 4),
            "is_extreme": self.is_extreme,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FundingArbitrage:
    """Funding rate arbitrage opportunity."""

    symbol: str
    long_exchange: str
    short_exchange: str
    rate_differential: float
    expected_profit_8h: float
    expected_profit_annual: float
    is_viable: bool
    min_capital: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "long_exchange": self.long_exchange,
            "short_exchange": self.short_exchange,
            "rate_differential_pct": round(self.rate_differential * 100, 4),
            "expected_profit_8h_pct": round(self.expected_profit_8h * 100, 4),
            "expected_profit_annual_pct": round(self.expected_profit_annual * 100, 2),
            "is_viable": self.is_viable,
            "min_capital": self.min_capital,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FundingConfig:
    """Funding rate analysis configuration."""

    # History settings
    history_size: int = 100  # Number of rates to keep

    # Signal thresholds
    positive_threshold: float = 0.0003  # 0.03% (bullish sentiment)
    negative_threshold: float = -0.0003  # -0.03% (bearish sentiment)
    extreme_threshold: float = 0.001  # 0.1% (extreme, potential reversal)

    # Arbitrage settings
    min_arb_spread: float = 0.0002  # 0.02% minimum spread
    arb_fee_estimate: float = 0.001  # 0.1% total fees (entry + exit)
    min_arb_capital: float = 10000  # Minimum capital for arb

    # Trend detection
    trend_window: int = 10  # Number of samples for trend


class FundingRateTracker:
    """
    Track and analyze perpetual funding rates.

    Features:
    - Multi-exchange funding rate tracking
    - Sentiment signal generation
    - Funding arbitrage detection
    - Rate trend analysis
    """

    def __init__(self, config: Optional[FundingConfig] = None):
        self.config = config or FundingConfig()
        self._rates: Dict[str, Dict[str, Deque[FundingRate]]] = {}  # symbol -> exchange -> rates
        self._latest: Dict[str, Dict[str, FundingRate]] = {}  # symbol -> exchange -> latest rate

    def add_rate(self, rate: FundingRate):
        """Add a funding rate observation."""
        symbol = rate.symbol
        exchange = rate.exchange

        # Initialize storage
        if symbol not in self._rates:
            self._rates[symbol] = {}
            self._latest[symbol] = {}

        if exchange not in self._rates[symbol]:
            self._rates[symbol][exchange] = deque(maxlen=self.config.history_size)

        self._rates[symbol][exchange].append(rate)
        self._latest[symbol][exchange] = rate

        logger.debug(f"Added funding rate: {symbol} {exchange} {rate.rate_pct:.4f}%")

    def add_rate_data(
        self,
        symbol: str,
        exchange: str,
        rate: float,
        next_funding_time: datetime,
        mark_price: float = 0,
        index_price: float = 0,
    ) -> FundingRate:
        """Add funding rate from raw data."""
        funding_rate = FundingRate(
            symbol=symbol,
            exchange=exchange,
            rate=rate,
            next_funding_time=next_funding_time,
            mark_price=mark_price,
            index_price=index_price,
        )
        self.add_rate(funding_rate)
        return funding_rate

    def get_latest(self, symbol: str, exchange: Optional[str] = None) -> Optional[FundingRate]:
        """Get latest funding rate."""
        if symbol not in self._latest:
            return None

        if exchange:
            return self._latest[symbol].get(exchange)

        # Return rate with most recent timestamp
        rates = list(self._latest[symbol].values())
        if rates:
            return max(rates, key=lambda r: r.timestamp)
        return None

    def get_average_rate(self, symbol: str, window: Optional[int] = None) -> float:
        """Get average funding rate across exchanges."""
        if symbol not in self._latest:
            return 0

        rates = [r.rate for r in self._latest[symbol].values()]
        return np.mean(rates) if rates else 0

    def get_rate_history(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        limit: int = 20,
    ) -> List[FundingRate]:
        """Get funding rate history."""
        if symbol not in self._rates:
            return []

        if exchange:
            history = list(self._rates[symbol].get(exchange, []))
        else:
            # Combine all exchanges
            history = []
            for ex_rates in self._rates[symbol].values():
                history.extend(list(ex_rates))
            history.sort(key=lambda r: r.timestamp, reverse=True)

        return history[:limit]

    def analyze_sentiment(self, symbol: str) -> FundingSignal:
        """
        Analyze funding rate for sentiment signal.

        Positive funding = longs pay shorts = bullish sentiment
        Negative funding = shorts pay longs = bearish sentiment
        Extreme funding = potential reversal

        Args:
            symbol: Trading symbol

        Returns:
            FundingSignal with analysis
        """
        if symbol not in self._latest or not self._latest[symbol]:
            return FundingSignal(
                symbol=symbol,
                signal="NEUTRAL",
                strength=0,
                avg_rate=0,
                rate_trend="stable",
                exchange_divergence=0,
                is_extreme=False,
                reasoning="No funding rate data available",
            )

        # Get current rates
        rates = [r.rate for r in self._latest[symbol].values()]
        avg_rate = np.mean(rates)

        # Calculate divergence between exchanges
        divergence = np.std(rates) if len(rates) > 1 else 0

        # Detect trend
        trend = self._detect_trend(symbol)

        # Check if extreme
        is_extreme = abs(avg_rate) > self.config.extreme_threshold

        # Generate signal
        if is_extreme:
            # Contrarian signal at extremes
            if avg_rate > self.config.extreme_threshold:
                signal = "SHORT"  # Fade extreme bullish
                reasoning = f"Extreme positive funding ({avg_rate * 100:.3f}%), potential reversal"
            else:
                signal = "LONG"  # Fade extreme bearish
                reasoning = f"Extreme negative funding ({avg_rate * 100:.3f}%), potential reversal"
            strength = min(1.0, abs(avg_rate) / self.config.extreme_threshold / 2)
        elif avg_rate > self.config.positive_threshold:
            signal = "LONG"  # Follow bullish sentiment
            reasoning = f"Positive funding ({avg_rate * 100:.3f}%) indicates bullish sentiment"
            strength = min(1.0, avg_rate / self.config.extreme_threshold)
        elif avg_rate < self.config.negative_threshold:
            signal = "SHORT"  # Follow bearish sentiment
            reasoning = f"Negative funding ({avg_rate * 100:.3f}%) indicates bearish sentiment"
            strength = min(1.0, abs(avg_rate) / self.config.extreme_threshold)
        else:
            signal = "NEUTRAL"
            reasoning = f"Neutral funding ({avg_rate * 100:.3f}%)"
            strength = 0

        return FundingSignal(
            symbol=symbol,
            signal=signal,
            strength=strength,
            avg_rate=avg_rate,
            rate_trend=trend,
            exchange_divergence=divergence,
            is_extreme=is_extreme,
            reasoning=reasoning,
        )

    def _detect_trend(self, symbol: str) -> str:
        """Detect funding rate trend."""
        if symbol not in self._rates:
            return "stable"

        # Get recent rates across exchanges
        recent_rates = []
        for exchange_rates in self._rates[symbol].values():
            recent = list(exchange_rates)[-self.config.trend_window :]
            recent_rates.extend(recent)

        if len(recent_rates) < 3:
            return "stable"

        # Sort by timestamp
        recent_rates.sort(key=lambda r: r.timestamp)

        # Calculate trend
        first_half = [r.rate for r in recent_rates[: len(recent_rates) // 2]]
        second_half = [r.rate for r in recent_rates[len(recent_rates) // 2 :]]

        if not first_half or not second_half:
            return "stable"

        avg_first = np.mean(first_half)
        avg_second = np.mean(second_half)

        change = avg_second - avg_first
        threshold = 0.0001  # 0.01%

        if change > threshold:
            return "rising"
        elif change < -threshold:
            return "falling"
        return "stable"

    def find_arbitrage(self, symbol: str) -> Optional[FundingArbitrage]:
        """
        Find funding rate arbitrage opportunities.

        Strategy: Long on exchange with negative funding,
        short on exchange with positive funding.

        Args:
            symbol: Trading symbol

        Returns:
            FundingArbitrage if opportunity exists
        """
        if symbol not in self._latest or len(self._latest[symbol]) < 2:
            return None

        rates = list(self._latest[symbol].items())

        # Find min and max rates
        min_rate_ex = min(rates, key=lambda x: x[1].rate)
        max_rate_ex = max(rates, key=lambda x: x[1].rate)

        # Calculate spread
        spread = max_rate_ex[1].rate - min_rate_ex[1].rate

        if spread < self.config.min_arb_spread:
            return None

        # Calculate expected profit (per 8 hours)
        gross_profit = spread
        net_profit = gross_profit - self.config.arb_fee_estimate
        annual_profit = net_profit * 3 * 365  # 3 times per day

        is_viable = net_profit > 0 and spread > self.config.min_arb_spread * 2

        return FundingArbitrage(
            symbol=symbol,
            long_exchange=min_rate_ex[0],  # Go long where funding is lowest
            short_exchange=max_rate_ex[0],  # Go short where funding is highest
            rate_differential=spread,
            expected_profit_8h=net_profit,
            expected_profit_annual=annual_profit,
            is_viable=is_viable,
            min_capital=self.config.min_arb_capital,
        )

    def get_top_funding(
        self,
        direction: Literal["positive", "negative"] = "positive",
        limit: int = 10,
    ) -> List[Dict]:
        """Get symbols with highest/lowest funding rates."""
        results = []

        for symbol, exchanges in self._latest.items():
            rates = [r.rate for r in exchanges.values()]
            avg_rate = np.mean(rates)

            results.append(
                {
                    "symbol": symbol,
                    "avg_rate": avg_rate,
                    "avg_rate_pct": avg_rate * 100,
                    "annualized_pct": avg_rate * 3 * 365 * 100,
                    "exchanges": len(exchanges),
                }
            )

        # Sort by rate
        if direction == "positive":
            results.sort(key=lambda x: x["avg_rate"], reverse=True)
        else:
            results.sort(key=lambda x: x["avg_rate"])

        return results[:limit]

    def get_all_arbitrage_opportunities(self) -> List[FundingArbitrage]:
        """Find all current arbitrage opportunities."""
        opportunities = []

        for symbol in self._latest.keys():
            arb = self.find_arbitrage(symbol)
            if arb and arb.is_viable:
                opportunities.append(arb)

        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit_annual, reverse=True)

        return opportunities

    def get_summary(self) -> Dict:
        """Get funding rate summary."""
        total_symbols = len(self._latest)
        total_positive = sum(1 for exs in self._latest.values() for r in exs.values() if r.rate > 0)
        total_negative = sum(1 for exs in self._latest.values() for r in exs.values() if r.rate < 0)

        arb_opportunities = self.get_all_arbitrage_opportunities()

        return {
            "total_symbols": total_symbols,
            "total_observations": sum(len(exs) for exs in self._latest.values()),
            "positive_funding": total_positive,
            "negative_funding": total_negative,
            "sentiment_ratio": total_positive / (total_positive + total_negative)
            if (total_positive + total_negative) > 0
            else 0.5,
            "arbitrage_opportunities": len(arb_opportunities),
            "top_arb_annual_pct": arb_opportunities[0].expected_profit_annual * 100
            if arb_opportunities
            else 0,
        }


def create_funding_tracker(config: Optional[FundingConfig] = None) -> FundingRateTracker:
    """Factory function to create funding rate tracker."""
    return FundingRateTracker(config=config)
