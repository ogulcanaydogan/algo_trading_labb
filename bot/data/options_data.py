"""
Options Data Integration - IV, Greeks, and Options Flow.

Provides options market data for sentiment analysis
and volatility-based trading signals.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type."""

    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Single options contract."""

    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: datetime
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def moneyness(self) -> float:
        """Moneyness relative to underlying (requires spot price)."""
        return 0.0  # Set externally

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "last_price": self.last_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OptionsChain:
    """Full options chain for an underlying."""

    underlying: str
    spot_price: float
    calls: List[OptionContract]
    puts: List[OptionContract]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def all_contracts(self) -> List[OptionContract]:
        return self.calls + self.puts

    @property
    def total_call_volume(self) -> int:
        return sum(c.volume for c in self.calls)

    @property
    def total_put_volume(self) -> int:
        return sum(p.volume for p in self.puts)

    @property
    def put_call_ratio(self) -> float:
        """Volume-based put/call ratio."""
        call_vol = self.total_call_volume
        if call_vol == 0:
            return 0.0
        return self.total_put_volume / call_vol

    @property
    def put_call_oi_ratio(self) -> float:
        """Open interest-based put/call ratio."""
        call_oi = sum(c.open_interest for c in self.calls)
        if call_oi == 0:
            return 0.0
        put_oi = sum(p.open_interest for p in self.puts)
        return put_oi / call_oi

    def get_atm_iv(self) -> float:
        """Get at-the-money implied volatility."""
        atm_calls = sorted(self.calls, key=lambda c: abs(c.strike - self.spot_price))
        atm_puts = sorted(self.puts, key=lambda p: abs(p.strike - self.spot_price))

        ivs = []
        if atm_calls:
            ivs.append(atm_calls[0].implied_volatility)
        if atm_puts:
            ivs.append(atm_puts[0].implied_volatility)

        return np.mean(ivs) if ivs else 0.0

    def get_iv_skew(self) -> float:
        """Calculate IV skew (25-delta put IV - 25-delta call IV)."""
        # Find 25-delta options
        put_25d = [p for p in self.puts if -0.30 < p.delta < -0.20]
        call_25d = [c for c in self.calls if 0.20 < c.delta < 0.30]

        if not put_25d or not call_25d:
            return 0.0

        put_iv = np.mean([p.implied_volatility for p in put_25d])
        call_iv = np.mean([c.implied_volatility for c in call_25d])

        return put_iv - call_iv

    def to_dict(self) -> Dict:
        return {
            "underlying": self.underlying,
            "spot_price": self.spot_price,
            "total_call_volume": self.total_call_volume,
            "total_put_volume": self.total_put_volume,
            "put_call_ratio": round(self.put_call_ratio, 4),
            "put_call_oi_ratio": round(self.put_call_oi_ratio, 4),
            "atm_iv": round(self.get_atm_iv(), 4),
            "iv_skew": round(self.get_iv_skew(), 4),
            "num_calls": len(self.calls),
            "num_puts": len(self.puts),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OptionsFlow:
    """Large options trade (unusual activity)."""

    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: datetime
    side: str  # "buy" or "sell"
    size: int
    price: float
    premium: float  # Total premium = size * price * 100
    spot_at_trade: float
    iv_at_trade: float
    is_sweep: bool = False  # Multi-exchange sweep
    is_block: bool = False  # Block trade
    sentiment: str = "neutral"  # "bullish", "bearish", "neutral"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "option_type": self.option_type.value,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "premium": self.premium,
            "spot_at_trade": self.spot_at_trade,
            "iv_at_trade": self.iv_at_trade,
            "is_sweep": self.is_sweep,
            "is_block": self.is_block,
            "sentiment": self.sentiment,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class VolatilitySurface:
    """Implied volatility surface."""

    underlying: str
    spot_price: float
    strikes: List[float]
    expiries: List[datetime]
    iv_matrix: np.ndarray  # Shape: (len(expiries), len(strikes))
    timestamp: datetime = field(default_factory=datetime.now)

    def get_iv(self, strike: float, expiry: datetime) -> float:
        """Interpolate IV for given strike and expiry."""
        # Find nearest indices
        strike_idx = np.argmin(np.abs(np.array(self.strikes) - strike))
        expiry_diffs = [(e - expiry).total_seconds() for e in self.expiries]
        expiry_idx = np.argmin(np.abs(expiry_diffs))

        return float(self.iv_matrix[expiry_idx, strike_idx])

    def get_term_structure(self) -> Dict[str, float]:
        """Get ATM IV term structure."""
        atm_idx = np.argmin(np.abs(np.array(self.strikes) - self.spot_price))
        return {
            exp.isoformat(): float(self.iv_matrix[i, atm_idx])
            for i, exp in enumerate(self.expiries)
        }

    def to_dict(self) -> Dict:
        return {
            "underlying": self.underlying,
            "spot_price": self.spot_price,
            "strikes": self.strikes,
            "expiries": [e.isoformat() for e in self.expiries],
            "term_structure": self.get_term_structure(),
            "timestamp": self.timestamp.isoformat(),
        }


class BlackScholes:
    """Black-Scholes option pricing and Greeks."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price."""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price."""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def delta(
        S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
    ) -> float:
        """Calculate delta."""
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate gamma (same for calls and puts)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(
        S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType
    ) -> float:
        """Calculate theta (per day)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

        return (term1 + term2) / 365  # Daily theta

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate vega (per 1% IV change)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) / 100

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        """Calculate rho (per 1% rate change)."""
        if T <= 0:
            return 0.0
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    @staticmethod
    def implied_volatility(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """Calculate implied volatility using Newton-Raphson."""
        if T <= 0:
            return 0.0

        # Initial guess
        sigma = 0.3

        for _ in range(max_iterations):
            if option_type == OptionType.CALL:
                calc_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                calc_price = BlackScholes.put_price(S, K, T, r, sigma)

            diff = calc_price - price
            if abs(diff) < tolerance:
                return sigma

            vega = BlackScholes.vega(S, K, T, r, sigma) * 100
            if vega < 1e-10:
                break

            sigma = sigma - diff / vega
            sigma = max(0.01, min(5.0, sigma))  # Bound between 1% and 500%

        return sigma


@dataclass
class OptionsAnalyticsConfig:
    """Options analytics configuration."""

    risk_free_rate: float = 0.05
    min_premium_threshold: float = 50000  # Min premium for unusual activity
    iv_spike_threshold: float = 0.2  # 20% IV change
    pcr_extreme_low: float = 0.5  # Bullish extreme
    pcr_extreme_high: float = 1.5  # Bearish extreme


class OptionsAnalytics:
    """
    Options market analytics for trading signals.

    Analyzes:
    - Put/Call ratios (sentiment)
    - Implied volatility (expectations)
    - Options flow (smart money)
    - IV skew (tail risk pricing)
    """

    def __init__(self, config: Optional[OptionsAnalyticsConfig] = None):
        self.config = config or OptionsAnalyticsConfig()
        self._chains: Dict[str, OptionsChain] = {}
        self._flow_history: List[OptionsFlow] = []
        self._iv_history: Dict[str, List[Dict[str, Any]]] = {}

    def update_chain(self, chain: OptionsChain):
        """Update options chain data."""
        self._chains[chain.underlying] = chain

        # Track IV history
        if chain.underlying not in self._iv_history:
            self._iv_history[chain.underlying] = []

        self._iv_history[chain.underlying].append(
            {
                "timestamp": chain.timestamp,
                "atm_iv": chain.get_atm_iv(),
                "skew": chain.get_iv_skew(),
                "pcr": chain.put_call_ratio,
            }
        )

        # Keep last 1000 data points
        if len(self._iv_history[chain.underlying]) > 1000:
            self._iv_history[chain.underlying] = self._iv_history[chain.underlying][-1000:]

    def add_flow(self, flow: OptionsFlow):
        """Add options flow trade."""
        # Determine sentiment
        if flow.option_type == OptionType.CALL:
            flow.sentiment = "bullish" if flow.side == "buy" else "bearish"
        else:
            flow.sentiment = "bearish" if flow.side == "buy" else "bullish"

        self._flow_history.append(flow)

        # Keep last 10000 flows
        if len(self._flow_history) > 10000:
            self._flow_history = self._flow_history[-10000:]

    def get_market_sentiment(self, underlying: str) -> Dict[str, Any]:
        """
        Get overall market sentiment from options data.

        Returns sentiment signals:
        - pcr_signal: Put/call ratio interpretation
        - iv_signal: IV level interpretation
        - skew_signal: IV skew interpretation
        - flow_signal: Recent flow interpretation
        """
        if underlying not in self._chains:
            return {"error": "No options data available"}

        chain = self._chains[underlying]
        pcr = chain.put_call_ratio
        atm_iv = chain.get_atm_iv()
        skew = chain.get_iv_skew()

        # PCR signal
        if pcr < self.config.pcr_extreme_low:
            pcr_signal = {"value": pcr, "interpretation": "bullish_extreme", "score": 1.0}
        elif pcr > self.config.pcr_extreme_high:
            pcr_signal = {"value": pcr, "interpretation": "bearish_extreme", "score": -1.0}
        else:
            normalized = (pcr - 1.0) / 0.5  # Center around 1.0
            pcr_signal = {"value": pcr, "interpretation": "neutral", "score": -normalized}

        # IV signal (high IV = fear, low IV = complacency)
        iv_history = self._iv_history.get(underlying, [])
        if len(iv_history) >= 20:
            recent_ivs = [h["atm_iv"] for h in iv_history[-20:]]
            iv_percentile = sum(1 for iv in recent_ivs if iv <= atm_iv) / len(recent_ivs)

            if iv_percentile > 0.8:
                iv_signal = {
                    "value": atm_iv,
                    "percentile": iv_percentile,
                    "interpretation": "high_fear",
                }
            elif iv_percentile < 0.2:
                iv_signal = {
                    "value": atm_iv,
                    "percentile": iv_percentile,
                    "interpretation": "low_fear",
                }
            else:
                iv_signal = {
                    "value": atm_iv,
                    "percentile": iv_percentile,
                    "interpretation": "normal",
                }
        else:
            iv_signal = {"value": atm_iv, "interpretation": "insufficient_history"}

        # Skew signal (positive skew = put premium = downside fear)
        if skew > 0.05:
            skew_signal = {"value": skew, "interpretation": "downside_fear"}
        elif skew < -0.05:
            skew_signal = {"value": skew, "interpretation": "upside_speculation"}
        else:
            skew_signal = {"value": skew, "interpretation": "balanced"}

        # Flow signal (recent unusual activity)
        recent_flows = [
            f
            for f in self._flow_history
            if f.underlying == underlying and (datetime.now() - f.timestamp).total_seconds() < 3600
        ]

        bullish_premium = sum(f.premium for f in recent_flows if f.sentiment == "bullish")
        bearish_premium = sum(f.premium for f in recent_flows if f.sentiment == "bearish")

        total_premium = bullish_premium + bearish_premium
        if total_premium > 0:
            flow_ratio = (bullish_premium - bearish_premium) / total_premium
            if flow_ratio > 0.3:
                flow_signal = {
                    "bullish_premium": bullish_premium,
                    "bearish_premium": bearish_premium,
                    "interpretation": "bullish_flow",
                }
            elif flow_ratio < -0.3:
                flow_signal = {
                    "bullish_premium": bullish_premium,
                    "bearish_premium": bearish_premium,
                    "interpretation": "bearish_flow",
                }
            else:
                flow_signal = {
                    "bullish_premium": bullish_premium,
                    "bearish_premium": bearish_premium,
                    "interpretation": "mixed_flow",
                }
        else:
            flow_signal = {"interpretation": "no_unusual_activity"}

        # Overall sentiment score (-1 to 1)
        scores = []
        if "score" in pcr_signal:
            scores.append(pcr_signal["score"])
        if skew_signal["interpretation"] == "downside_fear":
            scores.append(-0.5)
        elif skew_signal["interpretation"] == "upside_speculation":
            scores.append(0.5)
        if flow_signal.get("interpretation") == "bullish_flow":
            scores.append(0.5)
        elif flow_signal.get("interpretation") == "bearish_flow":
            scores.append(-0.5)

        overall_score = np.mean(scores) if scores else 0.0

        return {
            "underlying": underlying,
            "timestamp": datetime.now().isoformat(),
            "pcr_signal": pcr_signal,
            "iv_signal": iv_signal,
            "skew_signal": skew_signal,
            "flow_signal": flow_signal,
            "overall_sentiment": {
                "score": round(overall_score, 3),
                "interpretation": "bullish"
                if overall_score > 0.2
                else "bearish"
                if overall_score < -0.2
                else "neutral",
            },
        }

    def detect_unusual_activity(
        self, underlying: Optional[str] = None, lookback_hours: int = 24
    ) -> List[OptionsFlow]:
        """Detect unusual options activity."""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        flows = [
            f
            for f in self._flow_history
            if f.timestamp >= cutoff
            and f.premium >= self.config.min_premium_threshold
            and (underlying is None or f.underlying == underlying)
        ]

        # Sort by premium (largest first)
        flows.sort(key=lambda f: f.premium, reverse=True)

        return flows[:50]  # Return top 50

    def get_iv_term_structure(self, underlying: str) -> Dict[str, float]:
        """Get IV term structure from chain."""
        if underlying not in self._chains:
            return {}

        chain = self._chains[underlying]

        # Group by expiry
        expiry_ivs: Dict[datetime, List[float]] = {}
        for contract in chain.all_contracts:
            # Only use near-ATM options
            if abs(contract.strike - chain.spot_price) / chain.spot_price < 0.1:
                if contract.expiry not in expiry_ivs:
                    expiry_ivs[contract.expiry] = []
                expiry_ivs[contract.expiry].append(contract.implied_volatility)

        return {exp.isoformat(): round(np.mean(ivs), 4) for exp, ivs in sorted(expiry_ivs.items())}

    def calculate_greeks(
        self,
        spot: float,
        strike: float,
        expiry: datetime,
        option_type: OptionType,
        iv: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calculate option Greeks."""
        T = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
        r = self.config.risk_free_rate

        # Get IV from price if not provided
        if iv is None and price is not None:
            iv = BlackScholes.implied_volatility(price, spot, strike, T, r, option_type)
        elif iv is None:
            iv = 0.3  # Default

        return {
            "delta": round(BlackScholes.delta(spot, strike, T, r, iv, option_type), 4),
            "gamma": round(BlackScholes.gamma(spot, strike, T, r, iv), 6),
            "theta": round(BlackScholes.theta(spot, strike, T, r, iv, option_type), 4),
            "vega": round(BlackScholes.vega(spot, strike, T, r, iv), 4),
            "rho": round(BlackScholes.rho(spot, strike, T, r, iv, option_type), 4),
            "implied_volatility": round(iv, 4),
        }

    def get_max_pain(self, underlying: str) -> Dict[str, Any]:
        """
        Calculate max pain strike price.

        Max pain is the strike price where option holders
        would experience maximum losses at expiration.
        """
        if underlying not in self._chains:
            return {"error": "No options data"}

        chain = self._chains[underlying]

        # Get all strikes
        strikes = sorted(set(c.strike for c in chain.all_contracts))

        min_pain = float("inf")
        max_pain_strike = chain.spot_price

        for test_strike in strikes:
            total_pain = 0

            # Call holders lose if price below strike
            for call in chain.calls:
                if test_strike < call.strike:
                    # Calls expire worthless
                    total_pain += call.open_interest * call.last_price * 100

            # Put holders lose if price above strike
            for put in chain.puts:
                if test_strike > put.strike:
                    # Puts expire worthless
                    total_pain += put.open_interest * put.last_price * 100

            if total_pain < min_pain:
                min_pain = total_pain
                max_pain_strike = test_strike

        return {
            "underlying": underlying,
            "max_pain_strike": max_pain_strike,
            "current_price": chain.spot_price,
            "distance_pct": round((max_pain_strike - chain.spot_price) / chain.spot_price * 100, 2),
        }

    def get_gamma_exposure(self, underlying: str) -> Dict[str, Any]:
        """
        Calculate dealer gamma exposure (GEX).

        Positive GEX = Dealers are long gamma (price stabilizing)
        Negative GEX = Dealers are short gamma (price destabilizing)
        """
        if underlying not in self._chains:
            return {"error": "No options data"}

        chain = self._chains[underlying]
        spot = chain.spot_price

        total_gex = 0
        gex_by_strike: Dict[float, float] = {}

        for contract in chain.all_contracts:
            # Assume dealers are short options (retail is long)
            # Calls: dealers short gamma
            # Puts: dealers long gamma (negative delta hedge)

            gamma = contract.gamma
            oi = contract.open_interest

            if contract.option_type == OptionType.CALL:
                gex = -gamma * oi * 100 * spot  # Dealers short
            else:
                gex = gamma * oi * 100 * spot  # Dealers long puts (short gamma)

            total_gex += gex

            if contract.strike not in gex_by_strike:
                gex_by_strike[contract.strike] = 0
            gex_by_strike[contract.strike] += gex

        # Find flip point (where GEX changes sign)
        sorted_strikes = sorted(gex_by_strike.keys())
        flip_strike = None
        cumulative_gex = 0

        for strike in sorted_strikes:
            prev_cumulative = cumulative_gex
            cumulative_gex += gex_by_strike[strike]
            if prev_cumulative * cumulative_gex < 0:
                flip_strike = strike
                break

        return {
            "underlying": underlying,
            "total_gex": round(total_gex / 1e6, 2),  # In millions
            "gex_interpretation": "stabilizing" if total_gex > 0 else "destabilizing",
            "flip_strike": flip_strike,
            "current_price": spot,
        }


def create_options_analytics(config: Optional[OptionsAnalyticsConfig] = None) -> OptionsAnalytics:
    """Factory function to create options analytics."""
    return OptionsAnalytics(config=config)
