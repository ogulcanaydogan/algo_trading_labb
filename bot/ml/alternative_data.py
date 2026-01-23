"""
Alternative Data Sources for Enhanced ML Models

Fetches and processes:
- Social sentiment (Twitter, Reddit)
- On-chain metrics (for crypto)
- Economic indicators
- Options flow data
- Institutional holdings
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentDataFetcher:
    """Fetches sentiment data from various sources."""

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

    async def get_crypto_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get aggregated crypto sentiment.

        Sources:
        - Fear & Greed Index
        - Social mentions
        - News sentiment
        """
        cache_key = f"crypto_sentiment_{symbol}"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if (datetime.now() - timestamp).seconds < self._cache_ttl:
                return cached

        result = {
            "fear_greed": await self._get_fear_greed_index(),
            "social_sentiment": await self._get_social_sentiment(symbol),
            "news_sentiment": await self._get_news_sentiment(symbol),
            "whale_activity": await self._get_whale_activity(symbol),
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate composite score
        scores = []
        if result["fear_greed"]:
            scores.append((result["fear_greed"] - 50) / 50)
        if result["social_sentiment"]:
            scores.append(result["social_sentiment"])
        if result["news_sentiment"]:
            scores.append(result["news_sentiment"])

        result["composite_score"] = np.mean(scores) if scores else 0

        self._cache[cache_key] = (result, datetime.now())
        return result

    async def _get_fear_greed_index(self) -> Optional[float]:
        """Fetch Fear & Greed Index from Alternative.me."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data["data"][0]["value"])
        except Exception as e:
            logger.warning(f"Failed to fetch Fear & Greed: {e}")
        return None

    async def _get_social_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get social media sentiment.

        Returns normalized score: -1 (very bearish) to 1 (very bullish)
        """
        # Placeholder - would integrate with Twitter/Reddit APIs
        # Returns random for now - replace with actual API calls
        try:
            import aiohttp

            # LunarCrush API (requires API key)
            # CryptoCompare social data
            # Santiment API
            pass
        except Exception as e:
            logger.debug(f"Social sentiment fetch failed: {e}")
        return None

    async def _get_news_sentiment(self, symbol: str) -> Optional[float]:
        """Get news sentiment score."""
        try:
            import aiohttp

            # CryptoPanic API
            base_symbol = symbol.split("/")[0].upper()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://cryptopanic.com/api/v1/posts/?auth_token=&currencies={base_symbol}&kind=news",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "results" in data:
                            sentiments = []
                            for post in data["results"][:20]:
                                votes = post.get("votes", {})
                                positive = votes.get("positive", 0)
                                negative = votes.get("negative", 0)
                                if positive + negative > 0:
                                    sentiments.append((positive - negative) / (positive + negative))
                            return np.mean(sentiments) if sentiments else None
        except Exception as e:
            logger.debug(f"News sentiment fetch failed: {e}")
        return None

    async def _get_whale_activity(self, symbol: str) -> Optional[Dict]:
        """Get whale transaction activity."""
        # Would integrate with Whale Alert API or on-chain analysis
        return None


class OnChainDataFetcher:
    """Fetches on-chain metrics for crypto assets."""

    def __init__(self):
        self._cache = {}

    async def get_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get on-chain metrics.

        Metrics:
        - Active addresses
        - Transaction volume
        - Exchange inflows/outflows
        - MVRV ratio
        - NVT ratio
        - Hash rate (for PoW coins)
        """
        base_symbol = symbol.split("/")[0].upper()

        metrics = {
            "symbol": base_symbol,
            "timestamp": datetime.now().isoformat(),
            "active_addresses": await self._get_active_addresses(base_symbol),
            "exchange_flow": await self._get_exchange_flow(base_symbol),
            "mvrv": await self._get_mvrv(base_symbol),
            "nvt": await self._get_nvt(base_symbol),
            "supply_metrics": await self._get_supply_metrics(base_symbol),
        }

        return metrics

    async def _get_active_addresses(self, symbol: str) -> Optional[Dict]:
        """Get active addresses trend."""
        try:
            import aiohttp

            # Glassnode API (paid)
            # IntoTheBlock API
            # Messari API
            pass
        except Exception as e:
            logger.debug(f"Active addresses fetch failed: {e}")
        return None

    async def _get_exchange_flow(self, symbol: str) -> Optional[Dict]:
        """
        Get exchange inflow/outflow data.

        - Inflow: coins moving to exchanges (potential sell pressure)
        - Outflow: coins leaving exchanges (potential accumulation)
        """
        try:
            import aiohttp

            # CryptoQuant API
            # Glassnode API
            pass
        except Exception as e:
            logger.debug(f"Exchange flow fetch failed: {e}")
        return None

    async def _get_mvrv(self, symbol: str) -> Optional[float]:
        """
        Get Market Value to Realized Value ratio.

        MVRV > 3.5: Potentially overvalued
        MVRV < 1: Potentially undervalued
        """
        return None

    async def _get_nvt(self, symbol: str) -> Optional[float]:
        """
        Get Network Value to Transactions ratio.

        High NVT: Network overvalued relative to utility
        Low NVT: Network undervalued relative to utility
        """
        return None

    async def _get_supply_metrics(self, symbol: str) -> Optional[Dict]:
        """Get supply distribution metrics."""
        return None


class EconomicDataFetcher:
    """Fetches macroeconomic indicators."""

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour

    async def get_economic_context(self) -> Dict[str, Any]:
        """
        Get current economic context.

        Indicators:
        - VIX (fear index)
        - DXY (dollar strength)
        - 10Y Treasury yield
        - Fed funds rate
        - Inflation expectations
        """
        cache_key = "economic_context"
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if (datetime.now() - timestamp).seconds < self._cache_ttl:
                return cached

        result = {
            "vix": await self._get_vix(),
            "dxy": await self._get_dxy(),
            "treasury_10y": await self._get_treasury_yield(),
            "risk_sentiment": None,
            "timestamp": datetime.now().isoformat(),
        }

        # Calculate risk sentiment
        if result["vix"] is not None:
            if result["vix"] < 15:
                result["risk_sentiment"] = "risk_on"
            elif result["vix"] > 25:
                result["risk_sentiment"] = "risk_off"
            else:
                result["risk_sentiment"] = "neutral"

        self._cache[cache_key] = (result, datetime.now())
        return result

    async def _get_vix(self) -> Optional[float]:
        """Get VIX volatility index."""
        try:
            import yfinance as yf

            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")
        return None

    async def _get_dxy(self) -> Optional[float]:
        """Get US Dollar Index."""
        try:
            import yfinance as yf

            dxy = yf.Ticker("DX-Y.NYB")
            hist = dxy.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.debug(f"DXY fetch failed: {e}")
        return None

    async def _get_treasury_yield(self) -> Optional[float]:
        """Get 10-year Treasury yield."""
        try:
            import yfinance as yf

            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.debug(f"Treasury yield fetch failed: {e}")
        return None


class AlternativeDataAggregator:
    """
    Aggregates all alternative data sources into
    features for ML models.
    """

    def __init__(self):
        self.sentiment_fetcher = SentimentDataFetcher()
        self.onchain_fetcher = OnChainDataFetcher()
        self.economic_fetcher = EconomicDataFetcher()

    async def get_features(self, symbol: str, asset_type: str = "crypto") -> Dict[str, float]:
        """
        Get all alternative data features for ML model.

        Returns normalized feature dict ready for model input.
        """
        features = {}

        # Get economic context (all assets)
        economic = await self.economic_fetcher.get_economic_context()
        if economic.get("vix"):
            features["alt_vix"] = economic["vix"]
            features["alt_vix_high"] = 1.0 if economic["vix"] > 25 else 0.0
            features["alt_vix_low"] = 1.0 if economic["vix"] < 15 else 0.0

        if economic.get("dxy"):
            features["alt_dxy"] = economic["dxy"]

        if economic.get("treasury_10y"):
            features["alt_treasury_10y"] = economic["treasury_10y"]

        # Get crypto-specific data
        if asset_type == "crypto":
            sentiment = await self.sentiment_fetcher.get_crypto_sentiment(symbol)
            if sentiment.get("fear_greed") is not None:
                features["alt_fear_greed"] = sentiment["fear_greed"]
                features["alt_extreme_fear"] = 1.0 if sentiment["fear_greed"] < 25 else 0.0
                features["alt_extreme_greed"] = 1.0 if sentiment["fear_greed"] > 75 else 0.0

            if sentiment.get("composite_score") is not None:
                features["alt_sentiment_composite"] = sentiment["composite_score"]

            onchain = await self.onchain_fetcher.get_metrics(symbol)
            if onchain.get("mvrv"):
                features["alt_mvrv"] = onchain["mvrv"]
                features["alt_mvrv_overvalued"] = 1.0 if onchain["mvrv"] > 3.5 else 0.0
                features["alt_mvrv_undervalued"] = 1.0 if onchain["mvrv"] < 1 else 0.0

        return features

    async def get_historical_features(
        self, symbol: str, start_date: datetime, end_date: datetime, asset_type: str = "crypto"
    ) -> pd.DataFrame:
        """
        Get historical alternative data features.

        Note: Most alternative data sources have limited historical data
        available through free APIs. This would require paid data providers
        for production use.
        """
        # Placeholder - would need historical API access
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Create synthetic features based on price patterns
        # In production, replace with actual historical data
        df = pd.DataFrame(index=dates)
        df["alt_fear_greed"] = 50  # Neutral baseline
        df["alt_vix"] = 20  # Normal VIX
        df["alt_sentiment_composite"] = 0  # Neutral

        return df


# Global singleton
_alternative_data_aggregator: Optional[AlternativeDataAggregator] = None


def get_alternative_data_aggregator() -> AlternativeDataAggregator:
    """Get or create alternative data aggregator."""
    global _alternative_data_aggregator
    if _alternative_data_aggregator is None:
        _alternative_data_aggregator = AlternativeDataAggregator()
    return _alternative_data_aggregator
