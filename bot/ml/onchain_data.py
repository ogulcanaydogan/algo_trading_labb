"""
On-Chain Data Integration.

Fetches and processes on-chain metrics for crypto trading:
- Exchange flows (deposits/withdrawals)
- Whale movements
- Funding rates
- Open interest
- Network activity

APIs used:
- Coinglass (funding rates, open interest)
- Glassnode (on-chain metrics) - requires API key
- CryptoQuant (exchange flows) - requires API key
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """On-chain metrics for a crypto asset."""
    symbol: str
    timestamp: datetime

    # Funding rates
    funding_rate: float = 0.0
    predicted_funding: float = 0.0
    funding_sentiment: str = "neutral"  # bullish/bearish/neutral

    # Open Interest
    open_interest: float = 0.0
    oi_change_24h: float = 0.0
    oi_weighted_funding: float = 0.0

    # Exchange flows
    exchange_netflow: float = 0.0  # Positive = inflow (bearish), Negative = outflow (bullish)
    exchange_reserve: float = 0.0
    reserve_change_24h: float = 0.0

    # Whale activity
    whale_transactions: int = 0
    large_tx_volume: float = 0.0

    # Network
    active_addresses: int = 0
    transaction_count: int = 0

    # Derived signals
    accumulation_score: float = 0.5  # 0=distribution, 1=accumulation
    smart_money_signal: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'funding_rate': self.funding_rate,
            'predicted_funding': self.predicted_funding,
            'funding_sentiment': self.funding_sentiment,
            'open_interest': self.open_interest,
            'oi_change_24h': self.oi_change_24h,
            'exchange_netflow': self.exchange_netflow,
            'whale_transactions': self.whale_transactions,
            'accumulation_score': self.accumulation_score,
            'smart_money_signal': self.smart_money_signal,
        }


class OnChainDataFetcher:
    """
    Fetches on-chain data from multiple sources.

    Free sources:
    - Coinglass API (funding rates, OI)
    - Alternative.me (Fear & Greed)

    Paid sources (require API keys):
    - Glassnode
    - CryptoQuant
    - IntoTheBlock
    """

    def __init__(
        self,
        glassnode_api_key: Optional[str] = None,
        cryptoquant_api_key: Optional[str] = None,
    ):
        self.glassnode_api_key = glassnode_api_key
        self.cryptoquant_api_key = cryptoquant_api_key

        # Cache
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes

        # Rate limiting
        self._last_request: Dict[str, float] = {}
        self._min_interval = 1.0  # seconds between requests

    async def _rate_limit(self, source: str):
        """Simple rate limiting."""
        now = datetime.now().timestamp()
        if source in self._last_request:
            elapsed = now - self._last_request[source]
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
        self._last_request[source] = datetime.now().timestamp()

    async def _fetch_json(self, url: str, headers: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch JSON from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"HTTP {response.status} from {url}")
                        return None
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    async def get_funding_rates(self, symbol: str = "BTC") -> Dict[str, float]:
        """Get funding rates from Coinglass."""
        await self._rate_limit('coinglass')

        # Map symbol
        symbol_map = {
            'BTC': 'BTC', 'ETH': 'ETH', 'SOL': 'SOL',
            'DOGE': 'DOGE', 'XRP': 'XRP', 'AVAX': 'AVAX'
        }
        cg_symbol = symbol_map.get(symbol.split('/')[0].upper(), 'BTC')

        url = f"https://open-api.coinglass.com/public/v2/funding?symbol={cg_symbol}"

        data = await self._fetch_json(url)

        if data and data.get('success') and data.get('data'):
            rates = data['data']
            # Average across exchanges
            avg_rate = np.mean([r.get('rate', 0) for r in rates if r.get('rate')])
            predicted = np.mean([r.get('predictedRate', 0) for r in rates if r.get('predictedRate')])

            return {
                'funding_rate': avg_rate,
                'predicted_funding': predicted,
                'exchanges': len(rates)
            }

        return {'funding_rate': 0.0, 'predicted_funding': 0.0, 'exchanges': 0}

    async def get_open_interest(self, symbol: str = "BTC") -> Dict[str, float]:
        """Get open interest data."""
        await self._rate_limit('coinglass_oi')

        symbol_map = {'BTC': 'BTC', 'ETH': 'ETH', 'SOL': 'SOL'}
        cg_symbol = symbol_map.get(symbol.split('/')[0].upper(), 'BTC')

        url = f"https://open-api.coinglass.com/public/v2/open_interest?symbol={cg_symbol}"

        data = await self._fetch_json(url)

        if data and data.get('success') and data.get('data'):
            oi_data = data['data']
            total_oi = sum(item.get('openInterest', 0) for item in oi_data)
            return {
                'open_interest': total_oi,
                'exchanges': len(oi_data)
            }

        return {'open_interest': 0.0, 'exchanges': 0}

    async def get_exchange_flows(self, symbol: str = "BTC") -> Dict[str, float]:
        """
        Get exchange flow data.

        Note: This is a simplified version. Real implementation would
        use CryptoQuant or Glassnode APIs which require paid access.
        """
        # Simulated/estimated based on public data
        # In production, integrate with CryptoQuant/Glassnode

        if self.cryptoquant_api_key:
            # Real CryptoQuant integration would go here
            pass

        # Return neutral values if no API key
        return {
            'exchange_netflow': 0.0,
            'exchange_reserve': 0.0,
            'reserve_change_24h': 0.0
        }

    async def get_whale_activity(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get whale transaction data.

        Real implementation would use:
        - Whale Alert API
        - Glassnode whale metrics
        - On-chain transaction monitoring
        """
        # Placeholder - in production, integrate with whale tracking APIs
        return {
            'whale_transactions': 0,
            'large_tx_volume': 0.0,
            'whale_accumulating': None
        }

    def _calculate_accumulation_score(self, metrics: OnChainMetrics) -> float:
        """
        Calculate accumulation/distribution score (0-1).

        Higher = more accumulation (bullish)
        Lower = more distribution (bearish)
        """
        score = 0.5  # Neutral baseline

        # Funding rate impact
        # Negative funding = longs paying shorts = bullish for spot
        if metrics.funding_rate < -0.01:
            score += 0.15
        elif metrics.funding_rate > 0.03:
            score -= 0.15

        # Exchange netflow impact
        # Negative netflow = coins leaving exchanges = bullish
        if metrics.exchange_netflow < 0:
            score += 0.2
        elif metrics.exchange_netflow > 0:
            score -= 0.2

        # OI change impact
        # Rising OI with price = bullish
        if metrics.oi_change_24h > 5:
            score += 0.1
        elif metrics.oi_change_24h < -5:
            score -= 0.1

        return np.clip(score, 0, 1)

    def _determine_smart_money_signal(self, metrics: OnChainMetrics) -> str:
        """Determine smart money signal based on on-chain data."""
        bullish_signals = 0
        bearish_signals = 0

        # Funding rate
        if metrics.funding_rate < -0.01:
            bullish_signals += 1
        elif metrics.funding_rate > 0.02:
            bearish_signals += 1

        # Exchange flows
        if metrics.exchange_netflow < 0:
            bullish_signals += 1
        elif metrics.exchange_netflow > 0:
            bearish_signals += 1

        # Accumulation score
        if metrics.accumulation_score > 0.6:
            bullish_signals += 1
        elif metrics.accumulation_score < 0.4:
            bearish_signals += 1

        if bullish_signals >= 2:
            return "bullish"
        elif bearish_signals >= 2:
            return "bearish"
        return "neutral"

    async def get_metrics(self, symbol: str) -> OnChainMetrics:
        """Get comprehensive on-chain metrics for a symbol."""
        base_symbol = symbol.split('/')[0].upper()

        # Check cache
        cache_key = f"{base_symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch all data concurrently
        funding_task = self.get_funding_rates(base_symbol)
        oi_task = self.get_open_interest(base_symbol)
        flow_task = self.get_exchange_flows(base_symbol)
        whale_task = self.get_whale_activity(base_symbol)

        funding, oi, flows, whales = await asyncio.gather(
            funding_task, oi_task, flow_task, whale_task
        )

        metrics = OnChainMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            funding_rate=funding.get('funding_rate', 0),
            predicted_funding=funding.get('predicted_funding', 0),
            open_interest=oi.get('open_interest', 0),
            exchange_netflow=flows.get('exchange_netflow', 0),
            exchange_reserve=flows.get('exchange_reserve', 0),
            whale_transactions=whales.get('whale_transactions', 0),
            large_tx_volume=whales.get('large_tx_volume', 0),
        )

        # Determine funding sentiment
        if metrics.funding_rate > 0.01:
            metrics.funding_sentiment = "bearish"  # Longs overleveraged
        elif metrics.funding_rate < -0.01:
            metrics.funding_sentiment = "bullish"  # Shorts overleveraged
        else:
            metrics.funding_sentiment = "neutral"

        # Calculate derived metrics
        metrics.accumulation_score = self._calculate_accumulation_score(metrics)
        metrics.smart_money_signal = self._determine_smart_money_signal(metrics)

        # Cache result
        self._cache[cache_key] = metrics

        return metrics

    async def get_features_for_ml(self, symbol: str) -> Dict[str, float]:
        """Get on-chain features formatted for ML models."""
        metrics = await self.get_metrics(symbol)

        return {
            'onchain_funding_rate': metrics.funding_rate,
            'onchain_predicted_funding': metrics.predicted_funding,
            'onchain_funding_bullish': 1.0 if metrics.funding_sentiment == 'bullish' else 0.0,
            'onchain_funding_bearish': 1.0 if metrics.funding_sentiment == 'bearish' else 0.0,
            'onchain_accumulation_score': metrics.accumulation_score,
            'onchain_smart_money_bullish': 1.0 if metrics.smart_money_signal == 'bullish' else 0.0,
            'onchain_smart_money_bearish': 1.0 if metrics.smart_money_signal == 'bearish' else 0.0,
        }


# Global instance
_onchain_fetcher: Optional[OnChainDataFetcher] = None


def get_onchain_fetcher() -> OnChainDataFetcher:
    """Get or create on-chain data fetcher."""
    global _onchain_fetcher
    if _onchain_fetcher is None:
        _onchain_fetcher = OnChainDataFetcher()
    return _onchain_fetcher


async def get_onchain_signal(symbol: str) -> Dict[str, Any]:
    """Convenience function to get on-chain signal."""
    fetcher = get_onchain_fetcher()
    metrics = await fetcher.get_metrics(symbol)
    return metrics.to_dict()
