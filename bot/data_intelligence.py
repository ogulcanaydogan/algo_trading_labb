"""
Data Intelligence Module - Multi-Source Market Intelligence.

Aggregates data from multiple sources to provide comprehensive
market intelligence for trading decisions.

Sources:
1. News & Headlines (via news APIs)
2. Social Sentiment (Twitter, Reddit)
3. Fear & Greed Index
4. On-Chain Data (for crypto)
5. Economic Indicators
6. Cross-Asset Correlations
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """A news item with sentiment."""
    title: str
    source: str
    timestamp: datetime
    sentiment: float  # -1 to 1
    relevance: float  # 0 to 1
    symbols: List[str]


@dataclass
class SentimentData:
    """Aggregated sentiment data."""
    news_sentiment: float  # -1 to 1
    social_sentiment: float  # -1 to 1
    fear_greed_index: float  # 0-100
    bullish_pct: float  # 0-100
    news_volume: int
    sentiment_trend: str  # improving, worsening, stable
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OnChainMetrics:
    """On-chain metrics for crypto."""
    exchange_inflow: float
    exchange_outflow: float
    net_flow: float  # negative = outflow (bullish)
    whale_transactions: int
    active_addresses: int
    nvt_ratio: float  # Network Value to Transactions
    mvrv_ratio: float  # Market Value to Realized Value
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def flow_signal(self) -> str:
        """Interpret exchange flow."""
        if self.net_flow < -1000000:  # Large outflow
            return "bullish"
        elif self.net_flow > 1000000:  # Large inflow
            return "bearish"
        return "neutral"


@dataclass
class EconomicContext:
    """Economic indicators and events."""
    fed_rate: float
    fed_stance: str  # hawkish, dovish, neutral
    cpi_yoy: float  # Inflation
    unemployment: float
    gdp_growth: float
    dxy_index: float  # Dollar strength
    vix_index: float  # Volatility index
    next_fomc: Optional[datetime] = None
    upcoming_events: List[str] = field(default_factory=list)


class DataCache:
    """Simple time-based cache for API responses."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        """Cache value with current timestamp."""
        self.cache[key] = (value, time.time())


class NewsAggregator:
    """
    Aggregates news from multiple sources.

    Sources:
    - CryptoPanic (crypto news)
    - NewsAPI (general financial news)
    - RSS feeds
    """

    def __init__(self):
        self.cache = DataCache(ttl_seconds=300)  # 5 min cache
        self.cryptopanic_token = os.getenv('CRYPTOPANIC_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')

    async def get_news(self, symbol: str, limit: int = 20) -> List[NewsItem]:
        """Get recent news for symbol."""
        cache_key = f"news_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        news_items = []

        # Try CryptoPanic for crypto
        if 'USD' in symbol or 'BTC' in symbol or 'ETH' in symbol:
            crypto_news = await self._fetch_cryptopanic(symbol, limit)
            news_items.extend(crypto_news)

        self.cache.set(cache_key, news_items)
        return news_items

    async def _fetch_cryptopanic(self, symbol: str, limit: int) -> List[NewsItem]:
        """Fetch news from CryptoPanic API."""
        items = []

        # Extract base currency
        base = symbol.split('/')[0].upper()
        if base in ['BTC', 'ETH', 'SOL', 'AVAX', 'XRP', 'LINK', 'DOGE']:
            currency = base
        else:
            return items

        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_token}&currencies={currency}&public=true"
            if not self.cryptopanic_token:
                # Use public feed (limited)
                url = f"https://cryptopanic.com/api/v1/posts/?currencies={currency}&public=true"

            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read())

            for post in data.get('results', [])[:limit]:
                # Simple sentiment from votes
                votes = post.get('votes', {})
                positive = votes.get('positive', 0)
                negative = votes.get('negative', 0)
                total = positive + negative + 1
                sentiment = (positive - negative) / total

                items.append(NewsItem(
                    title=post.get('title', ''),
                    source=post.get('source', {}).get('title', 'Unknown'),
                    timestamp=datetime.fromisoformat(post.get('published_at', '').replace('Z', '+00:00')),
                    sentiment=sentiment,
                    relevance=0.8 if currency.lower() in post.get('title', '').lower() else 0.5,
                    symbols=[currency]
                ))

        except Exception as e:
            logger.debug(f"CryptoPanic fetch failed: {e}")

        return items

    def analyze_sentiment(self, news_items: List[NewsItem]) -> float:
        """Calculate overall sentiment from news items."""
        if not news_items:
            return 0.0

        # Weight by relevance and recency
        total_weight = 0
        weighted_sentiment = 0

        now = datetime.now()
        for item in news_items:
            # Recency weight (decay over 24 hours)
            age_hours = (now - item.timestamp.replace(tzinfo=None)).total_seconds() / 3600
            recency_weight = max(0.1, 1 - age_hours / 24)

            weight = item.relevance * recency_weight
            weighted_sentiment += item.sentiment * weight
            total_weight += weight

        return weighted_sentiment / total_weight if total_weight > 0 else 0.0


class SentimentTracker:
    """
    Tracks sentiment from multiple sources.

    - Fear & Greed Index (Alternative.me)
    - Social sentiment aggregation
    - News sentiment
    """

    def __init__(self):
        self.cache = DataCache(ttl_seconds=600)  # 10 min cache
        self.news_aggregator = NewsAggregator()

    async def get_sentiment(self, symbol: str) -> SentimentData:
        """Get comprehensive sentiment data."""
        cache_key = f"sentiment_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Fetch all sentiment sources
        fear_greed = await self._fetch_fear_greed()
        news = await self.news_aggregator.get_news(symbol)
        news_sentiment = self.news_aggregator.analyze_sentiment(news)

        # Calculate trend (compare to previous)
        prev_key = f"prev_sentiment_{symbol}"
        prev = self.cache.get(prev_key)
        if prev and isinstance(prev, SentimentData):
            if news_sentiment > prev.news_sentiment + 0.1:
                trend = "improving"
            elif news_sentiment < prev.news_sentiment - 0.1:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            trend = "stable"

        sentiment = SentimentData(
            news_sentiment=news_sentiment,
            social_sentiment=0,  # Could add Twitter/Reddit API
            fear_greed_index=fear_greed,
            bullish_pct=50 + news_sentiment * 25,  # Simple conversion
            news_volume=len(news),
            sentiment_trend=trend
        )

        self.cache.set(cache_key, sentiment)
        self.cache.set(prev_key, sentiment)
        return sentiment

    async def _fetch_fear_greed(self) -> float:
        """Fetch Fear & Greed Index from Alternative.me."""
        cached = self.cache.get("fear_greed")
        if cached:
            return cached

        try:
            url = "https://api.alternative.me/fng/"
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=10) as response:
                data = json.loads(response.read())

            value = float(data['data'][0]['value'])
            self.cache.set("fear_greed", value)
            return value

        except Exception as e:
            logger.debug(f"Fear & Greed fetch failed: {e}")
            return 50.0  # Neutral default


class OnChainAnalyzer:
    """
    Analyzes on-chain data for crypto assets.

    Could integrate with:
    - Glassnode
    - CryptoQuant
    - IntoTheBlock
    """

    def __init__(self):
        self.cache = DataCache(ttl_seconds=900)  # 15 min cache

    async def get_metrics(self, symbol: str) -> OnChainMetrics:
        """Get on-chain metrics for crypto."""
        cache_key = f"onchain_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Default metrics (would be replaced with real API calls)
        metrics = OnChainMetrics(
            exchange_inflow=0,
            exchange_outflow=0,
            net_flow=0,
            whale_transactions=0,
            active_addresses=0,
            nvt_ratio=0,
            mvrv_ratio=1.0
        )

        # Try to fetch real data if API keys available
        base = symbol.split('/')[0].upper()
        if base in ['BTC', 'ETH']:
            # Could add Glassnode/CryptoQuant API calls here
            pass

        self.cache.set(cache_key, metrics)
        return metrics


class EconomicDataProvider:
    """
    Provides economic indicators and context.

    Sources:
    - FRED (Federal Reserve Economic Data)
    - Trading Economics
    - Yahoo Finance (for VIX, DXY)
    """

    def __init__(self):
        self.cache = DataCache(ttl_seconds=3600)  # 1 hour cache

    async def get_context(self) -> EconomicContext:
        """Get current economic context."""
        cached = self.cache.get("economic")
        if cached:
            return cached

        # Fetch VIX
        vix = await self._fetch_vix()

        context = EconomicContext(
            fed_rate=5.25,  # Would fetch from API
            fed_stance="neutral",
            cpi_yoy=3.2,
            unemployment=3.7,
            gdp_growth=2.5,
            dxy_index=104.0,
            vix_index=vix,
            upcoming_events=[]
        )

        self.cache.set("economic", context)
        return context

    async def _fetch_vix(self) -> float:
        """Fetch VIX index."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")
        return 20.0  # Default


class CrossAssetAnalyzer:
    """
    Analyzes correlations and relationships across assets.
    """

    def __init__(self):
        self.cache = DataCache(ttl_seconds=3600)

    async def get_correlations(self, symbol: str) -> Dict[str, float]:
        """Get correlation of symbol with major assets."""
        cache_key = f"correlations_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        correlations = {
            'BTC': 0.0,
            'SP500': 0.0,
            'DXY': 0.0,
            'GOLD': 0.0
        }

        try:
            import yfinance as yf

            # Fetch price history
            base = symbol.split('/')[0]
            if base in ['BTC', 'ETH', 'SOL']:
                ticker = f"{base}-USD"
            else:
                ticker = base

            # Get historical data
            end = datetime.now()
            start = end - timedelta(days=90)

            target = yf.download(ticker, start=start, end=end, progress=False)
            btc = yf.download("BTC-USD", start=start, end=end, progress=False)
            spy = yf.download("SPY", start=start, end=end, progress=False)

            if not target.empty and not btc.empty:
                # Calculate returns
                target_ret = target['Close'].pct_change().dropna()
                btc_ret = btc['Close'].pct_change().dropna()
                spy_ret = spy['Close'].pct_change().dropna()

                # Align dates
                common_idx = target_ret.index.intersection(btc_ret.index)
                if len(common_idx) > 20:
                    correlations['BTC'] = float(target_ret.loc[common_idx].corr(btc_ret.loc[common_idx]))

                common_idx = target_ret.index.intersection(spy_ret.index)
                if len(common_idx) > 20:
                    correlations['SP500'] = float(target_ret.loc[common_idx].corr(spy_ret.loc[common_idx]))

        except Exception as e:
            logger.debug(f"Correlation calculation failed: {e}")

        self.cache.set(cache_key, correlations)
        return correlations


class DataIntelligence:
    """
    Main orchestrator for multi-source data intelligence.

    Combines all data sources to provide comprehensive
    market intelligence for trading decisions.
    """

    def __init__(self):
        self.sentiment_tracker = SentimentTracker()
        self.onchain_analyzer = OnChainAnalyzer()
        self.economic_provider = EconomicDataProvider()
        self.cross_asset_analyzer = CrossAssetAnalyzer()

    async def get_full_context(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market context for a symbol.

        Returns all available intelligence data.
        """
        # Fetch all data concurrently
        sentiment, onchain, economic, correlations = await asyncio.gather(
            self.sentiment_tracker.get_sentiment(symbol),
            self.onchain_analyzer.get_metrics(symbol),
            self.economic_provider.get_context(),
            self.cross_asset_analyzer.get_correlations(symbol),
            return_exceptions=True
        )

        # Handle any errors
        if isinstance(sentiment, Exception):
            logger.warning(f"Sentiment fetch failed: {sentiment}")
            sentiment = SentimentData(0, 0, 50, 50, 0, "stable")

        if isinstance(onchain, Exception):
            logger.warning(f"Onchain fetch failed: {onchain}")
            onchain = OnChainMetrics(0, 0, 0, 0, 0, 0, 1)

        if isinstance(economic, Exception):
            logger.warning(f"Economic fetch failed: {economic}")
            economic = EconomicContext(5.25, "neutral", 3.2, 3.7, 2.5, 104, 20)

        if isinstance(correlations, Exception):
            logger.warning(f"Correlations fetch failed: {correlations}")
            correlations = {'BTC': 0, 'SP500': 0, 'DXY': 0, 'GOLD': 0}

        # Build comprehensive context
        context = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),

            'sentiment': {
                'news': sentiment.news_sentiment,
                'social': sentiment.social_sentiment,
                'fear_greed': sentiment.fear_greed_index,
                'bullish_pct': sentiment.bullish_pct,
                'trend': sentiment.sentiment_trend,
                'news_count': sentiment.news_volume
            },

            'onchain': {
                'exchange_flow': onchain.flow_signal,
                'net_flow': onchain.net_flow,
                'whale_activity': 'high' if onchain.whale_transactions > 100 else 'normal',
                'nvt_ratio': onchain.nvt_ratio,
                'mvrv_ratio': onchain.mvrv_ratio
            },

            'economic': {
                'fed_rate': economic.fed_rate,
                'fed_stance': economic.fed_stance,
                'vix': economic.vix_index,
                'dxy': economic.dxy_index,
                'inflation': economic.cpi_yoy
            },

            'correlations': correlations,

            'signals': self._generate_signals(sentiment, onchain, economic, correlations)
        }

        return context

    def _generate_signals(
        self,
        sentiment: SentimentData,
        onchain: OnChainMetrics,
        economic: EconomicContext,
        correlations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate trading signals from intelligence data."""
        signals = {
            'sentiment_signal': 'neutral',
            'onchain_signal': 'neutral',
            'macro_signal': 'neutral',
            'overall_bias': 'neutral',
            'confidence': 0.5
        }

        # Sentiment signal
        if sentiment.fear_greed_index < 25:
            signals['sentiment_signal'] = 'bullish'  # Extreme fear = buy
        elif sentiment.fear_greed_index > 75:
            signals['sentiment_signal'] = 'bearish'  # Extreme greed = sell
        elif sentiment.news_sentiment > 0.3:
            signals['sentiment_signal'] = 'bullish'
        elif sentiment.news_sentiment < -0.3:
            signals['sentiment_signal'] = 'bearish'

        # On-chain signal
        if onchain.flow_signal == 'bullish':
            signals['onchain_signal'] = 'bullish'
        elif onchain.flow_signal == 'bearish':
            signals['onchain_signal'] = 'bearish'

        # Macro signal
        if economic.vix_index > 30:
            signals['macro_signal'] = 'bearish'  # High fear
        elif economic.vix_index < 15:
            signals['macro_signal'] = 'bullish'  # Low fear
        elif economic.fed_stance == 'dovish':
            signals['macro_signal'] = 'bullish'
        elif economic.fed_stance == 'hawkish':
            signals['macro_signal'] = 'bearish'

        # Overall bias
        bullish_count = sum(1 for s in [signals['sentiment_signal'], signals['onchain_signal'], signals['macro_signal']] if s == 'bullish')
        bearish_count = sum(1 for s in [signals['sentiment_signal'], signals['onchain_signal'], signals['macro_signal']] if s == 'bearish')

        if bullish_count > bearish_count:
            signals['overall_bias'] = 'bullish'
            signals['confidence'] = 0.5 + bullish_count * 0.15
        elif bearish_count > bullish_count:
            signals['overall_bias'] = 'bearish'
            signals['confidence'] = 0.5 + bearish_count * 0.15
        else:
            signals['overall_bias'] = 'neutral'
            signals['confidence'] = 0.5

        return signals


# Global instance
_data_intelligence: Optional[DataIntelligence] = None

def get_data_intelligence() -> DataIntelligence:
    """Get or create the DataIntelligence instance."""
    global _data_intelligence
    if _data_intelligence is None:
        _data_intelligence = DataIntelligence()
    return _data_intelligence
