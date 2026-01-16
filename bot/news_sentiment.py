"""
News and Sentiment Analysis Integration.

Fetches news from multiple sources, analyzes sentiment,
and provides signals for trading decisions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class SentimentType(Enum):
    """Sentiment classification types."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class NewsSource(Enum):
    """Supported news sources."""
    NEWSAPI = "newsapi"
    ALPHA_VANTAGE = "alpha_vantage"
    CRYPTO_PANIC = "crypto_panic"
    FINNHUB = "finnhub"
    POLYGON = "polygon"


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str]
    sentiment_score: Optional[float] = None
    sentiment_type: Optional[SentimentType] = None
    relevance_score: float = 1.0
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "symbols": self.symbols,
            "sentiment_score": self.sentiment_score,
            "sentiment_type": self.sentiment_type.value if self.sentiment_type else None,
            "relevance_score": self.relevance_score,
        }


@dataclass
class SentimentSignal:
    """Sentiment-based trading signal."""
    symbol: str
    sentiment_score: float  # -1 to 1
    sentiment_type: SentimentType
    confidence: float
    news_count: int
    avg_relevance: float
    key_headlines: List[str]
    signal_strength: float  # 0 to 1
    recommended_action: str  # LONG, SHORT, FLAT
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sentiment_score": round(self.sentiment_score, 4),
            "sentiment_type": self.sentiment_type.value,
            "confidence": round(self.confidence, 4),
            "news_count": self.news_count,
            "avg_relevance": round(self.avg_relevance, 4),
            "key_headlines": self.key_headlines[:3],
            "signal_strength": round(self.signal_strength, 4),
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
        }


class NewsFetcher:
    """Fetches news from various sources."""

    # Symbol mappings for different APIs
    CRYPTO_KEYWORDS = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth", "ether"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"],
        "DOGE": ["dogecoin", "doge"],
    }

    COMMODITY_KEYWORDS = {
        "XAU": ["gold", "precious metals"],
        "XAG": ["silver", "precious metals"],
        "USOIL": ["oil", "crude", "wti", "brent", "petroleum"],
        "NATGAS": ["natural gas", "nat gas"],
    }

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        crypto_panic_key: Optional[str] = None,
        finnhub_key: Optional[str] = None,
        polygon_key: Optional[str] = None,
    ):
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_API_KEY")
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHAVANTAGE_API_KEY")
        self.crypto_panic_key = crypto_panic_key or os.getenv("CRYPTOPANIC_API_KEY")
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY")
        self.polygon_key = polygon_key or os.getenv("POLYGON_API_KEY")

        self._cache: Dict[str, Tuple[List[NewsArticle], datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)

    def fetch_news(
        self,
        symbol: str,
        source: NewsSource = NewsSource.NEWSAPI,
        limit: int = 20,
        hours_back: int = 24,
    ) -> List[NewsArticle]:
        """
        Fetch news for a symbol from specified source.

        Args:
            symbol: Asset symbol
            source: News source to use
            limit: Maximum articles to fetch
            hours_back: How far back to look

        Returns:
            List of NewsArticle objects
        """
        if not HAS_REQUESTS:
            logger.warning("requests library not installed")
            return []

        cache_key = f"{symbol}_{source.value}_{hours_back}"

        # Check cache
        if cache_key in self._cache:
            articles, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return articles[:limit]

        articles = []

        try:
            if source == NewsSource.NEWSAPI:
                articles = self._fetch_newsapi(symbol, limit, hours_back)
            elif source == NewsSource.CRYPTO_PANIC:
                articles = self._fetch_crypto_panic(symbol, limit)
            elif source == NewsSource.FINNHUB:
                articles = self._fetch_finnhub(symbol, limit)
            elif source == NewsSource.ALPHA_VANTAGE:
                articles = self._fetch_alpha_vantage(symbol, limit)
        except Exception as e:
            logger.error(f"Failed to fetch news from {source.value}: {e}")

        # Cache results
        self._cache[cache_key] = (articles, datetime.now())

        return articles[:limit]

    def _get_keywords(self, symbol: str) -> List[str]:
        """Get search keywords for a symbol."""
        # Clean symbol
        base = symbol.replace("/USDT", "").replace("/USD", "").replace("USD", "")

        # Check mappings
        if base in self.CRYPTO_KEYWORDS:
            return self.CRYPTO_KEYWORDS[base]
        if base in self.COMMODITY_KEYWORDS:
            return self.COMMODITY_KEYWORDS[base]

        # For stocks, use the ticker symbol
        return [base, symbol]

    def _fetch_newsapi(
        self,
        symbol: str,
        limit: int,
        hours_back: int,
    ) -> List[NewsArticle]:
        """Fetch from NewsAPI."""
        if not self.newsapi_key:
            return []

        keywords = self._get_keywords(symbol)
        query = " OR ".join(keywords)

        from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "pageSize": limit,
            "language": "en",
            "apiKey": self.newsapi_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("articles", []):
            try:
                pub_at = datetime.fromisoformat(
                    item.get("publishedAt", "").replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pub_at = datetime.now()

            article = NewsArticle(
                title=item.get("title", ""),
                description=item.get("description", ""),
                source=item.get("source", {}).get("name", "NewsAPI"),
                url=item.get("url", ""),
                published_at=pub_at,
                symbols=[symbol],
                raw_data=item,
            )
            articles.append(article)

        return articles

    def _fetch_crypto_panic(
        self,
        symbol: str,
        limit: int,
    ) -> List[NewsArticle]:
        """Fetch from CryptoPanic."""
        if not self.crypto_panic_key:
            return []

        # Extract crypto symbol
        base = symbol.replace("/USDT", "").replace("/USD", "").upper()

        url = f"https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.crypto_panic_key,
            "currencies": base,
            "public": "true",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("results", [])[:limit]:
            try:
                pub_at = datetime.fromisoformat(
                    item.get("published_at", "").replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pub_at = datetime.now()

            # CryptoPanic includes sentiment votes
            votes = item.get("votes", {})
            sentiment_hint = votes.get("positive", 0) - votes.get("negative", 0)

            article = NewsArticle(
                title=item.get("title", ""),
                description=item.get("title", ""),  # CryptoPanic has short titles
                source="CryptoPanic",
                url=item.get("url", ""),
                published_at=pub_at,
                symbols=[symbol],
                sentiment_score=sentiment_hint / (abs(sentiment_hint) + 1) if sentiment_hint else None,
                raw_data=item,
            )
            articles.append(article)

        return articles

    def _fetch_finnhub(
        self,
        symbol: str,
        limit: int,
    ) -> List[NewsArticle]:
        """Fetch from Finnhub."""
        if not self.finnhub_key:
            return []

        # Finnhub expects stock symbols
        clean_symbol = symbol.replace("/USDT", "").replace("/USD", "").upper()

        url = "https://finnhub.io/api/v1/company-news"
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        params = {
            "symbol": clean_symbol,
            "from": from_date,
            "to": to_date,
            "token": self.finnhub_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data[:limit] if isinstance(data, list) else []:
            try:
                pub_at = datetime.fromtimestamp(item.get("datetime", 0))
            except (ValueError, TypeError, OSError):
                pub_at = datetime.now()

            article = NewsArticle(
                title=item.get("headline", ""),
                description=item.get("summary", ""),
                source=item.get("source", "Finnhub"),
                url=item.get("url", ""),
                published_at=pub_at,
                symbols=[symbol],
                raw_data=item,
            )
            articles.append(article)

        return articles

    def _fetch_alpha_vantage(
        self,
        symbol: str,
        limit: int,
    ) -> List[NewsArticle]:
        """Fetch from Alpha Vantage."""
        if not self.alpha_vantage_key:
            return []

        keywords = self._get_keywords(symbol)

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol.replace("/USDT", "").replace("/USD", ""),
            "apikey": self.alpha_vantage_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("feed", [])[:limit]:
            try:
                pub_at = datetime.strptime(
                    item.get("time_published", ""),
                    "%Y%m%dT%H%M%S"
                )
            except (ValueError, TypeError):
                pub_at = datetime.now()

            # Alpha Vantage includes sentiment
            sentiment = item.get("overall_sentiment_score", 0)

            article = NewsArticle(
                title=item.get("title", ""),
                description=item.get("summary", ""),
                source=item.get("source", "Alpha Vantage"),
                url=item.get("url", ""),
                published_at=pub_at,
                symbols=[symbol],
                sentiment_score=sentiment,
                raw_data=item,
            )
            articles.append(article)

        return articles


class SentimentAnalyzer:
    """Analyzes sentiment from news text."""

    def __init__(
        self,
        use_transformers: bool = True,
        use_claude: bool = False,
        claude_client: Optional[Any] = None,
    ):
        self.use_transformers = use_transformers and HAS_TRANSFORMERS
        self.use_claude = use_claude
        self.claude_client = claude_client

        self._transformer_model = None

        # Sentiment keywords for rule-based fallback
        self.bullish_keywords = [
            "surge", "soar", "rally", "bullish", "breakout", "all-time high",
            "adoption", "partnership", "approval", "upgrade", "growth",
            "buy", "accumulate", "outperform", "beat", "strong",
        ]
        self.bearish_keywords = [
            "crash", "plunge", "dump", "bearish", "breakdown", "low",
            "hack", "scam", "fraud", "ban", "regulation", "sell",
            "decline", "loss", "weak", "underperform", "miss",
        ]

    def _get_transformer_model(self):
        """Lazy load transformer model."""
        if self._transformer_model is None and self.use_transformers:
            try:
                self._transformer_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device="mps" if os.uname().sysname == "Darwin" else "cpu",
                )
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")
                self.use_transformers = False
        return self._transformer_model

    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score: -1 (bearish) to 1 (bullish)
            confidence: 0 to 1
        """
        if not text:
            return 0.0, 0.0

        # Try transformer model first
        if self.use_transformers:
            model = self._get_transformer_model()
            if model:
                try:
                    result = model(text[:512])[0]  # Limit text length
                    label = result["label"].lower()
                    score = result["score"]

                    if "positive" in label:
                        return score, score
                    elif "negative" in label:
                        return -score, score
                    else:
                        return 0.0, score
                except Exception as e:
                    logger.warning(f"Transformer analysis failed: {e}")

        # Fall back to TextBlob
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                confidence = 1 - subjectivity  # More objective = more confident
                return polarity, confidence
            except Exception:
                pass

        # Fall back to keyword-based
        return self._keyword_sentiment(text)

    def _keyword_sentiment(self, text: str) -> Tuple[float, float]:
        """Simple keyword-based sentiment analysis."""
        text_lower = text.lower()

        bullish_count = sum(1 for kw in self.bullish_keywords if kw in text_lower)
        bearish_count = sum(1 for kw in self.bearish_keywords if kw in text_lower)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 0.3  # Neutral with low confidence

        score = (bullish_count - bearish_count) / total
        confidence = min(0.8, total * 0.1)  # More keywords = more confident

        return score, confidence

    def analyze_article(self, article: NewsArticle) -> NewsArticle:
        """Analyze sentiment of a news article."""
        # Combine title and description
        text = f"{article.title}. {article.description}"

        score, confidence = self.analyze_text(text)

        article.sentiment_score = score
        article.sentiment_type = self._score_to_type(score)

        return article

    def _score_to_type(self, score: float) -> SentimentType:
        """Convert numerical score to sentiment type."""
        if score >= 0.5:
            return SentimentType.VERY_BULLISH
        elif score >= 0.2:
            return SentimentType.BULLISH
        elif score <= -0.5:
            return SentimentType.VERY_BEARISH
        elif score <= -0.2:
            return SentimentType.BEARISH
        else:
            return SentimentType.NEUTRAL


class NewsSentimentEngine:
    """
    Complete news sentiment analysis engine.

    Fetches news, analyzes sentiment, and generates trading signals.

    Usage:
        engine = NewsSentimentEngine()

        # Get sentiment signal for a symbol
        signal = engine.get_sentiment_signal("BTC/USDT")

        # Get signals for multiple symbols
        signals = engine.get_all_signals(["BTC/USDT", "ETH/USDT", "AAPL"])
    """

    def __init__(
        self,
        data_dir: str = "data/news_sentiment",
        sentiment_threshold: float = 0.3,
        min_news_count: int = 3,
        lookback_hours: int = 24,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sentiment_threshold = sentiment_threshold
        self.min_news_count = min_news_count
        self.lookback_hours = lookback_hours

        self.fetcher = NewsFetcher()
        self.analyzer = SentimentAnalyzer()

        # Cache signals
        self._signal_cache: Dict[str, Tuple[SentimentSignal, datetime]] = {}
        self._cache_ttl = timedelta(minutes=30)

    def get_sentiment_signal(
        self,
        symbol: str,
        sources: Optional[List[NewsSource]] = None,
    ) -> SentimentSignal:
        """
        Get sentiment signal for a symbol.

        Args:
            symbol: Asset symbol
            sources: List of news sources to use

        Returns:
            SentimentSignal with aggregated sentiment
        """
        # Check cache
        if symbol in self._signal_cache:
            signal, cached_at = self._signal_cache[symbol]
            if datetime.now() - cached_at < self._cache_ttl:
                return signal

        # Default sources
        if sources is None:
            if "BTC" in symbol or "ETH" in symbol:
                sources = [NewsSource.NEWSAPI, NewsSource.CRYPTO_PANIC]
            elif "/" not in symbol:  # Stock
                sources = [NewsSource.FINNHUB, NewsSource.NEWSAPI]
            else:  # Commodity
                sources = [NewsSource.NEWSAPI]

        # Fetch news from all sources
        all_articles: List[NewsArticle] = []
        for source in sources:
            try:
                articles = self.fetcher.fetch_news(
                    symbol,
                    source=source,
                    limit=10,
                    hours_back=self.lookback_hours,
                )
                all_articles.extend(articles)
            except Exception as e:
                logger.warning(f"Failed to fetch from {source.value}: {e}")

        # Analyze sentiment
        for article in all_articles:
            if article.sentiment_score is None:
                self.analyzer.analyze_article(article)

        # Aggregate sentiment
        signal = self._aggregate_sentiment(symbol, all_articles)

        # Cache
        self._signal_cache[symbol] = (signal, datetime.now())

        return signal

    def _aggregate_sentiment(
        self,
        symbol: str,
        articles: List[NewsArticle],
    ) -> SentimentSignal:
        """Aggregate sentiment from multiple articles."""
        if not articles:
            return SentimentSignal(
                symbol=symbol,
                sentiment_score=0.0,
                sentiment_type=SentimentType.NEUTRAL,
                confidence=0.0,
                news_count=0,
                avg_relevance=0.0,
                key_headlines=[],
                signal_strength=0.0,
                recommended_action="FLAT",
            )

        # Weight by recency and relevance
        now = datetime.now()
        weighted_scores = []
        weights = []
        headlines = []

        for article in articles:
            if article.sentiment_score is None:
                continue

            # Recency weight (exponential decay)
            hours_old = (now - article.published_at).total_seconds() / 3600
            recency_weight = np.exp(-hours_old / 24)

            # Combined weight
            weight = recency_weight * article.relevance_score

            weighted_scores.append(article.sentiment_score * weight)
            weights.append(weight)
            headlines.append(article.title)

        if not weights:
            return SentimentSignal(
                symbol=symbol,
                sentiment_score=0.0,
                sentiment_type=SentimentType.NEUTRAL,
                confidence=0.0,
                news_count=len(articles),
                avg_relevance=np.mean([a.relevance_score for a in articles]),
                key_headlines=headlines[:3],
                signal_strength=0.0,
                recommended_action="FLAT",
            )

        # Calculate weighted average
        total_weight = sum(weights)
        avg_sentiment = sum(weighted_scores) / total_weight if total_weight > 0 else 0

        # Calculate confidence based on consistency and volume
        sentiment_std = np.std([a.sentiment_score for a in articles if a.sentiment_score])
        consistency = max(0, 1 - sentiment_std)

        volume_confidence = min(1, len(articles) / 10)  # Max at 10 articles
        confidence = consistency * 0.5 + volume_confidence * 0.5

        # Determine sentiment type and signal
        sentiment_type = self.analyzer._score_to_type(avg_sentiment)

        # Signal strength (0 to 1)
        signal_strength = abs(avg_sentiment) * confidence

        # Recommended action
        if avg_sentiment >= self.sentiment_threshold and confidence >= 0.4:
            action = "LONG"
        elif avg_sentiment <= -self.sentiment_threshold and confidence >= 0.4:
            action = "SHORT"
        else:
            action = "FLAT"

        # If not enough news, reduce confidence in signal
        if len(articles) < self.min_news_count:
            action = "FLAT"
            signal_strength *= 0.5

        return SentimentSignal(
            symbol=symbol,
            sentiment_score=avg_sentiment,
            sentiment_type=sentiment_type,
            confidence=confidence,
            news_count=len(articles),
            avg_relevance=np.mean([a.relevance_score for a in articles]),
            key_headlines=headlines[:5],
            signal_strength=signal_strength,
            recommended_action=action,
        )

    def get_all_signals(
        self,
        symbols: List[str],
    ) -> Dict[str, SentimentSignal]:
        """Get sentiment signals for multiple symbols."""
        signals = {}
        for symbol in symbols:
            try:
                signals[symbol] = self.get_sentiment_signal(symbol)
            except Exception as e:
                logger.error(f"Failed to get signal for {symbol}: {e}")

        return signals

    def save_signals(self, signals: Dict[str, SentimentSignal]) -> None:
        """Save signals to disk."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "signals": {k: v.to_dict() for k, v in signals.items()},
        }

        filepath = self.data_dir / "latest_signals.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
