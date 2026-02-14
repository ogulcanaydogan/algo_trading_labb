"""
News Sentiment Fetcher with VADER Analysis

Fetches headlines from free sources and analyzes sentiment using VADER.
Provides features for ML model integration.

Free Sources:
- Google News RSS
- Yahoo Finance RSS
- Finviz (scraping)

Features Generated:
- sentiment_score_24h: Rolling 24h average sentiment (-1 to 1)
- sentiment_momentum: Sentiment change over time
- mention_volume: Relative buzz indicator
- sentiment_std: Sentiment volatility (disagreement)
"""

from __future__ import annotations

import logging
import re
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# VADER sentiment analyzer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    logger.warning("vaderSentiment not installed. Run: pip install vaderSentiment")

# HTTP requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# XML parsing for RSS
try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False

# HTML parsing for scraping
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


@dataclass
class NewsItem:
    """Single news item with sentiment."""
    title: str
    source: str
    url: str
    published_at: datetime
    symbol: str
    sentiment_score: float  # -1 to 1 (VADER compound)
    sentiment_pos: float    # Positive component
    sentiment_neg: float    # Negative component
    sentiment_neu: float    # Neutral component
    raw_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
            "symbol": self.symbol,
            "sentiment_score": round(self.sentiment_score, 4),
            "sentiment_pos": round(self.sentiment_pos, 4),
            "sentiment_neg": round(self.sentiment_neg, 4),
            "sentiment_neu": round(self.sentiment_neu, 4),
        }


@dataclass
class SentimentSnapshot:
    """Aggregated sentiment for a symbol at a point in time."""
    symbol: str
    timestamp: datetime
    sentiment_score: float      # Average compound score
    sentiment_std: float        # Standard deviation (disagreement)
    sentiment_pos_ratio: float  # Ratio of positive articles
    sentiment_neg_ratio: float  # Ratio of negative articles
    mention_count: int          # Number of articles
    sources: List[str]          # Unique sources
    top_headlines: List[str]    # Most relevant headlines
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "sentiment_score": round(self.sentiment_score, 4),
            "sentiment_std": round(self.sentiment_std, 4),
            "sentiment_pos_ratio": round(self.sentiment_pos_ratio, 4),
            "sentiment_neg_ratio": round(self.sentiment_neg_ratio, 4),
            "mention_count": self.mention_count,
            "sources": self.sources,
            "top_headlines": self.top_headlines[:3],
        }


class VADERAnalyzer:
    """VADER sentiment analyzer wrapper with financial domain adjustments."""
    
    # Financial domain lexicon adjustments
    FINANCIAL_LEXICON = {
        # Bullish terms
        "bullish": 2.5,
        "rally": 2.0,
        "surge": 2.0,
        "soar": 2.5,
        "breakout": 1.8,
        "outperform": 1.5,
        "upgrade": 1.8,
        "beat": 1.5,
        "record high": 2.5,
        "all-time high": 2.5,
        "ath": 2.5,
        "moon": 2.0,
        "pump": 1.5,
        "accumulate": 1.2,
        "buy": 1.0,
        "long": 1.0,
        
        # Bearish terms
        "bearish": -2.5,
        "crash": -3.0,
        "plunge": -2.5,
        "dump": -2.0,
        "selloff": -2.0,
        "sell-off": -2.0,
        "downgrade": -1.8,
        "miss": -1.5,
        "disappointing": -1.5,
        "bankruptcy": -3.0,
        "fraud": -3.0,
        "hack": -2.5,
        "breach": -2.0,
        "lawsuit": -1.5,
        "investigation": -1.5,
        "short": -1.0,
        "sell": -1.0,
        "rekt": -2.5,
        "capitulation": -2.5,
        
        # Neutral but important
        "halving": 0.5,  # Generally bullish for crypto
        "regulation": -0.3,  # Slight negative
        "sec": -0.2,  # Slight negative
        "fed": 0.0,  # Depends on context
    }
    
    def __init__(self):
        if not HAS_VADER:
            raise ImportError("vaderSentiment required")
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Add financial lexicon
        self.analyzer.lexicon.update(self.FINANCIAL_LEXICON)
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment.
        
        Returns:
            Dict with 'compound', 'pos', 'neg', 'neu' scores
        """
        if not text:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        
        # Clean text
        text = self._preprocess(text)
        
        # Get VADER scores
        scores = self.analyzer.polarity_scores(text)
        
        return {
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
        }
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for better analysis."""
        # Convert to lowercase but preserve CAPS emphasis
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s!?.,\'-]', ' ', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text


class NewsSentimentFetcher:
    """
    Fetches news from free sources and analyzes sentiment.
    
    Supports:
    - Google News RSS
    - Yahoo Finance RSS
    - Finviz headlines (scraping)
    """
    
    # Symbol to search term mapping
    SYMBOL_KEYWORDS = {
        # Stocks
        "TSLA": ["tesla", "tsla", "elon musk"],
        "AAPL": ["apple", "aapl", "iphone", "tim cook"],
        "NVDA": ["nvidia", "nvda", "jensen huang"],
        "MSFT": ["microsoft", "msft", "satya nadella"],
        "GOOGL": ["google", "alphabet", "googl"],
        "AMZN": ["amazon", "amzn", "jeff bezos", "andy jassy"],
        "META": ["meta", "facebook", "instagram", "zuckerberg"],
        "AMD": ["amd", "advanced micro devices", "lisa su"],
        
        # Crypto
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth", "vitalik"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"],
        "DOGE": ["dogecoin", "doge"],
    }
    
    # User agent for requests
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    
    def __init__(self, cache_ttl_minutes: int = 15):
        """
        Initialize fetcher.
        
        Args:
            cache_ttl_minutes: Cache TTL in minutes
        """
        if not HAS_VADER:
            raise ImportError("vaderSentiment required")
        if not HAS_REQUESTS:
            raise ImportError("requests required")
            
        self.analyzer = VADERAnalyzer()
        self._cache: Dict[str, Tuple[List[NewsItem], datetime]] = {}
        self._cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._rate_limit_delay = 1.0  # seconds between requests
        self._last_request = 0.0
    
    def get_keywords(self, symbol: str) -> List[str]:
        """Get search keywords for a symbol."""
        # Clean symbol
        clean = symbol.upper().replace("/USD", "").replace("/USDT", "").replace("USD", "")
        
        if clean in self.SYMBOL_KEYWORDS:
            return self.SYMBOL_KEYWORDS[clean]
        
        # Default to symbol itself
        return [clean.lower(), symbol.lower()]
    
    def fetch_news(
        self,
        symbol: str,
        hours_back: int = 24,
        max_items: int = 50,
    ) -> List[NewsItem]:
        """
        Fetch and analyze news for a symbol.
        
        Args:
            symbol: Asset symbol (e.g., "TSLA", "BTC/USD")
            hours_back: How far back to fetch
            max_items: Maximum items to return
            
        Returns:
            List of NewsItem with sentiment scores
        """
        cache_key = f"{symbol}_{hours_back}"
        
        # Check cache
        if cache_key in self._cache:
            items, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return items[:max_items]
        
        all_items: List[NewsItem] = []
        
        # Fetch from multiple sources
        try:
            google_items = self._fetch_google_news(symbol, hours_back)
            all_items.extend(google_items)
        except Exception as e:
            logger.warning(f"Google News fetch failed: {e}")
        
        try:
            yahoo_items = self._fetch_yahoo_finance(symbol, hours_back)
            all_items.extend(yahoo_items)
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed: {e}")
        
        if HAS_BS4:
            try:
                finviz_items = self._fetch_finviz(symbol)
                all_items.extend(finviz_items)
            except Exception as e:
                logger.warning(f"Finviz fetch failed: {e}")
        
        # Deduplicate by URL hash
        seen_hashes = set()
        unique_items = []
        for item in all_items:
            url_hash = hashlib.md5(item.url.encode()).hexdigest()[:12]
            if url_hash not in seen_hashes:
                seen_hashes.add(url_hash)
                unique_items.append(item)
        
        # Sort by date (newest first)
        unique_items.sort(key=lambda x: x.published_at, reverse=True)
        
        # Cache results
        self._cache[cache_key] = (unique_items, datetime.now())
        
        return unique_items[:max_items]
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request = time.time()
    
    def _fetch_google_news(self, symbol: str, hours_back: int) -> List[NewsItem]:
        """Fetch from Google News RSS."""
        keywords = self.get_keywords(symbol)
        query = "+".join(keywords[0].split())  # Use first keyword
        
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        self._rate_limit()
        
        response = requests.get(
            url,
            headers={"User-Agent": self.USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        
        items = []
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        root = ET.fromstring(response.content)
        
        for item in root.findall(".//item"):
            try:
                title = item.find("title").text or ""
                link = item.find("link").text or ""
                pub_date_str = item.find("pubDate").text or ""
                source = item.find("source").text if item.find("source") is not None else "Google News"
                
                # Parse date
                try:
                    pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                except ValueError:
                    pub_date = datetime.now()
                
                if pub_date < cutoff:
                    continue
                
                # Analyze sentiment
                scores = self.analyzer.analyze(title)
                
                news_item = NewsItem(
                    title=title,
                    source=source,
                    url=link,
                    published_at=pub_date,
                    symbol=symbol,
                    sentiment_score=scores["compound"],
                    sentiment_pos=scores["pos"],
                    sentiment_neg=scores["neg"],
                    sentiment_neu=scores["neu"],
                )
                items.append(news_item)
                
            except Exception as e:
                logger.debug(f"Failed to parse item: {e}")
                continue
        
        return items
    
    def _fetch_yahoo_finance(self, symbol: str, hours_back: int) -> List[NewsItem]:
        """Fetch from Yahoo Finance RSS."""
        # Clean symbol for Yahoo
        clean_symbol = symbol.replace("/USD", "").replace("/USDT", "").upper()
        
        # Yahoo Finance RSS for specific symbol
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={clean_symbol}&region=US&lang=en-US"
        
        self._rate_limit()
        
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException:
            # Try general finance news
            url = "https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US"
            response = requests.get(
                url,
                headers={"User-Agent": self.USER_AGENT},
                timeout=10,
            )
            response.raise_for_status()
        
        items = []
        cutoff = datetime.now() - timedelta(hours=hours_back)
        keywords = [k.lower() for k in self.get_keywords(symbol)]
        
        root = ET.fromstring(response.content)
        
        for item in root.findall(".//item"):
            try:
                title = item.find("title").text or ""
                description = item.find("description").text or ""
                link = item.find("link").text or ""
                pub_date_str = item.find("pubDate").text or ""
                
                # Check relevance to symbol
                text_lower = (title + " " + description).lower()
                if not any(kw in text_lower for kw in keywords):
                    continue
                
                # Parse date
                try:
                    pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
                    pub_date = pub_date.replace(tzinfo=None)
                except ValueError:
                    pub_date = datetime.now()
                
                if pub_date < cutoff:
                    continue
                
                # Analyze sentiment (combine title and description)
                full_text = f"{title}. {description}"
                scores = self.analyzer.analyze(full_text)
                
                news_item = NewsItem(
                    title=title,
                    source="Yahoo Finance",
                    url=link,
                    published_at=pub_date,
                    symbol=symbol,
                    sentiment_score=scores["compound"],
                    sentiment_pos=scores["pos"],
                    sentiment_neg=scores["neg"],
                    sentiment_neu=scores["neu"],
                    raw_text=description[:500],
                )
                items.append(news_item)
                
            except Exception as e:
                logger.debug(f"Failed to parse item: {e}")
                continue
        
        return items
    
    def _fetch_finviz(self, symbol: str) -> List[NewsItem]:
        """Scrape headlines from Finviz (stocks only)."""
        if not HAS_BS4:
            return []
        
        # Clean symbol for Finviz (stocks only, no crypto)
        clean_symbol = symbol.replace("/USD", "").replace("/USDT", "").upper()
        
        # Skip crypto symbols
        crypto_symbols = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "DOT", "LINK"}
        if clean_symbol in crypto_symbols:
            return []
        
        url = f"https://finviz.com/quote.ashx?t={clean_symbol}"
        
        self._rate_limit()
        
        response = requests.get(
            url,
            headers={"User-Agent": self.USER_AGENT},
            timeout=10,
        )
        response.raise_for_status()
        
        items = []
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find news table
        news_table = soup.find("table", {"id": "news-table"})
        if not news_table:
            return []
        
        current_date = None
        
        for row in news_table.find_all("tr"):
            try:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                
                # Parse date/time
                date_cell = cells[0].text.strip()
                if len(date_cell) > 8:  # Contains date
                    date_part, time_part = date_cell.split()[:2]
                    current_date = datetime.strptime(date_part, "%b-%d-%y").date()
                else:
                    time_part = date_cell
                
                if current_date is None:
                    current_date = datetime.now().date()
                
                # Parse time
                try:
                    time_obj = datetime.strptime(time_part, "%I:%M%p").time()
                except ValueError:
                    time_obj = datetime.now().time()
                
                pub_date = datetime.combine(current_date, time_obj)
                
                # Get headline
                link_tag = cells[1].find("a")
                if not link_tag:
                    continue
                
                title = link_tag.text.strip()
                link = link_tag.get("href", "")
                source = cells[1].find("span")
                source = source.text.strip() if source else "Finviz"
                
                # Analyze sentiment
                scores = self.analyzer.analyze(title)
                
                news_item = NewsItem(
                    title=title,
                    source=source,
                    url=link,
                    published_at=pub_date,
                    symbol=symbol,
                    sentiment_score=scores["compound"],
                    sentiment_pos=scores["pos"],
                    sentiment_neg=scores["neg"],
                    sentiment_neu=scores["neu"],
                )
                items.append(news_item)
                
            except Exception as e:
                logger.debug(f"Failed to parse Finviz row: {e}")
                continue
        
        return items
    
    def aggregate_sentiment(
        self,
        symbol: str,
        hours_back: int = 24,
    ) -> SentimentSnapshot:
        """
        Get aggregated sentiment for a symbol.
        
        Args:
            symbol: Asset symbol
            hours_back: Lookback period
            
        Returns:
            SentimentSnapshot with aggregated metrics
        """
        items = self.fetch_news(symbol, hours_back=hours_back)
        
        if not items:
            return SentimentSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                sentiment_score=0.0,
                sentiment_std=0.0,
                sentiment_pos_ratio=0.5,
                sentiment_neg_ratio=0.5,
                mention_count=0,
                sources=[],
                top_headlines=[],
            )
        
        scores = [item.sentiment_score for item in items]
        
        # Calculate metrics
        avg_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        pos_count = sum(1 for s in scores if s > 0.05)
        neg_count = sum(1 for s in scores if s < -0.05)
        total = len(scores)
        
        return SentimentSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            sentiment_score=float(avg_score),
            sentiment_std=float(std_score),
            sentiment_pos_ratio=pos_count / total if total > 0 else 0.5,
            sentiment_neg_ratio=neg_count / total if total > 0 else 0.5,
            mention_count=len(items),
            sources=list(set(item.source for item in items)),
            top_headlines=[item.title for item in items[:5]],
        )


class SentimentFeatureGenerator:
    """
    Generate ML features from sentiment data.
    
    Features:
    - sentiment_score_24h: Rolling 24h average sentiment
    - sentiment_momentum: Sentiment change (current vs previous period)
    - mention_volume: Normalized mention count
    - sentiment_std: Disagreement indicator
    - sentiment_pos_ratio: Bullish ratio
    - sentiment_extreme: Extreme sentiment indicator
    """
    
    def __init__(self, fetcher: Optional[NewsSentimentFetcher] = None):
        """
        Initialize feature generator.
        
        Args:
            fetcher: NewsSentimentFetcher instance (created if not provided)
        """
        self.fetcher = fetcher or NewsSentimentFetcher()
        self._history: Dict[str, List[SentimentSnapshot]] = defaultdict(list)
        self._baseline_mentions: Dict[str, float] = {}
    
    def update_sentiment(self, symbol: str) -> SentimentSnapshot:
        """
        Fetch and store latest sentiment for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Latest SentimentSnapshot
        """
        snapshot = self.fetcher.aggregate_sentiment(symbol, hours_back=24)
        self._history[symbol].append(snapshot)
        
        # Keep last 7 days of history
        cutoff = datetime.now() - timedelta(days=7)
        self._history[symbol] = [
            s for s in self._history[symbol]
            if s.timestamp > cutoff
        ]
        
        return snapshot
    
    def get_features(self, symbol: str) -> Dict[str, float]:
        """
        Get sentiment features for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dict of feature name -> value
        """
        # Get current sentiment
        current = self.update_sentiment(symbol)
        history = self._history[symbol]
        
        features = {}
        
        # Current sentiment score
        features["sentiment_score_24h"] = current.sentiment_score
        
        # Sentiment momentum (change from previous period)
        if len(history) >= 2:
            prev_score = history[-2].sentiment_score
            features["sentiment_momentum"] = current.sentiment_score - prev_score
        else:
            features["sentiment_momentum"] = 0.0
        
        # Normalized mention volume
        baseline = self._baseline_mentions.get(symbol, 10.0)
        features["mention_volume"] = min(3.0, current.mention_count / baseline)
        
        # Update baseline (exponential moving average)
        if current.mention_count > 0:
            if symbol not in self._baseline_mentions:
                self._baseline_mentions[symbol] = float(current.mention_count)
            else:
                self._baseline_mentions[symbol] = (
                    0.9 * self._baseline_mentions[symbol] +
                    0.1 * current.mention_count
                )
        
        # Sentiment volatility (disagreement)
        features["sentiment_std"] = current.sentiment_std
        
        # Bullish/bearish ratios
        features["sentiment_pos_ratio"] = current.sentiment_pos_ratio
        features["sentiment_neg_ratio"] = current.sentiment_neg_ratio
        
        # Extreme sentiment indicator
        features["sentiment_extreme"] = 1.0 if abs(current.sentiment_score) > 0.5 else 0.0
        
        # Contrarian signal (fade extremes)
        if current.sentiment_score > 0.6:
            features["sentiment_contrarian"] = -0.5  # Too bullish, fade
        elif current.sentiment_score < -0.6:
            features["sentiment_contrarian"] = 0.5   # Too bearish, fade
        else:
            features["sentiment_contrarian"] = 0.0
        
        return features
    
    def get_features_df(
        self,
        symbol: str,
        price_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Get sentiment features aligned to a price DataFrame index.
        
        Args:
            symbol: Asset symbol
            price_index: DatetimeIndex from price data
            
        Returns:
            DataFrame with sentiment features
        """
        features = self.get_features(symbol)
        
        # Create DataFrame with constant features (forward-filled from latest)
        df = pd.DataFrame(index=price_index)
        
        for name, value in features.items():
            df[name] = value
        
        return df


def get_sentiment_features(
    symbol: str,
    price_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Convenience function to get sentiment features for a symbol.
    
    Args:
        symbol: Asset symbol
        price_df: Optional price DataFrame (for alignment)
        
    Returns:
        Dict of sentiment features
    """
    generator = SentimentFeatureGenerator()
    return generator.get_features(symbol)


# Test/demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== News Sentiment Module Test ===\n")
    
    fetcher = NewsSentimentFetcher()
    
    # Test with TSLA
    symbol = "TSLA"
    print(f"Fetching news for {symbol}...")
    
    items = fetcher.fetch_news(symbol, hours_back=48)
    print(f"Found {len(items)} news items\n")
    
    # Show sample
    for item in items[:5]:
        sentiment_label = "🟢" if item.sentiment_score > 0.1 else "🔴" if item.sentiment_score < -0.1 else "⚪"
        print(f"{sentiment_label} [{item.sentiment_score:+.2f}] {item.title[:60]}...")
        print(f"   Source: {item.source}, Date: {item.published_at}")
        print()
    
    # Get aggregated sentiment
    snapshot = fetcher.aggregate_sentiment(symbol)
    print(f"\n=== Aggregated Sentiment for {symbol} ===")
    print(f"Score: {snapshot.sentiment_score:+.3f}")
    print(f"Std: {snapshot.sentiment_std:.3f}")
    print(f"Mentions: {snapshot.mention_count}")
    print(f"Bullish ratio: {snapshot.sentiment_pos_ratio:.1%}")
    print(f"Bearish ratio: {snapshot.sentiment_neg_ratio:.1%}")
    
    # Get ML features
    generator = SentimentFeatureGenerator(fetcher)
    features = generator.get_features(symbol)
    
    print(f"\n=== ML Features for {symbol} ===")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
