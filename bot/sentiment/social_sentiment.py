"""
Social Media Sentiment Analyzer

Fetches and analyzes sentiment from Reddit and other social sources.
Uses free tier APIs (no auth required for public data).

Sources:
- Reddit (via public JSON API - no auth needed)
  - r/wallstreetbets
  - r/stocks
  - r/investing
  - r/cryptocurrency (for crypto)
  
Features Generated:
- social_sentiment: Average sentiment from social posts
- social_volume: Post/comment volume indicator
- wsb_mentions: WallStreetBets specific buzz
- social_momentum: Change in social sentiment
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# VADER sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

# HTTP requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class SocialPost:
    """Single social media post with sentiment."""
    title: str
    body: str
    source: str  # e.g., "reddit:wallstreetbets"
    url: str
    created_at: datetime
    symbol: str
    sentiment_score: float
    upvotes: int
    comments: int
    author: str
    
    @property
    def engagement_score(self) -> float:
        """Calculate engagement score."""
        return np.log1p(self.upvotes) + np.log1p(self.comments) * 2


class RedditSentimentFetcher:
    """
    Fetches sentiment from Reddit using public JSON API.
    
    No authentication required for public subreddit data.
    Uses Reddit's .json endpoint.
    """
    
    # Subreddits to monitor by asset type
    STOCK_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "stockmarket"]
    CRYPTO_SUBREDDITS = ["cryptocurrency", "bitcoin", "ethereum", "CryptoMarkets"]
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 10
    
    # User agent (Reddit requires descriptive user agent)
    USER_AGENT = (
        "python:algo_trading_sentiment:v1.0 "
        "(by /u/trading_bot_research)"
    )
    
    def __init__(self):
        if not HAS_VADER:
            raise ImportError("vaderSentiment required")
        if not HAS_REQUESTS:
            raise ImportError("requests required")
            
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Add trading slang to lexicon
        self._add_trading_lexicon()
        
        self._cache: Dict[str, Tuple[List[SocialPost], datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)
        self._last_request = 0.0
    
    def _add_trading_lexicon(self):
        """Add trading/meme slang to VADER lexicon."""
        trading_words = {
            # Bullish
            "diamond hands": 2.5,
            "diamondhands": 2.5,
            "💎🙌": 2.5,
            "to the moon": 3.0,
            "moon": 2.0,
            "tendies": 1.5,
            "apes": 1.0,
            "hodl": 1.5,
            "bullish": 2.5,
            "yolo": 1.0,
            "calls": 0.5,
            "squeeze": 2.0,
            "gamma": 1.0,
            "rocket": 2.0,
            "🚀": 2.5,
            "undervalued": 1.5,
            
            # Bearish
            "paper hands": -2.0,
            "paperhands": -2.0,
            "bagholding": -2.0,
            "bagholder": -2.0,
            "puts": -0.5,
            "short": -0.5,
            "overvalued": -1.5,
            "dump": -2.0,
            "rug": -3.0,
            "rug pull": -3.0,
            "rekt": -2.5,
            "bearish": -2.5,
            "loss porn": -1.5,
            "guh": -2.0,
            "drill": -2.0,
            "🐻": -1.5,
            
            # Neutral/context-dependent
            "wsb": 0.0,
            "dd": 0.5,  # Due diligence usually positive
            "retard": 0.0,  # WSB term, neutral there
            "smooth brain": 0.0,
        }
        self.analyzer.lexicon.update(trading_words)
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        min_interval = 60.0 / self.REQUESTS_PER_MINUTE
        elapsed = now - self._last_request
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request = time.time()
    
    def _get_subreddits(self, symbol: str) -> List[str]:
        """Get relevant subreddits for a symbol."""
        clean = symbol.upper().replace("/USD", "").replace("/USDT", "")
        
        crypto_symbols = {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "DOT", "LINK", "AVAX"}
        
        if clean in crypto_symbols:
            return self.CRYPTO_SUBREDDITS
        else:
            return self.STOCK_SUBREDDITS
    
    def _search_keywords(self, symbol: str) -> List[str]:
        """Get search keywords for a symbol."""
        clean = symbol.upper().replace("/USD", "").replace("/USDT", "")
        
        # Symbol-specific keywords
        keywords = {
            "TSLA": ["TSLA", "Tesla", "$TSLA"],
            "AAPL": ["AAPL", "Apple", "$AAPL"],
            "NVDA": ["NVDA", "Nvidia", "$NVDA"],
            "AMD": ["AMD", "$AMD"],
            "GME": ["GME", "GameStop", "$GME"],
            "AMC": ["AMC", "$AMC"],
            "BTC": ["BTC", "Bitcoin"],
            "ETH": ["ETH", "Ethereum"],
        }
        
        if clean in keywords:
            return keywords[clean]
        
        return [clean, f"${clean}"]
    
    def fetch_subreddit_posts(
        self,
        subreddit: str,
        symbol: str,
        limit: int = 25,
    ) -> List[SocialPost]:
        """
        Fetch posts from a subreddit matching a symbol.
        
        Args:
            subreddit: Subreddit name
            symbol: Symbol to search for
            limit: Max posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        cache_key = f"{subreddit}_{symbol}"
        
        if cache_key in self._cache:
            posts, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return posts
        
        keywords = self._search_keywords(symbol)
        
        self._rate_limit()
        
        # Search subreddit
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q": " OR ".join(keywords),
            "restrict_sr": "on",
            "sort": "new",
            "limit": limit,
            "t": "day",  # Past 24 hours
        }
        
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": self.USER_AGENT},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"Reddit fetch failed for r/{subreddit}: {e}")
            return []
        
        posts = []
        
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            
            try:
                title = post_data.get("title", "")
                body = post_data.get("selftext", "")
                
                # Skip if symbol not mentioned
                full_text = f"{title} {body}".upper()
                if not any(kw.upper() in full_text for kw in keywords):
                    continue
                
                # Parse timestamp
                created_utc = post_data.get("created_utc", 0)
                created_at = datetime.fromtimestamp(created_utc)
                
                # Analyze sentiment
                text_for_analysis = f"{title}. {body[:1000]}"
                scores = self.analyzer.polarity_scores(text_for_analysis)
                
                post = SocialPost(
                    title=title,
                    body=body[:500],
                    source=f"reddit:{subreddit}",
                    url=f"https://reddit.com{post_data.get('permalink', '')}",
                    created_at=created_at,
                    symbol=symbol,
                    sentiment_score=scores["compound"],
                    upvotes=post_data.get("ups", 0),
                    comments=post_data.get("num_comments", 0),
                    author=post_data.get("author", ""),
                )
                posts.append(post)
                
            except Exception as e:
                logger.debug(f"Failed to parse post: {e}")
                continue
        
        self._cache[cache_key] = (posts, datetime.now())
        return posts
    
    def fetch_all_posts(
        self,
        symbol: str,
        limit_per_sub: int = 25,
    ) -> List[SocialPost]:
        """
        Fetch posts from all relevant subreddits.
        
        Args:
            symbol: Symbol to search
            limit_per_sub: Max posts per subreddit
            
        Returns:
            Combined list of posts
        """
        subreddits = self._get_subreddits(symbol)
        all_posts = []
        
        for sub in subreddits:
            try:
                posts = self.fetch_subreddit_posts(sub, symbol, limit_per_sub)
                all_posts.extend(posts)
            except Exception as e:
                logger.warning(f"Failed to fetch r/{sub}: {e}")
        
        # Sort by engagement
        all_posts.sort(key=lambda p: p.engagement_score, reverse=True)
        
        return all_posts
    
    def aggregate_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Get aggregated social sentiment metrics.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dict of sentiment metrics
        """
        posts = self.fetch_all_posts(symbol)
        
        if not posts:
            return {
                "social_sentiment": 0.0,
                "social_volume": 0.0,
                "social_engagement": 0.0,
                "wsb_sentiment": 0.0,
                "wsb_mentions": 0,
            }
        
        # Calculate metrics
        scores = [p.sentiment_score for p in posts]
        engagements = [p.engagement_score for p in posts]
        
        # Weighted average by engagement
        weights = np.array(engagements) + 1
        weighted_sentiment = np.average(scores, weights=weights)
        
        # WSB-specific metrics
        wsb_posts = [p for p in posts if "wallstreetbets" in p.source]
        wsb_sentiment = np.mean([p.sentiment_score for p in wsb_posts]) if wsb_posts else 0.0
        
        return {
            "social_sentiment": float(weighted_sentiment),
            "social_volume": min(3.0, len(posts) / 10.0),  # Normalized
            "social_engagement": float(np.mean(engagements)),
            "wsb_sentiment": float(wsb_sentiment),
            "wsb_mentions": len(wsb_posts),
        }


class SocialFeatureGenerator:
    """
    Generate ML features from social sentiment.
    
    Features:
    - social_sentiment: Engagement-weighted sentiment
    - social_volume: Normalized post volume
    - social_momentum: Sentiment change over time
    - wsb_buzz: WallStreetBets activity indicator
    """
    
    def __init__(self, fetcher: Optional[RedditSentimentFetcher] = None):
        self.fetcher = fetcher or RedditSentimentFetcher()
        self._history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    
    def update(self, symbol: str) -> Dict[str, float]:
        """
        Update and get latest social features.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dict of social features
        """
        metrics = self.fetcher.aggregate_sentiment(symbol)
        
        # Store in history
        self._history[symbol].append({
            "timestamp": datetime.now(),
            **metrics,
        })
        
        # Keep last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        self._history[symbol] = [
            h for h in self._history[symbol]
            if h.get("timestamp", datetime.now()) > cutoff
        ]
        
        return metrics
    
    def get_features(self, symbol: str) -> Dict[str, float]:
        """
        Get ML-ready social sentiment features.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dict of features
        """
        current = self.update(symbol)
        history = self._history[symbol]
        
        features = {
            "social_sentiment": current["social_sentiment"],
            "social_volume": current["social_volume"],
            "social_engagement": current["social_engagement"],
        }
        
        # Social momentum
        if len(history) >= 2:
            prev = history[-2].get("social_sentiment", 0)
            features["social_momentum"] = current["social_sentiment"] - prev
        else:
            features["social_momentum"] = 0.0
        
        # WSB buzz indicator
        features["wsb_buzz"] = min(1.0, current["wsb_mentions"] / 10.0)
        
        # Extreme social sentiment
        features["social_extreme"] = 1.0 if abs(current["social_sentiment"]) > 0.5 else 0.0
        
        return features


def get_social_features(symbol: str) -> Dict[str, float]:
    """
    Convenience function to get social features.
    
    Args:
        symbol: Asset symbol
        
    Returns:
        Dict of social sentiment features
    """
    generator = SocialFeatureGenerator()
    return generator.get_features(symbol)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Social Sentiment Module Test ===\n")
    
    fetcher = RedditSentimentFetcher()
    
    # Test with TSLA
    symbol = "TSLA"
    print(f"Fetching Reddit posts for {symbol}...")
    
    posts = fetcher.fetch_all_posts(symbol)
    print(f"Found {len(posts)} posts\n")
    
    # Show sample
    for post in posts[:5]:
        sentiment_label = "🟢" if post.sentiment_score > 0.1 else "🔴" if post.sentiment_score < -0.1 else "⚪"
        print(f"{sentiment_label} [{post.sentiment_score:+.2f}] {post.title[:50]}...")
        print(f"   Source: {post.source}, ⬆️ {post.upvotes}, 💬 {post.comments}")
        print()
    
    # Get aggregated metrics
    metrics = fetcher.aggregate_sentiment(symbol)
    print(f"\n=== Aggregated Social Sentiment for {symbol} ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Get ML features
    generator = SocialFeatureGenerator(fetcher)
    features = generator.get_features(symbol)
    
    print(f"\n=== ML Features for {symbol} ===")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
