"""
Sentiment Analysis Module

Provides news and social sentiment signals as trading features.

Features:
- News sentiment from Google News, Yahoo Finance, Finviz
- Social sentiment from Reddit (WSB, stocks, crypto subs)
- Combined ML features for model integration

Usage:
    from bot.sentiment import get_combined_features
    
    features = get_combined_features("TSLA")
    # Returns dict with sentiment_score_24h, sentiment_momentum, etc.
"""

from .news_sentiment import (
    NewsSentimentFetcher,
    SentimentFeatureGenerator,
    get_sentiment_features,
)

from .social_sentiment import (
    RedditSentimentFetcher,
    SocialFeatureGenerator,
    get_social_features,
)

from .feature_integration import (
    CombinedSentimentFeatureGenerator,
    SentimentConfig,
    integrate_sentiment_features,
    generate_synthetic_sentiment_history,
)


def get_combined_features(symbol: str) -> dict:
    """
    Get combined sentiment features for a symbol.
    
    Args:
        symbol: Asset symbol (e.g., "TSLA", "BTC/USD")
        
    Returns:
        Dict with all sentiment features
    """
    generator = CombinedSentimentFeatureGenerator()
    return generator.get_features(symbol)


__all__ = [
    # News sentiment
    "NewsSentimentFetcher",
    "SentimentFeatureGenerator",
    "get_sentiment_features",
    # Social sentiment
    "RedditSentimentFetcher",
    "SocialFeatureGenerator",
    "get_social_features",
    # Combined features
    "CombinedSentimentFeatureGenerator",
    "SentimentConfig",
    "integrate_sentiment_features",
    "generate_synthetic_sentiment_history",
    "get_combined_features",
]
