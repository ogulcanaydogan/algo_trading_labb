"""
Sentiment Feature Integration

Combines news and social sentiment into ML features
for integration with the trading model.

This module provides:
1. Combined sentiment features from multiple sources
2. Integration with the ML feature engineering pipeline
3. Historical sentiment feature generation for backtesting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import sentiment modules
try:
    from .news_sentiment import NewsSentimentFetcher, SentimentFeatureGenerator
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False
    logger.warning("News sentiment module not available")

try:
    from .social_sentiment import RedditSentimentFetcher, SocialFeatureGenerator
    HAS_SOCIAL = True
except ImportError:
    HAS_SOCIAL = False
    logger.warning("Social sentiment module not available")


@dataclass
class SentimentConfig:
    """Configuration for sentiment features."""
    # Feature weights
    news_weight: float = 0.6
    social_weight: float = 0.4
    
    # Update frequency
    update_interval_minutes: int = 30
    
    # Feature options
    include_momentum: bool = True
    include_volume: bool = True
    include_contrarian: bool = True
    include_wsb: bool = True
    
    # Lookback periods
    short_window_hours: int = 6
    long_window_hours: int = 24


class CombinedSentimentFeatureGenerator:
    """
    Generates combined sentiment features from news and social sources.
    
    Features:
    - sentiment_combined: Weighted average of news + social
    - sentiment_score_24h: News sentiment (rolling 24h)
    - sentiment_momentum: Change in sentiment
    - mention_volume: Normalized buzz indicator
    - social_sentiment: Social media sentiment
    - social_volume: Social post volume
    - wsb_buzz: WallStreetBets activity (if applicable)
    - sentiment_disagreement: News vs social divergence
    """
    
    FEATURE_NAMES = [
        "sentiment_combined",
        "sentiment_score_24h",
        "sentiment_momentum",
        "mention_volume",
        "sentiment_std",
        "social_sentiment",
        "social_volume",
        "social_momentum",
        "wsb_buzz",
        "sentiment_disagreement",
        "sentiment_extreme",
        "sentiment_contrarian",
    ]
    
    def __init__(self, config: Optional[SentimentConfig] = None):
        """
        Initialize combined feature generator.
        
        Args:
            config: Optional configuration
        """
        self.config = config or SentimentConfig()
        
        self._news_gen: Optional[SentimentFeatureGenerator] = None
        self._social_gen: Optional[SocialFeatureGenerator] = None
        
        if HAS_NEWS:
            self._news_gen = SentimentFeatureGenerator()
        
        if HAS_SOCIAL:
            self._social_gen = SocialFeatureGenerator()
        
        self._last_update: Dict[str, datetime] = {}
        self._cache: Dict[str, Dict[str, float]] = {}
    
    def _should_update(self, symbol: str) -> bool:
        """Check if we should fetch new data."""
        if symbol not in self._last_update:
            return True
        
        elapsed = datetime.now() - self._last_update[symbol]
        return elapsed > timedelta(minutes=self.config.update_interval_minutes)
    
    def get_features(self, symbol: str, force_update: bool = False) -> Dict[str, float]:
        """
        Get combined sentiment features for a symbol.
        
        Args:
            symbol: Asset symbol
            force_update: Force refresh even if cached
            
        Returns:
            Dict of feature name -> value
        """
        # Check cache
        if not force_update and not self._should_update(symbol):
            if symbol in self._cache:
                return self._cache[symbol]
        
        features = {}
        news_features = {}
        social_features = {}
        
        # Get news features
        if self._news_gen:
            try:
                news_features = self._news_gen.get_features(symbol)
            except Exception as e:
                logger.warning(f"Failed to get news features: {e}")
        
        # Get social features (with timeout protection)
        if self._social_gen:
            try:
                import signal
                import threading
                
                result = [{}]
                def fetch_social():
                    try:
                        result[0] = self._social_gen.get_features(symbol)
                    except Exception as e:
                        logger.warning(f"Social fetch error: {e}")
                
                thread = threading.Thread(target=fetch_social)
                thread.daemon = True
                thread.start()
                thread.join(timeout=10)  # 10 second timeout
                
                if thread.is_alive():
                    logger.warning("Social sentiment fetch timed out")
                else:
                    social_features = result[0]
                    
            except Exception as e:
                logger.warning(f"Failed to get social features: {e}")
        
        # News features
        features["sentiment_score_24h"] = news_features.get("sentiment_score_24h", 0.0)
        features["sentiment_momentum"] = news_features.get("sentiment_momentum", 0.0)
        features["mention_volume"] = news_features.get("mention_volume", 0.0)
        features["sentiment_std"] = news_features.get("sentiment_std", 0.0)
        
        # Social features
        features["social_sentiment"] = social_features.get("social_sentiment", 0.0)
        features["social_volume"] = social_features.get("social_volume", 0.0)
        features["social_momentum"] = social_features.get("social_momentum", 0.0)
        features["wsb_buzz"] = social_features.get("wsb_buzz", 0.0)
        
        # Combined sentiment (weighted average)
        news_score = features["sentiment_score_24h"]
        social_score = features["social_sentiment"]
        
        # Only combine if both have valid data
        if news_features and social_features:
            features["sentiment_combined"] = (
                self.config.news_weight * news_score +
                self.config.social_weight * social_score
            )
        elif news_features:
            features["sentiment_combined"] = news_score
        elif social_features:
            features["sentiment_combined"] = social_score
        else:
            features["sentiment_combined"] = 0.0
        
        # Disagreement (divergence between news and social)
        features["sentiment_disagreement"] = abs(news_score - social_score)
        
        # Extreme sentiment indicator
        features["sentiment_extreme"] = (
            1.0 if abs(features["sentiment_combined"]) > 0.5 else 0.0
        )
        
        # Contrarian signal
        combined = features["sentiment_combined"]
        if combined > 0.6:
            features["sentiment_contrarian"] = -0.5
        elif combined < -0.6:
            features["sentiment_contrarian"] = 0.5
        else:
            features["sentiment_contrarian"] = 0.0
        
        # Update cache
        self._cache[symbol] = features
        self._last_update[symbol] = datetime.now()
        
        return features
    
    def get_features_df(
        self,
        symbol: str,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get sentiment features aligned to price DataFrame.
        
        For real-time, fills forward the latest features.
        For backtesting, uses synthetic historical data.
        
        Args:
            symbol: Asset symbol
            price_df: Price DataFrame with datetime index
            
        Returns:
            DataFrame with sentiment features
        """
        # Get current features
        features = self.get_features(symbol)
        
        # Create aligned DataFrame
        df = pd.DataFrame(index=price_df.index)
        
        for name in self.FEATURE_NAMES:
            df[name] = features.get(name, 0.0)
        
        return df
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of feature names."""
        return CombinedSentimentFeatureGenerator.FEATURE_NAMES.copy()


def generate_synthetic_sentiment_history(
    price_df: pd.DataFrame,
    symbol: str,
    correlation_with_returns: float = 0.3,
) -> pd.DataFrame:
    """
    Generate synthetic historical sentiment for backtesting.
    
    This creates sentiment features that have a configurable
    correlation with price returns, useful for testing.
    
    Args:
        price_df: Price DataFrame with close column
        symbol: Asset symbol
        correlation_with_returns: Target correlation
        
    Returns:
        DataFrame with synthetic sentiment features
    """
    n = len(price_df)
    
    # Calculate returns
    returns = price_df["close"].pct_change().fillna(0)
    
    # Generate correlated noise
    noise = np.random.randn(n)
    
    # Create correlated sentiment
    sentiment_base = (
        correlation_with_returns * returns.values / (returns.std() + 1e-8) +
        np.sqrt(1 - correlation_with_returns**2) * noise
    )
    
    # Normalize to [-1, 1]
    sentiment_base = np.clip(sentiment_base / 3, -1, 1)
    
    # Create DataFrame
    df = pd.DataFrame(index=price_df.index)
    
    # Sentiment features
    df["sentiment_combined"] = sentiment_base
    df["sentiment_score_24h"] = sentiment_base * 0.8 + np.random.randn(n) * 0.1
    df["sentiment_score_24h"] = df["sentiment_score_24h"].clip(-1, 1)
    
    # Momentum (lagged difference)
    df["sentiment_momentum"] = df["sentiment_combined"].diff().fillna(0)
    
    # Volume (correlated with volatility)
    volatility = returns.rolling(10).std().fillna(0)
    df["mention_volume"] = (volatility / volatility.mean()).clip(0, 3).fillna(1.0)
    
    # Standard deviation
    df["sentiment_std"] = np.random.uniform(0.1, 0.4, n)
    
    # Social features
    df["social_sentiment"] = sentiment_base * 0.7 + np.random.randn(n) * 0.15
    df["social_sentiment"] = df["social_sentiment"].clip(-1, 1)
    df["social_volume"] = df["mention_volume"] * 0.8 + np.random.uniform(0, 0.3, n)
    df["social_momentum"] = df["social_sentiment"].diff().fillna(0)
    
    # WSB buzz (spiky, correlated with volume)
    df["wsb_buzz"] = np.random.exponential(0.2, n).clip(0, 1)
    
    # Disagreement
    df["sentiment_disagreement"] = abs(
        df["sentiment_score_24h"] - df["social_sentiment"]
    )
    
    # Extreme indicator
    df["sentiment_extreme"] = (abs(df["sentiment_combined"]) > 0.5).astype(float)
    
    # Contrarian
    df["sentiment_contrarian"] = np.where(
        df["sentiment_combined"] > 0.6, -0.5,
        np.where(df["sentiment_combined"] < -0.6, 0.5, 0.0)
    )
    
    return df


def integrate_sentiment_features(
    price_df: pd.DataFrame,
    symbol: str,
    use_live: bool = False,
    synthetic_correlation: float = 0.3,
) -> pd.DataFrame:
    """
    Add sentiment features to a price DataFrame.
    
    Args:
        price_df: Price DataFrame with OHLCV
        symbol: Asset symbol
        use_live: If True, fetch live data (for last row only)
        synthetic_correlation: Correlation for synthetic data
        
    Returns:
        DataFrame with sentiment features added
    """
    df = price_df.copy()
    
    if use_live:
        # Get live features for the last row
        generator = CombinedSentimentFeatureGenerator()
        features = generator.get_features(symbol)
        
        # Use synthetic for historical, live for latest
        sentiment_df = generate_synthetic_sentiment_history(
            df.iloc[:-1], symbol, synthetic_correlation
        )
        
        # Add live features for last row
        last_row = {name: features.get(name, 0.0) for name in generator.FEATURE_NAMES}
        sentiment_df = pd.concat([
            sentiment_df,
            pd.DataFrame([last_row], index=[df.index[-1]])
        ])
    else:
        # All synthetic for backtesting
        sentiment_df = generate_synthetic_sentiment_history(
            df, symbol, synthetic_correlation
        )
    
    # Merge
    for col in sentiment_df.columns:
        df[col] = sentiment_df[col].values
    
    return df


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Sentiment Feature Integration Test ===\n")
    
    # Create sample price data
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    price_df = pd.DataFrame({
        "open": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "high": 101 + np.cumsum(np.random.randn(100) * 0.5),
        "low": 99 + np.cumsum(np.random.randn(100) * 0.5),
        "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "volume": np.random.randint(1000, 10000, 100),
    }, index=dates)
    
    # Generate synthetic sentiment
    sentiment_df = generate_synthetic_sentiment_history(price_df, "TSLA")
    
    print("Synthetic sentiment features:")
    print(sentiment_df.tail())
    print()
    
    # Test live features
    if HAS_NEWS or HAS_SOCIAL:
        generator = CombinedSentimentFeatureGenerator()
        features = generator.get_features("TSLA")
        
        print("\nLive sentiment features for TSLA:")
        for name, value in features.items():
            print(f"  {name}: {value:.4f}")
    
    # Test integration
    integrated = integrate_sentiment_features(price_df, "TSLA", use_live=False)
    print(f"\nIntegrated DataFrame shape: {integrated.shape}")
    print(f"New columns: {[c for c in integrated.columns if 'sentiment' in c or 'social' in c or 'wsb' in c]}")
