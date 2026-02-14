"""
Sentiment Impact Test

Tests the impact of sentiment features on model accuracy.
Compares TSLA model performance with and without sentiment.

Usage:
    python scripts/test_sentiment_impact.py
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_tsla_data(days: int = 365) -> pd.DataFrame:
    """Fetch TSLA historical data."""
    try:
        import yfinance as yf
        
        ticker = yf.Ticker("TSLA")
        df = ticker.history(period=f"{days}d", interval="1h")
        
        if df.empty:
            logger.warning("No data from yfinance, using synthetic")
            return create_synthetic_data(days)
        
        # Rename columns
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        logger.warning(f"yfinance fetch failed: {e}")
        return create_synthetic_data(days)


def create_synthetic_data(days: int = 365) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    n_hours = days * 8  # ~8 market hours per day
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_hours,
        freq="h"
    )
    
    # Random walk with trend
    returns = np.random.randn(n_hours) * 0.02 + 0.0001
    prices = 200 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_hours) * 0.005),
        "high": prices * (1 + abs(np.random.randn(n_hours) * 0.01)),
        "low": prices * (1 - abs(np.random.randn(n_hours) * 0.01)),
        "close": prices,
        "volume": np.random.randint(100000, 1000000, n_hours),
    }, index=dates)
    
    return df


def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic technical features."""
    features = df.copy()
    close = df["close"]
    
    # Returns
    features["return_1"] = close.pct_change(1)
    features["return_5"] = close.pct_change(5)
    features["return_10"] = close.pct_change(10)
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        ma = close.rolling(period).mean()
        features[f"ma_{period}_dist"] = (close - ma) / close
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features["rsi"] = 100 - (100 / (1 + rs))
    features["rsi_norm"] = (features["rsi"] - 50) / 50
    
    # Volatility
    features["volatility_10"] = close.pct_change().rolling(10).std()
    features["volatility_20"] = close.pct_change().rolling(20).std()
    
    # Volume
    features["volume_ma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    
    # Target
    features["target"] = (close.shift(-1) > close).astype(int)
    
    return features.dropna()


def add_sentiment_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add sentiment features to DataFrame."""
    from bot.sentiment.feature_integration import (
        generate_synthetic_sentiment_history,
        CombinedSentimentFeatureGenerator,
    )
    
    # Generate synthetic historical sentiment with some predictive power
    sentiment_df = generate_synthetic_sentiment_history(
        df, symbol, correlation_with_returns=0.2
    )
    
    # Try to get live features for the last row
    try:
        generator = CombinedSentimentFeatureGenerator()
        live_features = generator.get_features(symbol)
        
        # Update last row with live data
        for col in sentiment_df.columns:
            if col in live_features:
                sentiment_df.iloc[-1, sentiment_df.columns.get_loc(col)] = live_features[col]
        
        logger.info(f"Live sentiment for {symbol}: score={live_features.get('sentiment_combined', 0):.3f}")
    except Exception as e:
        logger.warning(f"Could not get live sentiment: {e}")
    
    # Merge
    result = df.copy()
    for col in sentiment_df.columns:
        result[col] = sentiment_df[col].values
    
    return result


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """Train model and return metrics."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def run_comparison():
    """Run model comparison with and without sentiment."""
    print("=" * 60)
    print("SENTIMENT FEATURE IMPACT TEST")
    print("=" * 60)
    print()
    
    # Fetch data
    logger.info("Fetching TSLA data...")
    price_df = fetch_tsla_data(days=180)
    logger.info(f"Got {len(price_df)} rows of data")
    
    # Create technical features
    logger.info("Creating technical features...")
    tech_df = create_technical_features(price_df)
    
    # Add sentiment features
    logger.info("Adding sentiment features...")
    full_df = add_sentiment_features(tech_df, "TSLA")
    
    # Define feature sets
    base_features = [
        "return_1", "return_5", "return_10",
        "ma_5_dist", "ma_10_dist", "ma_20_dist", "ma_50_dist",
        "rsi_norm", "volatility_10", "volatility_20", "volume_ma_ratio",
    ]
    
    sentiment_features = [
        "sentiment_combined", "sentiment_score_24h", "sentiment_momentum",
        "mention_volume", "sentiment_std", "social_sentiment",
        "social_volume", "social_momentum", "wsb_buzz",
        "sentiment_disagreement", "sentiment_extreme", "sentiment_contrarian",
    ]
    
    # Prepare data
    all_features = base_features + sentiment_features
    
    # Ensure all columns exist
    for col in all_features:
        if col not in full_df.columns:
            full_df[col] = 0.0
    
    # Split data (time-based)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:]
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Prepare datasets
    X_train_base = train_df[base_features].values
    X_train_full = train_df[all_features].values
    X_test_base = test_df[base_features].values
    X_test_full = test_df[all_features].values
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    
    # Train models
    print("\n--- Training Models ---\n")
    
    logger.info("Training baseline model (technical only)...")
    baseline_results = train_and_evaluate(
        X_train_base, y_train, X_test_base, y_test,
        "Technical Only"
    )
    
    logger.info("Training enhanced model (technical + sentiment)...")
    enhanced_results = train_and_evaluate(
        X_train_full, y_train, X_test_full, y_test,
        "Technical + Sentiment"
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<15} {'Baseline':<15} {'With Sentiment':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for metric in ["accuracy", "precision", "recall", "f1"]:
        base_val = baseline_results[metric]
        full_val = enhanced_results[metric]
        improvement = (full_val - base_val) * 100
        sign = "+" if improvement >= 0 else ""
        
        print(f"{metric.capitalize():<15} {base_val:.4f}{'':>7} {full_val:.4f}{'':>7} {sign}{improvement:.2f}%")
    
    print()
    
    # Feature importance (for enhanced model)
    print("\n--- Top Features (Enhanced Model) ---\n")
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train_full, y_train)
    
    importance = dict(zip(all_features, model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Feature':<30} {'Importance':<15}")
    print("-" * 45)
    
    for name, imp in sorted_importance[:15]:
        is_sentiment = name in sentiment_features
        marker = "📊" if is_sentiment else "📈"
        print(f"{marker} {name:<28} {imp:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    accuracy_gain = (enhanced_results["accuracy"] - baseline_results["accuracy"]) * 100
    
    if accuracy_gain > 0:
        print(f"\n✅ Sentiment features IMPROVED accuracy by {accuracy_gain:.2f}%")
    elif accuracy_gain < 0:
        print(f"\n⚠️ Sentiment features DECREASED accuracy by {abs(accuracy_gain):.2f}%")
    else:
        print(f"\n➡️ Sentiment features had NO IMPACT on accuracy")
    
    # Count sentiment features in top 10
    top_10 = [name for name, _ in sorted_importance[:10]]
    sentiment_in_top_10 = sum(1 for f in top_10 if f in sentiment_features)
    
    print(f"\n📊 {sentiment_in_top_10} sentiment features in top 10 important features")
    
    # Sentiment features summary
    sentiment_importance = sum(
        imp for name, imp in sorted_importance if name in sentiment_features
    )
    print(f"📊 Total sentiment feature importance: {sentiment_importance:.2%}")
    
    return baseline_results, enhanced_results


def test_live_sentiment():
    """Test live sentiment fetching."""
    print("\n" + "=" * 60)
    print("LIVE SENTIMENT TEST")
    print("=" * 60 + "\n")
    
    try:
        from bot.sentiment import (
            NewsSentimentFetcher,
            RedditSentimentFetcher,
            get_combined_features,
        )
        
        symbol = "TSLA"
        
        # News sentiment
        print(f"📰 Fetching news sentiment for {symbol}...")
        try:
            news_fetcher = NewsSentimentFetcher()
            news_items = news_fetcher.fetch_news(symbol, hours_back=24)
            snapshot = news_fetcher.aggregate_sentiment(symbol)
            
            print(f"   Found {len(news_items)} news items")
            print(f"   News sentiment score: {snapshot.sentiment_score:+.3f}")
            
            if news_items:
                print(f"\n   Recent headlines:")
                for item in news_items[:3]:
                    emoji = "🟢" if item.sentiment_score > 0.1 else "🔴" if item.sentiment_score < -0.1 else "⚪"
                    print(f"   {emoji} [{item.sentiment_score:+.2f}] {item.title[:50]}...")
        except Exception as e:
            print(f"   ❌ News fetch failed: {e}")
        
        # Social sentiment
        print(f"\n🐦 Fetching social sentiment for {symbol}...")
        try:
            reddit_fetcher = RedditSentimentFetcher()
            metrics = reddit_fetcher.aggregate_sentiment(symbol)
            
            print(f"   Social sentiment: {metrics['social_sentiment']:+.3f}")
            print(f"   WSB mentions: {metrics['wsb_mentions']}")
        except Exception as e:
            print(f"   ❌ Social fetch failed: {e}")
        
        # Combined features
        print(f"\n🔗 Getting combined features for {symbol}...")
        try:
            features = get_combined_features(symbol)
            
            print(f"\n   {'Feature':<25} {'Value':>10}")
            print("   " + "-" * 37)
            for name, value in features.items():
                print(f"   {name:<25} {value:>10.4f}")
        except Exception as e:
            print(f"   ❌ Combined features failed: {e}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the project root and venv is activated")


if __name__ == "__main__":
    # Test live sentiment first
    test_live_sentiment()
    
    # Run comparison
    print()
    run_comparison()
