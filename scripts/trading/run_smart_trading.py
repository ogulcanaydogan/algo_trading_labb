#!/usr/bin/env python3
"""
Smart Trading Runner - Uses ML, Market Regime Detection, and Strategy Selection.

This script demonstrates the full AI-enhanced trading pipeline:
1. Fetch market data
2. Detect market regime (Bull/Bear/Sideways/Volatile)
3. Select optimal strategy for current conditions
4. Generate ML-enhanced predictions
5. Execute trades with proper risk management
6. Get LLM-powered analysis and suggestions

Usage:
    python run_smart_trading.py [--symbol BTC/USDT] [--train] [--backtest]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load unified configuration
from bot.config import load_config

app_config = load_config()

from bot.strategy import StrategyConfig
from bot.backtesting import Backtester, BacktestResult

# New ML modules
from bot.ml import MLPredictor, MarketRegimeClassifier, FeatureEngineer

# New strategy library
from bot.strategies import StrategySelector

# LLM advisor (optional)
from bot.llm import LLMAdvisor, PerformanceAnalyzer


def load_market_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Load market data from yfinance or ccxt."""
    try:
        import yfinance as yf

        # Convert crypto symbols for yfinance
        yf_symbol = symbol.replace("/", "-")
        if "USDT" in symbol:
            yf_symbol = symbol.split("/")[0] + "-USD"

        print(f"Fetching {days} days of data for {yf_symbol}...")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d", interval="1h")

        if df.empty:
            raise ValueError(f"No data returned for {yf_symbol}")

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"stock splits": "splits"})

        return df[["open", "high", "low", "close", "volume"]]

    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data(days * 24)


def generate_synthetic_data(num_bars: int = 8760) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    import numpy as np

    np.random.seed(42)

    # Generate random walk price
    returns = np.random.normal(0.0002, 0.02, num_bars)
    close = 50000 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, num_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, num_bars)))
    open_price = close * (1 + np.random.normal(0, 0.005, num_bars))
    volume = np.random.uniform(100, 10000, num_bars)

    # Create datetime index
    dates = pd.date_range(end=datetime.now(), periods=num_bars, freq="1h")

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


def run_regime_analysis(df: pd.DataFrame) -> None:
    """Analyze and display market regime."""
    print("\n" + "="*60)
    print("MARKET REGIME ANALYSIS")
    print("="*60)

    classifier = MarketRegimeClassifier()
    analysis = classifier.classify(df)

    print(f"Current Regime: {analysis.regime.value.upper()}")
    print(f"Confidence: {analysis.confidence:.1%}")
    print(f"Trend Strength: {analysis.trend_strength:+.2f}")
    print(f"Volatility: {analysis.volatility_level} ({analysis.volatility_percentile:.0f}%ile)")
    print(f"ADX: {analysis.adx_value:.1f}")
    print(f"Momentum: {analysis.momentum_score:+.2f}")
    print(f"Support: ${analysis.support_level:,.2f}")
    print(f"Resistance: ${analysis.resistance_level:,.2f}")
    print(f"Regime Duration: {analysis.regime_duration} bars")
    print(f"\nRecommended Strategy: {analysis.recommended_strategy}")
    print(f"\nReasoning:")
    for reason in analysis.reasoning:
        print(f"  - {reason}")

    # Get strategy parameters for this regime
    params = classifier.get_strategy_parameters(analysis.regime)
    print(f"\nSuggested Adjustments:")
    print(f"  Position Size: {params['position_size_multiplier']:.1f}x")
    print(f"  Stop Loss: {params['stop_loss_multiplier']:.1f}x")
    print(f"  Take Profit: {params['take_profit_multiplier']:.1f}x")


def run_strategy_selection(df: pd.DataFrame) -> None:
    """Run strategy selector and generate combined signal."""
    print("\n" + "="*60)
    print("STRATEGY SELECTION & SIGNAL GENERATION")
    print("="*60)

    selector = StrategySelector(use_multi_strategy=True)
    result = selector.select_and_generate(df)

    print(f"Selected Strategy: {result.selected_strategy}")
    print(f"Market Regime: {result.regime_analysis.regime.value}")
    print(f"\nPrimary Signal:")
    print(f"  Decision: {result.primary_signal.decision}")
    print(f"  Confidence: {result.primary_signal.confidence:.1%}")
    print(f"  Reason: {result.primary_signal.reason}")

    if result.primary_signal.entry_price:
        print(f"  Entry: ${result.primary_signal.entry_price:,.2f}")
        print(f"  Stop Loss: ${result.primary_signal.stop_loss:,.2f}")
        print(f"  Take Profit: ${result.primary_signal.take_profit:,.2f}")

    print(f"\nStrategy Agreement: {result.agreement_score:.1%}")
    print(f"Supporting Strategies: {', '.join(result.supporting_strategies) or 'None'}")
    print(f"Final Confidence: {result.final_confidence:.1%}")
    print(f"Position Size Multiplier: {result.position_size_multiplier:.2f}x")


def train_ml_model(df: pd.DataFrame) -> MLPredictor:
    """Train the ML predictor on historical data."""
    print("\n" + "="*60)
    print("ML MODEL TRAINING")
    print("="*60)

    predictor = MLPredictor(model_type="random_forest")

    print("Training Random Forest model...")
    metrics = predictor.train(df, test_size=0.2)

    print(f"\nTraining Results:")
    print(f"  Accuracy: {metrics.accuracy:.1%}")
    print(f"  Cross-Val: {metrics.cross_val_mean:.1%} +/- {metrics.cross_val_std:.1%}")
    print(f"  Train Samples: {metrics.train_samples}")
    print(f"  Test Samples: {metrics.test_samples}")

    # Show top features
    top_features = predictor.get_feature_importance(top_n=10)
    if top_features:
        print(f"\nTop 10 Features:")
        for name, importance in top_features:
            print(f"  {name}: {importance:.4f}")

    # Save model
    predictor.save("smart_predictor")
    print("\nModel saved to data/models/")

    return predictor


def run_ml_prediction(predictor: MLPredictor, df: pd.DataFrame) -> None:
    """Generate ML prediction."""
    print("\n" + "="*60)
    print("ML PREDICTION")
    print("="*60)

    prediction = predictor.predict(df)

    print(f"Recommended Action: {prediction.action}")
    print(f"Confidence: {prediction.confidence:.1%}")
    print(f"\nProbabilities:")
    print(f"  LONG:  {prediction.probability_long:.1%}")
    print(f"  SHORT: {prediction.probability_short:.1%}")
    print(f"  FLAT:  {prediction.probability_flat:.1%}")
    print(f"\nExpected Return: {prediction.expected_return:.4%}")
    print(f"Features Used: {prediction.features_used}")
    print(f"Model: {prediction.model_type}")


def run_backtest_with_costs(df: pd.DataFrame) -> BacktestResult:
    """Run backtest with realistic costs."""
    print("\n" + "="*60)
    print("BACKTEST WITH REALISTIC COSTS")
    print("="*60)

    # Use unified config for strategy parameters
    default_symbol = app_config.crypto.symbols[0] if app_config.crypto.symbols else "BTC/USDT"
    config = StrategyConfig(
        symbol=default_symbol,
        timeframe=app_config.strategy.timeframe,
        ema_fast=app_config.strategy.ema_fast,
        ema_slow=app_config.strategy.ema_slow,
        risk_per_trade_pct=app_config.trading.risk_per_trade_pct,
        stop_loss_pct=app_config.trading.stop_loss_pct,
        take_profit_pct=app_config.trading.take_profit_pct,
    )

    backtester = Backtester(
        strategy_config=config,
        initial_balance=app_config.trading.initial_capital,
        commission_pct=0.1,   # 0.1% per trade
        slippage_pct=0.05,    # 0.05% slippage
    )

    result = backtester.run(df)
    result.print_summary()

    return result


def run_llm_analysis(result: BacktestResult) -> None:
    """Get LLM-powered analysis (if available)."""
    print("\n" + "="*60)
    print("LLM-POWERED ANALYSIS")
    print("="*60)

    advisor = LLMAdvisor(model="llama3")

    if not advisor.is_available():
        print("Ollama not available. Using rule-based analysis...")
        print("\nTo enable LLM analysis:")
        print("  1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  2. Pull a model: ollama pull llama3")
        print("  3. Run again")

    # Get advice (falls back to rule-based if LLM unavailable)
    metrics = {
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_trades": result.total_trades,
    }

    advice = advisor.get_strategy_advice(
        symbol="BTC/USDT",
        timeframe="1h",
        regime="unknown",
        metrics=metrics,
        current_strategy="ema_crossover",
        recent_trades=[t.to_dict() for t in result.trades[-10:]],
    )

    print(f"\nAssessment:")
    print(f"  {advice.assessment[:500]}...")

    print(f"\nParameter Suggestions:")
    for param, suggestion in advice.parameter_suggestions.items():
        print(f"  {param}: {suggestion}")

    if advice.alternative_strategies:
        print(f"\nAlternative Strategies:")
        for strategy in advice.alternative_strategies:
            print(f"  - {strategy}")

    if advice.risk_recommendations:
        print(f"\nRisk Recommendations:")
        for rec in advice.risk_recommendations[:3]:
            print(f"  - {rec}")

    print(f"\nAdvice Confidence: {advice.confidence:.1%}")


def run_performance_analysis(result: BacktestResult) -> None:
    """Run comprehensive performance analysis."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*60)

    analyzer = PerformanceAnalyzer()
    trades = [t.to_dict() for t in result.trades]
    report = analyzer.analyze(trades, result.equity_curve, "ema_crossover")

    print(f"\nSummary: {report.summary}")

    print(f"\nStrategy Scores:")
    for dimension, score in report.strategy_scores.items():
        print(f"  {dimension}: {score}/100")

    print(f"\nStrengths:")
    for s in report.strengths:
        print(f"  + {s}")

    print(f"\nWeaknesses:")
    for w in report.weaknesses:
        print(f"  - {w}")

    print(f"\n{report.risk_assessment}")

    print(f"\nImprovement Priority:")
    for i, p in enumerate(report.improvement_priority, 1):
        print(f"  {i}. {p}")


def main():
    # Get default symbol from config
    default_symbol = app_config.crypto.symbols[0] if app_config.crypto.symbols else "BTC/USDT"

    parser = argparse.ArgumentParser(description="Smart Trading with ML and Strategy Selection")
    parser.add_argument("--symbol", default=default_symbol, help=f"Trading symbol (default: {default_symbol})")
    parser.add_argument("--days", type=int, default=180, help="Days of historical data")
    parser.add_argument("--train", action="store_true", help="Train ML model")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    print("="*60)
    print("SMART TRADING SYSTEM")
    print(f"Symbol: {args.symbol}")
    print(f"Data: {args.days} days")
    print("="*60)

    # Load data
    df = load_market_data(args.symbol, args.days)
    print(f"\nLoaded {len(df)} bars of data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Current price: ${df['close'].iloc[-1]:,.2f}")

    # Run analyses
    run_regime_analysis(df)
    run_strategy_selection(df)

    if args.train or args.all:
        predictor = train_ml_model(df)
        run_ml_prediction(predictor, df)

    if args.backtest or args.all:
        result = run_backtest_with_costs(df)
        run_performance_analysis(result)
        run_llm_analysis(result)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
