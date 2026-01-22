#!/usr/bin/env python3
"""
Apply Accuracy Improvements to All Models

Runs the accuracy improvement pipeline on all trained models
to maximize prediction accuracy.
"""

import asyncio
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.ml.accuracy_improver import AccuracyImprover, improve_model_accuracy
from bot.ml.walk_forward import run_walk_forward_analysis
from bot.ml.rl_optimizer import RLTradingOptimizer
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/training")
MODEL_DIR = Path("data/models")


def load_training_data(symbol: str) -> tuple:
    """Load training data for a symbol."""
    symbol_clean = symbol.replace("/", "_")
    data_file = DATA_DIR / f"{symbol_clean}_extended.parquet"

    if not data_file.exists():
        logger.warning(f"No data file for {symbol}")
        return None, None

    df = pd.read_parquet(data_file)

    # Add features
    from scripts.ml.comprehensive_train import EnhancedFeatureEngineer
    engineer = EnhancedFeatureEngineer()
    df = engineer.add_enhanced_features(df)

    # Create target
    future_return = df['close'].shift(-24) / df['close'] - 1
    df['target'] = 0
    df.loc[future_return > 0.01, 'target'] = 1
    df.loc[future_return < -0.01, 'target'] = -1

    df = df.dropna()

    # Select features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'returns', 'log_returns']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df['target']
    prices = df['close']

    return X, y, prices


def apply_improvements_to_symbol(symbol: str):
    """Apply all improvements to a single symbol."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Improving models for {symbol}")
    logger.info("=" * 60)

    # Load data
    result = load_training_data(symbol)
    if result[0] is None:
        return

    X, y, prices = result

    if len(X) < 1000:
        logger.warning(f"Insufficient data for {symbol}")
        return

    # 1. Apply accuracy improvement
    logger.info("\n--- Applying Accuracy Improvement Pipeline ---")
    improver = AccuracyImprover(
        target_accuracy=0.65,
        use_feature_selection=True,
        use_class_balancing=True,
        use_ensemble=True,
        use_calibration=True,
        use_threshold_optimization=True
    )

    try:
        improved_model, result = improver.improve(X, y)

        # Save improved model
        symbol_clean = symbol.replace("/", "_")
        model_path = MODEL_DIR / f"{symbol_clean}_improved_model.pkl"
        joblib.dump(improved_model, model_path)

        # Save metadata
        import json
        meta = {
            'symbol': symbol,
            'original_accuracy': result.original_accuracy,
            'improved_accuracy': result.improved_accuracy,
            'improvement_pct': result.improvement_pct,
            'best_technique': result.best_technique,
            'feature_count': result.feature_count,
            'calibrated': result.calibrated,
            'ensemble_used': result.ensemble_used
        }
        meta_path = MODEL_DIR / f"{symbol_clean}_improved_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved improved model: {result.improved_accuracy:.2%} accuracy")
    except Exception as e:
        logger.error(f"Accuracy improvement failed: {e}")

    # 2. Run walk-forward analysis
    logger.info("\n--- Running Walk-Forward Analysis ---")
    try:
        base_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        wf_summary = run_walk_forward_analysis(
            symbol, X, y, base_model, prices,
            train_size=5000, test_size=500
        )

        # Save walk-forward results
        wf_meta = {
            'symbol': symbol,
            'n_folds': wf_summary.n_folds,
            'mean_accuracy': wf_summary.mean_accuracy,
            'std_accuracy': wf_summary.std_accuracy,
            'sharpe_ratio': wf_summary.sharpe_ratio,
            'max_drawdown': wf_summary.max_drawdown,
            'win_rate': wf_summary.win_rate,
            'profit_factor': wf_summary.profit_factor
        }
        wf_path = MODEL_DIR / f"{symbol_clean}_walkforward_meta.json"
        with open(wf_path, 'w') as f:
            json.dump(wf_meta, f, indent=2)
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}")

    # 3. Train RL optimizer
    logger.info("\n--- Training RL Optimizer ---")
    try:
        rl_optimizer = RLTradingOptimizer(state_dim=min(X.shape[1], 50))

        # Use subset of features for RL
        X_rl = X.iloc[:, :min(50, X.shape[1])]

        history = rl_optimizer.train(
            prices, X_rl,
            episodes=50,
            initial_capital=10000
        )

        # Evaluate
        eval_results = rl_optimizer.evaluate(prices, X_rl)

        logger.info(f"RL Results: Return={eval_results['total_return']:.2%}, "
                   f"Buy&Hold={eval_results['buy_hold_return']:.2%}, "
                   f"Outperformance={eval_results['outperformance']:.2%}")

        # Save RL results
        rl_meta = {
            'symbol': symbol,
            'total_return': eval_results['total_return'],
            'buy_hold_return': eval_results['buy_hold_return'],
            'outperformance': eval_results['outperformance'],
            'num_trades': eval_results['num_trades']
        }
        rl_path = MODEL_DIR / f"{symbol_clean}_rl_meta.json"
        with open(rl_path, 'w') as f:
            json.dump(rl_meta, f, indent=2)
    except Exception as e:
        logger.error(f"RL training failed: {e}")


def main():
    """Apply improvements to all symbols."""
    logger.info("=" * 60)
    logger.info("APPLYING IMPROVEMENTS TO ALL MODELS")
    logger.info("=" * 60)

    # Find all symbols with extended data
    symbols = []
    for f in DATA_DIR.glob("*_extended.parquet"):
        symbol = f.stem.replace("_extended", "").replace("_", "/")
        symbols.append(symbol)

    logger.info(f"Found {len(symbols)} symbols with data")

    for symbol in symbols:
        try:
            apply_improvements_to_symbol(symbol)
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("ALL IMPROVEMENTS APPLIED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
