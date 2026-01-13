#!/usr/bin/env python3
"""
ML Model Training Script.

Train machine learning models on fetched historical data.
Supports XGBoost, RandomForest, and GradientBoosting models.

Usage:
    # Train all models on all available data
    python run_ml_training.py

    # Train specific model type
    python run_ml_training.py --model xgboost

    # Train on specific symbol
    python run_ml_training.py --symbol BTC/USDT

    # Backtest after training
    python run_ml_training.py --backtest
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.ml.predictor import MLPredictor, ModelMetrics
from bot.ml.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_training_data(data_dir: str = "data/training") -> Dict[str, pd.DataFrame]:
    """Load all available training data."""
    data_path = Path(data_dir)
    datasets = {}

    if not data_path.exists():
        logger.error(f"Training data directory not found: {data_dir}")
        return datasets

    for file in data_path.glob("*.parquet"):
        if file.name == "combined_training.parquet":
            continue

        try:
            df = pd.read_parquet(file)

            # Parse symbol from filename
            parts = file.stem.rsplit("_", 1)
            symbol = parts[0].replace("_", "/") if len(parts) >= 1 else file.stem
            timeframe = parts[1] if len(parts) == 2 else "1h"

            key = f"{symbol}_{timeframe}"
            datasets[key] = df
            logger.info(f"Loaded {key}: {len(df):,} records")

        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    return datasets


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for ML training."""
    # Ensure required columns exist
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create datetime index
    ohlcv = df.copy()
    if "datetime" not in ohlcv.columns:
        ohlcv["datetime"] = pd.to_datetime(ohlcv["timestamp"], unit="ms")

    ohlcv = ohlcv.set_index("datetime")
    ohlcv = ohlcv.sort_index()

    # Keep only OHLCV columns
    ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]

    return ohlcv


def train_model(
    ohlcv: pd.DataFrame,
    model_type: str = "random_forest",
    model_name: str = "default",
    test_size: float = 0.2,
) -> Dict:
    """Train a single ML model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()} model: {model_name}")
    logger.info(f"{'='*60}")

    try:
        predictor = MLPredictor(
            model_type=model_type,
            model_dir=f"data/models/{model_name}",
        )

        # Train
        metrics = predictor.train(ohlcv, test_size=test_size)

        # Save model
        predictor.save(model_name)

        # Get feature importance
        top_features = predictor.get_feature_importance(top_n=10)

        result = {
            "status": "success",
            "model_type": model_type,
            "model_name": model_name,
            "metrics": {
                "accuracy": round(metrics.accuracy, 4),
                "cross_val_mean": round(metrics.cross_val_mean, 4),
                "cross_val_std": round(metrics.cross_val_std, 4),
                "train_samples": metrics.train_samples,
                "test_samples": metrics.test_samples,
            },
            "top_features": [
                {"feature": f, "importance": round(imp, 4)}
                for f, imp in top_features
            ],
        }

        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Cross-validation: {metrics.cross_val_mean:.4f} +/- {metrics.cross_val_std:.4f}")
        logger.info(f"Training samples: {metrics.train_samples:,}")
        logger.info(f"Test samples: {metrics.test_samples:,}")

        return result

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {
            "status": "error",
            "model_type": model_type,
            "model_name": model_name,
            "error": str(e),
        }


def backtest_model(
    ohlcv: pd.DataFrame,
    model_name: str,
    model_type: str = "random_forest",
    initial_balance: float = 10000.0,
) -> Dict:
    """Backtest a trained model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Backtesting {model_name}")
    logger.info(f"{'='*60}")

    try:
        predictor = MLPredictor(
            model_type=model_type,
            model_dir=f"data/models/{model_name}",
        )

        # Load model
        if not predictor.load(model_name):
            return {"status": "error", "error": "Model not found"}

        # Run backtest
        results = predictor.backtest_predictions(ohlcv, initial_balance)

        logger.info(f"Initial Balance: ${results['initial_balance']:,.2f}")
        logger.info(f"Final Balance: ${results['final_balance']:,.2f}")
        logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")

        return {"status": "success", **results}

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {"status": "error", "error": str(e)}


def train_all_models(
    datasets: Dict[str, pd.DataFrame],
    model_types: List[str] = None,
    run_backtest: bool = False,
) -> Dict:
    """Train all model types on all datasets."""
    if model_types is None:
        model_types = ["random_forest", "xgboost", "gradient_boosting"]

    results = {
        "training_started": datetime.now().isoformat(),
        "datasets": list(datasets.keys()),
        "model_types": model_types,
        "models": [],
        "backtests": [],
    }

    for dataset_key, df in datasets.items():
        logger.info(f"\n{'#'*70}")
        logger.info(f"# DATASET: {dataset_key}")
        logger.info(f"{'#'*70}")

        try:
            ohlcv = prepare_ohlcv(df)
            logger.info(f"Prepared OHLCV data: {len(ohlcv):,} rows")
            logger.info(f"Date range: {ohlcv.index.min()} to {ohlcv.index.max()}")

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            continue

        # Train each model type
        for model_type in model_types:
            model_name = f"{dataset_key.replace('/', '_')}_{model_type}"

            train_result = train_model(
                ohlcv=ohlcv,
                model_type=model_type,
                model_name=model_name,
            )
            results["models"].append(train_result)

            # Run backtest if requested
            if run_backtest and train_result["status"] == "success":
                backtest_result = backtest_model(
                    ohlcv=ohlcv,
                    model_name=model_name,
                    model_type=model_type,
                )
                results["backtests"].append({
                    "model_name": model_name,
                    **backtest_result,
                })

    results["training_completed"] = datetime.now().isoformat()

    # Summary
    successful = sum(1 for m in results["models"] if m["status"] == "success")
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Models trained: {successful}/{len(results['models'])}")

    if results["backtests"]:
        profitable = sum(1 for b in results["backtests"]
                        if b.get("status") == "success" and b.get("total_return_pct", 0) > 0)
        logger.info(f"Profitable backtests: {profitable}/{len(results['backtests'])}")

    return results


def compare_models(results: Dict) -> None:
    """Print model comparison table."""
    if not results.get("models"):
        return

    successful_models = [m for m in results["models"] if m["status"] == "success"]
    if not successful_models:
        return

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model Name':<45} {'Accuracy':>10} {'CV Mean':>10} {'CV Std':>10}")
    print("-" * 80)

    # Sort by accuracy
    sorted_models = sorted(successful_models,
                          key=lambda x: x["metrics"]["accuracy"],
                          reverse=True)

    for m in sorted_models:
        print(f"{m['model_name']:<45} "
              f"{m['metrics']['accuracy']:>10.4f} "
              f"{m['metrics']['cross_val_mean']:>10.4f} "
              f"{m['metrics']['cross_val_std']:>10.4f}")

    print("=" * 80)

    # Find best model
    best = sorted_models[0]
    print(f"\nBest Model: {best['model_name']}")
    print(f"Accuracy: {best['metrics']['accuracy']:.4f}")

    if results.get("backtests"):
        print("\n" + "=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"{'Model Name':<45} {'Return %':>12} {'Win Rate':>10} {'Trades':>8}")
        print("-" * 80)

        sorted_bt = sorted(
            [b for b in results["backtests"] if b.get("status") == "success"],
            key=lambda x: x.get("total_return_pct", 0),
            reverse=True
        )

        for bt in sorted_bt:
            print(f"{bt['model_name']:<45} "
                  f"{bt.get('total_return_pct', 0):>11.2f}% "
                  f"{bt.get('win_rate', 0):>9.2f}% "
                  f"{bt.get('total_trades', 0):>8}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models on historical market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data-dir", "-d",
        default="data/training",
        help="Directory containing training data (default: data/training)"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["random_forest", "xgboost", "gradient_boosting", "all"],
        default="all",
        help="Model type to train (default: all)"
    )
    parser.add_argument(
        "--symbol", "-s",
        help="Train only on specific symbol (e.g., BTC/USDT)"
    )
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="Run backtest after training"
    )
    parser.add_argument(
        "--output", "-o",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Load data
    logger.info("Loading training data...")
    datasets = load_training_data(args.data_dir)

    if not datasets:
        logger.error("No training data found. Run data fetcher first:")
        logger.error("  python run_data_fetcher.py --symbols BTC/USDT,ETH/USDT --days 365")
        sys.exit(1)

    # Filter by symbol if specified
    if args.symbol:
        symbol_key = args.symbol.replace("/", "_")
        datasets = {k: v for k, v in datasets.items() if symbol_key in k}
        if not datasets:
            logger.error(f"No data found for symbol: {args.symbol}")
            sys.exit(1)

    # Determine model types
    if args.model == "all":
        model_types = ["random_forest", "xgboost", "gradient_boosting"]
    else:
        model_types = [args.model]

    # Train models
    results = train_all_models(
        datasets=datasets,
        model_types=model_types,
        run_backtest=args.backtest,
    )

    # Compare models
    compare_models(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    else:
        # Save to default location
        results_file = Path("data/models/training_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
