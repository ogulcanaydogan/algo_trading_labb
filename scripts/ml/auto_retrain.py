#!/usr/bin/env python3
"""
Automated Model Retraining Script.

Runs on a schedule (e.g., weekly) to retrain ML models with fresh data.
Can be run via cron or as a standalone script.

Usage:
    python scripts/ml/auto_retrain.py --symbols BTC/USDT,ETH/USDT --days 90

Cron example (run every Sunday at 2 AM):
    0 2 * * 0 cd /path/to/algo_trading_lab && python scripts/ml/auto_retrain.py
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from bot.ml.data_quality import (
    build_quality_report,
    save_quality_report,
    validate_feature_leakage,
    validate_target_alignment,
)
from bot.ml.feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoRetrainer:
    """Automated model retraining system."""

    def __init__(
        self,
        model_dir: str = "data/models",
        min_accuracy_threshold: float = 0.55,
        max_model_age_days: int = 7,
        label_horizon: int = 5,
        use_triple_barrier: bool = False,
        atr_multiplier: float = 2.0,
        min_return: float = 0.001,
        report_dir: str = "data/reports",
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.min_accuracy_threshold = min_accuracy_threshold
        self.max_model_age_days = max_model_age_days

        self.label_horizon = label_horizon
        self.use_triple_barrier = use_triple_barrier
        self.atr_multiplier = atr_multiplier
        self.min_return = min_return
        self.report_dir = report_dir

        # Training parameters
        self.lookback_period = 20  # Bars to look back for features
        self.future_period = label_horizon  # Bars to look ahead for labels

    def should_retrain(self, symbol: str, model_type: str = "gradient_boosting") -> bool:
        """Check if model needs retraining."""
        symbol_clean = symbol.replace("/", "_")
        meta_path = self.model_dir / f"{symbol_clean}_{model_type}_meta.json"

        if not meta_path.exists():
            logger.info(f"No existing model for {symbol} - training required")
            return True

        with open(meta_path) as f:
            meta = json.load(f)

        # Check age
        trained_at = meta.get("trained_at", "")
        if trained_at:
            trained_date = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
            age_days = (datetime.now() - trained_date.replace(tzinfo=None)).days

            if age_days > self.max_model_age_days:
                logger.info(f"Model for {symbol} is {age_days} days old - retraining")
                return True

        # Check accuracy degradation (would need live tracking)
        cv_accuracy = meta.get("cv_accuracy", 0)
        if cv_accuracy < self.min_accuracy_threshold:
            logger.info(f"Model accuracy {cv_accuracy:.2%} below threshold - retraining")
            return True

        logger.info(f"Model for {symbol} is up to date")
        return False

    def fetch_training_data(
        self,
        symbol: str,
        days: int = 90,
    ) -> pd.DataFrame:
        """Fetch historical data for training."""
        try:
            import ccxt
        except ImportError:
            logger.error("ccxt not installed - run: pip install ccxt")
            return pd.DataFrame()

        exchange = ccxt.binance({
            'enableRateLimit': True,
        })

        # Convert symbol format
        ccxt_symbol = symbol.replace("/", "")

        # Fetch OHLCV data
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        timeframe = "1h"

        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=1000,
            )

            # Fetch more if needed
            while len(ohlcv) < days * 24:
                last_timestamp = ohlcv[-1][0]
                more = exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=last_timestamp + 1,
                    limit=1000,
                )
                if not more:
                    break
                ohlcv.extend(more)

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

    def create_features(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Create features and labels from OHLCV data."""
        if df.empty:
            return np.array([]), np.array([]), []

        # Technical indicators as features
        df = df.copy()

        # Price returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        df["return_20"] = df["close"].pct_change(20)

        # Moving averages
        df["sma_5"] = df["close"].rolling(5).mean()
        df["sma_10"] = df["close"].rolling(10).mean()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Price vs MAs
        df["price_sma5_ratio"] = df["close"] / df["sma_5"]
        df["price_sma10_ratio"] = df["close"] / df["sma_10"]
        df["price_sma20_ratio"] = df["close"] / df["sma_20"]

        # Volatility
        df["volatility_5"] = df["return_1"].rolling(5).std()
        df["volatility_20"] = df["return_1"].rolling(20).std()

        # Volume features
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_ratio"] = df["atr"] / df["close"]

        # Build labels with volatility-aware thresholds
        fe = FeatureEngineer()
        df = fe.build_labels(
            df,
            horizon=self.label_horizon,
            use_triple_barrier=self.use_triple_barrier,
            atr_multiplier=self.atr_multiplier,
            min_return=self.min_return,
        )

        # Drop NaN rows
        df = df.dropna()

        if len(df) < 100:
            logger.warning("Not enough data after feature creation")
            return np.array([]), np.array([]), []

        # Select features (after dropping NaNs)
        feature_cols = [
            "return_1", "return_5", "return_10", "return_20",
            "price_sma5_ratio", "price_sma10_ratio", "price_sma20_ratio",
            "volatility_5", "volatility_20",
            "volume_ratio",
            "rsi", "macd_hist", "bb_position", "atr_ratio"
        ]

        leakage = validate_feature_leakage(feature_cols)
        if leakage:
            logger.warning(f"Leakage columns detected and removed: {leakage}")
            feature_cols = [c for c in feature_cols if c not in leakage]

        alignment_warnings = validate_target_alignment(
            df,
            target_col="target_return",
            horizon=self.label_horizon,
        )
        for warning in alignment_warnings:
            logger.warning(warning)

        target_label = "target_class"
        if self.use_triple_barrier and "target_triple_barrier" in df.columns:
            target_label = "target_triple_barrier"

        report = build_quality_report(
            df,
            feature_cols=feature_cols,
            target_col=target_label,
            symbol=symbol,
            metadata={
                "model_type": "auto_retrain",
                "label_horizon": self.label_horizon,
                "use_triple_barrier": self.use_triple_barrier,
            },
            alignment_warnings=alignment_warnings,
        )
        report_path = save_quality_report(report, report_dir=self.report_dir)
        logger.info(f"Data quality report saved to {report_path}")

        X = df[feature_cols].values
        y = df[target_label].values

        return X, y, feature_cols

    def train_model(
        self,
        symbol: str,
        days: int = 90,
        model_type: str = "gradient_boosting",
    ) -> Dict:
        """Train or retrain a model."""
        logger.info(f"Training {model_type} model for {symbol}...")

        # Fetch data
        df = self.fetch_training_data(symbol, days)
        if df.empty:
            return {"success": False, "error": "No data fetched"}

        # Create features
        X, y, feature_names = self.create_features(df, symbol=symbol)
        if len(X) == 0:
            return {"success": False, "error": "Feature creation failed"}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Time series - don't shuffle
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create model
        if model_type == "gradient_boosting":
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
            )
        else:
            return {"success": False, "error": f"Unknown model type: {model_type}"}

        # Train
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_accuracy = cv_scores.mean()

        logger.info(f"Train accuracy: {train_accuracy:.2%}")
        logger.info(f"Test accuracy: {test_accuracy:.2%}")
        logger.info(f"CV accuracy: {cv_accuracy:.2%} (+/- {cv_scores.std()*2:.2%})")

        # Check if model meets threshold
        if cv_accuracy < self.min_accuracy_threshold:
            logger.warning(f"Model accuracy {cv_accuracy:.2%} below threshold {self.min_accuracy_threshold:.2%}")

        # Save model
        symbol_clean = symbol.replace("/", "_")
        model_path = self.model_dir / f"{symbol_clean}_{model_type}_model.pkl"
        scaler_path = self.model_dir / f"{symbol_clean}_{model_type}_scaler.pkl"
        meta_path = self.model_dir / f"{symbol_clean}_{model_type}_meta.json"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save metadata
        meta = {
            "symbol": symbol,
            "model_type": model_type,
            "trained_at": datetime.now().isoformat(),
            "training_days": days,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "cv_accuracy": float(cv_accuracy),
            "cv_std": float(cv_scores.std()),
            "feature_names": feature_names,
            "feature_importances": dict(zip(
                feature_names,
                [float(x) for x in model.feature_importances_]
            )) if hasattr(model, "feature_importances_") else {},
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved model to {model_path}")

        return {
            "success": True,
            "symbol": symbol,
            "model_type": model_type,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_accuracy": cv_accuracy,
            "model_path": str(model_path),
        }

    def retrain_all(
        self,
        symbols: List[str],
        days: int = 90,
        force: bool = False,
    ) -> List[Dict]:
        """Retrain all models that need it."""
        results = []

        for symbol in symbols:
            for model_type in ["gradient_boosting", "random_forest"]:
                if force or self.should_retrain(symbol, model_type):
                    result = self.train_model(symbol, days, model_type)
                    results.append(result)
                else:
                    results.append({
                        "success": True,
                        "symbol": symbol,
                        "model_type": model_type,
                        "skipped": True,
                        "reason": "Model up to date",
                    })

        return results


def main():
    parser = argparse.ArgumentParser(description="Automated ML Model Retraining")
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT",
        help="Comma-separated list of symbols to train",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of historical data to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if models are up to date",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.55,
        help="Minimum accuracy threshold",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=5,
        help="Forward label horizon in bars",
    )
    parser.add_argument(
        "--use-triple-barrier",
        action="store_true",
        help="Use triple-barrier labels",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=2.0,
        help="ATR multiplier for triple-barrier labels",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=0.001,
        help="Minimum return threshold for labels",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="data/reports",
        help="Report output directory",
    )

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    retrainer = AutoRetrainer(
        min_accuracy_threshold=args.min_accuracy,
        label_horizon=args.label_horizon,
        use_triple_barrier=args.use_triple_barrier,
        atr_multiplier=args.atr_multiplier,
        min_return=args.min_return,
        report_dir=args.report_dir,
    )

    logger.info(f"Starting automated retraining for {len(symbols)} symbols...")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Force: {args.force}")

    results = retrainer.retrain_all(symbols, args.days, args.force)

    # Summary
    logger.info("\n" + "="*50)
    logger.info("RETRAINING SUMMARY")
    logger.info("="*50)

    for result in results:
        symbol = result.get("symbol", "unknown")
        model_type = result.get("model_type", "unknown")
        success = result.get("success", False)
        skipped = result.get("skipped", False)

        if skipped:
            logger.info(f"  {symbol} ({model_type}): SKIPPED - {result.get('reason')}")
        elif success:
            cv_acc = result.get("cv_accuracy", 0)
            logger.info(f"  {symbol} ({model_type}): SUCCESS - CV accuracy: {cv_acc:.2%}")
        else:
            error = result.get("error", "Unknown error")
            logger.info(f"  {symbol} ({model_type}): FAILED - {error}")

    logger.info("="*50)


if __name__ == "__main__":
    from typing import Tuple, List, Dict
    main()
