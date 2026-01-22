#!/usr/bin/env python3
"""
Automatic Model Retraining Cron Script

Runs daily/weekly to:
1. Fetch latest data
2. Check for concept drift
3. Retrain models if needed
4. Apply accuracy improvements
5. Deploy updated models
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'data/logs/auto_retrain_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/training")
MODEL_DIR = Path("data/models")
BACKUP_DIR = Path("data/models/backups")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Symbols to maintain
ALL_SYMBOLS = {
    'crypto': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT'],
    'forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD'],
    'indices': ['SPX500/USD', 'NAS100/USD', 'US30/USD'],
    'stocks': ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META'],
    'commodities': ['WTICO/USD', 'XAU/USD', 'XAG/USD']
}


class ModelRetrainer:
    """Handles model retraining logic."""

    def __init__(self, symbol: str, asset_type: str):
        self.symbol = symbol
        self.asset_type = asset_type
        self.symbol_clean = symbol.replace("/", "_").replace("-", "_")

    def should_retrain(self) -> tuple:
        """
        Check if model needs retraining.

        Returns:
            (should_retrain, reason)
        """
        # Check model age
        model_path = MODEL_DIR / f"{self.symbol_clean}_random_forest_model.pkl"
        meta_path = MODEL_DIR / f"{self.symbol_clean}_random_forest_meta.json"

        if not model_path.exists():
            return True, "No existing model"

        # Check age
        model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
        if model_age > timedelta(days=7):
            return True, f"Model is {model_age.days} days old"

        # Check performance degradation from meta
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            accuracy = meta.get('accuracy', 0)
            if accuracy < 0.35:  # Below random for 3-class
                return True, f"Low accuracy: {accuracy:.2%}"

        # Check for drift (would need recent predictions vs actuals)
        drift_file = MODEL_DIR / f"{self.symbol_clean}_drift.json"
        if drift_file.exists():
            with open(drift_file) as f:
                drift = json.load(f)
            if drift.get('drift_detected', False):
                return True, f"Drift detected: {drift.get('severity', 0):.2%}"

        return False, "Model is current"

    async def fetch_latest_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch latest data for incremental training."""
        try:
            if self.asset_type == 'crypto':
                import ccxt
                exchange = ccxt.binance({'enableRateLimit': True})
                binance_symbol = self.symbol.replace("/", "")

                since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
                ohlcv = exchange.fetch_ohlcv(binance_symbol, '1h', since=since, limit=days*24)

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df

            else:
                import yfinance as yf
                symbol_map = {
                    'forex': lambda s: s.replace("/", "") + "=X",
                    'indices': lambda s: {"SPX500/USD": "^GSPC", "NAS100/USD": "^NDX", "US30/USD": "^DJI"}.get(s, s),
                    'stocks': lambda s: s,
                    'commodities': lambda s: {"WTICO/USD": "CL=F", "XAU/USD": "GC=F", "XAG/USD": "SI=F"}.get(s, s)
                }
                yf_symbol = symbol_map[self.asset_type](self.symbol)
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(period=f"{days}d", interval="1h")
                if not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {self.symbol}: {e}")
            return pd.DataFrame()

    def backup_model(self):
        """Backup existing model before retraining."""
        model_path = MODEL_DIR / f"{self.symbol_clean}_random_forest_model.pkl"
        if model_path.exists():
            backup_name = f"{self.symbol_clean}_random_forest_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            backup_path = BACKUP_DIR / backup_name
            import shutil
            shutil.copy(model_path, backup_path)
            logger.info(f"Backed up model to {backup_path}")

    async def retrain(self) -> dict:
        """
        Retrain the model.

        Returns:
            Training results dict
        """
        logger.info(f"Retraining {self.symbol}...")

        # Backup existing
        self.backup_model()

        # Load existing data
        data_file = DATA_DIR / f"{self.symbol_clean}_extended.parquet"
        if data_file.exists():
            existing_df = pd.read_parquet(data_file)
        else:
            existing_df = pd.DataFrame()

        # Fetch latest data
        new_df = await self.fetch_latest_data(days=60)

        if new_df.empty:
            return {'success': False, 'reason': 'No new data'}

        # Combine data
        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df]).drop_duplicates()
            # Keep last 2 years
            cutoff = datetime.now() - timedelta(days=730)
            combined = combined[combined.index > cutoff]
        else:
            combined = new_df

        # Save combined data
        combined.to_parquet(data_file)

        # Train model
        try:
            from scripts.ml.comprehensive_train import EnhancedFeatureEngineer, ModelTrainer

            engineer = EnhancedFeatureEngineer()
            trainer = ModelTrainer()

            df = engineer.add_enhanced_features(combined)

            # Create target
            future_return = df['close'].shift(-24) / df['close'] - 1
            df['target'] = 0
            df.loc[future_return > 0.01, 'target'] = 1
            df.loc[future_return < -0.01, 'target'] = -1
            df = df.dropna()

            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'returns', 'log_returns']
            feature_cols = [c for c in df.columns if c not in exclude_cols]

            X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = df['target']

            if len(X) < 500:
                return {'success': False, 'reason': 'Insufficient data'}

            # Train with best model type
            best_accuracy = 0
            best_results = None
            best_type = None

            for model_type in ['random_forest', 'gradient_boosting']:
                results = trainer.train_model(self.symbol, X, y, model_type, tune_hyperparams=False)
                if results['accuracy'] > best_accuracy:
                    best_accuracy = results['accuracy']
                    best_results = results
                    best_type = model_type

            if best_results:
                trainer.save_model(self.symbol, best_type, best_results)
                return {
                    'success': True,
                    'accuracy': best_accuracy,
                    'model_type': best_type,
                    'samples': len(X)
                }

        except Exception as e:
            logger.error(f"Retrain failed for {self.symbol}: {e}")
            return {'success': False, 'reason': str(e)}

        return {'success': False, 'reason': 'Unknown error'}


async def run_auto_retrain(force: bool = False, asset_types: list = None):
    """
    Run automatic retraining for all symbols.

    Args:
        force: Force retrain even if not needed
        asset_types: List of asset types to retrain (default: all)
    """
    logger.info("=" * 60)
    logger.info("AUTOMATIC MODEL RETRAINING")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 60)

    if asset_types is None:
        asset_types = list(ALL_SYMBOLS.keys())

    results = {
        'timestamp': datetime.now().isoformat(),
        'retrained': [],
        'skipped': [],
        'failed': []
    }

    for asset_type in asset_types:
        symbols = ALL_SYMBOLS.get(asset_type, [])

        for symbol in symbols:
            retrainer = ModelRetrainer(symbol, asset_type)

            should_retrain, reason = retrainer.should_retrain()

            if force or should_retrain:
                logger.info(f"\n{'='*40}")
                logger.info(f"Retraining {symbol} ({asset_type}): {reason}")
                logger.info("=" * 40)

                result = await retrainer.retrain()

                if result['success']:
                    results['retrained'].append({
                        'symbol': symbol,
                        'accuracy': result['accuracy'],
                        'model_type': result['model_type']
                    })
                    logger.info(f"✓ {symbol}: {result['accuracy']:.2%} accuracy")
                else:
                    results['failed'].append({
                        'symbol': symbol,
                        'reason': result['reason']
                    })
                    logger.error(f"✗ {symbol}: {result['reason']}")
            else:
                results['skipped'].append({
                    'symbol': symbol,
                    'reason': reason
                })
                logger.info(f"Skip {symbol}: {reason}")

    # Save results
    results_file = Path("data/logs") / f"retrain_results_{datetime.now().strftime('%Y%m%d')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RETRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Retrained: {len(results['retrained'])}")
    logger.info(f"Skipped: {len(results['skipped'])}")
    logger.info(f"Failed: {len(results['failed'])}")

    if results['retrained']:
        avg_accuracy = np.mean([r['accuracy'] for r in results['retrained']])
        logger.info(f"Average accuracy: {avg_accuracy:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Auto-retrain trading models')
    parser.add_argument('--force', action='store_true', help='Force retrain all models')
    parser.add_argument('--asset-types', nargs='+', choices=['crypto', 'forex', 'indices', 'stocks', 'commodities'],
                       help='Asset types to retrain')
    args = parser.parse_args()

    asyncio.run(run_auto_retrain(force=args.force, asset_types=args.asset_types))


if __name__ == "__main__":
    main()
