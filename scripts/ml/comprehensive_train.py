#!/usr/bin/env python3
"""
Comprehensive Model Training Script

Fetches extended historical data (2+ years) and trains all models
with enhanced features and hyperparameter tuning.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bot.ml.feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data/training")
MODEL_DIR = Path("data/models")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Symbols to train
CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"]
FOREX_SYMBOLS = ["EUR/USD", "GBP/USD", "USD/JPY"]
INDEX_SYMBOLS = ["SPX500/USD", "NAS100/USD"]
STOCK_SYMBOLS = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "AMZN"]
COMMODITY_SYMBOLS = ["WTICO/USD", "XAU/USD"]


class DataFetcher:
    """Fetches historical data from multiple sources."""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()

    async def fetch_crypto_data(self, symbol: str, days: int = 730) -> pd.DataFrame:
        """Fetch crypto data from Binance."""
        import ccxt

        exchange = ccxt.binance({'enableRateLimit': True})
        binance_symbol = symbol.replace("/", "")

        logger.info(f"Fetching {days} days of {symbol} data from Binance...")

        all_ohlcv = []
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(binance_symbol, '1h', since=since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000:
                    break
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error fetching {symbol}: {e}")
                break

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        logger.info(f"  Fetched {len(df)} candles for {symbol}")
        return df

    async def fetch_stock_data(self, symbol: str, years: int = 2) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        import yfinance as yf

        logger.info(f"Fetching {years} years of {symbol} data from Yahoo Finance...")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{years}y", interval="1h")

        if df.empty:
            # Try daily data if hourly not available
            df = ticker.history(period=f"{years}y", interval="1d")

        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            logger.info(f"  Fetched {len(df)} candles for {symbol}")

        return df

    async def fetch_forex_data(self, symbol: str, years: int = 2) -> pd.DataFrame:
        """Fetch forex data from Yahoo Finance."""
        import yfinance as yf

        # Convert symbol format
        yf_symbol = symbol.replace("/", "") + "=X"

        logger.info(f"Fetching {years} years of {symbol} data...")

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{years}y", interval="1h")

        if df.empty:
            df = ticker.history(period=f"{years}y", interval="1d")

        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            logger.info(f"  Fetched {len(df)} candles for {symbol}")

        return df

    async def fetch_index_data(self, symbol: str, years: int = 2) -> pd.DataFrame:
        """Fetch index data from Yahoo Finance."""
        import yfinance as yf

        # Map to Yahoo Finance symbols
        yf_map = {
            "SPX500/USD": "^GSPC",
            "NAS100/USD": "^NDX",
        }

        yf_symbol = yf_map.get(symbol, symbol)
        logger.info(f"Fetching {years} years of {symbol} data...")

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{years}y", interval="1h")

        if df.empty:
            df = ticker.history(period=f"{years}y", interval="1d")

        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            if 'volume' not in df.columns:
                df['volume'] = 0
            df = df[['open', 'high', 'low', 'close', 'volume']]
            logger.info(f"  Fetched {len(df)} candles for {symbol}")

        return df

    def save_data(self, df: pd.DataFrame, symbol: str):
        """Save data to parquet file."""
        filename = symbol.replace("/", "_").replace("-", "_") + "_extended.parquet"
        filepath = DATA_DIR / filename
        df.to_parquet(filepath)
        logger.info(f"  Saved to {filepath}")

    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from parquet file."""
        filename = symbol.replace("/", "_").replace("-", "_") + "_extended.parquet"
        filepath = DATA_DIR / filename
        if filepath.exists():
            return pd.read_parquet(filepath)
        return None


class EnhancedFeatureEngineer:
    """Enhanced feature engineering with more indicators."""

    def __init__(self):
        self.base_engineer = FeatureEngineer()

    def add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators."""
        df = df.copy()

        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Multiple timeframe momentum
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # Volatility indicators
        for period in [10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'atr_{period}'] = self._calculate_atr(df, period)

        # RSI with multiple periods
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)

        # MACD variants
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 17, 9)]:
            macd, macd_signal, macd_hist = self._calculate_macd(df['close'], fast, slow, signal)
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = macd_signal
            df[f'macd_hist_{fast}_{slow}'] = macd_hist

        # Bollinger Bands
        for period in [20, 50]:
            for std in [1.5, 2, 2.5]:
                upper, middle, lower = self._calculate_bollinger(df['close'], period, std)
                df[f'bb_upper_{period}_{std}'] = upper
                df[f'bb_lower_{period}_{std}'] = lower
                df[f'bb_width_{period}_{std}'] = (upper - lower) / middle
                df[f'bb_position_{period}_{std}'] = (df['close'] - lower) / (upper - lower)

        # Stochastic
        for period in [14, 21]:
            k, d = self._calculate_stochastic(df, period)
            df[f'stoch_k_{period}'] = k
            df[f'stoch_d_{period}'] = d

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['obv'] = self._calculate_obv(df)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Price patterns
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10) < 0.1).astype(int)

        # Trend strength
        df['adx'] = self._calculate_adx(df, 14)

        # Support/Resistance proximity
        df['distance_to_high_20'] = (df['close'] - df['high'].rolling(20).max()) / df['close']
        df['distance_to_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']

        # Time features (if datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple:
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _calculate_bollinger(self, prices: pd.Series, period: int, std_dev: float) -> Tuple:
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> Tuple:
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(3).mean()
        return k, d

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._calculate_atr(df, 1) * period
        plus_di = 100 * (plus_dm.rolling(period).sum() / (tr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).sum() / (tr + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return adx


class ModelTrainer:
    """Trains and evaluates models."""

    def __init__(self):
        self.feature_engineer = EnhancedFeatureEngineer()
        self.results = {}

    def prepare_data(self, df: pd.DataFrame, lookahead: int = 24) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training."""
        # Add enhanced features
        df = self.feature_engineer.add_enhanced_features(df)

        # Create target: 1 if price goes up by threshold, -1 if down, 0 otherwise
        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        threshold = 0.01  # 1% threshold

        df['target'] = 0
        df.loc[future_return > threshold, 'target'] = 1
        df.loc[future_return < -threshold, 'target'] = -1

        # Remove rows with NaN
        df = df.dropna()

        # Select feature columns (exclude OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'returns', 'log_returns']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df['target']

        # Replace infinities
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X, y

    def train_model(self, symbol: str, X: pd.DataFrame, y: pd.Series,
                    model_type: str = 'random_forest', tune_hyperparams: bool = True) -> Dict:
        """Train a model with optional hyperparameter tuning."""

        logger.info(f"Training {model_type} for {symbol} ({len(X)} samples, {len(X.columns)} features)")

        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)

        # Define model and hyperparameter grid
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            } if tune_hyperparams else {}
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            } if tune_hyperparams else {}
        else:
            try:
                from xgboost import XGBClassifier
                base_model = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                } if tune_hyperparams else {}
            except ImportError:
                logger.warning("XGBoost not available, using RandomForest")
                return self.train_model(symbol, X, y, 'random_forest', tune_hyperparams)

        # Train-test split (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Hyperparameter tuning or direct training
        if tune_hyperparams and param_grid:
            logger.info(f"  Running hyperparameter search...")
            # Use smaller grid for speed
            small_grid = {k: v[:2] for k, v in param_grid.items()}
            grid_search = GridSearchCV(
                base_model, small_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
            logger.info(f"  Best params: {grid_search.best_params_}")
        else:
            model = base_model
            model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')

        results = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': len(X.columns)
        }

        logger.info(f"  Accuracy: {accuracy:.2%}, CV: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

        return results

    def save_model(self, symbol: str, model_type: str, results: Dict):
        """Save model and metadata."""
        symbol_clean = symbol.replace("/", "_").replace("-", "_")

        # Save model
        model_path = MODEL_DIR / f"{symbol_clean}_{model_type}_model.pkl"
        joblib.dump(results['model'], model_path)

        # Save scaler
        scaler_path = MODEL_DIR / f"{symbol_clean}_{model_type}_scaler.pkl"
        joblib.dump(results['scaler'], scaler_path)

        # Save metadata
        import json
        meta = {
            'symbol': symbol,
            'model_type': model_type,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'cv_mean': results['cv_mean'],
            'cv_std': results['cv_std'],
            'train_samples': results['train_samples'],
            'test_samples': results['test_samples'],
            'n_features': results['n_features'],
            'feature_names': results['feature_names'],
            'trained_at': datetime.now().isoformat()
        }
        meta_path = MODEL_DIR / f"{symbol_clean}_{model_type}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"  Saved model to {model_path}")


async def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE MODEL TRAINING")
    logger.info("=" * 60)

    fetcher = DataFetcher()
    trainer = ModelTrainer()

    all_symbols = []

    # Collect all symbols by type
    crypto_list = [(s, 'crypto') for s in CRYPTO_SYMBOLS]
    forex_list = [(s, 'forex') for s in FOREX_SYMBOLS]
    index_list = [(s, 'index') for s in INDEX_SYMBOLS]
    stock_list = [(s, 'stock') for s in STOCK_SYMBOLS]
    commodity_list = [(s, 'commodity') for s in COMMODITY_SYMBOLS]

    all_symbols = crypto_list + forex_list + index_list + stock_list + commodity_list

    results_summary = []

    for symbol, symbol_type in all_symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol} ({symbol_type})")
        logger.info("=" * 60)

        # Fetch or load data
        df = fetcher.load_data(symbol)

        if df is None or len(df) < 1000:
            try:
                if symbol_type == 'crypto':
                    df = await fetcher.fetch_crypto_data(symbol, days=730)
                elif symbol_type == 'stock':
                    df = await fetcher.fetch_stock_data(symbol, years=2)
                elif symbol_type == 'forex':
                    df = await fetcher.fetch_forex_data(symbol, years=2)
                elif symbol_type in ['index', 'commodity']:
                    df = await fetcher.fetch_index_data(symbol, years=2)

                if not df.empty:
                    fetcher.save_data(df, symbol)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        if df is None or len(df) < 500:
            logger.warning(f"Insufficient data for {symbol}, skipping")
            continue

        # Prepare data
        try:
            X, y = trainer.prepare_data(df)
        except Exception as e:
            logger.error(f"Failed to prepare data for {symbol}: {e}")
            continue

        if len(X) < 500:
            logger.warning(f"Insufficient samples for {symbol} after feature engineering")
            continue

        # Train multiple model types
        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            try:
                results = trainer.train_model(symbol, X, y, model_type, tune_hyperparams=True)
                trainer.save_model(symbol, model_type, results)

                results_summary.append({
                    'symbol': symbol,
                    'type': symbol_type,
                    'model': model_type,
                    'accuracy': results['accuracy'],
                    'f1': results['f1'],
                    'cv_mean': results['cv_mean']
                })
            except Exception as e:
                logger.error(f"Failed to train {model_type} for {symbol}: {e}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values('accuracy', ascending=False)

        for _, row in summary_df.iterrows():
            logger.info(f"{row['symbol']:15} | {row['model']:20} | Acc: {row['accuracy']:.2%} | F1: {row['f1']:.2%}")

        # Best model per symbol
        logger.info("\n" + "-" * 40)
        logger.info("Best model per symbol:")
        best_per_symbol = summary_df.loc[summary_df.groupby('symbol')['accuracy'].idxmax()]
        for _, row in best_per_symbol.iterrows():
            logger.info(f"  {row['symbol']}: {row['model']} ({row['accuracy']:.2%})")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
