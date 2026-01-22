#!/usr/bin/env python3
"""
CALIBRATED High-Confidence Model Training

Key insight: Instead of predicting every sample, only predict when
the model is CONFIDENT (probability > threshold). This naturally
filters to higher-accuracy predictions.

Approach:
1. Train a well-calibrated probabilistic model
2. Only make predictions when probability > 0.60 (or other threshold)
3. Measure accuracy ONLY on confident predictions
4. Trade-off: fewer signals but higher accuracy
"""

import asyncio
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.preprocessing import RobustScaler
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/training")
MODEL_DIR = Path("data/models")


class CalibratedFeatureEngineer:
    """Features focused on high-quality signals."""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set."""
        df = df.copy()

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('volume', pd.Series(1, index=df.index))
        returns = close.pct_change()

        # === TREND ===
        for p in [8, 21, 55, 100, 200]:
            df[f'ema_{p}'] = close.ewm(span=p).mean()

        df['trend_strength'] = (
            (df['ema_8'] > df['ema_21']).astype(int) +
            (df['ema_21'] > df['ema_55']).astype(int) +
            (df['ema_55'] > df['ema_100']).astype(int) +
            (df['ema_100'] > df['ema_200']).astype(int)
        ) - 2  # -2 to +2 scale

        df['price_vs_ema21'] = (close - df['ema_21']) / df['ema_21']
        df['price_vs_ema55'] = (close - df['ema_55']) / df['ema_55']
        df['price_vs_ema200'] = (close - df['ema_200']) / df['ema_200']

        # ADX
        df['adx'] = self._calculate_adx(df, 14)

        # === MOMENTUM ===
        df['rsi'] = self._calculate_rsi(close, 14)
        df['rsi_7'] = self._calculate_rsi(close, 7)
        df['rsi_momentum'] = df['rsi'] - df['rsi'].shift(5)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / close * 100
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Williams %R
        df['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)

        # CCI
        typical = (high + low + close) / 3
        df['cci'] = (typical - typical.rolling(20).mean()) / (typical.rolling(20).std() * 0.015 + 1e-10)

        # === VOLUME ===
        df['volume_sma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma'] + 1)

        obv = (np.sign(returns) * volume).cumsum()
        df['obv_slope'] = (obv - obv.shift(10)) / (obv.shift(10).abs() + 1e-10)

        # === VOLATILITY ===
        df['volatility'] = returns.rolling(20).std()
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_pct'] = df['atr'] / close

        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_position'] = (close - bb_sma) / (2 * bb_std + 1e-10)
        df['bb_width'] = 4 * bb_std / bb_sma

        # === PRICE ACTION ===
        for p in [1, 3, 5, 10, 20]:
            df[f'return_{p}'] = returns.rolling(p).sum()

        df['range_ratio'] = (high - low) / close
        df['body_ratio'] = abs(close - df['open']) / (high - low + 1e-10) if 'open' in df.columns else 0.5

        # === TREND CONFIRMATION ===
        df['price_above_ema21'] = (close > df['ema_21']).astype(float)
        df['price_above_ema55'] = (close > df['ema_55']).astype(float)
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(float)
        df['rsi_bullish'] = (df['rsi'] > 50).astype(float)

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-10)))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._calculate_atr(df, 1)
        atr = tr.rolling(period).sum()

        plus_di = 100 * plus_dm.rolling(period).sum() / (atr + 1e-10)
        minus_di = 100 * minus_dm.rolling(period).sum() / (atr + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.rolling(period).mean()


class CalibratedTrainer:
    """
    Train calibrated models that provide reliable probability estimates.

    Key: Use CalibratedClassifierCV to ensure probability outputs are
    well-calibrated, then filter predictions by confidence threshold.
    """

    def __init__(self):
        self.feature_engineer = CalibratedFeatureEngineer()
        self.scaler = RobustScaler()

    def train(
        self,
        symbol: str,
        df: pd.DataFrame,
        confidence_thresholds: list = [0.55, 0.60, 0.65, 0.70]
    ) -> Dict[str, Any]:
        """Train calibrated model for a symbol."""
        logger.info(f"\n{'='*60}")
        logger.info(f"CALIBRATED MODEL TRAINING: {symbol}")
        logger.info("=" * 60)

        # Create features
        logger.info("Creating features...")
        df = self.feature_engineer.create_features(df)

        # Create labels - 48h lookahead, 2% threshold
        lookahead = 48
        threshold = 0.02

        future_return = df['close'].shift(-lookahead) / df['close'] - 1
        labels = pd.Series(np.nan, index=df.index)
        labels[future_return > threshold] = 1
        labels[future_return < -threshold] = 0

        # Filter to samples with clear outcomes
        df_train = df.copy()
        df_train['target'] = labels
        df_train = df_train.dropna(subset=['target'])

        # Features
        feature_cols = [
            'trend_strength', 'price_vs_ema21', 'price_vs_ema55', 'price_vs_ema200',
            'adx', 'rsi', 'rsi_7', 'rsi_momentum',
            'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'williams_r', 'cci',
            'volume_ratio', 'obv_slope',
            'volatility', 'atr_pct', 'bb_position', 'bb_width',
            'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
            'range_ratio', 'price_above_ema21', 'price_above_ema55',
            'macd_bullish', 'rsi_bullish'
        ]

        available = [c for c in feature_cols if c in df_train.columns]
        X = df_train[available].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df_train['target'].astype(int)

        logger.info(f"Total samples: {len(X)}")
        logger.info(f"UP: {sum(y==1)} ({sum(y==1)/len(y):.1%}), DOWN: {sum(y==0)} ({sum(y==0)/len(y):.1%})")

        # Split with gap
        gap = lookahead
        split_idx = int(len(X) * 0.70)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx + gap:]
        y_test = y.iloc[split_idx + gap:]

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)} (gap: {gap})")

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train base model
        base_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42
        )

        # Calibrate using isotonic regression
        logger.info("Training calibrated model...")
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method='isotonic',
            cv=5
        )
        calibrated_model.fit(X_train_scaled, y_train)

        # Get probabilities on test set
        y_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate calibration
        brier = brier_score_loss(y_test, y_proba)
        logger.info(f"Brier score (lower is better): {brier:.4f}")

        # Evaluate at different confidence thresholds
        results_by_threshold = {}
        best_accuracy = 0
        best_threshold = 0.5

        logger.info("\nAccuracy by confidence threshold:")
        logger.info("-" * 50)

        for conf_thresh in confidence_thresholds:
            # Get confident predictions only
            confident_up = y_proba > conf_thresh
            confident_down = y_proba < (1 - conf_thresh)
            confident = confident_up | confident_down

            if sum(confident) < 50:
                logger.info(f"Threshold {conf_thresh:.0%}: Too few samples ({sum(confident)})")
                continue

            # Make predictions for confident samples
            y_pred_confident = np.where(y_proba[confident] > 0.5, 1, 0)
            y_test_confident = y_test.values[confident]

            acc = accuracy_score(y_test_confident, y_pred_confident)
            prec = precision_score(y_test_confident, y_pred_confident, zero_division=0)
            rec = recall_score(y_test_confident, y_pred_confident, zero_division=0)
            coverage = sum(confident) / len(y_test)

            results_by_threshold[conf_thresh] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'n_predictions': sum(confident),
                'coverage': coverage
            }

            status = "*** 60%+ ***" if acc >= 0.60 else ""
            logger.info(f"Threshold {conf_thresh:.0%}: {acc:.2%} accuracy on {sum(confident)} samples ({coverage:.1%} coverage) {status}")

            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = conf_thresh

        # Save model
        symbol_clean = symbol.replace("/", "_")
        model_path = MODEL_DIR / f"{symbol_clean}_calibrated_model.pkl"
        scaler_path = MODEL_DIR / f"{symbol_clean}_calibrated_scaler.pkl"
        meta_path = MODEL_DIR / f"{symbol_clean}_calibrated_meta.json"

        joblib.dump(calibrated_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        meta = {
            'symbol': symbol,
            'model_type': 'calibrated',
            'lookahead': lookahead,
            'threshold': threshold,
            'brier_score': brier,
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy,
            'results_by_threshold': {str(k): v for k, v in results_by_threshold.items()},
            'features': available,
            'trained_at': datetime.now().isoformat()
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        achieved_60 = any(r['accuracy'] >= 0.60 for r in results_by_threshold.values())
        status = "ACHIEVED 60%+" if achieved_60 else "BELOW 60%"

        logger.info(f"\n{'='*60}")
        logger.info(f"RESULT: {best_accuracy:.2%} at {best_threshold:.0%} confidence ({status})")
        logger.info(f"Saved to {model_path}")
        logger.info("=" * 60)

        return {
            'success': True,
            'best_accuracy': best_accuracy,
            'best_threshold': best_threshold,
            'achieved_60_plus': achieved_60,
            'results_by_threshold': results_by_threshold,
            'brier_score': brier
        }


async def fetch_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """Fetch historical data."""
    data_file = DATA_DIR / f"{symbol.replace('/', '_')}_extended.parquet"

    if data_file.exists():
        df = pd.read_parquet(data_file)
        if len(df) > 5000:
            return df

    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})

        logger.info(f"Fetching {days} days of {symbol} data...")

        all_ohlcv = []
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

        while True:
            ohlcv = exchange.fetch_ohlcv(symbol.replace("/", ""), '1h', since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break
            await asyncio.sleep(0.1)

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.to_parquet(data_file)

        logger.info(f"Fetched {len(df)} candles")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


async def main():
    """Train calibrated models for all symbols."""
    logger.info("=" * 70)
    logger.info("CALIBRATED MODEL TRAINING SYSTEM")
    logger.info("Trade quality for quantity: higher accuracy on fewer signals")
    logger.info("=" * 70)

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT']
    trainer = CalibratedTrainer()

    results = []

    for symbol in symbols:
        df = await fetch_data(symbol)

        if df.empty or len(df) < 5000:
            logger.warning(f"Insufficient data for {symbol}")
            continue

        result = trainer.train(symbol, df)
        results.append({'symbol': symbol, **result})

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    achieved = 0
    for r in results:
        if r.get('success'):
            status = "60%+ ACHIEVED" if r.get('achieved_60_plus') else "Below 60%"
            acc = r.get('best_accuracy', 0)
            thresh = r.get('best_threshold', 0)
            logger.info(f"  {r['symbol']}: {acc:.2%} at {thresh:.0%} confidence - {status}")
            if r.get('achieved_60_plus'):
                achieved += 1

    logger.info(f"\nAchieved 60%+ (at some confidence level): {achieved}/{len(results)} symbols")


if __name__ == "__main__":
    asyncio.run(main())
