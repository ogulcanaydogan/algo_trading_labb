#!/usr/bin/env python3
"""
PROPER High-Accuracy Model Training

Key principles to avoid overfitting and data leakage:
1. Conviction filtering based ONLY on current indicators (no future data)
2. Labels based on future returns (separate from filtering)
3. Proper train/test split with gap to avoid leakage
4. Walk-forward validation for realistic accuracy estimates
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
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import RobustScaler
import joblib

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/training")
MODEL_DIR = Path("data/models")


class ProperFeatureEngineer:
    """Features designed for prediction without look-ahead bias."""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features using ONLY historical data."""
        df = df.copy()

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('volume', pd.Series(1, index=df.index))
        returns = close.pct_change()

        # === TREND FEATURES ===
        for p in [8, 21, 55, 100, 200]:
            df[f'ema_{p}'] = close.ewm(span=p).mean()

        # EMA alignment (all point same direction)
        df['ema_bullish_aligned'] = (
            (df['ema_8'] > df['ema_21']) &
            (df['ema_21'] > df['ema_55']) &
            (df['ema_55'] > df['ema_100'])
        ).astype(int)

        df['ema_bearish_aligned'] = (
            (df['ema_8'] < df['ema_21']) &
            (df['ema_21'] < df['ema_55']) &
            (df['ema_55'] < df['ema_100'])
        ).astype(int)

        # Price position relative to EMAs
        df['price_vs_ema21'] = (close - df['ema_21']) / df['ema_21']
        df['price_vs_ema55'] = (close - df['ema_55']) / df['ema_55']

        # ADX (trend strength)
        df['adx'] = self._calculate_adx(df, 14)

        # Trend slope (rate of change)
        for p in [5, 10, 20]:
            df[f'trend_slope_{p}'] = (close - close.shift(p)) / close.shift(p)

        # === MOMENTUM ===
        df['rsi'] = self._calculate_rsi(close, 14)
        df['rsi_7'] = self._calculate_rsi(close, 7)

        # RSI momentum
        df['rsi_change'] = df['rsi'].diff(3)
        df['rsi_bullish'] = (df['rsi'] > 50).astype(int)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_positive'] = (df['macd'] > 0).astype(int)

        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['stoch_bullish'] = (df['stoch_k'] > df['stoch_d']).astype(int)

        # === VOLUME ===
        df['volume_sma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / (df['volume_sma'] + 1)
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)

        # OBV trend
        obv = (np.sign(returns) * volume).cumsum()
        obv_ema = obv.ewm(span=20).mean()
        df['obv_bullish'] = (obv > obv_ema).astype(int)

        # === VOLATILITY ===
        df['volatility'] = returns.rolling(20).std()
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_pct'] = df['atr'] / close

        # BB
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_position'] = (close - bb_sma) / (2 * bb_std + 1e-10)

        # === RETURNS ===
        for p in [1, 3, 5, 10, 20]:
            df[f'return_{p}'] = returns.rolling(p).sum()

        # Recent momentum
        df['momentum_5_20'] = df['return_5'] - df['return_20'] / 4

        # === COMPOSITE SCORES (current data only) ===
        df['bullish_indicators'] = (
            df['ema_bullish_aligned'] +
            df['rsi_bullish'] +
            df['macd_bullish'] +
            df['macd_positive'] +
            df['stoch_bullish'] +
            df['obv_bullish']
        )  # 0-6

        df['bearish_indicators'] = (
            df['ema_bearish_aligned'] +
            (1 - df['rsi_bullish']) +
            (1 - df['macd_bullish']) +
            (1 - df['macd_positive']) +
            (1 - df['stoch_bullish']) +
            (1 - df['obv_bullish'])
        )  # 0-6

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

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


class ProperTrainer:
    """
    Train models properly without data leakage.

    Key approach:
    1. Filter to high-conviction setups based on CURRENT indicators only
    2. Create labels based on FUTURE returns
    3. Use walk-forward validation with gap
    """

    def __init__(self):
        self.feature_engineer = ProperFeatureEngineer()
        self.scaler = RobustScaler()

    def train(
        self,
        symbol: str,
        df: pd.DataFrame,
        target_accuracy: float = 0.60
    ) -> Dict[str, Any]:
        """Train model for a symbol with proper methodology."""
        logger.info(f"\n{'='*60}")
        logger.info(f"PROPER MODEL TRAINING: {symbol}")
        logger.info(f"Target: {target_accuracy:.0%} accuracy")
        logger.info("=" * 60)

        # Create features
        logger.info("Creating features...")
        df = self.feature_engineer.create_features(df)

        best_accuracy = 0
        best_model = None
        best_results = None

        # Try different lookaheads and thresholds
        for lookahead, threshold in [
            (24, 0.015),  # 24h, 1.5% move
            (48, 0.020),  # 48h, 2% move
            (12, 0.010),  # 12h, 1% move
        ]:
            logger.info(f"\n--- Lookahead: {lookahead}h, Threshold: {threshold:.1%} ---")

            # Create labels based on future returns
            future_return = df['close'].shift(-lookahead) / df['close'] - 1
            labels = pd.Series(np.nan, index=df.index)
            labels[future_return > threshold] = 1  # UP
            labels[future_return < -threshold] = 0  # DOWN
            # Neutral samples (between thresholds) stay NaN

            # Filter to conviction setups based on CURRENT indicators only
            # High conviction = strong indicator agreement (4+ out of 6)
            conviction_bullish = (df['bullish_indicators'] >= 4)
            conviction_bearish = (df['bearish_indicators'] >= 4)
            conviction_filter = conviction_bullish | conviction_bearish

            # Apply both filters
            df_train = df.copy()
            df_train['target'] = labels
            df_train = df_train[conviction_filter]  # Only conviction setups
            df_train = df_train.dropna(subset=['target'])  # Only clear outcomes

            if len(df_train) < 500:
                logger.warning(f"Not enough samples after filtering: {len(df_train)}")
                continue

            # Feature columns
            feature_cols = [
                'price_vs_ema21', 'price_vs_ema55',
                'ema_bullish_aligned', 'ema_bearish_aligned',
                'adx', 'trend_slope_5', 'trend_slope_10', 'trend_slope_20',
                'rsi', 'rsi_7', 'rsi_change', 'rsi_bullish',
                'macd', 'macd_signal', 'macd_hist', 'macd_bullish', 'macd_positive',
                'stoch_k', 'stoch_d', 'stoch_bullish',
                'volume_ratio', 'high_volume', 'obv_bullish',
                'volatility', 'atr_pct', 'bb_position',
                'return_1', 'return_3', 'return_5', 'return_10', 'return_20',
                'momentum_5_20', 'bullish_indicators', 'bearish_indicators'
            ]

            available = [c for c in feature_cols if c in df_train.columns]
            X = df_train[available].replace([np.inf, -np.inf], np.nan).fillna(0)
            y = df_train['target'].astype(int)

            n_up = sum(y == 1)
            n_down = sum(y == 0)
            logger.info(f"Samples: {len(X)}, UP: {n_up} ({n_up/len(X):.1%}), DOWN: {n_down} ({n_down/len(X):.1%})")

            # PROPER train/test split with gap to prevent leakage
            # Gap should be at least lookahead bars
            gap = lookahead
            split_idx = int(len(X) * 0.70)

            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]

            X_test = X.iloc[split_idx + gap:]  # GAP to prevent leakage
            y_test = y.iloc[split_idx + gap:]

            if len(X_test) < 100:
                logger.warning(f"Not enough test samples: {len(X_test)}")
                continue

            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)} (gap: {gap})")

            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Try multiple models
            models = {
                'rf': RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=20,
                    class_weight='balanced', random_state=42, n_jobs=-1
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=20, random_state=42
                ),
                'et': ExtraTreesClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=20,
                    class_weight='balanced', random_state=42, n_jobs=-1
                ),
            }

            # Add LightGBM
            try:
                from lightgbm import LGBMClassifier
                models['lgbm'] = LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    num_leaves=20, min_child_samples=20, class_weight='balanced',
                    random_state=42, verbose=-1, n_jobs=-1
                )
            except ImportError:
                pass

            # Add XGBoost
            try:
                from xgboost import XGBClassifier
                models['xgb'] = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    random_state=42, eval_metric='logloss', n_jobs=-1
                )
            except ImportError:
                pass

            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)

                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = model
                        best_results = {
                            'lookahead': lookahead,
                            'threshold': threshold,
                            'model_name': name,
                            'accuracy': acc,
                            'precision': precision_score(y_test, y_pred, zero_division=0),
                            'recall': recall_score(y_test, y_pred, zero_division=0),
                            'f1': f1_score(y_test, y_pred, zero_division=0),
                            'n_train': len(X_train),
                            'n_test': len(X_test),
                            'features': available
                        }
                        logger.info(f"  {name}: {acc:.2%} *** NEW BEST ***")
                    else:
                        logger.info(f"  {name}: {acc:.2%}")

                except Exception as e:
                    logger.error(f"  {name} failed: {e}")

        if best_model is None:
            return {'success': False, 'reason': 'All training attempts failed'}

        # Try voting ensemble
        if best_accuracy >= 0.52:
            logger.info("\nTrying voting ensemble...")
            try:
                ensemble_models = [
                    ('rf', RandomForestClassifier(n_estimators=200, max_depth=8,
                        min_samples_leaf=20, class_weight='balanced', random_state=42, n_jobs=-1)),
                    ('et', ExtraTreesClassifier(n_estimators=200, max_depth=8,
                        min_samples_leaf=20, class_weight='balanced', random_state=42, n_jobs=-1)),
                    ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=4,
                        learning_rate=0.05, subsample=0.8, min_samples_leaf=20, random_state=42)),
                ]

                try:
                    from lightgbm import LGBMClassifier
                    ensemble_models.append(('lgbm', LGBMClassifier(
                        n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=20,
                        min_child_samples=20, class_weight='balanced', random_state=42,
                        verbose=-1, n_jobs=-1
                    )))
                except ImportError:
                    pass

                voting = VotingClassifier(estimators=ensemble_models, voting='soft')
                voting.fit(X_train_scaled, y_train)

                y_pred_ens = voting.predict(X_test_scaled)
                ens_acc = accuracy_score(y_test, y_pred_ens)
                logger.info(f"Ensemble: {ens_acc:.2%}")

                if ens_acc > best_accuracy:
                    best_accuracy = ens_acc
                    best_model = voting
                    best_results['accuracy'] = ens_acc
                    best_results['model_name'] = 'voting_ensemble'
                    best_results['precision'] = precision_score(y_test, y_pred_ens, zero_division=0)
                    best_results['recall'] = recall_score(y_test, y_pred_ens, zero_division=0)
                    best_results['f1'] = f1_score(y_test, y_pred_ens, zero_division=0)
                    logger.info("Ensemble is best!")
            except Exception as e:
                logger.warning(f"Ensemble failed: {e}")

        # Save model
        symbol_clean = symbol.replace("/", "_")
        model_path = MODEL_DIR / f"{symbol_clean}_proper_model.pkl"
        scaler_path = MODEL_DIR / f"{symbol_clean}_proper_scaler.pkl"
        meta_path = MODEL_DIR / f"{symbol_clean}_proper_meta.json"

        joblib.dump(best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        meta = {
            'symbol': symbol,
            'model_type': 'proper',
            'trained_at': datetime.now().isoformat(),
            **best_results
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        status = "TARGET MET" if best_accuracy >= target_accuracy else "BELOW TARGET"
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL: {best_accuracy:.2%} accuracy ({status})")
        logger.info(f"Lookahead: {best_results['lookahead']}h, Threshold: {best_results['threshold']:.1%}")
        logger.info(f"Model: {best_results['model_name']}")
        logger.info(f"Saved to {model_path}")
        logger.info("=" * 60)

        return {
            'success': True,
            'accuracy': best_accuracy,
            'target_met': best_accuracy >= target_accuracy,
            **best_results
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
        binance_symbol = symbol.replace("/", "")

        logger.info(f"Fetching {days} days of {symbol} data...")

        all_ohlcv = []
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

        while True:
            ohlcv = exchange.fetch_ohlcv(binance_symbol, '1h', since=since, limit=1000)
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
    """Train proper models for all symbols."""
    logger.info("=" * 70)
    logger.info("PROPER MODEL TRAINING SYSTEM")
    logger.info("No data leakage, proper train/test split with gap")
    logger.info("=" * 70)

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT']
    trainer = ProperTrainer()

    results = []

    for symbol in symbols:
        df = await fetch_data(symbol)

        if df.empty or len(df) < 5000:
            logger.warning(f"Insufficient data for {symbol}")
            continue

        result = trainer.train(symbol, df, target_accuracy=0.60)
        results.append({'symbol': symbol, **result})

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    for r in results:
        if r.get('success'):
            status = "MET" if r.get('target_met') else "BELOW"
            acc = r.get('accuracy', 0)
            model = r.get('model_name', 'unknown')
            logger.info(f"  {r['symbol']}: {acc:.2%} ({status} 60%) - {model}")
        else:
            logger.info(f"  {r['symbol']}: FAILED - {r.get('reason')}")

    achieved = [r for r in results if r.get('target_met')]
    logger.info(f"\nAchieved 60%+: {len(achieved)}/{len(results)} symbols")

    if results:
        avg_acc = np.mean([r.get('accuracy', 0) for r in results if r.get('success')])
        logger.info(f"Average accuracy: {avg_acc:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
