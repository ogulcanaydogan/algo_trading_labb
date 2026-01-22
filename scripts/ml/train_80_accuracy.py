#!/usr/bin/env python3
"""
Advanced ML Training for 80%+ Accuracy

Implements:
1. 730 days of training data
2. Enhanced feature engineering (60+ indicators)
3. Triple-barrier labeling for better targets
4. Multiple models: RF, GB, XGBoost, LightGBM
5. Performance-weighted ensemble
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not available")


def fetch_data(symbol: str, days: int = 730) -> Optional[pd.DataFrame]:
    """Fetch historical data using yfinance."""
    import yfinance as yf

    try:
        if "/" in symbol:
            yf_symbol = symbol.replace("/USDT", "-USD").replace("/USD", "-USD")
        else:
            yf_symbol = symbol

        logger.info(f"Fetching {days} days of hourly data for {symbol}...")
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=f"{days}d", interval="1h")

        if df.empty:
            logger.error(f"No data for {symbol}")
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        logger.info(f"  Got {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with 60+ indicators.
    """
    df = df.copy()

    # === PRICE-BASED FEATURES ===

    # Multiple EMAs
    for period in [5, 8, 13, 21, 34, 55, 89, 144, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']

    # EMA crossovers
    df['ema_8_21_cross'] = (df['ema_8'] > df['ema_21']).astype(int)
    df['ema_21_55_cross'] = (df['ema_21'] > df['ema_55']).astype(int)
    df['ema_55_200_cross'] = (df['ema_55'] > df['ema_200']).astype(int)

    # Price momentum
    for period in [1, 2, 4, 8, 12, 24, 48, 72, 168]:
        df[f'return_{period}h'] = df['close'].pct_change(period)

    # === VOLATILITY FEATURES ===

    # ATR (Average True Range)
    for period in [14, 21, 50]:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(period).mean()
        df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

    # Bollinger Bands
    for period in [20, 50]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = sma + 2 * std
        df[f'bb_lower_{period}'] = sma - 2 * std
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

    # Historical volatility
    for period in [12, 24, 48, 168]:
        df[f'volatility_{period}h'] = df['return_1h'].rolling(period).std() * np.sqrt(period)

    # Volatility regime
    df['vol_regime'] = pd.qcut(df['volatility_24h'].rank(method='first'), q=5, labels=[0, 1, 2, 3, 4]).astype(float)

    # === MOMENTUM INDICATORS ===

    # RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # RSI divergence
    df['rsi_divergence'] = df['rsi_14'].diff(5) - (df['close'].pct_change(5) * 100)

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

    # Stochastic
    for period in [14, 21]:
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()

    # Williams %R
    df['williams_r'] = -100 * (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min())

    # CCI (Commodity Channel Index)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(20).mean()
    mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

    # === VOLUME FEATURES ===

    # Volume momentum
    for period in [4, 12, 24]:
        df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']

    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=21).mean()
    df['obv_divergence'] = df['obv'] - df['obv_ema']

    # VWAP approximation
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']

    # === PATTERN FEATURES ===

    # Candlestick patterns
    df['body'] = df['close'] - df['open']
    df['body_pct'] = df['body'] / df['open']
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 0.0001)

    # Doji detection
    df['is_doji'] = (abs(df['body_pct']) < 0.001).astype(int)

    # Trend strength (ADX approximation)
    df['trend_strength'] = abs(df['ema_21_dist']) * 100

    # === TIME FEATURES ===

    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # === MULTI-TIMEFRAME FEATURES ===

    # 4-hour aggregates
    df['high_4h'] = df['high'].rolling(4).max()
    df['low_4h'] = df['low'].rolling(4).min()
    df['close_4h_change'] = df['close'].pct_change(4)

    # Daily aggregates
    df['high_24h'] = df['high'].rolling(24).max()
    df['low_24h'] = df['low'].rolling(24).min()
    df['range_24h'] = (df['high_24h'] - df['low_24h']) / df['close']

    return df


def create_triple_barrier_labels(
    df: pd.DataFrame,
    profit_take: float = 0.02,
    stop_loss: float = 0.01,
    max_holding: int = 24
) -> pd.Series:
    """
    Triple-barrier labeling for better targets.

    Labels:
    - 2: LONG (hit profit target)
    - 1: FLAT (hit max holding time without trigger)
    - 0: SHORT (hit stop loss)
    """
    labels = []

    for i in range(len(df) - max_holding):
        entry_price = df['close'].iloc[i]

        # Look forward up to max_holding periods
        for j in range(1, max_holding + 1):
            if i + j >= len(df):
                labels.append(1)  # FLAT
                break

            future_high = df['high'].iloc[i + j]
            future_low = df['low'].iloc[i + j]

            # Check profit target (upper barrier)
            if (future_high - entry_price) / entry_price >= profit_take:
                labels.append(2)  # LONG
                break

            # Check stop loss (lower barrier)
            if (entry_price - future_low) / entry_price >= stop_loss:
                labels.append(0)  # SHORT
                break

            # Max holding time reached (vertical barrier)
            if j == max_holding:
                # Use final return to decide
                final_price = df['close'].iloc[i + j]
                ret = (final_price - entry_price) / entry_price
                if ret > 0.005:
                    labels.append(2)  # LONG
                elif ret < -0.005:
                    labels.append(0)  # SHORT
                else:
                    labels.append(1)  # FLAT

    # Pad remaining rows
    labels.extend([1] * max_holding)

    return pd.Series(labels, index=df.index)


def prepare_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare features and labels for training."""

    # Create labels
    df['target'] = create_triple_barrier_labels(df)

    # Get feature columns (exclude OHLCV and target)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'target',
                    'dividends', 'stock splits', 'capital gains']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('bb_upper') and not c.startswith('bb_lower')]

    # Remove rows with NaN
    df_clean = df.dropna()

    X = df_clean[feature_cols].values
    y = df_clean['target'].values

    # Handle any remaining infinities
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"  Features: {len(feature_cols)}, Samples: {len(X)}")
    logger.info(f"  Class distribution: LONG={sum(y==2)}, FLAT={sum(y==1)}, SHORT={sum(y==0)}")

    return X, y, feature_cols


def train_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    symbol: str,
    output_dir: Path
) -> Dict[str, float]:
    """Train multiple models and save them."""

    results = {}

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    # === RANDOM FOREST ===
    logger.info("  Training Random Forest...")
    rf_scores = []
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

    for train_idx, val_idx in tscv.split(X_scaled):
        rf_model.fit(X_scaled[train_idx], y[train_idx])
        pred = rf_model.predict(X_scaled[val_idx])
        rf_scores.append(accuracy_score(y[val_idx], pred))

    rf_cv = np.mean(rf_scores)
    results['random_forest'] = rf_cv
    logger.info(f"    Random Forest CV: {rf_cv:.2%} ± {np.std(rf_scores):.2%}")

    # Train final model on all data
    rf_model.fit(X_scaled, y)

    # === GRADIENT BOOSTING ===
    logger.info("  Training Gradient Boosting...")
    gb_scores = []
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    )

    for train_idx, val_idx in tscv.split(X_scaled):
        gb_model.fit(X_scaled[train_idx], y[train_idx])
        pred = gb_model.predict(X_scaled[val_idx])
        gb_scores.append(accuracy_score(y[val_idx], pred))

    gb_cv = np.mean(gb_scores)
    results['gradient_boosting'] = gb_cv
    logger.info(f"    Gradient Boosting CV: {gb_cv:.2%} ± {np.std(gb_scores):.2%}")

    gb_model.fit(X_scaled, y)

    # === XGBOOST ===
    if XGB_AVAILABLE:
        logger.info("  Training XGBoost...")
        xgb_scores = []
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )

        for train_idx, val_idx in tscv.split(X_scaled):
            xgb_model.fit(X_scaled[train_idx], y[train_idx])
            pred = xgb_model.predict(X_scaled[val_idx])
            xgb_scores.append(accuracy_score(y[val_idx], pred))

        xgb_cv = np.mean(xgb_scores)
        results['xgboost'] = xgb_cv
        logger.info(f"    XGBoost CV: {xgb_cv:.2%} ± {np.std(xgb_scores):.2%}")

        xgb_model.fit(X_scaled, y)

    # === LIGHTGBM ===
    if LGB_AVAILABLE:
        logger.info("  Training LightGBM...")
        lgb_scores = []
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multiclass',
            num_class=3,
            random_state=42,
            verbose=-1
        )

        for train_idx, val_idx in tscv.split(X_scaled):
            lgb_model.fit(X_scaled[train_idx], y[train_idx])
            pred = lgb_model.predict(X_scaled[val_idx])
            lgb_scores.append(accuracy_score(y[val_idx], pred))

        lgb_cv = np.mean(lgb_scores)
        results['lightgbm'] = lgb_cv
        logger.info(f"    LightGBM CV: {lgb_cv:.2%} ± {np.std(lgb_scores):.2%}")

        lgb_model.fit(X_scaled, y)

    # === SAVE MODELS ===
    symbol_clean = symbol.replace('/', '_')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scaler
    joblib.dump(scaler, output_dir / f"{symbol_clean}_scaler_v2.pkl")

    # Save models
    joblib.dump(rf_model, output_dir / f"{symbol_clean}_random_forest_model.pkl")
    joblib.dump(scaler, output_dir / f"{symbol_clean}_random_forest_scaler.pkl")

    joblib.dump(gb_model, output_dir / f"{symbol_clean}_gradient_boosting_model.pkl")
    joblib.dump(scaler, output_dir / f"{symbol_clean}_gradient_boosting_scaler.pkl")

    if XGB_AVAILABLE:
        joblib.dump(xgb_model, output_dir / f"{symbol_clean}_xgboost_model.pkl")
        joblib.dump(scaler, output_dir / f"{symbol_clean}_xgboost_scaler.pkl")

    if LGB_AVAILABLE:
        joblib.dump(lgb_model, output_dir / f"{symbol_clean}_lightgbm_model.pkl")
        joblib.dump(scaler, output_dir / f"{symbol_clean}_lightgbm_scaler.pkl")

    # Save metadata
    metadata = {
        'symbol': symbol,
        'trained_at': datetime.now().isoformat(),
        'features': feature_cols,
        'num_features': len(feature_cols),
        'num_samples': len(y),
        'class_distribution': {
            'long': int(sum(y == 2)),
            'flat': int(sum(y == 1)),
            'short': int(sum(y == 0))
        },
        'models': {
            'random_forest': {'cv_accuracy': float(rf_cv)},
            'gradient_boosting': {'cv_accuracy': float(gb_cv)},
        }
    }

    if XGB_AVAILABLE:
        metadata['models']['xgboost'] = {'cv_accuracy': float(xgb_cv)}
    if LGB_AVAILABLE:
        metadata['models']['lightgbm'] = {'cv_accuracy': float(lgb_cv)}

    with open(output_dir / f"{symbol_clean}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Models saved to {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train ML models for 80%+ accuracy")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--output-dir", type=str, default="data/models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    all_results = {}

    logger.info("=" * 60)
    logger.info("ADVANCED ML TRAINING FOR 80%+ ACCURACY")
    logger.info("=" * 60)
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Training data: {args.days} days")
    logger.info(f"Features: 60+ technical indicators")
    logger.info(f"Labeling: Triple-barrier method")
    logger.info("=" * 60)

    for symbol in args.symbols:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*40}")

        # Fetch data
        df = fetch_data(symbol, args.days)
        if df is None:
            continue

        # Engineer features
        logger.info("  Engineering features...")
        df = engineer_features(df)

        # Prepare dataset
        X, y, feature_cols = prepare_dataset(df)

        if len(X) < 500:
            logger.warning(f"  Insufficient data for {symbol}: {len(X)} samples")
            continue

        # Train models
        results = train_models(X, y, feature_cols, symbol, output_dir)
        all_results[symbol] = results

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)

    total_accuracy = []
    for symbol, results in all_results.items():
        logger.info(f"\n{symbol}:")
        for model, accuracy in results.items():
            logger.info(f"  {model}: {accuracy:.2%}")
            total_accuracy.append(accuracy)

    if total_accuracy:
        avg = np.mean(total_accuracy)
        best = max(total_accuracy)
        logger.info(f"\nAverage CV Accuracy: {avg:.2%}")
        logger.info(f"Best Model Accuracy: {best:.2%}")

        if best >= 0.75:
            logger.info("Good progress toward 80% target!")
        elif best >= 0.65:
            logger.info("Moderate progress - consider more data or features")
        else:
            logger.info("Need significant improvements for 80% target")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
