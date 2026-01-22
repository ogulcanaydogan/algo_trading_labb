"""
Feature Engineering Module for ML-based Trading Predictions.

Extracts technical and statistical features from OHLCV data for ML models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import warnings
import pandas as pd

# Suppress PerformanceWarning for DataFrame fragmentation
# The copy() at the end of extract_features handles defragmentation
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    lookback_periods: List[int] = None

    def __post_init__(self):
        if self.ema_periods is None:
            self.ema_periods = [5, 10, 20, 50, 100]
        if self.lookback_periods is None:
            self.lookback_periods = [1, 3, 5, 10, 20]


class FeatureEngineer:
    """
    Extract features from OHLCV data for ML models.

    Creates a rich feature set including:
    - Price-based features (returns, momentum)
    - Technical indicators (EMA, RSI, MACD, BB, ATR, ADX)
    - Volume features
    - Statistical features (volatility, skewness)
    - Lagged features for sequence modeling
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def extract_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from OHLCV data.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with all extracted features
        """
        df = ohlcv.copy()
        df = df.sort_index()

        # Basic price features
        df = self._add_price_features(df)

        # Technical indicators
        df = self._add_ema_features(df)
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)

        # Statistical features
        df = self._add_statistical_features(df)

        # Time-based features
        df = self._add_time_features(df)

        # Lagged features
        df = self._add_lagged_features(df)

        # Target variable (future returns)
        df = self._add_target(df)

        # Sanitize data: replace infinity with NaN, then handle
        df = self._sanitize_features(df)

        # Drop rows with NaN values
        df = df.dropna()

        # Defragment DataFrame to fix PerformanceWarning
        df = df.copy()

        return df

    def _sanitize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize features by handling infinity and extreme values.

        This ensures the data is suitable for ML models that can't handle
        infinity or very large values.
        """
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Replace infinity with NaN
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Clip extreme values to prevent overflow
        # Use 99th percentile as clip threshold for each column
        for col in numeric_cols:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        # Clip to reasonable range based on data distribution
                        lower = col_data.quantile(0.001)
                        upper = col_data.quantile(0.999)
                        # Ensure we have valid numeric values for comparison
                        if (pd.notna(lower) and pd.notna(upper) and
                            isinstance(lower, (int, float)) and
                            isinstance(upper, (int, float)) and
                            float(lower) < float(upper)):
                            df[col] = (
                                df[col]
                                .astype(float)
                                .clip(lower=float(lower), upper=float(upper))
                            )
                except (TypeError, ValueError):
                    # Skip columns that can't be processed
                    continue

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns at different horizons
        for period in self.config.lookback_periods:
            df[f"return_{period}"] = df["close"].pct_change(period)
            df[f"log_return_{period}"] = np.log(df["close"] / df["close"].shift(period))

        # Price position relative to high/low
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

        # Gap (open vs previous close)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Intraday range
        df["intraday_range"] = (df["high"] - df["low"]) / df["close"]

        # Body size (open to close)
        df["body_size"] = abs(df["close"] - df["open"]) / df["close"]

        # Upper/lower shadows
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

        return df

    def _add_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA-based features."""
        for period in self.config.ema_periods:
            ema = EMAIndicator(close=df["close"], window=period).ema_indicator()
            df[f"ema_{period}"] = ema
            df[f"ema_{period}_dist"] = (df["close"] - ema) / df["close"]

        # EMA crossover signals
        if len(self.config.ema_periods) >= 2:
            fast_ema = df[f"ema_{self.config.ema_periods[0]}"]
            slow_ema = df[f"ema_{self.config.ema_periods[1]}"]
            df["ema_cross"] = (fast_ema - slow_ema) / df["close"]
            df["ema_cross_signal"] = np.where(fast_ema > slow_ema, 1, -1)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        rsi = RSIIndicator(close=df["close"], window=self.config.rsi_period)
        df["rsi"] = rsi.rsi()
        df["rsi_normalized"] = (df["rsi"] - 50) / 50

        # Stochastic
        stoch = StochasticOscillator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
            smooth_window=3
        )
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()
        df["stoch_cross"] = df["stoch_k"] - df["stoch_d"]

        # MACD
        macd = MACD(
            close=df["close"],
            window_slow=self.config.macd_slow,
            window_fast=self.config.macd_fast,
            window_sign=self.config.macd_signal
        )
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
        df["macd_normalized"] = df["macd_diff"] / df["close"]

        # ADX (trend strength)
        adx = ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.config.adx_period
        )
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()
        df["adx_trend"] = df["adx_pos"] - df["adx_neg"]

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # Bollinger Bands
        bb = BollingerBands(
            close=df["close"],
            window=self.config.bb_period,
            window_dev=self.config.bb_std
        )
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
        df["bb_position"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"] + 1e-8)

        # ATR
        atr = AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=self.config.atr_period
        )
        df["atr"] = atr.average_true_range()
        df["atr_normalized"] = df["atr"] / df["close"]

        # Historical volatility
        for period in [5, 10, 20]:
            df[f"volatility_{period}"] = df["close"].pct_change().rolling(period).std() * np.sqrt(252)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume change
        df["volume_change"] = df["volume"].pct_change()

        # Volume moving averages
        for period in [5, 10, 20]:
            df[f"volume_ma_{period}"] = df["volume"].rolling(period).mean()
            df[f"volume_ratio_{period}"] = df["volume"] / df[f"volume_ma_{period}"]

        # OBV
        obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        df["obv"] = obv.on_balance_volume()
        df["obv_change"] = df["obv"].pct_change()

        # VWAP (if we have enough data)
        try:
            vwap = VolumeWeightedAveragePrice(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                volume=df["volume"]
            )
            df["vwap"] = vwap.volume_weighted_average_price()
            df["vwap_dist"] = (df["close"] - df["vwap"]) / df["close"]
        except Exception:
            df["vwap"] = df["close"]
            df["vwap_dist"] = 0

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        returns = df["close"].pct_change()

        for period in [10, 20, 50]:
            # Rolling mean and std
            df[f"mean_{period}"] = returns.rolling(period).mean()
            df[f"std_{period}"] = returns.rolling(period).std()

            # Skewness
            df[f"skew_{period}"] = returns.rolling(period).skew()

            # Kurtosis
            df[f"kurt_{period}"] = returns.rolling(period).kurt()

            # Z-score of current return
            df[f"zscore_{period}"] = (returns - df[f"mean_{period}"]) / (df[f"std_{period}"] + 1e-8)

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features for sequence modeling."""
        lag_features = ["return_1", "rsi_normalized", "macd_normalized", "bb_position", "volume_change"]

        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 2, 3, 5]:
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for capturing temporal patterns.

        Features include:
        - Hour of day (cyclical encoding)
        - Day of week (cyclical encoding)
        - Month (cyclical encoding)
        - Session indicators (Asian, European, US)
        - Time since significant events
        - Day-of-month effects
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

            # === Hour of Day (cyclical encoding for 24-hour cycle) ===
            hour = df.index.hour
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

            # === Day of Week (cyclical encoding for 7-day cycle) ===
            dayofweek = df.index.dayofweek
            df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7)

            # === Month (cyclical encoding for 12-month cycle) ===
            month = df.index.month
            df["month_sin"] = np.sin(2 * np.pi * month / 12)
            df["month_cos"] = np.cos(2 * np.pi * month / 12)

            # === Day of Month Effects ===
            day = df.index.day
            df["day_of_month"] = day / 31  # Normalized

            # Month-end effect (last 5 days)
            days_in_month = df.index.to_series().apply(
                lambda x: pd.Timestamp(x.year, x.month, 1) + pd.offsets.MonthEnd(0)
            ).dt.day
            df["is_month_end"] = ((days_in_month - day) <= 5).astype(int)

            # Month-start effect (first 5 days)
            df["is_month_start"] = (day <= 5).astype(int)

            # === Trading Session Indicators ===
            # Crypto trades 24/7, but traditional market hours still matter for volume

            # Asian session (00:00 - 09:00 UTC)
            df["is_asian_session"] = ((hour >= 0) & (hour < 9)).astype(int)

            # European session (07:00 - 16:00 UTC)
            df["is_european_session"] = ((hour >= 7) & (hour < 16)).astype(int)

            # US session (13:00 - 22:00 UTC)
            df["is_us_session"] = ((hour >= 13) & (hour < 22)).astype(int)

            # Overlap periods (high liquidity)
            # London-NY overlap (13:00 - 16:00 UTC)
            df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)

            # === Weekend/Off-hours Indicators ===
            df["is_weekend"] = (dayofweek >= 5).astype(int)

            # Low liquidity hours (generally 20:00 - 05:00 UTC)
            df["is_low_liquidity"] = ((hour >= 20) | (hour < 5)).astype(int)

            # === Time Since Events ===
            # Bars since session open
            session_start = pd.Series((hour == 0) | (hour == 9) | (hour == 14), index=df.index)
            session_groups = (session_start != session_start.shift()).cumsum()
            df["bars_since_session_open"] = session_start.groupby(session_groups).cumcount()
            df["bars_since_session_open"] = df["bars_since_session_open"].clip(upper=50) / 50

            # === Minute of Hour (for intraday data) ===
            if hasattr(df.index, 'minute'):
                minute = df.index.minute
                df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
                df["minute_cos"] = np.cos(2 * np.pi * minute / 60)

            # === Quarter Indicators ===
            quarter = df.index.quarter
            df["is_q1"] = (quarter == 1).astype(int)
            df["is_q4"] = (quarter == 4).astype(int)  # Year-end effects

            # === Special Day Indicators ===
            # First trading day of month
            df["is_first_of_month"] = (day == 1).astype(int)

            # Monday effect
            df["is_monday"] = (dayofweek == 0).astype(int)

            # Friday effect
            df["is_friday"] = (dayofweek == 4).astype(int)

        # Defragment DataFrame after many inserts
        df = df.copy()
        return df

    def _add_target(self, df: pd.DataFrame, forward_periods: int = 1) -> pd.DataFrame:
        """
        Add improved target variables for ML training.

        Creates multiple forward-looking targets:
        1. Single-period return (original)
        2. Multi-horizon returns (1, 3, 5, 10 bars)
        3. Risk-adjusted targets considering volatility
        4. Trend continuation targets

        The multi-class target uses dynamic thresholds based on recent volatility
        to adapt to changing market conditions.
        """
        # Defragment DataFrame to avoid PerformanceWarning
        df = df.copy()

        # Single period return (original)
        df["target_return"] = df["close"].pct_change(forward_periods).shift(-forward_periods)

        # Multi-horizon returns for better signal
        for horizon in [1, 3, 5, 10]:
            df[f"future_return_{horizon}"] = df["close"].pct_change(horizon).shift(-horizon)

        # Classification targets
        df["target_direction"] = np.where(df["target_return"] > 0, 1, 0)

        # Improved multi-class target with dynamic threshold
        # Use rolling volatility for adaptive threshold
        rolling_vol = df["close"].pct_change().rolling(20).std()

        # Dynamic threshold: 0.5 * recent volatility (adapts to market conditions)
        # Use a minimum threshold to avoid too many signals in low-vol environments
        min_threshold = 0.001  # 0.1% minimum threshold
        dynamic_threshold = np.maximum(rolling_vol * 0.5, min_threshold)

        # Fill NaN thresholds with a reasonable default
        dynamic_threshold = dynamic_threshold.fillna(min_threshold)

        # Multi-class target (LONG=2, FLAT=1, SHORT=0)
        df["target_class"] = np.where(
            df["target_return"] > dynamic_threshold, 2,  # LONG
            np.where(df["target_return"] < -dynamic_threshold, 0, 1)  # SHORT or FLAT
        )

        # Trend continuation target (does trend continue for multiple bars?)
        # This helps the model learn sustained moves vs noise
        future_3 = df["future_return_3"].fillna(0)
        future_5 = df["future_return_5"].fillna(0)

        # Strong trend: positive returns at multiple horizons
        df["target_strong_trend"] = np.where(
            (future_3 > dynamic_threshold) & (future_5 > dynamic_threshold * 1.5), 2,  # Strong bullish
            np.where(
                (future_3 < -dynamic_threshold) & (future_5 < -dynamic_threshold * 1.5), 0,  # Strong bearish
                1  # No strong trend
            )
        )

        # Risk-adjusted target (Sharpe-like for each bar)
        # Reward moves relative to recent volatility
        df["target_risk_adjusted"] = df["target_return"] / (rolling_vol + 1e-8)

        return df

    def add_triple_barrier_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 10,
        atr_multiplier: float = 2.0,
        min_return: float = 0.001,
    ) -> pd.DataFrame:
        """
        Add triple-barrier labels based on ATR scaled thresholds.

        Labels: 2=LONG, 1=FLAT, 0=SHORT
        """
        if "atr" not in df.columns:
            atr = AverageTrueRange(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=self.config.atr_period,
            ).average_true_range()
            df["atr"] = atr

        entries = df["close"]
        atr = df["atr"].fillna(method="ffill")
        upper = entries * (1 + (atr_multiplier * atr / entries).clip(lower=min_return))
        lower = entries * (1 - (atr_multiplier * atr / entries).clip(lower=min_return))

        labels = np.ones(len(df), dtype=int)
        for idx in range(len(df)):
            end = min(idx + horizon, len(df) - 1)
            window = df.iloc[idx + 1 : end + 1]
            if window.empty:
                continue
            hit_upper = (window["high"] >= upper.iloc[idx]).any()
            hit_lower = (window["low"] <= lower.iloc[idx]).any()
            if hit_upper and not hit_lower:
                labels[idx] = 2
            elif hit_lower and not hit_upper:
                labels[idx] = 0
            else:
                labels[idx] = 1

        df["target_triple_barrier"] = labels
        return df

    def build_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        use_triple_barrier: bool = False,
        atr_multiplier: float = 2.0,
        min_return: float = 0.001,
    ) -> pd.DataFrame:
        """
        Build target labels with volatility-adjusted thresholds.
        """
        df = self._add_target(df, forward_periods=horizon)
        if use_triple_barrier:
            df = self.add_triple_barrier_labels(
                df,
                horizon=horizon,
                atr_multiplier=atr_multiplier,
                min_return=min_return,
            )
        return df

    def validate_feature_alignment(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        target_col: str = "target_return",
        horizon: int = 1,
        min_corr: float = 0.9,
    ) -> Tuple[bool, str]:
        """Validate that target_return matches expected forward returns."""
        expected = df[price_col].pct_change(horizon).shift(-horizon)
        aligned = pd.concat([expected, df[target_col]], axis=1).dropna()
        if aligned.empty:
            return False, "Target alignment failed: no overlapping samples."
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if corr is None or np.isnan(corr):
            return False, "Target alignment failed: correlation undefined."
        if corr < min_corr:
            return False, f"Target alignment low: corr={corr:.2f} (min {min_corr})."
        return True, f"Target alignment OK: corr={corr:.2f}."

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names (excluding targets)."""
        exclude = ["target_return", "target_direction", "target_class",
                   "open", "high", "low", "close", "volume"]
        return [col for col in self.feature_columns if col not in exclude]

    @property
    def feature_columns(self) -> List[str]:
        """List of all feature columns."""
        return [
            # Price features
            "return_1", "return_3", "return_5", "return_10", "return_20",
            "log_return_1", "log_return_3", "log_return_5",
            "price_position", "gap", "intraday_range", "body_size",
            "upper_shadow", "lower_shadow",
            # EMA features
            "ema_5_dist", "ema_10_dist", "ema_20_dist", "ema_50_dist",
            "ema_cross", "ema_cross_signal",
            # Momentum
            "rsi", "rsi_normalized", "stoch_k", "stoch_d", "stoch_cross",
            "macd", "macd_signal", "macd_diff", "macd_normalized",
            "adx", "adx_pos", "adx_neg", "adx_trend",
            # Volatility
            "bb_width", "bb_position", "atr_normalized",
            "volatility_5", "volatility_10", "volatility_20",
            # Volume
            "volume_change", "volume_ratio_5", "volume_ratio_10",
            "obv_change", "vwap_dist",
            # Statistical
            "mean_10", "std_10", "skew_10", "kurt_10", "zscore_10",
            "mean_20", "std_20", "skew_20", "kurt_20", "zscore_20",
            # Time-based features
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "day_of_month",
            "is_month_end", "is_month_start",
            "is_asian_session", "is_european_session", "is_us_session",
            "is_overlap", "is_weekend", "is_low_liquidity",
            "bars_since_session_open",
            "is_q1", "is_q4", "is_first_of_month",
            "is_monday", "is_friday",
        ]
