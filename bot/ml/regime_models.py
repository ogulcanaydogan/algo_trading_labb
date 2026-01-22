"""
Market Regime-Specific Models

Different market regimes require different trading strategies:
1. Bull Market: Momentum, buy dips
2. Bear Market: Mean reversion, sell rallies
3. Sideways/Range: Mean reversion, fade extremes
4. High Volatility: Smaller positions, wider stops
5. Low Volatility: Larger positions, trend following

This module trains and manages regime-specific models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CRASH = "crash"
    RECOVERY = "recovery"


@dataclass
class RegimeDetection:
    """Result of regime detection."""
    regime: MarketRegime
    confidence: float
    volatility_percentile: float
    trend_strength: float
    support_level: float
    resistance_level: float


class RegimeDetector:
    """
    Detects current market regime from price data.
    """

    def __init__(
        self,
        trend_lookback: int = 50,
        volatility_lookback: int = 20,
        breakout_threshold: float = 0.02
    ):
        self.trend_lookback = trend_lookback
        self.volatility_lookback = volatility_lookback
        self.breakout_threshold = breakout_threshold
        self.volatility_history: List[float] = []

    def detect(self, prices: pd.Series, volumes: Optional[pd.Series] = None) -> RegimeDetection:
        """
        Detect current market regime.

        Args:
            prices: Price series
            volumes: Optional volume series

        Returns:
            RegimeDetection with regime and metrics
        """
        if len(prices) < self.trend_lookback:
            return RegimeDetection(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.5,
                volatility_percentile=0.5,
                trend_strength=0,
                support_level=prices.min(),
                resistance_level=prices.max()
            )

        # Calculate metrics
        returns = prices.pct_change().dropna()

        # Trend detection
        sma_short = prices.rolling(10).mean().iloc[-1]
        sma_long = prices.rolling(self.trend_lookback).mean().iloc[-1]
        current_price = prices.iloc[-1]

        trend_strength = (sma_short - sma_long) / sma_long if sma_long > 0 else 0

        # Volatility
        current_vol = returns.iloc[-self.volatility_lookback:].std()
        historical_vol = returns.std()
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1

        self.volatility_history.append(current_vol)
        if len(self.volatility_history) > 100:
            vol_percentile = sum(1 for v in self.volatility_history if v < current_vol) / len(self.volatility_history)
        else:
            vol_percentile = 0.5

        # Support/Resistance
        lookback_prices = prices.iloc[-self.trend_lookback:]
        support = lookback_prices.min()
        resistance = lookback_prices.max()

        # Breakout detection
        range_size = resistance - support
        is_breakout_up = current_price > resistance * (1 - self.breakout_threshold / 2)
        is_breakout_down = current_price < support * (1 + self.breakout_threshold / 2)

        # Crash detection (rapid decline with high volume)
        recent_return = (current_price - prices.iloc[-5]) / prices.iloc[-5]
        is_crash = recent_return < -0.1 and vol_ratio > 2

        # Recovery detection (bounce from lows)
        is_recovery = recent_return > 0.05 and current_price > support * 1.05

        # Determine regime
        regime = MarketRegime.SIDEWAYS
        confidence = 0.5

        if is_crash:
            regime = MarketRegime.CRASH
            confidence = min(0.9, 0.6 + abs(recent_return))
        elif is_recovery:
            regime = MarketRegime.RECOVERY
            confidence = 0.65
        elif is_breakout_up:
            regime = MarketRegime.BREAKOUT
            confidence = 0.7
        elif is_breakout_down:
            regime = MarketRegime.BEAR_TREND
            confidence = 0.7
        elif vol_percentile > 0.8:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = vol_percentile
        elif vol_percentile < 0.2:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 1 - vol_percentile
        elif trend_strength > 0.02:
            regime = MarketRegime.BULL_TREND
            confidence = min(0.85, 0.5 + trend_strength * 10)
        elif trend_strength < -0.02:
            regime = MarketRegime.BEAR_TREND
            confidence = min(0.85, 0.5 + abs(trend_strength) * 10)

        return RegimeDetection(
            regime=regime,
            confidence=confidence,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            support_level=support,
            resistance_level=resistance
        )


class RegimeSpecificModel:
    """
    Model trained specifically for a market regime.
    """

    def __init__(
        self,
        regime: MarketRegime,
        model_type: str = 'random_forest'
    ):
        self.regime = regime
        self.model_type = model_type
        self.model: Optional[BaseEstimator] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.accuracy = 0.0
        self.n_samples = 0

    def train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Train model on regime-specific data.

        Returns accuracy.
        """
        if len(X) < 100:
            logger.warning(f"Insufficient data for {self.regime.value}: {len(X)} samples")
            return 0.0

        self.feature_names = list(X.columns)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Create model with regime-specific hyperparameters
        if self.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
            # More conservative for volatile regimes
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,  # Shallower to avoid overfitting to noise
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            # Deeper for trending regimes
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=25,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        else:
            # Balanced for sideways
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        self.model.fit(X_train, y_train)
        self.accuracy = accuracy_score(y_test, self.model.predict(X_test))
        self.n_samples = len(X)

        logger.info(f"{self.regime.value} model: {self.accuracy:.2%} accuracy on {self.n_samples} samples")

        return self.accuracy

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using trained model."""
        if self.model is None:
            raise ValueError("Model not trained")

        # Align features
        X_aligned = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X
        X_scaled = self.scaler.transform(X_aligned)

        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model not trained or doesn't support probabilities")

        X_aligned = X[self.feature_names] if set(self.feature_names).issubset(X.columns) else X
        X_scaled = self.scaler.transform(X_aligned)

        return self.model.predict_proba(X_scaled)


class RegimeModelEnsemble:
    """
    Ensemble of regime-specific models.

    Selects appropriate model based on detected regime.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.regime_detector = RegimeDetector()
        self.models: Dict[MarketRegime, RegimeSpecificModel] = {}
        self.fallback_model: Optional[BaseEstimator] = None
        self.fallback_scaler = StandardScaler()

    def train_all_regimes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series
    ) -> Dict[str, float]:
        """
        Train models for all regimes.

        Args:
            X: Full feature DataFrame
            y: Full target Series
            prices: Price series for regime detection

        Returns:
            Dict of regime -> accuracy
        """
        logger.info(f"Training regime-specific models for {self.symbol}")

        # Detect regime for each sample
        regimes = []
        for i in range(len(prices)):
            if i < 50:
                regimes.append(MarketRegime.SIDEWAYS)
            else:
                detection = self.regime_detector.detect(prices.iloc[:i+1])
                regimes.append(detection.regime)

        regime_series = pd.Series(regimes, index=prices.index)

        # Train model for each regime
        results = {}

        for regime in MarketRegime:
            mask = regime_series == regime
            X_regime = X[mask]
            y_regime = y[mask]

            if len(X_regime) >= 100:
                model = RegimeSpecificModel(regime)
                accuracy = model.train(X_regime, y_regime)
                self.models[regime] = model
                results[regime.value] = accuracy
            else:
                logger.info(f"Skipping {regime.value}: only {len(X_regime)} samples")

        # Train fallback model on all data
        X_scaled = self.fallback_scaler.fit_transform(X)
        split_idx = int(len(X) * 0.8)

        self.fallback_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.fallback_model.fit(X_scaled[:split_idx], y.iloc[:split_idx])
        fallback_acc = accuracy_score(y.iloc[split_idx:], self.fallback_model.predict(X_scaled[split_idx:]))
        results['fallback'] = fallback_acc

        logger.info(f"Trained {len(self.models)} regime models + fallback")

        return results

    def predict(
        self,
        X: pd.DataFrame,
        prices: pd.Series
    ) -> Tuple[np.ndarray, List[MarketRegime]]:
        """
        Predict using regime-appropriate model.

        Args:
            X: Features
            prices: Prices for regime detection

        Returns:
            Predictions and detected regimes
        """
        predictions = []
        regimes_used = []

        for i in range(len(X)):
            # Detect regime
            price_history = prices.iloc[:i+1] if i < len(prices) else prices
            detection = self.regime_detector.detect(price_history)

            # Select model
            if detection.regime in self.models:
                model = self.models[detection.regime]
                pred = model.predict(X.iloc[[i]])[0]
            elif self.fallback_model is not None:
                X_scaled = self.fallback_scaler.transform(X.iloc[[i]])
                pred = self.fallback_model.predict(X_scaled)[0]
            else:
                pred = 0  # HOLD

            predictions.append(pred)
            regimes_used.append(detection.regime)

        return np.array(predictions), regimes_used

    def get_regime_signal(
        self,
        X: pd.DataFrame,
        prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Get trading signal with regime context.

        Returns comprehensive signal with regime info.
        """
        detection = self.regime_detector.detect(prices)

        if detection.regime in self.models:
            model = self.models[detection.regime]
            pred = model.predict(X.iloc[[-1]])[0]
            proba = model.predict_proba(X.iloc[[-1]])[0]
            model_accuracy = model.accuracy
        elif self.fallback_model is not None:
            X_scaled = self.fallback_scaler.transform(X.iloc[[-1]])
            pred = self.fallback_model.predict(X_scaled)[0]
            proba = self.fallback_model.predict_proba(X_scaled)[0]
            model_accuracy = 0.5
        else:
            pred = 0
            proba = [0.33, 0.34, 0.33]
            model_accuracy = 0.33

        # Adjust confidence based on regime
        base_confidence = max(proba)

        # Reduce confidence in volatile/uncertain regimes
        if detection.regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
            confidence = base_confidence * 0.8
        elif detection.regime in [MarketRegime.BULL_TREND, MarketRegime.BEAR_TREND]:
            confidence = base_confidence * 1.1  # Boost in clear trends
        else:
            confidence = base_confidence

        confidence = np.clip(confidence, 0, 0.95)

        # Map prediction to action
        action_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
        action = action_map.get(pred, 'HOLD')

        return {
            'action': action,
            'confidence': confidence,
            'regime': detection.regime.value,
            'regime_confidence': detection.confidence,
            'volatility_percentile': detection.volatility_percentile,
            'trend_strength': detection.trend_strength,
            'support': detection.support_level,
            'resistance': detection.resistance_level,
            'model_accuracy': model_accuracy,
            'raw_probabilities': proba
        }

    def save(self, path: Path):
        """Save all models."""
        path.mkdir(parents=True, exist_ok=True)

        for regime, model in self.models.items():
            model_path = path / f"{self.symbol.replace('/', '_')}_{regime.value}.pkl"
            joblib.dump(model, model_path)

        if self.fallback_model is not None:
            fallback_path = path / f"{self.symbol.replace('/', '_')}_fallback.pkl"
            joblib.dump((self.fallback_model, self.fallback_scaler), fallback_path)

    def load(self, path: Path):
        """Load all models."""
        for regime in MarketRegime:
            model_path = path / f"{self.symbol.replace('/', '_')}_{regime.value}.pkl"
            if model_path.exists():
                self.models[regime] = joblib.load(model_path)

        fallback_path = path / f"{self.symbol.replace('/', '_')}_fallback.pkl"
        if fallback_path.exists():
            self.fallback_model, self.fallback_scaler = joblib.load(fallback_path)


# Factory function
def create_regime_ensemble(
    symbol: str,
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.Series
) -> RegimeModelEnsemble:
    """
    Create and train regime ensemble for symbol.

    Args:
        symbol: Trading symbol
        X: Features
        y: Labels
        prices: Price series

    Returns:
        Trained RegimeModelEnsemble
    """
    ensemble = RegimeModelEnsemble(symbol)
    results = ensemble.train_all_regimes(X, y, prices)

    logger.info(f"\nRegime Model Results for {symbol}:")
    for regime, accuracy in results.items():
        logger.info(f"  {regime}: {accuracy:.2%}")

    return ensemble
