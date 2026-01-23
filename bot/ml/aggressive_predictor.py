"""
Aggressive Profit Hunter - ML Predictor for High-Frequency Profit Generation.

Designed to achieve 1%+ daily returns through:
- Frequent trading with short prediction horizons
- Aggressive leverage on high-confidence signals
- Real-time learning from trade mistakes
- Multi-timeframe signal confirmation
- Adaptive position sizing based on recent performance
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        AdaBoostClassifier,
        ExtraTreesClassifier,
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .feature_engineer import FeatureEngineer
from .regime_classifier import MarketRegimeClassifier, MarketRegime

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels for position sizing."""

    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5


@dataclass
class TradeOutcome:
    """Record of a trade outcome for learning."""

    timestamp: datetime
    predicted_action: str
    actual_direction: str  # "UP", "DOWN", "FLAT"
    confidence: float
    features_snapshot: np.ndarray
    pnl_pct: float
    was_correct: bool
    regime: str
    holding_period: int  # bars


@dataclass
class AggressiveSignal:
    """Signal from aggressive predictor."""

    action: Literal["LONG", "SHORT", "FLAT"]
    confidence: float
    strength: SignalStrength
    recommended_leverage: float  # 1.0 - 10.0
    position_size_pct: float  # % of capital to use
    expected_move_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    max_hold_bars: int

    # Multi-timeframe agreement
    tf_1m_signal: Optional[str] = None
    tf_5m_signal: Optional[str] = None
    tf_15m_signal: Optional[str] = None
    tf_1h_signal: Optional[str] = None
    timeframe_agreement: float = 0.0

    # Learning metrics
    recent_win_rate: float = 0.5
    model_performance_score: float = 0.5
    regime: str = "unknown"

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "strength": self.strength.name,
            "recommended_leverage": round(self.recommended_leverage, 2),
            "position_size_pct": round(self.position_size_pct, 4),
            "expected_move_pct": round(self.expected_move_pct, 4),
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "max_hold_bars": self.max_hold_bars,
            "timeframe_agreement": round(self.timeframe_agreement, 4),
            "recent_win_rate": round(self.recent_win_rate, 4),
            "regime": self.regime,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LearningConfig:
    """Configuration for real-time learning."""

    # Learning window
    max_history: int = 500  # Max trades to remember
    learning_rate: float = 0.1  # How fast to adapt

    # Performance thresholds
    min_win_rate_for_aggressive: float = 0.55  # Need 55%+ win rate for leverage
    win_rate_boost_threshold: float = 0.65  # Boost sizing at 65%+ win rate

    # Adaptation speed
    fast_adapt_window: int = 10  # Fast adaptation on last 10 trades
    slow_adapt_window: int = 50  # Slow adaptation on last 50 trades

    # Mistake penalty
    consecutive_loss_penalty: float = 0.2  # Reduce size by 20% per consecutive loss
    max_consecutive_losses: int = 5  # After 5 losses, go conservative


@dataclass
class AggressiveConfig:
    """Configuration for aggressive predictor."""

    # Prediction horizons (bars)
    short_horizon: int = 1  # Scalping
    medium_horizon: int = 3  # Quick swing
    long_horizon: int = 8  # Day trade

    # Confidence thresholds
    min_confidence_to_trade: float = 0.55
    high_confidence_threshold: float = 0.70
    extreme_confidence_threshold: float = 0.80

    # Leverage settings
    base_leverage: float = 2.0
    max_leverage: float = 10.0
    leverage_confidence_multiplier: float = 0.1  # +0.1 leverage per 1% confidence above 55%

    # Position sizing
    base_position_pct: float = 0.05  # 5% of capital base
    max_position_pct: float = 0.25  # 25% max per trade

    # Risk per trade
    base_stop_loss_pct: float = 0.02  # 2% stop loss
    base_take_profit_pct: float = 0.03  # 3% take profit (1.5:1 R:R)

    # Multi-timeframe
    require_tf_agreement: bool = True
    min_tf_agreement: float = 0.6  # 60% timeframes must agree

    # Learning
    enable_learning: bool = True
    learning_config: LearningConfig = field(default_factory=LearningConfig)


class MistakeLearner:
    """
    Learns from trading mistakes to improve future predictions.

    Tracks:
    - Feature patterns that led to losses
    - Conditions where model underperforms
    - Regime-specific accuracy
    """

    def __init__(self, config: LearningConfig):
        self.config = config
        self.trade_history: Deque[TradeOutcome] = deque(maxlen=config.max_history)

        # Performance tracking
        self.regime_performance: Dict[str, List[bool]] = {}
        self.confidence_calibration: Dict[str, List[Tuple[float, bool]]] = {}
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        # Feature importance adjustment (learned from mistakes)
        self.feature_penalties: Dict[int, float] = {}  # feature_idx -> penalty

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for learning."""
        self.trade_history.append(outcome)

        # Update consecutive counts
        if outcome.was_correct:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Track regime performance
        regime = outcome.regime
        if regime not in self.regime_performance:
            self.regime_performance[regime] = []
        self.regime_performance[regime].append(outcome.was_correct)

        # Keep only recent history
        if len(self.regime_performance[regime]) > self.config.slow_adapt_window:
            self.regime_performance[regime] = self.regime_performance[regime][
                -self.config.slow_adapt_window :
            ]

        # Track confidence calibration
        conf_bucket = f"{int(outcome.confidence * 10) / 10:.1f}"
        if conf_bucket not in self.confidence_calibration:
            self.confidence_calibration[conf_bucket] = []
        self.confidence_calibration[conf_bucket].append((outcome.confidence, outcome.was_correct))

        logger.debug(
            f"Recorded outcome: {outcome.predicted_action} -> {'correct' if outcome.was_correct else 'wrong'}, "
            f"consecutive_losses={self.consecutive_losses}"
        )

    def get_win_rate(self, window: Optional[int] = None) -> float:
        """Get recent win rate."""
        if not self.trade_history:
            return 0.5

        history = list(self.trade_history)
        if window:
            history = history[-window:]

        if not history:
            return 0.5

        wins = sum(1 for t in history if t.was_correct)
        return wins / len(history)

    def get_regime_win_rate(self, regime: str) -> float:
        """Get win rate for specific regime."""
        history = self.regime_performance.get(regime, [])
        if not history:
            return 0.5
        return sum(history) / len(history)

    def get_confidence_adjustment(self, raw_confidence: float) -> float:
        """
        Adjust confidence based on historical calibration.

        If we've been overconfident, reduce. If underconfident, increase.
        """
        conf_bucket = f"{int(raw_confidence * 10) / 10:.1f}"
        calibration = self.confidence_calibration.get(conf_bucket, [])

        if len(calibration) < 10:
            return raw_confidence

        # Calculate actual accuracy at this confidence level
        actual_accuracy = sum(1 for _, correct in calibration if correct) / len(calibration)

        # Adjust: if we say 70% confident but only 50% accurate, reduce
        adjustment_factor = actual_accuracy / raw_confidence if raw_confidence > 0 else 1.0

        # Smooth adjustment
        adjusted = raw_confidence * (0.7 + 0.3 * adjustment_factor)
        return max(0.3, min(0.95, adjusted))

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on recent performance.

        Returns 0.2 - 2.0 multiplier.
        """
        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return 0.2  # Very conservative

        # Penalty for consecutive losses
        loss_penalty = 1.0 - (self.consecutive_losses * self.config.consecutive_loss_penalty)
        loss_penalty = max(0.3, loss_penalty)

        # Bonus for consecutive wins
        win_bonus = 1.0 + min(0.5, self.consecutive_wins * 0.1)

        # Recent win rate factor
        fast_win_rate = self.get_win_rate(self.config.fast_adapt_window)
        slow_win_rate = self.get_win_rate(self.config.slow_adapt_window)

        # Combine: fast adaptation weighted more
        combined_rate = fast_win_rate * 0.6 + slow_win_rate * 0.4

        # Scale to multiplier
        if combined_rate < 0.4:
            rate_factor = 0.3
        elif combined_rate < 0.5:
            rate_factor = 0.6
        elif combined_rate < 0.55:
            rate_factor = 0.9
        elif combined_rate < 0.6:
            rate_factor = 1.1
        elif combined_rate < 0.65:
            rate_factor = 1.3
        else:
            rate_factor = 1.5

        return loss_penalty * win_bonus * rate_factor

    def should_trade(self, regime: str) -> Tuple[bool, str]:
        """
        Determine if we should trade given current learning state.

        Returns (should_trade, reason).
        """
        # Too many consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"Too many consecutive losses ({self.consecutive_losses})"

        # Regime underperforming
        regime_rate = self.get_regime_win_rate(regime)
        if regime_rate < 0.35 and len(self.regime_performance.get(regime, [])) >= 20:
            return False, f"Regime {regime} underperforming (win rate: {regime_rate:.2%})"

        return True, "OK"


class AggressiveProfitHunter:
    """
    Aggressive ML Predictor designed for high-frequency profit generation.

    Key features:
    1. Multi-horizon predictions (scalping to day trading)
    2. Real-time learning from mistakes
    3. Adaptive leverage based on confidence and performance
    4. Multi-timeframe signal confirmation
    5. Regime-specific model selection
    """

    def __init__(
        self,
        config: Optional[AggressiveConfig] = None,
        model_dir: str = "data/models",
    ):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required")

        self.config = config or AggressiveConfig()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self.regime_classifier = MarketRegimeClassifier()
        self.scaler = StandardScaler()

        # Multi-horizon models
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}

        # Real-time learner
        self.learner = MistakeLearner(self.config.learning_config)

        # Online learning model (updates in real-time)
        self.online_model = SGDClassifier(
            loss="log_loss",
            learning_rate="adaptive",
            eta0=0.01,
            random_state=42,
        )
        self.online_trained = False

        self.feature_names: List[str] = []
        self.is_trained = False

        self._init_models()

    def _init_models(self) -> None:
        """Initialize diverse model ensemble."""
        # Fast model for quick decisions
        self.models["fast_rf"] = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model_weights["fast_rf"] = 1.0

        # Deep model for complex patterns
        self.models["deep_rf"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.model_weights["deep_rf"] = 1.2

        # Extra trees for diversity
        self.models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.model_weights["extra_trees"] = 1.0

        # Gradient boosting for sequential learning
        self.models["gradient_boost"] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.model_weights["gradient_boost"] = 1.1

        # AdaBoost for hard examples
        self.models["adaboost"] = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42,
        )
        self.model_weights["adaboost"] = 0.9

        if HAS_XGBOOST:
            self.models["xgboost"] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
            self.model_weights["xgboost"] = 1.3

        if HAS_LIGHTGBM:
            self.models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            self.model_weights["lightgbm"] = 1.3

    def _create_multi_horizon_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create targets for multiple prediction horizons."""
        df = df.copy()

        # Short horizon (1 bar) - scalping
        df["target_short"] = (
            df["close"].pct_change(self.config.short_horizon).shift(-self.config.short_horizon) > 0
        ).astype(int)

        # Medium horizon (3 bars) - quick swing
        df["target_medium"] = (
            df["close"].pct_change(self.config.medium_horizon).shift(-self.config.medium_horizon)
            > 0
        ).astype(int)

        # Long horizon (8 bars) - day trade
        df["target_long"] = (
            df["close"].pct_change(self.config.long_horizon).shift(-self.config.long_horizon) > 0
        ).astype(int)

        # Combined target: majority vote weighted by horizon importance
        # Short-term more important for frequent trading
        df["target_combined"] = (
            (df["target_short"] * 0.5 + df["target_medium"] * 0.3 + df["target_long"] * 0.2) >= 0.5
        ).astype(int)

        # Strong signal target: all horizons agree
        df["target_strong"] = (
            (df["target_short"] == df["target_medium"]) & (df["target_medium"] == df["target_long"])
        ).astype(int) * df["target_combined"]

        return df

    def _extract_aggressive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features optimized for short-term prediction."""
        # Base features from engineer
        df = self.feature_engineer.extract_features(df)

        # Add short-term momentum features
        for period in [1, 2, 3, 5]:
            df[f"momentum_{period}"] = df["close"].pct_change(period)
            df[f"momentum_{period}_acc"] = df[f"momentum_{period}"].diff()

        # Volume surge detection
        df["volume_surge"] = df["volume"] / df["volume"].rolling(10).mean()
        df["volume_trend"] = df["volume"].rolling(3).mean() / df["volume"].rolling(10).mean()

        # Price action features
        df["body_pct"] = (df["close"] - df["open"]) / df["open"]
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]

        # Breakout detection
        df["high_breakout"] = (df["high"] > df["high"].rolling(10).max().shift(1)).astype(int)
        df["low_breakout"] = (df["low"] < df["low"].rolling(10).min().shift(1)).astype(int)

        # Mean reversion signals
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_position"] = (df["close"] - bb_mid) / (2 * bb_std + 1e-10)
        df["mean_reversion_signal"] = -df["bb_position"]  # Negative when above band

        # Trend strength
        df["trend_strength"] = (
            abs(df["close"].rolling(10).mean() - df["close"].rolling(30).mean()) / df["close"]
        )

        # Volatility regime
        df["volatility_percentile"] = df["close"].pct_change().rolling(20).std().rank(pct=True)

        return df

    def train(self, ohlcv: pd.DataFrame, symbol: str = "unknown") -> Dict:
        """
        Train the aggressive predictor.

        Args:
            ohlcv: OHLCV DataFrame
            symbol: Symbol name

        Returns:
            Training metrics
        """
        logger.info(f"Training aggressive predictor for {symbol}...")

        # Extract features and targets
        df = self._extract_aggressive_features(ohlcv)
        df = self._create_multi_horizon_targets(df)
        df = df.dropna()

        if len(df) < 500:
            raise ValueError(f"Need 500+ rows, got {len(df)}")

        # Get feature columns
        exclude = {
            "target_short",
            "target_medium",
            "target_long",
            "target_combined",
            "target_strong",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        self.feature_names = [
            c
            for c in df.columns
            if c not in exclude
            and not c.startswith("target_")
            and not c.startswith("future_")
            and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        X = df[self.feature_names].values
        y = df["target_combined"].values
        y_strong = df["target_strong"].values

        # Replace inf/nan
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)

        # Time-series split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train all models
        metrics = {"models": {}}

        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)

                # Evaluate
                train_acc = model.score(X_train_scaled, y_train)
                test_acc = model.score(X_test_scaled, y_test)

                # Predict probabilities
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test_scaled)
                    # Calculate profit potential (sum of correct confident predictions)
                    pred = model.predict(X_test_scaled)
                    correct_confident = sum(
                        1 for i in range(len(pred)) if pred[i] == y_test[i] and max(proba[i]) > 0.6
                    )
                    profit_potential = correct_confident / len(y_test)
                else:
                    profit_potential = test_acc

                # Update model weight based on performance
                self.model_weights[name] = max(0.5, min(2.0, test_acc * 2))

                metrics["models"][name] = {
                    "train_accuracy": round(train_acc, 4),
                    "test_accuracy": round(test_acc, 4),
                    "profit_potential": round(profit_potential, 4),
                    "weight": round(self.model_weights[name], 4),
                }

                logger.info(f"  {name}: train={train_acc:.4f}, test={test_acc:.4f}")

            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")

        # Initialize online model
        if len(X_train_scaled) > 0:
            # Warm start with batch data
            self.online_model.partial_fit(X_train_scaled, y_train, classes=[0, 1])
            self.online_trained = True

        # Calculate overall metrics
        all_test_acc = [m["test_accuracy"] for m in metrics["models"].values()]

        metrics["overall"] = {
            "ensemble_accuracy": round(np.mean(all_test_acc), 4),
            "best_model": max(metrics["models"].items(), key=lambda x: x[1]["test_accuracy"])[0],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": len(self.feature_names),
            "trained_at": datetime.now().isoformat(),
        }

        self.is_trained = True
        logger.info(
            f"Training complete: ensemble_accuracy={metrics['overall']['ensemble_accuracy']:.4f}"
        )

        return metrics

    def predict(self, ohlcv: pd.DataFrame) -> AggressiveSignal:
        """
        Generate aggressive trading signal.

        Args:
            ohlcv: Recent OHLCV data (50+ bars)

        Returns:
            AggressiveSignal with action and sizing
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        # Get current regime
        try:
            regime_analysis = self.regime_classifier.classify(ohlcv)
            current_regime = regime_analysis.regime.value
        except Exception:
            current_regime = "unknown"

        # Check if we should trade based on learning
        should_trade, reason = self.learner.should_trade(current_regime)
        if not should_trade:
            logger.info(f"Learner suggests no trade: {reason}")
            return self._flat_signal(current_regime, reason)

        # Extract features
        df = self._extract_aggressive_features(ohlcv)
        if len(df) == 0:
            return self._flat_signal(current_regime, "No data")

        X = df[self.feature_names].iloc[-1:].values
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        predictions = []
        confidences = []

        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_scaled)[0]
                    pred = int(np.argmax(proba))
                    conf = float(max(proba))
                else:
                    pred = int(model.predict(X_scaled)[0])
                    conf = 0.6

                weight = self.model_weights.get(name, 1.0)
                predictions.append((pred, conf, weight))
                confidences.append(conf)

            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not predictions:
            return self._flat_signal(current_regime, "No predictions")

        # Weighted voting
        up_score = sum(w * c for p, c, w in predictions if p == 1)
        down_score = sum(w * c for p, c, w in predictions if p == 0)
        total_weight = sum(w for _, _, w in predictions)

        up_prob = up_score / total_weight if total_weight > 0 else 0.5
        down_prob = down_score / total_weight if total_weight > 0 else 0.5

        # Normalize
        total_prob = up_prob + down_prob
        if total_prob > 0:
            up_prob /= total_prob
            down_prob /= total_prob

        # Determine action
        raw_confidence = max(up_prob, down_prob)

        # Apply learned confidence adjustment
        if self.config.enable_learning:
            confidence = self.learner.get_confidence_adjustment(raw_confidence)
        else:
            confidence = raw_confidence

        # Determine action and strength
        if confidence < self.config.min_confidence_to_trade:
            action = "FLAT"
            strength = SignalStrength.WEAK
        elif up_prob > down_prob:
            action = "LONG"
            if confidence >= self.config.extreme_confidence_threshold:
                strength = SignalStrength.EXTREME
            elif confidence >= self.config.high_confidence_threshold:
                strength = SignalStrength.VERY_STRONG
            elif confidence >= 0.6:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
        else:
            action = "SHORT"
            if confidence >= self.config.extreme_confidence_threshold:
                strength = SignalStrength.EXTREME
            elif confidence >= self.config.high_confidence_threshold:
                strength = SignalStrength.VERY_STRONG
            elif confidence >= 0.6:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE

        # Calculate leverage and position size
        leverage, position_pct = self._calculate_sizing(confidence, strength, current_regime)

        # Calculate stop loss and take profit
        atr = df["close"].pct_change().rolling(14).std().iloc[-1] if len(df) > 14 else 0.02
        stop_loss_pct = max(self.config.base_stop_loss_pct, atr * 2)
        take_profit_pct = stop_loss_pct * 1.5  # 1.5:1 R:R minimum

        # Expected move
        expected_move = take_profit_pct * confidence - stop_loss_pct * (1 - confidence)

        # Max hold based on strength
        max_hold = {
            SignalStrength.EXTREME: self.config.long_horizon * 2,
            SignalStrength.VERY_STRONG: self.config.long_horizon,
            SignalStrength.STRONG: self.config.medium_horizon,
            SignalStrength.MODERATE: self.config.short_horizon,
            SignalStrength.WEAK: 1,
        }[strength]

        return AggressiveSignal(
            action=action,
            confidence=confidence,
            strength=strength,
            recommended_leverage=leverage,
            position_size_pct=position_pct,
            expected_move_pct=expected_move,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_hold_bars=max_hold,
            recent_win_rate=self.learner.get_win_rate(),
            model_performance_score=np.mean(confidences) if confidences else 0.5,
            regime=current_regime,
        )

    def _calculate_sizing(
        self,
        confidence: float,
        strength: SignalStrength,
        regime: str,
    ) -> Tuple[float, float]:
        """Calculate leverage and position size."""
        # Base values
        leverage = self.config.base_leverage
        position_pct = self.config.base_position_pct

        # Confidence adjustment
        confidence_excess = max(0, confidence - self.config.min_confidence_to_trade)
        leverage += confidence_excess * self.config.leverage_confidence_multiplier * 10

        # Strength multiplier
        strength_mult = {
            SignalStrength.EXTREME: 2.0,
            SignalStrength.VERY_STRONG: 1.5,
            SignalStrength.STRONG: 1.2,
            SignalStrength.MODERATE: 1.0,
            SignalStrength.WEAK: 0.5,
        }[strength]

        position_pct *= strength_mult

        # Learning adjustment
        if self.config.enable_learning:
            learning_mult = self.learner.get_position_size_multiplier()
            position_pct *= learning_mult

            # Reduce leverage if underperforming
            if learning_mult < 0.8:
                leverage *= learning_mult

        # Regime adjustment
        regime_mult = {
            "strong_bull": 1.3,
            "bull": 1.1,
            "sideways": 0.8,
            "bear": 0.7,
            "strong_bear": 0.5,
            "volatile": 0.6,
        }.get(regime.lower(), 1.0)

        position_pct *= regime_mult

        # Clamp to limits
        leverage = max(1.0, min(self.config.max_leverage, leverage))
        position_pct = max(0.01, min(self.config.max_position_pct, position_pct))

        return leverage, position_pct

    def _flat_signal(self, regime: str, reason: str = "") -> AggressiveSignal:
        """Return a FLAT signal."""
        return AggressiveSignal(
            action="FLAT",
            confidence=0.5,
            strength=SignalStrength.WEAK,
            recommended_leverage=1.0,
            position_size_pct=0.0,
            expected_move_pct=0.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.03,
            max_hold_bars=0,
            recent_win_rate=self.learner.get_win_rate(),
            model_performance_score=0.5,
            regime=regime,
        )

    def record_trade_outcome(
        self,
        predicted_action: str,
        actual_pnl_pct: float,
        confidence: float,
        features: np.ndarray,
        regime: str,
        holding_period: int,
    ) -> None:
        """
        Record trade outcome for learning.

        Call this after each trade completes to improve future predictions.
        """
        # Determine if correct
        if predicted_action == "LONG":
            was_correct = actual_pnl_pct > 0
            actual_direction = "UP" if actual_pnl_pct > 0 else "DOWN"
        elif predicted_action == "SHORT":
            was_correct = actual_pnl_pct > 0  # Short profits when price goes down
            actual_direction = "DOWN" if actual_pnl_pct > 0 else "UP"
        else:
            was_correct = abs(actual_pnl_pct) < 0.001  # FLAT is correct if no move
            actual_direction = "FLAT"

        outcome = TradeOutcome(
            timestamp=datetime.now(),
            predicted_action=predicted_action,
            actual_direction=actual_direction,
            confidence=confidence,
            features_snapshot=features,
            pnl_pct=actual_pnl_pct,
            was_correct=was_correct,
            regime=regime,
            holding_period=holding_period,
        )

        self.learner.record_outcome(outcome)

        # Update online model if we have features
        if self.online_trained and len(features) > 0:
            try:
                target = 1 if actual_direction == "UP" else 0
                self.online_model.partial_fit(features.reshape(1, -1), [target])
            except Exception as e:
                logger.debug(f"Online model update failed: {e}")

        logger.info(
            f"Trade outcome recorded: {predicted_action} -> {'WIN' if was_correct else 'LOSS'} "
            f"({actual_pnl_pct:+.2%}), win_rate={self.learner.get_win_rate():.2%}"
        )

    def save(self, name: str = "aggressive_predictor") -> None:
        """Save model and learning state."""
        import joblib

        # Save models
        for model_name, model in self.models.items():
            path = self.model_dir / f"{name}_{model_name}.pkl"
            joblib.dump(model, path)

        # Save scaler
        scaler_path = self.model_dir / f"{name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        # Save online model
        if self.online_trained:
            online_path = self.model_dir / f"{name}_online.pkl"
            joblib.dump(self.online_model, online_path)

        # Save metadata and learning state
        meta = {
            "feature_names": self.feature_names,
            "model_weights": self.model_weights,
            "is_trained": self.is_trained,
            "online_trained": self.online_trained,
            "config": {
                "short_horizon": self.config.short_horizon,
                "medium_horizon": self.config.medium_horizon,
                "long_horizon": self.config.long_horizon,
                "min_confidence_to_trade": self.config.min_confidence_to_trade,
                "base_leverage": self.config.base_leverage,
                "max_leverage": self.config.max_leverage,
            },
            "learning_state": {
                "consecutive_losses": self.learner.consecutive_losses,
                "consecutive_wins": self.learner.consecutive_wins,
                "regime_performance": {
                    k: list(v) for k, v in self.learner.regime_performance.items()
                },
            },
            "saved_at": datetime.now().isoformat(),
        }

        meta_path = self.model_dir / f"{name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Aggressive predictor saved to {self.model_dir}")

    def load(self, name: str = "aggressive_predictor") -> bool:
        """Load model and learning state."""
        import joblib

        meta_path = self.model_dir / f"{name}_meta.json"
        if not meta_path.exists():
            return False

        with open(meta_path) as f:
            meta = json.load(f)

        self.feature_names = meta["feature_names"]
        self.model_weights = meta["model_weights"]
        self.is_trained = meta["is_trained"]
        self.online_trained = meta.get("online_trained", False)

        # Load models
        for model_name in self.models.keys():
            path = self.model_dir / f"{name}_{model_name}.pkl"
            if path.exists():
                self.models[model_name] = joblib.load(path)

        # Load scaler
        scaler_path = self.model_dir / f"{name}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

        # Load online model
        online_path = self.model_dir / f"{name}_online.pkl"
        if online_path.exists():
            self.online_model = joblib.load(online_path)

        # Restore learning state
        learning_state = meta.get("learning_state", {})
        self.learner.consecutive_losses = learning_state.get("consecutive_losses", 0)
        self.learner.consecutive_wins = learning_state.get("consecutive_wins", 0)

        regime_perf = learning_state.get("regime_performance", {})
        self.learner.regime_performance = {k: list(v) for k, v in regime_perf.items()}

        logger.info(f"Aggressive predictor loaded from {self.model_dir}")
        return True

    def get_stats(self) -> Dict:
        """Get current predictor statistics."""
        return {
            "is_trained": self.is_trained,
            "models_count": len(self.models),
            "features_count": len(self.feature_names),
            "learning": {
                "win_rate_fast": self.learner.get_win_rate(10),
                "win_rate_slow": self.learner.get_win_rate(50),
                "consecutive_losses": self.learner.consecutive_losses,
                "consecutive_wins": self.learner.consecutive_wins,
                "trades_recorded": len(self.learner.trade_history),
            },
            "model_weights": self.model_weights,
        }


def create_aggressive_predictor(
    max_leverage: float = 5.0,
    enable_learning: bool = True,
    model_dir: str = "data/models",
) -> AggressiveProfitHunter:
    """Factory function to create aggressive predictor."""
    config = AggressiveConfig(
        max_leverage=max_leverage,
        enable_learning=enable_learning,
    )
    return AggressiveProfitHunter(config=config, model_dir=model_dir)
