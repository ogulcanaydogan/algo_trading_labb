"""
Online Learning Module.

Enables incremental learning from live trading outcomes.
Models adapt continuously from each trade without full retraining.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union
import copy

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeExperience:
    """A single trade experience for learning."""

    timestamp: datetime
    symbol: str
    features: np.ndarray  # Feature vector at entry
    action: str  # LONG, SHORT, FLAT
    action_idx: int  # 0=SHORT, 1=FLAT, 2=LONG
    outcome: str  # WIN, LOSS, BREAKEVEN
    pnl: float  # Profit/loss amount
    pnl_pct: float  # Profit/loss percentage
    entry_price: float
    exit_price: float
    hold_time_minutes: int
    confidence: float  # Model's confidence at entry
    regime: str  # Market regime at entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "action_idx": self.action_idx,
            "outcome": self.outcome,
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 4),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "hold_time_minutes": self.hold_time_minutes,
            "confidence": round(self.confidence, 4),
            "regime": self.regime,
        }

    @property
    def reward(self) -> float:
        """Calculate reward for this experience."""
        # Reward based on PnL and outcome
        if self.outcome == "WIN":
            return self.pnl_pct * 100 + 1.0  # Bonus for winning
        elif self.outcome == "LOSS":
            return self.pnl_pct * 100 - 0.5  # Penalty for losing
        else:
            return 0.0  # Breakeven


class ExperienceBuffer:
    """
    Circular buffer for storing recent trade experiences.

    Features:
    - Fixed-size buffer with automatic eviction of oldest experiences
    - Prioritized sampling based on reward magnitude
    - Support for batch sampling for training
    - Persistence to disk
    """

    def __init__(
        self,
        max_size: int = 2000,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize the experience buffer.

        Args:
            max_size: Maximum number of experiences to store
            persist_path: Path to persist buffer state
        """
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None
        self._buffer: Deque[TradeExperience] = deque(maxlen=max_size)
        self._lock = threading.RLock()

        if self.persist_path:
            self._load()

    def add(self, experience: TradeExperience) -> None:
        """Add a new experience to the buffer."""
        with self._lock:
            self._buffer.append(experience)

            # Auto-persist periodically
            if self.persist_path and len(self._buffer) % 100 == 0:
                self._save()

    def add_trade(
        self,
        symbol: str,
        features: np.ndarray,
        action: str,
        pnl: float,
        pnl_pct: float,
        entry_price: float,
        exit_price: float,
        hold_time_minutes: int,
        confidence: float = 0.5,
        regime: str = "unknown",
    ) -> TradeExperience:
        """
        Convenience method to create and add an experience.

        Args:
            symbol: Trading symbol
            features: Feature vector at entry time
            action: Action taken (LONG, SHORT, FLAT)
            pnl: Absolute profit/loss
            pnl_pct: Percentage profit/loss
            entry_price: Entry price
            exit_price: Exit price
            hold_time_minutes: Duration of trade
            confidence: Model's confidence
            regime: Market regime

        Returns:
            The created TradeExperience
        """
        action_idx = {"SHORT": 0, "FLAT": 1, "LONG": 2}.get(action.upper(), 1)

        if pnl_pct > 0.001:
            outcome = "WIN"
        elif pnl_pct < -0.001:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        experience = TradeExperience(
            timestamp=datetime.now(),
            symbol=symbol,
            features=features,
            action=action.upper(),
            action_idx=action_idx,
            outcome=outcome,
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_price=entry_price,
            exit_price=exit_price,
            hold_time_minutes=hold_time_minutes,
            confidence=confidence,
            regime=regime,
        )

        self.add(experience)
        return experience

    def sample_batch(
        self,
        batch_size: int = 64,
        prioritized: bool = True,
    ) -> List[TradeExperience]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample
            prioritized: Whether to use prioritized sampling

        Returns:
            List of sampled experiences
        """
        with self._lock:
            if len(self._buffer) == 0:
                return []

            actual_size = min(batch_size, len(self._buffer))

            if prioritized:
                # Prioritize by absolute reward magnitude
                weights = np.array([abs(exp.reward) + 0.1 for exp in self._buffer])
                weights = weights / weights.sum()

                indices = np.random.choice(
                    len(self._buffer),
                    size=actual_size,
                    replace=False,
                    p=weights,
                )
                return [self._buffer[i] for i in indices]
            else:
                # Uniform sampling
                indices = np.random.choice(
                    len(self._buffer),
                    size=actual_size,
                    replace=False,
                )
                return [self._buffer[i] for i in indices]

    def get_recent(self, n: int = 100) -> List[TradeExperience]:
        """Get the n most recent experiences."""
        with self._lock:
            return list(self._buffer)[-n:]

    def get_features_labels(
        self,
        n: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature matrix and labels for training.

        Args:
            n: Number of experiences to use (None = all)

        Returns:
            Tuple of (features array, labels array)
        """
        with self._lock:
            experiences = list(self._buffer)
            if n is not None:
                experiences = experiences[-n:]

            if not experiences:
                return np.array([]), np.array([])

            features = np.array([exp.features for exp in experiences])
            labels = np.array([exp.action_idx for exp in experiences])

            return features, labels

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            if not self._buffer:
                return {
                    "size": 0,
                    "max_size": self.max_size,
                    "fill_ratio": 0.0,
                }

            experiences = list(self._buffer)
            wins = sum(1 for e in experiences if e.outcome == "WIN")
            losses = sum(1 for e in experiences if e.outcome == "LOSS")
            total_pnl = sum(e.pnl for e in experiences)

            return {
                "size": len(self._buffer),
                "max_size": self.max_size,
                "fill_ratio": len(self._buffer) / self.max_size,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(experiences) if experiences else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(experiences) if experiences else 0,
                "avg_confidence": np.mean([e.confidence for e in experiences]),
                "oldest": experiences[0].timestamp.isoformat() if experiences else None,
                "newest": experiences[-1].timestamp.isoformat() if experiences else None,
            }

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def _save(self) -> None:
        """Save buffer to disk."""
        if not self.persist_path:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

        # Save only metadata (features can be large)
        data = {
            "timestamp": datetime.now().isoformat(),
            "size": len(self._buffer),
            "experiences": [
                {
                    **exp.to_dict(),
                    "features_shape": list(exp.features.shape)
                    if exp.features is not None
                    else None,
                }
                for exp in self._buffer
            ],
        }

        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load buffer from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            # Note: Features are not persisted for space efficiency
            # The buffer needs to be repopulated during live trading
            logger.info(f"Loaded buffer metadata: {data.get('size', 0)} experiences")

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load buffer: {e}")


class OnlineLearningManager:
    """
    Manages incremental model updates from live trading.

    Features:
    - Experience buffer for storing trade outcomes
    - Periodic fine-tuning based on accumulated experiences
    - Rollback capability if performance degrades
    - Support for multiple model types

    Usage:
        manager = OnlineLearningManager(
            model=ml_predictor,
            buffer_size=2000,
            update_frequency=50,  # Update every 50 trades
        )

        # After each trade completes
        manager.record_trade_outcome(trade_result)

        # The manager will automatically fine-tune when conditions are met
    """

    def __init__(
        self,
        model: Any,
        buffer_size: int = 2000,
        update_frequency: int = 50,
        min_samples_for_update: int = 100,
        validation_split: float = 0.2,
        max_epochs_per_update: int = 5,
        improvement_threshold: float = 0.01,
        data_dir: str = "data/online_learning",
    ):
        """
        Initialize the online learning manager.

        Args:
            model: The ML model to update (must have fit/predict methods)
            buffer_size: Size of experience buffer
            update_frequency: Update model every N trades
            min_samples_for_update: Minimum samples required for update
            validation_split: Fraction of data for validation
            max_epochs_per_update: Max training epochs per update
            improvement_threshold: Minimum improvement to accept update
            data_dir: Directory for persistence
        """
        self.model = model
        self.update_frequency = update_frequency
        self.min_samples_for_update = min_samples_for_update
        self.validation_split = validation_split
        self.max_epochs = max_epochs_per_update
        self.improvement_threshold = improvement_threshold
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Experience buffer
        self.buffer = ExperienceBuffer(
            max_size=buffer_size,
            persist_path=str(self.data_dir / "experience_buffer.json"),
        )

        # State tracking
        self.trade_count = 0
        self.update_count = 0
        self._last_update = datetime.now()
        self._model_backup: Any = None  # Backup for rollback
        self._performance_history: List[Dict[str, float]] = []

        # Lock for thread safety
        self._lock = threading.RLock()

        self._load_state()

    def record_trade_outcome(
        self,
        symbol: str,
        features: np.ndarray,
        action: str,
        pnl: float,
        pnl_pct: float,
        entry_price: float,
        exit_price: float,
        hold_time_minutes: int,
        confidence: float = 0.5,
        regime: str = "unknown",
    ) -> bool:
        """
        Record a completed trade outcome.

        Args:
            symbol: Trading symbol
            features: Feature vector at entry
            action: Action taken
            pnl: Profit/loss
            pnl_pct: Percentage P/L
            entry_price: Entry price
            exit_price: Exit price
            hold_time_minutes: Trade duration
            confidence: Model's confidence
            regime: Market regime

        Returns:
            True if an update was triggered
        """
        with self._lock:
            # Add to buffer
            self.buffer.add_trade(
                symbol=symbol,
                features=features,
                action=action,
                pnl=pnl,
                pnl_pct=pnl_pct,
                entry_price=entry_price,
                exit_price=exit_price,
                hold_time_minutes=hold_time_minutes,
                confidence=confidence,
                regime=regime,
            )

            self.trade_count += 1

            # Check if update is triggered
            if self.trade_count % self.update_frequency == 0:
                if len(self.buffer) >= self.min_samples_for_update:
                    return self._perform_update()

            return False

    def _perform_update(self) -> bool:
        """Perform incremental model update."""
        logger.info(f"Starting online update #{self.update_count + 1}")

        # Get training data
        X, y = self.buffer.get_features_labels()
        if len(X) < self.min_samples_for_update:
            logger.warning(f"Not enough samples: {len(X)} < {self.min_samples_for_update}")
            return False

        # Split for validation
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Get baseline performance
        baseline_accuracy = self._evaluate_model(X_val, y_val)

        # Backup current model
        self._backup_model()

        try:
            # Fine-tune model
            self._fine_tune_model(X_train, y_train, X_val, y_val)

            # Evaluate updated model
            new_accuracy = self._evaluate_model(X_val, y_val)

            improvement = new_accuracy - baseline_accuracy

            logger.info(
                f"Online update: baseline={baseline_accuracy:.4f}, "
                f"new={new_accuracy:.4f}, improvement={improvement:.4f}"
            )

            # Accept or rollback
            if improvement >= self.improvement_threshold:
                self.update_count += 1
                self._last_update = datetime.now()
                self._performance_history.append(
                    {
                        "timestamp": self._last_update.isoformat(),
                        "baseline_accuracy": baseline_accuracy,
                        "new_accuracy": new_accuracy,
                        "improvement": improvement,
                        "accepted": True,
                    }
                )
                self._save_state()
                logger.info("Update accepted")
                return True
            else:
                self._rollback_model()
                self._performance_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "baseline_accuracy": baseline_accuracy,
                        "new_accuracy": new_accuracy,
                        "improvement": improvement,
                        "accepted": False,
                    }
                )
                logger.info("Update rejected, model rolled back")
                return False

        except Exception as e:
            logger.error(f"Online update failed: {e}")
            self._rollback_model()
            return False

    def _fine_tune_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fine-tune the model on new data."""
        if self.model is None:
            return

        # Try different fine-tuning approaches based on model type
        if hasattr(self.model, "partial_fit"):
            # Incremental learning (SGD, MiniBatch)
            for _ in range(self.max_epochs):
                self.model.partial_fit(X_train, y_train)
        elif hasattr(self.model, "fit"):
            # Full fit (for models that don't support partial_fit)
            # Use early stopping if available
            fit_params = {}
            if hasattr(self.model, "n_estimators"):
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["verbose"] = False
            self.model.fit(X_train, y_train, **fit_params)
        else:
            logger.warning("Model doesn't support fit or partial_fit")

    def _evaluate_model(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy on validation set."""
        if self.model is None or len(X) == 0:
            return 0.0

        try:
            predictions = self.model.predict(X)
            accuracy = np.mean(predictions == y)
            return float(accuracy)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0

    def _backup_model(self) -> None:
        """Create backup of current model state."""
        if self.model is not None:
            try:
                self._model_backup = copy.deepcopy(self.model)
            except Exception as e:
                logger.warning(f"Could not backup model: {e}")
                self._model_backup = None

    def _rollback_model(self) -> None:
        """Rollback to backed up model state."""
        if self._model_backup is not None:
            try:
                self.model = self._model_backup
                logger.info("Model rolled back to previous state")
            except Exception as e:
                logger.error(f"Rollback failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get online learning status."""
        buffer_stats = self.buffer.get_statistics()

        return {
            "trade_count": self.trade_count,
            "update_count": self.update_count,
            "last_update": self._last_update.isoformat(),
            "update_frequency": self.update_frequency,
            "buffer": buffer_stats,
            "performance_history": self._performance_history[-10:],
        }

    def force_update(self) -> bool:
        """Force an immediate model update."""
        with self._lock:
            return self._perform_update()

    def reset(self) -> None:
        """Reset the manager state."""
        with self._lock:
            self.trade_count = 0
            self.update_count = 0
            self.buffer.clear()
            self._performance_history = []
            self._save_state()

    def _get_state_file(self) -> Path:
        return self.data_dir / "online_learning_state.json"

    def _save_state(self) -> None:
        """Save manager state."""
        state = {
            "trade_count": self.trade_count,
            "update_count": self.update_count,
            "last_update": self._last_update.isoformat(),
            "performance_history": self._performance_history[-50:],
        }
        with open(self._get_state_file(), "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load manager state."""
        state_file = self._get_state_file()
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                self.trade_count = state.get("trade_count", 0)
                self.update_count = state.get("update_count", 0)
                self._last_update = datetime.fromisoformat(
                    state.get("last_update", datetime.now().isoformat())
                )
                self._performance_history = state.get("performance_history", [])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load state: {e}")


class StreamingFeatureEngineer:
    """
    Maintains rolling state for incremental feature calculation.

    Avoids recalculating entire history for each prediction.
    """

    def __init__(
        self,
        lookback: int = 100,
        ema_periods: Optional[List[int]] = None,
    ):
        """
        Initialize streaming feature engineer.

        Args:
            lookback: Number of bars to keep in buffer
            ema_periods: EMA periods to track
        """
        self.lookback = lookback
        self.ema_periods = ema_periods or [9, 21, 50]

        # Rolling buffers
        self._price_buffer: Deque[float] = deque(maxlen=lookback)
        self._volume_buffer: Deque[float] = deque(maxlen=lookback)
        self._high_buffer: Deque[float] = deque(maxlen=lookback)
        self._low_buffer: Deque[float] = deque(maxlen=lookback)

        # EMA states (for incremental calculation)
        self._ema_states: Dict[int, float] = {}

        # RSI state
        self._gains: Deque[float] = deque(maxlen=14)
        self._losses: Deque[float] = deque(maxlen=14)

        # Statistics
        self._bar_count = 0

    def update(
        self,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> np.ndarray:
        """
        Update state with new bar and return current features.

        Args:
            open_: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume

        Returns:
            Feature vector for current state
        """
        # Update buffers
        prev_close = self._price_buffer[-1] if self._price_buffer else close
        self._price_buffer.append(close)
        self._volume_buffer.append(volume)
        self._high_buffer.append(high)
        self._low_buffer.append(low)

        # Update EMAs
        for period in self.ema_periods:
            self._update_ema(close, period)

        # Update RSI components
        change = close - prev_close
        if change > 0:
            self._gains.append(change)
            self._losses.append(0)
        else:
            self._gains.append(0)
            self._losses.append(abs(change))

        self._bar_count += 1

        return self._calculate_features()

    def _update_ema(self, value: float, period: int) -> None:
        """Update EMA for a specific period."""
        if period not in self._ema_states:
            self._ema_states[period] = value
        else:
            multiplier = 2 / (period + 1)
            self._ema_states[period] = (
                value - self._ema_states[period]
            ) * multiplier + self._ema_states[period]

    def _calculate_features(self) -> np.ndarray:
        """Calculate current feature vector."""
        if len(self._price_buffer) < 2:
            return np.zeros(15)  # Return zeros if not enough data

        prices = np.array(self._price_buffer)
        volumes = np.array(self._volume_buffer)
        highs = np.array(self._high_buffer)
        lows = np.array(self._low_buffer)

        features = []

        # 1. Returns
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0]
        features.append(returns[-1] if len(returns) > 0 else 0)  # Latest return

        # 2. Volatility (rolling std of returns)
        if len(returns) >= 10:
            features.append(np.std(returns[-10:]))
        else:
            features.append(np.std(returns) if len(returns) > 1 else 0)

        # 3. EMA distances
        current_price = prices[-1]
        for period in self.ema_periods:
            ema = self._ema_states.get(period, current_price)
            features.append((current_price - ema) / ema if ema != 0 else 0)

        # 4. RSI
        avg_gain = np.mean(self._gains) if self._gains else 0
        avg_loss = np.mean(self._losses) if self._losses else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)  # Normalized 0-1

        # 5. MACD-like feature
        if 9 in self._ema_states and 21 in self._ema_states:
            macd = self._ema_states[9] - self._ema_states[21]
            features.append(macd / current_price if current_price != 0 else 0)
        else:
            features.append(0)

        # 6. Volume ratio
        avg_volume = np.mean(volumes) if len(volumes) > 0 else 1
        features.append(volumes[-1] / avg_volume if avg_volume > 0 else 1)

        # 7. Price position in range
        if len(highs) >= 20 and len(lows) >= 20:
            range_high = np.max(highs[-20:])
            range_low = np.min(lows[-20:])
            if range_high != range_low:
                features.append((current_price - range_low) / (range_high - range_low))
            else:
                features.append(0.5)
        else:
            features.append(0.5)

        # 8. Trend (linear regression slope)
        if len(prices) >= 10:
            x = np.arange(10)
            slope, _ = np.polyfit(x, prices[-10:], 1)
            features.append(slope / current_price if current_price != 0 else 0)
        else:
            features.append(0)

        # 9. Momentum (rate of change)
        if len(prices) >= 10:
            roc = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] != 0 else 0
            features.append(roc)
        else:
            features.append(0)

        # 10. ATR-like volatility
        if len(highs) >= 14:
            true_ranges = []
            for i in range(-14, -1):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prices[i - 1]) if i > -len(prices) else 0,
                    abs(lows[i] - prices[i - 1]) if i > -len(prices) else 0,
                )
                true_ranges.append(tr)
            atr = np.mean(true_ranges) / current_price if current_price != 0 else 0
            features.append(atr)
        else:
            features.append(0)

        # Pad to consistent size
        while len(features) < 15:
            features.append(0)

        return np.array(features[:15], dtype=np.float32)

    def reset(self) -> None:
        """Reset all state."""
        self._price_buffer.clear()
        self._volume_buffer.clear()
        self._high_buffer.clear()
        self._low_buffer.clear()
        self._ema_states.clear()
        self._gains.clear()
        self._losses.clear()
        self._bar_count = 0

    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence."""
        return {
            "bar_count": self._bar_count,
            "prices": list(self._price_buffer)[-20:],
            "ema_states": dict(self._ema_states),
        }
