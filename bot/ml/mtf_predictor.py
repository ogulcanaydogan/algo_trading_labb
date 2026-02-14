"""
Multi-Timeframe ML Predictor

Combines predictions from multiple horizon models for stronger conviction:
- Short: 3h horizon model
- Medium: 8h horizon model  
- Long: 24h horizon model

Conviction levels:
- HIGH: All 3 models agree on direction
- MEDIUM: 2/3 models agree
- NO TRADE: Models disagree
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Conviction(Enum):
    """Trading conviction level based on model agreement."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NO_TRADE = "no_trade"


class Direction(Enum):
    """Trading direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe model."""
    horizon: int  # in hours
    direction: Direction
    probability: float  # probability of the predicted direction
    confidence: float  # max(prob_long, prob_short)
    prob_long: float
    prob_short: float
    model_loaded: bool = True


@dataclass
class MTFPrediction:
    """Combined multi-timeframe prediction."""
    timestamp: datetime
    symbol: str
    direction: Direction
    conviction: Conviction
    agreement_score: float  # 0-1, how much models agree
    signals: Dict[str, TimeframeSignal]  # keyed by horizon (e.g., "3h")
    
    # Combined metrics
    avg_confidence: float
    weighted_prob_long: float
    weighted_prob_short: float
    
    # Decision
    should_trade: bool
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction.value,
            "conviction": self.conviction.value,
            "agreement_score": round(self.agreement_score, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "weighted_prob_long": round(self.weighted_prob_long, 4),
            "weighted_prob_short": round(self.weighted_prob_short, 4),
            "should_trade": self.should_trade,
            "reason": self.reason,
            "signals": {
                k: {
                    "horizon": v.horizon,
                    "direction": v.direction.value,
                    "probability": round(v.probability, 4),
                    "confidence": round(v.confidence, 4),
                }
                for k, v in self.signals.items()
            }
        }


class MTFPredictor:
    """
    Multi-Timeframe ML Predictor.
    
    Loads and combines predictions from multiple horizon models:
    - Short (3h): Captures fast momentum moves
    - Medium (8h): Captures intraday trends
    - Long (24h): Captures multi-day direction
    
    Trading rules:
    - HIGH conviction: All 3 agree → Trade with full size
    - MEDIUM conviction: 2/3 agree → Trade with reduced size
    - NO TRADE: Disagreement → Skip trade
    """
    
    # Weights for combining signals (higher timeframes more weight)
    DEFAULT_WEIGHTS = {
        3: 0.25,   # Short-term
        8: 0.35,   # Medium-term
        24: 0.40,  # Long-term
    }
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.55  # Minimum per-model confidence
    HIGH_CONFIDENCE = 0.60  # High confidence threshold
    
    def __init__(
        self,
        symbol: str,
        horizons: List[int] = None,
        model_dir: str = "data/models_v6_improved",
        weights: Dict[int, float] = None,
        min_conviction: Conviction = Conviction.MEDIUM,
    ):
        """
        Initialize multi-timeframe predictor.
        
        Args:
            symbol: Trading symbol (e.g., "TSLA")
            horizons: List of prediction horizons in hours (default: [3, 8, 24])
            model_dir: Directory containing trained models
            weights: Custom weights for each horizon
            min_conviction: Minimum conviction to trade (default: MEDIUM)
        """
        self.symbol = symbol.replace("/", "_")
        self.horizons = horizons or [3, 8, 24]
        self.model_dir = Path(model_dir)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.min_conviction = min_conviction
        
        # Normalize weights
        total_weight = sum(self.weights.get(h, 0.33) for h in self.horizons)
        self.weights = {h: self.weights.get(h, 0.33) / total_weight for h in self.horizons}
        
        # Load models
        self.models: Dict[int, object] = {}
        self.scalers: Dict[int, object] = {}
        self.features: Dict[int, List[str]] = {}
        self.loaded_horizons: List[int] = []
        
        self._load_models()
        
    def _load_models(self):
        """Load all timeframe models."""
        for horizon in self.horizons:
            model_path = self.model_dir / f"{self.symbol}_{horizon}h_binary_ensemble_v6.pkl"
            scaler_path = self.model_dir / f"{self.symbol}_{horizon}h_binary_scaler_v6.pkl"
            features_path = self.model_dir / f"{self.symbol}_{horizon}h_selected_features_v6.json"
            
            # Also check without horizon suffix (for existing 8h models)
            if not model_path.exists() and horizon == 8:
                model_path = self.model_dir / f"{self.symbol}_binary_ensemble_v6.pkl"
                scaler_path = self.model_dir / f"{self.symbol}_binary_scaler_v6.pkl"
                features_path = self.model_dir / f"{self.symbol}_selected_features_v6.json"
            
            if model_path.exists() and scaler_path.exists():
                try:
                    self.models[horizon] = joblib.load(model_path)
                    self.scalers[horizon] = joblib.load(scaler_path)
                    
                    if features_path.exists():
                        with open(features_path) as f:
                            self.features[horizon] = json.load(f)
                    else:
                        self.features[horizon] = []
                    
                    self.loaded_horizons.append(horizon)
                    logger.info(f"Loaded {self.symbol} {horizon}h model")
                except Exception as e:
                    logger.warning(f"Failed to load {horizon}h model: {e}")
            else:
                logger.warning(f"Model not found for {self.symbol} {horizon}h: {model_path}")
        
        if not self.loaded_horizons:
            raise ValueError(f"No models loaded for {self.symbol}. Train models first.")
        
        logger.info(f"MTFPredictor loaded {len(self.loaded_horizons)} models for {self.symbol}: {self.loaded_horizons}h")
    
    def predict(self, features_df: pd.DataFrame) -> MTFPrediction:
        """
        Make multi-timeframe prediction.
        
        Args:
            features_df: DataFrame with computed features (from feature engineering)
            
        Returns:
            MTFPrediction with combined signal
        """
        signals = {}
        
        for horizon in self.loaded_horizons:
            signal = self._predict_single(features_df, horizon)
            signals[f"{horizon}h"] = signal
        
        return self._combine_signals(signals)
    
    def _predict_single(self, features_df: pd.DataFrame, horizon: int) -> TimeframeSignal:
        """Get prediction from a single timeframe model."""
        if horizon not in self.loaded_horizons:
            return TimeframeSignal(
                horizon=horizon,
                direction=Direction.FLAT,
                probability=0.5,
                confidence=0.5,
                prob_long=0.5,
                prob_short=0.5,
                model_loaded=False,
            )
        
        model = self.models[horizon]
        scaler = self.scalers[horizon]
        feature_names = self.features[horizon]
        
        try:
            # Get latest row
            if len(features_df) == 0:
                raise ValueError("Empty features DataFrame")
            
            latest = features_df.iloc[-1:]
            
            # Select features
            if feature_names:
                missing = [f for f in feature_names if f not in latest.columns]
                if missing:
                    logger.warning(f"{horizon}h model missing features: {missing[:5]}...")
                    # Fill missing with 0
                    for f in missing:
                        latest[f] = 0
                X = latest[feature_names].values
            else:
                # Use all numeric features
                X = latest.select_dtypes(include=[np.number]).values
            
            # Handle NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale
            X_scaled = scaler.transform(X)
            
            # Predict
            proba = model.predict_proba(X_scaled)[0]
            prob_short = proba[0] if len(proba) > 0 else 0.5
            prob_long = proba[1] if len(proba) > 1 else 0.5
            
            # Determine direction
            if prob_long > prob_short:
                direction = Direction.LONG
                probability = prob_long
            elif prob_short > prob_long:
                direction = Direction.SHORT
                probability = prob_short
            else:
                direction = Direction.FLAT
                probability = 0.5
            
            confidence = max(prob_long, prob_short)
            
            return TimeframeSignal(
                horizon=horizon,
                direction=direction,
                probability=probability,
                confidence=confidence,
                prob_long=prob_long,
                prob_short=prob_short,
                model_loaded=True,
            )
            
        except Exception as e:
            logger.error(f"Error predicting {horizon}h: {e}")
            return TimeframeSignal(
                horizon=horizon,
                direction=Direction.FLAT,
                probability=0.5,
                confidence=0.5,
                prob_long=0.5,
                prob_short=0.5,
                model_loaded=True,
            )
    
    def _combine_signals(self, signals: Dict[str, TimeframeSignal]) -> MTFPrediction:
        """Combine signals from multiple timeframes."""
        timestamp = datetime.now()
        
        # Extract directions and calculate agreement
        directions = [s.direction for s in signals.values() if s.model_loaded]
        n_long = sum(1 for d in directions if d == Direction.LONG)
        n_short = sum(1 for d in directions if d == Direction.SHORT)
        n_total = len(directions)
        
        # Calculate weighted probabilities
        weighted_prob_long = 0.0
        weighted_prob_short = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        for key, signal in signals.items():
            if not signal.model_loaded:
                continue
            horizon = signal.horizon
            weight = self.weights.get(horizon, 0.33)
            weighted_prob_long += signal.prob_long * weight
            weighted_prob_short += signal.prob_short * weight
            confidence_sum += signal.confidence
            total_weight += weight
        
        if total_weight > 0:
            weighted_prob_long /= total_weight
            weighted_prob_short /= total_weight
            avg_confidence = confidence_sum / len([s for s in signals.values() if s.model_loaded])
        else:
            weighted_prob_long = 0.5
            weighted_prob_short = 0.5
            avg_confidence = 0.5
        
        # Determine conviction level
        if n_total == 0:
            conviction = Conviction.NO_TRADE
            direction = Direction.FLAT
            agreement_score = 0.0
        elif n_long == n_total:
            # All models agree LONG
            conviction = Conviction.HIGH
            direction = Direction.LONG
            agreement_score = 1.0
        elif n_short == n_total:
            # All models agree SHORT
            conviction = Conviction.HIGH
            direction = Direction.SHORT
            agreement_score = 1.0
        elif n_long > n_short and n_long >= n_total - 1:
            # 2/3 or more agree LONG
            conviction = Conviction.MEDIUM
            direction = Direction.LONG
            agreement_score = n_long / n_total
        elif n_short > n_long and n_short >= n_total - 1:
            # 2/3 or more agree SHORT
            conviction = Conviction.MEDIUM
            direction = Direction.SHORT
            agreement_score = n_short / n_total
        else:
            # No agreement
            conviction = Conviction.NO_TRADE
            # Use weighted direction
            if weighted_prob_long > weighted_prob_short:
                direction = Direction.LONG
            elif weighted_prob_short > weighted_prob_long:
                direction = Direction.SHORT
            else:
                direction = Direction.FLAT
            agreement_score = max(n_long, n_short) / n_total if n_total > 0 else 0.0
        
        # Determine if we should trade
        should_trade = (
            conviction.value in [Conviction.HIGH.value, Conviction.MEDIUM.value] and
            avg_confidence >= self.MIN_CONFIDENCE and
            direction != Direction.FLAT
        )
        
        # Filter by minimum conviction
        if conviction == Conviction.MEDIUM and self.min_conviction == Conviction.HIGH:
            should_trade = False
        
        # Build reason
        signal_summary = ", ".join([
            f"{k}:{s.direction.value}({s.confidence:.0%})" 
            for k, s in signals.items() if s.model_loaded
        ])
        
        if should_trade:
            reason = f"{conviction.value.upper()} conviction {direction.value}: {signal_summary}"
        else:
            reason = f"NO TRADE ({conviction.value}): {signal_summary}"
        
        return MTFPrediction(
            timestamp=timestamp,
            symbol=self.symbol,
            direction=direction,
            conviction=conviction,
            agreement_score=agreement_score,
            signals=signals,
            avg_confidence=avg_confidence,
            weighted_prob_long=weighted_prob_long,
            weighted_prob_short=weighted_prob_short,
            should_trade=should_trade,
            reason=reason,
        )
    
    def get_position_size_multiplier(self, conviction: Conviction) -> float:
        """Get position size multiplier based on conviction."""
        multipliers = {
            Conviction.HIGH: 1.0,
            Conviction.MEDIUM: 0.6,
            Conviction.LOW: 0.3,
            Conviction.NO_TRADE: 0.0,
        }
        return multipliers.get(conviction, 0.0)


def create_mtf_predictor(
    symbol: str,
    model_dir: str = "data/models_v6_improved",
    horizons: List[int] = None,
) -> Optional[MTFPredictor]:
    """
    Factory function to create MTFPredictor.
    
    Returns None if models are not available.
    """
    try:
        return MTFPredictor(
            symbol=symbol,
            horizons=horizons or [3, 8, 24],
            model_dir=model_dir,
        )
    except Exception as e:
        logger.error(f"Failed to create MTFPredictor for {symbol}: {e}")
        return None


# Global predictors cache
_mtf_predictors: Dict[str, MTFPredictor] = {}


def get_mtf_predictor(symbol: str, model_dir: str = "data/models_v6_improved") -> Optional[MTFPredictor]:
    """Get or create cached MTFPredictor."""
    global _mtf_predictors
    
    key = f"{symbol}_{model_dir}"
    if key not in _mtf_predictors:
        predictor = create_mtf_predictor(symbol, model_dir)
        if predictor:
            _mtf_predictors[key] = predictor
    
    return _mtf_predictors.get(key)
