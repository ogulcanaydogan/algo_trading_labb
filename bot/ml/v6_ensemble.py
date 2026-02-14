"""
V6 Ensemble Predictor - Combines Multiple V6 Binary Models.

Uses walk-forward accuracy-weighted voting to combine predictions from:
- TSLA, BTC_USDT, XRP_USDT models

Ensemble Strategies:
1. Weighted Average: Weight predictions by walk-forward accuracy
2. Majority Vote: Simple majority across models
3. Confidence-Weighted: Weight by both accuracy and model confidence
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class V6ModelInfo:
    """Information about a loaded V6 model."""
    symbol: str
    model: Any
    scaler: Any
    features: List[str]
    wf_accuracy: float  # Walk-forward accuracy (primary weight)
    test_accuracy: float
    hc_accuracy: float  # High-confidence accuracy
    confidence_threshold: float
    asset_class: str
    horizon: int
    

@dataclass
class V6EnsemblePrediction:
    """Combined prediction from V6 ensemble."""
    signal: Literal["LONG", "SHORT", "NEUTRAL"]
    combined_probability: float  # Weighted probability
    ensemble_confidence: float  # How confident is the ensemble
    agreement_score: float  # How much models agree (0-1)
    individual_predictions: Dict[str, Dict]
    weights_used: Dict[str, float]
    strategy: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "signal": self.signal,
            "combined_probability": round(self.combined_probability, 4),
            "ensemble_confidence": round(self.ensemble_confidence, 4),
            "agreement_score": round(self.agreement_score, 4),
            "individual_predictions": self.individual_predictions,
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "strategy": self.strategy,
            "timestamp": self.timestamp.isoformat(),
        }


class V6EnsemblePredictor:
    """
    Ensemble predictor combining multiple V6 binary classification models.
    
    Features:
    - Loads V6 models (TSLA, BTC_USDT, XRP_USDT, etc.)
    - Combines predictions using walk-forward accuracy weighting
    - Supports multiple ensemble strategies
    - Provides confidence metrics and agreement scores
    """
    
    # Default model directory
    DEFAULT_MODEL_DIR = Path("data/models_v6_improved")
    
    # Symbols to load by default (sorted by walk-forward accuracy)
    # SPX500_USD: 55.67%, XRP_USDT: 55.64%, TSLA: 55.11%, BTC_USDT: 51.37%
    DEFAULT_SYMBOLS = ["SPX500_USD", "XRP_USDT", "TSLA", "BTC_USDT"]
    
    def __init__(
        self,
        model_dir: Optional[Path] = None,
        symbols: Optional[List[str]] = None,
        strategy: Literal["weighted_avg", "majority_vote", "confidence_weighted"] = "weighted_avg",
        min_agreement: float = 0.5,
        probability_threshold: float = 0.55,
    ):
        """
        Initialize V6 Ensemble Predictor.
        
        Args:
            model_dir: Directory containing V6 models
            symbols: List of symbols to load (default: TSLA, BTC_USDT, XRP_USDT)
            strategy: Ensemble strategy to use
            min_agreement: Minimum agreement between models for valid signal
            probability_threshold: Threshold for converting probability to signal
        """
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        self.symbols = symbols or self.DEFAULT_SYMBOLS
        self.strategy = strategy
        self.min_agreement = min_agreement
        self.probability_threshold = probability_threshold
        
        # Loaded models
        self.models: Dict[str, V6ModelInfo] = {}
        self.is_loaded = False
        
        # Computed weights (normalized walk-forward accuracies)
        self.weights: Dict[str, float] = {}
        
    def load_models(self) -> bool:
        """
        Load all V6 models from disk.
        
        Returns:
            True if at least one model loaded successfully
        """
        loaded_count = 0
        
        for symbol in self.symbols:
            try:
                model_info = self._load_single_model(symbol)
                if model_info:
                    self.models[symbol] = model_info
                    loaded_count += 1
                    logger.info(
                        f"Loaded V6 model: {symbol} "
                        f"(wf_acc={model_info.wf_accuracy:.3f}, "
                        f"features={len(model_info.features)})"
                    )
            except Exception as e:
                logger.warning(f"Failed to load V6 model for {symbol}: {e}")
        
        if loaded_count > 0:
            self._compute_weights()
            self.is_loaded = True
            logger.info(f"V6 Ensemble loaded {loaded_count} models, weights: {self.weights}")
        else:
            logger.warning("No V6 models loaded!")
            
        return loaded_count > 0
    
    def _load_single_model(self, symbol: str) -> Optional[V6ModelInfo]:
        """Load a single V6 model and its metadata."""
        model_dir = Path(self.model_dir)
        
        # File paths
        model_path = model_dir / f"{symbol}_binary_ensemble_v6.pkl"
        scaler_path = model_dir / f"{symbol}_binary_scaler_v6.pkl"
        meta_path = model_dir / f"{symbol}_binary_meta_v6.json"
        features_path = model_dir / f"{symbol}_selected_features_v6.json"
        
        # Check all required files exist
        if not all(p.exists() for p in [model_path, scaler_path, meta_path, features_path]):
            logger.debug(f"Missing files for {symbol}")
            return None
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Load features
        with open(features_path, "r") as f:
            features = json.load(f)
        
        return V6ModelInfo(
            symbol=symbol,
            model=model,
            scaler=scaler,
            features=features,
            wf_accuracy=meta.get("walk_forward", {}).get("avg_acc", 0.5),
            test_accuracy=meta.get("metrics", {}).get("test_accuracy", 0.5),
            hc_accuracy=meta.get("metrics", {}).get("hc_accuracy", 0.5),
            confidence_threshold=meta.get("config", {}).get("confidence_threshold", 0.55),
            asset_class=meta.get("asset_class", "unknown"),
            horizon=meta.get("config", {}).get("horizon", 1),
        )
    
    def _compute_weights(self) -> None:
        """Compute normalized weights based on walk-forward accuracy."""
        if not self.models:
            return
        
        # Use walk-forward accuracy as weight
        total_wf = sum(m.wf_accuracy for m in self.models.values())
        
        if total_wf > 0:
            self.weights = {
                symbol: info.wf_accuracy / total_wf
                for symbol, info in self.models.items()
            }
        else:
            # Equal weights fallback
            n = len(self.models)
            self.weights = {symbol: 1.0 / n for symbol in self.models.keys()}
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Optional[np.ndarray]:
        """
        Prepare features for a specific model.
        
        Args:
            df: DataFrame with OHLCV and computed features
            symbol: Symbol to prepare features for
            
        Returns:
            Scaled feature array or None if features missing
        """
        if symbol not in self.models:
            return None
        
        model_info = self.models[symbol]
        
        # Check for missing features
        missing = [f for f in model_info.features if f not in df.columns]
        if missing:
            # Only log once per symbol to avoid spam
            if not hasattr(self, '_missing_warned'):
                self._missing_warned = set()
            if symbol not in self._missing_warned:
                logger.warning(f"Missing features for {symbol}: {missing[:5]}... (suppressing further warnings)")
                self._missing_warned.add(symbol)
            return None
        
        # Extract features
        X = df[model_info.features].iloc[-1:].values
        
        # Handle inf/nan
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale
        try:
            X_scaled = model_info.scaler.transform(X)
            return X_scaled
        except Exception as e:
            logger.warning(f"Scaling failed for {symbol}: {e}")
            return None
    
    def predict_single(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Optional[Dict]:
        """
        Get prediction from a single model.
        
        Args:
            df: DataFrame with features
            symbol: Model symbol
            
        Returns:
            Dict with prediction, probability, confidence
        """
        X = self.prepare_features(df, symbol)
        if X is None:
            return None
        
        model_info = self.models[symbol]
        
        try:
            # Get probability (V6 models are binary: 0=DOWN, 1=UP)
            if hasattr(model_info.model, "predict_proba"):
                proba = model_info.model.predict_proba(X)[0]
                prob_up = proba[1] if len(proba) > 1 else proba[0]
                prob_down = proba[0] if len(proba) > 1 else 1 - prob_up
            else:
                # Fallback if no predict_proba
                pred = model_info.model.predict(X)[0]
                prob_up = 0.7 if pred == 1 else 0.3
                prob_down = 1 - prob_up
            
            # Determine signal
            if prob_up > model_info.confidence_threshold:
                signal = "LONG"
            elif prob_down > model_info.confidence_threshold:
                signal = "SHORT"
            else:
                signal = "NEUTRAL"
            
            return {
                "signal": signal,
                "prob_up": float(prob_up),
                "prob_down": float(prob_down),
                "confidence": float(max(prob_up, prob_down)),
                "wf_accuracy": model_info.wf_accuracy,
                "threshold": model_info.confidence_threshold,
            }
            
        except Exception as e:
            logger.warning(f"Prediction failed for {symbol}: {e}")
            return None
    
    def predict(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None,
    ) -> V6EnsemblePrediction:
        """
        Generate ensemble prediction.
        
        Args:
            df: DataFrame with OHLCV and computed features
            strategy: Override default strategy
            
        Returns:
            V6EnsemblePrediction with combined signal
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        strategy = strategy or self.strategy
        
        # Get predictions from all models
        predictions = {}
        for symbol in self.models.keys():
            pred = self.predict_single(df, symbol)
            if pred:
                predictions[symbol] = pred
        
        if not predictions:
            # Return neutral if no predictions
            return V6EnsemblePrediction(
                signal="NEUTRAL",
                combined_probability=0.5,
                ensemble_confidence=0.0,
                agreement_score=0.0,
                individual_predictions={},
                weights_used={},
                strategy=strategy,
            )
        
        # Combine predictions based on strategy
        if strategy == "weighted_avg":
            result = self._weighted_average(predictions)
        elif strategy == "majority_vote":
            result = self._majority_vote(predictions)
        elif strategy == "confidence_weighted":
            result = self._confidence_weighted(predictions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Build final prediction
        signal, prob, confidence, agreement = result
        
        # Apply minimum agreement threshold
        if agreement < self.min_agreement:
            signal = "NEUTRAL"
            confidence *= 0.5
        
        return V6EnsemblePrediction(
            signal=signal,
            combined_probability=prob,
            ensemble_confidence=confidence,
            agreement_score=agreement,
            individual_predictions=predictions,
            weights_used={k: self.weights.get(k, 0) for k in predictions.keys()},
            strategy=strategy,
        )
    
    def _weighted_average(
        self,
        predictions: Dict[str, Dict],
    ) -> Tuple[str, float, float, float]:
        """
        Combine predictions using walk-forward accuracy weighting.
        
        Returns: (signal, combined_prob, confidence, agreement)
        """
        weighted_prob_up = 0.0
        weighted_prob_down = 0.0
        total_weight = 0.0
        
        for symbol, pred in predictions.items():
            weight = self.weights.get(symbol, 0.0)
            weighted_prob_up += weight * pred["prob_up"]
            weighted_prob_down += weight * pred["prob_down"]
            total_weight += weight
        
        if total_weight == 0:
            return "NEUTRAL", 0.5, 0.0, 0.0
        
        # Normalize
        avg_prob_up = weighted_prob_up / total_weight
        avg_prob_down = weighted_prob_down / total_weight
        
        # Determine signal
        if avg_prob_up > self.probability_threshold:
            signal = "LONG"
            confidence = avg_prob_up
        elif avg_prob_down > self.probability_threshold:
            signal = "SHORT"
            confidence = avg_prob_down
        else:
            signal = "NEUTRAL"
            confidence = max(avg_prob_up, avg_prob_down)
        
        # Calculate agreement (how many models agree with ensemble)
        agreement_count = sum(
            1 for p in predictions.values()
            if p["signal"] == signal
        )
        agreement = agreement_count / len(predictions)
        
        return signal, avg_prob_up, confidence, agreement
    
    def _majority_vote(
        self,
        predictions: Dict[str, Dict],
    ) -> Tuple[str, float, float, float]:
        """
        Simple majority voting.
        
        Returns: (signal, combined_prob, confidence, agreement)
        """
        votes = {"LONG": 0, "SHORT": 0, "NEUTRAL": 0}
        probs = {"LONG": [], "SHORT": [], "NEUTRAL": []}
        
        for pred in predictions.values():
            votes[pred["signal"]] += 1
            probs[pred["signal"]].append(pred["confidence"])
        
        # Winner is signal with most votes
        winner = max(votes.keys(), key=lambda k: votes[k])
        agreement = votes[winner] / len(predictions)
        
        # Average confidence of winning votes
        if probs[winner]:
            avg_conf = np.mean(probs[winner])
        else:
            avg_conf = 0.5
        
        # Combined probability (average prob_up for LONG, prob_down for SHORT)
        if winner == "LONG":
            combined_prob = np.mean([p["prob_up"] for p in predictions.values()])
        elif winner == "SHORT":
            combined_prob = np.mean([p["prob_down"] for p in predictions.values()])
        else:
            combined_prob = 0.5
        
        return winner, combined_prob, avg_conf, agreement
    
    def _confidence_weighted(
        self,
        predictions: Dict[str, Dict],
    ) -> Tuple[str, float, float, float]:
        """
        Weight by both walk-forward accuracy AND prediction confidence.
        
        Returns: (signal, combined_prob, confidence, agreement)
        """
        vote_scores = {"LONG": 0.0, "SHORT": 0.0, "NEUTRAL": 0.0}
        
        for symbol, pred in predictions.items():
            wf_weight = self.weights.get(symbol, 0.0)
            pred_conf = pred["confidence"]
            # Combined weight = accuracy * confidence
            combined_weight = wf_weight * pred_conf
            vote_scores[pred["signal"]] += combined_weight
        
        total = sum(vote_scores.values())
        if total == 0:
            return "NEUTRAL", 0.5, 0.0, 0.0
        
        # Winner is highest weighted signal
        winner = max(vote_scores.keys(), key=lambda k: vote_scores[k])
        confidence = vote_scores[winner] / total
        
        # Agreement
        agreement_count = sum(
            1 for p in predictions.values()
            if p["signal"] == winner
        )
        agreement = agreement_count / len(predictions)
        
        # Combined probability
        if winner == "LONG":
            combined_prob = np.mean([p["prob_up"] for p in predictions.values()])
        elif winner == "SHORT":
            combined_prob = np.mean([p["prob_down"] for p in predictions.values()])
        else:
            combined_prob = 0.5
        
        return winner, combined_prob, confidence, agreement
    
    def get_model_stats(self) -> Dict[str, Dict]:
        """Get statistics for all loaded models."""
        stats = {}
        for symbol, info in self.models.items():
            stats[symbol] = {
                "weight": self.weights.get(symbol, 0.0),
                "wf_accuracy": info.wf_accuracy,
                "test_accuracy": info.test_accuracy,
                "hc_accuracy": info.hc_accuracy,
                "asset_class": info.asset_class,
                "horizon": info.horizon,
                "features": len(info.features),
            }
        return stats


# Factory function
def create_v6_ensemble(
    model_dir: Optional[Path] = None,
    symbols: Optional[List[str]] = None,
    strategy: str = "weighted_avg",
) -> Optional[V6EnsemblePredictor]:
    """
    Create and load a V6 ensemble predictor.
    
    Args:
        model_dir: Directory with V6 models
        symbols: Symbols to include
        strategy: Ensemble strategy
        
    Returns:
        Loaded V6EnsemblePredictor or None if load failed
    """
    ensemble = V6EnsemblePredictor(
        model_dir=model_dir,
        symbols=symbols,
        strategy=strategy,
    )
    
    if ensemble.load_models():
        return ensemble
    else:
        return None


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    ensemble = create_v6_ensemble()
    if ensemble:
        print("\nLoaded V6 Ensemble Models:")
        print("-" * 50)
        for symbol, stats in ensemble.get_model_stats().items():
            print(f"  {symbol}:")
            print(f"    Weight: {stats['weight']:.3f}")
            print(f"    WF Accuracy: {stats['wf_accuracy']:.3f}")
            print(f"    Features: {stats['features']}")
        print("-" * 50)
    else:
        print("Failed to load ensemble")
