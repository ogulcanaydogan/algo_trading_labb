"""
Machine Learning module for predictive trading.

This module provides:
- Feature engineering for ML models
- XGBoost/RandomForest predictors
- Market regime classification
- Model training and persistence
- Auto-retraining scheduler
"""

from .feature_engineer import FeatureEngineer
from .predictor import MLPredictor, PredictionResult
from .regime_classifier import MarketRegimeClassifier, MarketRegime, RegimeAnalysis

# Enhanced predictor with improved accuracy
from .enhanced_predictor import (
    EnhancedMLPredictor,
    EnhancedPredictionResult,
    EnhancedModelConfig,
    FeatureSelector,
    create_enhanced_predictor,
)

# Auto-retraining components
from .auto_retrainer import (
    AutoRetrainingScheduler,
    ModelHealth,
    ModelHealthStatus,
    RetrainingJob,
    create_auto_retrainer,
)

# Retraining pipeline components
from .training.retraining_pipeline import (
    RetrainingPipeline,
    RetrainingConfig,
    RetrainingTrigger,
    RetrainingResult,
    PerformanceMetrics,
    ModelPerformanceMonitor,
    DataDriftDetector,
)

# Walk-forward strategy optimizer
from .walk_forward_optimizer import (
    WalkForwardStrategyOptimizer,
    ParameterSpace,
    ParameterDriftTracker,
    WindowOptimizationResult,
    WalkForwardOptimizationResults,
)

# Online learning
from .online_learning import (
    ExperienceBuffer,
    OnlineLearningManager,
    StreamingFeatureEngineer,
    TradeExperience,
)

# Aggressive profit hunter
from .aggressive_predictor import (
    AggressiveProfitHunter,
    AggressiveSignal,
    AggressiveConfig,
    SignalStrength,
    MistakeLearner,
    TradeOutcome,
    create_aggressive_predictor,
)

# Profit optimizer
from .profit_optimizer import (
    ProfitOptimizer,
    EntryOptimizer,
    ExitOptimizer,
    EntrySignal,
    ExitSignal,
    TradeState,
    OptimizedLevels,
)

__all__ = [
    # Core ML
    "FeatureEngineer",
    "MLPredictor",
    "PredictionResult",
    "MarketRegimeClassifier",
    "MarketRegime",
    "RegimeAnalysis",
    # Enhanced predictor
    "EnhancedMLPredictor",
    "EnhancedPredictionResult",
    "EnhancedModelConfig",
    "FeatureSelector",
    "create_enhanced_predictor",
    # Auto-retraining
    "AutoRetrainingScheduler",
    "ModelHealth",
    "ModelHealthStatus",
    "RetrainingJob",
    "create_auto_retrainer",
    # Retraining pipeline
    "RetrainingPipeline",
    "RetrainingConfig",
    "RetrainingTrigger",
    "RetrainingResult",
    "PerformanceMetrics",
    "ModelPerformanceMonitor",
    "DataDriftDetector",
    # Walk-forward optimizer
    "WalkForwardStrategyOptimizer",
    "ParameterSpace",
    "ParameterDriftTracker",
    "WindowOptimizationResult",
    "WalkForwardOptimizationResults",
    # Online learning
    "ExperienceBuffer",
    "OnlineLearningManager",
    "StreamingFeatureEngineer",
    "TradeExperience",
    # Aggressive profit hunter
    "AggressiveProfitHunter",
    "AggressiveSignal",
    "AggressiveConfig",
    "SignalStrength",
    "MistakeLearner",
    "TradeOutcome",
    "create_aggressive_predictor",
    # Profit optimizer
    "ProfitOptimizer",
    "EntryOptimizer",
    "ExitOptimizer",
    "EntrySignal",
    "ExitSignal",
    "TradeState",
    "OptimizedLevels",
]
