#!/usr/bin/env python3
"""
Feature Selection with SHAP.

Analyzes feature importance and selects best features for improved generalization.

Usage:
    python scripts/ml/feature_selection.py --symbol BTC/USDT
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_prepared_data(symbol: str):
    """Load prepared data from training pipeline."""
    from scripts.ml.hyperparameter_tuning import prepare_data
    return prepare_data(symbol)


def analyze_feature_importance(X, y, feature_names, model=None):
    """Analyze feature importance using multiple methods."""
    results = {}

    # 1. Random Forest importance
    logger.info("Computing RF feature importance...")
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)

    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    results['rf_importance'] = rf_importance

    # 2. Mutual Information
    logger.info("Computing mutual information...")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)

    results['mutual_info'] = mi_importance

    # 3. SHAP values (if available)
    if SHAP_AVAILABLE:
        logger.info("Computing SHAP values...")
        try:
            # Use a subset for SHAP (it's slow)
            sample_size = min(1000, len(X))
            X_sample = X[:sample_size]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # For multi-class, average absolute SHAP values
            if isinstance(shap_values, list):
                shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0)

            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': shap_importance
            }).sort_values('shap_importance', ascending=False)

            results['shap'] = shap_df

        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")

    # 4. Correlation with target
    logger.info("Computing correlations...")
    correlations = []
    for i, feat in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(abs(corr) if not np.isnan(corr) else 0)

    corr_df = pd.DataFrame({
        'feature': feature_names,
        'abs_correlation': correlations
    }).sort_values('abs_correlation', ascending=False)

    results['correlation'] = corr_df

    return results, model


def select_features(importance_results, n_features=20, method='ensemble'):
    """Select top features based on importance analysis."""
    if method == 'rf':
        return importance_results['rf_importance'].head(n_features)['feature'].tolist()

    elif method == 'shap' and 'shap' in importance_results:
        return importance_results['shap'].head(n_features)['feature'].tolist()

    elif method == 'ensemble':
        # Combine rankings from multiple methods
        features = list(importance_results['rf_importance']['feature'])

        # Score each feature by its rank in each method
        scores = {}
        for feat in features:
            score = 0
            n_methods = 0

            # RF rank
            rf_rank = importance_results['rf_importance'][
                importance_results['rf_importance']['feature'] == feat
            ].index[0]
            score += len(features) - rf_rank
            n_methods += 1

            # MI rank
            mi_rank = importance_results['mutual_info'][
                importance_results['mutual_info']['feature'] == feat
            ].index[0]
            score += len(features) - mi_rank
            n_methods += 1

            # SHAP rank (if available)
            if 'shap' in importance_results:
                shap_df = importance_results['shap']
                shap_match = shap_df[shap_df['feature'] == feat]
                if len(shap_match) > 0:
                    shap_rank = shap_match.index[0]
                    score += len(features) - shap_rank
                    n_methods += 1

            scores[feat] = score / n_methods

        # Sort by ensemble score
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:n_features]]

    return features[:n_features]


def evaluate_feature_subset(X, y, feature_indices, feature_names):
    """Evaluate model performance with feature subset."""
    X_subset = X[:, feature_indices]

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X_subset, y, cv=tscv, scoring='accuracy')

    return {
        'n_features': len(feature_indices),
        'features': [feature_names[i] for i in feature_indices],
        'cv_mean': scores.mean(),
        'cv_std': scores.std()
    }


def run_feature_selection(symbol: str, max_features: int = 30):
    """Run complete feature selection pipeline."""
    logger.info(f"Running feature selection for {symbol}")

    # Load data
    X, y, feature_names = load_prepared_data(symbol)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

    # Analyze importance
    importance_results, model = analyze_feature_importance(
        X_scaled, y, feature_names
    )

    # Evaluate different feature counts
    logger.info("\nEvaluating feature subsets...")
    evaluation_results = []

    for n_feat in [10, 15, 20, 25, 30, len(feature_names)]:
        if n_feat > len(feature_names):
            continue

        selected = select_features(importance_results, n_feat, method='ensemble')
        indices = [feature_names.index(f) for f in selected]

        result = evaluate_feature_subset(X_scaled, y, indices, feature_names)
        evaluation_results.append(result)

        logger.info(f"  {n_feat} features: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})")

    # Find optimal feature count
    best_result = max(evaluation_results, key=lambda x: x['cv_mean'])
    logger.info(f"\nBest: {best_result['n_features']} features with {best_result['cv_mean']:.4f} accuracy")

    # Save results
    symbol_clean = symbol.replace('/', '_')

    report = {
        'symbol': symbol,
        'total_features': len(feature_names),
        'optimal_features': best_result['n_features'],
        'optimal_accuracy': best_result['cv_mean'],
        'selected_features': best_result['features'],
        'importance_rankings': {
            'rf': importance_results['rf_importance'].head(20).to_dict('records'),
            'mutual_info': importance_results['mutual_info'].head(20).to_dict('records'),
        },
        'evaluation_results': evaluation_results,
        'created_at': datetime.now().isoformat()
    }

    if 'shap' in importance_results:
        report['importance_rankings']['shap'] = importance_results['shap'].head(20).to_dict('records')

    # Save report
    report_path = REPORT_DIR / f"feature_selection_{symbol_clean}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to {report_path}")

    # Save selected features for future use
    meta_path = MODEL_DIR / f"{symbol_clean}_selected_features.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'symbol': symbol,
            'selected_features': best_result['features'],
            'accuracy': best_result['cv_mean'],
            'created_at': datetime.now().isoformat()
        }, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description='Feature selection with SHAP')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--max-features', type=int, default=30, help='Max features to select')
    parser.add_argument('--all-symbols', action='store_true', help='Analyze all symbols')
    args = parser.parse_args()

    if args.all_symbols:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    else:
        symbols = [args.symbol]

    for symbol in symbols:
        try:
            run_feature_selection(symbol, args.max_features)
        except Exception as e:
            logger.error(f"Failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
