import numpy as np
import pandas as pd

from bot.cross_market_analysis import (
    CrossMarketAnalyzer,
    CorrelationRegime,
)


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_pair_correlation_positive_regime():
    analyzer = CrossMarketAnalyzer(rolling_window=5, min_correlation_period=5)
    analyzer.add_returns("BTC", _series([0.01, 0.02, 0.015, 0.01, 0.02, 0.03]), "crypto")
    analyzer.add_returns("AAPL", _series([0.011, 0.021, 0.016, 0.011, 0.019, 0.031]), "stock")

    pair = analyzer.calculate_pair_correlation("BTC", "AAPL")
    assert pair is not None
    assert pair.regime in {CorrelationRegime.HIGH_POSITIVE, CorrelationRegime.MODERATE_POSITIVE}
    assert bool(pair.is_significant) is True
    assert pair.market_type1 == "crypto"
    assert pair.market_type2 == "stock"


def test_detect_correlation_shift():
    analyzer = CrossMarketAnalyzer(rolling_window=5, min_correlation_period=10)
    # Early windows strongly positive correlation, only last few windows flip to strong negative
    r1 = _series([0.01 + 0.001 * i for i in range(60)])
    r2 = _series(
        [0.01 + 0.001 * i for i in range(55)]
        + [-0.01 - 0.001 * i for i in range(5)]
    )
    analyzer.add_returns("BTC", r1, "crypto")
    analyzer.add_returns("GLD", r2, "commodity")

    shifts = analyzer.detect_correlation_shifts(threshold=0.5)
    assert len(shifts) >= 1
    # Ensure we recorded a shift for the BTC/GLD pair
    assert any({shift.asset1, shift.asset2} == {"BTC", "GLD"} for shift in shifts)


def test_correlation_matrix_handles_missing():
    analyzer = CrossMarketAnalyzer(min_correlation_period=5)
    analyzer.add_returns("BTC", _series([0.01, 0.02, -0.01, 0.005, 0.01]), "crypto")
    # Only one asset available; matrix should be 1x1 with 1.0 on diagonal
    matrix = analyzer.get_correlation_matrix()
    assert matrix.shape == (1, 1)
    assert matrix.iloc[0, 0] == 1.0
