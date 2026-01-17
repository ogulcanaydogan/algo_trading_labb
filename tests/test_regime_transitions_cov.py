from datetime import datetime, timedelta

from bot.regime_transitions import RegimeTransitionAnalyzer


def test_transition_matrix_and_prediction():
    analyzer = RegimeTransitionAnalyzer()
    now = datetime.now()
    regimes = ["BULL", "BULL", "BEAR", "VOLATILE", "BULL"]
    timestamps = [now - timedelta(hours=4 - i) for i in range(len(regimes))]
    prices = [100, 102, 97, 95, 101]

    analyzer.load_regime_history(regimes, timestamps, prices)
    matrix = analyzer.calculate_transition_matrix()

    # Only count changes in regime, so consecutive duplicates don't inflate transitions
    assert matrix.total_transitions == 3
    # Probabilities for any row should sum to 1.0
    for probs in matrix.matrix.values():
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    # Predict next regime from current state
    current = analyzer.get_current_regime()
    prediction = analyzer.predict_next_regime(current)
    assert prediction
    assert current is not None


def test_regime_distribution_and_timeline():
    analyzer = RegimeTransitionAnalyzer()
    now = datetime.now()
    regimes = ["BULL", "BEAR", "BULL"]
    timestamps = [now - timedelta(hours=2 - i) for i in range(len(regimes))]
    prices = [100, 95, 105]

    analyzer.load_regime_history(regimes, timestamps, prices)
    distribution = analyzer.get_regime_distribution()
    timeline = analyzer.get_regime_timeline(limit=10)

    assert set(distribution.keys()) == set(regimes)
    assert len(timeline) == len(regimes)
