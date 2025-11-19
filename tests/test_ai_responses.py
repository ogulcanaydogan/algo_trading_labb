from bot.ai import FeatureSnapshot, PredictionSnapshot, QuestionAnsweringEngine
from bot.state import BotState
from bot.strategy import StrategyConfig


def build_engine() -> QuestionAnsweringEngine:
    return QuestionAnsweringEngine(StrategyConfig(symbol="XAU/USD", timeframe="1h"))


def test_coalesce_lines_removes_duplicates_and_whitespace():
    engine = build_engine()
    merged = engine._coalesce_lines(
        [
            "  Duplicate sentence.  ",
            "Duplicate sentence.",
            "",
            "Unique insight.",
            "Unique insight.",
        ]
    )
    assert merged == "Duplicate sentence. Unique insight."


def test_ai_answer_includes_prediction_and_macro_once():
    engine = build_engine()
    state = BotState(symbol="XAU/USD")
    snapshot = PredictionSnapshot(
        recommended_action="LONG",
        confidence=0.6,
        probability_long=0.6,
        probability_short=0.3,
        probability_flat=0.1,
        expected_move_pct=1.23,
        summary="",
        features=FeatureSnapshot(
            ema_gap_pct=0.45,
            momentum_pct=0.12,
            rsi_distance_from_mid=1.2,
            volatility_pct=0.8,
        ),
        macro_bias=0.1,
        macro_confidence=0.5,
        macro_summary="Macro text",
    )
    answer = engine._answer_ai_question(state, snapshot, None)
    assert answer.count("AI model leans LONG") == 1
    assert "Expected move" in answer
