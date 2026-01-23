"""
Market Analyst using Claude API.

Provides AI-powered market analysis, trade explanations, and strategy advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from .client import ClaudeClient, ClaudeResponse


@dataclass
class MarketAnalysis:
    """Result of market analysis."""

    symbol: str
    market_type: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    confidence: float
    key_factors: List[str]
    risk_factors: List[str]
    short_term_outlook: str
    recommendation: str
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market_type": self.market_type,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "key_factors": self.key_factors,
            "risk_factors": self.risk_factors,
            "short_term_outlook": self.short_term_outlook,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeExplanation:
    """Explanation for a trade decision."""

    symbol: str
    action: str
    reasoning: str
    supporting_factors: List[str]
    risk_assessment: str
    confidence_level: str


class MarketAnalyst:
    """
    AI-powered market analyst using Claude.

    Features:
    - Market regime analysis
    - Trade explanations for audit trails
    - Strategy recommendations
    - News sentiment analysis (when price moves >2%)

    Uses haiku model for routine analysis (cost-effective)
    and sonnet for detailed explanations (higher quality).

    Usage:
        analyst = MarketAnalyst(client)
        analysis = analyst.analyze_market("BTC/USDT", market_data, regime="volatile")
        explanation = analyst.explain_trade("BTC/USDT", "BUY", trade_context)
    """

    # System prompts for different tasks
    MARKET_ANALYSIS_SYSTEM = """You are an expert cryptocurrency and financial markets analyst.
Analyze market data objectively and provide actionable insights.
Always structure your response as valid JSON.
Be concise but thorough. Focus on:
- Current market regime and trend
- Key support/resistance levels
- Volume and momentum analysis
- Risk factors
Do not provide financial advice, only analysis."""

    TRADE_EXPLANATION_SYSTEM = """You are a trading system explainer.
Your job is to explain why a trading system made a specific decision.
Be clear, factual, and educational.
Structure your response as valid JSON."""

    STRATEGY_ADVICE_SYSTEM = """You are a quantitative trading strategy consultant.
Provide objective analysis of trading strategies and suggest improvements.
Focus on risk management, position sizing, and market conditions.
Structure your response as valid JSON."""

    def __init__(
        self,
        client: Optional[ClaudeClient] = None,
        daily_budget: float = 5.0,
    ):
        self.client = client or ClaudeClient(daily_budget=daily_budget)

    def analyze_market(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        regime: str = "unknown",
        market_type: str = "crypto",
        detailed: bool = False,
    ) -> Optional[MarketAnalysis]:
        """
        Analyze market conditions for a symbol.

        Args:
            symbol: Trading symbol
            market_data: Recent OHLCV and indicator data
            regime: Current detected regime
            market_type: Type of market
            detailed: If True, use sonnet for better analysis

        Returns:
            MarketAnalysis or None if budget exceeded
        """
        prompt = f"""Analyze the current market conditions for {symbol}.

Market Type: {market_type}
Current Regime: {regime}

Recent Price Data:
{json.dumps(market_data.get("recent_prices", {}), indent=2)}

Technical Indicators:
{json.dumps(market_data.get("indicators", {}), indent=2)}

Provide your analysis as JSON with these fields:
- sentiment: "bullish", "bearish", or "neutral"
- confidence: 0.0 to 1.0
- key_factors: list of 3-5 key bullish/bearish factors
- risk_factors: list of 2-3 risk factors
- short_term_outlook: 1-2 sentence outlook
- recommendation: brief trading recommendation"""

        model = "sonnet" if detailed else "haiku"

        response = self.client.complete(
            prompt=prompt,
            system=self.MARKET_ANALYSIS_SYSTEM,
            model=model,
            max_tokens=800,
            temperature=0.3,
            purpose="market_analysis",
        )

        if response is None:
            return None

        try:
            # Parse JSON response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content)

            return MarketAnalysis(
                symbol=symbol,
                market_type=market_type,
                sentiment=data.get("sentiment", "neutral"),
                confidence=data.get("confidence", 0.5),
                key_factors=data.get("key_factors", []),
                risk_factors=data.get("risk_factors", []),
                short_term_outlook=data.get("short_term_outlook", ""),
                recommendation=data.get("recommendation", ""),
                timestamp=datetime.now(),
            )
        except json.JSONDecodeError:
            print(f"Failed to parse analysis response: {response.content[:100]}")
            return None

    def explain_trade(
        self,
        symbol: str,
        action: str,
        context: Dict[str, Any],
    ) -> Optional[TradeExplanation]:
        """
        Explain why a trade was executed.

        Args:
            symbol: Trading symbol
            action: Trade action (BUY/SELL)
            context: Trade context (signals, regime, prices, etc.)

        Returns:
            TradeExplanation or None
        """
        prompt = f"""Explain why the trading system executed a {action} order for {symbol}.

Trade Context:
- Signal: {context.get("signal", "N/A")}
- Market Regime: {context.get("regime", "N/A")}
- Confidence: {context.get("confidence", "N/A")}
- Price: ${context.get("price", "N/A")}
- Indicators: {json.dumps(context.get("indicators", {}), indent=2)}

Provide explanation as JSON:
- reasoning: 2-3 sentence explanation of why this trade was made
- supporting_factors: list of 3-5 factors that supported this decision
- risk_assessment: brief assessment of trade risk
- confidence_level: "high", "medium", or "low" """

        response = self.client.complete(
            prompt=prompt,
            system=self.TRADE_EXPLANATION_SYSTEM,
            model="haiku",  # Use cheap model for explanations
            max_tokens=500,
            temperature=0.3,
            purpose="trade_explanation",
        )

        if response is None:
            return None

        try:
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            data = json.loads(content)

            return TradeExplanation(
                symbol=symbol,
                action=action,
                reasoning=data.get("reasoning", ""),
                supporting_factors=data.get("supporting_factors", []),
                risk_assessment=data.get("risk_assessment", ""),
                confidence_level=data.get("confidence_level", "medium"),
            )
        except json.JSONDecodeError:
            return None

    def get_strategy_advice(
        self,
        portfolio_status: Dict[str, Any],
        recent_performance: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> Optional[str]:
        """
        Get strategic advice based on portfolio performance.

        Args:
            portfolio_status: Current portfolio state
            recent_performance: Recent P&L and metrics
            market_conditions: Current market regimes

        Returns:
            Strategy advice text or None
        """
        prompt = f"""Review this trading portfolio and provide strategic advice.

Portfolio Status:
{json.dumps(portfolio_status, indent=2)}

Recent Performance (7 days):
{json.dumps(recent_performance, indent=2)}

Market Conditions:
{json.dumps(market_conditions, indent=2)}

Provide strategic advice covering:
1. Position sizing adjustments
2. Risk management recommendations
3. Market regime considerations
4. Specific action items

Format as JSON:
- position_advice: string
- risk_advice: string
- regime_advice: string
- action_items: list of strings"""

        response = self.client.complete(
            prompt=prompt,
            system=self.STRATEGY_ADVICE_SYSTEM,
            model="sonnet",  # Use quality model for strategy
            max_tokens=1000,
            temperature=0.5,
            purpose="strategy_advice",
        )

        if response is None:
            return None

        return response.content

    def analyze_price_move(
        self,
        symbol: str,
        price_change_pct: float,
        timeframe: str = "1h",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Analyze significant price moves (>2% recommended).

        Args:
            symbol: Trading symbol
            price_change_pct: Price change percentage
            timeframe: Time period of the move
            context: Additional context (news, events)

        Returns:
            Analysis text or None
        """
        direction = "up" if price_change_pct > 0 else "down"

        prompt = f"""{symbol} moved {abs(price_change_pct):.1f}% {direction} in the last {timeframe}.

Additional context:
{json.dumps(context or {}, indent=2)}

Briefly explain:
1. What likely caused this move?
2. Is this move likely to continue or reverse?
3. Key levels to watch

Keep response under 150 words."""

        response = self.client.complete(
            prompt=prompt,
            system="You are a concise market analyst. Explain price movements factually.",
            model="haiku",
            max_tokens=300,
            temperature=0.3,
            purpose="price_move_analysis",
        )

        return response.content if response else None

    def generate_daily_market_summary(
        self,
        portfolio_state: Dict[str, Any],
        recent_trades: List[Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Generate a comprehensive daily market summary suitable for Telegram.

        Args:
            portfolio_state: Current portfolio state including:
                - total_value: Current portfolio value
                - positions: Dict of symbol -> position info
                - cash_balance: Available cash
                - daily_pnl: Today's P&L
                - daily_pnl_pct: Today's P&L percentage
            recent_trades: List of recent trades with:
                - symbol, action, price, quantity, timestamp, pnl
            market_data: Market data including:
                - prices: Dict of symbol -> current price
                - regimes: Dict of symbol -> detected regime
                - changes_24h: Dict of symbol -> 24h percentage change

        Returns:
            Formatted text summary suitable for Telegram or None if budget exceeded
        """
        # Build trade summary
        trades_summary = "None today"
        if recent_trades:
            trade_lines = []
            for t in recent_trades[:5]:  # Limit to 5 most recent
                pnl_str = f" (P&L: ${t.get('pnl', 0):+,.2f})" if "pnl" in t else ""
                trade_lines.append(
                    f"- {t.get('action', 'N/A')} {t.get('symbol', 'N/A')} "
                    f"@ ${t.get('price', 0):,.2f}{pnl_str}"
                )
            trades_summary = "\n".join(trade_lines)

        # Build position summary
        positions_summary = "No open positions"
        positions = portfolio_state.get("positions", {})
        if positions:
            pos_lines = []
            for symbol, pos in list(positions.items())[:5]:
                unrealized = pos.get("unrealized_pnl", 0)
                pos_lines.append(
                    f"- {symbol}: {pos.get('size', 0):.4f} @ ${pos.get('entry_price', 0):,.2f} "
                    f"(Unrealized: ${unrealized:+,.2f})"
                )
            positions_summary = "\n".join(pos_lines)

        # Build regime summary
        regimes = market_data.get("regimes", {})
        regime_summary = (
            ", ".join(f"{sym}: {reg}" for sym, reg in list(regimes.items())[:5])
            if regimes
            else "Not detected"
        )

        prompt = f"""Generate a comprehensive daily trading summary for Telegram notification.

PORTFOLIO STATUS:
- Total Value: ${portfolio_state.get("total_value", 0):,.2f}
- Cash Balance: ${portfolio_state.get("cash_balance", 0):,.2f}
- Daily P&L: ${portfolio_state.get("daily_pnl", 0):,.2f} ({portfolio_state.get("daily_pnl_pct", 0):+.2f}%)

CURRENT POSITIONS:
{positions_summary}

TODAY'S TRADES:
{trades_summary}

MARKET REGIMES:
{regime_summary}

24H PRICE CHANGES:
{json.dumps(market_data.get("changes_24h", {}), indent=2)}

Generate a daily summary that includes:
1. MARKET REGIME ANALYSIS: Brief assessment of current market conditions (2-3 sentences)
2. KEY PRICE MOVEMENTS: Notable price changes and what they might indicate (2-3 sentences)
3. PORTFOLIO PERFORMANCE: Analysis of today's P&L and position performance (2-3 sentences)
4. RISK ASSESSMENT: Current risk level and any concerns (1-2 sentences)
5. TOMORROW'S OUTLOOK: Brief outlook for the next trading day (1-2 sentences)

Format the response as plain text suitable for Telegram:
- Use simple formatting (no markdown except basic line breaks)
- Keep total response under 300 words
- Be concise but informative
- Include relevant numbers and percentages
- Start each section with the section name followed by colon"""

        response = self.client.complete(
            prompt=prompt,
            system="You are a professional trading analyst generating daily summaries for traders. "
            "Be concise, data-driven, and actionable. Format for mobile messaging apps.",
            model="sonnet",  # Use sonnet for quality analysis
            max_tokens=1000,
            temperature=0.4,
            purpose="daily_market_summary",
        )

        if response is None:
            return None

        # Clean up and format for Telegram
        summary = response.content.strip()
        # Add header and timestamp
        header = (
            f"DAILY MARKET SUMMARY\n{'=' * 25}\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        )
        footer = f"\n\n{'=' * 25}\nGenerated by AI Market Analyst"

        return header + summary + footer

    def explain_trade_briefly(
        self,
        symbol: str,
        action: str,
        price: float,
        regime: str,
        confidence: float,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate a brief AI explanation of why a trade makes sense.
        Uses Claude haiku for cost efficiency.

        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            action: Trade action ("BUY" or "SELL")
            price: Trade price
            regime: Current market regime (e.g., "trending", "volatile", "ranging")
            confidence: Signal confidence (0.0 to 1.0)
            indicators: Optional technical indicators at time of trade

        Returns:
            Brief explanation text (2-3 sentences) or None if budget exceeded
        """
        indicators_str = ""
        if indicators:
            indicator_parts = []
            for k, v in indicators.items():
                if isinstance(v, float):
                    indicator_parts.append(f"{k}: {v:.2f}")
                else:
                    indicator_parts.append(f"{k}: {v}")
            indicators_str = ", ".join(indicator_parts)

        prompt = f"""Briefly explain this trade decision in 2-3 sentences:

Trade: {action} {symbol} at ${price:,.2f}
Market Regime: {regime}
Signal Confidence: {confidence:.0%}
{f"Indicators: {indicators_str}" if indicators_str else ""}

Focus on:
- Why this trade aligns with current market conditions
- The key factor(s) driving this decision
- Any relevant risk consideration

Keep response under 60 words. No bullet points, just flowing sentences."""

        response = self.client.complete(
            prompt=prompt,
            system="You explain algorithmic trading decisions concisely. Be factual and brief.",
            model="haiku",  # Use haiku for cost efficiency
            max_tokens=150,
            temperature=0.3,
            purpose="trade_explanation_brief",
        )

        return response.content.strip() if response else None

    def generate_weekly_strategy_review(
        self,
        weekly_performance: Dict[str, Any],
        trade_history: List[Dict[str, Any]],
        market_conditions: Dict[str, Any],
    ) -> Optional[str]:
        """
        Generate strategic recommendations based on weekly performance.

        Args:
            weekly_performance: Week's performance data including:
                - total_pnl: Total P&L for the week
                - total_pnl_pct: Percentage return
                - total_trades: Number of trades
                - winning_trades: Number of winning trades
                - losing_trades: Number of losing trades
                - win_rate: Win rate percentage
                - avg_win: Average winning trade
                - avg_loss: Average losing trade
                - max_drawdown: Maximum drawdown during week
                - sharpe_ratio: Weekly Sharpe ratio (if available)
            trade_history: List of trades from the week
            market_conditions: Summary of market conditions including:
                - dominant_regimes: Most common regimes during week
                - volatility_level: Overall volatility assessment
                - trend_direction: General market trend

        Returns:
            Strategic review and recommendations or None if budget exceeded
        """
        # Analyze trade patterns
        trade_patterns = {"symbols": {}, "actions": {"BUY": 0, "SELL": 0}}
        for trade in trade_history:
            sym = trade.get("symbol", "unknown")
            trade_patterns["symbols"][sym] = trade_patterns["symbols"].get(sym, 0) + 1
            action = trade.get("action", "").upper()
            if action in trade_patterns["actions"]:
                trade_patterns["actions"][action] += 1

        most_traded = sorted(trade_patterns["symbols"].items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        prompt = f"""Provide a strategic weekly review and recommendations for this trading portfolio.

WEEKLY PERFORMANCE SUMMARY:
- Total P&L: ${weekly_performance.get("total_pnl", 0):,.2f} ({weekly_performance.get("total_pnl_pct", 0):+.2f}%)
- Total Trades: {weekly_performance.get("total_trades", 0)}
- Win Rate: {weekly_performance.get("win_rate", 0):.1f}%
- Winning Trades: {weekly_performance.get("winning_trades", 0)} (Avg: ${weekly_performance.get("avg_win", 0):,.2f})
- Losing Trades: {weekly_performance.get("losing_trades", 0)} (Avg: ${weekly_performance.get("avg_loss", 0):,.2f})
- Maximum Drawdown: {weekly_performance.get("max_drawdown", 0):.2f}%
- Sharpe Ratio: {weekly_performance.get("sharpe_ratio", "N/A")}

TRADING PATTERNS:
- Most Traded: {", ".join(f"{s[0]} ({s[1]} trades)" for s in most_traded) if most_traded else "N/A"}
- Buy/Sell Ratio: {trade_patterns["actions"]["BUY"]} buys / {trade_patterns["actions"]["SELL"]} sells

MARKET CONDITIONS:
- Dominant Regimes: {", ".join(market_conditions.get("dominant_regimes", ["N/A"]))}
- Volatility Level: {market_conditions.get("volatility_level", "N/A")}
- Trend Direction: {market_conditions.get("trend_direction", "N/A")}

Provide a strategic review covering:

1. PERFORMANCE ANALYSIS:
   - Assessment of the week's results
   - Comparison of win rate vs profit factor
   - Drawdown analysis

2. STRATEGY EFFECTIVENESS:
   - How well did the strategy adapt to market regimes?
   - Were position sizes appropriate for volatility?
   - Trading frequency assessment

3. RECOMMENDATIONS:
   - Specific adjustments to improve performance
   - Risk management suggestions
   - Position sizing recommendations

4. NEXT WEEK OUTLOOK:
   - What market conditions to expect
   - Strategy adjustments to consider
   - Key levels or events to watch

Format as plain text suitable for Telegram:
- Use clear section headers
- Keep each section to 2-4 bullet points
- Total response under 400 words
- Be specific and actionable"""

        response = self.client.complete(
            prompt=prompt,
            system="You are a quantitative trading strategist providing weekly portfolio reviews. "
            "Focus on actionable insights and data-driven recommendations. "
            "Be constructive even when performance is poor.",
            model="sonnet",  # Use sonnet for strategic analysis
            max_tokens=1200,
            temperature=0.5,
            purpose="weekly_strategy_review",
        )

        if response is None:
            return None

        # Format for Telegram
        review = response.content.strip()
        header = f"WEEKLY STRATEGY REVIEW\n{'=' * 25}\nWeek ending: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        footer = f"\n\n{'=' * 25}\nGenerated by AI Strategy Analyst"

        return header + review + footer

    def get_daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        positions: Dict[str, Any],
        signals: Dict[str, Any],
    ) -> Optional[str]:
        """
        Generate daily trading summary.

        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Today's P&L
            positions: Current positions
            signals: Today's signals

        Returns:
            Summary text or None
        """
        pnl_pct = (
            (daily_pnl / (portfolio_value - daily_pnl)) * 100 if portfolio_value != daily_pnl else 0
        )

        prompt = f"""Generate a brief daily trading summary.

Portfolio: ${portfolio_value:,.2f}
Daily P&L: ${daily_pnl:,.2f} ({pnl_pct:+.2f}%)

Positions:
{json.dumps(positions, indent=2)}

Today's Signals:
{json.dumps(signals, indent=2)}

Write a 3-4 sentence summary covering:
- Overall performance
- Key position changes
- Tomorrow's outlook

Be concise and professional."""

        response = self.client.complete(
            prompt=prompt,
            system="You are a trading system generating daily summaries for users.",
            model="haiku",
            max_tokens=200,
            temperature=0.5,
            purpose="daily_summary",
        )

        return response.content if response else None
