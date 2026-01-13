"""
LLM Client for Strategy Development and News Analysis
Connects to local Ollama instance for trading strategy assistance
"""
import json
from typing import Any, Dict, List, Optional

import requests


class LLMClient:
    """Client for interacting with local LLM (Ollama)"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral"):
        self.base_url = base_url
        self.model = model
        self.timeout = 120  # 2 minutes timeout

    def ask(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Ask LLM a question and get response

        Args:
            prompt: User question
            system_prompt: System role (optional)
            temperature: Creativity level (0.0-1.0)

        Returns:
            LLM response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            return f"LLM connection error: {e}"

    def analyze_news(
        self,
        news_items: List[Dict[str, Any]],
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Analyze news and return sentiment + impact

        Args:
            news_items: List of news (title, summary, etc.)
            symbol: Symbol (BTC/USDT, NVDA, etc.)

        Returns:
            {
                "sentiment": "bullish" | "bearish" | "neutral",
                "impact": "low" | "medium" | "high" | "critical",
                "bias_score": -1.0 to 1.0,
                "confidence": 0.0 to 1.0,
                "summary": "Brief explanation",
                "catalysts": ["catalyst 1", "catalyst 2", ...]
            }
        """
        # Convert news to text
        news_text = "\n".join(
            [
                f"- {item.get('title', '')} ({item.get('published', 'N/A')})"
                for item in news_items[-10:]  # Last 10 news
            ]
        )

        system_prompt = """You are a financial analyst and trading expert.
You analyze news and evaluate the potential impact on an asset's price.
Respond in JSON format."""

        prompt = f"""
The following news is related to {symbol}:

{news_text}

Please analyze this news and respond in the following JSON format:

{{
  "sentiment": "bullish or bearish or neutral",
  "impact": "low or medium or high or critical",
  "bias_score": number between -1.0 and 1.0 (negative=bearish, positive=bullish),
  "confidence": number between 0.0 and 1.0 (analysis reliability),
  "summary": "Brief summary (max 2 sentences)",
  "catalysts": ["main catalyst 1", "main catalyst 2"]
}}

Return ONLY JSON, do not add any other explanation.
"""

        response = self.ask(prompt, system_prompt=system_prompt, temperature=0.3)

        # Parse JSON
        try:
            # Extract JSON (if inside markdown code block)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            return json.loads(response)
        except json.JSONDecodeError:
            # Return default values if parsing fails
            return {
                "sentiment": "neutral",
                "impact": "low",
                "bias_score": 0.0,
                "confidence": 0.3,
                "summary": "LLM analysis could not be parsed",
                "catalysts": []
            }

    def suggest_strategy(
        self,
        symbol: str,
        historical_performance: Dict[str, Any],
        market_conditions: Dict[str, Any],
    ) -> str:
        """
        Strategy suggestion based on current performance and market conditions

        Args:
            symbol: Symbol
            historical_performance: Backtest results
            market_conditions: Market state (volatility, trend, etc.)

        Returns:
            Strategy suggestion text
        """
        system_prompt = """You are an algorithmic trading strategist and quantitative analyst.
You analyze backtest results and market conditions to suggest strategy improvements."""

        prompt = f"""
Symbol: {symbol}

Current Performance:
- Sharpe Ratio: {historical_performance.get('sharpe_ratio', 'N/A')}
- Win Rate: {historical_performance.get('win_rate', 'N/A')}%
- Max Drawdown: {historical_performance.get('max_drawdown_pct', 'N/A')}%
- Total Return: {historical_performance.get('total_pnl_pct', 'N/A')}%
- Total Trades: {historical_performance.get('total_trades', 'N/A')}

Market Conditions:
- Volatility: {market_conditions.get('volatility', 'N/A')}
- Trend: {market_conditions.get('trend', 'N/A')}
- RSI: {market_conditions.get('rsi', 'N/A')}

To improve this performance:
1. Which parameters should I adjust?
2. Which additional indicators should I add?
3. How can risk management be optimized?
4. Which strategy type is more suitable in these market conditions? (trend-following, mean-reversion, etc.)

Please provide concrete, actionable suggestions.
"""

        return self.ask(prompt, system_prompt=system_prompt, temperature=0.7)

    def optimize_parameters(
        self,
        symbol: str,
        current_params: Dict[str, Any],
        performance_history: List[Dict[str, Any]],
    ) -> str:
        """
        Parameter optimization suggestion

        Args:
            symbol: Symbol
            current_params: Current parameters
            performance_history: Performance of different parameter combinations

        Returns:
            Optimization suggestion
        """
        system_prompt = """You are a parameter optimization expert.
You interpret grid search or Bayesian optimization results and suggest the best approach."""

        # Get top 5 combinations
        top_5 = sorted(
            performance_history,
            key=lambda x: x.get("sharpe_ratio", 0),
            reverse=True,
        )[:5]

        prompt = f"""
Symbol: {symbol}

Current Parameters:
{json.dumps(current_params, indent=2)}

Top 5 Combinations:
{json.dumps(top_5, indent=2)}

Based on these results:
1. Which parameters affect performance the most?
2. What is the relationship between parameters?
3. Is there overfitting risk?
4. What are the recommended new parameter ranges?

Provide concrete numerical suggestions.
"""

        return self.ask(prompt, system_prompt=system_prompt, temperature=0.5)

    def explain_trade(
        self,
        trade_data: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> str:
        """
        Explain why a trade was opened/closed

        Args:
            trade_data: Trade details
            market_context: Market state

        Returns:
            Explanation text
        """
        system_prompt = """You are a trading educator.
You explain trades in a clear and understandable way."""

        prompt = f"""
Trade Details:
- Side: {trade_data.get('side', 'N/A')}
- Entry: ${trade_data.get('entry_price', 'N/A')}
- Exit: ${trade_data.get('exit_price', 'N/A')}
- P&L: {trade_data.get('pnl_pct', 'N/A')}%
- Exit Reason: {trade_data.get('exit_reason', 'N/A')}

Market State:
- EMA Fast: {market_context.get('ema_fast', 'N/A')}
- EMA Slow: {market_context.get('ema_slow', 'N/A')}
- RSI: {market_context.get('rsi', 'N/A')}
- Price: ${market_context.get('price', 'N/A')}

Why did we open this trade and why did it close this way?
Explain with technical analysis (2-3 sentences).
"""

        return self.ask(prompt, system_prompt=system_prompt, temperature=0.5)

    def health_check(self) -> bool:
        """Check if LLM service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


# Global instance
_llm_client: Optional[LLMClient] = None


def get_llm_client(model: str = "mistral") -> LLMClient:
    """Get singleton LLM client"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient(model=model)
    return _llm_client


if __name__ == "__main__":
    # Test
    client = LLMClient()

    print("LLM Health Check...")
    if client.health_check():
        print("LLM service is running!")
    else:
        print("LLM service not responding. Run 'ollama serve'.")
        exit(1)

    print("\nTest Question...")
    response = client.ask("When does EMA crossover strategy for Bitcoin open a long position? Brief explanation.")
    print(f"Response: {response}")

    print("\nNews Analysis Test...")
    test_news = [
        {"title": "Fed announced it will continue raising rates", "published": "2025-11-01"},
        {"title": "Bitcoin ETF approvals approaching", "published": "2025-11-01"},
    ]
    analysis = client.analyze_news(test_news, "BTC/USDT")
    print(f"Analysis: {json.dumps(analysis, indent=2, ensure_ascii=False)}")
