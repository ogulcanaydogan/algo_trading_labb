"""
AI Brain V2 - Advanced Adaptive Trading Intelligence.

A self-learning, multi-source AI system that continuously adapts
to maximize profit across all market conditions.

Key Features:
1. Multi-Source Data Integration (news, sentiment, on-chain, economic)
2. Reinforcement Learning for optimal action selection
3. Meta-Learning across assets and timeframes
4. Continuous adaptation with experience replay
5. Market regime awareness with dynamic strategy switching
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Complete market context for decision making."""
    symbol: str
    price: float
    timestamp: datetime

    # Technical indicators
    trend: str  # up, down, sideways
    volatility: float  # 0-1 scale
    rsi: float
    macd_signal: float

    # Market regime
    regime: str  # bull, bear, crash, sideways, volatile
    regime_confidence: float

    # Sentiment
    news_sentiment: float  # -1 to 1
    social_sentiment: float  # -1 to 1
    fear_greed_index: float  # 0-100

    # Cross-market signals
    btc_correlation: float  # -1 to 1
    sp500_trend: str
    dxy_trend: str  # Dollar index
    vix_level: float

    # On-chain (for crypto)
    whale_activity: str  # accumulating, distributing, neutral
    exchange_flow: str  # inflow, outflow, neutral

    # Economic
    fed_stance: str  # hawkish, dovish, neutral
    next_event: Optional[str] = None

    def to_features(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        features = [
            self.volatility,
            self.rsi / 100,
            self.macd_signal,
            self.regime_confidence,
            self.news_sentiment,
            self.social_sentiment,
            self.fear_greed_index / 100,
            self.btc_correlation,
            self.vix_level / 100,
            1 if self.trend == 'up' else (-1 if self.trend == 'down' else 0),
            1 if self.regime == 'bull' else (-1 if self.regime in ['bear', 'crash'] else 0),
            1 if self.whale_activity == 'accumulating' else (-1 if self.whale_activity == 'distributing' else 0),
            1 if self.exchange_flow == 'outflow' else (-1 if self.exchange_flow == 'inflow' else 0),
            1 if self.fed_stance == 'dovish' else (-1 if self.fed_stance == 'hawkish' else 0),
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class TradingExperience:
    """A single trading experience for learning."""
    context: MarketContext
    action: str  # BUY, SELL, HOLD
    position_size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_time: int = 0  # minutes
    was_stopped: bool = False
    was_target_hit: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def reward(self) -> float:
        """Calculate reward for reinforcement learning."""
        if self.exit_price is None:
            return 0.0

        # Base reward is P&L percentage
        base_reward = self.pnl_pct * 100

        # Bonus for hitting target
        if self.was_target_hit:
            base_reward *= 1.2

        # Penalty for being stopped out
        if self.was_stopped:
            base_reward *= 0.8

        # Efficiency bonus (more profit in less time)
        if self.hold_time > 0 and self.pnl_pct > 0:
            efficiency = self.pnl_pct / (self.hold_time / 60)  # profit per hour
            base_reward += efficiency * 10

        return base_reward


@dataclass
class ActionStrategy:
    """A trading strategy with learned parameters."""
    name: str
    regime: str
    confidence_threshold: float
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop: bool
    dca_enabled: bool
    max_hold_hours: int

    # Performance tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

    def update_performance(self, pnl: float, won: bool):
        """Update strategy performance."""
        self.total_trades += 1
        self.total_pnl += pnl
        if won:
            self.winning_trades += 1


class ExperienceReplayBuffer:
    """Prioritized experience replay for learning."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

    def add(self, experience: TradingExperience):
        """Add experience with priority based on reward magnitude."""
        priority = abs(experience.reward) + 0.01  # Small epsilon for zero rewards
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[TradingExperience]:
        """Sample experiences with priority weighting."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        # Convert to numpy for weighted sampling
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        return [self.buffer[i] for i in indices]

    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self.buffer:
            return {}

        rewards = [exp.reward for exp in self.buffer]
        return {
            'count': len(self.buffer),
            'avg_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'positive_pct': sum(1 for r in rewards if r > 0) / len(rewards)
        }


class AdaptiveQNetwork:
    """
    Simple Q-Learning network for action selection.

    State: Market context features
    Actions: BUY, SELL, HOLD with different position sizes
    """

    def __init__(self, state_dim: int = 14, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.learning_rate = learning_rate

        # Action space: (action, position_size_level)
        # Actions: BUY, SELL, HOLD
        # Position sizes: 0.25, 0.5, 0.75, 1.0 of max
        self.actions = [
            ('BUY', 0.25), ('BUY', 0.5), ('BUY', 0.75), ('BUY', 1.0),
            ('SELL', 0.25), ('SELL', 0.5), ('SELL', 0.75), ('SELL', 1.0),
            ('HOLD', 0.0)
        ]
        self.n_actions = len(self.actions)

        # Q-table approximation with linear weights
        self.weights = np.random.randn(state_dim, self.n_actions) * 0.01
        self.bias = np.zeros(self.n_actions)

        # Exploration parameters
        self.epsilon = 0.3  # Start with 30% exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Discount factor
        self.gamma = 0.95

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given state."""
        return np.dot(state, self.weights) + self.bias

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[str, float, int]:
        """Select action using epsilon-greedy policy."""
        if explore and random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # Exploit: best action
            q_values = self.get_q_values(state)
            action_idx = np.argmax(q_values)

        action, size = self.actions[action_idx]
        return action, size, action_idx

    def update(self, state: np.ndarray, action_idx: int, reward: float,
               next_state: Optional[np.ndarray] = None):
        """Update Q-values using temporal difference learning."""
        current_q = self.get_q_values(state)[action_idx]

        if next_state is not None:
            next_q = np.max(self.get_q_values(next_state))
            target = reward + self.gamma * next_q
        else:
            target = reward

        # TD error
        td_error = target - current_q

        # Update weights (gradient descent)
        self.weights[:, action_idx] += self.learning_rate * td_error * state
        self.bias[action_idx] += self.learning_rate * td_error

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return td_error

    def save(self, path: str):
        """Save model weights."""
        np.savez(path, weights=self.weights, bias=self.bias, epsilon=self.epsilon)

    def load(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            data = np.load(path)
            self.weights = data['weights']
            self.bias = data['bias']
            self.epsilon = float(data['epsilon'])


class StrategyEvolver:
    """
    Evolves trading strategies using genetic algorithm principles.

    Keeps a population of strategies and evolves them based on performance.
    """

    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.strategies: Dict[str, List[ActionStrategy]] = {}  # regime -> strategies
        self.generation = 0

        # Initialize population for each regime
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population of strategies."""
        regimes = ['bull', 'bear', 'sideways', 'volatile', 'crash']

        for regime in regimes:
            self.strategies[regime] = []
            for i in range(self.population_size):
                strategy = self._create_random_strategy(regime, i)
                self.strategies[regime].append(strategy)

    def _create_random_strategy(self, regime: str, idx: int) -> ActionStrategy:
        """Create a random strategy with regime-appropriate defaults."""
        # Base parameters vary by regime
        if regime == 'bull':
            conf_range = (0.5, 0.7)
            sl_range = (0.01, 0.03)
            tp_range = (0.02, 0.06)
            size_range = (0.5, 1.0)
        elif regime == 'bear':
            conf_range = (0.6, 0.8)
            sl_range = (0.01, 0.02)
            tp_range = (0.01, 0.03)
            size_range = (0.25, 0.5)
        elif regime == 'crash':
            conf_range = (0.75, 0.9)
            sl_range = (0.005, 0.01)
            tp_range = (0.01, 0.02)
            size_range = (0.1, 0.25)
        elif regime == 'volatile':
            conf_range = (0.65, 0.8)
            sl_range = (0.02, 0.04)
            tp_range = (0.03, 0.08)
            size_range = (0.25, 0.5)
        else:  # sideways
            conf_range = (0.55, 0.7)
            sl_range = (0.01, 0.02)
            tp_range = (0.015, 0.03)
            size_range = (0.4, 0.7)

        return ActionStrategy(
            name=f"{regime}_strategy_{idx}",
            regime=regime,
            confidence_threshold=random.uniform(*conf_range),
            position_size_pct=random.uniform(*size_range),
            stop_loss_pct=random.uniform(*sl_range),
            take_profit_pct=random.uniform(*tp_range),
            trailing_stop=random.random() > 0.5,
            dca_enabled=random.random() > 0.7,
            max_hold_hours=random.randint(1, 48)
        )

    def get_best_strategy(self, regime: str) -> ActionStrategy:
        """Get best performing strategy for regime."""
        if regime not in self.strategies:
            regime = 'sideways'  # Default

        strategies = self.strategies[regime]

        # Score by: win_rate * 0.4 + sharpe * 0.3 + total_pnl * 0.3
        def score(s: ActionStrategy) -> float:
            if s.total_trades < 3:
                return -1  # Not enough data
            return s.win_rate * 0.4 + s.sharpe_ratio * 0.3 + (s.total_pnl / 1000) * 0.3

        best = max(strategies, key=score)
        return best

    def evolve(self):
        """Evolve strategies based on performance."""
        self.generation += 1

        for regime, strategies in self.strategies.items():
            # Sort by performance
            sorted_strategies = sorted(
                strategies,
                key=lambda s: s.win_rate * s.total_pnl if s.total_trades > 0 else -1,
                reverse=True
            )

            # Keep top 50%
            survivors = sorted_strategies[:self.population_size // 2]

            # Create offspring through crossover and mutation
            offspring = []
            while len(offspring) < self.population_size // 2:
                parent1, parent2 = random.sample(survivors, 2)
                child = self._crossover(parent1, parent2, regime, len(offspring))
                child = self._mutate(child)
                offspring.append(child)

            self.strategies[regime] = survivors + offspring

    def _crossover(self, p1: ActionStrategy, p2: ActionStrategy,
                   regime: str, idx: int) -> ActionStrategy:
        """Create child strategy from two parents."""
        return ActionStrategy(
            name=f"{regime}_gen{self.generation}_{idx}",
            regime=regime,
            confidence_threshold=(p1.confidence_threshold + p2.confidence_threshold) / 2,
            position_size_pct=(p1.position_size_pct + p2.position_size_pct) / 2,
            stop_loss_pct=(p1.stop_loss_pct + p2.stop_loss_pct) / 2,
            take_profit_pct=(p1.take_profit_pct + p2.take_profit_pct) / 2,
            trailing_stop=random.choice([p1.trailing_stop, p2.trailing_stop]),
            dca_enabled=random.choice([p1.dca_enabled, p2.dca_enabled]),
            max_hold_hours=random.choice([p1.max_hold_hours, p2.max_hold_hours])
        )

    def _mutate(self, strategy: ActionStrategy, mutation_rate: float = 0.2) -> ActionStrategy:
        """Apply random mutations to strategy."""
        if random.random() < mutation_rate:
            strategy.confidence_threshold *= random.uniform(0.9, 1.1)
            strategy.confidence_threshold = np.clip(strategy.confidence_threshold, 0.4, 0.95)

        if random.random() < mutation_rate:
            strategy.position_size_pct *= random.uniform(0.8, 1.2)
            strategy.position_size_pct = np.clip(strategy.position_size_pct, 0.1, 1.0)

        if random.random() < mutation_rate:
            strategy.stop_loss_pct *= random.uniform(0.8, 1.2)
            strategy.stop_loss_pct = np.clip(strategy.stop_loss_pct, 0.005, 0.1)

        if random.random() < mutation_rate:
            strategy.take_profit_pct *= random.uniform(0.8, 1.2)
            strategy.take_profit_pct = np.clip(strategy.take_profit_pct, 0.01, 0.15)

        return strategy


class AIBrainV2:
    """
    Advanced Adaptive Trading Intelligence.

    Combines multiple learning approaches:
    1. Q-Learning for action selection
    2. Genetic algorithm for strategy evolution
    3. Experience replay for continuous learning
    4. Multi-source data integration
    """

    def __init__(self, data_dir: str = "data/ai_brain_v2"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.q_network = AdaptiveQNetwork()
        self.strategy_evolver = StrategyEvolver()
        self.experience_buffer = ExperienceReplayBuffer(capacity=10000)

        # Symbol-specific learnings
        self.symbol_stats: Dict[str, Dict] = {}

        # Learning parameters
        self.batch_size = 32
        self.learn_every_n_trades = 5
        self.evolve_every_n_trades = 50
        self.trade_count = 0

        # Performance tracking
        self.total_reward = 0.0
        self.recent_rewards: deque = deque(maxlen=100)

        # Load saved state
        self._load_state()

        logger.info("AI Brain V2 initialized")

    def _load_state(self):
        """Load saved brain state."""
        state_path = self.data_dir / "brain_state.json"
        q_path = self.data_dir / "q_network.npz"

        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.trade_count = state.get('trade_count', 0)
            self.total_reward = state.get('total_reward', 0.0)
            self.symbol_stats = state.get('symbol_stats', {})
            logger.info(f"Loaded brain state: {self.trade_count} trades, {self.total_reward:.2f} total reward")

        if q_path.exists():
            self.q_network.load(str(q_path))
            logger.info("Loaded Q-network weights")

    def _save_state(self):
        """Save brain state."""
        state = {
            'trade_count': self.trade_count,
            'total_reward': self.total_reward,
            'symbol_stats': self.symbol_stats,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.data_dir / "brain_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        self.q_network.save(str(self.data_dir / "q_network.npz"))

    async def get_trading_decision(
        self,
        symbol: str,
        ml_signal: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get optimal trading decision combining all intelligence sources.

        Args:
            symbol: Trading symbol
            ml_signal: Signal from ML predictor (action, confidence, etc.)
            market_data: Current market data (price, indicators, etc.)

        Returns:
            Enhanced trading decision with optimal parameters
        """
        # Build market context
        context = await self._build_market_context(symbol, ml_signal, market_data)

        # Get Q-network action
        state = context.to_features()
        q_action, q_size, action_idx = self.q_network.select_action(state, explore=True)

        # Get best strategy for current regime
        strategy = self.strategy_evolver.get_best_strategy(context.regime)

        # Combine ML signal with RL decision
        ml_action = ml_signal.get('action', 'HOLD')
        ml_confidence = ml_signal.get('confidence', 0.5)

        # Decision logic:
        # - If ML and RL agree, high confidence
        # - If they disagree, use the one with better recent performance
        # - Apply strategy parameters

        if ml_action == q_action:
            # Agreement - boost confidence
            final_action = ml_action
            final_confidence = min(0.95, ml_confidence * 1.2)
            agreement = "ML+RL agree"
        else:
            # Disagreement - evaluate recent performance
            symbol_stats = self.symbol_stats.get(symbol, {})
            ml_recent_accuracy = symbol_stats.get('ml_accuracy', 0.5)
            rl_recent_accuracy = symbol_stats.get('rl_accuracy', 0.5)

            if ml_recent_accuracy > rl_recent_accuracy:
                final_action = ml_action
                final_confidence = ml_confidence
                agreement = "ML override"
            else:
                final_action = q_action
                final_confidence = ml_confidence * 0.9  # Slightly lower confidence
                agreement = "RL override"

        # Apply strategy thresholds
        if final_confidence < strategy.confidence_threshold:
            final_action = 'HOLD'
            reason = f"Below threshold ({final_confidence:.2f} < {strategy.confidence_threshold:.2f})"
        else:
            reason = f"{agreement}, confidence {final_confidence:.2f}"

        # Calculate position size
        position_size = strategy.position_size_pct * q_size

        # Build enhanced decision
        decision = {
            'action': final_action,
            'confidence': final_confidence,
            'position_size_pct': position_size,
            'stop_loss_pct': strategy.stop_loss_pct,
            'take_profit_pct': strategy.take_profit_pct,
            'trailing_stop': strategy.trailing_stop,
            'dca_enabled': strategy.dca_enabled,
            'max_hold_hours': strategy.max_hold_hours,
            'reason': reason,
            'regime': context.regime,
            'strategy_name': strategy.name,
            'q_action': q_action,
            'ml_action': ml_action,
            'context': {
                'trend': context.trend,
                'volatility': context.volatility,
                'news_sentiment': context.news_sentiment,
                'fear_greed': context.fear_greed_index
            }
        }

        logger.info(f"[{symbol}] AI Decision: {final_action} ({reason})")
        return decision

    async def _build_market_context(
        self,
        symbol: str,
        ml_signal: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> MarketContext:
        """Build complete market context from available data."""
        # Extract from ML signal metadata
        meta = ml_signal.get('signal_meta', {})

        # Get additional data (with defaults)
        sentiment = await self._get_sentiment(symbol)
        macro = await self._get_macro_context()
        onchain = await self._get_onchain_data(symbol) if 'USDT' in symbol else {}

        return MarketContext(
            symbol=symbol,
            price=market_data.get('price', 0),
            timestamp=datetime.now(),

            # Technical
            trend=meta.get('trend', 'neutral'),
            volatility=meta.get('volatility_score', 0.5) if isinstance(meta.get('volatility_score'), float) else 0.5,
            rsi=meta.get('rsi', 50),
            macd_signal=meta.get('macd_signal', 0),

            # Regime
            regime=meta.get('regime', 'unknown'),
            regime_confidence=meta.get('regime_confidence', 0.5),

            # Sentiment
            news_sentiment=sentiment.get('news', 0),
            social_sentiment=sentiment.get('social', 0),
            fear_greed_index=sentiment.get('fear_greed', 50),

            # Cross-market
            btc_correlation=macro.get('btc_correlation', 0),
            sp500_trend=macro.get('sp500_trend', 'neutral'),
            dxy_trend=macro.get('dxy_trend', 'neutral'),
            vix_level=macro.get('vix', 20),

            # On-chain
            whale_activity=onchain.get('whale_activity', 'neutral'),
            exchange_flow=onchain.get('exchange_flow', 'neutral'),

            # Economic
            fed_stance=macro.get('fed_stance', 'neutral'),
            next_event=macro.get('next_event')
        )

    async def _get_sentiment(self, symbol: str) -> Dict[str, float]:
        """Get sentiment data for symbol."""
        # Try to use existing news reasoner
        try:
            from bot.intelligence.news_reasoner import NewsReasoner
            reasoner = NewsReasoner()
            score = await reasoner.analyze_sentiment(symbol)
            return {
                'news': score,
                'social': 0,  # Could integrate Twitter/Reddit later
                'fear_greed': 50  # Could fetch from alternative.me API
            }
        except Exception as e:
            logger.debug(f"Sentiment fetch failed: {e}")
            return {'news': 0, 'social': 0, 'fear_greed': 50}

    async def _get_macro_context(self) -> Dict[str, Any]:
        """Get macro economic context."""
        # This could be enhanced with real API calls
        return {
            'btc_correlation': 0.7,
            'sp500_trend': 'up',
            'dxy_trend': 'neutral',
            'vix': 18,
            'fed_stance': 'neutral',
            'next_event': None
        }

    async def _get_onchain_data(self, symbol: str) -> Dict[str, str]:
        """Get on-chain data for crypto."""
        # Could integrate Glassnode, CryptoQuant APIs
        return {
            'whale_activity': 'neutral',
            'exchange_flow': 'neutral'
        }

    def record_trade_outcome(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        hold_time_minutes: int,
        was_stopped: bool,
        was_target_hit: bool,
        context: Optional[MarketContext] = None
    ):
        """
        Record trade outcome for learning.

        This is called after every trade closes.
        """
        self.trade_count += 1

        # Create experience (use simple context if not provided)
        if context is None:
            context = MarketContext(
                symbol=symbol,
                price=entry_price,
                timestamp=datetime.now(),
                trend='neutral', volatility=0.5, rsi=50, macd_signal=0,
                regime='unknown', regime_confidence=0.5,
                news_sentiment=0, social_sentiment=0, fear_greed_index=50,
                btc_correlation=0, sp500_trend='neutral', dxy_trend='neutral', vix_level=20,
                whale_activity='neutral', exchange_flow='neutral', fed_stance='neutral'
            )

        experience = TradingExperience(
            context=context,
            action=action,
            position_size=1.0,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time=hold_time_minutes,
            was_stopped=was_stopped,
            was_target_hit=was_target_hit
        )

        # Add to experience buffer
        self.experience_buffer.add(experience)

        # Track reward
        reward = experience.reward
        self.total_reward += reward
        self.recent_rewards.append(reward)

        # Update symbol stats
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = {
                'trades': 0, 'wins': 0, 'total_pnl': 0,
                'ml_correct': 0, 'rl_correct': 0
            }

        stats = self.symbol_stats[symbol]
        stats['trades'] += 1
        stats['total_pnl'] += pnl
        if pnl > 0:
            stats['wins'] += 1

        # Update strategy performance
        strategy = self.strategy_evolver.get_best_strategy(context.regime)
        strategy.update_performance(pnl, pnl > 0)

        # Learn from experience
        if self.trade_count % self.learn_every_n_trades == 0:
            self._learn_from_experiences()

        # Evolve strategies periodically
        if self.trade_count % self.evolve_every_n_trades == 0:
            self.strategy_evolver.evolve()
            logger.info(f"Strategies evolved to generation {self.strategy_evolver.generation}")

        # Save state
        self._save_state()

        logger.info(f"[{symbol}] Recorded trade: {action} -> PnL {pnl_pct*100:.2f}%, Reward {reward:.2f}")

    def _learn_from_experiences(self):
        """Learn from batch of experiences using Q-learning."""
        if len(self.experience_buffer.buffer) < self.batch_size:
            return

        batch = self.experience_buffer.sample(self.batch_size)

        total_td_error = 0
        for exp in batch:
            state = exp.context.to_features()

            # Map action to index
            action_map = {'BUY': 0, 'SELL': 4, 'HOLD': 8}
            action_idx = action_map.get(exp.action, 8)

            # Update Q-network
            td_error = self.q_network.update(state, action_idx, exp.reward, None)
            total_td_error += abs(td_error)

        avg_td_error = total_td_error / len(batch)
        logger.info(f"Learning update: avg TD error = {avg_td_error:.4f}, epsilon = {self.q_network.epsilon:.3f}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        buffer_stats = self.experience_buffer.get_statistics()

        # Calculate recent performance
        recent_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0

        # Strategy performance by regime
        strategy_perf = {}
        for regime, strategies in self.strategy_evolver.strategies.items():
            best = max(strategies, key=lambda s: s.win_rate if s.total_trades > 0 else 0)
            strategy_perf[regime] = {
                'best_strategy': best.name,
                'win_rate': best.win_rate,
                'total_pnl': best.total_pnl,
                'trades': best.total_trades
            }

        return {
            'total_trades': self.trade_count,
            'total_reward': self.total_reward,
            'recent_avg_reward': recent_reward,
            'q_network_epsilon': self.q_network.epsilon,
            'strategy_generation': self.strategy_evolver.generation,
            'experience_buffer': buffer_stats,
            'strategy_performance': strategy_perf,
            'symbol_stats': self.symbol_stats
        }


# Global instance
_ai_brain_v2: Optional[AIBrainV2] = None

def get_ai_brain_v2() -> AIBrainV2:
    """Get or create the AI Brain V2 instance."""
    global _ai_brain_v2
    if _ai_brain_v2 is None:
        _ai_brain_v2 = AIBrainV2()
    return _ai_brain_v2
