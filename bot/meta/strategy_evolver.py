"""
Strategy Evolver.

Genetic algorithm-based strategy evolution:
- Population of strategy configurations
- Fitness evaluation (Sharpe, win rate, drawdown)
- Crossover and mutation
- Automatic discovery of profitable strategies
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


@dataclass
class StrategyGene:
    """A single strategy configuration (individual in GA)."""
    gene_id: str
    name: str

    # Entry parameters
    entry_rsi_threshold: float  # Buy when RSI below this
    entry_momentum_threshold: float  # Momentum requirement
    entry_volume_threshold: float  # Volume ratio requirement
    entry_regime_filter: List[str]  # Allowed regimes

    # Exit parameters
    take_profit_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    max_hold_hours: int

    # Position sizing
    base_position_pct: float  # % of portfolio
    leverage_multiplier: float

    # Risk parameters
    max_daily_trades: int
    min_confidence: float
    cooldown_minutes: int

    # Shorting
    short_enabled: bool
    short_rsi_threshold: float  # Short when RSI above this

    # Fitness metrics (filled after evaluation)
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0

    generation: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionConfig:
    """Configuration for the evolution process."""
    population_size: int = 50
    elite_count: int = 5  # Top performers kept unchanged
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7

    # Fitness weights
    sharpe_weight: float = 0.40
    win_rate_weight: float = 0.30
    drawdown_weight: float = 0.20
    pnl_weight: float = 0.10

    # Constraints
    min_sharpe: float = 0.5
    min_win_rate: float = 0.45
    max_drawdown: float = 0.15

    # Evolution
    max_generations: int = 100
    convergence_threshold: float = 0.01  # Stop if improvement < 1%


class StrategyEvolver:
    """
    Evolves trading strategies using genetic algorithms.

    Features:
    - Random strategy generation
    - Fitness evaluation
    - Selection, crossover, mutation
    - Elitism (preserve best)
    - Automatic convergence detection
    """

    # Parameter ranges for random generation
    PARAM_RANGES = {
        "entry_rsi_threshold": (20, 45),
        "entry_momentum_threshold": (-0.02, 0.02),
        "entry_volume_threshold": (0.5, 2.0),
        "take_profit_pct": (0.02, 0.15),
        "stop_loss_pct": (0.01, 0.05),
        "trailing_stop_pct": (0.01, 0.04),
        "max_hold_hours": (1, 72),
        "base_position_pct": (0.02, 0.10),
        "leverage_multiplier": (1.0, 3.0),
        "max_daily_trades": (3, 20),
        "min_confidence": (0.5, 0.8),
        "cooldown_minutes": (5, 60),
        "short_rsi_threshold": (55, 80),
    }

    REGIME_OPTIONS = [
        "BULL", "STRONG_BULL", "BEAR", "STRONG_BEAR",
        "SIDEWAYS", "HIGH_VOL", "LOW_VOL", "CRASH", "RECOVERY"
    ]

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        fitness_function: Optional[Callable[[StrategyGene], float]] = None,
    ):
        self.config = config or EvolutionConfig()
        self.fitness_function = fitness_function

        # Population
        self._population: List[StrategyGene] = []
        self._generation = 0

        # History
        self._best_fitness_history: List[float] = []
        self._avg_fitness_history: List[float] = []
        self._best_ever: Optional[StrategyGene] = None

        # Hall of fame
        self._hall_of_fame: List[StrategyGene] = []

        logger.info("StrategyEvolver initialized")

    def initialize_population(self):
        """Create initial random population."""
        self._population = []

        for i in range(self.config.population_size):
            gene = self._create_random_gene(f"gen0_ind{i}")
            self._population.append(gene)

        self._generation = 0
        logger.info(f"Initialized population with {len(self._population)} individuals")

    def _create_random_gene(self, gene_id: str) -> StrategyGene:
        """Create a random strategy gene."""
        # Random regime filter (2-5 regimes)
        num_regimes = random.randint(2, 5)
        regime_filter = random.sample(self.REGIME_OPTIONS, num_regimes)

        return StrategyGene(
            gene_id=gene_id,
            name=f"Strategy_{gene_id}",
            entry_rsi_threshold=self._random_param("entry_rsi_threshold"),
            entry_momentum_threshold=self._random_param("entry_momentum_threshold"),
            entry_volume_threshold=self._random_param("entry_volume_threshold"),
            entry_regime_filter=regime_filter,
            take_profit_pct=self._random_param("take_profit_pct"),
            stop_loss_pct=self._random_param("stop_loss_pct"),
            trailing_stop_pct=self._random_param("trailing_stop_pct"),
            max_hold_hours=int(self._random_param("max_hold_hours")),
            base_position_pct=self._random_param("base_position_pct"),
            leverage_multiplier=self._random_param("leverage_multiplier"),
            max_daily_trades=int(self._random_param("max_daily_trades")),
            min_confidence=self._random_param("min_confidence"),
            cooldown_minutes=int(self._random_param("cooldown_minutes")),
            short_enabled=random.random() > 0.5,
            short_rsi_threshold=self._random_param("short_rsi_threshold"),
            generation=self._generation,
        )

    def _random_param(self, param_name: str) -> float:
        """Generate random value for a parameter."""
        min_val, max_val = self.PARAM_RANGES[param_name]
        return random.uniform(min_val, max_val)

    def evaluate_population(
        self,
        backtest_function: Optional[Callable[[StrategyGene], Dict]] = None,
    ):
        """
        Evaluate fitness of all individuals.

        Args:
            backtest_function: Function that takes a StrategyGene and returns
                              {sharpe, win_rate, max_drawdown, total_pnl, total_trades}
        """
        for gene in self._population:
            if backtest_function:
                results = backtest_function(gene)
                gene.sharpe_ratio = results.get("sharpe", 0.0)
                gene.win_rate = results.get("win_rate", 0.0)
                gene.max_drawdown = results.get("max_drawdown", 0.0)
                gene.total_pnl = results.get("total_pnl", 0.0)
                gene.total_trades = results.get("total_trades", 0)

            # Calculate composite fitness
            gene.fitness = self._calculate_fitness(gene)

        # Sort by fitness
        self._population.sort(key=lambda g: g.fitness, reverse=True)

        # Update best ever
        if self._population and (
            self._best_ever is None or
            self._population[0].fitness > self._best_ever.fitness
        ):
            self._best_ever = copy.deepcopy(self._population[0])

        # Update hall of fame
        self._update_hall_of_fame()

        # Record history
        fitnesses = [g.fitness for g in self._population]
        self._best_fitness_history.append(max(fitnesses))
        self._avg_fitness_history.append(sum(fitnesses) / len(fitnesses))

        logger.info(
            f"Generation {self._generation}: "
            f"Best fitness={self._population[0].fitness:.4f}, "
            f"Avg={sum(fitnesses)/len(fitnesses):.4f}"
        )

    def _calculate_fitness(self, gene: StrategyGene) -> float:
        """Calculate composite fitness score."""
        # Use custom fitness function if provided
        if self.fitness_function:
            return self.fitness_function(gene)

        # Default fitness calculation
        fitness = 0.0

        # Sharpe ratio (higher is better)
        sharpe_score = min(1.0, gene.sharpe_ratio / 2.0)  # Normalize to 0-1
        fitness += sharpe_score * self.config.sharpe_weight

        # Win rate (higher is better)
        win_rate_score = gene.win_rate
        fitness += win_rate_score * self.config.win_rate_weight

        # Drawdown penalty (lower is better)
        drawdown_score = 1.0 - min(1.0, gene.max_drawdown / 0.20)  # 20% max
        fitness += drawdown_score * self.config.drawdown_weight

        # P&L bonus
        pnl_score = min(1.0, max(0.0, (gene.total_pnl + 1000) / 2000))  # Normalize
        fitness += pnl_score * self.config.pnl_weight

        # Penalties for constraint violations
        if gene.sharpe_ratio < self.config.min_sharpe:
            fitness *= 0.8
        if gene.win_rate < self.config.min_win_rate:
            fitness *= 0.8
        if gene.max_drawdown > self.config.max_drawdown:
            fitness *= 0.7

        # Bonus for sufficient trades (statistical significance)
        if gene.total_trades < 20:
            fitness *= 0.5
        elif gene.total_trades < 50:
            fitness *= 0.8

        return fitness

    def evolve_generation(self):
        """Create next generation through selection, crossover, mutation."""
        new_population = []

        # Elitism: Keep top performers
        elites = self._population[:self.config.elite_count]
        for elite in elites:
            elite_copy = copy.deepcopy(elite)
            elite_copy.generation = self._generation + 1
            new_population.append(elite_copy)

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            # Selection (tournament)
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)

            # Update metadata
            child1.gene_id = f"gen{self._generation + 1}_ind{len(new_population)}"
            child1.generation = self._generation + 1
            child1.fitness = 0.0
            new_population.append(child1)

            if len(new_population) < self.config.population_size:
                child2.gene_id = f"gen{self._generation + 1}_ind{len(new_population)}"
                child2.generation = self._generation + 1
                child2.fitness = 0.0
                new_population.append(child2)

        self._population = new_population
        self._generation += 1

        logger.info(f"Evolved to generation {self._generation}")

    def _tournament_select(self, tournament_size: int = 3) -> StrategyGene:
        """Select individual via tournament selection."""
        tournament = random.sample(self._population, tournament_size)
        winner = max(tournament, key=lambda g: g.fitness)
        return winner

    def _crossover(
        self,
        parent1: StrategyGene,
        parent2: StrategyGene,
    ) -> Tuple[StrategyGene, StrategyGene]:
        """Perform crossover between two parents."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Uniform crossover for numeric parameters
        numeric_params = [
            "entry_rsi_threshold", "entry_momentum_threshold",
            "entry_volume_threshold", "take_profit_pct", "stop_loss_pct",
            "trailing_stop_pct", "max_hold_hours", "base_position_pct",
            "leverage_multiplier", "max_daily_trades", "min_confidence",
            "cooldown_minutes", "short_rsi_threshold"
        ]

        for param in numeric_params:
            if random.random() < 0.5:
                # Swap values
                val1 = getattr(child1, param)
                val2 = getattr(child2, param)
                setattr(child1, param, val2)
                setattr(child2, param, val1)

        # Special handling for regime filter (combine and redistribute)
        all_regimes = list(set(child1.entry_regime_filter + child2.entry_regime_filter))
        random.shuffle(all_regimes)
        split = len(all_regimes) // 2
        child1.entry_regime_filter = all_regimes[:max(2, split)]
        child2.entry_regime_filter = all_regimes[split:] if split < len(all_regimes) else all_regimes[:2]

        # Swap short_enabled with 50% probability
        if random.random() < 0.5:
            child1.short_enabled, child2.short_enabled = child2.short_enabled, child1.short_enabled

        return child1, child2

    def _mutate(self, gene: StrategyGene) -> StrategyGene:
        """Apply random mutation to a gene."""
        mutated = copy.deepcopy(gene)

        # Randomly mutate 1-3 parameters
        num_mutations = random.randint(1, 3)
        numeric_params = list(self.PARAM_RANGES.keys())

        for _ in range(num_mutations):
            param = random.choice(numeric_params)
            current_val = getattr(mutated, param)
            min_val, max_val = self.PARAM_RANGES[param]

            # Gaussian mutation
            mutation_strength = (max_val - min_val) * 0.2
            new_val = current_val + random.gauss(0, mutation_strength)
            new_val = max(min_val, min(max_val, new_val))

            # Convert to int if needed
            if param in ["max_hold_hours", "max_daily_trades", "cooldown_minutes"]:
                new_val = int(new_val)

            setattr(mutated, param, new_val)

        # Possibly mutate regime filter
        if random.random() < 0.2:
            if random.random() < 0.5 and len(mutated.entry_regime_filter) > 2:
                # Remove a regime
                mutated.entry_regime_filter.remove(random.choice(mutated.entry_regime_filter))
            else:
                # Add a regime
                available = [r for r in self.REGIME_OPTIONS if r not in mutated.entry_regime_filter]
                if available:
                    mutated.entry_regime_filter.append(random.choice(available))

        # Possibly flip short_enabled
        if random.random() < 0.1:
            mutated.short_enabled = not mutated.short_enabled

        return mutated

    def _update_hall_of_fame(self, max_size: int = 10):
        """Update hall of fame with best individuals."""
        # Add top performers
        for gene in self._population[:3]:
            if gene.fitness > 0.5:  # Minimum quality threshold
                # Check if similar already exists
                similar = any(
                    self._calculate_similarity(gene, hof) > 0.9
                    for hof in self._hall_of_fame
                )
                if not similar:
                    self._hall_of_fame.append(copy.deepcopy(gene))

        # Sort and trim
        self._hall_of_fame.sort(key=lambda g: g.fitness, reverse=True)
        self._hall_of_fame = self._hall_of_fame[:max_size]

    def _calculate_similarity(self, gene1: StrategyGene, gene2: StrategyGene) -> float:
        """Calculate similarity between two genes (0-1)."""
        similarities = []

        for param in self.PARAM_RANGES.keys():
            val1 = getattr(gene1, param)
            val2 = getattr(gene2, param)
            min_val, max_val = self.PARAM_RANGES[param]

            if max_val > min_val:
                sim = 1 - abs(val1 - val2) / (max_val - min_val)
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self._best_fitness_history) < 10:
            return False

        recent = self._best_fitness_history[-10:]
        improvement = (recent[-1] - recent[0]) / max(0.001, recent[0])

        return improvement < self.config.convergence_threshold

    def run_evolution(
        self,
        backtest_function: Callable[[StrategyGene], Dict],
        max_generations: Optional[int] = None,
    ) -> StrategyGene:
        """
        Run full evolution process.

        Args:
            backtest_function: Function to evaluate strategies
            max_generations: Override max generations

        Returns:
            Best strategy found
        """
        max_gen = max_generations or self.config.max_generations

        if not self._population:
            self.initialize_population()

        for gen in range(max_gen):
            # Evaluate
            self.evaluate_population(backtest_function)

            # Check convergence
            if self.check_convergence():
                logger.info(f"Converged at generation {self._generation}")
                break

            # Evolve
            if gen < max_gen - 1:
                self.evolve_generation()

        return self._best_ever

    def get_best_strategy(self) -> Optional[StrategyGene]:
        """Get the best strategy ever found."""
        return self._best_ever

    def get_top_strategies(self, n: int = 5) -> List[StrategyGene]:
        """Get top N strategies from current population."""
        return self._population[:n]

    def get_hall_of_fame(self) -> List[StrategyGene]:
        """Get hall of fame strategies."""
        return self._hall_of_fame

    def inject_strategy(self, gene: StrategyGene):
        """Inject a known good strategy into the population."""
        gene.generation = self._generation
        self._population.append(gene)
        logger.info(f"Injected strategy: {gene.name}")

    def get_evolution_stats(self) -> Dict:
        """Get evolution statistics."""
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": self._best_fitness_history[-1] if self._best_fitness_history else 0,
            "avg_fitness": self._avg_fitness_history[-1] if self._avg_fitness_history else 0,
            "fitness_history": self._best_fitness_history[-20:],
            "hall_of_fame_size": len(self._hall_of_fame),
            "best_ever": {
                "name": self._best_ever.name,
                "fitness": self._best_ever.fitness,
                "sharpe": self._best_ever.sharpe_ratio,
                "win_rate": self._best_ever.win_rate,
            } if self._best_ever else None,
        }

    def export_strategy(self, gene: StrategyGene) -> Dict:
        """Export strategy as a configuration dictionary."""
        return {
            "name": gene.name,
            "entry": {
                "rsi_threshold": gene.entry_rsi_threshold,
                "momentum_threshold": gene.entry_momentum_threshold,
                "volume_threshold": gene.entry_volume_threshold,
                "regime_filter": gene.entry_regime_filter,
            },
            "exit": {
                "take_profit_pct": gene.take_profit_pct,
                "stop_loss_pct": gene.stop_loss_pct,
                "trailing_stop_pct": gene.trailing_stop_pct,
                "max_hold_hours": gene.max_hold_hours,
            },
            "position": {
                "base_position_pct": gene.base_position_pct,
                "leverage_multiplier": gene.leverage_multiplier,
            },
            "risk": {
                "max_daily_trades": gene.max_daily_trades,
                "min_confidence": gene.min_confidence,
                "cooldown_minutes": gene.cooldown_minutes,
            },
            "shorting": {
                "enabled": gene.short_enabled,
                "rsi_threshold": gene.short_rsi_threshold,
            },
            "metrics": {
                "fitness": gene.fitness,
                "sharpe_ratio": gene.sharpe_ratio,
                "win_rate": gene.win_rate,
                "max_drawdown": gene.max_drawdown,
                "total_pnl": gene.total_pnl,
            },
        }


# Singleton
_strategy_evolver: Optional[StrategyEvolver] = None


def get_strategy_evolver() -> StrategyEvolver:
    """Get or create the StrategyEvolver singleton."""
    global _strategy_evolver
    if _strategy_evolver is None:
        _strategy_evolver = StrategyEvolver()
    return _strategy_evolver
