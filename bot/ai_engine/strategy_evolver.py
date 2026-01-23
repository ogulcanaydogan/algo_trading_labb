"""
Strategy Evolver - Genetic Algorithm for Strategy Discovery

Evolves trading strategies through:
1. Gene encoding of strategy rules
2. Fitness evaluation via backtesting
3. Selection of top performers
4. Crossover and mutation to create new strategies
5. Speciation for diverse strategy population

Can discover entirely new strategies without human input.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

import numpy as np

from .learning_db import LearningDatabase, EvolvedStrategy, get_learning_db

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Available indicator types for strategy genes."""

    EMA_CROSS = "ema_cross"
    RSI = "rsi"
    MACD = "macd"
    ADX = "adx"
    BOLLINGER = "bollinger"
    ATR_BREAK = "atr_break"
    VOLUME = "volume"
    PRICE_ACTION = "price_action"
    MOMENTUM = "momentum"


class ConditionOperator(Enum):
    """Operators for conditions."""

    GREATER = ">"
    LESS = "<"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    BETWEEN = "between"
    OUTSIDE = "outside"


@dataclass
class StrategyGene:
    """
    A single gene representing a trading rule component.

    Example genes:
    - RSI < 30 (buy signal)
    - EMA_12 cross_above EMA_26 (buy signal)
    - ADX > 25 (trend confirmation)
    """

    indicator: str
    operator: str
    value: float
    value2: Optional[float] = None  # For BETWEEN operator
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Importance weight
    signal_type: str = "entry"  # entry, exit, filter

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyGene":
        return cls(**data)


@dataclass
class StrategyChromosome:
    """
    A complete strategy encoded as a chromosome of genes.

    Contains:
    - Entry rules (when to buy/sell)
    - Exit rules (when to close)
    - Filter rules (confirm signals)
    - Risk parameters
    """

    id: str
    name: str
    entry_long_genes: List[StrategyGene]
    entry_short_genes: List[StrategyGene]
    exit_genes: List[StrategyGene]
    filter_genes: List[StrategyGene]
    risk_params: Dict[str, float]
    regime_preference: Optional[str] = None
    generation: int = 0
    fitness: float = 0.0
    parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entry_long_genes": [g.to_dict() for g in self.entry_long_genes],
            "entry_short_genes": [g.to_dict() for g in self.entry_short_genes],
            "exit_genes": [g.to_dict() for g in self.exit_genes],
            "filter_genes": [g.to_dict() for g in self.filter_genes],
            "risk_params": self.risk_params,
            "regime_preference": self.regime_preference,
            "generation": self.generation,
            "fitness": self.fitness,
            "parent_ids": self.parent_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyChromosome":
        return cls(
            id=data["id"],
            name=data["name"],
            entry_long_genes=[StrategyGene.from_dict(g) for g in data["entry_long_genes"]],
            entry_short_genes=[StrategyGene.from_dict(g) for g in data["entry_short_genes"]],
            exit_genes=[StrategyGene.from_dict(g) for g in data["exit_genes"]],
            filter_genes=[StrategyGene.from_dict(g) for g in data["filter_genes"]],
            risk_params=data["risk_params"],
            regime_preference=data.get("regime_preference"),
            generation=data.get("generation", 0),
            fitness=data.get("fitness", 0.0),
            parent_ids=data.get("parent_ids", []),
        )


class GeneFactory:
    """Factory for creating random genes."""

    INDICATOR_TEMPLATES = {
        "rsi": {
            "params": {"period": (7, 21)},
            "value_range": (20, 80),
            "operators": ["<", ">"],
        },
        "ema_fast": {
            "params": {"period": (5, 20)},
            "value_range": None,  # Compare to price or another EMA
            "operators": ["cross_above", "cross_below"],
        },
        "ema_slow": {
            "params": {"period": (15, 50)},
            "value_range": None,
            "operators": ["cross_above", "cross_below"],
        },
        "adx": {
            "params": {"period": (10, 20)},
            "value_range": (15, 35),
            "operators": [">", "<"],
        },
        "macd_hist": {
            "params": {"fast": (8, 15), "slow": (20, 30), "signal": (7, 12)},
            "value_range": (-0.5, 0.5),
            "operators": [">", "<", "cross_above", "cross_below"],
        },
        "bb_position": {
            "params": {"period": (15, 25), "std": (1.5, 2.5)},
            "value_range": (0, 1),  # 0 = lower band, 1 = upper band
            "operators": ["<", ">"],
        },
        "volume_ratio": {
            "params": {"period": (10, 30)},
            "value_range": (0.5, 2.0),
            "operators": [">", "<"],
        },
        "momentum": {
            "params": {"period": (5, 20)},
            "value_range": (-5, 5),  # Percent change
            "operators": [">", "<"],
        },
        "atr_distance": {
            "params": {"period": (10, 20)},
            "value_range": (0.5, 3.0),  # ATR multiples from MA
            "operators": [">", "<"],
        },
    }

    @classmethod
    def create_random_gene(cls, signal_type: str = "entry") -> StrategyGene:
        """Create a random gene."""
        indicator = random.choice(list(cls.INDICATOR_TEMPLATES.keys()))
        template = cls.INDICATOR_TEMPLATES[indicator]

        # Generate random params
        params = {}
        for param_name, (min_val, max_val) in template["params"].items():
            if isinstance(min_val, int):
                params[param_name] = random.randint(min_val, max_val)
            else:
                params[param_name] = round(random.uniform(min_val, max_val), 2)

        # Generate value
        value = 0.0
        if template["value_range"]:
            min_v, max_v = template["value_range"]
            value = round(random.uniform(min_v, max_v), 2)

        # Select operator
        operator = random.choice(template["operators"])

        return StrategyGene(
            indicator=indicator,
            operator=operator,
            value=value,
            params=params,
            weight=round(random.uniform(0.5, 1.5), 2),
            signal_type=signal_type,
        )

    @classmethod
    def create_random_risk_params(cls) -> Dict[str, float]:
        """Create random risk parameters."""
        return {
            "position_size_pct": round(random.uniform(5, 25), 1),
            "stop_loss_atr_mult": round(random.uniform(1.0, 3.0), 2),
            "take_profit_atr_mult": round(random.uniform(1.5, 4.0), 2),
            "trailing_stop_pct": round(random.uniform(0.5, 2.0), 2),
            "max_hold_bars": random.randint(10, 100),
        }


class StrategyEvolver:
    """
    Genetic algorithm for evolving trading strategies.

    Process:
    1. Initialize population of random strategies
    2. Evaluate fitness via backtesting
    3. Select top performers (elitism)
    4. Create offspring through crossover
    5. Apply mutations
    6. Repeat for N generations
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_size: int = 5,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.7,
        db: LearningDatabase = None,
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.db = db or get_learning_db()

        self.population: List[StrategyChromosome] = []
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self._strategy_counter = 0

    def _generate_strategy_id(self) -> str:
        """Generate unique strategy ID."""
        self._strategy_counter += 1
        return f"evolved_{self.generation}_{self._strategy_counter}"

    def _create_random_strategy(self) -> StrategyChromosome:
        """Create a completely random strategy."""
        # Random number of genes per section
        n_entry_long = random.randint(1, 4)
        n_entry_short = random.randint(1, 4)
        n_exit = random.randint(1, 3)
        n_filter = random.randint(0, 3)

        return StrategyChromosome(
            id=self._generate_strategy_id(),
            name=f"Strategy_Gen{self.generation}_{self._strategy_counter}",
            entry_long_genes=[GeneFactory.create_random_gene("entry") for _ in range(n_entry_long)],
            entry_short_genes=[
                GeneFactory.create_random_gene("entry") for _ in range(n_entry_short)
            ],
            exit_genes=[GeneFactory.create_random_gene("exit") for _ in range(n_exit)],
            filter_genes=[GeneFactory.create_random_gene("filter") for _ in range(n_filter)],
            risk_params=GeneFactory.create_random_risk_params(),
            generation=self.generation,
        )

    def initialize_population(self, seed_strategies: List[StrategyChromosome] = None):
        """Initialize population with random strategies and optional seeds."""
        self.population = []
        self.generation = 0

        # Add seed strategies if provided
        if seed_strategies:
            for strategy in seed_strategies[: self.elite_size]:
                strategy.generation = 0
                self.population.append(strategy)

        # Fill rest with random strategies
        while len(self.population) < self.population_size:
            self.population.append(self._create_random_strategy())

        logger.info(f"Initialized population with {len(self.population)} strategies")

    def evaluate_population(
        self,
        fitness_fn: Callable[[StrategyChromosome], float],
        parallel: bool = False,
    ):
        """Evaluate fitness of all strategies in population."""
        for strategy in self.population:
            try:
                strategy.fitness = fitness_fn(strategy)
            except Exception as e:
                logger.warning(f"Fitness evaluation failed for {strategy.id}: {e}")
                strategy.fitness = float("-inf")

        # Sort by fitness
        self.population.sort(key=lambda s: s.fitness, reverse=True)

        # Track best fitness
        if self.population:
            self.best_fitness_history.append(self.population[0].fitness)

    def _select_parent(self) -> StrategyChromosome:
        """Tournament selection for parent."""
        tournament_size = 3
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda s: s.fitness)

    def _crossover(
        self,
        parent1: StrategyChromosome,
        parent2: StrategyChromosome,
    ) -> StrategyChromosome:
        """Create offspring through crossover of two parents."""
        child_id = self._generate_strategy_id()

        # Crossover each gene section
        def crossover_genes(
            genes1: List[StrategyGene], genes2: List[StrategyGene]
        ) -> List[StrategyGene]:
            if not genes1 and not genes2:
                return []

            # Either take all from one parent or mix
            if random.random() < 0.3:
                return copy.deepcopy(genes1 if random.random() < 0.5 else genes2)

            # Mix genes
            result = []
            all_genes = genes1 + genes2
            n_genes = random.randint(1, max(1, len(all_genes) // 2 + 1))
            for gene in random.sample(all_genes, min(n_genes, len(all_genes))):
                result.append(copy.deepcopy(gene))
            return result

        # Crossover risk params
        risk_params = {}
        for key in parent1.risk_params:
            if random.random() < 0.5:
                risk_params[key] = parent1.risk_params[key]
            else:
                risk_params[key] = parent2.risk_params.get(key, parent1.risk_params[key])

        return StrategyChromosome(
            id=child_id,
            name=f"Strategy_Gen{self.generation + 1}_{self._strategy_counter}",
            entry_long_genes=crossover_genes(parent1.entry_long_genes, parent2.entry_long_genes),
            entry_short_genes=crossover_genes(parent1.entry_short_genes, parent2.entry_short_genes),
            exit_genes=crossover_genes(parent1.exit_genes, parent2.exit_genes),
            filter_genes=crossover_genes(parent1.filter_genes, parent2.filter_genes),
            risk_params=risk_params,
            regime_preference=random.choice([parent1.regime_preference, parent2.regime_preference]),
            generation=self.generation + 1,
            parent_ids=[parent1.id, parent2.id],
        )

    def _mutate(self, strategy: StrategyChromosome) -> StrategyChromosome:
        """Apply mutations to a strategy."""
        mutated = copy.deepcopy(strategy)

        def mutate_gene(gene: StrategyGene) -> StrategyGene:
            if random.random() < self.mutation_rate:
                # Mutate value
                if gene.value != 0:
                    gene.value *= random.uniform(0.8, 1.2)
                    gene.value = round(gene.value, 2)

                # Mutate params
                for key in gene.params:
                    if random.random() < 0.3:
                        if isinstance(gene.params[key], int):
                            gene.params[key] += random.randint(-2, 2)
                            gene.params[key] = max(1, gene.params[key])
                        else:
                            gene.params[key] *= random.uniform(0.9, 1.1)
                            gene.params[key] = round(gene.params[key], 2)

                # Mutate weight
                if random.random() < 0.2:
                    gene.weight = round(random.uniform(0.5, 1.5), 2)

            return gene

        def mutate_gene_list(genes: List[StrategyGene]) -> List[StrategyGene]:
            # Mutate existing genes
            genes = [mutate_gene(g) for g in genes]

            # Possibly add a new gene
            if random.random() < self.mutation_rate * 0.5:
                genes.append(GeneFactory.create_random_gene())

            # Possibly remove a gene
            if len(genes) > 1 and random.random() < self.mutation_rate * 0.3:
                genes.pop(random.randint(0, len(genes) - 1))

            return genes

        mutated.entry_long_genes = mutate_gene_list(mutated.entry_long_genes)
        mutated.entry_short_genes = mutate_gene_list(mutated.entry_short_genes)
        mutated.exit_genes = mutate_gene_list(mutated.exit_genes)
        mutated.filter_genes = mutate_gene_list(mutated.filter_genes)

        # Mutate risk params
        for key in mutated.risk_params:
            if random.random() < self.mutation_rate:
                mutated.risk_params[key] *= random.uniform(0.8, 1.2)
                mutated.risk_params[key] = round(mutated.risk_params[key], 2)

        mutated.id = self._generate_strategy_id()
        return mutated

    def evolve_generation(
        self,
        fitness_fn: Callable[[StrategyChromosome], float],
    ):
        """Evolve one generation."""
        self.generation += 1
        logger.info(f"Evolving generation {self.generation}")

        # Keep elite (top performers)
        new_population = []
        for i in range(min(self.elite_size, len(self.population))):
            elite = copy.deepcopy(self.population[i])
            elite.generation = self.generation
            new_population.append(elite)

        # Create offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self.population) >= 2:
                # Crossover
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
            else:
                # Clone and mutate
                parent = self._select_parent()
                child = copy.deepcopy(parent)
                child.id = self._generate_strategy_id()
                child.generation = self.generation

            # Apply mutation
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population

        # Evaluate new generation
        self.evaluate_population(fitness_fn)

        # Save top strategies to database
        for strategy in self.population[:3]:
            try:
                evolved = EvolvedStrategy(
                    id=None,
                    generation=self.generation,
                    fitness=strategy.fitness,
                    genes=strategy.to_dict(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    regime=strategy.regime_preference or "all",
                    is_active=True,
                    performance_live=None,
                )
                self.db.save_evolved_strategy(evolved)
            except Exception as e:
                logger.error(f"Failed to save evolved strategy: {e}")

        return self.population[0] if self.population else None

    async def run_evolution(
        self,
        n_generations: int,
        fitness_fn: Callable[[StrategyChromosome], float],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        seed_strategies: List[StrategyChromosome] = None,
    ) -> StrategyChromosome:
        """
        Run full evolution process.

        Args:
            n_generations: Number of generations to evolve
            fitness_fn: Function that evaluates strategy fitness
            progress_callback: Optional callback(generation, total, best_fitness)
            seed_strategies: Optional list of starting strategies

        Returns:
            Best evolved strategy
        """
        self.initialize_population(seed_strategies)
        self.evaluate_population(fitness_fn)

        if progress_callback:
            best = self.population[0].fitness if self.population else 0
            progress_callback(0, n_generations, best)

        for gen in range(n_generations):
            best_strategy = self.evolve_generation(fitness_fn)

            if progress_callback:
                best = best_strategy.fitness if best_strategy else 0
                progress_callback(gen + 1, n_generations, best)

            logger.info(
                f"Generation {self.generation}: "
                f"Best fitness = {best_strategy.fitness:.4f if best_strategy else 0}"
            )

            # Early stopping if fitness is very high
            if best_strategy and best_strategy.fitness > 3.0:  # Sharpe > 3
                logger.info("Early stopping: excellent strategy found")
                break

        return self.population[0] if self.population else None

    def get_best_strategies(self, n: int = 5) -> List[StrategyChromosome]:
        """Get top N strategies from current population."""
        return self.population[:n]

    def get_diverse_strategies(self, n: int = 5) -> List[StrategyChromosome]:
        """
        Get diverse strategies (different approaches).

        Ensures variety by selecting strategies with different
        primary indicators.
        """
        if not self.population:
            return []

        # Group by primary entry indicator
        groups: Dict[str, List[StrategyChromosome]] = {}
        for strategy in self.population:
            if strategy.entry_long_genes:
                primary = strategy.entry_long_genes[0].indicator
                if primary not in groups:
                    groups[primary] = []
                groups[primary].append(strategy)

        # Select best from each group
        diverse = []
        for group in groups.values():
            if group:
                best_in_group = max(group, key=lambda s: s.fitness)
                diverse.append(best_in_group)

        # Sort by fitness and return top N
        diverse.sort(key=lambda s: s.fitness, reverse=True)
        return diverse[:n]


# Global instance
_evolver: Optional[StrategyEvolver] = None


def get_strategy_evolver() -> StrategyEvolver:
    """Get or create global strategy evolver."""
    global _evolver
    if _evolver is None:
        _evolver = StrategyEvolver()
    return _evolver
