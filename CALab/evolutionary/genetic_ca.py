"""
Evolutionary Cellular Automata Framework
========================================

Evolve CA rules using genetic algorithms to discover novel behaviors,
self-replicating patterns, and complex emergent dynamics.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from functools import partial
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary CA."""
    population_size: int = 100
    n_generations: int = 1000
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    tournament_size: int = 5
    elitism_count: int = 5
    grid_size: Tuple[int, int] = (64, 64)
    n_steps: int = 100
    n_trials: int = 3  # Number of random initial conditions to test
    seed: int = 42


class CARuleGenome(ABC):
    """Abstract base class for CA rule genomes."""

    @abstractmethod
    def mutate(self, mutation_rate: float) -> 'CARuleGenome':
        """Mutate the genome."""
        pass

    @abstractmethod
    def crossover(self, other: 'CARuleGenome') -> Tuple['CARuleGenome', 'CARuleGenome']:
        """Crossover with another genome."""
        pass

    @abstractmethod
    def to_rule_function(self) -> Callable:
        """Convert genome to CA rule function."""
        pass

    @abstractmethod
    def copy(self) -> 'CARuleGenome':
        """Create a copy of the genome."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation."""
        pass


class ElementaryCAGenome(CARuleGenome):
    """Genome for elementary cellular automata (Wolfram rules)."""

    def __init__(self, rule_number: Optional[int] = None):
        """Initialize with random rule or specific rule number."""
        if rule_number is None:
            self.rule_number = random.randint(0, 255)
        else:
            self.rule_number = rule_number

        # Convert rule number to binary rule table
        self.rule_table = [(self.rule_number >> i) & 1 for i in range(8)]

    def mutate(self, mutation_rate: float) -> 'ElementaryCAGenome':
        """Mutate by flipping bits in rule table."""
        new_genome = self.copy()

        for i in range(8):
            if random.random() < mutation_rate:
                new_genome.rule_table[i] = 1 - new_genome.rule_table[i]

        # Update rule number
        new_genome.rule_number = sum(bit << i for i, bit in enumerate(new_genome.rule_table))
        return new_genome

    def crossover(self, other: CARuleGenome) -> Tuple[CARuleGenome, CARuleGenome]:
        """Single-point crossover."""
        if not isinstance(other, ElementaryCAGenome):
            # Fallback to copying if types don't match
            return self.copy(), other.copy()

        # Choose crossover point
        point = random.randint(1, 7)

        # Create offspring
        child1_table = self.rule_table[:point] + other.rule_table[point:]
        child2_table = other.rule_table[:point] + self.rule_table[point:]

        child1 = ElementaryCAGenome()
        child1.rule_table = child1_table
        child1.rule_number = sum(bit << i for i, bit in enumerate(child1_table))

        child2 = ElementaryCAGenome()
        child2.rule_table = child2_table
        child2.rule_number = sum(bit << i for i, bit in enumerate(child2_table))

        return child1, child2

    def to_rule_function(self) -> Callable:
        """Convert to CA rule function."""
        rule_table = self.rule_table.copy()

        def rule(current: int, neighbors: np.ndarray) -> int:
            # For elementary CA, use left and right neighbors
            left = int(neighbors[0]) if len(neighbors) > 0 else 0
            right = int(neighbors[2]) if len(neighbors) > 2 else 0

            # Convert to 3-bit index
            index = (left << 2) | (current << 1) | right
            return rule_table[index]

        return rule

    def copy(self) -> 'ElementaryCAGenome':
        """Create a copy."""
        new_genome = ElementaryCAGenome(self.rule_number)
        return new_genome

    def __str__(self) -> str:
        return f"ElementaryCA(rule={self.rule_number})"


class TotalisticCAGenome(CARuleGenome):
    """Genome for totalistic cellular automata."""

    def __init__(self, n_states: int = 2, radius: int = 1):
        """Initialize totalistic CA genome."""
        self.n_states = n_states
        self.radius = radius

        # Rule table: for each possible sum of neighbors, what state to take
        max_sum = n_states * (2 * radius + 1) ** 2  # Include center cell
        self.rule_table = [random.randint(0, n_states - 1) for _ in range(max_sum + 1)]

    def mutate(self, mutation_rate: float) -> 'TotalisticCAGenome':
        """Mutate rule table entries."""
        new_genome = self.copy()

        for i in range(len(new_genome.rule_table)):
            if random.random() < mutation_rate:
                new_genome.rule_table[i] = random.randint(0, self.n_states - 1)

        return new_genome

    def crossover(self, other: 'TotalisticCAGenome') -> Tuple['TotalisticCAGenome', 'TotalisticCAGenome']:
        """Uniform crossover."""
        child1 = self.copy()
        child2 = other.copy()

        for i in range(len(child1.rule_table)):
            if random.random() < 0.5:
                child1.rule_table[i], child2.rule_table[i] = child2.rule_table[i], child1.rule_table[i]

        return child1, child2

    def to_rule_function(self) -> Callable:
        """Convert to CA rule function."""
        rule_table = self.rule_table.copy()
        n_states = self.n_states

        def rule(current: int, neighbors: np.ndarray) -> int:
            # Sum all neighbors and center
            total = current + int(np.sum(neighbors))
            return rule_table[min(total, len(rule_table) - 1)]

        return rule

    def copy(self) -> 'TotalisticCAGenome':
        """Create a copy."""
        new_genome = TotalisticCAGenome(self.n_states, self.radius)
        new_genome.rule_table = self.rule_table.copy()
        return new_genome

    def __str__(self) -> str:
        return f"TotalisticCA(states={self.n_states}, radius={self.radius})"


class NeuralCAGenome(CARuleGenome):
    """Genome for neural CA (parameter vector)."""

    def __init__(self, n_params: int = 1000, params: Optional[np.ndarray] = None):
        """Initialize neural CA genome."""
        self.n_params = n_params

        if params is None:
            self.params = np.random.normal(0, 0.1, n_params)
        else:
            self.params = params.copy()

    def mutate(self, mutation_rate: float) -> 'NeuralCAGenome':
        """Gaussian mutation of parameters."""
        new_genome = self.copy()

        # Mutate each parameter
        for i in range(len(new_genome.params)):
            if random.random() < mutation_rate:
                # Gaussian perturbation
                new_genome.params[i] += np.random.normal(0, 0.1)

        return new_genome

    def crossover(self, other: 'NeuralCAGenome') -> Tuple['NeuralCAGenome', 'NeuralCAGenome']:
        """Blend crossover."""
        child1 = self.copy()
        child2 = other.copy()

        # Blend parameters
        alpha = np.random.uniform(0, 1, len(child1.params))
        child1.params = alpha * self.params + (1 - alpha) * other.params
        child2.params = (1 - alpha) * self.params + alpha * other.params

        return child1, child2

    def to_rule_function(self) -> Callable:
        """Convert to neural CA rule function."""
        # This would need integration with the neural CA framework
        # For now, return a placeholder
        params = self.params.copy()

        def rule(current: int, neighbors: np.ndarray) -> int:
            # Simplified neural computation
            # In practice, this would use the full neural CA
            input_sum = current + np.sum(neighbors)
            activation = np.tanh(np.dot(params[:10], np.array([input_sum, current, np.mean(neighbors)])))
            return 1 if activation > 0 else 0

        return rule

    def copy(self) -> 'NeuralCAGenome':
        """Create a copy."""
        return NeuralCAGenome(self.n_params, self.params)

    def __str__(self) -> str:
        return f"NeuralCA(params={self.n_params})"


class CAFIndividual:
    """Individual in the evolutionary CA population."""

    def __init__(self, genome: CARuleGenome, config: EvolutionaryConfig):
        self.genome = genome
        self.config = config
        self.fitness = 0.0
        self.fitness_components = {}
        self.metadata = {}

    def evaluate(self, fitness_functions: List[Callable]) -> float:
        """Evaluate fitness using multiple fitness functions."""
        total_fitness = 0.0
        self.fitness_components = {}

        for fitness_fn in fitness_functions:
            fitness, components = fitness_fn(self.genome, self.config)
            total_fitness += fitness
            self.fitness_components.update(components)

        self.fitness = total_fitness
        return self.fitness

    def __str__(self) -> str:
        return f"Individual(fitness={self.fitness:.3f}, genome={self.genome})"


class FitnessFunctions:
    """Collection of fitness functions for evolutionary CA."""

    @staticmethod
    def pattern_similarity_target(target_pattern: np.ndarray) -> Callable:
        """Fitness for matching a target pattern."""
        def fitness(genome: CARuleGenome, config: EvolutionaryConfig) -> Tuple[float, Dict[str, float]]:
            rule_fn = genome.to_rule_function()

            similarities = []
            for trial in range(config.n_trials):
                # Create random initial condition
                initial = np.random.choice([0, 1], size=config.grid_size, p=[0.7, 0.3])

                # Evolve CA
                grid = initial.copy()
                for step in range(config.n_steps):
                    new_grid = np.zeros_like(grid)
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            neighbors = []
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                        neighbors.append(grid[ni, nj])
                                    else:
                                        neighbors.append(0)  # Boundary condition

                            new_grid[i, j] = rule_fn(grid[i, j], np.array(neighbors))
                    grid = new_grid

                # Compare to target pattern
                if grid.shape == target_pattern.shape:
                    similarity = np.mean(grid == target_pattern)
                else:
                    # Resize if needed
                    similarity = 0.0

                similarities.append(similarity)

            avg_similarity = float(np.mean(similarities))
            return avg_similarity, {'pattern_similarity': avg_similarity}

        return fitness

    @staticmethod
    def complexity_fitness() -> Callable:
        """Fitness encouraging complex behavior."""
        def fitness(genome: CARuleGenome, config: EvolutionaryConfig) -> Tuple[float, Dict[str, float]]:
            rule_fn = genome.to_rule_function()

            complexities = []
            spatial_entropies = []
            temporal_diversities = []

            for trial in range(config.n_trials):
                # Create random initial condition
                initial = np.random.choice([0, 1], size=config.grid_size, p=[0.5, 0.5])

                # Evolve CA and collect states
                states = []
                grid = initial.copy()
                states.append(grid.copy())

                for step in range(config.n_steps):
                    new_grid = np.zeros_like(grid)
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            neighbors = []
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                        neighbors.append(grid[ni, nj])
                                    else:
                                        neighbors.append(0)

                            new_grid[i, j] = rule_fn(grid[i, j], np.array(neighbors))
                    grid = new_grid
                    states.append(grid.copy())

                # Calculate complexity metrics
                states_array = np.array(states)

                # Spatial entropy
                spatial_entropy = 0.0
                for state in states_array:
                    probs = np.bincount(state.flatten(), minlength=2) / state.size
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    spatial_entropy += entropy
                spatial_entropy /= len(states_array)

                # Temporal diversity (how much states change)
                temporal_diversity = 0.0
                for i in range(1, len(states_array)):
                    diff = np.mean(np.abs(states_array[i] - states_array[i-1]))
                    temporal_diversity += diff
                temporal_diversity /= len(states_array) - 1

                # Combined complexity score
                complexity = spatial_entropy * temporal_diversity
                complexities.append(complexity)
                spatial_entropies.append(spatial_entropy)
                temporal_diversities.append(temporal_diversity)

            avg_complexity = float(np.mean(complexities))
            avg_spatial_entropy = float(np.mean(spatial_entropies))
            avg_temporal_diversity = float(np.mean(temporal_diversities))

            return avg_complexity, {
                'spatial_entropy': avg_spatial_entropy,
                'temporal_diversity': avg_temporal_diversity,
                'complexity': avg_complexity
            }

        return fitness

    @staticmethod
    def self_replication_fitness() -> Callable:
        """Fitness for self-replicating patterns."""
        def fitness(genome: CARuleGenome, config: EvolutionaryConfig) -> Tuple[float, Dict[str, float]]:
            rule_fn = genome.to_rule_function()

            replication_scores = []
            for trial in range(config.n_trials):
                # Start with a small seed pattern
                grid = np.zeros(config.grid_size)
                center_i, center_j = config.grid_size[0] // 2, config.grid_size[1] // 2

                # Create a small initial pattern
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if abs(di) + abs(dj) <= 1:  # Cross pattern
                            i, j = center_i + di, center_j + dj
                            if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                                grid[i, j] = 1

                initial_mass = np.sum(grid)

                # Evolve CA
                for step in range(config.n_steps):
                    new_grid = np.zeros_like(grid)
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            neighbors = []
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                        neighbors.append(grid[ni, nj])
                                    else:
                                        neighbors.append(0)

                            new_grid[i, j] = rule_fn(grid[i, j], np.array(neighbors))
                    grid = new_grid

                final_mass = np.sum(grid)

                # Replication score: how much the pattern grew
                if initial_mass > 0:
                    growth_ratio = final_mass / initial_mass
                    # Penalize excessive growth, reward moderate growth
                    if growth_ratio > 10:  # Too much growth
                        score = 0.1
                    elif growth_ratio > 2:  # Good replication
                        score = growth_ratio * 0.5
                    else:  # Poor replication
                        score = growth_ratio * 0.1
                else:
                    score = 0.0

                replication_scores.append(score)

            avg_score = float(np.mean(replication_scores))
            return avg_score, {'replication_score': avg_score}

        return fitness

    @staticmethod
    def stability_fitness() -> Callable:
        """Fitness for stable, persistent patterns."""
        def fitness(genome: CARuleGenome, config: EvolutionaryConfig) -> Tuple[float, Dict[str, float]]:
            rule_fn = genome.to_rule_function()

            stability_scores = []
            for trial in range(config.n_trials):
                # Random initial condition
                initial = np.random.choice([0, 1], size=config.grid_size, p=[0.5, 0.5])

                # Evolve CA
                grid = initial.copy()
                states = [grid.copy()]

                for step in range(config.n_steps):
                    new_grid = np.zeros_like(grid)
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            neighbors = []
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                        neighbors.append(grid[ni, nj])
                                    else:
                                        neighbors.append(0)

                            new_grid[i, j] = rule_fn(grid[i, j], np.array(neighbors))
                    grid = new_grid
                    states.append(grid.copy())

                # Calculate stability (how similar final states are to each other)
                final_states = states[-10:]  # Last 10 states
                stability = 0
                count = 0
                for i in range(len(final_states)):
                    for j in range(i+1, len(final_states)):
                        similarity = np.mean(final_states[i] == final_states[j])
                        stability += similarity
                        count += 1

                if count > 0:
                    stability /= count
                else:
                    stability = 0.0

                stability_scores.append(stability)

            avg_stability = float(np.mean(stability_scores))
            return avg_stability, {'stability': avg_stability}

        return fitness


class EvolutionaryCA:
    """Main evolutionary CA system."""

    def __init__(self, config: EvolutionaryConfig, genome_type: str = 'elementary'):
        self.config = config
        self.genome_type = genome_type

        # Initialize population
        self.population = self._initialize_population()

        # Statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation = 0

    def _initialize_population(self) -> List[CAFIndividual]:
        """Initialize random population."""
        population = []

        for _ in range(self.config.population_size):
            if self.genome_type == 'elementary':
                genome = ElementaryCAGenome()
            elif self.genome_type == 'totalistic':
                genome = TotalisticCAGenome()
            elif self.genome_type == 'neural':
                genome = NeuralCAGenome()
            else:
                raise ValueError(f"Unknown genome type: {self.genome_type}")

            individual = CAFIndividual(genome, self.config)
            population.append(individual)

        return population

    def evolve(self, fitness_functions: List[Callable], n_generations: Optional[int] = None) -> CAFIndividual:
        """Evolve the population."""
        if n_generations is None:
            n_generations = self.config.n_generations

        for gen in range(n_generations):
            print(f"Generation {gen + 1}/{n_generations}")

            # Evaluate population
            self._evaluate_population(fitness_functions)

            # Record statistics
            fitnesses = [ind.fitness for ind in self.population]
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(np.mean(fitnesses))

            print(f"Best: {max(fitnesses):.3f}, Avg: {np.mean(fitnesses):.3f}")
            # Create new population
            new_population = []

            # Elitism
            elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.config.elitism_count]
            new_population.extend(elite)

            # Generate offspring
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                if random.random() < self.config.crossover_rate:
                    child1_genome, child2_genome = parent1.genome.crossover(parent2.genome)
                else:
                    child1_genome = parent1.genome.copy()
                    child2_genome = parent2.genome.copy()

                # Mutate
                child1_genome = child1_genome.mutate(self.config.mutation_rate)
                child2_genome = child2_genome.mutate(self.config.mutation_rate)

                child1 = CAFIndividual(child1_genome, self.config)
                child2 = CAFIndividual(child2_genome, self.config)

                new_population.extend([child1, child2])

            # Trim to exact population size
            self.population = new_population[:self.config.population_size]
            self.generation += 1

        # Return best individual
        best = max(self.population, key=lambda x: x.fitness)
        return best

    def _evaluate_population(self, fitness_functions: List[Callable]) -> None:
        """Evaluate entire population."""
        for individual in self.population:
            individual.evaluate(fitness_functions)

    def _tournament_selection(self) -> CAFIndividual:
        """Tournament selection."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0,
            'avg_fitness': self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }

    def save_population(self, filename: str) -> None:
        """Save population to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.population, f)

    def load_population(self, filename: str) -> None:
        """Load population from file."""
        with open(filename, 'rb') as f:
            self.population = pickle.load(f)


# Utility functions
def create_target_pattern(name: str, size: Tuple[int, int]) -> np.ndarray:
    """Create common target patterns."""
    pattern = np.zeros(size)

    if name == 'glider':
        # Conway's Game of Life glider
        pattern[1, 3] = 1
        pattern[2, 1] = 1
        pattern[2, 3] = 1
        pattern[3, 2] = 1
        pattern[3, 3] = 1

    elif name == 'block':
        # Still life block
        pattern[15:17, 15:17] = 1

    elif name == 'blinker':
        # Oscillator
        pattern[16, 14:17] = 1

    elif name == 'beacon':
        # Another oscillator
        pattern[14:16, 14:16] = 1
        pattern[16:18, 16:18] = 1

    return pattern


def run_evolution_experiment(genome_type: str = 'elementary',
                           fitness_type: str = 'complexity',
                           n_generations: int = 100) -> EvolutionaryCA:
    """Run a complete evolution experiment."""
    config = EvolutionaryConfig(
        population_size=50,
        n_generations=n_generations,
        grid_size=(32, 32),
        n_steps=50
    )

    # Create fitness functions
    fitness_functions = []

    if fitness_type == 'complexity':
        fitness_functions.append(FitnessFunctions.complexity_fitness())
    elif fitness_type == 'replication':
        fitness_functions.append(FitnessFunctions.self_replication_fitness())
    elif fitness_type == 'stability':
        fitness_functions.append(FitnessFunctions.stability_fitness())
    elif fitness_type == 'pattern':
        target = create_target_pattern('glider', config.grid_size)
        fitness_functions.append(FitnessFunctions.pattern_similarity_target(target))

    # Create evolutionary system
    evo_ca = EvolutionaryCA(config, genome_type)

    # Evolve
    best_individual = evo_ca.evolve(fitness_functions)

    print("Evolution completed!")
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness}")

    return evo_ca