#!/usr/bin/env python3
"""CA Rule Evolution: Synthesis of Cellular Automata and Evolutionary Algorithms.

This experiment demonstrates the power of algorithmic synthesis by using
evolutionary algorithms to discover cellular automata rules that produce
desired patterns or behaviors.

Key Innovation: Instead of hand-coding CA rules, we evolve them to solve
specific problems, potentially discovering novel computational patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Any, Callable, Optional, Tuple
import sys
import os
from dataclasses import dataclass
import json

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from EALab.algorithms.base import (
    EvolutionaryAlgorithm, Individual, Population, EvolutionaryParams,
    SelectionType
)
import numpy as np


class ElementaryCA:
    """Simple Elementary Cellular Automaton for rule evolution."""

    def __init__(self, width: int, rule_number: int):
        self.width = width
        self.rule_number = rule_number
        self.grid = np.zeros(width, dtype=int)

        # Convert rule number to binary rule table
        self.rule_table = [(rule_number >> i) & 1 for i in range(8)]

    def set_single_cell(self):
        """Initialize with single 1 in the center."""
        self.grid = np.zeros(self.width, dtype=int)
        center = self.width // 2
        self.grid[center] = 1

    def evolve_spacetime(self, generations: int) -> np.ndarray:
        """Evolve CA and return spacetime diagram."""
        spacetime = np.zeros((generations + 1, self.width), dtype=int)
        spacetime[0] = self.grid.copy()

        for t in range(generations):
            new_grid = np.zeros(self.width, dtype=int)
            for i in range(self.width):
                # Get neighborhood (with wrapping)
                left = self.grid[(i-1) % self.width]
                center = self.grid[i]
                right = self.grid[(i+1) % self.width]

                # Convert to rule index
                rule_index = 4*left + 2*center + right
                new_grid[i] = self.rule_table[rule_index]

            self.grid = new_grid
            spacetime[t+1] = self.grid.copy()

        return spacetime


@dataclass
class CATarget:
    """Defines a target pattern or behavior for CA evolution."""
    target_type: str  # "pattern", "behavior", "property"
    data: Any  # Target pattern array, behavior description, or property value
    weight: float = 1.0  # Importance weight in fitness
    
    def __post_init__(self):
        """Validate target specification."""
        valid_types = ["pattern", "behavior", "property", "glider", "oscillator"]
        if self.target_type not in valid_types:
            raise ValueError(f"Invalid target type: {self.target_type}")


class CAGenome:
    """Genome representation for CA rules."""
    
    def __init__(self, ca_type: str = "elementary"):
        """Initialize CA genome.
        
        Args:
            ca_type: Type of CA ("elementary", "life_like", "totalistic")
        """
        self.ca_type = ca_type
        
        if ca_type == "elementary":
            # Elementary CA: 8-bit rule (256 possible rules)
            self.genes = np.random.randint(0, 2, 8)  # Binary representation
        elif ca_type == "life_like":
            # Life-like: Birth and survival conditions
            self.birth_genes = np.random.randint(0, 2, 9)    # 0-8 neighbors
            self.survival_genes = np.random.randint(0, 2, 9)  # 0-8 neighbors
        elif ca_type == "totalistic":
            # General totalistic with multiple states
            self.rule_table = np.random.randint(0, 3, 27)  # 3^3 possible neighborhoods
        else:
            raise ValueError(f"Unknown CA type: {ca_type}")
    
    def to_rule_number(self) -> int:
        """Convert elementary genes to Wolfram rule number."""
        if self.ca_type != "elementary":
            raise ValueError("Rule number only applies to elementary CA")
        
        return sum(gene * (2 ** i) for i, gene in enumerate(self.genes))
    
    def to_life_like_string(self) -> str:
        """Convert life-like genes to B/S notation."""
        if self.ca_type != "life_like":
            raise ValueError("Life-like string only applies to life-like CA")
        
        birth = ''.join(str(i) for i, gene in enumerate(self.birth_genes) if gene)
        survival = ''.join(str(i) for i, gene in enumerate(self.survival_genes) if gene)
        
        return f"B{birth}/S{survival}"
    
    def mutate(self, mutation_rate: float = 0.1) -> 'CAGenome':
        """Create a mutated copy of the genome."""
        mutant = CAGenome(self.ca_type)
        
        if self.ca_type == "elementary":
            mutant.genes = self.genes.copy()
            # Flip bits with probability mutation_rate
            mask = np.random.random(8) < mutation_rate
            mutant.genes[mask] = 1 - mutant.genes[mask]
        
        elif self.ca_type == "life_like":
            mutant.birth_genes = self.birth_genes.copy()
            mutant.survival_genes = self.survival_genes.copy()
            
            # Mutate birth conditions
            mask = np.random.random(9) < mutation_rate
            mutant.birth_genes[mask] = 1 - mutant.birth_genes[mask]
            
            # Mutate survival conditions  
            mask = np.random.random(9) < mutation_rate
            mutant.survival_genes[mask] = 1 - mutant.survival_genes[mask]
        
        elif self.ca_type == "totalistic":
            mutant.rule_table = self.rule_table.copy()
            # Add small random perturbations
            mask = np.random.random(27) < mutation_rate
            mutant.rule_table[mask] = np.random.randint(0, 3, np.sum(mask))
        
        return mutant
    
    def crossover(self, other: 'CAGenome') -> Tuple['CAGenome', 'CAGenome']:
        """Create two offspring through crossover."""
        if self.ca_type != other.ca_type:
            raise ValueError("Cannot crossover different CA types")
        
        child1 = CAGenome(self.ca_type)
        child2 = CAGenome(self.ca_type)
        
        if self.ca_type == "elementary":
            # Single-point crossover
            crossover_point = np.random.randint(1, 8)
            child1.genes = np.concatenate([self.genes[:crossover_point], 
                                         other.genes[crossover_point:]])
            child2.genes = np.concatenate([other.genes[:crossover_point], 
                                         self.genes[crossover_point:]])
        
        elif self.ca_type == "life_like":
            # Uniform crossover for birth and survival
            mask = np.random.random(9) < 0.5
            
            child1.birth_genes = np.where(mask, self.birth_genes, other.birth_genes)
            child1.survival_genes = np.where(mask, self.survival_genes, other.survival_genes)
            
            child2.birth_genes = np.where(mask, other.birth_genes, self.birth_genes)
            child2.survival_genes = np.where(mask, other.survival_genes, self.survival_genes)
        
        elif self.ca_type == "totalistic":
            # Multi-point crossover
            points = sorted(np.random.choice(27, size=2, replace=False))
            child1.rule_table = np.concatenate([
                self.rule_table[:points[0]],
                other.rule_table[points[0]:points[1]],
                self.rule_table[points[1]:]
            ])
            child2.rule_table = np.concatenate([
                other.rule_table[:points[0]],
                self.rule_table[points[0]:points[1]],
                other.rule_table[points[1]:]
            ])
        
        return child1, child2


class CARuleEvolutionGA(EvolutionaryAlgorithm):
    """Evolutionary algorithm for evolving CA rules."""
    
    def __init__(self,
                 targets: List[CATarget],
                 ca_type: str = "elementary",
                 grid_size: int = 100,
                 generations_per_eval: int = 100,
                 params: Optional[EvolutionaryParams] = None):
        """Initialize CA rule evolution.
        
        Args:
            targets: List of target patterns/behaviors
            ca_type: Type of CA to evolve
            grid_size: Size of CA grid for evaluation
            generations_per_eval: CA generations per fitness evaluation
            params: EA parameters
        """
        self.targets = targets
        self.ca_type = ca_type
        self.grid_size = grid_size
        self.generations_per_eval = generations_per_eval
        
        if params is None:
            params = EvolutionaryParams(
                population_size=50,
                max_generations=100,
                mutation_rate=0.1,
                crossover_rate=0.7,
                selection_type=SelectionType.TOURNAMENT,
                tournament_size=5,
                elitism_rate=0.2
            )
        
        super().__init__(self._evaluate_ca_fitness, params)
    
    def _evaluate_ca_fitness(self, genome: CAGenome) -> float:
        """Evaluate fitness of a CA genome based on targets."""
        total_fitness = 0.0
        
        for target in self.targets:
            if target.target_type == "pattern":
                fitness = self._evaluate_pattern_fitness(genome, target)
            elif target.target_type == "behavior":
                fitness = self._evaluate_behavior_fitness(genome, target)
            elif target.target_type == "property":
                fitness = self._evaluate_property_fitness(genome, target)
            elif target.target_type == "glider":
                fitness = self._evaluate_glider_fitness(genome, target)
            elif target.target_type == "oscillator":
                fitness = self._evaluate_oscillator_fitness(genome, target)
            else:
                fitness = 0.0
            
            total_fitness += fitness * target.weight
        
        return total_fitness
    
    def _create_ca_from_genome(self, genome: CAGenome) -> Any:
        """Create a CA instance from genome."""
        if self.ca_type == "elementary":
            rule_num = genome.to_rule_number()
            ca = ElementaryCA(self.grid_size, rule_num)
            ca.set_single_cell()  # Standard initial condition
            return ca
        elif self.ca_type == "life_like":
            # Would need GameOfLife implementation with custom rules
            # For now, return placeholder
            return None
        else:
            raise ValueError(f"CA type {self.ca_type} not implemented")
    
    def _evaluate_pattern_fitness(self, genome: CAGenome, target: CATarget) -> float:
        """Evaluate fitness based on pattern matching."""
        ca = self._create_ca_from_genome(genome)
        if ca is None:
            return 0.0
        
        # Run CA and get final pattern
        ca.evolve(self.generations_per_eval)
        final_pattern = ca.grid
        target_pattern = target.data
        
        # Calculate pattern similarity (normalized correlation)
        if len(final_pattern) != len(target_pattern):
            # Resize or crop as needed
            min_len = min(len(final_pattern), len(target_pattern))
            final_pattern = final_pattern[:min_len]
            target_pattern = target_pattern[:min_len]
        
        # Correlation coefficient
        correlation = np.corrcoef(final_pattern, target_pattern)[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.0
    
    def _evaluate_behavior_fitness(self, genome: CAGenome, target: CATarget) -> float:
        """Evaluate fitness based on behavioral properties."""
        ca = self._create_ca_from_genome(genome)
        if ca is None:
            return 0.0
        
        # Get space-time diagram
        spacetime = ca.evolve_spacetime(self.generations_per_eval)
        
        behavior = target.data  # Expected to be a behavior description
        
        if behavior == "chaotic":
            # Measure entropy/randomness
            flat = spacetime.flatten()
            probs = np.bincount(flat) / len(flat)
            entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-10))
            return entropy / 10.0  # Normalize
        
        elif behavior == "periodic":
            # Detect periodicity
            for period in range(1, min(20, spacetime.shape[0] // 2)):
                if np.array_equal(spacetime[-1], spacetime[-(period + 1)]):
                    return 1.0
            return 0.0
        
        elif behavior == "stable":
            # Check for convergence to stable state
            if spacetime.shape[0] > 10:
                last_states = spacetime[-5:]
                if np.all(last_states == last_states[0]):
                    return 1.0
            return 0.0
        
        return 0.0
    
    def _evaluate_property_fitness(self, genome: CAGenome, target: CATarget) -> float:
        """Evaluate fitness based on emergent properties."""
        ca = self._create_ca_from_genome(genome)
        if ca is None:
            return 0.0
        
        spacetime = ca.evolve_spacetime(self.generations_per_eval)
        
        property_name = target.data["property"]
        target_value = target.data["value"]
        
        if property_name == "density":
            # Final density of active cells
            final_density = np.mean(spacetime[-1])
            error = abs(final_density - target_value)
            return max(0, 1 - error)
        
        elif property_name == "complexity":
            # Measure using compression ratio
            import zlib
            compressed_size = len(zlib.compress(spacetime.tobytes()))
            original_size = spacetime.nbytes
            complexity = compressed_size / original_size
            error = abs(complexity - target_value)
            return max(0, 1 - error)
        
        return 0.0
    
    def _evaluate_glider_fitness(self, genome: CAGenome, target: CATarget) -> float:
        """Evaluate fitness for glider-like patterns."""
        ca = self._create_ca_from_genome(genome)
        if ca is None:
            return 0.0
        
        spacetime = ca.evolve_spacetime(self.generations_per_eval)
        
        # Simple glider detection: look for diagonal movement
        glider_score = 0.0
        
        for t in range(spacetime.shape[0] - 5):
            for x in range(2, spacetime.shape[1] - 2):
                # Check for diagonal pattern
                if (spacetime[t, x] == 1 and 
                    spacetime[t + 1, (x + 1) % spacetime.shape[1]] == 1 and
                    spacetime[t + 2, (x + 2) % spacetime.shape[1]] == 1):
                    glider_score += 1
        
        return min(1.0, glider_score / 100.0)  # Normalize
    
    def _evaluate_oscillator_fitness(self, genome: CAGenome, target: CATarget) -> float:
        """Evaluate fitness for oscillating patterns."""
        ca = self._create_ca_from_genome(genome)
        if ca is None:
            return 0.0
        
        spacetime = ca.evolve_spacetime(self.generations_per_eval)
        
        # Look for periodic behavior
        target_period = target.data.get("period", 2)
        
        # Check last portion of evolution for periodicity
        if spacetime.shape[0] > target_period * 2:
            test_region = spacetime[-target_period * 2:]
            
            for start in range(target_period):
                if np.array_equal(test_region[start], 
                                test_region[start + target_period]):
                    return 1.0
        
        return 0.0
    
    def create_individual(self) -> Individual:
        """Create a random individual."""
        genome = CAGenome(self.ca_type)
        return Individual(genome=genome)
    
    def initialize_population(self) -> Population:
        """Initialize population with random CA genomes."""
        individuals = [self.create_individual() 
                      for _ in range(self.params.population_size)]
        return Population(individuals)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate a CA genome."""
        mutated = individual.copy()
        mutated.genome = individual.genome.mutate(self.params.mutation_rate)
        mutated.fitness = None  # Reset fitness
        return mutated
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover two CA genomes."""
        child1_genome, child2_genome = parent1.genome.crossover(parent2.genome)
        
        child1 = Individual(genome=child1_genome)
        child2 = Individual(genome=child2_genome)
        
        return child1, child2


class CAEvolutionExperiment:
    """Orchestrates CA rule evolution experiments."""
    
    def __init__(self):
        """Initialize experiment."""
        self.results: List[Dict[str, Any]] = []
    
    def run_glider_evolution(self) -> Dict[str, Any]:
        """Evolve CA rules that produce glider-like patterns."""
        print("Evolving CA rules for glider patterns...")
        
        targets = [
            CATarget("glider", {"speed": 1}, weight=1.0),
            CATarget("behavior", "periodic", weight=0.5)
        ]
        
        params = EvolutionaryParams(
            population_size=30,
            max_generations=50,
            mutation_rate=0.15,
            crossover_rate=0.8,
            selection_type=SelectionType.TOURNAMENT,
            tournament_size=3
        )
        
        ga = CARuleEvolutionGA(
            targets=targets,
            ca_type="elementary",
            grid_size=100,
            generations_per_eval=100,
            params=params
        )
        
        population, stats = ga.evolve()
        best_individual = population.best()
        
        result = {
            "experiment": "glider_evolution",
            "best_rule": best_individual.genome.to_rule_number(),
            "best_fitness": best_individual.fitness,
            "convergence_generation": stats["convergence_generation"],
            "total_evaluations": stats["total_evaluations"],
            "history": ga.history
        }
        
        self.results.append(result)
        return result
    
    def run_density_evolution(self, target_density: float = 0.5) -> Dict[str, Any]:
        """Evolve CA rules for specific final density."""
        print(f"Evolving CA rules for density {target_density}...")
        
        targets = [
            CATarget("property", {"property": "density", "value": target_density}, weight=1.0)
        ]
        
        params = EvolutionaryParams(
            population_size=40,
            max_generations=75,
            mutation_rate=0.1,
            crossover_rate=0.7,
            elitism_rate=0.25
        )
        
        ga = CARuleEvolutionGA(
            targets=targets,
            ca_type="elementary",
            grid_size=200,
            generations_per_eval=150,
            params=params
        )
        
        population, stats = ga.evolve()
        best_individual = population.best()
        
        result = {
            "experiment": "density_evolution",
            "target_density": target_density,
            "best_rule": best_individual.genome.to_rule_number(),
            "best_fitness": best_individual.fitness,
            "convergence_generation": stats["convergence_generation"],
            "total_evaluations": stats["total_evaluations"],
            "history": ga.history
        }
        
        self.results.append(result)
        return result
    
    def visualize_evolution(self, result: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize the evolution process and best result."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"CA Rule Evolution: {result['experiment']}", fontsize=16)
        
        history = result["history"]
        generations = [h["generation"] for h in history]
        
        # Fitness evolution
        ax = axes[0, 0]
        best_fitness = [h["best_fitness"] for h in history]
        mean_fitness = [h["mean_fitness"] for h in history]
        
        ax.plot(generations, best_fitness, 'b-', label='Best', linewidth=2)
        ax.plot(generations, mean_fitness, 'r--', label='Mean', alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Diversity evolution
        ax = axes[0, 1]
        diversity = [h["diversity"] for h in history]
        ax.plot(generations, diversity, 'g-', linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Population Diversity")
        ax.set_title("Diversity Over Time")
        ax.grid(True, alpha=0.3)
        
        # Best CA rule visualization
        ax = axes[1, 0]
        best_rule = result["best_rule"]
        ca = ElementaryCA(100, best_rule)
        ca.set_single_cell()
        spacetime = ca.evolve_spacetime(50)
        
        ax.imshow(spacetime, cmap='binary', aspect='auto')
        ax.set_xlabel("Cell Position")
        ax.set_ylabel("Time Step")
        ax.set_title(f"Best Rule: {best_rule} (Fitness: {result['best_fitness']:.4f})")
        
        # Rule information
        ax = axes[1, 1]
        ax.axis('off')
        
        info_text = f"""
Best Rule Information:
Rule Number: {best_rule}
Final Fitness: {result['best_fitness']:.6f}
Convergence: Gen {result.get('convergence_generation', 'N/A')}
Total Evaluations: {result['total_evaluations']}

Rule Binary: {format(best_rule, '08b')}
Classification: {self._get_rule_classification(best_rule)}
        """
        
        ax.text(0.1, 0.9, info_text.strip(), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def _get_rule_classification(self, rule_num: int) -> str:
        """Get Wolfram classification for rule."""
        from CALab.core.rules import ElementaryRule
        
        try:
            rule = ElementaryRule(rule_num)
            classification = rule.get_classification()
            if classification:
                return f"Class {classification.value}"
        except:
            pass
        
        return "Unknown"
    
    def run_comprehensive_experiment(self) -> Dict[str, Any]:
        """Run multiple evolution experiments."""
        print("Running comprehensive CA rule evolution experiments...")
        
        # Experiment 1: Glider evolution
        glider_result = self.run_glider_evolution()
        
        # Experiment 2: Density evolution (multiple targets)
        density_results = []
        for density in [0.2, 0.5, 0.8]:
            result = self.run_density_evolution(density)
            density_results.append(result)
        
        # Combine results
        comprehensive_result = {
            "glider_evolution": glider_result,
            "density_evolution": density_results,
            "summary": {
                "total_experiments": len(density_results) + 1,
                "best_glider_rule": glider_result["best_rule"],
                "best_glider_fitness": glider_result["best_fitness"],
                "density_results": [
                    {
                        "target": r["target_density"],
                        "rule": r["best_rule"], 
                        "fitness": r["best_fitness"]
                    } for r in density_results
                ]
            }
        }
        
        return comprehensive_result
    
    def export_results(self, filepath: str) -> None:
        """Export all results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results exported to {filepath}")


def main():
    """Run CA rule evolution demonstration."""
    print("CA Rule Evolution: Synthesis of CA and EA")
    print("=" * 50)
    
    experiment = CAEvolutionExperiment()
    
    # Run comprehensive experiment
    results = experiment.run_comprehensive_experiment()
    
    print("\nExperiment Summary:")
    print("=" * 30)
    summary = results["summary"]
    print(f"Total experiments run: {summary['total_experiments']}")
    print(f"Best glider rule: {summary['best_glider_rule']} "
          f"(fitness: {summary['best_glider_fitness']:.4f})")
    
    print("\nDensity evolution results:")
    for result in summary["density_results"]:
        print(f"  Target {result['target']}: Rule {result['rule']} "
              f"(fitness: {result['fitness']:.4f})")
    
    # Visualize best results
    experiment.visualize_evolution(
        results["glider_evolution"], 
        save_path="glider_evolution.png"
    )
    
    # Export results
    experiment.export_results("ca_evolution_results.json")
    
    print("\n" + "=" * 50)
    print("CA Rule Evolution complete!")
    print("This demonstrates how evolutionary algorithms can discover")
    print("novel CA rules that exhibit desired behaviors - a perfect")
    print("example of algorithmic synthesis creating emergent intelligence.")


if __name__ == "__main__":
    main()