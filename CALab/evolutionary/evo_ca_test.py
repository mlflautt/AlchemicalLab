#!/usr/bin/env python3
"""
Evolutionary CA Test Script
===========================

Demonstrates evolving cellular automata rules for various objectives.
"""

import numpy as np
import matplotlib.pyplot as plt
from CALab.evolutionary.genetic_ca import (
    EvolutionaryCA, EvolutionaryConfig, FitnessFunctions,
    ElementaryCAGenome, create_target_pattern
)


def test_basic_evolution():
    """Test basic evolutionary CA with complexity fitness."""
    print("Testing Evolutionary CA with Complexity Fitness")
    print("=" * 50)

    # Configuration
    config = EvolutionaryConfig(
        population_size=20,  # Small for quick testing
        n_generations=10,    # Few generations for demo
        grid_size=(16, 16),
        n_steps=20,
        n_trials=2
    )

    # Create evolutionary system
    evo_ca = EvolutionaryCA(config, genome_type='elementary')

    # Fitness functions
    fitness_functions = [FitnessFunctions.complexity_fitness()]

    print(f"Initial population size: {len(evo_ca.population)}")
    print("Evolving...")

    # Evolve
    best_individual = evo_ca.evolve(fitness_functions, n_generations=5)

    print("Evolution completed!")
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness:.4f}")

    # Show fitness history
    stats = evo_ca.get_statistics()
    print(f"Final best fitness: {stats['best_fitness']:.4f}")
    print(f"Final average fitness: {stats['avg_fitness']:.4f}")

    return evo_ca


def test_pattern_evolution():
    """Test evolving CA to match a target pattern."""
    print("\nTesting Pattern Matching Evolution")
    print("=" * 50)

    # Create target pattern (glider)
    target = create_target_pattern('glider', (16, 16))
    print("Target pattern: Glider")

    # Configuration
    config = EvolutionaryConfig(
        population_size=20,
        n_generations=10,
        grid_size=(16, 16),
        n_steps=10,
        n_trials=2
    )

    # Create evolutionary system
    evo_ca = EvolutionaryCA(config, genome_type='elementary')

    # Fitness function for pattern matching
    fitness_functions = [FitnessFunctions.pattern_similarity_target(target)]

    print("Evolving towards glider pattern...")

    # Evolve
    best_individual = evo_ca.evolve(fitness_functions, n_generations=5)

    print("Pattern evolution completed!")
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness:.4f}")

    return evo_ca


def test_replication_evolution():
    """Test evolving self-replicating patterns."""
    print("\nTesting Self-Replication Evolution")
    print("=" * 50)

    # Configuration
    config = EvolutionaryConfig(
        population_size=20,
        n_generations=10,
        grid_size=(16, 16),
        n_steps=15,
        n_trials=2
    )

    # Create evolutionary system
    evo_ca = EvolutionaryCA(config, genome_type='elementary')

    # Fitness function for self-replication
    fitness_functions = [FitnessFunctions.self_replication_fitness()]

    print("Evolving self-replicating patterns...")

    # Evolve
    best_individual = evo_ca.evolve(fitness_functions, n_generations=5)

    print("Replication evolution completed!")
    print(f"Best individual: {best_individual}")
    print(f"Best fitness: {best_individual.fitness:.4f}")

    return evo_ca


def visualize_evolution_results(evo_ca, title):
    """Visualize evolution results."""
    stats = evo_ca.get_statistics()

    if stats['best_fitness_history']:
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        plt.plot(stats['best_fitness_history'], label='Best Fitness')
        plt.plot(stats['avg_fitness_history'], label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'{title} - Fitness Evolution')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # Show best individual details
        best = max(evo_ca.population, key=lambda x: x.fitness)
        plt.bar(range(len(best.fitness_components)),
                list(best.fitness_components.values()))
        plt.xticks(range(len(best.fitness_components)),
                   list(best.fitness_components.keys()), rotation=45)
        plt.ylabel('Component Value')
        plt.title(f'Best Individual Components\nFitness: {best.fitness:.4f}')
        plt.tight_layout()

        plt.savefig(f'evolutionary_ca_{title.lower().replace(" ", "_")}_results.png',
                    dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Results saved as: evolutionary_ca_{title.lower().replace(' ', '_')}_results.png")


def run_comprehensive_test():
    """Run comprehensive evolutionary CA tests."""
    print("Evolutionary CA Framework Test Suite")
    print("=" * 60)

    try:
        # Test 1: Complexity evolution
        print("\n1. COMPLEXITY EVOLUTION TEST")
        evo_complexity = test_basic_evolution()
        visualize_evolution_results(evo_complexity, "Complexity Evolution")

        # Test 2: Pattern matching
        print("\n2. PATTERN MATCHING TEST")
        evo_pattern = test_pattern_evolution()
        visualize_evolution_results(evo_pattern, "Pattern Matching")

        # Test 3: Self-replication
        print("\n3. SELF-REPLICATION TEST")
        evo_replication = test_replication_evolution()
        visualize_evolution_results(evo_replication, "Self-Replication")

        print("\n" + "=" * 60)
        print("✅ All evolutionary CA tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Genome types: Elementary CA rules")
        print("- Fitness functions: Complexity, Pattern matching, Self-replication")
        print("- Evolutionary operators: Selection, Crossover, Mutation")
        print("- Population dynamics and statistics tracking")

        print("\nNext Steps:")
        print("- Implement additional genome types (Totalistic, Neural)")
        print("- Add more fitness functions")
        print("- Scale to larger populations and longer evolution")
        print("- Integrate with visualization system")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("This might be due to missing dependencies or implementation issues.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_test()