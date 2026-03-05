#!/usr/bin/env python3
"""Hello Emergence: First AlchemicalLab Synthesis Experiment.

This simple demonstration shows the basic synthesis of CA, EA, and NN
concepts in the AlchemicalLab framework. It serves as both a test of
the system and an introduction to emergent algorithmic synthesis.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add lab paths
lab_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, lab_root)

def hello_cellular_automata():
    """Demonstrate basic CA functionality."""
    print("🔬 Testing Cellular Automata Lab...")
    
    try:
        from CALab.models.elementary import ElementaryCA
        from CALab.ca_core.grids.hexagonal import HexagonalCA
        
        # Test elementary CA
        ca = ElementaryCA(size=50, rule_number=30)
        ca.set_single_cell()
        spacetime = ca.evolve_spacetime(25)
        
        print(f"  ✓ Elementary CA (Rule 30): {spacetime.shape} spacetime")
        
        # Test hexagonal CA
        hex_ca = HexagonalCA(30, 30)
        hex_ca.randomize(density=0.3)
        hex_ca.evolve(steps=10)
        
        print(f"  ✓ Hexagonal CA: {hex_ca.get_statistics()}")
        
        return spacetime, hex_ca.hex_grid.grid
        
    except ImportError as e:
        print(f"  ❌ CA Lab import error: {e}")
        return None, None


def hello_evolutionary_algorithms():
    """Demonstrate basic EA functionality."""
    print("🧬 Testing Evolutionary Algorithms Lab...")
    
    try:
        from EALab.algorithms.base import EvolutionaryParams, Individual, Population
        
        # Simple fitness function: maximize sum of binary array
        def fitness_func(genome):
            return sum(genome) / len(genome)
        
        # Create test parameters
        params = EvolutionaryParams(
            population_size=20,
            max_generations=10,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        # Create test individuals
        individuals = []
        for _ in range(5):
            genome = np.random.randint(0, 2, 10)
            individual = Individual(genome=genome, fitness=fitness_func(genome))
            individuals.append(individual)
        
        population = Population(individuals)
        population.update_statistics()
        
        print(f"  ✓ EA Framework: Pop size {len(population)}")
        print(f"  ✓ Best fitness: {population.best().fitness:.3f}")
        print(f"  ✓ Mean fitness: {population.mean_fitness():.3f}")
        
        return population
        
    except ImportError as e:
        print(f"  ❌ EA Lab import error: {e}")
        return None


def hello_synthesis():
    """Demonstrate basic synthesis capability."""
    print("⚗️ Testing Synthesis Lab...")
    
    # Simple CA-EA hybrid: evolve a pattern that matches a target
    target_pattern = np.array([1, 1, 0, 1, 0, 1, 1, 0])
    
    def ca_fitness(rule_bits):
        """Fitness function: how well does CA rule reproduce target pattern."""
        # Convert 8-bit rule to rule number
        rule_num = sum(bit * (2 ** i) for i, bit in enumerate(rule_bits))
        
        try:
            from CALab.models.elementary import ElementaryCA
            ca = ElementaryCA(size=len(target_pattern), rule_number=rule_num)
            ca.randomize(density=0.5)
            ca.evolve(generations=5)
            
            # Compare final state to target
            final_state = ca.grid[:len(target_pattern)]
            similarity = np.sum(final_state == target_pattern) / len(target_pattern)
            return similarity
        except:
            return 0.0
    
    # Simple evolutionary search for best rule
    best_rule = None
    best_fitness = 0.0
    
    for _ in range(20):  # Quick search
        rule_bits = np.random.randint(0, 2, 8)
        fitness = ca_fitness(rule_bits)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_rule = rule_bits.copy()
    
    rule_number = sum(bit * (2 ** i) for i, bit in enumerate(best_rule))
    
    print(f"  ✓ CA-EA Synthesis: Best rule {rule_number}")
    print(f"  ✓ Target match: {best_fitness:.3f}")
    
    return rule_number, best_fitness


def visualize_emergence(ca_spacetime, hex_grid, best_rule):
    """Create a simple visualization of emergent patterns."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Hello Emergence: AlchemicalLab Demonstration", fontsize=16)
    
    # Elementary CA spacetime
    if ca_spacetime is not None:
        axes[0].imshow(ca_spacetime, cmap='binary', aspect='auto')
        axes[0].set_title("Elementary CA (Rule 30)")
        axes[0].set_xlabel("Space")
        axes[0].set_ylabel("Time")
    else:
        axes[0].text(0.5, 0.5, "CA Lab\nNot Available", 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # Hexagonal CA state
    if hex_grid is not None:
        axes[1].imshow(hex_grid, cmap='viridis', aspect='equal')
        axes[1].set_title("Hexagonal CA")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
    else:
        axes[1].text(0.5, 0.5, "Hex CA\nNot Available", 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    # Synthesis result
    if best_rule is not None:
        try:
            from CALab.models.elementary import ElementaryCA
            ca = ElementaryCA(size=30, rule_number=best_rule)
            ca.set_single_cell()
            result = ca.evolve_spacetime(20)
            
            axes[2].imshow(result, cmap='plasma', aspect='auto')
            axes[2].set_title(f"Evolved Rule {best_rule}")
            axes[2].set_xlabel("Space")
            axes[2].set_ylabel("Time")
        except:
            axes[2].text(0.5, 0.5, f"Evolved Rule\n{best_rule}", 
                        ha='center', va='center', transform=axes[2].transAxes)
    else:
        axes[2].text(0.5, 0.5, "Synthesis\nNot Available", 
                    ha='center', va='center', transform=axes[2].transAxes)
    
    plt.tight_layout()
    plt.savefig("hello_emergence.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Visualization saved as 'hello_emergence.png'")


def main():
    """Run the Hello Emergence demonstration."""
    print("🧪 AlchemicalLab: Hello Emergence")
    print("=" * 50)
    print("Testing the synthesis of computational paradigms...")
    print()
    
    # Test each lab component
    ca_spacetime, hex_grid = hello_cellular_automata()
    population = hello_evolutionary_algorithms()
    best_rule, synthesis_fitness = hello_synthesis()
    
    print()
    print("🎉 Synthesis Results:")
    print("-" * 30)
    
    if ca_spacetime is not None:
        print(f"✓ CA Lab: Working ({ca_spacetime.shape} patterns)")
    else:
        print("⚠ CA Lab: Limited functionality")
    
    if population is not None:
        print(f"✓ EA Lab: Working ({len(population)} individuals)")
    else:
        print("⚠ EA Lab: Limited functionality")
    
    if best_rule is not None:
        print(f"✓ Synthesis: Working (Rule {best_rule}, fit={synthesis_fitness:.3f})")
    else:
        print("⚠ Synthesis: Limited functionality")
    
    print()
    print("Creating emergence visualization...")
    visualize_emergence(ca_spacetime, hex_grid, best_rule)
    
    print()
    print("🔮 Philosophical Reflection:")
    print("-" * 40)
    print("Even this simple demonstration shows emergent behavior:")
    print("• CA Rule 30 creates complex patterns from simple rules")
    print("• Hexagonal grids show different symmetries than square grids")  
    print("• Evolution discovers rules without explicit programming")
    print("• Synthesis combines algorithms to solve novel problems")
    print()
    print("This is the essence of the AlchemicalLab:")
    print("Simple algorithms + Synthesis → Emergent Intelligence")
    print()
    print("🚀 Next Steps:")
    print("- Run SynthLab/hybrid_systems/ca_rule_evolution.py for advanced synthesis")
    print("- Explore CALab/experiments/research/hex_vs_square_comparison.py")
    print("- Build your own synthesis experiments in SynthLab/")
    print()
    print("Welcome to the future of computational alchemy! 🧪⚡")


if __name__ == "__main__":
    main()