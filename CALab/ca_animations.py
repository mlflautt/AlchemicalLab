"""
Animated CA Visualizations
==========================

Create animated visualizations showing Conway's Game of Life evolution across many timesteps.
Includes real-time statistics, pattern analysis, and multiple initial conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import available CA systems
try:
    from evolutionary.genetic_ca import EvolutionaryCA, EvolutionaryConfig, FitnessFunctions
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False
    print("Warning: Evolutionary CA not available")


@dataclass
class AnimationConfig:
    """Configuration for CA animations."""
    grid_size: Tuple[int, int] = (64, 64)
    n_steps: int = 200
    interval: int = 50  # ms between frames
    fps: int = 20
    dpi: int = 100
    figsize: Tuple[int, int] = (12, 8)
    output_dir: str = 'ca_animations'
    save_animation: bool = True


class ConwayAnimator:
    """Creates animated visualizations of Conway's Game of Life."""

    def __init__(self, config: Optional[AnimationConfig] = None):
        self.config = config or AnimationConfig()
        self.gol_cmap = self._create_gol_colormap()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _create_gol_colormap(self) -> LinearSegmentedColormap:
        """Create colormap for Conway's Game of Life."""
        gol_colors = ['black', 'white']
        return LinearSegmentedColormap.from_list('gol', gol_colors)

    def animate_conway_life(self, initial_pattern: str = 'random', n_steps: int = 200) -> animation.Animation:
        """Animate Conway's Game of Life evolution with statistics."""
        print("Animating Conway's Game of Life...")

        # Initialize grid
        if initial_pattern == 'glider':
            grid = self._create_glider()
        elif initial_pattern == 'glider_gun':
            grid = self._create_glider_gun()
        elif initial_pattern == 'random':
            grid = np.random.choice([0, 1], size=self.config.grid_size, p=[0.7, 0.3])
        else:
            grid = np.random.choice([0, 1], size=self.config.grid_size, p=[0.5, 0.5])

        # Setup animation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figsize, dpi=self.config.dpi)

        # Main grid visualization
        im1 = ax1.imshow(grid, cmap=self.gol_cmap, interpolation='nearest', vmin=0, vmax=1)
        ax1.set_title("Conway's Game of Life")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Population over time
        steps = []
        populations = []
        line1, = ax2.plot([], [], 'b-', linewidth=2, label='Population')
        ax2.set_xlim(0, n_steps)
        ax2.set_ylim(0, grid.size)
        ax2.set_title('Population Over Time')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Live Cells')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Entropy over time
        entropies = []
        line2, = ax3.plot([], [], 'r-', linewidth=2, label='Spatial Entropy')
        ax3.set_xlim(0, n_steps)
        ax3.set_ylim(0, 1)
        ax3.set_title('Spatial Entropy')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Entropy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Pattern density
        densities = []
        line3, = ax4.plot([], [], 'g-', linewidth=2, label='Pattern Density')
        ax4.set_xlim(0, n_steps)
        ax4.set_ylim(0, 1)
        ax4.set_title('Pattern Density')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        def animate(frame):
            nonlocal grid
            # Update CA
            new_grid = np.zeros_like(grid)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    # Count neighbors
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                neighbors += grid[ni, nj]

                    # Apply rules
                    if grid[i, j] == 1:  # Alive
                        if neighbors in [2, 3]:
                            new_grid[i, j] = 1
                    else:  # Dead
                        if neighbors == 3:
                            new_grid[i, j] = 1

            grid = new_grid
            im1.set_array(grid)

            # Update statistics
            steps.append(frame)
            populations.append(np.sum(grid))

            # Calculate spatial entropy
            if np.sum(grid) > 0:
                probs = np.bincount(grid.flatten(), minlength=2) / grid.size
                entropy = -np.sum(probs * np.log(probs + 1e-10))
            else:
                entropy = 0.0
            entropies.append(entropy)

            # Calculate pattern density (local variation)
            if grid.shape[0] > 1 and grid.shape[1] > 1:
                # Simple density measure: fraction of cells that differ from their neighbors
                diff_count = 0
                total_checks = 0
                for i in range(grid.shape[0] - 1):
                    for j in range(grid.shape[1] - 1):
                        if grid[i, j] != grid[i, j+1] or grid[i, j] != grid[i+1, j]:
                            diff_count += 1
                        total_checks += 1
                density = diff_count / total_checks if total_checks > 0 else 0
            else:
                density = 0.0
            densities.append(density)

            # Update plots
            line1.set_data(steps, populations)
            line2.set_data(steps, entropies)
            line3.set_data(steps, densities)

            ax1.set_title(f"Conway's Game of Life - Step {frame}")
            return [im1, line1, line2, line3]

        anim = animation.FuncAnimation(
            fig, animate, frames=n_steps, interval=self.config.interval, blit=True
        )

        if self.config.save_animation:
            filename = os.path.join(self.config.output_dir, f'conway_life_{initial_pattern}.gif')
            anim.save(filename, writer='pillow', fps=self.config.fps)
            print(f"Animation saved to {filename}")

        return anim

    def animate_evolutionary_ca(self, genome_type: str = 'elementary', n_generations: int = 20):
        """Animate evolutionary CA discovering patterns."""
        if not EVO_AVAILABLE:
            print("Evolutionary CA not available")
            return None

        print(f"Animating evolutionary {genome_type} CA...")

        # Setup evolution
        evo_config = EvolutionaryConfig(  # type: ignore
            population_size=10,  # Smaller for animation
            n_generations=n_generations,
            grid_size=(32, 32),
            n_steps=20,  # Shorter simulations for animation
            n_trials=2
        )

        evo_ca = EvolutionaryCA(evo_config, genome_type)  # type: ignore
        fitness_fns = [FitnessFunctions.complexity_fitness()]  # type: ignore

        # Setup animation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), dpi=self.config.dpi)

        # Current best pattern
        best_grid = np.zeros((32, 32))
        im1 = ax1.imshow(best_grid, cmap=self.gol_cmap, interpolation='nearest', vmin=0, vmax=1)
        ax1.set_title("Current Best Pattern")
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # Fitness history
        fitness_history = []
        line1, = ax2.plot([], [], 'r-', linewidth=2, label='Best Fitness')
        line2, = ax2.plot([], [], 'b-', linewidth=2, label='Avg Fitness')
        ax2.set_xlim(0, n_generations)
        ax2.set_ylim(0, 2)  # Adjust based on fitness range
        ax2.set_title('Fitness Evolution')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Rule complexity (for elementary CA)
        complexities = []
        line3, = ax3.plot([], [], 'g-', linewidth=2, label='Rule Complexity')
        ax3.set_xlim(0, n_generations)
        ax3.set_ylim(0, 1)
        ax3.set_title('Rule Complexity')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Complexity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Pattern evolution over time
        pattern_history = []
        im2 = ax4.imshow(best_grid, cmap=self.gol_cmap, interpolation='nearest', vmin=0, vmax=1)
        ax4.set_title("Pattern Evolution")
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')

        generation = 0

        def animate(frame):
            nonlocal generation, best_grid

            if frame < n_generations:
                # Evolve one generation
                evo_ca.evolve(fitness_fns, n_generations=1)

                # Get current best
                best_individual = max(evo_ca.population, key=lambda x: x.fitness)
                fitness_history.append(evo_ca.best_fitness_history[-1])

                # Simulate best individual to get pattern
                best_grid = self._simulate_genome(best_individual.genome, evo_config)

                # Calculate rule complexity (simple measure)
                if hasattr(best_individual.genome, 'rule_table'):
                    # For elementary CA, count unique rules
                    rule_table = getattr(best_individual.genome, 'rule_table', [])
                    unique_rules = len(set(rule_table))
                    complexity = unique_rules / 8.0  # Normalize
                else:
                    complexity = 0.5  # Default for other types
                complexities.append(complexity)

                generation = frame

            # Update displays
            im1.set_array(best_grid)
            im2.set_array(best_grid)

            # Update fitness plot
            gens = list(range(len(evo_ca.best_fitness_history)))
            if len(gens) > 0:
                line1.set_data(gens, evo_ca.best_fitness_history)
                if hasattr(evo_ca, 'avg_fitness_history') and evo_ca.avg_fitness_history:
                    line2.set_data(gens, evo_ca.avg_fitness_history)
                # Adjust y-axis
                if evo_ca.best_fitness_history:
                    ax2.set_ylim(0, max(evo_ca.best_fitness_history) * 1.1)

            # Update complexity plot
            line3.set_data(list(range(len(complexities))), complexities)

            ax1.set_title(f"Best Pattern - Gen {generation}")
            ax2.set_title(f'Fitness Evolution - Best: {evo_ca.best_fitness_history[-1]:.3f}')
            return [im1, im2, line1, line2, line3]

        anim = animation.FuncAnimation(
            fig, animate, frames=n_generations, interval=self.config.interval * 3, blit=True
        )

        if self.config.save_animation:
            filename = os.path.join(self.config.output_dir, f'evolutionary_{genome_type}_ca.gif')
            anim.save(filename, writer='pillow', fps=self.config.fps)
            print(f"Animation saved to {filename}")

        return anim

    def _simulate_genome(self, genome, config) -> np.ndarray:
        """Simulate a CA genome to get final pattern."""
        # Create initial state
        initial = np.random.choice([0, 1], size=config.grid_size, p=[0.5, 0.5])
        grid = initial.copy()

        # Evolve for some steps
        for step in range(min(config.n_steps, 15)):  # Limit for animation
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

                    new_grid[i, j] = genome.to_rule_function()(grid[i, j], np.array(neighbors))
            grid = new_grid

        return grid

    def _create_glider(self) -> np.ndarray:
        """Create a glider pattern."""
        grid = np.zeros(self.config.grid_size, dtype=int)
        center_i, center_j = self.config.grid_size[0] // 2, self.config.grid_size[1] // 2

        # Glider pattern
        glider = [
            (0, 1),
            (1, 2),
            (2, 0), (2, 1), (2, 2)
        ]

        for di, dj in glider:
            i, j = center_i + di, center_j + dj
            if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                grid[i, j] = 1

        return grid

    def _create_glider_gun(self) -> np.ndarray:
        """Create Gosper glider gun pattern."""
        grid = np.zeros(self.config.grid_size, dtype=int)

        # Simplified glider gun (smaller version for animation)
        gun_pattern = [
            (5, 1), (5, 2), (6, 1), (6, 2), (5, 11), (6, 11), (7, 11),
            (4, 12), (3, 13), (3, 14), (8, 12), (9, 13), (9, 14),
            (6, 15), (4, 16), (5, 17), (6, 17), (7, 17), (6, 18),
            (8, 16), (3, 21), (4, 21), (5, 21), (3, 22), (4, 22), (5, 22),
            (2, 23), (6, 23), (1, 25), (2, 25), (6, 25), (7, 25)
        ]

        for i, j in gun_pattern:
            if i < grid.shape[0] and j < grid.shape[1]:
                grid[i, j] = 1

        return grid

    def animate_multiple_patterns(self, n_steps: int = 150) -> animation.Animation:
        """Animate multiple different initial patterns simultaneously."""
        print("Animating multiple Conway patterns...")

        # Create different initial patterns
        patterns = {
            'random': np.random.choice([0, 1], size=(32, 32), p=[0.7, 0.3]),
            'glider': self._create_glider(),
            'dense': np.random.choice([0, 1], size=(32, 32), p=[0.4, 0.6]),
            'sparse': np.random.choice([0, 1], size=(32, 32), p=[0.9, 0.1])
        }

        # Setup animation
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=self.config.dpi)
        axes = axes.flatten()

        images = []
        titles = []
        grids = []

        for i, (name, grid) in enumerate(patterns.items()):
            im = axes[i].imshow(grid, cmap=self.gol_cmap, interpolation='nearest', vmin=0, vmax=1)
            axes[i].set_title(f"{name.capitalize()} Initial")
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            images.append(im)
            titles.append(f"{name.capitalize()}")
            grids.append(grid.copy())

        def animate(frame):
            for i, grid in enumerate(grids):
                # Update each grid
                new_grid = np.zeros_like(grid)
                for x in range(grid.shape[0]):
                    for y in range(grid.shape[1]):
                        # Count neighbors
                        neighbors = 0
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                                    neighbors += grid[nx, ny]

                        # Apply rules
                        if grid[x, y] == 1:  # Alive
                            if neighbors in [2, 3]:
                                new_grid[x, y] = 1
                        else:  # Dead
                            if neighbors == 3:
                                new_grid[x, y] = 1

                grids[i] = new_grid
                images[i].set_array(new_grid)
                axes[i].set_title(f"{titles[i]} - Step {frame}")

            return images

        anim = animation.FuncAnimation(
            fig, animate, frames=n_steps, interval=self.config.interval, blit=True
        )

        if self.config.save_animation:
            filename = os.path.join(self.config.output_dir, 'conway_multiple_patterns.gif')
            anim.save(filename, writer='pillow', fps=self.config.fps)
            print(f"Animation saved to {filename}")

        return anim


def create_ca_animations_showcase():
    """Create a showcase of various CA animations."""
    animator = ConwayAnimator()

    animations = []

    # Conway's Game of Life patterns (just random for speed)
    try:
        anim = animator.animate_conway_life('random', n_steps=50)
        animations.append(("Conway_random", anim))
        print("Created animation for random Conway pattern")
    except Exception as e:
        print(f"Failed to create Conway animation: {e}")

    # Multiple patterns comparison
    try:
        anim = animator.animate_multiple_patterns(n_steps=40)
        animations.append(("Conway_multiple", anim))
        print("Created multiple patterns animation")
    except Exception as e:
        print(f"Failed to create multiple patterns animation: {e}")

    # Evolutionary CA animations (only elementary for speed)
    if EVO_AVAILABLE:
        try:
            anim = animator.animate_evolutionary_ca('elementary', n_generations=10)
            animations.append(("Evolutionary_elementary", anim))
            print("Created evolutionary animation for elementary CA")
        except Exception as e:
            print(f"Failed to create evolutionary animation: {e}")
    else:
        print("Evolutionary CA not available, skipping evolutionary animations")

    return animations


if __name__ == "__main__":
    # Create comprehensive CA animations showcase
    animations = create_ca_animations_showcase()

    print(f"\nCreated {len(animations)} animations successfully")
    print("Animation files saved in ca_animations/ directory")