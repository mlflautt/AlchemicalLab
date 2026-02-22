"""
Working Conway's Game of Life with proper patterns and boundary conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple
import matplotlib.colors as mcolors

plt.style.use('dark_background')

class ProperGameOfLife:
    """Conway's Game of Life with correct implementation."""
    
    def __init__(self, size: Tuple[int, int] = (100, 100), seed: int = 42):
        self.size = size
        self.generation = 0
        np.random.seed(seed)
        
        h, w = size
        self.grid = np.zeros((h, w), dtype=bool)
        
        # Add well-known stable patterns
        self._add_working_patterns()
        
        print(f"Game of Life initialized: {size}, {np.sum(self.grid)} initial cells")
    
    def _add_working_patterns(self):
        """Add patterns that are known to work."""
        h, w = self.size
        
        # Ensure we have enough space - place patterns away from edges
        margin = 15
        
        # 1. Multiple Gliders
        glider = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ], dtype=bool)
        
        glider_positions = [
            (margin, margin),
            (margin, w - margin - 10),
            (h - margin - 10, margin),
            (h//2, w//2 - 10)
        ]
        
        for y, x in glider_positions:
            if y + 3 < h - margin and x + 3 < w - margin:
                self.grid[y:y+3, x:x+3] = glider
        
        # 2. R-pentomino (creates chaos)
        r_pentomino = np.array([
            [0, 1, 1],
            [1, 1, 0], 
            [0, 1, 0]
        ], dtype=bool)
        
        if h > 50 and w > 50:
            self.grid[h//3:h//3+3, w//3:w//3+3] = r_pentomino
        
        # 3. Stable patterns
        # Block (2x2 stable)
        if h > 30 and w > 30:
            self.grid[20:22, 20:22] = True
        
        # Beehive (stable) 
        beehive = np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ], dtype=bool)
        
        if h > 40 and w > 40:
            self.grid[h-30:h-27, w-30:w-26] = beehive
        
        # 4. Oscillators
        # Blinker (period 2)
        if h > 25 and w > 25:
            self.grid[h//4, w//4:w//4+3] = True
        
        # Toad (period 2)
        toad = np.array([
            [0, 1, 1, 1],
            [1, 1, 1, 0]
        ], dtype=bool)
        
        if h > 30 and w > 30:
            self.grid[h-20:h-18, w-25:w-21] = toad
        
        # 5. Pulsar (period 3) - only if grid is large enough
        if h > 80 and w > 80:
            self._add_pulsar(h//2, w//2)
    
    def _add_pulsar(self, center_y: int, center_x: int):
        """Add pulsar oscillator (period 3)."""
        # Pulsar pattern (13x13)
        pulsar_coords = [
            # Top part
            (2, 4), (2, 5), (2, 6), (2, 10), (2, 11), (2, 12),
            (4, 2), (4, 7), (4, 9), (4, 14),
            (5, 2), (5, 7), (5, 9), (5, 14),
            (6, 2), (6, 7), (6, 9), (6, 14),
            (7, 4), (7, 5), (7, 6), (7, 10), (7, 11), (7, 12),
            # Middle gap at row 8
            (9, 4), (9, 5), (9, 6), (9, 10), (9, 11), (9, 12),
            (10, 2), (10, 7), (10, 9), (10, 14),
            (11, 2), (11, 7), (11, 9), (11, 14),
            (12, 2), (12, 7), (12, 9), (12, 14),
            (14, 4), (14, 5), (14, 6), (14, 10), (14, 11), (14, 12)
        ]
        
        for dy, dx in pulsar_coords:
            y, x = center_y - 7 + dy, center_x - 7 + dx
            if 0 <= y < self.size[0] and 0 <= x < self.size[1]:
                self.grid[y, x] = True
    
    def count_neighbors(self) -> np.ndarray:
        """Count neighbors with proper toroidal (wrapping) boundary conditions."""
        from scipy import ndimage
        
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Use wrap mode for toroidal boundary
        padded = np.pad(self.grid.astype(int), 1, mode='wrap')
        neighbors = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        return neighbors
    
    def step(self):
        """Execute one generation with correct Conway rules."""
        neighbors = self.count_neighbors()
        
        # Conway's Game of Life rules:
        # 1. Any live cell with 2-3 neighbors survives
        # 2. Any dead cell with exactly 3 neighbors becomes alive
        # 3. All other cells die or stay dead
        
        new_grid = np.zeros_like(self.grid)
        
        # Birth: dead cells with exactly 3 neighbors
        new_grid |= (~self.grid) & (neighbors == 3)
        
        # Survival: living cells with 2 or 3 neighbors  
        new_grid |= self.grid & ((neighbors == 2) | (neighbors == 3))
        
        self.grid = new_grid
        self.generation += 1
        
        return {
            'generation': self.generation,
            'alive_count': int(np.sum(self.grid)),
            'density': float(np.mean(self.grid))
        }

class WorkingLifeVisualizer:
    """Dark mode visualizer for working Game of Life."""
    
    def __init__(self, life: ProperGameOfLife):
        self.life = life
        
        # Dark theme setup
        plt.rcParams.update({
            'figure.facecolor': '#0a0a0a',
            'axes.facecolor': '#1a1a1a',
            'text.color': '#ffffff',
            'axes.labelcolor': '#ffffff',
            'xtick.color': '#ffffff',
            'ytick.color': '#ffffff'
        })
        
        self.fig, (self.ax_grid, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Custom colormap: dead=dark, alive=bright cyan
        colors = ['#0a0a0a', '#00ffff']
        self.cmap = mcolors.ListedColormap(colors)
        
        # Grid setup
        self.ax_grid.set_facecolor('#1a1a1a')
        self.im = self.ax_grid.imshow(self.life.grid, cmap=self.cmap, interpolation='nearest')
        
        h, w = self.life.size
        self.ax_grid.set_xlim(-0.5, w-0.5)
        self.ax_grid.set_ylim(-0.5, h-0.5)
        
        # Subtle grid lines
        step = max(5, min(w, h) // 20)
        self.ax_grid.set_xticks(np.arange(-0.5, w, step), minor=True)
        self.ax_grid.set_yticks(np.arange(-0.5, h, step), minor=True)
        self.ax_grid.grid(which='minor', color='#333333', linestyle='-', linewidth=0.2, alpha=0.5)
        
        self.ax_grid.set_title("Conway's Game of Life", color='#ffffff', fontsize=16, pad=20)
        
        # Statistics setup
        self.stats_history = []
        self.generation_data = []
        self.alive_data = []
        
        print("Working Life visualizer ready!")
    
    def update(self):
        """Update the visualization."""
        # Step the simulation
        stats = self.life.step()
        
        # Update grid display
        self.im.set_data(self.life.grid)
        
        title = f"Generation {stats['generation']} | Population: {stats['alive_count']} | Density: {stats['density']:.4f}"
        self.ax_grid.set_title(title, color='#ffffff', fontsize=14)
        
        # Update statistics
        self.stats_history.append(stats)
        self.generation_data.append(stats['generation'])
        self.alive_data.append(stats['alive_count'])
        
        # Plot statistics
        self.ax_stats.clear()
        self.ax_stats.set_facecolor('#1a1a1a')
        
        if len(self.generation_data) > 1:
            self.ax_stats.plot(self.generation_data, self.alive_data, '#00ff88', linewidth=2.5)
            self.ax_stats.set_xlabel('Generation', color='#ffffff')
            self.ax_stats.set_ylabel('Population', color='#ffffff')
            self.ax_stats.set_title('Population Over Time', color='#ffffff', fontsize=14)
            self.ax_stats.grid(True, color='#333333', alpha=0.3)
            
            # Style the axes
            self.ax_stats.tick_params(colors='#ffffff')
            for spine in self.ax_stats.spines.values():
                spine.set_color('#555555')
        
        # Add stats text
        stats_text = f"Generation: {stats['generation']}\\n"
        stats_text += f"Population: {stats['alive_count']}\\n"
        stats_text += f"Density: {stats['density']:.5f}\\n"
        stats_text += f"Rule: B3/S23 (Conway)"
        
        self.ax_stats.text(0.02, 0.98, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontsize=12, color='#ffffff',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#333333', 
                                  edgecolor='#555555', alpha=0.9))
        
        # Print progress
        if stats['generation'] % 25 == 0:
            print(f"Generation {stats['generation']}: {stats['alive_count']} cells alive")
        
        return stats
    
    def run_animation(self, steps_per_second: float = 10, max_generations: int = 500):
        """Run the animation."""
        generation_count = 0
        
        def animate(frame):
            nonlocal generation_count
            if generation_count < max_generations:
                self.update()
                generation_count += 1
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, 
                                    blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        return ani
    
    def run_static_demo(self):
        """Show static snapshots at key generations."""
        generations = [0, 50, 100, 200]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle("Conway's Game of Life Evolution", color='white', fontsize=16)
        
        current_life = ProperGameOfLife(self.life.size, seed=42)  # Fresh copy
        
        for idx, target_gen in enumerate(generations):
            # Evolve to target generation
            while current_life.generation < target_gen:
                current_life.step()
            
            ax = axes[idx // 2, idx % 2]
            ax.set_facecolor('#1a1a1a')
            ax.imshow(current_life.grid, cmap=self.cmap, interpolation='nearest')
            ax.set_title(f"Generation {target_gen}: {np.sum(current_life.grid)} cells", color='white')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main demo function."""
    print("Conway's Game of Life - Working Implementation")
    print("=" * 50)
    
    # Create Game of Life
    life = ProperGameOfLife(size=(120, 120), seed=42)
    viz = WorkingLifeVisualizer(life)
    
    initial_count = np.sum(life.grid)
    print(f"Starting with {initial_count} alive cells")
    print("Patterns included:")
    print("- Gliders (moving)")
    print("- R-pentomino (chaotic growth)")
    print("- Oscillators (blinkers, toads, pulsar)")  
    print("- Stable patterns (blocks, beehives)")
    print()
    
    try:
        choice = input("Choose demo (1=Animation, 2=Static snapshots): ").strip()
        
        if choice == "2":
            viz.run_static_demo()
        else:
            print("Starting animation...")
            print("Watch for:")
            print("- Gliders moving across the grid")
            print("- Oscillating patterns")
            print("- Chaotic growth from R-pentomino")
            print("- Population stabilization")
            viz.run_animation(steps_per_second=12, max_generations=400)
            
    except KeyboardInterrupt:
        print("\\nDemo stopped by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        # Fallback - show that it works
        print("\\nRunning quick test...")
        for i in range(10):
            stats = life.step()
            print(f"Gen {stats['generation']}: {stats['alive_count']} alive")

if __name__ == "__main__":
    main()