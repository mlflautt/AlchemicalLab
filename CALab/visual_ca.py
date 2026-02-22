"""
Visual Cellular Automata with Grid Displays
===========================================

Traditional CA with proper visual grid output and hexagonal grid support.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
from typing import Tuple, List, Dict, Optional
import time

class SquareGridCA:
    """Traditional square grid cellular automaton."""
    
    def __init__(self, size: Tuple[int, int] = (50, 50), rule: str = "B3/S23", 
                 initial_density: float = 0.3, seed: Optional[int] = None):
        self.size = size
        self.rule = rule
        self.generation = 0
        
        # Parse B/S rule
        self.birth_conditions, self.survival_conditions = self._parse_rule(rule)
        
        # Initialize with better patterns for visualization
        if seed is not None:
            np.random.seed(seed)
        
        h, w = size
        if initial_density > 0:
            # Random initialization
            self.grid = np.random.random((h, w)) < initial_density
        else:
            # Start with interesting pattern
            self.grid = np.zeros((h, w), dtype=bool)
            self._add_gliders()
        
        print(f"Square Grid CA: {rule}, size {size}")
    
    def _parse_rule(self, rule: str) -> Tuple[List[int], List[int]]:
        """Parse B/S notation."""
        parts = rule.split('/')
        birth = [int(d) for d in parts[0][1:]]
        survival = [int(d) for d in parts[1][1:]]
        return birth, survival
    
    def _add_gliders(self):
        """Add some glider patterns for interesting dynamics."""
        h, w = self.size
        
        # Glider pattern
        glider = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ], dtype=bool)
        
        # Place a few gliders
        positions = [(10, 10), (20, 30), (30, 15)]
        for y, x in positions:
            if y + 3 < h and x + 3 < w:
                self.grid[y:y+3, x:x+3] = glider
        
        # Add an oscillator (blinker)
        if h > 25 and w > 25:
            self.grid[25, 23:26] = True
    
    def count_neighbors(self) -> np.ndarray:
        """Count neighbors using convolution."""
        from scipy import ndimage
        
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        padded = np.pad(self.grid, 1, mode='wrap')
        neighbors = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        return neighbors
    
    def step(self):
        """Execute one generation."""
        neighbors = self.count_neighbors()
        new_grid = np.zeros_like(self.grid)
        
        # Birth conditions
        for count in self.birth_conditions:
            new_grid |= (~self.grid) & (neighbors == count)
        
        # Survival conditions  
        for count in self.survival_conditions:
            new_grid |= self.grid & (neighbors == count)
        
        self.grid = new_grid
        self.generation += 1
        
        return {
            'generation': self.generation,
            'alive_count': int(np.sum(self.grid)),
            'density': float(np.mean(self.grid))
        }

class HexGridCA:
    """Hexagonal grid cellular automaton."""
    
    def __init__(self, size: Tuple[int, int] = (40, 40), rule: str = "B2/S34", 
                 initial_density: float = 0.3, seed: Optional[int] = None):
        self.size = size
        self.rule = rule
        self.generation = 0
        
        # Parse rule - hex grids have 6 neighbors
        self.birth_conditions, self.survival_conditions = self._parse_rule(rule)
        
        if seed is not None:
            np.random.seed(seed)
        
        h, w = size
        self.grid = np.random.random((h, w)) < initial_density
        
        print(f"Hex Grid CA: {rule}, size {size}")
    
    def _parse_rule(self, rule: str) -> Tuple[List[int], List[int]]:
        """Parse B/S notation."""
        parts = rule.split('/')
        birth = [int(d) for d in parts[0][1:]]
        survival = [int(d) for d in parts[1][1:]]
        return birth, survival
    
    def get_hex_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Get 6 hexagonal neighbors."""
        h, w = self.size
        
        if i % 2 == 0:  # Even row
            neighbors = [
                ((i-1) % h, (j-1) % w),  # NW
                ((i-1) % h, j),          # NE  
                (i, (j-1) % w),          # W
                (i, (j+1) % w),          # E
                ((i+1) % h, (j-1) % w),  # SW
                ((i+1) % h, j)           # SE
            ]
        else:  # Odd row
            neighbors = [
                ((i-1) % h, j),          # NW
                ((i-1) % h, (j+1) % w),  # NE
                (i, (j-1) % w),          # W
                (i, (j+1) % w),          # E
                ((i+1) % h, j),          # SW
                ((i+1) % h, (j+1) % w)   # SE
            ]
        
        return neighbors
    
    def count_neighbors(self) -> np.ndarray:
        """Count neighbors for all cells."""
        h, w = self.size
        neighbors = np.zeros((h, w), dtype=int)
        
        for i in range(h):
            for j in range(w):
                neighbor_coords = self.get_hex_neighbors(i, j)
                count = sum(self.grid[ni, nj] for ni, nj in neighbor_coords)
                neighbors[i, j] = count
        
        return neighbors
    
    def step(self):
        """Execute one generation."""
        neighbors = self.count_neighbors()
        new_grid = np.zeros_like(self.grid)
        
        # Birth conditions
        for count in self.birth_conditions:
            new_grid |= (~self.grid) & (neighbors == count)
        
        # Survival conditions
        for count in self.survival_conditions:
            new_grid |= self.grid & (neighbors == count)
        
        self.grid = new_grid
        self.generation += 1
        
        return {
            'generation': self.generation,
            'alive_count': int(np.sum(self.grid)),
            'density': float(np.mean(self.grid))
        }

class CAVisualizer:
    """Visualizer with proper grid display."""
    
    def __init__(self, ca_system, grid_type: str = "square"):
        self.ca = ca_system
        self.grid_type = grid_type
        
        # Create figure with grid and stats
        self.fig, (self.ax_grid, self.ax_stats) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Setup grid display
        if grid_type == "hex":
            self._setup_hex_display()
        else:
            self._setup_square_display()
        
        # Setup stats
        self.stats_history = []
        self.generation_data = []
        self.alive_data = []
        
        print(f"Visualizer ready for {grid_type} grid")
    
    def _setup_square_display(self):
        """Setup square grid display."""
        self.ax_grid.set_title("Square Grid Cellular Automaton")
        self.ax_grid.set_aspect('equal')
        h, w = self.ca.size
        
        # Initial display
        self.im = self.ax_grid.imshow(self.ca.grid, cmap='binary', interpolation='nearest')
        self.ax_grid.set_xlim(-0.5, w-0.5)
        self.ax_grid.set_ylim(-0.5, h-0.5)
        
        # Add grid lines for better visibility
        self.ax_grid.set_xticks(np.arange(-0.5, w, 5), minor=True)
        self.ax_grid.set_yticks(np.arange(-0.5, h, 5), minor=True)
        self.ax_grid.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    def _setup_hex_display(self):
        """Setup hexagonal grid display.""" 
        self.ax_grid.set_title("Hexagonal Grid Cellular Automaton")
        self.ax_grid.set_aspect('equal')
        
        h, w = self.ca.size
        
        # Calculate hex positions
        self.hex_positions = {}
        self.hex_patches = []
        
        hex_radius = 0.4
        x_spacing = hex_radius * 1.5
        y_spacing = hex_radius * np.sqrt(3)
        
        for i in range(h):
            for j in range(w):
                # Offset every other row
                x = j * x_spacing + (0.75 * hex_radius if i % 2 == 1 else 0)
                y = i * y_spacing * 0.75
                
                self.hex_positions[(i, j)] = (x, y)
                
                # Create hexagon patch
                hex_patch = RegularPolygon((x, y), 6, radius=hex_radius, 
                                         facecolor='white', edgecolor='gray', linewidth=0.5)
                self.ax_grid.add_patch(hex_patch)
                self.hex_patches.append(hex_patch)
        
        # Set display limits
        all_x = [pos[0] for pos in self.hex_positions.values()]
        all_y = [pos[1] for pos in self.hex_positions.values()]
        self.ax_grid.set_xlim(min(all_x) - 1, max(all_x) + 1)
        self.ax_grid.set_ylim(min(all_y) - 1, max(all_y) + 1)
        self.ax_grid.axis('off')
    
    def update_square_display(self):
        """Update square grid display."""
        self.im.set_data(self.ca.grid)
        
        # Add generation info
        stats = self.ca.get_stats() if hasattr(self.ca, 'get_stats') else {
            'generation': self.ca.generation,
            'alive_count': int(np.sum(self.ca.grid)),
            'density': float(np.mean(self.ca.grid))
        }
        
        title = f"Generation {stats['generation']} | Alive: {stats['alive_count']} | Density: {stats['density']:.3f}"
        self.ax_grid.set_title(title)
        
        return stats
    
    def update_hex_display(self):
        """Update hexagonal grid display."""
        h, w = self.ca.size
        
        # Update hex colors
        patch_idx = 0
        for i in range(h):
            for j in range(w):
                alive = self.ca.grid[i, j]
                color = 'black' if alive else 'white'
                self.hex_patches[patch_idx].set_facecolor(color)
                patch_idx += 1
        
        # Update title
        stats = {
            'generation': self.ca.generation,
            'alive_count': int(np.sum(self.ca.grid)),
            'density': float(np.mean(self.ca.grid))
        }
        
        title = f"Generation {stats['generation']} | Alive: {stats['alive_count']} | Density: {stats['density']:.3f}"
        self.ax_grid.set_title(title)
        
        return stats
    
    def update_stats(self, stats):
        """Update statistics plot."""
        self.stats_history.append(stats)
        self.generation_data.append(stats['generation'])
        self.alive_data.append(stats['alive_count'])
        
        self.ax_stats.clear()
        
        if len(self.generation_data) > 1:
            self.ax_stats.plot(self.generation_data, self.alive_data, 'b-', linewidth=2, label='Alive Cells')
            self.ax_stats.set_xlabel('Generation')
            self.ax_stats.set_ylabel('Alive Cells')
            self.ax_stats.set_title('Population Over Time')
            self.ax_stats.grid(True, alpha=0.3)
            self.ax_stats.legend()
        
        # Add current stats as text
        stats_text = f"Rule: {self.ca.rule}\\n"
        stats_text += f"Generation: {stats['generation']}\\n"
        stats_text += f"Population: {stats['alive_count']}\\n"
        stats_text += f"Density: {stats['density']:.4f}"
        
        self.ax_stats.text(0.02, 0.98, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def update(self):
        """Update visualization."""
        # Step the CA
        self.ca.step()
        
        # Update display based on grid type
        if self.grid_type == "hex":
            stats = self.update_hex_display()
        else:
            stats = self.update_square_display()
        
        # Update statistics
        self.update_stats(stats)
        
        # Print occasional updates
        if stats['generation'] % 25 == 0:
            print(f"Generation {stats['generation']}: {stats}")
    
    def run_animation(self, steps_per_second: float = 4, max_generations: int = 200):
        """Run the animation."""
        generation_count = 0
        
        def animate(frame):
            nonlocal generation_count
            if generation_count < max_generations:
                self.update()
                generation_count += 1
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, blit=False)
        
        plt.tight_layout()
        plt.show()
        return ani
    
    def run_static_steps(self, steps: int = 50):
        """Run static visualization showing key generations."""
        print(f"Running {steps} generations...")
        
        # Show initial state
        plt.figure(figsize=(12, 4))
        
        # Initial state
        plt.subplot(1, 3, 1)
        plt.imshow(self.ca.grid, cmap='binary')
        plt.title(f"Generation 0 (Initial)")
        plt.axis('off')
        
        # Run to middle
        for _ in range(steps // 2):
            self.ca.step()
        
        plt.subplot(1, 3, 2)
        plt.imshow(self.ca.grid, cmap='binary')
        plt.title(f"Generation {steps // 2}")
        plt.axis('off')
        
        # Run to end
        for _ in range(steps // 2):
            self.ca.step()
        
        plt.subplot(1, 3, 3)
        plt.imshow(self.ca.grid, cmap='binary')
        plt.title(f"Generation {steps}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def demo_square_grid():
    """Demonstrate square grid CA."""
    print("\\n" + "="*50)
    print("SQUARE GRID CELLULAR AUTOMATON DEMO")
    print("="*50)
    
    # Conway's Game of Life
    ca = SquareGridCA(size=(60, 60), rule="B3/S23", initial_density=0.0, seed=42)
    viz = CAVisualizer(ca, "square")
    
    print("Conway's Game of Life with gliders and oscillators")
    print("Close the window or press Ctrl+C to continue...")
    
    try:
        viz.run_animation(steps_per_second=6, max_generations=150)
    except KeyboardInterrupt:
        print("Animation stopped by user")
    except Exception as e:
        print(f"Animation failed: {e}")
        print("Showing static snapshots instead...")
        viz.run_static_steps(30)

def demo_hex_grid():
    """Demonstrate hexagonal grid CA."""
    print("\\n" + "="*50)
    print("HEXAGONAL GRID CELLULAR AUTOMATON DEMO")
    print("="*50)
    
    # Hexagonal CA with different rule
    ca = HexGridCA(size=(25, 25), rule="B2/S34", initial_density=0.4, seed=42)
    viz = CAVisualizer(ca, "hex")
    
    print("Hexagonal Grid CA - Rule B2/S34")
    print("Close the window or press Ctrl+C to continue...")
    
    try:
        viz.run_animation(steps_per_second=4, max_generations=100)
    except KeyboardInterrupt:
        print("Animation stopped by user")
    except Exception as e:
        print(f"Animation failed: {e}")
        print("Running headless simulation...")
        
        for i in range(20):
            stats = ca.step()
            if i % 5 == 0:
                print(f"Generation {stats['generation']}: {stats}")

def interactive_demo():
    """Interactive demo allowing user choice."""
    print("\\nAlchemicalLab Visual Cellular Automata")
    print("="*45)
    print("1. Square Grid - Conway's Game of Life")
    print("2. Square Grid - HighLife (B36/S23)")  
    print("3. Hexagonal Grid - B2/S34")
    print("4. Hexagonal Grid - B12/S1")
    print("5. Both demonstrations")
    
    try:
        choice = input("\\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            ca = SquareGridCA(size=(50, 50), rule="B3/S23", initial_density=0.0, seed=42)
            viz = CAVisualizer(ca, "square")
            viz.run_animation(steps_per_second=5, max_generations=200)
            
        elif choice == "2":
            ca = SquareGridCA(size=(60, 60), rule="B36/S23", initial_density=0.3, seed=42)
            viz = CAVisualizer(ca, "square") 
            viz.run_animation(steps_per_second=5, max_generations=200)
            
        elif choice == "3":
            ca = HexGridCA(size=(30, 30), rule="B2/S34", initial_density=0.4, seed=42)
            viz = CAVisualizer(ca, "hex")
            viz.run_animation(steps_per_second=4, max_generations=150)
            
        elif choice == "4":
            ca = HexGridCA(size=(25, 25), rule="B12/S1", initial_density=0.2, seed=42)
            viz = CAVisualizer(ca, "hex")
            viz.run_animation(steps_per_second=4, max_generations=150)
            
        else:
            demo_square_grid()
            demo_hex_grid()
            
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    interactive_demo()