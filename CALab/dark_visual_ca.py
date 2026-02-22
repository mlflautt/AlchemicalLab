"""
Dark Mode Visual Cellular Automata
==================================

Improved CA with better initial patterns and dark mode visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
from typing import Tuple, List, Dict, Optional
import matplotlib.colors as mcolors

# Set dark mode style
plt.style.use('dark_background')

class SquareGridCA:
    """Square grid cellular automaton with better pattern initialization."""
    
    def __init__(self, size: Tuple[int, int] = (50, 50), rule: str = "B3/S23", 
                 initial_density: float = 0.3, seed: Optional[int] = None):
        self.size = size
        self.rule = rule
        self.generation = 0
        
        # Parse B/S rule
        self.birth_conditions, self.survival_conditions = self._parse_rule(rule)
        
        if seed is not None:
            np.random.seed(seed)
        
        h, w = size
        self.grid = np.zeros((h, w), dtype=bool)
        
        if initial_density > 0:
            # Random initialization
            self.grid = np.random.random((h, w)) < initial_density
        else:
            # Add interesting patterns that actually survive
            self._add_stable_patterns()
        
        print(f"Square Grid CA: {rule}, size {size}")
    
    def _parse_rule(self, rule: str) -> Tuple[List[int], List[int]]:
        """Parse B/S notation."""
        parts = rule.split('/')
        birth = [int(d) for d in parts[0][1:]]
        survival = [int(d) for d in parts[1][1:]]
        return birth, survival
    
    def _add_stable_patterns(self):
        """Add patterns that will actually survive and create interesting dynamics."""
        h, w = self.size
        
        # Glider (moving pattern)
        glider = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ], dtype=bool)
        
        # R-pentomino (chaotic growth pattern)
        r_pentomino = np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0]
        ], dtype=bool)
        
        # Oscillators
        blinker = np.array([[1, 1, 1]], dtype=bool)
        
        toad = np.array([
            [0, 1, 1, 1],
            [1, 1, 1, 0]
        ], dtype=bool)
        
        beacon = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=bool)
        
        # Place patterns with proper spacing
        patterns = [
            (glider, (5, 5)),
            (glider, (10, 25)), 
            (glider, (25, 10)),
            (r_pentomino, (15, 15)),
            (blinker, (30, 5)),
            (toad, (35, 20)),
            (beacon, (5, 35))
        ]
        
        for pattern, (start_y, start_x) in patterns:
            ph, pw = pattern.shape
            if start_y + ph < h and start_x + pw < w:
                self.grid[start_y:start_y+ph, start_x:start_x+pw] |= pattern
        
        # Add some random scattered cells for chaos
        for _ in range(20):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            self.grid[y, x] = True
    
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
    
    def __init__(self, size: Tuple[int, int] = (30, 30), rule: str = "B2/S34", 
                 initial_density: float = 0.4, seed: Optional[int] = None):
        self.size = size
        self.rule = rule
        self.generation = 0
        
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

class DarkCAVisualizer:
    """Dark mode visualizer with improved styling."""
    
    def __init__(self, ca_system, grid_type: str = "square"):
        self.ca = ca_system
        self.grid_type = grid_type
        
        # Create figure with dark background
        plt.rcParams.update({
            'figure.facecolor': '#1a1a1a',
            'axes.facecolor': '#2d2d2d',
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        })
        
        self.fig, (self.ax_grid, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.patch.set_facecolor('#1a1a1a')
        
        # Setup displays
        if grid_type == "hex":
            self._setup_hex_display()
        else:
            self._setup_square_display()
        
        # Stats tracking
        self.stats_history = []
        self.generation_data = []
        self.alive_data = []
        
        print(f"Dark mode visualizer ready for {grid_type} grid")
    
    def _setup_square_display(self):
        """Setup square grid with dark theme."""
        self.ax_grid.set_facecolor('#2d2d2d')
        h, w = self.ca.size
        
        # Custom colormap for dark theme: dead=dark gray, alive=bright cyan
        colors = ['#1a1a1a', '#00ffff']  # Dark gray, bright cyan
        cmap = mcolors.ListedColormap(colors)
        
        self.im = self.ax_grid.imshow(self.ca.grid, cmap=cmap, interpolation='nearest')
        
        self.ax_grid.set_xlim(-0.5, w-0.5)
        self.ax_grid.set_ylim(-0.5, h-0.5)
        
        # Dark grid lines
        self.ax_grid.set_xticks(np.arange(-0.5, w, 5), minor=True)
        self.ax_grid.set_yticks(np.arange(-0.5, h, 5), minor=True)
        self.ax_grid.grid(which='minor', color='#404040', linestyle='-', linewidth=0.3, alpha=0.5)
        
        self.ax_grid.set_title("Square Grid Cellular Automaton", color='white', fontsize=14)
    
    def _setup_hex_display(self):
        """Setup hexagonal grid with dark theme."""
        self.ax_grid.set_facecolor('#2d2d2d')
        h, w = self.ca.size
        
        self.hex_positions = {}
        self.hex_patches = []
        
        hex_radius = 0.4
        x_spacing = hex_radius * 1.5
        y_spacing = hex_radius * np.sqrt(3)
        
        for i in range(h):
            for j in range(w):
                x = j * x_spacing + (0.75 * hex_radius if i % 2 == 1 else 0)
                y = i * y_spacing * 0.75
                
                self.hex_positions[(i, j)] = (x, y)
                
                # Dark hex patches
                hex_patch = RegularPolygon((x, y), 6, radius=hex_radius, 
                                         facecolor='#1a1a1a', edgecolor='#404040', linewidth=0.5)
                self.ax_grid.add_patch(hex_patch)
                self.hex_patches.append(hex_patch)
        
        all_x = [pos[0] for pos in self.hex_positions.values()]
        all_y = [pos[1] for pos in self.hex_positions.values()]
        self.ax_grid.set_xlim(min(all_x) - 1, max(all_x) + 1)
        self.ax_grid.set_ylim(min(all_y) - 1, max(all_y) + 1)
        self.ax_grid.axis('off')
        self.ax_grid.set_title("Hexagonal Grid Cellular Automaton", color='white', fontsize=14)
    
    def update_square_display(self):
        """Update square grid display."""
        self.im.set_data(self.ca.grid)
        
        stats = {
            'generation': self.ca.generation,
            'alive_count': int(np.sum(self.ca.grid)),
            'density': float(np.mean(self.ca.grid))
        }
        
        title = f"Gen {stats['generation']} | Pop: {stats['alive_count']} | Density: {stats['density']:.3f}"
        self.ax_grid.set_title(title, color='white', fontsize=12)
        
        return stats
    
    def update_hex_display(self):
        """Update hexagonal grid display."""
        h, w = self.ca.size
        
        patch_idx = 0
        for i in range(h):
            for j in range(w):
                alive = self.ca.grid[i, j]
                color = '#00ffff' if alive else '#1a1a1a'  # Cyan or dark
                self.hex_patches[patch_idx].set_facecolor(color)
                patch_idx += 1
        
        stats = {
            'generation': self.ca.generation,
            'alive_count': int(np.sum(self.ca.grid)),
            'density': float(np.mean(self.ca.grid))
        }
        
        title = f"Gen {stats['generation']} | Pop: {stats['alive_count']} | Density: {stats['density']:.3f}"
        self.ax_grid.set_title(title, color='white', fontsize=12)
        
        return stats
    
    def update_stats(self, stats):
        """Update statistics with dark theme."""
        self.stats_history.append(stats)
        self.generation_data.append(stats['generation'])
        self.alive_data.append(stats['alive_count'])
        
        self.ax_stats.clear()
        self.ax_stats.set_facecolor('#2d2d2d')
        
        if len(self.generation_data) > 1:
            self.ax_stats.plot(self.generation_data, self.alive_data, '#00ff41', linewidth=2.5, label='Population')
            self.ax_stats.set_xlabel('Generation', color='white')
            self.ax_stats.set_ylabel('Alive Cells', color='white')
            self.ax_stats.set_title('Population Dynamics', color='white', fontsize=14)
            self.ax_stats.grid(True, color='#404040', alpha=0.3)
            self.ax_stats.legend(facecolor='#2d2d2d', edgecolor='#404040')
            
            # Color the axes
            self.ax_stats.tick_params(colors='white')
            self.ax_stats.spines['bottom'].set_color('#404040')
            self.ax_stats.spines['left'].set_color('#404040')
            self.ax_stats.spines['top'].set_color('#404040')
            self.ax_stats.spines['right'].set_color('#404040')
        
        # Stats text box with dark theme
        stats_text = f"Rule: {self.ca.rule}\\n"
        stats_text += f"Generation: {stats['generation']}\\n"
        stats_text += f"Population: {stats['alive_count']}\\n"
        stats_text += f"Density: {stats['density']:.4f}"
        
        self.ax_stats.text(0.02, 0.98, stats_text, transform=self.ax_stats.transAxes,
                          verticalalignment='top', fontsize=11, color='white',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#404040', 
                                  edgecolor='#606060', alpha=0.8))
    
    def update(self):
        """Update visualization."""
        self.ca.step()
        
        if self.grid_type == "hex":
            stats = self.update_hex_display()
        else:
            stats = self.update_square_display()
        
        self.update_stats(stats)
        
        # Progress updates
        if stats['generation'] % 20 == 0:
            print(f"Generation {stats['generation']}: Population = {stats['alive_count']}")
    
    def run_animation(self, steps_per_second: float = 5, max_generations: int = 300):
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

def demo_conway_life():
    """Demo Conway's Game of Life with proper patterns."""
    print("\\n" + "="*60)
    print("CONWAY'S GAME OF LIFE - Dark Mode")
    print("="*60)
    
    ca = SquareGridCA(size=(80, 80), rule="B3/S23", initial_density=0.0, seed=42)
    viz = DarkCAVisualizer(ca, "square")
    
    print("Starting with gliders, oscillators, and chaotic patterns...")
    print("Watch for:")
    print("- Moving gliders")
    print("- Oscillating blinkers and toads")
    print("- Chaotic R-pentomino evolution")
    
    try:
        viz.run_animation(steps_per_second=8, max_generations=200)
    except Exception as e:
        print(f"Animation failed: {e}")

def demo_hex_ca():
    """Demo hexagonal cellular automaton."""
    print("\\n" + "="*60)
    print("HEXAGONAL CELLULAR AUTOMATON - Dark Mode")
    print("="*60)
    
    ca = HexGridCA(size=(35, 35), rule="B2/S34", initial_density=0.3, seed=42)
    viz = DarkCAVisualizer(ca, "hex")
    
    print("Hexagonal grid with 6-neighbor rule B2/S34...")
    
    try:
        viz.run_animation(steps_per_second=6, max_generations=150)
    except Exception as e:
        print(f"Animation failed: {e}")

def interactive_demo():
    """Interactive demo menu."""
    print("\\nAlchemicalLab - Dark Mode Visual Cellular Automata")
    print("="*55)
    print("1. Conway's Game of Life (Square Grid)")
    print("2. Hexagonal Grid CA (B2/S34)")
    print("3. HighLife (Square Grid B36/S23)")
    print("4. Both demonstrations")
    
    try:
        choice = input("\\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            demo_conway_life()
        elif choice == "2":
            demo_hex_ca()
        elif choice == "3":
            ca = SquareGridCA(size=(70, 70), rule="B36/S23", initial_density=0.25, seed=42)
            viz = DarkCAVisualizer(ca, "square")
            viz.run_animation(steps_per_second=6, max_generations=200)
        else:
            demo_conway_life()
            demo_hex_ca()
            
    except KeyboardInterrupt:
        print("\\nDemo interrupted")
    except Exception as e:
        print(f"Demo failed: {e}")

if __name__ == "__main__":
    interactive_demo()