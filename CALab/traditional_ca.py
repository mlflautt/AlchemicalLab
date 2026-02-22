"""
Traditional Cellular Automata for AlchemicalLab
===============================================

Implementation of classic cellular automata rules and generations:
- Conway's Game of Life
- Elementary cellular automata (Wolfram rules)
- Multi-state CA (Brian's Brain, Wireworld, etc.)
- Totalistic rules
- Outer totalistic rules
- Life-like rules (B/S notation)

Focus on proper CA simulation fundamentals.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, List, Dict, Callable, Optional
from enum import Enum
import time

class CAType(Enum):
    """Types of cellular automata."""
    ELEMENTARY = "elementary"
    LIFE = "life" 
    BRIAN_BRAIN = "brian_brain"
    WIREWORLD = "wireworld"
    TOTALISTIC = "totalistic"
    OUTER_TOTALISTIC = "outer_totalistic"

class ElementaryCA:
    """1D Elementary Cellular Automaton (Wolfram rules)."""
    
    def __init__(self, size: int = 200, rule: int = 30, seed: Optional[int] = None):
        self.size = size
        self.rule = rule
        self.generation = 0
        
        # Convert rule to lookup table
        self.rule_table = self._rule_to_table(rule)
        
        # Initialize grid
        if seed is not None:
            np.random.seed(seed)
        
        self.current_row = np.random.randint(0, 2, size).astype(bool)
        self.history = [self.current_row.copy()]
        
        print(f"Elementary CA Rule {rule} initialized, size {size}")
    
    def _rule_to_table(self, rule: int) -> Dict[Tuple[bool, bool, bool], bool]:
        """Convert Wolfram rule number to lookup table."""
        rule_binary = format(rule, '08b')  # 8-bit binary representation
        
        patterns = [
            (True, True, True),    # 111
            (True, True, False),   # 110
            (True, False, True),   # 101
            (True, False, False),  # 100
            (False, True, True),   # 011
            (False, True, False),  # 010
            (False, False, True),  # 001
            (False, False, False)  # 000
        ]
        
        lookup = {}
        for i, pattern in enumerate(patterns):
            lookup[pattern] = rule_binary[i] == '1'
        
        return lookup
    
    def step(self):
        """Execute one generation."""
        new_row = np.zeros_like(self.current_row)
        
        for i in range(self.size):
            # Get neighborhood with periodic boundary conditions
            left = self.current_row[(i - 1) % self.size]
            center = self.current_row[i]
            right = self.current_row[(i + 1) % self.size]
            
            # Apply rule
            pattern = (left, center, right)
            new_row[i] = self.rule_table[pattern]
        
        self.current_row = new_row
        self.history.append(self.current_row.copy())
        self.generation += 1
        
        return self.get_stats()
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'generation': self.generation,
            'alive_count': int(np.sum(self.current_row)),
            'density': float(np.mean(self.current_row)),
            'rule': self.rule
        }
    
    def get_history_array(self, max_generations: int = 100) -> np.ndarray:
        """Get history as 2D array for visualization."""
        history_len = min(len(self.history), max_generations)
        return np.array(self.history[-history_len:])

class GameOfLife:
    """Conway's Game of Life and Life-like cellular automata."""
    
    def __init__(self, size: Tuple[int, int] = (100, 100), 
                 rule: str = "B3/S23", seed: Optional[int] = None):
        self.size = size
        self.rule = rule
        self.generation = 0
        
        # Parse rule (B/S notation)
        self.birth_conditions, self.survival_conditions = self._parse_rule(rule)
        
        # Initialize grid
        if seed is not None:
            np.random.seed(seed)
        
        h, w = size
        self.grid = np.random.random((h, w)) < 0.3
        self.previous_grids = []
        
        print(f"Game of Life initialized: {rule}, size {size}")
    
    def _parse_rule(self, rule: str) -> Tuple[List[int], List[int]]:
        """Parse B/S rule notation (e.g., 'B3/S23')."""
        parts = rule.split('/')
        
        birth_part = parts[0][1:]  # Remove 'B'
        survival_part = parts[1][1:]  # Remove 'S'
        
        birth = [int(d) for d in birth_part]
        survival = [int(d) for d in survival_part]
        
        return birth, survival
    
    def count_neighbors(self, grid: np.ndarray) -> np.ndarray:
        """Count neighbors for all cells using convolution."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1], 
                          [1, 1, 1]])
        
        # Pad with periodic boundary conditions
        padded = np.pad(grid, 1, mode='wrap')
        
        # Convolve to count neighbors
        from scipy import ndimage
        neighbor_count = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        return neighbor_count
    
    def step(self):
        """Execute one generation."""
        neighbors = self.count_neighbors(self.grid)
        
        # Apply birth/survival rules
        new_grid = np.zeros_like(self.grid)
        
        # Birth: dead cells with right number of neighbors
        for birth_count in self.birth_conditions:
            new_grid |= (~self.grid) & (neighbors == birth_count)
        
        # Survival: living cells with right number of neighbors
        for survival_count in self.survival_conditions:
            new_grid |= self.grid & (neighbors == survival_count)
        
        # Store previous grid for oscillation detection
        self.previous_grids.append(self.grid.copy())
        if len(self.previous_grids) > 10:
            self.previous_grids.pop(0)
        
        self.grid = new_grid
        self.generation += 1
        
        return self.get_stats()
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            'generation': self.generation,
            'alive_count': int(np.sum(self.grid)),
            'density': float(np.mean(self.grid)),
            'rule': self.rule
        }
    
    def detect_oscillation(self) -> Optional[int]:
        """Detect if pattern is oscillating."""
        if len(self.previous_grids) < 2:
            return None
        
        for period in range(1, min(6, len(self.previous_grids))):
            if period < len(self.previous_grids):
                if np.array_equal(self.grid, self.previous_grids[-period]):
                    return period
        
        return None

class BriansBrain:
    """Brian's Brain 3-state cellular automaton."""
    
    def __init__(self, size: Tuple[int, int] = (100, 100), seed: Optional[int] = None):
        self.size = size
        self.generation = 0
        
        # Initialize grid: 0=dead, 1=firing, 2=refractory
        if seed is not None:
            np.random.seed(seed)
        
        h, w = size
        self.grid = np.random.choice([0, 1, 2], size=(h, w), p=[0.8, 0.1, 0.1])
        
        print(f"Brian's Brain initialized, size {size}")
    
    def count_neighbors(self, grid: np.ndarray, state: int) -> np.ndarray:
        """Count neighbors in specific state."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Create binary grid for specific state
        state_grid = (grid == state).astype(int)
        
        # Pad with wrapping
        padded = np.pad(state_grid, 1, mode='wrap')
        
        # Convolve
        from scipy import ndimage
        neighbor_count = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        return neighbor_count
    
    def step(self):
        """Execute one generation."""
        firing_neighbors = self.count_neighbors(self.grid, 1)  # Count firing neighbors
        
        new_grid = np.zeros_like(self.grid)
        
        # Rules:
        # 1. Dead cells (0) with exactly 2 firing neighbors become firing (1)
        new_grid[(self.grid == 0) & (firing_neighbors == 2)] = 1
        
        # 2. Firing cells (1) always become refractory (2)
        new_grid[self.grid == 1] = 2
        
        # 3. Refractory cells (2) always become dead (0)
        new_grid[self.grid == 2] = 0
        
        self.grid = new_grid
        self.generation += 1
        
        return self.get_stats()
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        unique, counts = np.unique(self.grid, return_counts=True)
        state_counts = dict(zip(unique, counts))
        
        return {
            'generation': self.generation,
            'dead_count': state_counts.get(0, 0),
            'firing_count': state_counts.get(1, 0), 
            'refractory_count': state_counts.get(2, 0),
            'total_active': state_counts.get(1, 0) + state_counts.get(2, 0)
        }

class TotalisticCA:
    """Totalistic cellular automaton (state depends on sum of neighborhood)."""
    
    def __init__(self, size: Tuple[int, int] = (100, 100), 
                 states: int = 3, rule: List[int] = None, seed: Optional[int] = None):
        self.size = size
        self.states = states  # Number of possible states (0 to states-1)
        self.generation = 0
        
        # Default rule if none provided
        if rule is None:
            # Example: state becomes (sum of neighborhood) mod states
            self.rule = list(range(states * 9))  # Max sum is 8 neighbors * (states-1)
        else:
            self.rule = rule
        
        # Initialize grid
        if seed is not None:
            np.random.seed(seed)
        
        h, w = size
        self.grid = np.random.randint(0, states, (h, w))
        
        print(f"Totalistic CA initialized: {states} states, size {size}")
    
    def get_neighborhood_sum(self, grid: np.ndarray) -> np.ndarray:
        """Get sum of 3x3 neighborhood for each cell."""
        kernel = np.array([[1, 1, 1],
                          [1, 1, 1],  # Include center cell
                          [1, 1, 1]])
        
        # Pad with wrapping
        padded = np.pad(grid, 1, mode='wrap')
        
        # Convolve
        from scipy import ndimage
        neighborhood_sum = ndimage.convolve(padded, kernel, mode='constant')[1:-1, 1:-1]
        
        return neighborhood_sum
    
    def step(self):
        """Execute one generation."""
        sums = self.get_neighborhood_sum(self.grid)
        
        # Apply rule based on neighborhood sum
        new_grid = np.zeros_like(self.grid)
        
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                total = sums[i, j]
                if total < len(self.rule):
                    new_grid[i, j] = self.rule[total] % self.states
                else:
                    new_grid[i, j] = total % self.states
        
        self.grid = new_grid
        self.generation += 1
        
        return self.get_stats()
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        unique, counts = np.unique(self.grid, return_counts=True)
        state_counts = dict(zip(unique, counts))
        
        return {
            'generation': self.generation,
            'state_counts': state_counts,
            'entropy': self._calculate_entropy(counts),
            'states': self.states
        }
    
    def _calculate_entropy(self, counts: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        total = np.sum(counts)
        if total == 0:
            return 0.0
        
        probabilities = counts / total
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        return -np.sum(probabilities * np.log2(probabilities))

class CAVisualizer:
    """Visualizer for traditional cellular automata."""
    
    def __init__(self, ca_system, ca_type: CAType):
        self.ca = ca_system
        self.ca_type = ca_type
        
        # Setup visualization
        if ca_type == CAType.ELEMENTARY:
            self.fig, (self.ax_space_time, self.ax_current) = plt.subplots(1, 2, figsize=(12, 6))
        else:
            self.fig, (self.ax_grid, self.ax_stats) = plt.subplots(1, 2, figsize=(12, 6))
        
        self.step_count = 0
        self.stats_history = []
    
    def update_elementary(self):
        """Update elementary CA visualization."""
        # Space-time diagram
        history = self.ca.get_history_array(100)
        
        self.ax_space_time.clear()
        self.ax_space_time.imshow(history, cmap='binary', aspect='auto', origin='upper')
        self.ax_space_time.set_title(f'Elementary CA Rule {self.ca.rule} - Space-Time')
        self.ax_space_time.set_xlabel('Position')
        self.ax_space_time.set_ylabel('Generation')
        
        # Current row
        self.ax_current.clear()
        self.ax_current.plot(self.ca.current_row.astype(int), 'ko-', markersize=2)
        self.ax_current.set_title(f'Generation {self.ca.generation}')
        self.ax_current.set_ylim(-0.1, 1.1)
        self.ax_current.set_xlabel('Position')
        self.ax_current.set_ylabel('State')
    
    def update_2d(self):
        """Update 2D CA visualization."""
        # Grid display
        self.ax_grid.clear()
        
        if self.ca_type == CAType.BRIAN_BRAIN:
            # Color mapping for Brian's Brain
            colors = ['black', 'white', 'gray']  # dead, firing, refractory
            cmap = plt.matplotlib.colors.ListedColormap(colors)
            self.ax_grid.imshow(self.ca.grid, cmap=cmap, vmin=0, vmax=2)
        elif self.ca_type == CAType.TOTALISTIC:
            self.ax_grid.imshow(self.ca.grid, cmap='viridis', vmin=0, vmax=self.ca.states-1)
        else:  # Life-like
            self.ax_grid.imshow(self.ca.grid, cmap='binary')
        
        self.ax_grid.set_title(f'Generation {self.ca.generation}')
        self.ax_grid.axis('off')
        
        # Statistics
        stats = self.ca.get_stats()
        self.stats_history.append(stats)
        
        self.ax_stats.clear()
        if len(self.stats_history) > 1:
            generations = [s['generation'] for s in self.stats_history]
            
            if self.ca_type == CAType.LIFE:
                alive_counts = [s['alive_count'] for s in self.stats_history]
                self.ax_stats.plot(generations, alive_counts, 'g-', label='Alive')
            elif self.ca_type == CAType.BRIAN_BRAIN:
                firing = [s['firing_count'] for s in self.stats_history]
                refractory = [s['refractory_count'] for s in self.stats_history]
                self.ax_stats.plot(generations, firing, 'r-', label='Firing')
                self.ax_stats.plot(generations, refractory, 'gray', label='Refractory')
            elif self.ca_type == CAType.TOTALISTIC:
                entropy = [s['entropy'] for s in self.stats_history]
                self.ax_stats.plot(generations, entropy, 'b-', label='Entropy')
            
            self.ax_stats.legend()
        
        self.ax_stats.set_title('Statistics')
        self.ax_stats.set_xlabel('Generation')
    
    def update(self):
        """Update visualization based on CA type."""
        self.ca.step()
        
        if self.ca_type == CAType.ELEMENTARY:
            self.update_elementary()
        else:
            self.update_2d()
        
        self.step_count += 1
        
        # Print stats occasionally
        if self.step_count % 20 == 0:
            stats = self.ca.get_stats()
            print(f"Generation {stats['generation']}: {stats}")
    
    def run_animation(self, steps_per_second: float = 5, max_steps: int = 200):
        """Run animated visualization."""
        step_count = 0
        
        def animate(frame):
            nonlocal step_count
            if step_count < max_steps:
                self.update()
                step_count += 1
            return []
        
        interval = int(1000 / steps_per_second)
        ani = animation.FuncAnimation(self.fig, animate, interval=interval, blit=False)
        
        plt.tight_layout()
        plt.show()
        return ani

def run_ca_comparison():
    """Run comparison of different CA types."""
    print("Traditional Cellular Automata Comparison")
    print("=" * 50)
    
    # Test different CA types
    ca_systems = [
        (ElementaryCA(size=100, rule=30, seed=42), CAType.ELEMENTARY, "Rule 30"),
        (ElementaryCA(size=100, rule=110, seed=42), CAType.ELEMENTARY, "Rule 110"),
        (GameOfLife(size=(50, 50), rule="B3/S23", seed=42), CAType.LIFE, "Conway's Life"),
        (GameOfLife(size=(50, 50), rule="B36/S23", seed=42), CAType.LIFE, "HighLife"),
        (BriansBrain(size=(50, 50), seed=42), CAType.BRIAN_BRAIN, "Brian's Brain"),
        (TotalisticCA(size=(50, 50), states=4, seed=42), CAType.TOTALISTIC, "4-state Totalistic")
    ]
    
    for ca, ca_type, name in ca_systems:
        print(f"\\nTesting {name}:")
        
        # Run for several generations
        for i in range(10):
            stats = ca.step()
            if i == 0 or i == 9:
                print(f"  Generation {stats['generation']}: {stats}")

if __name__ == "__main__":
    print("AlchemicalLab Traditional Cellular Automata")
    print("=" * 45)
    
    # Run comparison
    run_ca_comparison()
    
    print("\\nChoose a CA to visualize:")
    print("1. Elementary Rule 30")
    print("2. Elementary Rule 110") 
    print("3. Conway's Game of Life")
    print("4. Brian's Brain")
    print("5. Totalistic CA")
    
    try:
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            ca = ElementaryCA(size=200, rule=30, seed=42)
            viz = CAVisualizer(ca, CAType.ELEMENTARY)
        elif choice == "2":
            ca = ElementaryCA(size=200, rule=110, seed=42)
            viz = CAVisualizer(ca, CAType.ELEMENTARY)
        elif choice == "3":
            ca = GameOfLife(size=(80, 80), rule="B3/S23", seed=42)
            viz = CAVisualizer(ca, CAType.LIFE)
        elif choice == "4":
            ca = BriansBrain(size=(80, 80), seed=42)
            viz = CAVisualizer(ca, CAType.BRIAN_BRAIN)
        elif choice == "5":
            ca = TotalisticCA(size=(60, 60), states=5, seed=42)
            viz = CAVisualizer(ca, CAType.TOTALISTIC)
        else:
            print("Invalid choice, using Conway's Life")
            ca = GameOfLife(size=(80, 80), rule="B3/S23", seed=42)
            viz = CAVisualizer(ca, CAType.LIFE)
        
        print("\\nStarting visualization...")
        viz.run_animation(steps_per_second=8, max_steps=300)
        
    except KeyboardInterrupt:
        print("\\nVisualization interrupted")
    except Exception as e:
        print(f"Error: {e}")
        print("Running headless mode...")
        
        # Headless fallback
        ca = GameOfLife(size=(40, 40), rule="B3/S23", seed=42)
        for i in range(20):
            stats = ca.step()
            if i % 5 == 0:
                print(f"Generation {stats['generation']}: {stats}")