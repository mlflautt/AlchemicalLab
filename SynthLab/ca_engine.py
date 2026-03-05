"""
Cellular Automata Engine for Audio Synthesis.

Implements multiple CA rules including:
- Conway's Game of Life
- Life Without Death (Live cells never die)
- HighLife
- Brian's Brain
- Day & Night
- Rule 30/90/110 (1D CAs extended to 2D)
- Seeds
- And more...
"""

import numpy as np
from typing import Tuple, Callable, Optional, List, Dict, Any
from enum import Enum
from scipy import ndimage


class CARule(str, Enum):
    """Supported CA rules."""
    CONWAY = "conway"
    LIFE_WITHOUT_DEATH = "life_without_death"
    HIGHLIFE = "highlife"
    BRIANS_BRAIN = "brians_brain"
    DAY_NIGHT = "day_night"
    SEEDS = "seeds"
    REPLICATOR = "replicator"
    MORLEY = "morley"
    RULE_30 = "rule30"
    RULE_90 = "rule90"
    RULE_110 = "rule110"
    WINDOW = "window"
    SERIOUS = "serious"
    INVERSE_LIFE = "inverse_life"


class CANeighborhood(str, Enum):
    """Neighborhood types."""
    MOORE = "moore"       # 8 neighbors
    VON_NEUMANN = "von_neumann"  # 4 neighbors (N, S, E, W)
    HEX = "hex"          # Hexagonal grid (6 neighbors)


class CAEngine:
    """
    Cellular Automata engine for audio/terrain synthesis.
    
    Supports multiple rules, neighborhoods, and state counts.
    """
    
    # Rule definitions: (birth_conditions, survival_conditions, states)
    RULE_DEFINITIONS = {
        CARule.CONWAY: ([3], [2, 3], 2),
        CARule.LIFE_WITHOUT_DEATH: ([3], [2, 3, 4, 5, 6, 7, 8], 2),
        CARule.HIGHLIFE: ([3, 6], [2, 3], 2),
        CARule.BRIANS_BRAIN: ([2], [1, 2], 3),  # 0=dying, 1=alive, 2=dying
        CARule.DAY_NIGHT: ([3, 6, 7, 8], [3, 4, 6, 7, 8], 2),
        CARule.SEEDS: ([2], [], 2),  # Explodes
        CARule.REPLICATOR: ([1, 3, 5, 7], [1, 3, 5, 7], 2),
        CARule.MORLEY: ([2, 4, 5], [2, 4], 2),
        CARule.RULE_30: ([], [1], 2),  # Chaotic 1D-like
        CARule.RULE_90: ([], [1], 2),  # Sierpinski
        CARule.RULE_110: ([3, 4, 5], [2, 3], 2),  # Turing complete
        CARule.WINDOW: ([2], [2, 3, 4], 2),
        CARule.SERIOUS: ([2], [3, 4, 5], 2),
        CARule.INVERSE_LIFE: ([3], [0, 1, 2, 4, 5, 6, 7, 8], 2),
    }
    
    def __init__(
        self,
        size: Tuple[int, int],
        rule: CARule = CARule.CONWAY,
        neighborhood: CANeighborhood = CANeighborhood.MOORE,
        states: int = 2,
        seed: Optional[int] = None
    ):
        self.size = size
        self.rule = rule
        self.neighborhood = neighborhood
        self.generation = 0
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize grid
        self.grid = np.zeros(size, dtype=np.uint8)
        
        # Get rule parameters
        if rule in self.RULE_DEFINITIONS:
            birth, survival, rule_states = self.RULE_DEFINITIONS[rule]
            self.birth_conditions = set(birth)
            self.survival_conditions = set(survival)
            self.rule_states = min(rule_states, states)
        else:
            self.birth_conditions = {3}
            self.survival_conditions = {2, 3}
            self.rule_states = states
        
        # Build neighborhood kernel
        self.kernel = self._build_kernel()
    
    def _build_kernel(self) -> np.ndarray:
        """Build convolution kernel for neighborhood counting."""
        if self.neighborhood == CANeighborhood.MOORE:
            # 8-neighbor kernel
            return np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ], dtype=np.uint8)
        elif self.neighborhood == CANeighborhood.VON_NEUMANN:
            # 4-neighbor kernel (von Neumann)
            return np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ], dtype=np.uint8)
        else:
            return self.kernel
    
    def initialize_random(self, density: float = 0.3):
        """Initialize grid with random state."""
        self.grid = (np.random.random(self.size) < density).astype(np.uint8)
        self.generation = 0
    
    def initialize_pattern(self, pattern: np.ndarray, center: Tuple[int, int] = None):
        """Initialize with a specific pattern."""
        h, w = pattern.shape
        if center is None:
            center = (self.size[0] // 2 - h // 2, self.size[1] // 2 - w // 2)
        
        y_start = max(0, center[0])
        x_start = max(0, center[1])
        y_end = min(self.size[0], y_start + h)
        x_end = min(self.size[1], x_start + w)
        
        py_start = y_start - center[0]
        px_start = x_start - center[1]
        
        self.grid[y_start:y_end, x_start:x_end] = pattern[
            py_start:py_start + (y_end - y_start),
            px_start:px_start + (x_end - x_start)
        ]
    
    def step(self):
        """Advance CA by one generation."""
        # Count neighbors
        padded = np.pad(self.grid, 1, mode='wrap')
        neighbor_count = ndimage.convolve(
            padded.astype(np.int16),
            self.kernel,
            mode='constant'
        )[1:-1, 1:-1]
        
        new_grid = np.zeros_like(self.grid)
        
        if self.rule == CARule.BRIANS_BRAIN:
            # Special handling for 3-state rule
            # 0 = dead, 1 = alive, 2 = dying
            alive = self.grid == 1
            dying = self.grid == 2
            dead = self.grid == 0
            
            # Dead cells become alive if exactly 2 neighbors
            new_grid[dead & (neighbor_count == 2)] = 1
            
            # Alive cells become dying
            new_grid[alive] = 2
            
            # Dying cells become dead
            new_grid[dying] = 0
            
        elif self.rule in [CARule.RULE_30, CARule.RULE_90, CARule.RULE_110]:
            # 1D CA extended to 2D (apply rule to each row)
            for y in range(self.size[0]):
                row = self.grid[y, :]
                new_row = self._apply_1d_rule(row, self.rule)
                new_grid[y, :] = new_row
            self.generation += 1
            self.grid = new_grid
            return
        
        else:
            # Standard 2-state rules
            alive = self.grid == 1
            dead = self.grid == 0
            
            # Birth: dead cell with exact number of neighbors
            new_grid[dead & np.isin(neighbor_count, list(self.birth_conditions))] = 1
            
            # Survival: alive cell with exact number of neighbors
            new_grid[alive & np.isin(neighbor_count, list(self.survival_conditions))] = 1
        
        self.grid = new_grid
        self.generation += 1
    
    def _apply_1d_rule(self, row: np.ndarray, rule: CARule) -> np.ndarray:
        """Apply 1D CA rule to a row."""
        n = len(row)
        new_row = np.zeros(n, dtype=np.uint8)
        
        # Extended row with wrap
        extended = np.concatenate([row[-1:], row, row[:1]])
        
        if rule == CARule.RULE_30:
            # Rule 30: 000->0, 001->1, 010->1, 011->1, 100->1, 101->0, 110->0, 111->0
            for i in range(n):
                pattern = (extended[i] << 2) | (extended[i+1] << 1) | extended[i+2]
                new_row[i] = 0b00111110 >> (7 - pattern) & 1
                
        elif rule == CARule.RULE_90:
            # Rule 90: XOR of left and right neighbors
            new_row = extended[:n] ^ extended[2:]
            
        elif rule == CARule.RULE_110:
            # Rule 110: 000->0, 001->1, 010->1, 011->1, 100->0, 101->1, 110->1, 111->0
            for i in range(n):
                pattern = (extended[i] << 2) | (extended[i+1] << 1) | extended[i+2]
                new_row[i] = 0b01101110 >> (7 - pattern) & 1
        
        return new_row
    
    def step_n(self, n: int):
        """Advance CA by n generations."""
        for _ in range(n):
            self.step()
    
    def get_terrain_height(self, normalize: bool = True) -> np.ndarray:
        """Get normalized terrain height from CA state."""
        if normalize:
            return self.grid.astype(np.float64)
        return self.grid.copy()
    
    def get_density(self) -> float:
        """Get current density of alive cells."""
        return np.mean(self.grid)
    
    def get_activity(self) -> float:
        """Get activity level (cells that changed in last step)."""
        # This requires storing previous state
        return 0.5  # Placeholder


class HexCAEngine:
    """
    Hexagonal Grid Cellular Automata.
    
    Uses axial coordinate system for hex grids.
    """
    
    # Hex neighbor offsets (axial coordinates)
    HEX_DIRECTIONS = [
        (+1, 0), (+1, -1), (0, -1),
        (-1, 0), (-1, +1), (0, +1)
    ]
    
    def __init__(self, radius: int, rule: str = "conway_hex"):
        self.radius = radius
        self.rule = rule
        self.generation = 0
        
        # Size is (2*radius + 1)
        size = 2 * radius + 1
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.coords = self._hex_coords()
    
    def _hex_coords(self) -> List[Tuple[int, int]]:
        """Generate valid hex coordinates within radius."""
        coords = []
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                if abs(q + r) <= self.radius:
                    coords.append((q + self.radius, r + self.radius))
        return coords
    
    def initialize_random(self, density: float = 0.3):
        """Initialize with random state."""
        for q, r in self.coords:
            self.grid[q, r] = (np.random.random() < density)
        self.generation = 0
    
    def _get_neighbors(self, q: int, r: int) -> List[Tuple[int, int]]:
        """Get neighbor coordinates in axial system."""
        q_adj = q - self.radius
        r_adj = r - self.radius
        
        neighbors = []
        for dq, dr in self.HEX_DIRECTIONS:
            nq = q_adj + dq
            nr = r_adj + dr
            if 0 <= nq < 2*self.radius + 1 and 0 <= nr < 2*self.radius + 1:
                if abs(nq + nr - self.radius) <= self.radius:
                    neighbors.append((nq + self.radius, nr + self.radius))
        return neighbors
    
    def step(self):
        """Advance hex CA by one generation."""
        new_grid = np.zeros_like(self.grid)
        
        birth_set = {2, 3}  # Default Conway-like
        survival_set = {3, 4}
        
        if self.rule == "seeds_hex":
            birth_set = {2}
            survival_set = set()
        
        for q, r in self.coords:
            neighbors = self._get_neighbors(q, r)
            alive_count = sum(self.grid[nq, nr] for nq, nr in neighbors)
            
            if self.grid[q, r] == 1:
                new_grid[q, r] = 1 if alive_count in survival_set else 0
            else:
                new_grid[q, r] = 1 if alive_count in birth_set else 0
        
        self.grid = new_grid
        self.generation += 1
    
    def get_terrain_height(self) -> Dict[Tuple[int, int], float]:
        """Get terrain heights as dict."""
        return {(q, r): float(self.grid[q, r]) for q, r in self.coords}
    
    def to_array(self, size: Tuple[int, int] = None) -> np.ndarray:
        """Convert to dense array for visualization."""
        if size is None:
            size = (2 * self.radius + 1, 2 * self.radius + 1)
        
        arr = np.zeros(size)
        for q, r in self.coords:
            arr[q, r] = self.grid[q, r]
        return arr


class MultiStateCAEngine(CAEngine):
    """Extended CA engine with more than 2 states."""
    
    def __init__(
        self,
        size: Tuple[int, int],
        rule: CARule,
        states: int = 4,
        seed: Optional[int] = None
    ):
        super().__init__(size, rule, CANeighborhood.MOORE, states, seed)
        self.rule_states = states
    
    def step(self):
        """Advance multi-state CA."""
        padded = np.pad(self.grid, 1, mode='wrap')
        neighbor_count = ndimage.convolve(
            padded.astype(np.int16),
            self.kernel,
            mode='constant'
        )[1:-1, 1:-1]
        
        new_grid = self.grid.copy()
        
        for state in range(self.rule_states):
            cells = self.grid == state
            
            # State transitions based on neighbor count
            if state < self.rule_states - 1:
                # Cells progress based on neighbor activity
                active_neighbors = neighbor_count[cells]
                new_grid[cells & (active_neighbors > 3)] = (state + 1) % self.rule_states
            else:
                # Highest state can die
                new_grid[self.grid == state] = 0
        
        self.grid = new_grid
        self.generation += 1


def get_rule_info(rule: str) -> Dict[str, Any]:
    """Get information about a CA rule."""
    try:
        rule_enum = CARule(rule.lower())
    except ValueError:
        return {"name": rule, "description": "Unknown rule", "type": "custom"}
    
    descriptions = {
        CARule.CONWAY: {
            "name": "Conway's Game of Life",
            "description": "The classic CA - creates gliders, oscillators, and chaos",
            "type": "standard",
            "behavior": "chaotic"
        },
        CARule.LIFE_WITHOUT_DEATH: {
            "name": "Life Without Death",
            "description": "Cells never die, creating organic growth patterns",
            "type": "growth",
            "behavior": "expanding"
        },
        CARule.HIGHLIFE: {
            "name": "HighLife",
            "description": "Like Conway but with 6-neighbor birth - creates interesting structures",
            "type": "standard",
            "behavior": "chaotic"
        },
        CARule.BRIANS_BRAIN: {
            "name": "Brian's Brain",
            "description": "3-state rule creating moving wave patterns",
            "type": "three_state",
            "behavior": "wave_like"
        },
        CARule.DAY_NIGHT: {
            "name": "Day & Night",
            "description": "Symmetric rule with rich structures",
            "type": "standard",
            "behavior": "complex"
        },
        CARule.SEEDS: {
            "name": "Seeds",
            "description": "Explosive growth - all seeds spread rapidly",
            "type": "explosive",
            "behavior": "expanding"
        },
        CARule.REPLICATOR: {
            "name": "Replicator",
            "description": "Creates replicating patterns",
            "type": "replicator",
            "behavior": "replicating"
        },
        CARule.MORLEY: {
            "name": "Morley",
            "description": "Creates moving ships and puffers",
            "type": "standard",
            "behavior": "moving"
        },
        CARule.RULE_30: {
            "name": "Rule 30",
            "description": "Chaotic 1D CA - great for textures",
            "type": "1d_chaotic",
            "behavior": "chaotic"
        },
        CARule.RULE_90: {
            "name": "Rule 90",
            "description": "Creates Sierpinski triangles",
            "type": "1d_fractal",
            "behavior": "fractal"
        },
        CARule.RULE_110: {
            "name": "Rule 110",
            "description": "Turing complete - complex behavior",
            "type": "1d_complex",
            "behavior": "complex"
        },
    }
    
    return descriptions.get(rule_enum, {"name": rule_enum.value, "description": "", "type": "standard"})


# Preset patterns for common CA structures
COMMON_PATTERNS = {
    "glider": np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8),
    
    "blinker": np.array([
        [1, 1, 1]
    ], dtype=np.uint8),
    
    "block": np.array([
        [1, 1],
        [1, 1]
    ], dtype=np.uint8),
    
    "beehive": np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ], dtype=np.uint8),
    
    "lwss": np.array([  # Lightweight spaceship
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [0, 1, 1, 1, 0]
    ], dtype=np.uint8),
    
    "pulsar": np.array([  # Partial pulsar
        [0,0,1,1,1,0,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,0,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,0,1,0,0,0,0,1],
        [0,0,1,1,1,0,0,0,0,1,1,1,0,0],
    ], dtype=np.uint8),
}


if __name__ == '__main__':
    # Test the CA engine
    print("Testing CA Engine...")
    
    # Create Conway's Game of Life
    ca = CAEngine((50, 50), CARule.CONWAY, seed=42)
    ca.initialize_random(0.3)
    
    print(f"Initial density: {ca.get_density():.3f}")
    
    # Evolve for 50 generations
    for _ in range(50):
        ca.step()
    
    print(f"After 50 generations: {ca.get_density():.3f}")
    print(f"Grid sum: {ca.grid.sum()}")
    
    # Test rule info
    print("\nRule info:")
    info = get_rule_info("conway")
    print(f"  {info}")
    
    # Test hex grid
    print("\nTesting Hex CA...")
    hex_ca = HexCAEngine(20)
    hex_ca.initialize_random(0.3)
    hex_ca.step()
    print(f"Hex CA initialized with {len(hex_ca.coords)} cells")
