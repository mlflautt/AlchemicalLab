"""Hexagonal grid implementation for cellular automata.

Hexagonal grids are more natural and isotropic than square grids,
providing equal distances to all neighbors and avoiding artifacts
from diagonal vs orthogonal neighbors.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math


class HexOrientation(Enum):
    """Hexagon orientation types."""
    POINTY_TOP = "pointy"  # ⬡ Pointy top hexagons
    FLAT_TOP = "flat"      # ⬢ Flat top hexagons


class HexCoordinateSystem(Enum):
    """Coordinate systems for hexagonal grids."""
    OFFSET = "offset"      # Offset coordinates (row, col)
    AXIAL = "axial"        # Axial coordinates (q, r)
    CUBIC = "cubic"        # Cubic coordinates (x, y, z) where x+y+z=0
    DOUBLED = "doubled"    # Doubled coordinates for easier algorithms


@dataclass
class HexCoord:
    """Hexagonal coordinate representation."""
    q: int  # Column (axial)
    r: int  # Row (axial)
    
    @property
    def s(self) -> int:
        """Third cubic coordinate (s = -q - r)."""
        return -self.q - self.r
    
    def to_offset(self, orientation: HexOrientation = HexOrientation.POINTY_TOP) -> Tuple[int, int]:
        """Convert to offset coordinates."""
        if orientation == HexOrientation.POINTY_TOP:
            col = self.q
            row = self.r + (self.q - (self.q & 1)) // 2
        else:  # FLAT_TOP
            row = self.r
            col = self.q + (self.r - (self.r & 1)) // 2
        return (row, col)
    
    @classmethod
    def from_offset(cls, row: int, col: int, 
                   orientation: HexOrientation = HexOrientation.POINTY_TOP) -> 'HexCoord':
        """Create from offset coordinates."""
        if orientation == HexOrientation.POINTY_TOP:
            q = col
            r = row - (col - (col & 1)) // 2
        else:  # FLAT_TOP
            r = row
            q = col - (row - (row & 1)) // 2
        return cls(q, r)
    
    def distance_to(self, other: 'HexCoord') -> int:
        """Calculate hexagonal distance to another coordinate."""
        return (abs(self.q - other.q) + abs(self.q + self.r - other.q - other.r) + 
                abs(self.r - other.r)) // 2
    
    def __add__(self, other: 'HexCoord') -> 'HexCoord':
        """Add two hex coordinates."""
        return HexCoord(self.q + other.q, self.r + other.r)
    
    def __sub__(self, other: 'HexCoord') -> 'HexCoord':
        """Subtract two hex coordinates."""
        return HexCoord(self.q - other.q, self.r - other.r)
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash((self.q, self.r))
    
    def __eq__(self, other: 'HexCoord') -> bool:
        """Check equality."""
        return self.q == other.q and self.r == other.r


class HexagonalGrid:
    """Hexagonal grid for cellular automata.
    
    This implementation supports multiple coordinate systems and
    provides efficient neighbor lookups and transformations.
    """
    
    # Neighbor directions for pointy-top hexagons (axial coordinates)
    POINTY_NEIGHBORS = [
        HexCoord(1, 0),   # East
        HexCoord(1, -1),  # Northeast
        HexCoord(0, -1),  # Northwest
        HexCoord(-1, 0),  # West
        HexCoord(-1, 1),  # Southwest
        HexCoord(0, 1),   # Southeast
    ]
    
    # Neighbor directions for flat-top hexagons (axial coordinates)
    FLAT_NEIGHBORS = [
        HexCoord(1, 0),   # Southeast
        HexCoord(0, 1),   # South
        HexCoord(-1, 1),  # Southwest
        HexCoord(-1, 0),  # Northwest
        HexCoord(0, -1),  # North
        HexCoord(1, -1),  # Northeast
    ]
    
    def __init__(self, 
                 width: int, 
                 height: int,
                 orientation: HexOrientation = HexOrientation.POINTY_TOP,
                 wrap: bool = True):
        """Initialize hexagonal grid.
        
        Args:
            width: Grid width in hexagons
            height: Grid height in hexagons
            orientation: Hexagon orientation
            wrap: Whether to wrap at edges (toroidal topology)
        """
        self.width = width
        self.height = height
        self.orientation = orientation
        self.wrap = wrap
        
        # Initialize grid storage (using offset coordinates internally)
        self.grid = np.zeros((height, width), dtype=np.int32)
        
        # Cache for coordinate conversions
        self._coord_cache: Dict[Tuple[int, int], HexCoord] = {}
        self._reverse_cache: Dict[HexCoord, Tuple[int, int]] = {}
        
        # Precompute valid coordinates
        self._init_coordinates()
    
    def _init_coordinates(self) -> None:
        """Precompute coordinate mappings."""
        for row in range(self.height):
            for col in range(self.width):
                hex_coord = HexCoord.from_offset(row, col, self.orientation)
                self._coord_cache[(row, col)] = hex_coord
                self._reverse_cache[hex_coord] = (row, col)
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get neighbor positions for a cell.
        
        Args:
            row: Row in offset coordinates
            col: Column in offset coordinates
            
        Returns:
            List of (row, col) tuples for valid neighbors
        """
        hex_coord = self._coord_cache[(row, col)]
        neighbors = []
        
        neighbor_dirs = (self.POINTY_NEIGHBORS if self.orientation == HexOrientation.POINTY_TOP
                        else self.FLAT_NEIGHBORS)
        
        for direction in neighbor_dirs:
            neighbor = hex_coord + direction
            
            if self.wrap:
                # Wrap coordinates
                offset = neighbor.to_offset(self.orientation)
                wrapped_row = offset[0] % self.height
                wrapped_col = offset[1] % self.width
                neighbors.append((wrapped_row, wrapped_col))
            else:
                # Check bounds
                offset = neighbor.to_offset(self.orientation)
                if 0 <= offset[0] < self.height and 0 <= offset[1] < self.width:
                    neighbors.append(offset)
        
        return neighbors
    
    def get_neighbor_values(self, row: int, col: int) -> np.ndarray:
        """Get values of neighboring cells.
        
        Args:
            row: Row in offset coordinates
            col: Column in offset coordinates
            
        Returns:
            Array of neighbor values
        """
        neighbors = self.get_neighbors(row, col)
        return np.array([self.grid[n[0], n[1]] for n in neighbors])
    
    def distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate hexagonal distance between two positions.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            Hexagonal distance
        """
        hex1 = self._coord_cache[pos1]
        hex2 = self._coord_cache[pos2]
        return hex1.distance_to(hex2)
    
    def get_ring(self, center: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
        """Get all cells at a specific distance from center.
        
        Args:
            center: Center position (row, col)
            radius: Distance from center
            
        Returns:
            List of positions forming a ring
        """
        if radius == 0:
            return [center]
        
        center_hex = self._coord_cache[center]
        ring = []
        
        # Start at a corner of the ring
        current = center_hex + HexCoord(radius, 0)
        
        # Walk around the ring
        neighbor_dirs = (self.POINTY_NEIGHBORS if self.orientation == HexOrientation.POINTY_TOP
                        else self.FLAT_NEIGHBORS)
        
        for direction_idx in range(6):
            for _ in range(radius):
                offset = current.to_offset(self.orientation)
                if self.wrap:
                    wrapped = (offset[0] % self.height, offset[1] % self.width)
                    ring.append(wrapped)
                elif 0 <= offset[0] < self.height and 0 <= offset[1] < self.width:
                    ring.append(offset)
                
                # Move to next hex in ring
                current = current + neighbor_dirs[(direction_idx + 2) % 6]
        
        return ring
    
    def get_spiral(self, center: Tuple[int, int], radius: int) -> List[Tuple[int, int]]:
        """Get all cells within a radius in spiral order.
        
        Args:
            center: Center position (row, col)
            radius: Maximum distance from center
            
        Returns:
            List of positions in spiral order
        """
        spiral = []
        for r in range(radius + 1):
            spiral.extend(self.get_ring(center, r))
        return spiral
    
    def to_pixel_coordinates(self, row: int, col: int, 
                           hex_size: float = 10.0) -> Tuple[float, float]:
        """Convert hex grid position to pixel coordinates.
        
        Args:
            row: Row in offset coordinates
            col: Column in offset coordinates
            hex_size: Size of hexagon (radius)
            
        Returns:
            (x, y) pixel coordinates
        """
        if self.orientation == HexOrientation.POINTY_TOP:
            x = hex_size * (math.sqrt(3) * col + math.sqrt(3)/2 * (row & 1))
            y = hex_size * (3/2 * row)
        else:  # FLAT_TOP
            x = hex_size * (3/2 * col)
            y = hex_size * (math.sqrt(3)/2 * col + math.sqrt(3) * row)
        
        return (x, y)
    
    def from_pixel_coordinates(self, x: float, y: float, 
                              hex_size: float = 10.0) -> Optional[Tuple[int, int]]:
        """Convert pixel coordinates to hex grid position.
        
        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            hex_size: Size of hexagon (radius)
            
        Returns:
            (row, col) grid position or None if out of bounds
        """
        if self.orientation == HexOrientation.POINTY_TOP:
            q = (math.sqrt(3)/3 * x - 1/3 * y) / hex_size
            r = (2/3 * y) / hex_size
        else:  # FLAT_TOP
            q = (2/3 * x) / hex_size
            r = (-1/3 * x + math.sqrt(3)/3 * y) / hex_size
        
        # Round to nearest hex
        rx = round(q)
        ry = round(r)
        rz = round(-q - r)
        
        x_diff = abs(rx - q)
        y_diff = abs(ry - r)
        z_diff = abs(rz - (-q - r))
        
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        
        hex_coord = HexCoord(rx, ry)
        offset = hex_coord.to_offset(self.orientation)
        
        if 0 <= offset[0] < self.height and 0 <= offset[1] < self.width:
            return offset
        return None
    
    def rotate_60(self, clockwise: bool = True) -> None:
        """Rotate the entire grid by 60 degrees.
        
        Args:
            clockwise: Direction of rotation
        """
        new_grid = np.zeros_like(self.grid)
        
        for row in range(self.height):
            for col in range(self.width):
                hex_coord = self._coord_cache[(row, col)]
                
                if clockwise:
                    # Rotate clockwise: (q, r, s) -> (-r, -s, -q)
                    rotated = HexCoord(-hex_coord.r, -hex_coord.s)
                else:
                    # Rotate counter-clockwise: (q, r, s) -> (-s, -q, -r)
                    rotated = HexCoord(-hex_coord.s, -hex_coord.q)
                
                offset = rotated.to_offset(self.orientation)
                if self.wrap:
                    wrapped = (offset[0] % self.height, offset[1] % self.width)
                    new_grid[wrapped] = self.grid[row, col]
                elif 0 <= offset[0] < self.height and 0 <= offset[1] < self.width:
                    new_grid[offset] = self.grid[row, col]
        
        self.grid = new_grid
    
    def reflect(self, axis: str = "q") -> None:
        """Reflect the grid across an axis.
        
        Args:
            axis: Axis to reflect across ("q", "r", or "s")
        """
        new_grid = np.zeros_like(self.grid)
        
        for row in range(self.height):
            for col in range(self.width):
                hex_coord = self._coord_cache[(row, col)]
                
                if axis == "q":
                    reflected = HexCoord(hex_coord.q, -hex_coord.r - hex_coord.q)
                elif axis == "r":
                    reflected = HexCoord(-hex_coord.q - hex_coord.r, hex_coord.r)
                else:  # axis == "s"
                    reflected = HexCoord(-hex_coord.q, -hex_coord.r)
                
                offset = reflected.to_offset(self.orientation)
                if self.wrap:
                    wrapped = (offset[0] % self.height, offset[1] % self.width)
                    new_grid[wrapped] = self.grid[row, col]
                elif 0 <= offset[0] < self.height and 0 <= offset[1] < self.width:
                    new_grid[offset] = self.grid[row, col]
        
        self.grid = new_grid


class HexagonalCA:
    """Cellular automaton on hexagonal grid.
    
    This provides a complete CA implementation specifically for hex grids,
    with support for various rules and update schemes.
    """
    
    def __init__(self,
                 width: int,
                 height: int,
                 rule_function: Optional[callable] = None,
                 orientation: HexOrientation = HexOrientation.POINTY_TOP,
                 wrap: bool = True):
        """Initialize hexagonal CA.
        
        Args:
            width: Grid width
            height: Grid height
            rule_function: Function (current, neighbors) -> next_state
            orientation: Hexagon orientation
            wrap: Whether to wrap at edges
        """
        self.hex_grid = HexagonalGrid(width, height, orientation, wrap)
        self.rule_function = rule_function or self._default_rule
        self.generation = 0
        self.history = []
        
    def _default_rule(self, current: int, neighbors: np.ndarray) -> int:
        """Default rule: Game of Life adapted for hex grid.
        
        Hex Life: B2/S34 (birth on 2, survive on 3 or 4)
        """
        alive_neighbors = np.sum(neighbors)
        
        if current == 0:  # Dead cell
            return 1 if alive_neighbors == 2 else 0
        else:  # Alive cell
            return 1 if alive_neighbors in [3, 4] else 0
    
    def step(self) -> None:
        """Perform one CA step."""
        new_grid = np.zeros_like(self.hex_grid.grid)
        
        for row in range(self.hex_grid.height):
            for col in range(self.hex_grid.width):
                current = self.hex_grid.grid[row, col]
                neighbors = self.hex_grid.get_neighbor_values(row, col)
                new_grid[row, col] = self.rule_function(current, neighbors)
        
        self.hex_grid.grid = new_grid
        self.generation += 1
    
    def evolve(self, steps: int, record_history: bool = False) -> Optional[List[np.ndarray]]:
        """Evolve the CA for multiple steps.
        
        Args:
            steps: Number of steps to evolve
            record_history: Whether to record history
            
        Returns:
            History of states if recording, else None
        """
        history = [] if record_history else None
        
        for _ in range(steps):
            if record_history:
                history.append(self.hex_grid.grid.copy())
            self.step()
        
        return history
    
    def randomize(self, density: float = 0.5) -> None:
        """Randomize the grid.
        
        Args:
            density: Density of alive cells
        """
        self.hex_grid.grid = np.random.choice(
            [0, 1], 
            size=(self.hex_grid.height, self.hex_grid.width),
            p=[1-density, density]
        )
        self.generation = 0
    
    def clear(self) -> None:
        """Clear the grid."""
        self.hex_grid.grid.fill(0)
        self.generation = 0
    
    def set_pattern(self, pattern: np.ndarray, center: Tuple[int, int]) -> None:
        """Place a pattern on the grid.
        
        Args:
            pattern: 2D pattern array
            center: Center position for pattern
        """
        p_height, p_width = pattern.shape
        
        for i in range(p_height):
            for j in range(p_width):
                row = (center[0] - p_height // 2 + i) % self.hex_grid.height
                col = (center[1] - p_width // 2 + j) % self.hex_grid.width
                self.hex_grid.grid[row, col] = pattern[i, j]
    
    def find_oscillators(self, max_period: int = 10) -> Optional[int]:
        """Detect oscillating patterns.
        
        Args:
            max_period: Maximum period to check
            
        Returns:
            Period if oscillator detected, else None
        """
        initial = self.hex_grid.grid.copy()
        
        for period in range(1, max_period + 1):
            self.step()
            if np.array_equal(self.hex_grid.grid, initial):
                return period
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate grid statistics.
        
        Returns:
            Dictionary of statistics
        """
        total_cells = self.hex_grid.grid.size
        alive_cells = np.sum(self.hex_grid.grid)
        
        return {
            'generation': self.generation,
            'total_cells': total_cells,
            'alive_cells': int(alive_cells),
            'density': float(alive_cells / total_cells),
            'width': self.hex_grid.width,
            'height': self.hex_grid.height,
            'orientation': self.hex_grid.orientation.value,
            'wrap': self.hex_grid.wrap
        }