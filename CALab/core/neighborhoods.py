"""Neighborhood definitions for cellular automata."""

import numpy as np
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
from enum import Enum


class Neighborhood(ABC):
    """Abstract base class for neighborhood definitions."""
    
    @abstractmethod
    def get_neighbors(self, grid: np.ndarray, position: Tuple[int, ...], 
                     boundary: 'BoundaryCondition') -> np.ndarray:
        """Get neighbor states for a position.
        
        Args:
            grid: The CA grid
            position: Position to get neighbors for
            boundary: Boundary condition to apply
            
        Returns:
            Array of neighbor states
        """
        pass
    
    @abstractmethod
    def get_offsets(self) -> List[Tuple[int, ...]]:
        """Get relative positions of neighbors.
        
        Returns:
            List of offset tuples
        """
        pass


class MooreNeighborhood(Neighborhood):
    """Moore neighborhood (8 neighbors in 2D, 26 in 3D)."""
    
    def __init__(self, dimensions: int = 2, radius: int = 1):
        """Initialize Moore neighborhood.
        
        Args:
            dimensions: Number of dimensions (1, 2, or 3)
            radius: Radius of neighborhood
        """
        self.dimensions = dimensions
        self.radius = radius
        self._offsets = self._generate_offsets()
    
    def _generate_offsets(self) -> List[Tuple[int, ...]]:
        """Generate offset positions for Moore neighborhood."""
        offsets = []
        
        if self.dimensions == 1:
            for i in range(-self.radius, self.radius + 1):
                if i != 0:
                    offsets.append((i,))
        
        elif self.dimensions == 2:
            for i in range(-self.radius, self.radius + 1):
                for j in range(-self.radius, self.radius + 1):
                    if i != 0 or j != 0:
                        offsets.append((i, j))
        
        elif self.dimensions == 3:
            for i in range(-self.radius, self.radius + 1):
                for j in range(-self.radius, self.radius + 1):
                    for k in range(-self.radius, self.radius + 1):
                        if i != 0 or j != 0 or k != 0:
                            offsets.append((i, j, k))
        
        else:
            raise ValueError(f"Unsupported dimensions: {self.dimensions}")
        
        return offsets
    
    def get_offsets(self) -> List[Tuple[int, ...]]:
        """Get relative positions of neighbors."""
        return self._offsets.copy()
    
    def get_neighbors(self, grid: np.ndarray, position: Tuple[int, ...], 
                     boundary: 'BoundaryCondition') -> np.ndarray:
        """Get neighbor states using Moore neighborhood."""
        from .automaton import BoundaryCondition
        
        neighbors = []
        shape = grid.shape
        
        for offset in self._offsets:
            neighbor_pos = tuple(p + o for p, o in zip(position, offset))
            
            if boundary == BoundaryCondition.PERIODIC:
                # Wrap around edges
                neighbor_pos = tuple(n % s for n, s in zip(neighbor_pos, shape))
                neighbors.append(grid[neighbor_pos])
            
            elif boundary == BoundaryCondition.FIXED:
                # Use 0 for out-of-bounds
                if all(0 <= n < s for n, s in zip(neighbor_pos, shape)):
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(0)
            
            elif boundary == BoundaryCondition.REFLECTIVE:
                # Reflect at boundaries
                reflected_pos = tuple(
                    min(max(0, n), s-1) for n, s in zip(neighbor_pos, shape)
                )
                neighbors.append(grid[reflected_pos])
            
            elif boundary == BoundaryCondition.ABSORBING:
                # Boundaries absorb (stay same state)
                if all(0 <= n < s for n, s in zip(neighbor_pos, shape)):
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(grid[position])
        
        return np.array(neighbors)


class VonNeumannNeighborhood(Neighborhood):
    """Von Neumann neighborhood (4 neighbors in 2D, 6 in 3D)."""
    
    def __init__(self, dimensions: int = 2, radius: int = 1):
        """Initialize Von Neumann neighborhood.
        
        Args:
            dimensions: Number of dimensions (1, 2, or 3)
            radius: Manhattan distance for neighborhood
        """
        self.dimensions = dimensions
        self.radius = radius
        self._offsets = self._generate_offsets()
    
    def _generate_offsets(self) -> List[Tuple[int, ...]]:
        """Generate offset positions for Von Neumann neighborhood."""
        offsets = []
        
        if self.dimensions == 1:
            for i in range(-self.radius, self.radius + 1):
                if i != 0:
                    offsets.append((i,))
        
        elif self.dimensions == 2:
            for dist in range(1, self.radius + 1):
                offsets.extend([
                    (-dist, 0), (dist, 0),
                    (0, -dist), (0, dist)
                ])
            
            # For radius > 1, add intermediate points
            if self.radius > 1:
                for i in range(-self.radius, self.radius + 1):
                    for j in range(-self.radius, self.radius + 1):
                        if abs(i) + abs(j) <= self.radius and (i != 0 or j != 0):
                            if (i, j) not in offsets:
                                offsets.append((i, j))
        
        elif self.dimensions == 3:
            for dist in range(1, self.radius + 1):
                offsets.extend([
                    (-dist, 0, 0), (dist, 0, 0),
                    (0, -dist, 0), (0, dist, 0),
                    (0, 0, -dist), (0, 0, dist)
                ])
            
            # For radius > 1, add intermediate points
            if self.radius > 1:
                for i in range(-self.radius, self.radius + 1):
                    for j in range(-self.radius, self.radius + 1):
                        for k in range(-self.radius, self.radius + 1):
                            if abs(i) + abs(j) + abs(k) <= self.radius and \
                               (i != 0 or j != 0 or k != 0):
                                if (i, j, k) not in offsets:
                                    offsets.append((i, j, k))
        
        else:
            raise ValueError(f"Unsupported dimensions: {self.dimensions}")
        
        return offsets
    
    def get_offsets(self) -> List[Tuple[int, ...]]:
        """Get relative positions of neighbors."""
        return self._offsets.copy()
    
    def get_neighbors(self, grid: np.ndarray, position: Tuple[int, ...], 
                     boundary: 'BoundaryCondition') -> np.ndarray:
        """Get neighbor states using Von Neumann neighborhood."""
        from .automaton import BoundaryCondition
        
        neighbors = []
        shape = grid.shape
        
        for offset in self._offsets:
            neighbor_pos = tuple(p + o for p, o in zip(position, offset))
            
            if boundary == BoundaryCondition.PERIODIC:
                neighbor_pos = tuple(n % s for n, s in zip(neighbor_pos, shape))
                neighbors.append(grid[neighbor_pos])
            
            elif boundary == BoundaryCondition.FIXED:
                if all(0 <= n < s for n, s in zip(neighbor_pos, shape)):
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(0)
            
            elif boundary == BoundaryCondition.REFLECTIVE:
                reflected_pos = tuple(
                    min(max(0, n), s-1) for n, s in zip(neighbor_pos, shape)
                )
                neighbors.append(grid[reflected_pos])
            
            elif boundary == BoundaryCondition.ABSORBING:
                if all(0 <= n < s for n, s in zip(neighbor_pos, shape)):
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(grid[position])
        
        return np.array(neighbors)


class HexagonalNeighborhood(Neighborhood):
    """Hexagonal grid neighborhood (6 neighbors)."""
    
    def __init__(self):
        """Initialize hexagonal neighborhood."""
        self.dimensions = 2
        self._offsets = self._generate_offsets()
    
    def _generate_offsets(self) -> List[Tuple[int, int]]:
        """Generate offset positions for hexagonal neighborhood."""
        # Hex neighbors depend on whether row is even or odd
        # These are for pointy-top hexagons
        return [
            (-1, 0),  # Top
            (1, 0),   # Bottom
            (0, -1),  # Top-left
            (0, 1),   # Top-right
            (1, -1),  # Bottom-left
            (1, 1)    # Bottom-right
        ]
    
    def get_offsets(self) -> List[Tuple[int, int]]:
        """Get relative positions of neighbors."""
        return self._offsets.copy()
    
    def get_neighbors(self, grid: np.ndarray, position: Tuple[int, int], 
                     boundary: 'BoundaryCondition') -> np.ndarray:
        """Get neighbor states using hexagonal neighborhood."""
        from .automaton import BoundaryCondition
        
        row, col = position
        neighbors = []
        shape = grid.shape
        
        # Adjust offsets based on row parity for proper hex grid
        if row % 2 == 0:
            # Even row
            actual_offsets = [
                (-1, 0),   # Top
                (1, 0),    # Bottom
                (0, -1),   # Left
                (0, 1),    # Right
                (-1, -1),  # Top-left
                (1, -1)    # Bottom-left
            ]
        else:
            # Odd row
            actual_offsets = [
                (-1, 0),   # Top
                (1, 0),    # Bottom  
                (0, -1),   # Left
                (0, 1),    # Right
                (-1, 1),   # Top-right
                (1, 1)     # Bottom-right
            ]
        
        for offset in actual_offsets:
            neighbor_pos = (row + offset[0], col + offset[1])
            
            if boundary == BoundaryCondition.PERIODIC:
                neighbor_pos = (neighbor_pos[0] % shape[0], neighbor_pos[1] % shape[1])
                neighbors.append(grid[neighbor_pos])
            
            elif boundary == BoundaryCondition.FIXED:
                if 0 <= neighbor_pos[0] < shape[0] and 0 <= neighbor_pos[1] < shape[1]:
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(0)
            
            elif boundary == BoundaryCondition.REFLECTIVE:
                reflected_pos = (
                    min(max(0, neighbor_pos[0]), shape[0]-1),
                    min(max(0, neighbor_pos[1]), shape[1]-1)
                )
                neighbors.append(grid[reflected_pos])
            
            elif boundary == BoundaryCondition.ABSORBING:
                if 0 <= neighbor_pos[0] < shape[0] and 0 <= neighbor_pos[1] < shape[1]:
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(grid[position])
        
        return np.array(neighbors)


class ExtendedNeighborhood(Neighborhood):
    """Custom neighborhood with specified offsets."""
    
    def __init__(self, offsets: List[Tuple[int, ...]]):
        """Initialize extended neighborhood with custom offsets.
        
        Args:
            offsets: List of offset tuples defining the neighborhood
        """
        self.offsets = offsets
        self.dimensions = len(offsets[0]) if offsets else 2
    
    def get_offsets(self) -> List[Tuple[int, ...]]:
        """Get relative positions of neighbors."""
        return self.offsets.copy()
    
    def get_neighbors(self, grid: np.ndarray, position: Tuple[int, ...], 
                     boundary: 'BoundaryCondition') -> np.ndarray:
        """Get neighbor states using custom neighborhood."""
        from .automaton import BoundaryCondition
        
        neighbors = []
        shape = grid.shape
        
        for offset in self.offsets:
            neighbor_pos = tuple(p + o for p, o in zip(position, offset))
            
            if boundary == BoundaryCondition.PERIODIC:
                neighbor_pos = tuple(n % s for n, s in zip(neighbor_pos, shape))
                neighbors.append(grid[neighbor_pos])
            
            elif boundary == BoundaryCondition.FIXED:
                if all(0 <= n < s for n, s in zip(neighbor_pos, shape)):
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(0)
            
            elif boundary == BoundaryCondition.REFLECTIVE:
                reflected_pos = tuple(
                    min(max(0, n), s-1) for n, s in zip(neighbor_pos, shape)
                )
                neighbors.append(grid[reflected_pos])
            
            elif boundary == BoundaryCondition.ABSORBING:
                if all(0 <= n < s for n, s in zip(neighbor_pos, shape)):
                    neighbors.append(grid[neighbor_pos])
                else:
                    neighbors.append(grid[position])
        
        return np.array(neighbors)