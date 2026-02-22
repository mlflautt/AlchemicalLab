"""Base cellular automaton implementation."""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from abc import ABC, abstractmethod
import copy
from enum import Enum


class BoundaryCondition(Enum):
    """Boundary condition types for cellular automata."""
    PERIODIC = "periodic"  # Toroidal topology
    FIXED = "fixed"  # Fixed boundary values
    REFLECTIVE = "reflective"  # Mirror at boundaries
    ABSORBING = "absorbing"  # Boundaries absorb activity


class CAState:
    """Represents the state of a cellular automaton at a given time."""
    
    def __init__(self, grid: np.ndarray, generation: int = 0, metadata: Optional[Dict[str, Any]] = None):
        """Initialize CA state.
        
        Args:
            grid: The state grid (can be 1D, 2D, or 3D)
            generation: Current generation number
            metadata: Optional metadata about the state
        """
        self.grid = grid.copy()
        self.generation = generation
        self.metadata = metadata or {}
        self.shape = grid.shape
        self.dimensions = len(grid.shape)
        
    def copy(self) -> 'CAState':
        """Create a deep copy of the state."""
        return CAState(
            grid=self.grid.copy(),
            generation=self.generation,
            metadata=copy.deepcopy(self.metadata)
        )
    
    def __eq__(self, other: 'CAState') -> bool:
        """Check equality with another state."""
        if not isinstance(other, CAState):
            return False
        return (
            np.array_equal(self.grid, other.grid) and
            self.generation == other.generation
        )
    
    def __hash__(self) -> int:
        """Hash the state for use in sets/dicts."""
        return hash((self.grid.tobytes(), self.generation))


class CellularAutomaton(ABC):
    """Abstract base class for cellular automata."""
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        states: int = 2,
        boundary: BoundaryCondition = BoundaryCondition.PERIODIC,
        neighborhood: Optional['Neighborhood'] = None,
        rule: Optional['Rule'] = None
    ):
        """Initialize cellular automaton.
        
        Args:
            shape: Shape of the CA grid
            states: Number of possible states per cell
            boundary: Boundary condition type
            neighborhood: Neighborhood definition
            rule: Rule for state transitions
        """
        self.shape = shape
        self.states = states
        self.boundary = boundary
        self.neighborhood = neighborhood
        self.rule = rule
        self.dimensions = len(shape)
        
        # Initialize grid with zeros
        self.grid = np.zeros(shape, dtype=np.int32)
        self.next_grid = np.zeros(shape, dtype=np.int32)
        
        # Track history
        self.generation = 0
        self.history: List[CAState] = []
        self.track_history = False
        
        # Statistics
        self.stats: Dict[str, Any] = {}
        
    @abstractmethod
    def step(self) -> None:
        """Perform one generation step."""
        pass
    
    @abstractmethod
    def apply_rule(self, neighbors: np.ndarray, current_state: int) -> int:
        """Apply the CA rule to determine next state.
        
        Args:
            neighbors: Array of neighbor states
            current_state: Current state of the cell
            
        Returns:
            Next state of the cell
        """
        pass
    
    def get_neighbors(self, position: Tuple[int, ...]) -> np.ndarray:
        """Get neighbors for a given position.
        
        Args:
            position: Position in the grid
            
        Returns:
            Array of neighbor states
        """
        if self.neighborhood is None:
            raise ValueError("No neighborhood defined")
        return self.neighborhood.get_neighbors(self.grid, position, self.boundary)
    
    def evolve(self, generations: int) -> List[CAState]:
        """Evolve the CA for multiple generations.
        
        Args:
            generations: Number of generations to evolve
            
        Returns:
            List of states if tracking history, else empty list
        """
        states = []
        for _ in range(generations):
            if self.track_history:
                states.append(self.get_state())
            self.step()
        return states
    
    def reset(self) -> None:
        """Reset the CA to initial state."""
        self.grid.fill(0)
        self.generation = 0
        self.history.clear()
        self.stats.clear()
    
    def randomize(self, density: float = 0.5, states: Optional[List[int]] = None) -> None:
        """Randomize the grid.
        
        Args:
            density: Density of non-zero states (for binary CA)
            states: List of possible states to use
        """
        if states is None:
            if self.states == 2:
                self.grid = np.random.choice([0, 1], size=self.shape, p=[1-density, density])
            else:
                self.grid = np.random.randint(0, self.states, size=self.shape)
        else:
            self.grid = np.random.choice(states, size=self.shape)
        self.generation = 0
    
    def set_pattern(self, pattern: np.ndarray, position: Optional[Tuple[int, ...]] = None) -> None:
        """Place a pattern in the grid.
        
        Args:
            pattern: Pattern array
            position: Position to place pattern (center if None)
        """
        if position is None:
            # Center the pattern
            position = tuple((g - p) // 2 for g, p in zip(self.shape, pattern.shape))
        
        # Calculate slices for pattern placement
        slices = tuple(slice(pos, min(pos + psize, gsize)) 
                      for pos, psize, gsize in zip(position, pattern.shape, self.shape))
        pattern_slices = tuple(slice(0, s.stop - s.start) for s in slices)
        
        self.grid[slices] = pattern[pattern_slices]
    
    def get_state(self) -> CAState:
        """Get current state."""
        return CAState(
            grid=self.grid.copy(),
            generation=self.generation,
            metadata=self.get_metadata()
        )
    
    def set_state(self, state: CAState) -> None:
        """Set the CA to a specific state."""
        self.grid = state.grid.copy()
        self.generation = state.generation
        if hasattr(state, 'metadata'):
            self.stats.update(state.metadata)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about current state."""
        return {
            'shape': self.shape,
            'states': self.states,
            'boundary': self.boundary.value,
            'generation': self.generation,
            'population': np.sum(self.grid > 0),
            'density': np.mean(self.grid > 0),
            **self.stats
        }
    
    def find_patterns(self, pattern: np.ndarray) -> List[Tuple[int, ...]]:
        """Find occurrences of a pattern in the grid.
        
        Args:
            pattern: Pattern to search for
            
        Returns:
            List of positions where pattern occurs
        """
        from scipy.signal import correlate
        
        if self.dimensions != len(pattern.shape):
            raise ValueError("Pattern dimensions must match grid dimensions")
        
        # Use correlation to find pattern
        corr = correlate(self.grid, pattern, mode='valid')
        pattern_sum = np.sum(pattern)
        matches = np.where(corr == pattern_sum)
        
        return list(zip(*matches))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about current state."""
        stats = {
            'generation': self.generation,
            'population': int(np.sum(self.grid > 0)),
            'density': float(np.mean(self.grid > 0)),
            'entropy': self._calculate_entropy(),
            'state_distribution': self._get_state_distribution()
        }
        return stats
    
    def _calculate_entropy(self) -> float:
        """Calculate Shannon entropy of the current state."""
        from scipy.stats import entropy
        counts = np.bincount(self.grid.flatten(), minlength=self.states)
        probs = counts / counts.sum()
        return float(entropy(probs, base=2))
    
    def _get_state_distribution(self) -> Dict[int, float]:
        """Get distribution of states in the grid."""
        counts = np.bincount(self.grid.flatten(), minlength=self.states)
        total = counts.sum()
        return {i: float(c/total) for i, c in enumerate(counts)}
    
    def save(self, filename: str) -> None:
        """Save CA state to file."""
        np.savez_compressed(
            filename,
            grid=self.grid,
            generation=self.generation,
            shape=self.shape,
            states=self.states,
            boundary=self.boundary.value,
            metadata=self.get_metadata()
        )
    
    def load(self, filename: str) -> None:
        """Load CA state from file."""
        data = np.load(filename, allow_pickle=True)
        self.grid = data['grid']
        self.generation = int(data['generation'])
        self.shape = tuple(data['shape'])
        self.states = int(data['states'])
        if 'metadata' in data:
            self.stats.update(data['metadata'].item())