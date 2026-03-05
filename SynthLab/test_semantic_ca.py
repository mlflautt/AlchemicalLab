"""
Simplified test version of Semantic CA for immediate testing
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

class SimpleSemanticCA:
    """Simplified semantic CA for immediate testing."""
    
    def __init__(self, size=(50, 50), seed=42):
        self.size = size
        self.key = random.PRNGKey(seed)
        h, w = size
        
        # Initialize with simpler state
        keys = random.split(self.key, 5)
        self.state = {
            'alive': random.bernoulli(keys[0], 0.3, (h, w)),
            'species': random.randint(keys[1], (h, w), 0, 5),
            'energy': random.uniform(keys[2], (h, w), minval=0.0, maxval=1.0),
            'color': random.uniform(keys[3], (h, w, 3), minval=0.0, maxval=1.0),
            'resources': random.uniform(keys[4], (h, w), minval=0.0, maxval=1.0)
        }
    
    def get_neighbors(self, grid, i, j):
        """Get 3x3 neighborhood with boundary handling."""
        h, w = grid.shape
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = (i + di) % h, (j + dj) % w
                neighbors.append(grid[ni, nj])
        return jnp.array(neighbors)
    
    def apply_rules(self):
        """Apply CA rules (simplified version)."""
        h, w = self.size
        new_state = {}
        
        # Initialize new state
        for key in self.state:
            new_state[key] = jnp.zeros_like(self.state[key])
        
        # Apply rules cell by cell
        for i in range(h):
            for j in range(w):
                # Get neighborhood info
                alive_neighbors = self.get_neighbors(self.state['alive'], i, j)
                alive_count = jnp.sum(alive_neighbors) - self.state['alive'][i, j]
                
                current_alive = self.state['alive'][i, j]
                current_energy = self.state['energy'][i, j]
                
                # Game of Life + Energy rules
                if current_alive:
                    # Survival: 2-3 neighbors and energy > 0.2
                    survives = (alive_count >= 2) & (alive_count <= 3) & (current_energy > 0.2)
                    new_energy = jnp.maximum(0.0, current_energy - 0.01)  # Energy decay
                else:
                    # Birth: exactly 3 neighbors
                    survives = alive_count == 3
                    new_energy = 0.5 if survives else 0.0
                
                # Update state
                new_state['alive'] = new_state['alive'].at[i, j].set(survives)
                new_state['energy'] = new_state['energy'].at[i, j].set(new_energy)
                
                # Keep other properties mostly stable with small changes
                new_state['species'] = new_state['species'].at[i, j].set(self.state['species'][i, j])
                new_state['resources'] = new_state['resources'].at[i, j].set(
                    jnp.maximum(0.0, self.state['resources'][i, j] - 0.001)
                )
                
                # Color evolution based on species and energy
                if survives:
                    species_color = self.state['species'][i, j] / 5.0  # Normalize species to [0,1]
                    energy_factor = new_energy
                    color = jnp.array([species_color, energy_factor, 0.5])
                else:
                    color = jnp.array([0.0, 0.0, 0.0])  # Dead = black
                
                new_state['color'] = new_state['color'].at[i, j].set(color)
        
        self.state = new_state
    
    def step(self):
        """Execute one simulation step."""
        self.apply_rules()
        return self.get_stats()
    
    def get_stats(self):
        """Get current simulation statistics."""
        return {
            'alive_count': int(jnp.sum(self.state['alive'])),
            'avg_energy': float(jnp.mean(self.state['energy'])),
            'species_diversity': int(len(jnp.unique(self.state['species']))),
            'total_resources': float(jnp.sum(self.state['resources']))
        }
    
    def get_visualization(self):
        """Get visualization data."""
        return {
            'alive': np.array(self.state['alive']),
            'species': np.array(self.state['species']),
            'energy': np.array(self.state['energy']),
            'color': np.array(self.state['color']),
            'resources': np.array(self.state['resources'])
        }

def run_simulation():
    """Run a test simulation."""
    print("🧬 AlchemicalLab Semantic CA Test")
    print("=" * 50)
    
    # Create world
    world = SimpleSemanticCA(size=(50, 50), seed=42)
    
    # Run simulation
    print("Initial state:")
    stats = world.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nRunning simulation steps...")
    for step in range(20):
        stats = world.step()
        
        if step % 5 == 0 or step < 5:  # Show frequent updates early, then sparse
            print(f"Step {step:2d}: Alive={stats['alive_count']:3d}, "
                  f"Energy={stats['avg_energy']:.3f}, "
                  f"Species={stats['species_diversity']:2d}, "
                  f"Resources={stats['total_resources']:.2f}")
    
    # Final visualization data
    viz = world.get_visualization()
    print(f"\nFinal grid size: {viz['alive'].shape}")
    print(f"Alive cells: {np.sum(viz['alive'])}")
    
    # Show a small sample of the grid
    print("\nSample of alive grid (10x10 corner):")
    sample = viz['alive'][:10, :10].astype(int)
    for row in sample:
        print(''.join(['█' if cell else '░' for cell in row]))
    
    print("\n🎮 Basic semantic CA working! Ready for game development...")
    return world

if __name__ == "__main__":
    world = run_simulation()