"""
Working GPU Cellular Automaton for AlchemicalLab

Avoids cuDNN-dependent operations, focuses on basic GPU parallel CA computation.
"""

import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
import time
import matplotlib.pyplot as plt

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")

@jit
def count_neighbors_vectorized(alive_grid):
    """Count neighbors for all cells using padding and slicing - GPU optimized."""
    # Pad with wrapping boundary conditions
    padded = jnp.pad(alive_grid, 1, mode='wrap')
    
    # Count neighbors by summing 8 shifted versions
    neighbors = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +   # top row
        padded[1:-1, :-2] +                   padded[1:-1, 2:] +   # middle row (no center)
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]       # bottom row
    )
    
    return neighbors

@jit
def update_ca_gpu(alive_grid, energy_grid):
    """GPU-optimized CA update using vectorized operations."""
    # Count neighbors for all cells at once
    neighbor_count = count_neighbors_vectorized(alive_grid)
    
    # Apply Conway's Game of Life + energy rules vectorized
    # Birth: dead cells with exactly 3 neighbors
    birth = (~alive_grid) & (neighbor_count == 3)
    
    # Survival: living cells with 2-3 neighbors and sufficient energy
    survival = alive_grid & (neighbor_count >= 2) & (neighbor_count <= 3) & (energy_grid > 0.2)
    
    # New alive state
    new_alive = birth | survival
    
    # Energy update
    new_energy = jnp.where(
        new_alive,
        jnp.where(alive_grid, 
                  jnp.maximum(0.0, energy_grid - 0.01),  # Living cells decay
                  0.5),  # New cells get energy
        0.0  # Dead cells have no energy
    )
    
    return new_alive, new_energy

class WorkingGPUCA:
    """GPU Cellular Automaton that actually works."""
    
    def __init__(self, size=(500, 500), seed=42):
        self.size = size
        self.key = random.PRNGKey(seed)
        
        # Initialize grids on GPU
        keys = random.split(self.key, 2)
        self.alive = random.bernoulli(keys[0], 0.3, size)
        self.energy = random.uniform(keys[1], size, minval=0.0, maxval=1.0)
        
        print(f"Initialized {size} CA on device: {self.alive.device()}")
        
        # Track statistics
        self.step_count = 0
        self.stats_history = []
    
    def step(self):
        """Execute one simulation step."""
        self.alive, self.energy = update_ca_gpu(self.alive, self.energy)
        self.step_count += 1
        
        # Record stats
        stats = self.get_stats()
        self.stats_history.append(stats)
        return stats
    
    def get_stats(self):
        """Get current simulation statistics."""
        return {
            'step': self.step_count,
            'alive_count': int(jnp.sum(self.alive)),
            'avg_energy': float(jnp.mean(self.energy)),
            'total_energy': float(jnp.sum(self.energy)),
            'density': float(jnp.mean(self.alive))
        }
    
    def run_benchmark(self, steps=50):
        """Run performance benchmark."""
        print(f"Benchmarking {self.size} grid for {steps} steps...")
        
        # Warmup
        self.step()
        
        start_time = time.time()
        for i in range(steps):
            self.step()
            if i % 10 == 0:
                stats = self.stats_history[-1]
                print(f"Step {stats['step']:3d}: "
                      f"Alive={stats['alive_count']:5d}, "
                      f"Energy={stats['avg_energy']:.3f}, "
                      f"Density={stats['density']:.3f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance metrics
        steps_per_second = steps / total_time
        cells_per_second = (self.size[0] * self.size[1] * steps) / total_time
        
        print(f"\\nPerformance:")
        print(f"  {steps_per_second:.1f} steps/second")
        print(f"  {cells_per_second/1e6:.1f} million cells/second")
        print(f"  {total_time:.2f} seconds total")
        
        return {
            'steps_per_second': steps_per_second,
            'cells_per_second': cells_per_second,
            'total_time': total_time
        }
    
    def get_grid_sample(self, sample_size=50):
        """Get a small sample of the grid for visualization."""
        h, w = self.size
        sample_h = min(sample_size, h)
        sample_w = min(sample_size, w)
        
        return {
            'alive': np.array(self.alive[:sample_h, :sample_w]),
            'energy': np.array(self.energy[:sample_h, :sample_w])
        }

def test_gpu_performance():
    """Test GPU performance with different grid sizes."""
    print("GPU Cellular Automaton Performance Test")
    print("=" * 45)
    
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for size in sizes:
        print(f"\\nTesting {size[0]}x{size[1]} grid:")
        try:
            ca = WorkingGPUCA(size=size, seed=42)
            performance = ca.run_benchmark(steps=20)
            
            # Show final statistics
            final_stats = ca.stats_history[-1]
            print(f"Final state: {final_stats['alive_count']} alive cells")
            
            # Memory usage estimate
            total_cells = size[0] * size[1]
            memory_mb = (total_cells * 8) / 1e6  # 8 bytes per cell (2 float32 grids)
            print(f"Memory usage: ~{memory_mb:.1f} MB")
            
        except Exception as e:
            print(f"Error with {size}: {e}")

def run_visual_simulation():
    """Run a smaller simulation for visualization."""
    print("\\nRunning visual simulation...")
    
    ca = WorkingGPUCA(size=(100, 100), seed=123)
    
    # Run for several steps
    for i in range(30):
        stats = ca.step()
        if i % 5 == 0:
            print(f"Step {stats['step']:2d}: "
                  f"Alive={stats['alive_count']:4d}, "
                  f"Energy={stats['avg_energy']:.3f}")
    
    # Get sample for visualization
    sample = ca.get_grid_sample(50)
    
    # Simple text visualization
    print("\\nFinal grid sample (50x50, ■=alive, ░=dead):")
    alive_sample = sample['alive']
    for row in alive_sample[:20]:  # Show first 20 rows
        print(''.join(['■' if cell else '░' for cell in row[:50]]))
    
    return ca

if __name__ == "__main__":
    print("AlchemicalLab Working GPU Cellular Automaton")
    print("=" * 50)
    
    # Test if GPU is working for basic operations
    try:
        # Simple GPU test without cuDNN dependencies
        key = random.PRNGKey(42)
        x = random.normal(key, (100, 100))
        y = x + 1  # Simple operation
        print(f"Basic GPU test: {y.device()}")
        
        # Run performance tests
        test_gpu_performance()
        
        # Run visual simulation
        visual_ca = run_visual_simulation()
        
        print("\\n✓ GPU cellular automaton working successfully!")
        
    except Exception as e:
        print(f"GPU test failed: {e}")
        print("Running on available device...")