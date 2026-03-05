"""
GPU-Optimized Semantic CA Test

Properly compiled JAX functions for GPU acceleration of cellular automaton simulations.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import numpy as np
import time

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Ensure we're using GPU if available
if len(jax.devices('gpu')) > 0:
    print(f"GPU devices: {jax.devices('gpu')}")
else:
    print("No GPU devices found")

@jit
def get_neighbors(grid, i, j):
    """Get 3x3 neighborhood around cell (i,j) with wrapping - GPU optimized."""
    h, w = grid.shape
    
    # Create neighborhood indices with wrapping
    rows = jnp.array([(i-1) % h, i, (i+1) % h])
    cols = jnp.array([(j-1) % w, j, (j+1) % w])
    
    # Extract neighborhood
    return grid[jnp.ix_(rows, cols)]

@jit
def count_alive_neighbors(neighborhood, center_alive):
    """Count alive neighbors excluding center cell."""
    return jnp.sum(neighborhood) - center_alive

@jit
def apply_ca_rules_single_cell(alive_grid, energy_grid, species_grid, i, j):
    """Apply CA rules to a single cell - GPU compiled."""
    current_alive = alive_grid[i, j]
    current_energy = energy_grid[i, j]
    current_species = species_grid[i, j]
    
    # Get neighborhood
    alive_neighbors = get_neighbors(alive_grid, i, j)
    neighbor_count = count_alive_neighbors(alive_neighbors, current_alive)
    
    # Game of Life + Energy rules
    new_alive = jnp.where(
        current_alive,
        # Alive cells: survive with 2-3 neighbors and energy > 0.2
        (neighbor_count >= 2) & (neighbor_count <= 3) & (current_energy > 0.2),
        # Dead cells: born with exactly 3 neighbors
        neighbor_count == 3
    )
    
    # Energy update
    new_energy = jnp.where(
        new_alive,
        jnp.where(current_alive, 
                  jnp.maximum(0.0, current_energy - 0.01),  # Decay for living cells
                  0.5),  # Energy for new cells
        0.0  # No energy for dead cells
    )
    
    return new_alive, new_energy, current_species

# Vectorize over entire grid - this is where GPU parallelism shines
@jit
def update_ca_grid(alive_grid, energy_grid, species_grid):
    """Update entire CA grid in parallel on GPU."""
    h, w = alive_grid.shape
    
    # Create coordinate grids
    i_coords, j_coords = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    
    # Vectorized application of rules
    new_alive = jnp.zeros_like(alive_grid)
    new_energy = jnp.zeros_like(energy_grid)
    new_species = species_grid.copy()  # Species doesn't change in this simple version
    
    # Apply rules to all cells in parallel
    for i in range(h):
        for j in range(w):
            alive, energy, species = apply_ca_rules_single_cell(
                alive_grid, energy_grid, species_grid, i, j
            )
            new_alive = new_alive.at[i, j].set(alive)
            new_energy = new_energy.at[i, j].set(energy)
    
    return new_alive, new_energy, new_species

# Even better: fully vectorized version
@jit 
def update_ca_grid_vectorized(alive_grid, energy_grid, species_grid):
    """Fully vectorized CA update - maximum GPU utilization."""
    h, w = alive_grid.shape
    
    # Pad grid for neighborhood calculations
    alive_padded = jnp.pad(alive_grid, 1, mode='wrap')
    
    # Calculate all neighborhoods at once using convolution
    kernel = jnp.ones((3, 3))
    kernel = kernel.at[1, 1].set(0)  # Don't count center cell
    
    # Convolve to count neighbors
    neighbor_counts = lax.conv_general_dilated(
        alive_padded[None, None, :, :].astype(jnp.float32),
        kernel[None, None, :, :],
        window_strides=(1, 1),
        padding='VALID'
    )[0, 0]
    
    # Apply Game of Life rules
    birth = (neighbor_counts == 3) & ~alive_grid
    survival = alive_grid & (neighbor_counts >= 2) & (neighbor_counts <= 3) & (energy_grid > 0.2)
    new_alive = birth | survival
    
    # Update energy
    new_energy = jnp.where(
        new_alive,
        jnp.where(alive_grid,
                  jnp.maximum(0.0, energy_grid - 0.01),  # Decay
                  0.5),  # New cells
        0.0  # Dead cells
    )
    
    return new_alive, new_energy, species_grid

class GPUSemanticCA:
    """GPU-accelerated semantic cellular automaton."""
    
    def __init__(self, size=(1000, 1000), seed=42):
        self.size = size
        self.key = random.PRNGKey(seed)
        h, w = size
        
        print(f"Initializing GPU CA with grid size: {h}x{w}")
        
        # Initialize on GPU
        keys = random.split(self.key, 4)
        
        with jax.default_device(jax.devices()[0]):  # Use first available device
            self.alive = random.bernoulli(keys[0], 0.3, (h, w))
            self.energy = random.uniform(keys[1], (h, w), minval=0.0, maxval=1.0)
            self.species = random.randint(keys[2], (h, w), 0, 5)
            self.resources = random.uniform(keys[3], (h, w), minval=0.0, maxval=1.0)
        
        print(f"Grid initialized on device: {self.alive.device()}")
    
    def step(self):
        """Execute one simulation step on GPU."""
        # Use the vectorized version for maximum GPU utilization
        self.alive, self.energy, self.species = update_ca_grid_vectorized(
            self.alive, self.energy, self.species
        )
    
    def get_stats(self):
        """Get simulation statistics."""
        return {
            'alive_count': int(jnp.sum(self.alive)),
            'avg_energy': float(jnp.mean(self.energy)),
            'total_energy': float(jnp.sum(self.energy)),
            'species_diversity': int(len(jnp.unique(self.species)))
        }

def benchmark_gpu_ca():
    """Benchmark GPU-accelerated cellular automaton."""
    print("\\nGPU Cellular Automaton Benchmark")
    print("=" * 40)
    
    # Test different grid sizes
    sizes = [(100, 100), (500, 500), (1000, 1000)]
    
    for size in sizes:
        print(f"\\nTesting {size[0]}x{size[1]} grid...")
        
        try:
            ca = GPUSemanticCA(size=size, seed=42)
            
            # Warmup - compile JIT functions
            print("  Warming up JIT compilation...")
            ca.step()
            
            # Benchmark
            steps = 10
            start_time = time.time()
            
            for step in range(steps):
                ca.step()
                if step == 0:
                    first_step_stats = ca.get_stats()
                    print(f"    Initial: {first_step_stats}")
            
            end_time = time.time()
            final_stats = ca.get_stats()
            
            total_time = end_time - start_time
            steps_per_second = steps / total_time
            cells_per_second = (size[0] * size[1] * steps) / total_time
            
            print(f"    Final: {final_stats}")
            print(f"    Performance: {steps_per_second:.1f} steps/sec, {cells_per_second/1e6:.1f}M cells/sec")
            
        except Exception as e:
            print(f"    Error: {e}")

def test_gpu_availability():
    """Test basic GPU functionality."""
    print("GPU Availability Test")
    print("=" * 20)
    
    # Test basic GPU operations
    try:
        # Create arrays on GPU
        key = random.PRNGKey(42)
        x = random.normal(key, (1000, 1000))
        
        print(f"Array device: {x.device()}")
        
        # Test computation
        y = jnp.dot(x, x.T)
        print(f"Computation completed, result shape: {y.shape}")
        print("GPU test successful!")
        
        return True
        
    except Exception as e:
        print(f"GPU test failed: {e}")
        return False

if __name__ == "__main__":
    print("AlchemicalLab GPU Test")
    print("=" * 30)
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    
    if gpu_available:
        # Run CA benchmark
        benchmark_gpu_ca()
    else:
        print("GPU not available - fix JAX CUDA installation")