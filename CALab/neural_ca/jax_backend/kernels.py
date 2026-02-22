"""
JAX Backend for Neural Cellular Automata
========================================

GPU-accelerated operations and utilities for neural CA computation.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.scipy import ndimage
import jax.lax as lax
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np


class JAXBackend:
    """JAX-based backend for neural CA operations."""

    def __init__(self, seed: int = 42):
        self.key = random.PRNGKey(seed)

    def init_weights(self, shapes: List[Tuple[int, ...]]) -> List[jnp.ndarray]:
        """Initialize neural network weights."""
        weights = []
        for shape in shapes:
            self.key, subkey = random.split(self.key)
            w = random.normal(subkey, shape) * 0.1
            weights.append(w)
        return weights

    @jit
    def conv2d(self, x: jnp.ndarray, kernel: jnp.ndarray,
               strides: Tuple[int, int] = (1, 1), padding: str = 'SAME') -> jnp.ndarray:
        """2D convolution operation."""
        return lax.conv_general_dilated(
            x[None, ...], kernel[None, None, ...],
            window_strides=strides,
            padding=padding.upper()
        )[0]

    @jit
    def max_pool2d(self, x: jnp.ndarray, pool_size: Tuple[int, int] = (2, 2),
                   strides: Tuple[int, int] = (2, 2)) -> jnp.ndarray:
        """2D max pooling."""
        return lax.reduce_window(
            x, -jnp.inf, lax.max,
            (pool_size[0], pool_size[1], 1, 1),
            (strides[0], strides[1], 1, 1),
            padding='VALID'
        )

    @jit
    def batch_norm(self, x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                   running_mean: jnp.ndarray, running_var: jnp.ndarray,
                   eps: float = 1e-5) -> jnp.ndarray:
        """Batch normalization."""
        mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(x, axis=(0, 1, 2), keepdims=True)

        # Update running statistics (simplified)
        new_running_mean = 0.9 * running_mean + 0.1 * mean.squeeze()
        new_running_var = 0.9 * running_var + 0.1 * var.squeeze()

        x_norm = (x - mean) / jnp.sqrt(var + eps)
        return gamma[None, None, None, :] * x_norm + beta[None, None, None, :]

    @jit
    def attention(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        """Multi-head attention mechanism."""
        # Simplified single-head attention
        scores = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(query.shape[-1])
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(weights, value)

    def create_ca_step_fn(self, update_fn: Callable) -> Callable:
        """Create a JIT-compiled CA step function."""
        @jit
        def step_fn(state: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
            return update_fn(state, params)
        return step_fn

    def create_ca_evolution_fn(self, step_fn: Callable, n_steps: int) -> Callable:
        """Create a function that evolves CA for multiple steps."""
        def evolve_fn(initial_state: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
            state = initial_state
            for _ in range(n_steps):
                state = step_fn(state, params)
            return state
        return jit(evolve_fn)

    @jit
    def compute_fft_features(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute frequency domain features."""
        # 2D FFT
        fft = jnp.fft.fft2(x)

        # Magnitude spectrum
        magnitude = jnp.abs(fft)

        # Phase spectrum
        phase = jnp.angle(fft)

        # Return concatenated features
        return jnp.concatenate([magnitude, phase], axis=-1)

    @jit
    def compute_entropy(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute spatial entropy of the grid."""
        # Flatten and compute histogram
        flat = x.flatten()
        hist = jnp.histogram(flat, bins=32, range=(flat.min(), flat.max()))[0]
        hist = hist / jnp.sum(hist)  # Normalize

        # Compute entropy
        entropy = -jnp.sum(hist * jnp.log(hist + 1e-8))
        return entropy

    @jit
    def compute_complexity(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute complexity measures (compression-based)."""
        # Simple compression ratio estimate using entropy
        entropy = self.compute_entropy(x)

        # Theoretical minimum bits per value
        min_bits = entropy / jnp.log(2.0)

        # Actual bits (assuming float32)
        actual_bits = 32.0

        # Compression ratio
        return actual_bits / (min_bits + 1e-8)

    def create_parallel_evolution_fn(self, step_fn: Callable, n_steps: int) -> Callable:
        """Create parallel evolution for multiple initial conditions."""
        evolve_single = self.create_ca_evolution_fn(step_fn, n_steps)

        @vmap
        def evolve_batch(initial_states: jnp.ndarray, params: Dict[str, Any]) -> jnp.ndarray:
            return evolve_single(initial_states, params)

        return evolve_batch

    @jit
    def apply_boundary_conditions(self, x: jnp.ndarray, bc_type: str = 'periodic') -> jnp.ndarray:
        """Apply boundary conditions to the grid."""
        if bc_type == 'periodic':
            return x  # Already handled by padding in convolutions
        elif bc_type == 'dirichlet':
            # Zero boundary conditions
            return jnp.pad(x[1:-1, 1:-1], 1, mode='constant', constant_values=0)
        elif bc_type == 'neumann':
            # Reflective boundary conditions
            return jnp.pad(x[1:-1, 1:-1], 1, mode='reflect')
        else:
            return x

    def create_adaptive_step_fn(self, base_step_fn: Callable,
                               adaptation_fn: Callable) -> Callable:
        """Create adaptive step function that modifies parameters over time."""
        @jit
        def adaptive_step(state: jnp.ndarray, params: Dict[str, Any], step_count: int) -> jnp.ndarray:
            # Adapt parameters based on current step
            adapted_params = adaptation_fn(params, step_count)
            return base_step_fn(state, adapted_params)

        return adaptive_step

    def memory_efficient_scan(self, step_fn: Callable, initial_state: jnp.ndarray,
                             n_steps: int) -> jnp.ndarray:
        """Memory-efficient evolution using scan."""
        def scan_fn(carry, x):
            state = carry
            new_state = step_fn(state, {})
            return new_state, new_state

        final_state, trajectory = lax.scan(scan_fn, initial_state, None, length=n_steps)
        return trajectory


class GPUBenchmark:
    """GPU performance benchmarking for neural CA."""

    def __init__(self, backend: JAXBackend):
        self.backend = backend

    def benchmark_step_time(self, step_fn: Callable, state_shape: Tuple[int, ...],
                           n_steps: int = 100, n_runs: int = 10) -> Dict[str, float]:
        """Benchmark CA step execution time."""
        # Create random initial state
        key = random.PRNGKey(0)
        initial_state = random.normal(key, state_shape)

        # Warmup
        _ = step_fn(initial_state, {})

        # Benchmark
        times = []
        for _ in range(n_runs):
            state = initial_state
            start_time = jax.time.time()

            for _ in range(n_steps):
                state = step_fn(state, {})

            end_time = jax.time.time()
            times.append((end_time - start_time) / n_steps)

        return {
            'mean_step_time': np.mean(times),
            'std_step_time': np.std(times),
            'steps_per_second': 1.0 / np.mean(times),
            'total_time': np.sum(times)
        }

    def benchmark_memory_usage(self, step_fn: Callable, state_shape: Tuple[int, ...],
                              n_steps: int = 100) -> Dict[str, float]:
        """Benchmark memory usage during evolution."""
        # This is a simplified memory benchmark
        # In practice, you'd use more sophisticated profiling tools

        key = random.PRNGKey(0)
        initial_state = random.normal(key, state_shape)

        # Track memory usage (simplified)
        initial_memory = 0  # Would need actual memory profiling

        state = initial_state
        for _ in range(n_steps):
            state = step_fn(state, {})

        final_memory = 0  # Would need actual memory profiling

        return {
            'initial_memory_mb': initial_memory / (1024**2),
            'final_memory_mb': final_memory / (1024**2),
            'memory_increase_mb': (final_memory - initial_memory) / (1024**2)
        }

    def benchmark_scaling(self, step_fn: Callable, base_size: int = 32,
                         max_size: int = 256) -> List[Dict[str, Any]]:
        """Benchmark performance scaling with grid size."""
        results = []

        for size in [32, 64, 128, 256]:
            if size > max_size:
                break

            state_shape = (size, size, 16)
            benchmark = self.benchmark_step_time(step_fn, state_shape, n_steps=50, n_runs=3)

            results.append({
                'grid_size': size,
                'cells': size * size,
                'step_time': benchmark['mean_step_time'],
                'steps_per_second': benchmark['steps_per_second']
            })

        return results


# Utility functions
def get_device_info() -> Dict[str, Any]:
    """Get information about available JAX devices."""
    return {
        'platform': jax.default_backend(),
        'devices': jax.devices(),
        'device_count': len(jax.devices()),
        'host_id': jax.host_id(),
        'process_count': jax.process_count()
    }


def optimize_for_gpu(fn: Callable) -> Callable:
    """Apply GPU optimizations to a function."""
    # JIT compilation
    jitted_fn = jit(fn)

    # Additional optimizations can be added here
    # - Rematerialization for memory efficiency
    # - Kernel fusion
    # - etc.

    return jitted_fn


def create_custom_kernel(kernel_type: str = 'sobel') -> jnp.ndarray:
    """Create custom convolution kernels."""
    if kernel_type == 'sobel':
        # Sobel operator for edge detection
        sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=jnp.float32)
        sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=jnp.float32)
        return jnp.stack([sobel_x, sobel_y])

    elif kernel_type == 'laplacian':
        # Laplacian for curvature detection
        return jnp.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=jnp.float32)

    elif kernel_type == 'gaussian':
        # Gaussian blur
        return jnp.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=jnp.float32) / 16.0

    else:
        # Identity kernel
        return jnp.eye(3, dtype=jnp.float32)