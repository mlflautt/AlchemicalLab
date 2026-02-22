"""
Cellular Automata Benchmarking Suite
====================================

Comprehensive benchmarking framework for comparing CA systems across:
- Performance metrics (execution time, memory usage, GPU utilization)
- Emergence metrics (complexity, entropy, diversity, stability)
- Scalability across grid sizes and evolution steps

Supports benchmarking of:
- Evolutionary CA (Elementary, Totalistic, Neural genomes)
- Neural CA (NeuralCA, DiffLogicCA, UniversalNCA)
- Traditional CA implementations
"""

import time
import psutil
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import json
import pickle
from functools import partial
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import os
import warnings
warnings.filterwarnings('ignore')

# Import CA systems
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from evolutionary.genetic_ca import EvolutionaryCA, EvolutionaryConfig, FitnessFunctions
    EVO_AVAILABLE = True
except ImportError as e:
    EVO_AVAILABLE = False
    print(f"Warning: Evolutionary CA not available: {e}")

try:
    from neural_ca.models.nca import NeuralCA, DiffLogicCA, UniversalNCA, CAConfig, NCATrainer
    NEURAL_AVAILABLE = True
except ImportError as e:
    NEURAL_AVAILABLE = False
    print(f"Warning: Neural CA not available: {e}")

try:
    from visualization.ca_visualizer import CAVisualizer, VisualizationConfig
    VIS_AVAILABLE = True
except ImportError as e:
    VIS_AVAILABLE = False
    print(f"Warning: Visualization not available: {e}")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    grid_sizes: Optional[List[Tuple[int, int]]] = None
    n_steps_range: Optional[List[int]] = None
    n_trials: int = 5
    warmup_trials: int = 2
    enable_gpu: bool = True
    memory_tracking: bool = True
    output_dir: str = 'benchmark_results'
    save_results: bool = True
    parallel_execution: bool = True
    max_workers: int = 4

    def __post_init__(self):
        if self.grid_sizes is None:
            self.grid_sizes = [(32, 32), (64, 64), (128, 128)]
        if self.n_steps_range is None:
            self.n_steps_range = [50, 100, 200]


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    execution_time: float
    memory_usage: float
    gpu_memory: Optional[float] = None
    cpu_usage: float = 0.0
    throughput: float = 0.0  # steps per second


@dataclass
class EmergenceMetrics:
    """Emergence and complexity measurements."""
    spatial_entropy: float
    temporal_diversity: float
    complexity_score: float
    stability_score: float
    pattern_diversity: float
    fractal_dimension: float
    information_content: float


class CABenchmarker:
    """Main benchmarking framework for cellular automata."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

        # Initialize tracking
        self.results = {}
        self.visualizer = CAVisualizer() if VIS_AVAILABLE else None

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # GPU detection
        try:
            self.has_gpu = jax.default_backend() == 'gpu'
        except:
            self.has_gpu = False
        if not self.has_gpu and self.config.enable_gpu:
            print("Warning: GPU requested but not available. Using CPU.")

    def benchmark_all_systems(self) -> Dict[str, Any]:
        """Run comprehensive benchmarks across all CA systems."""
        print("Starting comprehensive CA benchmarking...")

        # Define systems to benchmark based on availability
        systems = {}

        if EVO_AVAILABLE:
            systems['evolutionary'] = {
                'elementary': self._benchmark_evolutionary_ca,
                'totalistic': self._benchmark_evolutionary_ca,
                'neural_genome': self._benchmark_evolutionary_ca
            }

        if NEURAL_AVAILABLE:
            systems['neural'] = {
                'neural_ca': self._benchmark_neural_ca,
                'diff_logic_ca': self._benchmark_neural_ca,
                'universal_nca': self._benchmark_neural_ca
            }

        if not systems:
            raise RuntimeError("No CA systems available for benchmarking")

        all_results = {}

        for category, system_types in systems.items():
            print(f"\nBenchmarking {category} CA systems...")
            category_results = {}

            for system_name, benchmark_fn in system_types.items():
                print(f"  Testing {system_name}...")
                try:
                    results = benchmark_fn(system_name)
                    category_results[system_name] = results
                except Exception as e:
                    print(f"    Error benchmarking {system_name}: {e}")
                    category_results[system_name] = {'error': str(e)}

            all_results[category] = category_results

        # Save results
        if self.config.save_results:
            self._save_results(all_results)

        return all_results

    def _benchmark_evolutionary_ca(self, genome_type: str) -> Dict[str, Any]:
        """Benchmark evolutionary CA system."""
        if not EVO_AVAILABLE:
            return {'error': 'Evolutionary CA not available'}

        results = {}

        for grid_size in self.config.grid_sizes or [(32, 32), (64, 64), (128, 128)]:
            for n_steps in self.config.n_steps_range or [50, 100, 200]:
                key = f"{genome_type}_{grid_size[0]}x{grid_size[1]}_{n_steps}steps"

                # Configure evolutionary CA
                config = EvolutionaryConfig(
                    population_size=20,  # Smaller for benchmarking
                    n_generations=10,    # Fewer generations
                    grid_size=grid_size,
                    n_steps=n_steps,
                    n_trials=2
                )

                # Performance benchmark
                perf_metrics = self._measure_performance(
                    lambda: self._run_evolutionary_experiment(genome_type, config)
                )

                # Emergence benchmark
                emergence_metrics = self._measure_emergence_evolutionary(genome_type, config)

                results[key] = {
                    'performance': perf_metrics.__dict__,
                    'emergence': emergence_metrics.__dict__,
                    'config': {
                        'genome_type': genome_type,
                        'grid_size': grid_size,
                        'n_steps': n_steps
                    }
                }

        return results

    def _benchmark_neural_ca(self, model_type: str) -> Dict[str, Any]:
        """Benchmark neural CA system."""
        if not NEURAL_AVAILABLE:
            return {'error': 'Neural CA not available'}

        results = {}

        for grid_size in self.config.grid_sizes or [(32, 32), (64, 64), (128, 128)]:
            for n_steps in self.config.n_steps_range or [50, 100, 200]:
                key = f"{model_type}_{grid_size[0]}x{grid_size[1]}_{n_steps}steps"

                # Configure neural CA
                config = CAConfig(
                    grid_size=grid_size,
                    n_channels=8,  # Smaller for benchmarking
                    n_hidden=64,
                    fire_rate=0.5
                )

                # Performance benchmark
                perf_metrics = self._measure_performance(
                    lambda: self._run_neural_experiment(model_type, config, n_steps)
                )

                # Emergence benchmark
                emergence_metrics = self._measure_emergence_neural(model_type, config, n_steps)

                results[key] = {
                    'performance': perf_metrics.__dict__,
                    'emergence': emergence_metrics.__dict__,
                    'config': {
                        'model_type': model_type,
                        'grid_size': grid_size,
                        'n_steps': n_steps
                    }
                }

        return results

    def _measure_performance(self, experiment_fn: Callable) -> PerformanceMetrics:
        """Measure performance metrics for an experiment."""
        # Warmup
        for _ in range(self.config.warmup_trials):
            try:
                experiment_fn()
            except:
                pass  # Ignore warmup errors

        # Memory tracking
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # GPU memory if available
        initial_gpu_memory = None
        if self.has_gpu:
            try:
                initial_gpu_memory = self._get_gpu_memory_usage()
            except:
                initial_gpu_memory = None

        # CPU usage
        initial_cpu = psutil.cpu_percent(interval=None)

        # Time execution
        start_time = time.time()
        result = experiment_fn()
        end_time = time.time()

        execution_time = end_time - start_time

        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_usage = final_memory - initial_memory

        # Final GPU memory
        final_gpu_memory = None
        if self.has_gpu:
            try:
                final_gpu_memory = self._get_gpu_memory_usage()
                gpu_memory = final_gpu_memory - (initial_gpu_memory or 0)
            except:
                gpu_memory = None
        else:
            gpu_memory = None

        # CPU usage
        final_cpu = psutil.cpu_percent(interval=None)
        cpu_usage = (initial_cpu + final_cpu) / 2

        # Calculate throughput (assuming result has step count)
        throughput = getattr(result, 'steps', 1) / execution_time if hasattr(result, 'steps') else 0

        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            gpu_memory=gpu_memory,
            cpu_usage=cpu_usage,
            throughput=throughput
        )

    def _measure_emergence_evolutionary(self, genome_type: str, config) -> EmergenceMetrics:
        """Measure emergence metrics for evolutionary CA."""
        if not EVO_AVAILABLE:
            return EmergenceMetrics(0, 0, 0, 0, 0, 0, 0)

        # Run a quick evolution to get a representative individual
        evo_ca = EvolutionaryCA(config, genome_type)
        fitness_fns = [FitnessFunctions.complexity_fitness()]
        best_individual = evo_ca.evolve(fitness_fns, n_generations=5)

        # Simulate the best individual
        states = self._simulate_evolutionary_ca(best_individual.genome, config)

        # Calculate emergence metrics
        return self._calculate_emergence_metrics(states)

    def _measure_emergence_neural(self, model_type: str, config, n_steps: int) -> EmergenceMetrics:
        """Measure emergence metrics for neural CA."""
        if not NEURAL_AVAILABLE:
            return EmergenceMetrics(0, 0, 0, 0, 0, 0, 0)

        # Create and train a quick model
        if model_type == 'neural_ca':
            model = NeuralCA(config)
        elif model_type == 'diff_logic_ca':
            model = DiffLogicCA(config)
        else:  # universal_nca
            model = UniversalNCA(config)

        trainer = NCATrainer(model, config, n_steps=n_steps)
        trainer.train(n_epochs=50)

        # Generate pattern
        pattern = trainer.generate_pattern(steps=n_steps)

        # Convert to states (simulate evolution)
        states = [pattern]  # Simplified - in practice would track intermediate states

        # Calculate emergence metrics
        return self._calculate_emergence_metrics(states)

    def _calculate_emergence_metrics(self, states: List[Any]) -> EmergenceMetrics:
        """Calculate emergence and complexity metrics from state sequence."""
        if len(states) < 2:
            return EmergenceMetrics(0, 0, 0, 0, 0, 0, 0)

        # Spatial entropy
        spatial_entropies = []
        for state in states:
            if state.ndim == 3:  # Neural CA (H, W, C)
                state = state.mean(axis=-1)  # Average channels

            probs = np.bincount(state.astype(int).flatten(), minlength=2) / state.size
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            spatial_entropies.append(entropy)

        avg_spatial_entropy = float(np.mean(spatial_entropies))

        # Temporal diversity
        temporal_diversity = 0.0
        for i in range(1, len(states)):
            diff = np.mean(np.abs(states[i] - states[i-1]))
            temporal_diversity += diff
        temporal_diversity /= len(states) - 1

        # Complexity score (entropy * diversity)
        complexity_score = avg_spatial_entropy * temporal_diversity

        # Stability (similarity of final states)
        final_states = states[-min(10, len(states)):]
        stability = 0.0
        count = 0
        for i in range(len(final_states)):
            for j in range(i+1, len(final_states)):
                similarity = np.mean(final_states[i] == final_states[j])
                stability += similarity
                count += 1
        stability = stability / count if count > 0 else 0.0

        # Pattern diversity (variety of local patterns)
        pattern_diversity = self._calculate_pattern_diversity(states[-1])

        # Fractal dimension (box counting)
        fractal_dimension = self._calculate_fractal_dimension(states[-1])

        # Information content
        information_content = self._calculate_information_content(states)

        return EmergenceMetrics(
            spatial_entropy=float(avg_spatial_entropy),
            temporal_diversity=float(temporal_diversity),
            complexity_score=float(complexity_score),
            stability_score=float(stability),
            pattern_diversity=float(pattern_diversity),
            fractal_dimension=float(fractal_dimension),
            information_content=float(information_content)
        )

    def _calculate_pattern_diversity(self, grid: np.ndarray) -> float:
        """Calculate diversity of local patterns in the grid."""
        if grid.ndim == 3:
            grid = grid.mean(axis=-1)

        # Extract 3x3 patches
        patches = []
        for i in range(grid.shape[0] - 2):
            for j in range(grid.shape[1] - 2):
                patch = grid[i:i+3, j:j+3]
                patches.append(patch.flatten())

        if not patches:
            return 0.0

        # Calculate uniqueness of patches
        patches_array = np.array(patches)
        unique_patches = np.unique(patches_array, axis=0)
        diversity = len(unique_patches) / len(patches)

        return float(diversity)

    def _calculate_fractal_dimension(self, grid: np.ndarray) -> float:
        """Estimate fractal dimension using box counting."""
        if grid.ndim == 3:
            grid = grid.mean(axis=-1)

        # Simple box counting for binary grids
        binary_grid = (grid > np.mean(grid)).astype(int)

        # Different box sizes
        sizes = [2, 4, 8, 16]
        counts = []

        for size in sizes:
            if size > min(binary_grid.shape):
                continue

            count = 0
            for i in range(0, binary_grid.shape[0], size):
                for j in range(0, binary_grid.shape[1], size):
                    if np.any(binary_grid[i:i+size, j:j+size]):
                        count += 1
            counts.append(count)

        if len(counts) < 2:
            return 1.0

        # Linear regression for fractal dimension
        sizes_log = np.log(sizes[:len(counts)])
        counts_log = np.log(counts)

        if len(sizes_log) > 1:
            slope = np.polyfit(sizes_log, counts_log, 1)[0]
            return float(-slope)
        else:
            return 1.0

    def _calculate_information_content(self, states: List[np.ndarray]) -> float:
        """Calculate information content of the state sequence."""
        # Compress the state sequence and measure compression ratio
        state_bytes = pickle.dumps(states)
        original_bytes = len(states) * states[0].nbytes

        if original_bytes > 0:
            compression_ratio = len(state_bytes) / original_bytes
            information_content = 1.0 / compression_ratio  # Higher is more informative
        else:
            information_content = 0.0

        return float(information_content)

    def _run_evolutionary_experiment(self, genome_type: str, config):
        """Run evolutionary CA experiment for benchmarking."""
        if not EVO_AVAILABLE:
            return None

        evo_ca = EvolutionaryCA(config, genome_type)
        fitness_fns = [FitnessFunctions.complexity_fitness()]
        best = evo_ca.evolve(fitness_fns, n_generations=config.n_generations)

        return best

    def _run_neural_experiment(self, model_type: str, config, n_steps: int):
        """Run neural CA experiment for benchmarking."""
        if not NEURAL_AVAILABLE:
            return None

        if model_type == 'neural_ca':
            model = NeuralCA(config)
        elif model_type == 'diff_logic_ca':
            model = DiffLogicCA(config)
        else:
            model = UniversalNCA(config)

        trainer = NCATrainer(model, config, n_steps=n_steps)
        trainer.train(n_epochs=50)

        return trainer

    def _simulate_evolutionary_ca(self, genome, config) -> List[np.ndarray]:
        """Simulate evolutionary CA to get state sequence."""
        states = []

        # Create initial state
        initial = np.random.choice([0, 1], size=config.grid_size, p=[0.5, 0.5])
        grid = initial.copy()
        states.append(grid.copy())

        # Evolve
        for step in range(config.n_steps):
            new_grid = np.zeros_like(grid)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                neighbors.append(grid[ni, nj])
                            else:
                                neighbors.append(0)

                    new_grid[i, j] = genome.to_rule_function()(grid[i, j], np.array(neighbors))
            grid = new_grid
            states.append(grid.copy())

        return states

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            # This is a simplified version - in practice would use nvidia-ml-py or similar
            return 0.0  # Placeholder
        except:
            return 0.0

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to files."""
        # JSON for structured data
        json_path = os.path.join(self.config.output_dir, 'benchmark_results.json')
        with open(json_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._make_json_serializable(results)
            json.dump(json_results, f, indent=2)

        # Pickle for full Python objects
        pickle_path = os.path.join(self.config.output_dir, 'benchmark_results.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"Results saved to {self.config.output_dir}")

    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # JAX or numpy array
            return obj.tolist()
        else:
            return obj

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# Cellular Automata Benchmarking Report")
        report.append("=" * 50)
        report.append("")

        for category, systems in results.items():
            report.append(f"## {category.upper()} CA SYSTEMS")
            report.append("")

            for system_name, benchmarks in systems.items():
                if 'error' in benchmarks:
                    report.append(f"### {system_name}")
                    report.append(f"Error: {benchmarks['error']}")
                    report.append("")
                    continue

                report.append(f"### {system_name}")
                report.append("")

                # Performance summary
                perf_times = []
                perf_memory = []
                emergence_scores = []

                for benchmark_name, data in benchmarks.items():
                    perf = data.get('performance', {})
                    emergence = data.get('emergence', {})

                    if 'execution_time' in perf:
                        perf_times.append(perf['execution_time'])
                    if 'memory_usage' in perf:
                        perf_memory.append(perf['memory_usage'])
                    if 'complexity_score' in emergence:
                        emergence_scores.append(emergence['complexity_score'])

                if perf_times:
                    report.append(f"Average execution time: {np.mean(perf_times):.3f}s")
                if perf_memory:
                    report.append(f"Average memory usage: {np.mean(perf_memory):.1f}MB")
                if emergence_scores:
                    report.append(f"Average complexity score: {np.mean(emergence_scores):.3f}")
                report.append("")

        return "\n".join(report)


# Utility functions for running benchmarks
def run_quick_benchmark(output_dir: str = 'benchmark_results') -> Dict[str, Any]:
    """Run a quick benchmark across all systems."""
    config = BenchmarkConfig(
        grid_sizes=[(32, 32), (64, 64)],
        n_steps_range=[50, 100],
        n_trials=3,
        output_dir=output_dir
    )

    benchmarker = CABenchmarker(config)
    results = benchmarker.benchmark_all_systems()

    # Generate and save report
    report = benchmarker.generate_report(results)
    report_path = os.path.join(output_dir, 'benchmark_report.md')
    with open(report_path, 'w') as f:
        f.write(report)

    return results


def compare_systems(results: Dict[str, Any], metric: str = 'complexity_score') -> Dict[str, float]:
    """Compare systems based on a specific emergence metric."""
    comparisons = {}

    for category, systems in results.items():
        for system_name, benchmarks in systems.items():
            if 'error' in benchmarks:
                continue

            scores = []
            for benchmark_name, data in benchmarks.items():
                emergence = data.get('emergence', {})
                if metric in emergence:
                    scores.append(emergence[metric])

            if scores:
                comparisons[f"{category}_{system_name}"] = np.mean(scores)

    return comparisons


if __name__ == "__main__":
    # Run quick benchmark
    results = run_quick_benchmark()
    print("Benchmarking complete!")

    # Print summary
    comparisons = compare_systems(results, 'complexity_score')
    print("\nComplexity Score Comparison:")
    for system, score in sorted(comparisons.items(), key=lambda x: x[1], reverse=True):
        print(f"  {system}: {score:.3f}")