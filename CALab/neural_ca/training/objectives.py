"""
Training and Optimization Framework for Neural CA
==================================================

Contains training objectives, datasets, and optimizers for neural CA.
Supports supervised learning from target patterns and unsupervised discovery.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import optax
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from functools import partial


@dataclass
class TrainingConfig:
    """Configuration for neural CA training."""
    n_epochs: int = 1000
    batch_size: int = 8
    learning_rate: float = 1e-3
    n_steps: int = 64  # CA simulation steps
    patience: int = 100  # Early stopping patience
    seed: int = 42


class PatternDataset:
    """Dataset of target patterns for supervised training."""

    def __init__(self, patterns: List[jnp.ndarray]):
        """Initialize with list of target patterns."""
        self.patterns = patterns

    @classmethod
    def from_game_of_life(cls, size: int = 32) -> 'PatternDataset':
        """Create dataset from Game of Life patterns."""
        patterns = []

        # Glider
        glider = jnp.zeros((size, size))
        glider = glider.at[1, 3].set(1)
        glider = glider.at[2, 1].set(1)
        glider = glider.at[2, 3].set(1)
        glider = glider.at[3, 2].set(1)
        glider = glider.at[3, 3].set(1)
        patterns.append(glider)

        # Block (still life)
        block = jnp.zeros((size, size))
        block = block.at[size//2-1:size//2+1, size//2-1:size//2+1].set(1)
        patterns.append(block)

        # Blinker (oscillator)
        blinker = jnp.zeros((size, size))
        blinker = blinker.at[size//2, size//2-1:size//2+2].set(1)
        patterns.append(blinker)

        return cls(patterns)

    def __len__(self) -> int:
        return len(self.patterns)

    def __getitem__(self, idx: int) -> jnp.ndarray:
        return self.patterns[idx]


class LossFunctions:
    """Collection of loss functions for neural CA training."""

    @staticmethod
    def mse_loss(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Mean squared error loss."""
        return jnp.mean((predicted - target) ** 2)

    @staticmethod
    def pattern_similarity_loss(predicted: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Loss based on pattern similarity metrics."""
        # Binary cross-entropy for pattern matching
        pred_binary = jnp.clip(predicted, 0, 1)
        target_binary = jnp.clip(target, 0, 1)

        bce = -jnp.mean(
            target_binary * jnp.log(pred_binary + 1e-8) +
            (1 - target_binary) * jnp.log(1 - pred_binary + 1e-8)
        )

        # Add L2 regularization to encourage smooth patterns
        l2_reg = 0.01 * jnp.mean(predicted ** 2)

        return bce + l2_reg

    @staticmethod
    def emergence_loss(states: List[jnp.ndarray]) -> jnp.ndarray:
        """Unsupervised loss encouraging emergent behavior."""
        if len(states) < 2:
            return 0.0

        # Encourage state changes over time
        changes = [jnp.mean(jnp.abs(states[i+1] - states[i])) for i in range(len(states)-1)]
        change_loss = -jnp.mean(jnp.array(changes))  # Negative because we want changes

        # Encourage spatial complexity (high-frequency content)
        final_state = states[-1]
        fft = jnp.fft.fft2(final_state.mean(axis=-1))
        complexity_loss = -jnp.mean(jnp.abs(fft)) / jnp.prod(jnp.array(final_state.shape[:-1]))

        # Encourage stability in the final state
        stability_loss = jnp.mean(jnp.abs(final_state - states[-2])) if len(states) > 1 else 0.0

        return change_loss + 0.1 * complexity_loss + 0.01 * stability_loss

    @staticmethod
    def conservation_loss(states: List[jnp.ndarray], conserved_quantity: str = 'mass') -> jnp.ndarray:
        """Loss encouraging conservation of physical quantities."""
        if conserved_quantity == 'mass':
            # Conserve total mass (sum of all cells)
            masses = [jnp.sum(state) for state in states]
            mass_changes = jnp.array([masses[i+1] - masses[i] for i in range(len(masses)-1)])
            return jnp.mean(mass_changes ** 2)
        elif conserved_quantity == 'energy':
            # Conserve energy-like quantity
            energies = [jnp.sum(state ** 2) for state in states]
            energy_changes = jnp.array([energies[i+1] - energies[i] for i in range(len(energies)-1)])
            return jnp.mean(energy_changes ** 2)

        return 0.0


class NCATrainingObjectives:
    """Training objectives for different neural CA tasks."""

    @staticmethod
    def pattern_replication_objective(target_pattern: jnp.ndarray,
                                    n_steps: int = 50) -> Callable:
        """Objective to replicate a target pattern."""
        def objective(model_fn: Callable, params: Dict, initial_state: jnp.ndarray) -> jnp.ndarray:
            state = initial_state
            for _ in range(n_steps):
                state = model_fn(params, state)

            return LossFunctions.pattern_similarity_loss(state, target_pattern)
        return objective

    @staticmethod
    def self_organization_objective(n_steps: int = 100) -> Callable:
        """Objective encouraging self-organization."""
        def objective(model_fn: Callable, params: Dict, initial_state: jnp.ndarray) -> jnp.ndarray:
            states = []
            state = initial_state
            for _ in range(n_steps):
                states.append(state)
                state = model_fn(params, state)

            emergence_loss = LossFunctions.emergence_loss(states)
            conservation_loss = LossFunctions.conservation_loss(states, 'mass')

            return emergence_loss + 0.1 * conservation_loss
        return objective

    @staticmethod
    def oscillator_objective(period: int = 10, n_cycles: int = 5) -> Callable:
        """Objective to create oscillating patterns."""
        def objective(model_fn: Callable, params: Dict, initial_state: jnp.ndarray) -> jnp.ndarray:
            state = initial_state
            states = []

            # Collect states over multiple cycles
            for _ in range(period * n_cycles):
                states.append(state)
                state = model_fn(params, state)

            # Check periodicity
            period_errors = []
            for i in range(period):
                error = jnp.mean((states[i] - states[i + period]) ** 2)
                period_errors.append(error)

            return jnp.mean(jnp.array(period_errors))
        return objective

    @staticmethod
    def glider_objective() -> Callable:
        """Objective to create moving patterns (gliders)."""
        def objective(model_fn: Callable, params: Dict, initial_state: jnp.ndarray) -> jnp.ndarray:
            state = initial_state
            positions = []

            # Track center of mass over time
            for _ in range(50):
                center_mass = jnp.array([
                    jnp.sum(state * jnp.arange(state.shape[0])[:, None]) / jnp.sum(state),
                    jnp.sum(state * jnp.arange(state.shape[1])[None, :]) / jnp.sum(state)
                ])
                positions.append(center_mass)
                state = model_fn(params, state)

            # Check if center of mass is moving (indicating glider-like behavior)
            positions = jnp.array(positions)
            movement = jnp.mean(jnp.abs(positions[1:] - positions[:-1]))

            # Penalize if no movement or erratic movement
            if movement < 0.1:  # Too static
                return 1.0
            elif movement > 5.0:  # Too erratic
                return 1.0
            else:
                return 1.0 / movement  # Reward smooth movement
        return objective


class NCAOptimizer:
    """Advanced optimizer for neural CA training."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with learning rate schedule."""
        # Cosine decay learning rate schedule
        schedule = optax.cosine_decay_schedule(
            init_value=self.config.learning_rate,
            decay_steps=self.config.n_epochs,
            alpha=0.01
        )

        # Adam optimizer with weight decay
        optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=1e-4
        )

        # Add gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optimizer
        )

        return optimizer

    def create_multi_step_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer for multi-step CA training."""
        # Use a more conservative learning rate for stability
        schedule = optax.linear_schedule(
            init_value=self.config.learning_rate,
            end_value=self.config.learning_rate * 0.1,
            transition_steps=self.config.n_epochs
        )

        return optax.adam(learning_rate=schedule)


class CurriculumLearning:
    """Curriculum learning for neural CA training."""

    def __init__(self, stages: List[Dict[str, Any]]):
        """Initialize curriculum stages."""
        self.stages = stages
        self.current_stage = 0

    def get_current_config(self) -> Dict[str, Any]:
        """Get configuration for current curriculum stage."""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return self.stages[-1]  # Stay at final stage

    def advance_stage(self, performance_metric: float) -> bool:
        """Advance to next stage if performance threshold met."""
        if self.current_stage < len(self.stages) - 1:
            threshold = self.stages[self.current_stage].get('threshold', 0.0)
            if performance_metric < threshold:
                self.current_stage += 1
                return True
        return False

    @classmethod
    def create_pattern_curriculum(cls) -> 'CurriculumLearning':
        """Create curriculum for pattern learning."""
        stages = [
            {
                'n_steps': 10,
                'complexity': 'simple',
                'threshold': 0.5,
                'description': 'Learn basic patterns with short sequences'
            },
            {
                'n_steps': 25,
                'complexity': 'medium',
                'threshold': 0.3,
                'description': 'Increase sequence length and pattern complexity'
            },
            {
                'n_steps': 50,
                'complexity': 'complex',
                'threshold': 0.1,
                'description': 'Full complexity patterns with long sequences'
            }
        ]
        return cls(stages)

    @classmethod
    def create_emergence_curriculum(cls) -> 'CurriculumLearning':
        """Create curriculum for emergence discovery."""
        stages = [
            {
                'grid_size': (16, 16),
                'n_channels': 4,
                'threshold': 0.8,
                'description': 'Small grids, simple dynamics'
            },
            {
                'grid_size': (32, 32),
                'n_channels': 8,
                'threshold': 0.6,
                'description': 'Medium grids, more complex dynamics'
            },
            {
                'grid_size': (64, 64),
                'n_channels': 16,
                'threshold': 0.4,
                'description': 'Large grids, rich emergent behavior'
            }
        ]
        return cls(stages)


# Utility functions for training
def create_training_data(n_samples: int = 100,
                        grid_size: Tuple[int, int] = (32, 32),
                        n_channels: int = 16) -> jnp.ndarray:
    """Create random training data."""
    key = random.PRNGKey(42)
    return random.normal(key, (n_samples, grid_size[0], grid_size[1], n_channels))


def evaluate_nca_performance(model_fn: Callable,
                           params: Dict,
                           test_patterns: List[jnp.ndarray],
                           n_steps: int = 50) -> Dict[str, float]:
    """Evaluate neural CA performance on test patterns."""
    similarities = []

    for target in test_patterns:
        # Start with random initial state
        key = random.PRNGKey(0)
        initial = random.normal(key, target.shape + (16,))  # 16 channels

        # Evolve CA
        state = initial
        for _ in range(n_steps):
            state = model_fn(params, state)

        # Compute similarity to target
        similarity = -LossFunctions.pattern_similarity_loss(state.mean(axis=-1), target)
        similarities.append(float(similarity))

    return {
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities),
        'best_similarity': np.max(similarities)
    }