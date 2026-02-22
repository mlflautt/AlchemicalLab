"""
Neural Cellular Automata Framework
==================================

JAX-based differentiable cellular automata with learnable update rules.
Supports training CA to generate specific patterns and behaviors.

Key Components:
- NeuralCA: Basic differentiable CA with learned update rules
- DiffLogicCA: CA with differentiable logic gates
- UniversalNCA: General-purpose neural CA architecture
"""

import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from jax.nn import relu, sigmoid, tanh
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from functools import partial


@dataclass
class CAConfig:
    """Configuration for Neural CA."""
    grid_size: Tuple[int, int] = (64, 64)
    n_channels: int = 16  # Number of state channels per cell
    n_hidden: int = 128   # Hidden layer size
    kernel_size: int = 3  # Convolution kernel size
    n_layers: int = 2     # Number of update layers
    fire_rate: float = 0.5  # Stochastic update probability
    seed: int = 42


class NeuralCA(nn.Module):
    """Basic Neural Cellular Automaton with learned update rules."""

    config: CAConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the neural CA.

        Args:
            x: Input state tensor of shape (H, W, C)

        Returns:
            Updated state tensor of same shape
        """
        # Perception: extract neighborhood features
        y = self._perceive(x)

        # Update: apply neural network to compute state changes
        dx = self._update(y)

        # Stochastic update mask
        mask = self._stochastic_mask(x.shape)

        # Apply updates
        x = x + dx * mask

        return x

    def _perceive(self, x: jnp.ndarray) -> jnp.ndarray:
        """Extract neighborhood features using convolution."""
        # Identity perception (cell's own state)
        y = nn.Conv(
            features=self.config.n_channels,
            kernel_size=(1, 1),
            name='identity'
        )(x)

        # Sobel filters for gradient detection
        sobel_x = jnp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None, None, :]
        sobel_y = jnp.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])[None, None, None, :]

        # Apply Sobel filters
        grad_x = nn.Conv(
            features=self.config.n_channels,
            kernel_size=(3, 3),
            name='grad_x',
            kernel_init=lambda key, shape: sobel_x
        )(x)

        grad_y = nn.Conv(
            features=self.config.n_channels,
            kernel_size=(3, 3),
            name='grad_y',
            kernel_init=lambda key, shape: sobel_y
        )(x)

        # Concatenate all perceptions
        y = jnp.concatenate([y, grad_x, grad_y], axis=-1)

        return y

    def _update(self, y: jnp.ndarray) -> jnp.ndarray:
        """Neural update function."""
        # Dense layers for state computation
        for i in range(self.config.n_layers):
            y = nn.Dense(
                features=self.config.n_hidden,
                name=f'update_{i}'
            )(y)
            y = relu(y)

        # Output layer to compute state changes
        dx = nn.Dense(
            features=self.config.n_channels,
            name='output'
        )(y)

        return dx

    def _stochastic_mask(self, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Generate stochastic update mask."""
        # During training, randomly mask updates
        if self.is_mutable_collection('dropout'):
            key = self.make_rng('dropout')
            mask = random.bernoulli(
                key,
                self.config.fire_rate,
                shape[:-1] + (1,)  # One mask per cell, broadcast to channels
            )
            return mask
        else:
            # During inference, always update
            return jnp.ones(shape[:-1] + (1,))


class DiffLogicCA(nn.Module):
    """Differentiable Logic Cellular Automaton.

    Uses differentiable logic gates instead of traditional neural layers.
    Inspired by 'Differentiable Logic Cellular Automata' research.
    """

    config: CAConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with differentiable logic."""
        # Extract binary features from continuous state
        binary_features = self._binarize(x)

        # Apply differentiable logic operations
        logic_output = self._logic_layer(binary_features)

        # Convert back to continuous updates
        dx = self._continuous_output(logic_output)

        # Apply stochastic updates
        mask = self._stochastic_mask(x.shape)
        x = x + dx * mask

        return x

    def _binarize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert continuous states to binary features."""
        # Use sigmoid to get binary probabilities
        return sigmoid(x)

    def _logic_layer(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply differentiable logic operations."""
        # AND, OR, NOT operations using soft logic
        and_gate = jnp.minimum(x, jnp.roll(x, 1, axis=-1))
        or_gate = jnp.maximum(x, jnp.roll(x, 1, axis=-1))
        not_gate = 1 - x

        # Combine logic operations
        logic_features = jnp.concatenate([and_gate, or_gate, not_gate], axis=-1)

        # Learnable combination weights
        combined = nn.Dense(
            features=self.config.n_channels,
            name='logic_combine'
        )(logic_features)

        return combined

    def _continuous_output(self, logic_output: jnp.ndarray) -> jnp.ndarray:
        """Convert logic output back to continuous state updates."""
        return nn.Dense(
            features=self.config.n_channels,
            name='continuous_out'
        )(logic_output)

    def _stochastic_mask(self, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Generate stochastic update mask."""
        if self.is_mutable_collection('dropout'):
            key = self.make_rng('dropout')
            mask = random.bernoulli(
                key,
                self.config.fire_rate,
                shape[:-1] + (1,)
            )
            return mask
        else:
            return jnp.ones(shape[:-1] + (1,))


class UniversalNCA(nn.Module):
    """Universal Neural Cellular Automaton.

    General-purpose architecture capable of universal computation.
    Based on 'A Path to Universal Neural Cellular Automata' research.
    """

    config: CAConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Universal NCA forward pass."""
        # Multi-head attention for neighborhood communication
        attended = self._attention_layer(x)

        # Memory-augmented updates
        memory = self.variable('memory', 'memory', jnp.zeros, (self.config.grid_size[0], self.config.grid_size[1], self.config.n_channels))

        # Update memory and state
        new_memory, dx = self._memory_update(attended, memory.value)

        # Update memory variable
        memory.value = new_memory

        # Apply stochastic updates
        mask = self._stochastic_mask(x.shape)
        x = x + dx * mask

        return x

    def _attention_layer(self, x: jnp.ndarray) -> jnp.ndarray:
        """Multi-head attention over neighborhood."""
        # Simplified attention mechanism
        # In full implementation, this would use proper attention
        query = nn.Dense(self.config.n_hidden, name='query')(x)
        key = nn.Dense(self.config.n_hidden, name='key')(x)
        value = nn.Dense(self.config.n_hidden, name='value')(x)

        # Compute attention weights (simplified)
        attention = jnp.matmul(query, key.transpose(-2, -1)) / jnp.sqrt(self.config.n_hidden)
        attention = nn.softmax(attention, axis=-1)

        attended = jnp.matmul(attention, value)
        return attended

    def _memory_update(self, attended: jnp.ndarray, memory: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update memory and compute state changes."""
        # Combine attended features with memory
        combined = jnp.concatenate([attended, memory], axis=-1)

        # Memory update
        new_memory = nn.Dense(
            features=self.config.n_channels,
            name='memory_update'
        )(combined)
        new_memory = relu(new_memory)

        # State update
        dx = nn.Dense(
            features=self.config.n_channels,
            name='state_update'
        )(combined)

        return new_memory, dx

    def _stochastic_mask(self, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Generate stochastic update mask."""
        if self.is_mutable_collection('dropout'):
            key = self.make_rng('dropout')
            mask = random.bernoulli(
                key,
                self.config.fire_rate,
                shape[:-1] + (1,)
            )
            return mask
        else:
            return jnp.ones(shape[:-1] + (1,))


class NCATrainer:
    """Training framework for Neural Cellular Automata."""

    def __init__(self,
                 model: nn.Module,
                 config: CAConfig,
                 target_pattern: Optional[jnp.ndarray] = None,
                 n_steps: int = 64):
        """Initialize trainer.

        Args:
            model: Neural CA model to train
            config: Model configuration
            target_pattern: Target pattern to learn (optional)
            n_steps: Number of CA steps to simulate
        """
        self.model = model
        self.config = config
        self.target_pattern = target_pattern
        self.n_steps = n_steps

        # Initialize model
        self.key = random.PRNGKey(config.seed)
        dummy_input = jnp.zeros((config.grid_size[0], config.grid_size[1], config.n_channels))

        # Create train state
        variables = model.init(self.key, dummy_input)
        self.state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=optax.adam(1e-3)
        )

    def loss_function(self, params: Dict, batch: Dict) -> jnp.ndarray:
        """Compute training loss."""
        initial_state = batch['initial']
        target = batch['target']

        # Simulate CA evolution
        state = initial_state
        for _ in range(self.n_steps):
            state = self.model.apply({'params': params}, state)

        # Compute loss against target pattern
        if self.target_pattern is not None:
            loss = jnp.mean((state - self.target_pattern) ** 2)
        else:
            # Unsupervised loss: encourage interesting dynamics
            loss = self._dynamics_loss(state, initial_state)

        return loss

    def _dynamics_loss(self, final_state: jnp.ndarray, initial_state: jnp.ndarray) -> jnp.ndarray:
        """Unsupervised loss encouraging interesting dynamics."""
        # Encourage state changes
        change_loss = -jnp.mean(jnp.abs(final_state - initial_state))

        # Encourage spatial patterns (high frequency content)
        fft = jnp.fft.fft2(final_state.mean(axis=-1))
        pattern_loss = -jnp.mean(jnp.abs(fft))

        # Encourage stability (low change in later steps)
        stability_loss = jnp.mean(jnp.abs(final_state - final_state))

        return change_loss + 0.1 * pattern_loss + 0.01 * stability_loss

    @partial(jit, static_argnums=(0,))
    def train_step(self, state: train_state.TrainState, batch: Dict) -> Tuple[train_state.TrainState, Dict]:
        """Single training step."""
        loss_fn = lambda params: self.loss_function(params, batch)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        new_state = state.apply_gradients(grads=grads)

        metrics = {'loss': loss}
        return new_state, metrics

    def train(self, n_epochs: int = 1000, batch_size: int = 8) -> Dict[str, List[float]]:
        """Train the neural CA."""
        losses = []

        for epoch in range(n_epochs):
            # Generate random initial states
            batch = self._generate_batch(batch_size)

            # Training step
            self.state, metrics = self.train_step(self.state, batch)
            losses.append(float(metrics['loss']))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {metrics['loss']:.6f}")

        return {'losses': losses}

    def _generate_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate training batch."""
        self.key, subkey = random.split(self.key)

        # Random initial states
        initial = random.normal(
            subkey,
            (batch_size, self.config.grid_size[0], self.config.grid_size[1], self.config.n_channels)
        )

        # For now, target is the evolved state (self-supervised)
        target = initial  # Will be modified during evolution

        return {'initial': initial[0], 'target': target[0]}  # Single example for simplicity

    def generate_pattern(self, initial_state: Optional[jnp.ndarray] = None, steps: int = 100) -> jnp.ndarray:
        """Generate pattern using trained CA."""
        if initial_state is None:
            # Random initial state
            self.key, subkey = random.split(self.key)
            initial_state = random.normal(
                subkey,
                (self.config.grid_size[0], self.config.grid_size[1], self.config.n_channels)
            )

        state = initial_state
        for _ in range(steps):
            state = self.model.apply({'params': self.state.params}, state)

        return state


# Utility functions
def create_neural_ca(config: CAConfig) -> NeuralCA:
    """Create a basic Neural CA."""
    return NeuralCA(config)


def create_diff_logic_ca(config: CAConfig) -> DiffLogicCA:
    """Create a Differentiable Logic CA."""
    return DiffLogicCA(config)


def create_universal_nca(config: CAConfig) -> UniversalNCA:
    """Create a Universal Neural CA."""
    return UniversalNCA(config)


def train_nca_for_pattern(target_pattern: jnp.ndarray,
                         config: CAConfig,
                         n_epochs: int = 1000) -> NCATrainer:
    """Train a Neural CA to generate a specific pattern."""
    model = create_neural_ca(config)
    trainer = NCATrainer(model, config, target_pattern=target_pattern)
    trainer.train(n_epochs)
    return trainer