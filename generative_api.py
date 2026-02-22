"""
Unified Generative API for AlchemicalLab
========================================

A common interface for all generative systems in AlchemicalLab:
- StoryLab: Narrative generation
- NNLab: Neural generative models
- SynthLab: Emergent systems
- EALab: Evolutionary generation

This provides a consistent API for sampling, training, and evaluation
across different generative modalities.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Union, Callable, Protocol
from abc import ABC, abstractmethod
import numpy as np


class GenerativeModel(Protocol):
    """Protocol for all generative models in AlchemicalLab."""

    def sample(self, key: jax.random.PRNGKey, n_samples: int, **kwargs) -> Any:
        """Generate samples from the model."""
        ...

    def log_prob(self, samples: Any) -> jnp.ndarray:
        """Compute log probabilities of samples."""
        ...

    def train_step(self, batch: Any, **kwargs) -> Dict[str, Any]:
        """Single training step."""
        ...


class UnifiedGenerativeAPI:
    """Unified interface for all AlchemicalLab generative systems."""

    def __init__(self):
        self.models: Dict[str, GenerativeModel] = {}
        self.pipelines: Dict[str, Callable] = {}

    def register_model(self, name: str, model: GenerativeModel):
        """Register a generative model."""
        self.models[name] = model

    def register_pipeline(self, name: str, pipeline: Callable):
        """Register a generative pipeline."""
        self.pipelines[name] = pipeline

    def sample(self, model_name: str, key: Any, n_samples: int = 1, **kwargs) -> Any:
        """Sample from a registered model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")

        return self.models[model_name].sample(key, n_samples, **kwargs)

    def generate_story(self, world_dna: str, **kwargs) -> Dict[str, Any]:
        """Generate a complete story using StoryLab pipeline."""
        if 'story' not in self.pipelines:
            # Fallback implementation
            return self._basic_story_generation(world_dna, **kwargs)

        return self.pipelines['story'](world_dna, **kwargs)

    def generate_image(self, prompt: str, **kwargs) -> str:
        """Generate an image from text prompt."""
        if 'image' in self.pipelines:
            return self.pipelines['image'](prompt, **kwargs)

        # Fallback to basic implementation
        return f"Generated image for: {prompt}"

    def evolve_population(self, initial_pop: Any, fitness_fn: Callable, **kwargs) -> Any:
        """Evolve a population using EALab."""
        if 'evolution' not in self.pipelines:
            raise ValueError("Evolution pipeline not available")

        return self.pipelines['evolution'](initial_pop, fitness_fn, **kwargs)

    def simulate_ca(self, initial_state: jnp.ndarray, steps: int, **kwargs) -> jnp.ndarray:
        """Run cellular automaton simulation."""
        if 'ca' not in self.pipelines:
            raise ValueError("CA pipeline not available")

        return self.pipelines['ca'](initial_state, steps, **kwargs)

    def _basic_story_generation(self, world_dna: str, **kwargs) -> Dict[str, Any]:
        """Basic story generation fallback."""
        # Simple template-based generation
        story = f"In a world where {world_dna[:100]}..., a hero emerged to face the challenges ahead."

        return {
            'story': story,
            'dna': world_dna,
            'metadata': {'method': 'basic_template'}
        }

    def multimodal_generation(self, prompt: str, modalities: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Generate content across multiple modalities."""
        if modalities is None:
            modalities = ['text', 'image', 'simulation']

        results = {}

        if 'text' in modalities:
            results['story'] = self.generate_story(prompt, **kwargs)

        if 'image' in modalities:
            results['image'] = self.generate_image(prompt, **kwargs)

        if 'simulation' in modalities and 'ca' in self.pipelines:
            # Generate a CA pattern based on the prompt
            key = kwargs.get('key', jax.random.PRNGKey(0))
            initial_state = jax.random.bernoulli(key, 0.3, (50, 50))
            results['simulation'] = self.simulate_ca(initial_state, 100)

        return results

    def evaluate_generation(self, generated_content: Any, criteria: Dict[str, Callable]) -> Dict[str, float]:
        """Evaluate generated content against criteria."""
        scores = {}

        for criterion_name, criterion_fn in criteria.items():
            try:
                scores[criterion_name] = float(criterion_fn(generated_content))
            except Exception as e:
                scores[criterion_name] = 0.0
                print(f"Error evaluating {criterion_name}: {e}")

        return scores

    def get_available_models(self) -> List[str]:
        """Get list of available generative models."""
        return list(self.models.keys())

    def get_available_pipelines(self) -> List[str]:
        """Get list of available generative pipelines."""
        return list(self.pipelines.keys())
    
    def generate_world_from_ca(
        self,
        seed: int = 42,
        generations: int = 50,
        world_size: tuple = (100, 100),
        density: float = 0.3,
        complexity: str = "medium",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete world from CA simulation.
        
        Full pipeline: CA simulation → WorldState → WorldDNA → Story
        
        Args:
            seed: Random seed for reproducibility
            generations: Number of CA generations to run
            world_size: (height, width) of CA grid
            density: Initial cell density (0-1)
            complexity: "low", "medium", or "high" affecting generation parameters
            progress_callback: Optional callback(gen, world_state) for progress tracking
        
        Returns:
            Dict containing world_state, world_dna, entities, and optionally story
        """
        gen_map = {"low": 30, "medium": 50, "high": 100}
        actual_generations = gen_map.get(complexity, generations)
        
        try:
            from CALab.hybrid_ca_aggregator import run_hybrid_simulation
            from CALab.narrative_bridge import world_state_to_world_dna
            
            def default_progress(gen, state):
                if gen % 25 == 0:
                    print(f"  Generation {gen}: {len(state.species)} species, "
                          f"{len(state.characters)} characters")
            
            callback = progress_callback or default_progress
            
            world_state = run_hybrid_simulation(
                world_size=world_size,
                generations=actual_generations,
                density=density,
                seed=seed,
                progress_callback=callback
            )
            
            world_dna = world_state_to_world_dna(world_state)
            
            entities = {
                'species': [s.to_dict() for s in world_state.species.values()],
                'characters': [c.to_dict() for c in world_state.characters.values()],
                'locations': [l.to_dict() for l in world_state.locations.values()],
                'factions': [f.to_dict() for f in world_state.factions.values()],
                'story_arcs': [a.to_dict() for a in world_state.story_arcs.values()]
            }
            
            relationships = {
                'ecological': [r.to_dict() for r in world_state.ecological_relationships],
                'narrative': [r.to_dict() for r in world_state.narrative_relationships]
            }
            
            return {
                'world_state': world_state,
                'world_dna': world_dna,
                'entities': entities,
                'relationships': relationships,
                'stats': world_state.stats,
                'metadata': {
                    'seed': seed,
                    'generations': actual_generations,
                    'world_size': world_size,
                    'complexity': complexity
                }
            }
            
        except ImportError as e:
            return {
                'error': f"CA system not available: {e}",
                'world_dna': f"Simulated world from seed {seed}",
                'fallback': True
            }
    
    def simulate_and_narrate(
        self,
        seed: int = 42,
        generations: int = 30,
        world_size: tuple = (60, 60),
        theme: str = "",
        multi_step: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run CA simulation and generate narrative in one call.
        
        Combines CA world generation with LLM narrative generation.
        
        Args:
            seed: Random seed
            generations: CA simulation steps
            world_size: Grid dimensions
            theme: Optional theme for story generation
            multi_step: Use multi-step story generation
            **kwargs: Additional arguments for story generation
        
        Returns:
            Dict with world_state, world_dna, story, critique, and full results
        """
        world_result = self.generate_world_from_ca(
            seed=seed,
            generations=generations,
            world_size=world_size
        )
        
        if 'error' in world_result:
            return world_result
        
        story_result = self.generate_story(
            world_result['world_dna'],
            theme=theme,
            multi_step=multi_step,
            **kwargs
        )
        
        return {
            **world_result,
            'story': story_result.get('story', ''),
            'critique': story_result.get('critique', {}),
            'extra': story_result.get('extra', {})
        }
    
    def cross_modal_generate(
        self,
        prompt: str,
        ca_seed: Optional[int] = None,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate cross-modal content from a prompt.
        
        If ca_seed provided, uses CA-derived world. Otherwise, uses prompt directly.
        
        Args:
            prompt: Text prompt for generation
            ca_seed: If provided, generates CA world first
            modalities: List of output modalities ['text', 'world', 'ca_viz']
            **kwargs: Additional generation arguments
        
        Returns:
            Dict with content for each requested modality
        """
        if modalities is None:
            modalities = ['text']
        
        results = {'prompt': prompt}
        
        if ca_seed is not None and 'world' in modalities:
            world_result = self.generate_world_from_ca(seed=ca_seed, **kwargs)
            results['world'] = world_result
            
            if 'text' in modalities:
                story_result = self.generate_story(world_result.get('world_dna', prompt))
                results['story'] = story_result
        else:
            if 'text' in modalities:
                results['story'] = self.generate_story(prompt, **kwargs)
        
        if 'image' in modalities:
            results['image'] = self.generate_image(prompt, **kwargs)
        
        return results
    
    def get_cross_lab_network(
        self,
        world_state: Any
    ) -> Dict[str, Any]:
        """
        Get the combined network representation of a world state.
        
        Returns a NetworkX-compatible graph structure combining
        ecological and narrative entities and relationships.
        """
        try:
            from CALab.hybrid_ca_aggregator import HybridCAAggregator
            
            aggregator = HybridCAAggregator()
            aggregator.world_state = world_state
            G = aggregator.get_combined_network()
            
            return {
                'nodes': [
                    {
                        'id': node,
                        'type': G.nodes[node].get('node_type', 'unknown'),
                        **G.nodes[node]
                    }
                    for node in G.nodes()
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        **G.edges[u, v]
                    }
                    for u, v in G.edges()
                ],
                'stats': {
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges()
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def export_world(
        self,
        world_state: Any,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export a world state to various formats.
        
        Args:
            world_state: WorldState object to export
            format: "json", "markdown", or "graph"
            output_path: Optional file path to save output
        
        Returns:
            Exported content as string
        """
        content = ""
        
        if format == "json":
            content = world_state.to_json() if hasattr(world_state, 'to_json') else str(world_state.to_dict())
        elif format == "markdown":
            from CALab.narrative_bridge import world_state_to_world_dna
            content = world_state_to_world_dna(world_state)
        elif format == "graph":
            network_data = self.get_cross_lab_network(world_state)
            import json
            content = json.dumps(network_data, indent=2, default=str)
        
        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(content)
        
        return content


# Integration adapters for existing systems

class StoryLabAdapter:
    """Adapter for StoryLab integration."""

    def __init__(self):
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if StoryLab components are available."""
        try:
            from StoryLab.story_generator import generate_story_idea
            return True
        except ImportError:
            return False

    def generate_story(self, world_dna: str, **kwargs) -> Dict[str, Any]:
        """Generate story using StoryLab."""
        if not self.available:
            # Fallback
            return {
                'story': f"Generated story based on: {world_dna[:200]}...",
                'fallback': True
            }

        from StoryLab.story_generator import generate_story_idea

        try:
            result = generate_story_idea(world_dna, **kwargs)
            if len(result) == 3:
                story, critique, extra = result
                return {
                    'story': story,
                    'critique': critique,
                    'extra': extra
                }
            else:
                story, critique = result
                return {
                    'story': story,
                    'critique': critique
                }
        except Exception as e:
            return {
                'story': f"Story generation failed: {e}",
                'error': str(e)
            }


class NNLabAdapter:
    """Adapter for NNLab generative models."""

    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load available NNLab models."""
        # Try to load pure JAX models first
        try:
            from NNLab.generative_models import GAN, DiffusionModel, NormalizingFlow
            # Only initialize simple models to avoid CUDA issues
            print("NNLab generative models available")
            self.models_available = True
        except (ImportError, Exception) as e:
            print(f"NNLab generative models not available: {e}")
            self.models_available = False

        # Try Flax models
        try:
            from NNLab.architectures.models import VAE
            print("Flax models available but need initialization")
        except ImportError:
            print("Flax models not available")

    def sample_gan(self, key, n_samples=1):
        """Sample from GAN."""
        if 'gan' not in self.models:
            return jnp.zeros((n_samples, 784))

        return self.models['gan'].generate(key, n_samples)


class SynthLabAdapter:
    """Adapter for SynthLab CA systems."""

    def __init__(self):
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if SynthLab is available."""
        try:
            from SynthLab.hybrid_framework import SemanticCA
            return True
        except ImportError:
            return False

    def simulate_ca(self, initial_state: jnp.ndarray, steps: int, **kwargs) -> jnp.ndarray:
        """Run CA simulation."""
        if not self.available:
            # Simple CA fallback
            state = initial_state
            for _ in range(steps):
                # Conway's Game of Life rules (simplified)
                neighbors = jnp.zeros_like(state)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        neighbors += jnp.roll(jnp.roll(state, di, axis=0), dj, axis=1)

                state = jnp.where(
                    (neighbors == 3) | ((state == 1) & (neighbors == 2)), 1, 0
                )
            return state

        from SynthLab.hybrid_framework import SemanticCA

        try:
            ca = SemanticCA(grid_size=(initial_state.shape[0], initial_state.shape[1]))
            # Set initial state (simplified)
            ca.grid_state['alive'] = initial_state.astype(bool)

            for _ in range(steps):
                ca.step({}, {})

            return ca.grid_state['alive'].astype(jnp.float32)

        except Exception as e:
            print(f"SynthLab simulation error: {e}")
            return initial_state


# Global API instance
generative_api = UnifiedGenerativeAPI()

# Register adapters
story_adapter = StoryLabAdapter()
generative_api.register_pipeline('story', story_adapter.generate_story)

nn_adapter = NNLabAdapter()
if 'gan' in nn_adapter.models:
    generative_api.register_model('gan', nn_adapter.models['gan'])

synth_adapter = SynthLabAdapter()
generative_api.register_pipeline('ca', synth_adapter.simulate_ca)


def get_generative_api() -> UnifiedGenerativeAPI:
    """Get the global generative API instance."""
    return generative_api


# Example usage
if __name__ == "__main__":
    api = get_generative_api()

    print("Available models:", api.get_available_models())
    print("Available pipelines:", api.get_available_pipelines())

    # Test basic story generation
    key = jax.random.PRNGKey(42)
    story_result = api.generate_story("A world of floating islands and ancient magic")
    print("Generated story:", story_result.get('story', 'No story generated')[:200] + "...")

    # Test multimodal generation
    multimodal = api.multimodal_generation("A mystical forest with glowing trees", modalities=['text'])
    print("Multimodal result keys:", list(multimodal.keys()))