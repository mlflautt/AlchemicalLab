# Cross-Lab Integration Guide

This guide explains how to use the integrated CA → StoryLab pipeline for emergent world-building and narrative generation.

## Overview

The cross-lab integration connects:
- **CALab**: Cellular automata systems for ecosystem and narrative emergence
- **StoryLab**: LLM-based narrative generation with critic system
- **Unified API**: Single interface for multi-modal generation

## Quick Start

### Basic Usage

```python
from generative_api import get_generative_api

api = get_generative_api()

# Generate a world from CA simulation
result = api.generate_world_from_ca(
    seed=42,
    generations=50,
    world_size=(100, 100)
)

print(result['world_dna'])
```

### Full Pipeline: CA → Story

```python
from generative_api import get_generative_api

api = get_generative_api()

# Run CA simulation and generate narrative
result = api.simulate_and_narrate(
    seed=42,
    generations=30,
    world_size=(60, 60),
    theme="discovery"
)

print("World Stats:", result['stats'])
print("Story:", result['story'])
```

## Components

### 1. HybridCAAggregator (`CALab/hybrid_ca_aggregator.py`)

Combines ecosystem evolution and narrative emergence systems.

```python
from CALab.hybrid_ca_aggregator import HybridCAAggregator, run_hybrid_simulation

# Quick function
world_state = run_hybrid_simulation(
    world_size=(100, 100),
    generations=50,
    density=0.3,
    seed=42
)

# Full control
aggregator = HybridCAAggregator(world_size=(100, 100))
aggregator.initialize(density=0.3, seed=42)
aggregator.run(generations=100)
world_state = aggregator.get_world_state()

# Get combined network
network = aggregator.get_cross_lab_network()

# Map entities across systems
mappings = aggregator.cross_map_entities()
```

### 2. NarrativeBridge (`CALab/narrative_bridge.py`)

Converts CA WorldState to StoryLab's world_dna format.

```python
from CALab.narrative_bridge import WorldDNAGenerator, world_state_to_world_dna

# Quick conversion
dna = world_state_to_world_dna(world_state, world_name="My World")

# Save to file
world_state_to_world_dna(world_state, output_path="world_dna.md")

# Full control
generator = WorldDNAGenerator()
dna = generator.generate_world_dna(
    world_state,
    world_name="Emergent World",
    genre="Fantasy",
    model="DeepSeek-R1",
    temperature=0.7
)
```

### 3. CAAdapter (`StoryLab/ca_adapter.py`)

Direct integration with StoryLab's LLM system.

```python
from StoryLab.ca_adapter import CAAdapter, quick_ca_story

# Quick story from CA
result = quick_ca_story(
    generations=30,
    world_size=(60, 60),
    theme="adventure"
)

# Full control
adapter = CAAdapter(model_path="../qwen_q2.gguf")

# Load from existing WorldState
context = adapter.load_world_state(world_state)

# Generate story
story, critique, extra = adapter.generate_story(
    context,
    multi_step=True,
    theme="exploration"
)

# Sync entities to ChromaDB
adapter.sync_embeddings_to_db(context)

# Complete pipeline
result = adapter.run_full_pipeline(
    generations=50,
    world_size=(80, 80),
    world_name="Test World",
    multi_step=True
)
```

### 4. Unified API Extensions

New methods in `generative_api.py`:

```python
from generative_api import get_generative_api

api = get_generative_api()

# Generate world from CA
world = api.generate_world_from_ca(
    seed=42,
    generations=50,
    world_size=(100, 100),
    complexity="high"
)

# Simulate and narrate in one call
result = api.simulate_and_narrate(
    seed=42,
    generations=30,
    theme="mystery"
)

# Cross-modal generation
result = api.cross_modal_generate(
    prompt="A world of floating islands",
    ca_seed=42,
    modalities=['text', 'world']
)

# Get network representation
network = api.get_cross_lab_network(world_state)

# Export world
api.export_world(world_state, format="json", output_path="world.json")
api.export_world(world_state, format="markdown", output_path="world.md")
```

## Data Structures

### WorldState

The central data structure holding all CA-derived entities:

```python
@dataclass
class WorldState:
    generation: int
    world_size: Tuple[int, int]
    
    # Entities
    species: Dict[int, SpeciesData]
    characters: Dict[int, CharacterData]
    locations: Dict[int, LocationData]
    factions: Dict[int, FactionData]
    story_arcs: Dict[int, StoryArcData]
    
    # Relationships
    ecological_relationships: List[EcologicalRelationshipData]
    narrative_relationships: List[NarrativeRelationshipData]
    
    # World info
    biomes: Dict[str, Dict]
    regions: Dict[str, Dict]
    ca_grid: Optional[np.ndarray]
    stats: Dict[str, Any]
```

### Entity Data Classes

Each entity type has a corresponding dataclass:

- `SpeciesData`: Species from ecosystem evolution
- `CharacterData`: Characters from narrative emergence
- `LocationData`: Locations from narrative emergence
- `FactionData`: Factions from narrative emergence
- `StoryArcData`: Story arcs from relationship patterns

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedGenerativeAPI                     │
├─────────────────────────────────────────────────────────────┤
│  generate_world_from_ca()  simulate_and_narrate()           │
│  cross_modal_generate()    get_cross_lab_network()          │
│  export_world()            generate_story()                  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │   CALab     │   │   Bridge    │   │  StoryLab   │
    │             │   │             │   │             │
    │ Ecosystem   │──▶│ WorldDNA    │──▶│ LLM Gen     │
    │ Narrative   │   │ Generator   │   │ Critic      │
    │ Hybrid Agg  │   │             │   │ ChromaDB    │
    └─────────────┘   └─────────────┘   └─────────────┘
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   WorldState    │
                    │   (dataclass)   │
                    └─────────────────┘
```

## Testing

Run integration tests:

```bash
python tests/test_cross_lab.py
```

Quick tests only:

```bash
python tests/test_cross_lab.py --quick
```

## Configuration

### CA Simulation Parameters

- `world_size`: Grid dimensions (default: (100, 100))
- `generations`: Simulation steps (default: 50)
- `density`: Initial cell density 0-1 (default: 0.3)
- `seed`: Random seed for reproducibility

### Story Generation Parameters

- `model_path`: Path to GGUF model file
- `ngl`: GPU layers (default: 35)
- `multi_step`: Use hierarchical generation (default: False)
- `theme`: Optional theme focus
- `use_news`, `use_books`, etc.: API inspiration toggles

### Complexity Levels

- `low`: 30 generations, smaller world
- `medium`: 50 generations (default)
- `high`: 100 generations, full detail

## Examples

### Example 1: Quick World Generation

```python
from generative_api import get_generative_api

api = get_generative_api()

# Generate a small world quickly
result = api.generate_world_from_ca(
    seed=123,
    generations=20,
    world_size=(40, 40),
    complexity="low"
)

print(f"Generated {len(result['entities']['species'])} species")
print(f"Generated {len(result['entities']['characters'])} characters")
```

### Example 2: Full Narrative Generation

```python
from StoryLab.ca_adapter import CAAdapter

adapter = CAAdapter()

# Run complete pipeline
result = adapter.run_full_pipeline(
    generations=40,
    world_size=(70, 70),
    world_name="The Emergent Realm",
    genre="Fantasy",
    multi_step=True,
    theme="ancient mysteries"
)

print("=== World DNA ===")
print(result['world_dna'][:500])

print("\n=== Story ===")
print(result['story'][:500])
```

### Example 3: Entity Extraction

```python
from CALab.hybrid_ca_aggregator import run_hybrid_simulation
from CALab.narrative_bridge import generate_story_elements_from_species

world_state = run_hybrid_simulation(generations=30)

# Convert species to narrative elements
story_elements = generate_story_elements_from_species(world_state.species)

for element in story_elements[:3]:
    print(f"{element['name']}: {element['narrative_hooks']}")
```

## Troubleshooting

### Import Errors

If you get import errors, ensure:
1. You're running from the project root directory
2. All dependencies are installed (`pip install -r requirements.txt`)
3. The CALab and StoryLab directories have `__init__.py` files

### LLM Not Responding

1. Check that `llama.cpp` is built with CUDA support
2. Verify model paths are correct
3. Reduce `ngl` parameter if GPU memory is limited

### Empty World State

If the world state has few entities:
1. Increase `generations` parameter
2. Increase `density` parameter
3. Use larger `world_size`

## Future Enhancements

- GPU acceleration for CA simulation
- Real-time visualization integration
- Multi-model story generation
- Persistent world state storage
- Web interface for pipeline control
