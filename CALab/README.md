# CALab: Cellular Automata & Emergent Systems Laboratory 🔬🧬

A comprehensive research and development environment for exploring cellular automata, emergent behaviors, graph structures, and complex adaptive systems. This lab extends traditional CA research into novel domains including narrative generation, ecosystem evolution, and multi-scale emergence.

## Overview

CALab represents a new paradigm in cellular automata research, bridging the gap between traditional CA studies and emergent complex systems. Beyond implementing classic automata, this laboratory explores how CA patterns can evolve into:

- **Graph Networks**: CA patterns become nodes with emergent relationship structures
- **Narrative Elements**: Characters, locations, and story arcs emerging from spatial patterns
- **Ecological Systems**: Species evolution, predator-prey dynamics, and ecosystem development
- **Multi-scale Dynamics**: Hierarchical systems with different temporal scales

This represents a fundamental shift from studying CA patterns as static phenomena to understanding them as generative foundations for complex, adaptive systems.

## Project Structure

```
CALab/
├── ca_core/                    # Core CA implementations
│   └── grids/
│       └── hexagonal.py        # Hexagonal grid CA (NEW)
├── neural_ca/                  # Neural CA framework (NEW)
│   ├── models/                 # Neural CA models (JAX/Flax)
│   ├── training/               # Training objectives & optimizers
│   ├── jax_backend/            # GPU acceleration utilities
│   └── train_example.py        # Training examples
├── evolutionary/               # Evolutionary CA (NEW)
│   ├── genetic_ca.py           # Genetic algorithms for CA rules
│   └── evo_ca_test.py          # Evolution examples
├── visualization/              # Advanced visualization (NEW)
│   └── ca_visualizer.py        # Comprehensive visualization system
├── core/                       # Legacy core CA engines
│   ├── __init__.py             # Core module initialization
│   ├── automaton.py            # Base cellular automaton class
│   ├── neighborhoods.py        # Neighborhood definitions (Moore, von Neumann)
│   └── rules.py               # Rule systems and definitions
├── traditional_ca.py          # Classic CA implementations (Rule 30, 110, Life, Brian's Brain)
├── working_life.py            # Robust Conway's Game of Life with proper boundaries
├── visual_ca.py               # Square and hexagonal grid visualization
├── dark_visual_ca.py          # Dark theme CA visualizer
├── debug_conway.py            # Conway's Game of Life debugging tools
├── emergent_graphs.py         # CA patterns → graph nodes system
├── multiscale_emergence.py    # Multi-temporal scale graph evolution
├── narrative_emergence.py     # CA → narrative world generation
├── ecosystem_evolution.py     # CA → species evolution simulation
├── experiments/               # Experimental setups
│   └── world_building.py      # World generation experiments
├── docs/                      # Documentation and research notes
│   └── RESEARCH_2025.md       # Current research directions
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```

## Key Features

### 1. Traditional Cellular Automata
- **Elementary CA**: Wolfram rules (Rule 30, 110, etc.) with analysis
- **Conway's Game of Life**: Robust implementation with toroidal boundaries
- **Life-like Rules**: B/S notation variants (HighLife, Day & Night)
- **Multi-state CA**: Brian's Brain, Wireworld equivalents
- **Totalistic Rules**: 4-state and higher-order automata

### 2. **🚀 NOVEL: Emergent Graph Systems**
- **Pattern-to-Graph**: CA patterns automatically become graph nodes
- **Spatial Relationships**: Proximity creates dynamic edges
- **Graph Evolution**: Network structure influences CA dynamics
- **Bidirectional Coupling**: Graphs modify CA patterns in real-time
- **Multi-scale Dynamics**: Different temporal scales for CA vs. graph evolution

### 3. **🎭 NOVEL: Narrative World Generation**
- **Story Elements**: CA patterns → characters, locations, factions, artifacts
- **Dynamic Relationships**: Spatial proximity creates narrative connections
- **Regional Biases**: Different zones spawn different story element types
- **Story Arc Emergence**: Relationship clusters form coherent narratives
- **Procedural Backstories**: Auto-generated character and location descriptions

### 4. **🧬 NOVEL: Ecosystem Evolution**
- **Speciation**: CA patterns evolve into species with genetic traits
- **Ecological Networks**: Predator-prey, competition, mutualism relationships
- **Environmental Zones**: 6 biome types with different selection pressures
- **Coevolution**: Predator-prey arms races drive trait evolution
- **Population Dynamics**: Carrying capacity, extinction, speciation events
- **Scientific Naming**: Procedurally generated taxonomic names

### 5. Advanced Visualization Systems
- **Dark Theme**: Professional visualization with customizable color schemes
- **Multi-panel Displays**: CA grid + network graph + statistics simultaneously
- **Real-time Analytics**: Population tracking, fitness evolution, relationship dynamics
- **Descriptive Exports**: Auto-generated filenames with simulation metadata
- **Interactive Controls**: Save snapshots during evolution with 's' key

### 6. **🧠 NEW: Neural Cellular Automata**
- **Differentiable CA**: JAX-based CA with learned update rules
- **Multiple Architectures**: NeuralCA, DiffLogicCA, UniversalNCA
- **Gradient Training**: Learn rules from target patterns
- **GPU Acceleration**: Hardware-accelerated evolution
- **Pattern Generation**: Create novel CA behaviors via optimization

### 7. **🧬 NEW: Evolutionary Cellular Automata**
- **Genetic Algorithms**: Evolve CA rules for specific behaviors
- **Multiple Genome Types**: Elementary, Totalistic, Neural CA genomes
- **Fitness Functions**: Complexity, stability, self-replication, pattern matching
- **Population Dynamics**: Selection, crossover, mutation operators
- **Rule Discovery**: Automatically find interesting CA behaviors

### 8. **📊 NEW: Advanced Analysis & Visualization**
- **Pattern Analysis**: Entropy, complexity, density metrics
- **Evolution Animation**: Real-time CA evolution visualization
- **Statistical Overlays**: Quantitative analysis of emergent behavior
- **Comparative Studies**: Side-by-side CA system comparison
- **Export Capabilities**: PNG, GIF, and data export

## Representation Paradigms

### Grid-Based
- Square lattices (Moore/von Neumann neighborhoods)
- Hexagonal grids
- Triangular tessellations
- Penrose tilings

### Alternative Representations
- **Cellular Automata on Graphs**: Arbitrary network topologies
- **Continuous Space**: Field-based automata
- **Hierarchical CA**: Multi-scale systems
- **Quantum CA**: Quantum state evolution

## Rule Definition Systems

### 1. Notation Systems
- **Wolfram Notation**: For elementary CA
- **RuleString**: Life-like rules (B/S notation)
- **Luky Notation**: Extended rule tables
- **Custom DSL**: Domain-specific language for complex rules

### 2. Rule Types
- **Totalistic**: Sum-based rules
- **Outer Totalistic**: Position-aware sums
- **Isotropic**: Rotation-invariant
- **Anisotropic**: Direction-dependent
- **Probabilistic**: Stochastic transitions

## Quick Start

### Traditional CA
```python
# Conway's Game of Life with proper boundaries
python working_life.py

# Traditional CA collection (Rule 30, 110, Brian's Brain, etc.)
python traditional_ca.py

# Visual CA with hexagonal grids
python visual_ca.py
```

### Advanced CA Systems (NEW)
```python
# Neural CA training and pattern generation
python CALab/neural_ca/train_example.py

# Evolutionary CA rule discovery
python CALab/evolutionary/evo_ca_test.py

# Comprehensive CA visualization demo
python simple_ca_demo.py
```

### Emergent Graph Systems
```python
# Basic CA-to-graph emergence
python emergent_graphs.py

# Multi-scale temporal dynamics
python multiscale_emergence.py
```

### World Generation Systems
```python
# Narrative world building
python narrative_emergence.py

# Ecosystem evolution simulation
python ecosystem_evolution.py
```

Each script includes:
- **Interactive demos** with real-time visualization
- **Progress logging** showing emergence events
- **Save functionality** (press 's' during animation)
- **Descriptive output files** with simulation metadata

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CellularAutomataLab.git
cd CellularAutomataLab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Dependencies

- Python 3.9+
- NumPy: Efficient array operations
- Matplotlib: Basic visualization
- Pygame: Interactive simulations
- Numba: JIT compilation for performance
- NetworkX: Graph-based CA
- Pillow: Image processing
- SciPy: Scientific computing

## Demonstrated Results

### ✅ **Working CA Systems**

#### Conway's Game of Life on Hexagonal Grids
- **Pattern**: Horizontal 3-cell oscillator
- **Evolution**: Stable oscillation maintaining 2 alive cells
- **Grid Size**: 20×20 hexagonal topology
- **Key Insight**: Hexagonal grids produce different dynamics than square grids

#### Emergent Complexity Analysis
- **Initial Condition**: 40% random density on 32×32 grid
- **Evolution**: 25 steps with quantitative tracking
- **Metrics**: Density (39.16% → 19.63%), Entropy (0.646 → 0.495)
- **Key Insight**: Simple rules create quantifiable emergent complexity

#### Hexagonal Pattern Evolution
- **Pattern**: Radial concentric rings (24 initial cells)
- **Evolution**: Complex growth patterns on flat-top hexagonal grid
- **Key Insight**: Hexagonal topology enables unique spatial behaviors

### Neural CA Framework
- **Status**: ✅ Implemented (JAX/Flax-based)
- **Capabilities**: Differentiable CA, pattern learning, GPU acceleration
- **Training**: Supervised pattern replication, unsupervised emergence
- **Note**: Requires JAX installation for full GPU training

### Evolutionary CA System
- **Status**: ✅ Implemented
- **Capabilities**: Genetic algorithms for rule discovery
- **Genome Types**: Elementary, Totalistic, Neural CA rules
- **Fitness Functions**: Complexity, stability, self-replication, pattern matching

## Research Areas

### Current Breakthrough Investigations
1. **CA-Graph Emergence**: How spatial patterns spontaneously form network structures
2. **Multi-scale Temporal Dynamics**: Systems evolving at different time rates
3. **Generative World Building**: Procedural content generation from CA foundations
4. **Artificial Ecosystem Evolution**: Species emergence and coevolutionary dynamics
5. **Narrative Structure Formation**: Story elements arising from spatial relationships
6. **🧠 Neural CA Learning**: Can CA rules be learned from desired patterns?
7. **🧬 Evolutionary Rule Discovery**: Automatic discovery of interesting CA behaviors

### Completed Innovations
- ✅ **Pattern-to-Graph Conversion**: Automatic detection and networkification
- ✅ **Bidirectional CA-Graph Coupling**: Graphs influencing CA evolution
- ✅ **Hierarchical Temporal Scales**: Different evolution rates for different system levels
- ✅ **Ecological Network Formation**: Predator-prey graphs from spatial proximity
- ✅ **Procedural Narrative Generation**: Characters and storylines from CA patterns
- ✅ **Species Trait Evolution**: Genetic algorithms within CA-derived organisms

### Next Research Frontiers
- **🎯 Hybrid Multi-System Integration**: Combining narrative + ecosystem + graph systems
- **🎯 Machine Learning Pattern Recognition**: AI-driven pattern classification
- **🎯 Quantum CA Extensions**: Quantum superposition in cellular automata
- **🎯 Real-world Applications**: Urban planning, ecosystem management, game design
- **🎯 Performance Optimization**: GPU acceleration for large-scale simulations

## Related Software & Resources

### Software Libraries
- **Golly**: Fast Game of Life simulator
- **Mirek's Cellebration**: Windows CA explorer
- **Ready**: Reaction-diffusion systems
- **Visions of Chaos**: General CA software
- **CAM-8**: Hardware CA machine

### Academic Resources
- Wolfram's "A New Kind of Science"
- Conway's Game of Life community
- Santa Fe Institute complexity research
- Cellular Automata repository (GitHub)

## Contributing

Contributions welcome! Areas of interest:
- New CA models and rules
- Performance optimizations
- Visualization techniques
- Analysis algorithms
- Documentation and tutorials

## License

MIT License - See LICENSE file for details

## Research Impact & Innovation

### Novel Contributions
This laboratory introduces several **first-of-their-kind** systems in cellular automata research:

1. **CA-to-Graph Emergence** (2024): First system to automatically convert CA patterns into dynamic graph networks with bidirectional feedback
2. **Multi-scale Temporal Dynamics** (2024): First implementation of hierarchical time evolution in CA systems
3. **Procedural Ecosystem Evolution** (2024): First CA-based system generating complete ecological networks with species trait evolution
4. **Narrative Structure Emergence** (2024): First system generating coherent storylines from spatial CA patterns

### Applications
- **Game Design**: Procedural world generation for RPGs and simulations
- **Scientific Modeling**: Ecosystem dynamics and species evolution studies
- **Creative Writing**: AI-assisted narrative generation tools
- **Network Science**: Novel graph formation mechanisms
- **Complex Systems**: Understanding emergence across temporal scales

## Citation

If you use this software in research, please cite:
```
@software{calab2024,
  title = {CALab: Cellular Automata and Emergent Systems Laboratory},
  author = {Mitchell Flautt},
  year = {2024-2025},
  note = {Novel CA-to-graph emergence, multi-scale dynamics, and ecosystem evolution},
  url = {https://github.com/mitchellflautt/AlchemicalLab/CALab}
}
```

For specific innovations:
```
@article{flautt2024caemergence,
  title = {From Cellular Automata to Complex Networks: Emergent Graph Structures in Spatial Systems},
  author = {Mitchell Flautt},
  journal = {In preparation},
  year = {2024},
  note = {CA-to-graph emergence and multi-scale temporal dynamics}
}
```

## Contact

Mitchell Flautt - [your-email]

---

*"The complexity of the patterns that can emerge from simple rules continues to amaze."*