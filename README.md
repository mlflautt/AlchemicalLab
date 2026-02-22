# AlchemicalLab: Generative Systems Overview

AlchemicalLab is a comprehensive framework for generative AI systems, combining multiple approaches to create rich, emergent content across various modalities.

## 🏗️ Architecture

The system is organized into specialized labs, each focusing on different generative paradigms:

### 🎨 StoryLab - Narrative Generation
**Location:** `StoryLab/`
**Purpose:** Generate coherent narratives and world-building content

**Key Components:**
- `story_generator.py`: Multi-step story generation with LLM integration
- `api_integrations.py`: External API connections (news, books, images, TTS)
- `db_manager.py`: ChromaDB vector storage for story embeddings
- `critic.py`: Rule-based and LLM critique system

**Capabilities:**
- Hierarchical story generation (DNA → biomes → characters → plots)
- API-driven inspiration from real-world sources
- Vector similarity search and entity relationship mapping
- Multi-round critique and refinement

### 🧠 NNLab - Neural Generative Models
**Location:** `NNLab/`
**Purpose:** Advanced neural network-based generation

**Key Components:**
- `architectures/models.py`: Flax-based neural architectures
- `training/train.py`: JAX training utilities and optimizers
- `generative_models.py`: Pure JAX generative models (GANs, diffusion, flows)

**Capabilities:**
- **GANs**: Adversarial generation with discriminator networks
- **Diffusion Models**: DDPM implementation for high-quality generation
- **Normalizing Flows**: Flexible density estimation and sampling
- **VAE**: Latent space learning and reconstruction

### 🌊 SynthLab - Emergent Systems
**Location:** `SynthLab/`
**Purpose:** Cellular automata and emergent behavior simulation

**Key Components:**
- `hybrid_framework.py`: Multi-layer semantic CA with 5-layer architecture
- `ca_rule_evolution.py`: EA-optimized CA rule evolution
- `web/`: Real-time visualization server with D3.js interface

**Capabilities:**
- Multi-layer CA (biological, economic, cultural, physical, legacy)
- Evolutionary rule optimization using genetic algorithms
- Real-time web visualization with live statistics
- Hybrid CA-EA-NN integration

### 🧬 EALab - Evolutionary Algorithms
**Location:** `EALab/`
**Purpose:** Population-based optimization and generation

**Key Components:**
- `algorithms/`: GA, ES, DE implementations
- `operators/`: Selection, crossover, mutation operators
- `analysis/`: Population dynamics visualization

**Capabilities:**
- Multiple evolutionary algorithms (GA, ES, DE)
- Flexible operator system for custom evolution
- Population analysis and visualization
- Integration with other generative systems

## 🔗 Unified Generative API

All systems are accessible through a unified interface in `generative_api.py`:

```python
from generative_api import get_generative_api

api = get_generative_api()

# Generate a story
story = api.generate_story("A world of floating islands...")

# Run CA simulation
initial_state = jnp.ones((50, 50))
final_state = api.simulate_ca(initial_state, steps=100)

# Multi-modal generation
result = api.multimodal_generation("Crystal cave with glowing runes")
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install jax jaxlib numpy requests chromadb sentence-transformers networkx
# Optional: flax optax matplotlib seaborn (for full functionality)
```

### Basic Usage
```python
# Run the demonstration
python generative_demo.py

# Use individual systems
from StoryLab.story_generator import generate_story_idea
from SynthLab.hybrid_framework import SemanticCA
from NNLab.generative_models import GAN
```

## 📊 Current Status

| System | Core Status | Dependencies | GPU Support | Production Ready |
|--------|-------------|--------------|-------------|------------------|
| **StoryLab** | ✅ Working | ⚠️ Partial | ✅ CPU | ✅ With fallbacks |
| **NNLab** | ✅ Models | ❌ Flax missing | ⚠️ CUDA issues | ⚠️ Needs fixes |
| **SynthLab** | ✅ CA | ✅ Complete | ⚠️ CUDA issues | ✅ Core ready |
| **EALab** | ⚠️ Basic | ⚠️ Visualization | ✅ CPU | ⚠️ Needs polish |

## 🔧 Installation & Setup

### Full Installation
```bash
# Install core dependencies
pip install jax jaxlib numpy requests

# Install vector database
pip install chromadb sentence-transformers

# Install graph utilities
pip install networkx

# Optional: Full ML stack
pip install flax optax matplotlib seaborn
```

### Environment Setup
```bash
# Set environment variables for APIs
export NEWSAPI_KEY="your_key_here"
export REPLICATE_API_TOKEN="your_token_here"
```

## 🎯 Key Features

### Cross-System Integration
- **StoryLab ↔ NNLab**: Text generation enhanced with neural models
- **SynthLab ↔ EALab**: Evolutionary CA rule optimization
- **NNLab ↔ SynthLab**: Neural CA and hybrid architectures

### Robust Fallbacks
- Systems work with or without optional dependencies
- Graceful degradation when external APIs are unavailable
- Local alternatives for cloud services

### GPU Acceleration
- JAX-based systems support GPU acceleration
- CUDA compatibility issues currently limit GPU usage
- CPU implementations fully functional

## 📈 Development Roadmap

### Phase 1: Foundation Stabilization ✅
- [x] Fix import issues and dependency management
- [x] Implement graceful fallbacks
- [x] Create unified API interface

### Phase 2: Advanced Capabilities 🚧
- [ ] GPU acceleration fixes
- [ ] Full Flax/Optax integration
- [ ] Enhanced multimodal pipelines

### Phase 3: Production Deployment 📋
- [ ] Containerization and deployment
- [ ] Performance monitoring
- [ ] API documentation and examples

## 🧪 Testing

Run the comprehensive demonstration:
```bash
python generative_demo.py
```

Test individual systems:
```bash
# StoryLab
cd StoryLab && python -c "from story_generator import load_world_dna; print('DNA loaded')"

# NNLab
cd NNLab && JAX_PLATFORM_NAME=cpu python generative_models.py

# SynthLab
cd SynthLab && python -c "from hybrid_framework import SemanticCA; ca = SemanticCA(); print('CA ready')"
```

## 🤝 Contributing

The system is designed for extensibility:
- Add new generative models by implementing the `GenerativeModel` protocol
- Register new pipelines in the unified API
- Extend existing systems with new capabilities

## 📄 License

This project combines multiple generative approaches for research and development in artificial creativity and emergent systems.

---

**Built with ❤️ using JAX, Flax, and the power of emergence**