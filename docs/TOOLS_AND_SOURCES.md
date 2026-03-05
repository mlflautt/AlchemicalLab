# AlchemicalLab Auxiliary Tools & Sources

This document lists all external tools, libraries, frameworks, and research references used in AlchemicalLab.

## Core Dependencies

### Python Libraries

| Library | Version | Purpose | Source |
|---------|---------|---------|--------|
| numpy | >=1.21.0 | Numerical computing, arrays | pip install numpy |
| scipy | >=1.7.0 | Signal processing, convolutions | pip install scipy |
| matplotlib | >=3.7.0 | Visualization | pip install matplotlib |
| networkx | >=3.1.0 | Graph data structures | pip install networkx |
| chromadb | >=0.4.0 | Vector database | pip install chromadb |
| sentence-transformers | >=2.2.0 | Embeddings | pip install sentence-transformers |
| pyyaml | >=6.0.0 | Config files | pip install pyyaml |

### Optional Dependencies

| Library | Purpose | Source |
|---------|---------|--------|
| jax | GPU-accelerated ML | pip install jax |
| torch | Neural networks | pip install torch |
| deap | Evolutionary algorithms | pip install deap |

## Audio Synthesis References

### Wave Terrain Synthesis

1. **Terrain Synth** (Primary Reference)
   - Repository: https://github.com/aaronaanderson/Terrain
   - Author: Aaron Anderson
   - License: GPL-3.0
   - Description: Open source wave terrain synth - primary inspiration for our implementation

### CA in Music Research

2. **Chaosynth**
   - Paper: "evolving cellular automata music: From sound synthesis to composition"
   - Description: CA-controlled granular synthesis using ChaOs (Chemical Oscillator)

3. **CAMUS** (Cellular Automata MUisic System)
   - Paper: ResearchGate publication 228862474
   - Description: CA for macro-structural musical forms

4. **PQCA** (Partitioned Quantum Cellular Automata)
   - Paper: MDPI 2076-3417/13/4/2401
   - Description: Quantum CA for generative music

### Fractal Audio

5. **Fractal Dimension & Timbre**
   - Paper: DAFx - "Simulation of Textured Audio Harmonics"
   - Description: Using fractal dimension for timbre control

6. **1/f Music**
   - Reference: Ethan Hein Blog - "Fractal music"
   - Description: Power-law distributions in music

### Neural Audio Synthesis

7. **RAVE** (Realtime Audio Variational autoEncoder)
   - Repository: https://github.com/acids-ircam/RAVE
   - Paper: arXiv:2111.05011
   - Description: Fast neural audio synthesis

8. **Audio-Morph GAN**
   - Project: https://ahlab.org/project/audio-morph-gan/
   - Description: Controllable audio texture morphing

9. **DyNCA** (Dynamic Neural Cellular Automata)
   - Paper: CVPR 2023
   - Description: Real-time dynamic texture synthesis using NCA

## Frameworks

### JUCE (C++ Audio Framework)

| Aspect | Detail |
|--------|--------|
| Version | 7.0+ |
| Website | https://juce.com |
| Repository | https://github.com/juce-framework/JUCE |
| License | GPL-3.0 |

Used for: AU/VST3 plugin development in `SynthLab/synth_plugin/`

### Perlin Noise

| Implementation | Detail |
|----------------|--------|
| Source | https://github.com/Reputeless/PerlinNoise |
| Author | Reputeless |
| License | MIT |

### MTS-ESP (Microtuning)

| Implementation | Detail |
|----------------|--------|
| Source | https://github.com/ODDSound/MTS-ESP |
| Description | MIDI tuning standard for dynamic microtuning |

## Graph & Knowledge Systems

### Knowledge Graph

| Component | Implementation |
|-----------|----------------|
| SQLite | Standard Python sqlite3 |
| Vector Search | ChromaDB |
| Graph View | NetworkX |

### Obsidian Integration

| Feature | Implementation |
|---------|----------------|
| Sync | Custom Markdown vault sync |
| Graph View | D3.js visualization |

## Research Papers & Articles

### Cellular Automata

1. Stanford Encyclopedia of Philosophy - Cellular Automata
   - URL: https://plato.stanford.edu/entries/cellular-automata/

2. Musical Composition and Two-Dimensional Cellular Automata
   - Publisher: Wolfram
   - URL: https://content.wolfram.com/sites/13/2025/06/34-2-4.pdf

### Fractals in Music

3. "Fractals and Their Symphony with Music"
   - Publisher: Medium (@cgpt)
   - URL: https://medium.com/@cgpt/fractals-and-their-symphony-with-music

4. "From Fractal Geometry to Fractal Cognition"
   - Publisher: MDPI
   - URL: https://www.mdpi.com/2504-3110/9/10/654

### Generative Audio

5. "Generative Audio Extension and Morphing"
   - Publisher: arXiv
   - URL: https://arxiv.org/html/2602.16790v1

6. "Music Generation Using Deep Learning"
   - Publisher: IEEE Xplore

### Adaptive Soundscapes

7. Endel - Neuroscience-based soundscapes
   - URL: https://art.art/blog/in-the-flow-of-sound

8. Lullabyte Project
   - URL: https://lullabyte.eu/
   - Description: EU-funded research on music and sleep

## Text & World Building

### Ingestion Sources

| Source | Integration |
|--------|-------------|
| Wikipedia | research_topic_wikipedia() |
| Archive.org | fetch_archive_text() |
| Open Library | fetch_open_library_excerpt() |
| Project Gutenberg | fetch_book_excerpt() |
| NASA | research_science_nasa() |
| Freesound | fetch_audio_sample() |

### LLM Integration

| Provider | Integration |
|----------|-------------|
| Ollama | Local models (phi3.5, etc.) |
| OpenAI | API-based |
| Anthropic | API-based |

## Code References

### Existing Libraries Used

1. **scipy.ndimage** - Connected component labeling for pattern detection
2. **numpy.random** - Random number generation
3. **json** - Configuration and preset files
4. **gzip** - Compressed checkpoint storage

### Inspired By

1. **NCA (Neural Cellular Automata)**
   - Distill.pub: "Growing Neural Cellular Automata"
   - URL: https://distill.pub/2020/growing-ca/

2. **Generative Adversarial Networks**
   - Style transfer concepts from audio domain

## File Formats

### Configuration

| Format | Purpose |
|--------|---------|
| YAML | Main config (config.yaml) |
| JSON | Audio presets, checkpoints |
| Markdown | World DNA, documentation |

### Audio

| Format | Purpose |
|--------|---------|
| WAV | Generated audio output |
| MIDI | Pattern triggers (planned) |

## Development Tools

### Build Systems

| Tool | Purpose |
|------|---------|
| CMake | C++ plugin builds |
| pytest | Python testing |

### Version Control

| Tool | Usage |
|------|-------|
| Git | Version control |
| GitHub | Remote repository |

## License Summary

| Component | License |
|-----------|---------|
| AlchemicalLab | GPL-3.0 |
| Terrain Synth | GPL-3.0 |
| JUCE | GPL-3.0 |
| PerlinNoise | MIT |
| RAVE | AGPL-3.0 |
| MTS-ESP | GPL-3.0 |

## External Resources

### Tutorials & Guides

1. Wave Terrain Synthesis Basics
2. CA Music Generation
3. Fractal Audio Processing

### Communities

1. Reddit r/algorithmicmusic
2. Reddit r/wearethemusicmakers
3. KVR Audio - Developer forums

## Acknowledgments

- Aaron Anderson (Terrain synth)
- JUCE Framework developers
- Open source CA research community
- Generative audio research community
